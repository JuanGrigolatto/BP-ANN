import os 
import csv 
import pandas as pd 
import numpy as np
import learn2learn as l2l
from torch import mode, nn, optim, seed
import torch.utils.data as data
from tqdm.auto import tqdm 
import torch
import random
import matplotlib.pyplot as plt

# --- TUS IMPORTACIONES ---
from src.data.data_chargers.Clase_UCIDataset import UCIDataset
from src.data.data_chargers.MetaDataset import TaskDataset
from src.data.data_chargers.PatientWiseSet import PatientWiseDataset
from src.models.ConvolucionalV1 import Modelo_ConvolucionalV1

# =============================================================================
#  CONFIGURACIÓN GLOBAL
# =============================================================================
CURRENT_STAGE = 2  # <--- 1 = BASE (Multifuente), 2 = DELTA (Instancia Única)

CHECKPOINT_DIR = 'models/checkpoints'
LOG_DIR = 'metalearning/logs'
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

if CURRENT_STAGE == 1:
    EXP_NAME = 'STAGE1_BASE_Multifuente_MSL'
    MODE = 'patient_wise'   
    ALPHA_LOSS = 0.1        # Prioridad Magnitud (MSE)
    PRETRAINED_PATH = None
    
elif CURRENT_STAGE == 2:
    EXP_NAME = 'STAGE2_DELTA_Specialist_MSL_HighLR'
    MODE = 'traditional'    
    ALPHA_LOSS = 0.9        # Prioridad Dinámica (Pearson)
    PRETRAINED_PATH = os.path.join(CHECKPOINT_DIR, 'best_STAGE1_BASE_Multifuente_MSL.pt')

LATEST_CKPT_PATH = os.path.join(CHECKPOINT_DIR, f'checkpoint_{EXP_NAME}.pt')
BEST_CKPT_PATH = os.path.join(CHECKPOINT_DIR, f'best_{EXP_NAME}.pt')
CSV_LOG_PATH = os.path.join(LOG_DIR, f'log_{EXP_NAME}.csv')

# =============================================================================
#  CLASES AUXILIARES
# =============================================================================
class HybridLoss(nn.Module):
    def __init__(self, alpha=0.5):
        super().__init__()
        self.mse = nn.MSELoss()
        self.alpha = alpha 

    def forward(self, pred, target):
        mse_loss = self.mse(pred, target)
        vx = pred - torch.mean(pred)
        vy = target - torch.mean(target)
        cost = torch.sum(vx * vy) / (torch.sqrt(torch.sum(vx ** 2)) * torch.sqrt(torch.sum(vy ** 2)) + 1e-6)
        pearson_loss = 1 - cost
        return (1 - self.alpha) * mse_loss + self.alpha * pearson_loss

def save_checkpoint(state, is_best, filename=LATEST_CKPT_PATH):
    torch.save(state, filename)
    if is_best: torch.save(state, BEST_CKPT_PATH)

def log_to_csv(epoch, train_loss, valid_loss, filepath=CSV_LOG_PATH):
    file_exists = os.path.isfile(filepath)
    with open(filepath, mode='a', newline='') as f:
        writer = csv.writer(f)
        if not file_exists: writer.writerow(['epoch', 'train_loss', 'valid_loss'])
        writer.writerow([epoch, train_loss, valid_loss])

def get_batch_data(batch, mode, device):
    if mode == 'traditional':
        x, y = batch
        if device.type == 'cuda': x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        return x, y, None, None
    else:
        xs, ys, xq, yq = batch
        if device.type == 'cuda':
            xs, ys = xs.to(device, non_blocking=True), ys.to(device, non_blocking=True)
            xq, yq = xq.to(device, non_blocking=True), yq.to(device, non_blocking=True)
        return xs, ys, xq, yq

# =============================================================================
#  VALIDACIÓN (Con Multi-Step Loss para consistencia)
# =============================================================================
def validate_meta_epoch(maml, val_loader, lossfn, adapt_steps, shots, device, mode):
    meta_val_loss = 0.0
    num_batches = 0
    
    for batch in val_loader:
        x_sup_b, y_sup_b, x_qry_b, y_qry_b = get_batch_data(batch, mode, device)
        effective_batch_size = x_sup_b.size(0)
        batch_loss = 0.0
        
        for i in range(effective_batch_size):
            learner = maml.clone()
            
            if mode == 'traditional':
                x_task, y_task = x_sup_b[i], y_sup_b[i]
                x_support, y_support = x_task[:shots], y_task[:shots]
                x_query, y_query = x_task[shots:], y_task[shots:]
            else:
                x_support, y_support = x_sup_b[i], y_sup_b[i]
                x_query, y_query = x_qry_b[i], y_qry_b[i]

            if CURRENT_STAGE == 2:
                bias = y_support.mean(dim=0, keepdim=True)
                y_support = y_support - bias
                y_query = y_query - bias

            # --- MULTI-STEP EVALUATION ---
            query_losses = []
            for _ in range(adapt_steps):
                # 1. Adaptar
                support_preds = learner(x_support)
                support_loss = lossfn(support_preds, y_support)
                learner.adapt(support_loss)
                
                # 2. Evaluar inmediatamente (MSL)
                with torch.no_grad():
                    query_preds = learner(x_query)
                    q_loss = lossfn(query_preds, y_query)
                    query_losses.append(q_loss.item())
            
            # Promedio de todos los pasos (MSL)
            batch_loss += np.mean(query_losses)
        
        meta_val_loss += batch_loss / effective_batch_size
        num_batches += 1
        
    if num_batches == 0: return 0.0
    return meta_val_loss / num_batches

# =============================================================================
#  MAIN LOOP
# =============================================================================
def main(shots=5, tasks_per_batch=4, adapt_lr=0.01, meta_lr=0.001, 
         seed=42, num_epochs=200, patience=20,
         # Parametros PatientWise
         N_patient_group=4, p_support=5, q_query=10,
         # Parametros Annealing
         adapt_steps_start=5, adapt_steps_end=1, anneal_epochs=100):

    print(f"\n=== EJECUTANDO STAGE {CURRENT_STAGE}: {EXP_NAME} ===")
    print(f"Annealing: {adapt_steps_start} -> {adapt_steps_end} pasos en {anneal_epochs} épocas.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

    # Dataset & Dataloaders (Igual que antes)
    data_paths = [
        'data/processed/data_UCI/dataset_parte_1_por_picos.pt',
        'data/processed/data_UCI/dataset_parte_2_por_picos.pt',
        'data/processed/data_UCI/dataset_parte_3_por_picos.pt',
        'data/processed/data_UCI/dataset_parte_4_por_picos.pt',
    ]
    dataset_completo = UCIDataset(data_paths)
    all_pids = torch.tensor([dataset_completo[i][2] for i in range(len(dataset_completo))])
    unique_patients = list(set(all_pids.tolist()))
    random.shuffle(unique_patients)

    min_samples = (shots * 2) if MODE == 'traditional' else (p_support + q_query)
    patient_to_indices = {pid: [] for pid in unique_patients}
    for i in range(len(dataset_completo)):
        patient_to_indices[int(dataset_completo[i][2])].append(i)
    valid_IDs = [pid for pid, idxs in patient_to_indices.items() if len(idxs) >= min_samples]
    
    n_train = int(len(valid_IDs) * 0.70)
    train_patients = valid_IDs[:n_train]
    val_patients = valid_IDs[n_train : int(len(valid_IDs)*0.85)]
    test_patients = valid_IDs[int(len(valid_IDs)*0.85):]

    specific_test_path = os.path.join('data/processed/data_UCI', f'test_ids_{EXP_NAME}.pt')
    
    # Siempre lo guardamos (o sobrescribimos) para asegurar que corresponda 
    # EXACTAMENTE a la seed=42 y configuración actual.
    print(f"Guardando IDs de Test específicos para este experimento en: {specific_test_path}")
    torch.save({
        'test_patient_ids': test_patients,
        'seed_used': seed,       # Guardamos la seed por seguridad
        'stage': CURRENT_STAGE   # Guardamos el stage
    }, specific_test_path)

    if MODE == 'traditional':
        train_set = TaskDataset(train_patients, base_dataset=dataset_completo, num_shots=shots, gap=50)
        val_set = TaskDataset(val_patients, base_dataset=dataset_completo, num_shots=shots, gap=50)
    else:
        train_set = PatientWiseDataset(train_patients, base_dataset=dataset_completo, N_patients=N_patient_group, p_support=p_support, q_query=q_query)
        val_set = PatientWiseDataset(val_patients, base_dataset=dataset_completo, N_patients=N_patient_group, p_support=p_support, q_query=q_query)

    train_loader = data.DataLoader(train_set, batch_size=tasks_per_batch, shuffle=True, num_workers=0, drop_last=True)
    val_loader = data.DataLoader(val_set, batch_size=tasks_per_batch, shuffle=False, num_workers=0, drop_last=True)
 
    model = Modelo_ConvolucionalV1(in_channels=2, out_channels=2, long_signal=500).to(device)
    maml = l2l.algorithms.MAML(model, lr=adapt_lr, first_order=True, allow_unused=True)
    opt = optim.Adam(maml.parameters(), meta_lr)
    lossfn = HybridLoss(alpha=ALPHA_LOSS).to(device)

    start_epoch = 0; best_valid_loss = float('inf'); patience_counter = 0

    # Carga de Checkpoints
    if os.path.exists(LATEST_CKPT_PATH):
        print(f"Reanudando: {LATEST_CKPT_PATH}")
        ckpt = torch.load(LATEST_CKPT_PATH)
        maml.load_state_dict(ckpt['model_state_dict']) 
        opt.load_state_dict(ckpt['optimizer_state_dict']) 
        start_epoch = ckpt['epoch'] + 1; best_valid_loss = ckpt['best_loss']
    elif CURRENT_STAGE == 2 and PRETRAINED_PATH and os.path.exists(PRETRAINED_PATH):
        print(f"⚠️ CARGANDO BASE MULTIFUENTE: {PRETRAINED_PATH}")
        ckpt = torch.load(PRETRAINED_PATH)
        maml.load_state_dict(ckpt['model_state_dict'])
    
    # Cálculo del Stride para Annealing
    if adapt_steps_start > adapt_steps_end:
        anneal_stride = max(1, anneal_epochs // (adapt_steps_start - adapt_steps_end))
    else: anneal_stride = 9999

    # --- ENTRENAMIENTO ---
    for epoch in range(start_epoch, num_epochs):  
        # --- 1. ANNEALING DE PASOS ---
        steps_drop = epoch // anneal_stride
        current_steps = max(adapt_steps_start - steps_drop, adapt_steps_end)
        
        epoch_loss_accum = 0.0; num_batches = 0
        
        with tqdm(train_loader, desc=f"Ep {epoch+1} (Steps={current_steps})") as pbar:
            for batch in pbar:
                x_sup_b, y_sup_b, x_qry_b, y_qry_b = get_batch_data(batch, MODE, device)
                effective_bs = x_sup_b.size(0)
                batch_total_loss = 0.0

                for i in range(effective_bs):
                    learner = maml.clone()
                    if MODE == 'traditional':
                        x_task, y_task = x_sup_b[i], y_sup_b[i]
                        x_support, y_support = x_task[:shots], y_task[:shots]
                        x_query, y_query = x_task[shots:], y_task[shots:]
                    else:
                        x_support, y_support = x_sup_b[i], y_sup_b[i]
                        x_query, y_query = x_qry_b[i], y_qry_b[i]

                    if CURRENT_STAGE == 2:
                        bias = y_support.mean(dim=0, keepdim=True)
                        y_support = y_support - bias; y_query = y_query - bias

                    # --- 2. MULTI-STEP LOSS (MSL) ---
                    task_losses = []
                    for _ in range(current_steps): 
                        # Adaptar
                        pred = learner(x_support)
                        loss = lossfn(pred, y_support)
                        learner.adapt(loss)
                        
                        # Evaluar Query INMEDIATAMENTE (MSL)
                        q_pred = learner(x_query)
                        q_loss = lossfn(q_pred, y_query)
                        task_losses.append(q_loss)

                    # Promediar pérdidas de todos los pasos
                    # Esto estabiliza el gradiente porque optimiza la trayectoria completa
                    batch_total_loss += torch.stack(task_losses).mean()

                meta_loss = batch_total_loss / effective_bs
                epoch_loss_accum += meta_loss.item()
                num_batches += 1
                
                opt.zero_grad(); meta_loss.backward(); opt.step()
                pbar.set_postfix({'loss': meta_loss.item()})

        train_loss = epoch_loss_accum / num_batches
        print(f"Train Loss: {train_loss:.4f}")
        
        val_shots = shots if MODE == 'traditional' else p_support
        valid_loss = validate_meta_epoch(maml, val_loader, lossfn, current_steps, val_shots, device, MODE)
        print(f"Valid Loss: {valid_loss:.4f}")
        log_to_csv(epoch+1, train_loss, valid_loss)

        is_best = valid_loss < (best_valid_loss - 1e-4)
        if is_best:
            best_valid_loss = valid_loss; patience_counter = 0
            print(" ** Nuevo Récord **")
            save_checkpoint({'epoch': epoch, 'model_state_dict': maml.state_dict(), 'optimizer_state_dict': opt.state_dict(), 'best_loss': best_valid_loss}, True)
        else:
            patience_counter += 1
            print(f" Paciencia: {patience_counter}/{patience}")
            save_checkpoint({'epoch': epoch, 'model_state_dict': maml.state_dict(), 'optimizer_state_dict': opt.state_dict(), 'best_loss': best_valid_loss}, False)

        if patience_counter >= patience: print("Early Stopping."); break

if __name__ == '__main__':
    if CURRENT_STAGE == 1:
        main(shots=10, tasks_per_batch=4, adapt_lr=0.01, meta_lr=0.001, N_patient_group=2, p_support=10, q_query=20,
             adapt_steps_start=5, adapt_steps_end=1, anneal_epochs=100)
    else:
        # Delta empieza con más pasos y reduce
        main(shots=5, tasks_per_batch=4, adapt_lr=0.01, meta_lr=0.001,
             adapt_steps_start=5, adapt_steps_end=1, anneal_epochs=100)