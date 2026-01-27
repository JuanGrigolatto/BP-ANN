import os  
import csv 
import pandas as pd 
import numpy as np
import learn2learn as l2l
from torch import mode, nn, optim, seed
import matplotlib.pyplot as plt
import torch.utils.data as data
from tqdm.auto import tqdm 
import torch
import random
from collections import defaultdict
from src.data.data_chargers.Clase_UCIDataset import UCIDataset
from src.models.ConvolucionalV1 import Modelo_ConvolucionalV1
from src.data.data_chargers.MetaDataset import TaskDataset 

# --- CONFIGURACIÓN DE DIRECTORIOS ---
CHECKPOINT_DIR = 'models/checkpoints'
LOG_DIR = 'metalearning/logs'

# Nombre del experimento para identificar los archivos
EXPERIMENT_NAME = 'meta_intrapatient_GAP50_HYBRID_ANNEAL_5_epochs'
LATEST_CKPT_PATH = os.path.join(CHECKPOINT_DIR, f'checkpoint_{EXPERIMENT_NAME}.pt')
BEST_CKPT_PATH = os.path.join(CHECKPOINT_DIR, f'best_{EXPERIMENT_NAME}.pt')
CSV_LOG_PATH = os.path.join(LOG_DIR, f'log_{EXPERIMENT_NAME}.csv')

os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

# =============================================================================
# 1. FUNCIÓN DE PÉRDIDA HÍBRIDA (MSE + PEARSON)
# =============================================================================
class HybridLoss(nn.Module):
    def __init__(self, alpha=0.5):
        super().__init__()
        self.mse = nn.MSELoss()
        self.alpha = alpha # 0.5 equilibra valor absoluto y forma

    def forward(self, pred, target):
        # A. Error de Valor (MSE)
        mse_loss = self.mse(pred, target)
        
        # B. Error de Dinámica (Pearson Correlation)
        vx = pred - torch.mean(pred)
        vy = target - torch.mean(target)
        
        # Correlación (evitando división por cero)
        cost = torch.sum(vx * vy) / (torch.sqrt(torch.sum(vx ** 2)) * torch.sqrt(torch.sum(vy ** 2)) + 1e-6)
        
        # Maximizamos correlación => Minimizamos (1 - r)
        pearson_loss = 1 - cost
        
        return (1 - self.alpha) * mse_loss + self.alpha * pearson_loss

# =============================================================================
# 2. UTILIDADES DE VALIDACIÓN Y LOGGING
# =============================================================================
def validate_meta_epoch(maml, val_loader, lossfn, adapt_steps, shots, device):
    meta_val_loss = 0.0
    num_batches = 0
    
    for batch in val_loader:
        x_batch, y_batch = batch
        if device.type == 'cuda':
            x_batch, y_batch = x_batch.to(device, non_blocking=True), y_batch.to(device, non_blocking=True)
        
        effective_batch_size = x_batch.size(0)
        batch_loss = 0.0
        
        for i in range(effective_batch_size):
            learner = maml.clone()
            x_task, y_task = x_batch[i], y_batch[i]
            
            # Tu TaskDataset ya devuelve [Support ... Query] ordenados
            x_support, y_support = x_task[:shots], y_task[:shots]
            x_query, y_query = x_task[shots:], y_task[shots:]
            
            # Validación estándar (solo evaluamos al final)
            for _ in range(adapt_steps):
                support_preds = learner(x_support)
                support_loss = lossfn(support_preds, y_support)
                learner.adapt(support_loss)
            
            with torch.no_grad():
                query_preds = learner(x_query)
                query_loss = lossfn(query_preds, y_query)
                batch_loss += query_loss.item()
        
        meta_val_loss += batch_loss / effective_batch_size
        num_batches += 1
        
    if num_batches == 0: return 0.0
    return meta_val_loss / num_batches

def log_to_csv(epoch, train_loss, valid_loss, filepath=CSV_LOG_PATH):
    file_exists = os.path.isfile(filepath)
    with open(filepath, mode='a', newline='') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(['epoch', 'train_loss', 'valid_loss'])
        writer.writerow([epoch, train_loss, valid_loss])

def save_checkpoint(state, is_best, filename=LATEST_CKPT_PATH):
    torch.save(state, filename)
    if is_best:
        torch.save(state, BEST_CKPT_PATH)

def plot_from_csv(csv_path):
    try:
        df = pd.read_csv(csv_path)
        plt.figure(figsize=(8, 5))
        plt.plot(df['epoch'], df['train_loss'], label='Train (Hybrid)')
        plt.plot(df['epoch'], df['valid_loss'], label='Valid (Hybrid)')
        plt.title('Entrenamiento Meta-Learning Híbrido')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(LOG_DIR, 'training_curves.png'))
        print("Curvas guardadas.")
    except: pass

# =============================================================================
# 3. MAIN - BUCLE DE ENTRENAMIENTO
# =============================================================================
def main(shots=10, 
         gap=50,               # GAP TEMPORAL: Parámetro para tu TaskDataset
         tasks_per_batch=4, 
         adapt_lr=0.005, 
         meta_lr=0.001, 
         adapt_steps=10,       # Pasos iniciales (Jia)
         k_min_steps=2,        # Pasos finales (Jia)
         anneal_stride=12,     # Reducción cada 12 épocas
         seed=42, 
         num_epochs=150,       
         patience=20, 
         min_delta=1e-3):

    print(f"--- INICIANDO META-TRAINING MEJORADO ---")
    print(f"Estrategia: Intra-Patient con GAP={gap} | Loss: Hybrid | Opt: Annealing")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Dispositivo: {device}")
    
    # Semillas
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

    # Carga de Datos Base
    data_paths = [
        'data/processed/data_UCI/dataset_parte_1_por_picos.pt',
        'data/processed/data_UCI/dataset_parte_2_por_picos.pt',
        'data/processed/data_UCI/dataset_parte_3_por_picos.pt',
        'data/processed/data_UCI/dataset_parte_4_por_picos.pt',
    ]
    print("Cargando dataset base...")
    dataset_completo = UCIDataset(data_paths)

    # Split de Pacientes
    # Usamos la misma lógica de filtrado que tienes en tu clase TaskDataset
    # para saber cuántos pacientes válidos hay antes de splitear
    min_required = (2 * shots) + gap
    
    # Pre-calcular índices para filtrar rápido y hacer el split Train/Val
    temp_indices = defaultdict(list)
    for i in range(len(dataset_completo)):
        temp_indices[int(dataset_completo[i][2])].append(i)
        
    valid_patients = [pid for pid in temp_indices.keys() if len(temp_indices[pid]) >= min_required]
    random.shuffle(valid_patients) # Mezclar antes de dividir
    
    n_train = int(len(valid_patients) * 0.70)
    n_val = int(len(valid_patients) * 0.15)
    
    train_pids = valid_patients[:n_train]
    val_pids = valid_patients[n_train : n_train + n_val]
    test_pids = valid_patients[n_train + n_val:]
    
    print(f"Pacientes Válidos: {len(valid_patients)}")
    print(f"Split: Train {len(train_pids)} | Val {len(val_pids)} | Test {len(test_pids)}")
    
    # Guardamos los IDs de test para usarlos luego en el script de evaluación
    torch.save({'test_patient_ids': test_pids}, 'data/processed/data_UCI/few_shot_patient_data.pt')

    # --- INSTANCIAR TU TASK DATASET ---
    # Aquí pasamos el parámetro 'gap' que definiste en tu archivo
    train_set = TaskDataset(train_pids, base_dataset=dataset_completo, num_shots=shots, gap=gap)
    val_set = TaskDataset(val_pids, base_dataset=dataset_completo, num_shots=shots, gap=gap)

    train_loader = data.DataLoader(train_set, batch_size=tasks_per_batch, shuffle=True, num_workers=0)
    val_loader = data.DataLoader(val_set, batch_size=tasks_per_batch, shuffle=False, num_workers=0)
 
    # Modelo
    model = Modelo_ConvolucionalV1(in_channels=2, out_channels=2, long_signal=500)
    model.to(device)
    maml = l2l.algorithms.MAML(model, lr=adapt_lr, first_order=True, allow_unused=True)
    opt = optim.Adam(maml.parameters(), meta_lr)
    
    # LOSS HÍBRIDA
    lossfn = HybridLoss(alpha=0.5).to(device)

    # Estado inicial
    start_epoch = 0
    patience_counter = 0
    best_valid_loss = float('inf')

    # Reanudar si existe checkpoint
    if os.path.exists(LATEST_CKPT_PATH):
        print(f"Reanudando desde {LATEST_CKPT_PATH}")
        ckpt = torch.load(LATEST_CKPT_PATH)
        maml.load_state_dict(ckpt['model_state_dict']) 
        opt.load_state_dict(ckpt['optimizer_state_dict']) 
        start_epoch = ckpt['epoch'] + 1
        best_valid_loss = ckpt['best_loss']

    # --- BUCLE PRINCIPAL ---
    for epoch in range(start_epoch, num_epochs):  
        print(f"\n--- Epoch {epoch+1}/{num_epochs} ---")
        
        # 1. ANNEALING (Estrategia Jia et al.)
        current_steps = max(adapt_steps - (epoch // anneal_stride), k_min_steps)
        print(f"  -> Inner Steps: {current_steps}")

        epoch_loss_accum = 0.0   
        num_batches = 0
        
        with tqdm(train_loader, desc=f"Train") as pbar:
            for batch in pbar:
                x_batch, y_batch = batch
                if device.type == 'cuda':
                    x_batch, y_batch = x_batch.to(device, non_blocking=True), y_batch.to(device, non_blocking=True)
                
                bs = x_batch.size(0)
                batch_total_loss = 0.0

                for i in range(bs):
                    learner = maml.clone()
                    x_task, y_task = x_batch[i], y_batch[i]
                    
                    # Tu TaskDataset ya entrega la estructura: [Support | Query]
                    # Solo necesitamos cortar en el índice 'shots'
                    x_support, y_support = x_task[:shots], y_task[:shots]
                    x_query, y_query = x_task[shots:], y_task[shots:]

                    # 2. MULTI-STEP LOSS (Estrategia Jia et al.)
                    task_step_losses = []
                    for _ in range(current_steps): 
                        # Adaptar con Support
                        sup_preds = learner(x_support)
                        sup_loss = lossfn(sup_preds, y_support)
                        learner.adapt(sup_loss)
                        
                        # Evaluar con Query (y acumular loss en CADA paso)
                        qry_preds = learner(x_query)
                        qry_loss = lossfn(qry_preds, y_query)
                        task_step_losses.append(qry_loss)

                    # Promedio de losses de todos los pasos para estabilizar
                    batch_total_loss += torch.stack(task_step_losses).mean()

                # Actualización Meta-Modelo (Outer Loop)
                meta_loss = batch_total_loss / bs
                epoch_loss_accum += meta_loss.item()
                num_batches += 1
                
                opt.zero_grad()
                meta_loss.backward()
                opt.step()
                
                pbar.set_postfix({'loss': meta_loss.item()})

        # Métricas
        train_loss_avg = epoch_loss_accum / num_batches
        print(f"Train Loss: {train_loss_avg:.4f}")

        # Validación
        print("Validando...")
        valid_loss = validate_meta_epoch(maml, val_loader, lossfn, adapt_steps, shots, device)
        print(f"Valid Loss: {valid_loss:.4f}")

        log_to_csv(epoch+1, train_loss_avg, valid_loss)

        # Checkpointing & Early Stopping
        is_best = valid_loss < (best_valid_loss - min_delta)
        if is_best:
            best_valid_loss = valid_loss
            patience_counter = 0
            print(f" ** Nuevo Récord **")
        else:
            patience_counter += 1
            print(f" Sin mejora. Paciencia: {patience_counter}/{patience}")

        save_checkpoint({
            'epoch': epoch,
            'model_state_dict': maml.state_dict(),
            'optimizer_state_dict': opt.state_dict(),
            'best_loss': best_valid_loss,
            'patience_counter': patience_counter
        }, is_best)

        if patience_counter >= patience:
            print(f"EARLY STOPPING ACTIVADO.")
            break

    plot_from_csv(CSV_LOG_PATH)
    print("Experimento finalizado.")

if __name__ == '__main__':
    main(shots=10, 
         gap=50,               
         tasks_per_batch=4, 
         adapt_lr=0.01, 
         meta_lr=0.001, 
         adapt_steps=10,       
         k_min_steps=2,        
         anneal_stride=5,     
         num_epochs=150, 
         patience=20)