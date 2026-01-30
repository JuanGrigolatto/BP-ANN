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
from collections import defaultdict

# --- TUS IMPORTACIONES ---
from src.data.data_chargers.Clase_UCIDataset import UCIDataset
from src.models.ConvolucionalV1 import Modelo_ConvolucionalV1
# Asumimos que TaskDataset ya tiene el parámetro 'gap' implementado (como confirmamos antes)
from src.data.data_chargers.MetaDataset import TaskDataset 

# --- CONFIGURACIÓN DE RUTAS ---
EXPERIMENT_NAME = 'meta_DELTA_LEARNING_refine_alpha50'
PRETRAINED_MODEL_PATH = 'models/checkpoints/checkpoint_meta_DELTA_LEARNING_fast_anneal_k100_alpha90.pt' 
CHECKPOINT_DIR = 'models/checkpoints'
LOG_DIR = 'metalearning/logs'
LATEST_CKPT_PATH = os.path.join(CHECKPOINT_DIR, f'checkpoint_{EXPERIMENT_NAME}.pt')
BEST_CKPT_PATH = os.path.join(CHECKPOINT_DIR, f'best_{EXPERIMENT_NAME}.pt')
CSV_LOG_PATH = os.path.join(LOG_DIR, f'log_{EXPERIMENT_NAME}.csv')
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

# 1. HYBRID LOSS (MSE + PEARSON)
# Nota: Esta Loss funciona perfecto con Deltas porque Pearson es invariante a la escala/offset
class HybridLoss(nn.Module):
    def __init__(self, alpha=0.5):
        super().__init__()
        self.mse = nn.MSELoss()
        self.alpha = alpha 

    def forward(self, pred, target):
        # MSE mide el error en la variación predicha vs variación real
        mse_loss = self.mse(pred, target)
        
        # Pearson mide si la forma de la variación coincide
        vx = pred - torch.mean(pred)
        vy = target - torch.mean(target)
        cost = torch.sum(vx * vy) / (torch.sqrt(torch.sum(vx ** 2)) * torch.sqrt(torch.sum(vy ** 2)) + 1e-6)
        pearson_loss = 1 - cost
        
        return (1 - self.alpha) * mse_loss + self.alpha * pearson_loss

# 2. VALIDACIÓN (ADAPTADA PARA DELTA LEARNING)
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
            x_support, y_support = x_task[:shots], y_task[:shots]
            x_query, y_query = x_task[shots:], y_task[shots:]
            
            # ---------------------------------------------------------
            # ### !!! ATENCIÓN: APLICACIÓN DE DELTA LEARNING EN VAL !!!
            # ---------------------------------------------------------
            # 1. Calculamos el Bias (Promedio) del Support Set.
            #    Este es el valor "base" de presión del paciente.
            bias = y_support.mean(dim=0, keepdim=True)
            
            # 2. Centramos los datos.
            #    Ahora 'y_support_centered' tiene media ~0. 
            #    El modelo debe predecir fluctuaciones, no valores absolutos.
            y_support_centered = y_support - bias
            
            # 3. Centramos el Query con el MISMO bias del Support.
            #    Esto es clave: queremos ver si el modelo predice bien la variación futura
            #    usando la referencia pasada.
            y_query_centered = y_query - bias 
            
            # Adaptación (Inner Loop) con datos centrados
            for _ in range(adapt_steps):
                support_preds = learner(x_support)
                support_loss = lossfn(support_preds, y_support_centered)
                learner.adapt(support_loss)
            
            # Evaluación (Outer Loop) con datos centrados
            with torch.no_grad():
                query_preds = learner(x_query)
                query_loss = lossfn(query_preds, y_query_centered)
                batch_loss += query_loss.item()
        
        meta_val_loss += batch_loss / effective_batch_size
        num_batches += 1
        
    if num_batches == 0: return 0.0
    return meta_val_loss / num_batches

# Funciones auxiliares de log y guardado
def log_to_csv(epoch, train_loss, valid_loss, filepath=CSV_LOG_PATH):
    file_exists = os.path.isfile(filepath)
    with open(filepath, mode='a', newline='') as f:
        writer = csv.writer(f)
        if not file_exists: writer.writerow(['epoch', 'train_loss', 'valid_loss'])
        writer.writerow([epoch, train_loss, valid_loss])

def save_checkpoint(state, is_best, filename=LATEST_CKPT_PATH):
    torch.save(state, filename)
    if is_best: torch.save(state, BEST_CKPT_PATH)

# -----------------------------------------------------------------------------
# MAIN: BUCLE DE ENTRENAMIENTO PRINCIPAL
# -----------------------------------------------------------------------------
def main(shots=5, 
         gap=50, 
         tasks_per_batch=4, 
         adapt_lr=0.005, 
         meta_lr=0.0005, 
         
        
         adapt_steps_start=5,  
         adapt_steps_end=1,     
         anneal_epochs=100,      
         
         seed=42, 
         num_epochs=150, 
         patience=20):

    # Cálculo del Stride para el Annealing

    if adapt_steps_start > adapt_steps_end:
        anneal_stride = max(1, anneal_epochs // (adapt_steps_start - adapt_steps_end))
    else:
        anneal_stride = 9999 # Si start <= end, no hacemos annealing

    print(f"--- DELTA LEARNING META-TRAINING ---")
    print(f"Estrategia Annealing: {adapt_steps_start} -> {adapt_steps_end} pasos.")
    print(f"Velocidad de bajada: 1 paso cada {anneal_stride} épocas.")
    print(f"Objetivo: Predecir VARIACIONES (y - bias).")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Seeds
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

    # Carga de Datos
    data_paths = [
        'data/processed/data_UCI/dataset_parte_1_por_picos.pt',
        'data/processed/data_UCI/dataset_parte_2_por_picos.pt',
        'data/processed/data_UCI/dataset_parte_3_por_picos.pt',
        'data/processed/data_UCI/dataset_parte_4_por_picos.pt',
    ]
    dataset_completo = UCIDataset(data_paths)

    # Filtrado de Pacientes (Shots + Gap + Shots)
    min_required = (2 * shots) + gap
    temp_indices = defaultdict(list)
    for i in range(len(dataset_completo)):
        temp_indices[int(dataset_completo[i][2])].append(i)
    
    valid_patients = [pid for pid in temp_indices.keys() if len(temp_indices[pid]) >= min_required]
    random.shuffle(valid_patients)
    
    # Split
    n_train = int(len(valid_patients) * 0.70)
    train_pids = valid_patients[:n_train]
    val_pids = valid_patients[n_train : int(len(valid_patients)*0.85)]
    test_pids = valid_patients[int(len(valid_patients)*0.85):] 
    
    # Guardamos IDs de test para luego
    torch.save({'test_patient_ids': test_pids}, 'data/processed/data_UCI/few_shot_patient_data.pt')

    # Dataloaders (Usando tu TaskDataset externo)
    train_set = TaskDataset(train_pids, base_dataset=dataset_completo, num_shots=shots, gap=gap)
    val_set = TaskDataset(val_pids, base_dataset=dataset_completo, num_shots=shots, gap=gap)
    
    train_loader = data.DataLoader(train_set, batch_size=tasks_per_batch, shuffle=True, num_workers=0)
    val_loader = data.DataLoader(val_set, batch_size=tasks_per_batch, shuffle=False, num_workers=0)
 
    # Modelo y MAML
    model = Modelo_ConvolucionalV1(in_channels=2, out_channels=2, long_signal=500).to(device)
    maml = l2l.algorithms.MAML(model, lr=adapt_lr, first_order=True, allow_unused=True)
    opt = optim.Adam(maml.parameters(), meta_lr)
    lossfn = HybridLoss(alpha=0.75).to(device)

    start_epoch = 0
    best_valid_loss = float('inf')
    patience_counter = 0
    
    if os.path.exists(LATEST_CKPT_PATH):
        print(f"Reanudando refinamiento desde: {LATEST_CKPT_PATH}")
        ckpt = torch.load(LATEST_CKPT_PATH)
        maml.load_state_dict(ckpt['model_state_dict']) 
        opt.load_state_dict(ckpt['optimizer_state_dict']) 
        start_epoch = ckpt['epoch'] + 1
        best_valid_loss = ckpt['best_loss']
        
    # B) Si NO existe, cargamos el modelo PRE-ENTRENADO (Fase 1)
    elif os.path.exists(PRETRAINED_MODEL_PATH):
        print(f"⚠️ CARGANDO PESOS DE FASE 1: {PRETRAINED_MODEL_PATH}")
        print("--> Reiniciando optimizador y épocas para Fine-Tuning.")
        
        pretrained_ckpt = torch.load(PRETRAINED_MODEL_PATH)
        
        # Solo cargamos los pesos del modelo (maml), NO el optimizador
        maml.load_state_dict(pretrained_ckpt['model_state_dict'])
        
        # start_epoch se mantiene en 0
        # best_valid_loss se mantiene en infinito para forzar nuevos guardados
        
    else:
        print("¡OJO! No se encontró ni checkpoint nuevo ni pre-entrenado. Iniciando desde cero (NO RECOMENDADO).")
    
    """
    # Reanudar si existe checkpoint
    if os.path.exists(LATEST_CKPT_PATH):
        ckpt = torch.load(LATEST_CKPT_PATH)
        maml.load_state_dict(ckpt['model_state_dict']) 
        opt.load_state_dict(ckpt['optimizer_state_dict']) 
        start_epoch = ckpt['epoch'] + 1
        best_valid_loss = ckpt['best_loss']
        print(f"Reanudando desde época {start_epoch}")
    """
    # --- BUCLE DE ÉPOCAS ---
    for epoch in range(start_epoch, num_epochs):  
        
        # 1. CÁLCULO DINÁMICO DE PASOS (ANNEALING)
        steps_drop = epoch // anneal_stride
        # Nos aseguramos de no bajar de 'adapt_steps_end'
        current_steps = max(adapt_steps_start - steps_drop, adapt_steps_end)
        
        print(f"\nEpoch {epoch+1}/{num_epochs} | Inner Steps: {current_steps}")

        epoch_loss_accum = 0.0; num_batches = 0
        
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
                    x_support, y_support = x_task[:shots], y_task[:shots]
                    x_query, y_query = x_task[shots:], y_task[shots:]

                    # ------------------------------------------------------
                    # ### !!! CORAZÓN DEL DELTA LEARNING !!! ###
                    # ------------------------------------------------------
                    
                    # 1. Extraer el Bias (Promedio del Support)
                    #    El modelo "ve" el Support y calcula: "OK, el nivel base es 120/80"
                    bias = y_support.mean(dim=0, keepdim=True)
                    
                    # 2. Centrar Support
                    #    Al restar el bias, le decimos al modelo:
                    #    "No aprendas 120. Aprende que aquí la presión subió +2 respecto a la base".
                    y_support_centered = y_support - bias
                    
                    # 3. Centrar Query
                    #    Le pedimos al modelo que prediga la variación futura respecto a ESE MISMO bias base.
                    #    Esto prohíbe que el modelo simplemente 'copie' el promedio del support, 
                    #    porque el promedio de 'y_query_centered' no es necesariamente 0.
                    y_query_centered = y_query - bias 

                    # ------------------------------------------------------
                    # MULTI-STEP LOSS (Estrategia Jia et al.)
                    # ------------------------------------------------------
                    task_step_losses = []
                    for _ in range(current_steps): 
                        # Adaptar con datos CENTRADOS
                        sup_preds = learner(x_support)
                        sup_loss = lossfn(sup_preds, y_support_centered)
                        learner.adapt(sup_loss)
                        
                        # Evaluar con datos CENTRADOS
                        qry_preds = learner(x_query)
                        qry_loss = lossfn(qry_preds, y_query_centered)
                        task_step_losses.append(qry_loss)

                    # Promedio de todas las pérdidas de los pasos
                    batch_total_loss += torch.stack(task_step_losses).mean()

                # Actualización Meta-Modelo (Outer Loop)
                meta_loss = batch_total_loss / bs
                epoch_loss_accum += meta_loss.item()
                num_batches += 1
                
                opt.zero_grad()
                meta_loss.backward()
                opt.step()
                pbar.set_postfix({'loss': meta_loss.item()})

        train_loss = epoch_loss_accum / num_batches
        print(f"Train Loss: {train_loss:.4f}")

        # Validación
        valid_loss = validate_meta_epoch(maml, val_loader, lossfn, adapt_steps=current_steps, shots=shots, device=device)
        print(f"Valid Loss: {valid_loss:.4f}")
        log_to_csv(epoch+1, train_loss, valid_loss)

        # Checkpointing
        if valid_loss < (best_valid_loss - 1e-3):
            best_valid_loss = valid_loss
            patience_counter = 0
            save_checkpoint({
                'epoch': epoch,
                'model_state_dict': maml.state_dict(),
                'optimizer_state_dict': opt.state_dict(),
                'best_loss': best_valid_loss
            }, is_best=True)
            print(" ** Nuevo Récord **")
        else:
            patience_counter += 1
            save_checkpoint({
                'epoch': epoch,
                'model_state_dict': maml.state_dict(),
                'optimizer_state_dict': opt.state_dict(),
                'best_loss': best_valid_loss
            }, is_best=False)
            print(f" Paciencia: {patience_counter}/{patience}")

        if patience_counter >= patience:
            print("Early Stopping."); break

if __name__ == '__main__':
    main()