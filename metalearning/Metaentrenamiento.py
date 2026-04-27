"""
Módulo: train_meta_learning.py
Autor: Juan Marcos Grigolatto
Descripción: Script principal de Meta-Entrenamiento utilizando el algoritmo 
             MAML (Model-Agnostic Meta-Learning). Implementa dos paradigmas de 
             construcción de tareas (episodios) para evaluar la generalización:
             1. Modo 'traditional': Tareas formadas por ventanas aleatorias 
                sin distinción de sujeto (Línea Base / Baseline).
             2. Modo 'patient_wise': Tareas estrictamente separadas por paciente. 
                Obliga a la red a optimizar su inicialización para adaptarse 
                rápidamente a la fisiología de un nuevo individuo utilizando 
                pocos latidos de calibración (Support Set) y generalizar sobre 
                sus latidos futuros (Query Set).
"""
import os  
import csv 
import pandas as pd 
from src.data.data_chargers.MetaDataset import TaskDataset
from src.data.data_chargers.PatientWiseSet import PatientWiseDataset
#from src.models.InceptionTime import InceptionTime
from src.models.ConvolucionalV1 import Modelo_ConvolucionalV1
#from src.models.ConvolucionalV2 import Modelo_ConvolucionalV2
import numpy as np
import learn2learn as l2l
from torch import mode, nn, optim, seed
import matplotlib.pyplot as plt
import torch.utils.data as data
from tqdm.auto import tqdm 
import torch
import random
from src.data.data_chargers.Clase_UCIDataset import UCIDataset

CHECKPOINT_DIR = 'models/checkpoints'
LOG_DIR = 'metalearning/logs'

LATEST_CKPT_PATH = os.path.join(CHECKPOINT_DIR, 'checkpoint_latest_patientwise_s10q20_adapt5_PG2_reducido_2.pt')
BEST_CKPT_PATH = os.path.join(CHECKPOINT_DIR, 'best_meta_model_patientwise_s10q20_adapt5_PG2_reducido_2.pt')
CSV_LOG_PATH = os.path.join(LOG_DIR, 'training_log_patientwise_s10q20_adapt5_PG2_reducido_2.csv')

os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

def validate_meta_epoch(maml, val_loader, lossfn, adapt_steps, shots, device, mode='traditional'):
    """_summary_ Validación del modelo meta-entrenado al finalizar cada época. Se evalúa la capacidad de adaptación rápida a nuevas tareas (pacientes) utilizando un Support Set pequeño y generalizando sobre un Query Set.  

    Args:
        maml (_type_): _description_ es el meta-modelo que se va a evaluar. Se clona para cada tarea de validación y se adapta con el Support Set antes de evaluar en el Query Set.    
        val_loader (_type_): _description_ es el DataLoader que proporciona las tareas de validación. Cada batch contiene un conjunto de tareas (episodios) formados por Support Set y Query Set.  
        lossfn (_type_): _description_ es la función de pérdida utilizada para calcular el error en el Support Set durante la adaptación y en el Query Set durante la evaluación. En este caso, se utiliza MSELoss para regresión. 
        adapt_steps (_type_): _description_ es el número de pasos de adaptación (inner loop) que se realizan en cada tarea de validación. Durante estos pasos, el modelo se adapta utilizando el Support Set antes de evaluar en el Query Set.
        shots (_type_): _description_ es el número de muestras en el Support Set.
        device (_type_): _description_ es el dispositivo en el que se ejecuta el modelo (CPU o CUDA).
        mode (str, optional): _description_. Por defecto 'traditional'. Determina la forma en que se construyen las tareas de validación. En 'traditional', las tareas se forman por ventanas aleatorias de un único paciente. En 'patient_wise', las tareas se forman mezclando datos de diferentes pacientes, tanto en Support Set y como enQuery Set.

    Returns:
        _type_: _description_  
    """    
    meta_val_loss = 0.0
    num_batches = 0
    
    for batch in val_loader:
        if mode == 'traditional':
            x_batch, y_batch = batch
            if device.type == 'cuda':
                x_batch, y_batch = x_batch.to(device, non_blocking=True), y_batch.to(device, non_blocking=True)
            effective_batch_size = x_batch.size(0)
        else: 
            x_support_batch, y_support_batch, x_query_batch, y_query_batch = batch
            if device.type == 'cuda':
                x_support_batch = x_support_batch.to(device, non_blocking=True)
                y_support_batch = y_support_batch.to(device, non_blocking=True)
                x_query_batch = x_query_batch.to(device, non_blocking=True)
                y_query_batch = y_query_batch.to(device, non_blocking=True)
            effective_batch_size = x_support_batch.size(0)
            
        batch_loss = 0.0
        
        for i in range(effective_batch_size):
            learner = maml.clone()
            
            # Preparación de datos support/query
            if mode == 'traditional':
                x_task, y_task = x_batch[i], y_batch[i]
                x_support, y_support = x_task[:shots], y_task[:shots]
                x_query, y_query = x_task[shots:], y_task[shots:]
            else: # patient_wise
                x_support, y_support = x_support_batch[i], y_support_batch[i]
                x_query, y_query = x_query_batch[i], y_query_batch[i]
            
            # Inner Loop
            for _ in range(adapt_steps):
                support_preds = learner(x_support)
                support_loss = lossfn(support_preds, y_support)
                learner.adapt(support_loss)
            
            # Outer Loop
            with torch.no_grad():
                query_preds = learner(x_query)
                query_loss = lossfn(query_preds, y_query)
                batch_loss += query_loss.item()
        
        meta_val_loss += batch_loss / effective_batch_size
        num_batches += 1
        
    if num_batches == 0: return 0.0
    return meta_val_loss / num_batches

def save_checkpoint(state, is_best, filename=LATEST_CKPT_PATH):
    torch.save(state, filename)
    if is_best:
        torch.save(state, BEST_CKPT_PATH)

def log_to_csv(epoch, train_loss, valid_loss, filepath=CSV_LOG_PATH):
    """Guarda Epoch, Train Loss y Valid Loss"""
    file_exists = os.path.isfile(filepath)
    with open(filepath, mode='a', newline='') as f:
        writer = csv.writer(f)
        if not file_exists:
            # Cabecera correcta con 3 columnas
            writer.writerow(['epoch', 'train_loss', 'valid_loss'])
        writer.writerow([epoch, train_loss, valid_loss])

def plot_from_csv(csv_path):
    """Grafica comparativa de Train vs Validation"""
    try:
        df = pd.read_csv(csv_path)
        plt.figure(figsize=(8, 5), tight_layout=True)
        
        # Graficamos ambas curvas
        plt.plot(df['epoch'], df['train_loss'], marker='.', linestyle='-', label='Meta Train Loss', alpha=0.7)
        plt.plot(df['epoch'], df['valid_loss'], marker='o', linestyle='-', label='Meta Valid Loss', linewidth=2)
        
        plt.xlabel('Época')
        plt.ylabel('Loss (MSE)')
        plt.title('Curvas de Aprendizaje: Train vs Validation')
        plt.legend()
        plt.grid(True, which='both', linestyle='--', alpha=0.5)
        
        save_path = os.path.join(LOG_DIR, 'training_curves.png')
        plt.savefig(save_path)
        print(f"Gráfico guardado en: {save_path}")

    except Exception as e:
        print(f"No se pudo graficar desde CSV: {e}")

def main(mode='patient_wise', shots=5,tasks_per_batch=4, adapt_lr=0.01, meta_lr=0.001, adapt_steps=5, seed=42, num_epochs=500, patience=20, min_delta=1e-3,
         # Parametros modo patient_wise:
         N_patient_group=4, p_support=5, q_query=10):

    print(f"MODO SELECCIONADO: {mode.upper()}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Usando dispositivo: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    print(f"Fijando seed global: {seed}")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    data_paths = [
        'data/processed/data_UCI/dataset_parte_1_por_picos.pt',
        'data/processed/data_UCI/dataset_parte_2_por_picos.pt',
        'data/processed/data_UCI/dataset_parte_3_por_picos.pt',
        'data/processed/data_UCI/dataset_parte_4_por_picos.pt',
    ]

    print("Cargando datasets...")
    dataset_completo = UCIDataset(data_paths)

    all_pids = torch.tensor([dataset_completo[i][2] for i in range(len(dataset_completo))])
    unique_patients = all_pids.unique().tolist()

    random.shuffle(unique_patients)

    if mode == 'traditional':

        min_samples = shots * 2 
    else:
        
        min_samples = p_support + q_query
    
    patient_to_indices = {pid: [] for pid in unique_patients}
    for i in range(len(dataset_completo)):
        _, _, pid, _ = dataset_completo[i]
        patient_to_indices[int(pid)].append(i)
        
    valid_IDs_global = [pid for pid, idxs in patient_to_indices.items() if len(idxs) >= min_samples]
    unique_patients = valid_IDs_global

    total_patients = len(unique_patients)
    n_train = int(total_patients * 0.70)
    n_val = int(total_patients * 0.15)
    
    train_patients = unique_patients[:n_train]
    val_patients = unique_patients[n_train : n_train + n_val] # PACIENTES DE VALIDACIÓN
    test_patients = unique_patients[n_train + n_val:]
    
    print(f"Total: {total_patients} | Train: {len(train_patients)} | Val: {len(val_patients)} | Test: {len(test_patients)}")
    # Guardar lista de pacientes para evaluación few-shot
    torch.save({'test_patient_ids': test_patients}, 'data/processed/data_UCI/few_shot_patient_data.pt')

    if mode == 'traditional':
        train_tasksets = TaskDataset(list_IDs=train_patients, base_dataset=dataset_completo, num_shots=shots)
        val_tasksets = TaskDataset(list_IDs=val_patients, base_dataset=dataset_completo, num_shots=shots)
    elif mode == 'patient_wise':
        train_tasksets = PatientWiseDataset(list_IDs=train_patients, base_dataset=dataset_completo, 
                                            N_patients=N_patient_group, p_support=p_support, q_query=q_query)
        val_tasksets = PatientWiseDataset(list_IDs=val_patients, base_dataset=dataset_completo, 
                                          N_patients=N_patient_group, p_support=p_support, q_query=q_query)
    else:
        raise ValueError("Mode debe ser 'traditional' o 'patient_wise'")

    # Dataloaders
    train_loader = data.DataLoader(train_tasksets, batch_size=tasks_per_batch, shuffle=True, num_workers=0, pin_memory=True, drop_last=True)
    val_loader = data.DataLoader(val_tasksets, batch_size=tasks_per_batch, shuffle=False, num_workers=0, pin_memory=True, drop_last=True)
 
    model=Modelo_ConvolucionalV1(in_channels=2,out_channels=2, long_signal=500)
    model.to(device)
    maml = l2l.algorithms.MAML(model, lr=adapt_lr, first_order=True, allow_unused=True)
    opt = optim.Adam(maml.parameters(), meta_lr)
    lossfn = nn.MSELoss(reduction='mean')

    start_epoch = 0
    patience_counter = 0
    best_valid_loss = float('inf')

    
    if os.path.exists(LATEST_CKPT_PATH):
        print(f"Checkpoint encontrado en {LATEST_CKPT_PATH}. Reanudando...")
        checkpoint = torch.load(LATEST_CKPT_PATH)
        maml.load_state_dict(checkpoint['model_state_dict']) 
        opt.load_state_dict(checkpoint['optimizer_state_dict']) 
        best_valid_loss = checkpoint['best_loss']
        patience_counter = checkpoint.get('patience_counter', 0)
        print(f"Reanudando desde época {start_epoch}. Mejor Loss anterior: {best_valid_loss:.4f}")
    else:
        print("Iniciando entrenamiento desde cero.")
        if os.path.exists(CSV_LOG_PATH):
            os.remove(CSV_LOG_PATH) 

    # Entrenamiento del modelo

    for epoch in range(start_epoch, num_epochs):  
        print(f"\n--- Época {epoch+1}/{num_epochs} ---")
        epoch_loss_accum = 0.0   
        num_batches = 0
        
        with tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}") as pbar:
            for batch in pbar:
                meta_train_loss = 0.0
                
                if mode == 'traditional':
                    x_batch, y_batch = batch
                    if device.type == 'cuda':
                        x_batch, y_batch = x_batch.to(device, non_blocking=True), y_batch.to(device, non_blocking=True)
                    effective_batch_size = x_batch.size(0)
                else: # patient_wise
                    x_sup_batch, y_sup_batch, x_qry_batch, y_qry_batch = batch
                    if device.type == 'cuda':
                         x_sup_batch, y_sup_batch = x_sup_batch.to(device, non_blocking=True), y_sup_batch.to(device, non_blocking=True)
                         x_qry_batch, y_qry_batch = x_qry_batch.to(device, non_blocking=True), y_qry_batch.to(device, non_blocking=True)
                    effective_batch_size = x_sup_batch.size(0)

                for i in range(effective_batch_size):
                    learner = maml.clone()

                    if mode == 'traditional':
                        x_task, y_task = x_batch[i], y_batch[i]
                        x_support, y_support = x_task[:shots], y_task[:shots]
                        x_query, y_query = x_task[shots:], y_task[shots:]
                    else: # patient_wise
                        x_support, y_support = x_sup_batch[i], y_sup_batch[i]
                        x_query, y_query = x_qry_batch[i], y_qry_batch[i]

                    # Inner loop
                    for _ in range(adapt_steps): 
                        support_preds = learner(x_support)
                        support_loss = lossfn(support_preds, y_support)
                        learner.adapt(support_loss)

                    # Outer Loop
                    query_preds = learner(x_query)
                    query_loss = lossfn(query_preds, y_query)
                    meta_train_loss += query_loss

                meta_train_loss = meta_train_loss / effective_batch_size
                epoch_loss_accum += meta_train_loss.item()
                num_batches += 1
                pbar.set_postfix({'batch_loss': meta_train_loss.item()})

                opt.zero_grad()
                meta_train_loss.backward()
                opt.step()

        #Calculo de perdida promedio por época
        epoch_train_loss = epoch_loss_accum / num_batches
        print(f"Train Loss: {epoch_train_loss:.4f}")

        print("Validando...")
        shots_arg = shots if mode == 'traditional' else p_support
        valid_loss = validate_meta_epoch(maml, val_loader, lossfn, adapt_steps, shots_arg, device, mode=mode)
        print(f"Valid Loss: {valid_loss:.4f}")

        log_to_csv(epoch+1, epoch_train_loss, valid_loss)

        # Chequeo de mejor modelo
        is_best = valid_loss < (best_valid_loss - min_delta)
        if is_best:
            best_valid_loss = valid_loss
            patience_counter = 0
            print(f" Nuevo mejor modelo (Loss: {best_valid_loss:.4f})")
        else:
            patience_counter += 1 
            print(f" No hubo mejora significativa. Paciencia: {patience_counter}/{patience}")

        save_checkpoint({
            'epoch': epoch,
            'model_state_dict': maml.state_dict(),
            'optimizer_state_dict': opt.state_dict(),
            'best_loss': best_valid_loss,
            'patience_counter': patience_counter,
            'config': {'shots': shots, 'seed': seed}
        }, is_best)

        if patience_counter >= patience:
            print(f"\n[EARLY STOPPING] Se ha detenido el entrenamiento.")
            print(f"No hubo mejora de {min_delta} en las últimas {patience} épocas.")
            print(f"Mejor loss obtenido: {best_valid_loss:.4f}")
            break

    print("Entrenamiento completado.")

    plot_from_csv(CSV_LOG_PATH)
    
if __name__ == '__main__':
    main(mode='patient_wise', shots=10,tasks_per_batch=4, adapt_lr=0.005, meta_lr=0.001, adapt_steps=5, seed=42, num_epochs=500, patience=20, min_delta=1e-3,
         # Parametros modo patient_wise:
         N_patient_group=2, p_support=10, q_query=20)        