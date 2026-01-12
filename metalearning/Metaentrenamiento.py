import os  
import csv 
import pandas as pd 
from src.data.data_chargers.MetaDataset import TaskDataset
#from src.models.Modelo_conv import Modelo_Convolucional
#from src.models.InceptionTime import InceptionTime
from src.models.ConvolucionalV1 import Modelo_ConvolucionalV1
#from src.models.ConvolucionalV2 import Modelo_ConvolucionalV2
import numpy as np
import learn2learn as l2l
from torch import nn, optim, seed
import matplotlib.pyplot as plt
import torch.utils.data as data
from tqdm.auto import tqdm 
import torch
import random
from src.data.data_chargers.Clase_UCIDataset import UCIDataset

CHECKPOINT_DIR = 'models/checkpoints'
LOG_DIR = 'metalearning/logs'
LATEST_CKPT_PATH = os.path.join(CHECKPOINT_DIR, 'checkpoint_latest.pt')
BEST_CKPT_PATH = os.path.join(CHECKPOINT_DIR, 'best_meta_model_v1.pt')
CSV_LOG_PATH = os.path.join(LOG_DIR, 'training_log.csv')

os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

def validate_meta_epoch(maml, val_loader, lossfn, adapt_steps, shots, device):
    meta_val_loss = 0.0
    num_batches = 0
    
    # Importante: No queremos guardar gradientes del modelo PRINCIPAL, 
    # pero el learner.adapt SI necesita gradientes locales.
    # Por eso NO usamos 'with torch.no_grad():' globalmente aquí, es truculento en MAML.
    
    for batch in val_loader:
        x_batch, y_batch = batch
        if device.type == 'cuda':
            x_batch, y_batch = x_batch.to(device, non_blocking=True), y_batch.to(device, non_blocking=True)
            
        effective_batch_size = x_batch.size(0)
        batch_loss = 0.0
        
        for i in range(effective_batch_size):
            # Clonamos el modelo para validación (esto no afecta al modelo entrenado)
            learner = maml.clone()
            
            x_task, y_task = x_batch[i], y_batch[i]
            x_support, y_support = x_task[:shots], y_task[:shots]
            x_query, y_query = x_task[shots:], y_task[shots:]
            
            # Adaptación Rápida (Inner Loop) en el set de validación
            # Esto simula: "¿Qué tan bien aprende el modelo con un paciente nuevo?"
            for _ in range(adapt_steps):
                support_preds = learner(x_support)
                support_loss = lossfn(support_preds, y_support)
                learner.adapt(support_loss)
            
            # Evaluar en Query (Outer Loop)
            # Desactivamos gradientes aquí porque solo queremos medir el error, no entrenar
            with torch.no_grad():
                query_preds = learner(x_query)
                query_loss = lossfn(query_preds, y_query)
                batch_loss += query_loss.item()
        
        meta_val_loss += batch_loss / effective_batch_size
        num_batches += 1
        
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

def main(shots=5,tasks_per_batch=4, adapt_lr=0.01, meta_lr=0.001, adapt_steps=5, seed=42, num_epochs=500, patience=20, min_delta=1e-3):

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

    #unique_patients = merged_data['patient_ids'].unique().tolist()
    all_pids = torch.tensor([dataset_completo[i][2] for i in range(len(dataset_completo))])
    unique_patients = all_pids.unique().tolist()

    random.shuffle(unique_patients)
    
    total_patients = len(unique_patients)
    n_train = int(total_patients * 0.70)
    n_val = int(total_patients * 0.15)
    
    train_patients = unique_patients[:n_train]
    val_patients = unique_patients[n_train : n_train + n_val] # PACIENTES DE VALIDACIÓN
    test_patients = unique_patients[n_train + n_val:]
    
    print(f"Total: {total_patients} | Train: {len(train_patients)} | Val: {len(val_patients)} | Test: {len(test_patients)}")
    # Guardar lista de pacientes para evaluación few-shot
    torch.save({'test_patient_ids': test_patients}, 'data/processed/data_UCI/few_shot_patient_data.pt')

    train_tasksets = TaskDataset(list_IDs=train_patients, base_dataset=dataset_completo, num_shots=shots)
    train_loader = data.DataLoader(train_tasksets, batch_size=tasks_per_batch, shuffle=True, num_workers=0, pin_memory=True)
    
    val_tasksets = TaskDataset(list_IDs=val_patients, base_dataset=dataset_completo, num_shots=shots)
  
    val_loader = data.DataLoader(val_tasksets, batch_size=tasks_per_batch, shuffle=False, num_workers=0, pin_memory=True)
 
    model=Modelo_ConvolucionalV1(in_channels=2,out_channels=2, long_signal=500)
    model.to(device)
    maml = l2l.algorithms.MAML(model, lr=adapt_lr, first_order=True, allow_unused=True)
    opt = optim.Adam(maml.parameters(), meta_lr)
    lossfn = nn.MSELoss(reduction='mean')

    start_epoch = 0
    patience_counter = 0
    best_valid_loss = float('inf')

    # Verificamos si existe un archivo de "último estado"
    if os.path.exists(LATEST_CKPT_PATH):
        print(f"Checkpoint encontrado en {LATEST_CKPT_PATH}. Reanudando...")
        checkpoint = torch.load(LATEST_CKPT_PATH)
        maml.load_state_dict(checkpoint['model_state_dict']) # Cargamos pesos
        opt.load_state_dict(checkpoint['optimizer_state_dict']) # Cargamos estado del optimizador (CRÍTICO)
        start_epoch = checkpoint['epoch'] + 1 # Arrancamos en la siguiente época
        best_valid_loss = checkpoint['best_loss']
        patience_counter = checkpoint.get('patience_counter', 0)
        print(f"Reanudando desde época {start_epoch}. Mejor Loss anterior: {best_valid_loss:.4f}")
    else:
        print("Iniciando entrenamiento desde cero.")
        if os.path.exists(CSV_LOG_PATH):
            os.remove(CSV_LOG_PATH) # Limpiamos log viejo si empezamos de cero

    # Entrenamiento del modelo
    #Epochs bucle
    for epoch in range(start_epoch, num_epochs):  
        print(f"\n--- Época {epoch+1}/{num_epochs} ---")
        epoch_loss_accum = 0.0   
        num_batches = 0
        # Outer loop: iteraciones sobre lotes de tareas
        with tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}") as pbar:
            for batch in pbar:
                meta_train_loss = 0.0
                x_batch, y_batch = batch # batch: lote de tareas, x_batch: entradas, y_batch: etiquetas
                if device.type == 'cuda':
                    x_batch = x_batch.to(device, non_blocking=True)
                    y_batch = y_batch.to(device, non_blocking=True)
                #x_batch, y_batch = x_batch.to(device), y_batch.to(device)
                effective_batch_size = x_batch.size(0)
                # Bucle sobre las tareas en el batch
                for i in range(effective_batch_size):
                    learner = maml.clone()

                    x_task = x_batch[i]
                    y_task = y_batch[i]

                    x_support = x_task[:shots]
                    y_support = y_task[:shots]
                    x_query   = x_task[shots:]
                    y_query   = y_task[shots:]

                    # Inner loop: adaptación del modelo a la tarea
                    for _ in range(adapt_steps): #adapt_steps: cantidad de pasos de adaptación a la tarea
                        support_preds = learner(x_support)
                        support_loss = lossfn(support_preds, y_support)
                        learner.adapt(support_loss)

                    # Evaluación del modelo adaptado en el conjunto de consulta
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

        # 2. VALIDACIÓN (NUEVO)
        # Validamos sobre pacientes NO vistos
        print("Validando...")
        valid_loss = validate_meta_epoch(maml, val_loader, lossfn, adapt_steps, shots, device)
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
    main()        