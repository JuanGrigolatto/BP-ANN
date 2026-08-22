"""
Módulo: Metaentrenamiento.py
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
    """Validación del modelo meta-entrenado al finalizar cada época. Se evalúa la capacidad de adaptación rápida a nuevas tareas (pacientes) utilizando un Support Set pequeño y generalizando sobre un Query Set.  

    Args:
        maml: es el meta-modelo que se va a evaluar. Se clona para cada tarea de validación y se adapta con el Support Set antes de evaluar en el Query Set.    
        val_loader: es el DataLoader que proporciona las tareas de validación. Cada batch contiene un conjunto de tareas (episodios) formados por Support Set y Query Set.  
        lossfn: es la función de pérdida utilizada para calcular el error en el Support Set durante la adaptación y en el Query Set durante la evaluación. En este caso, se utiliza MSELoss para regresión. 
        adapt_steps: es el número de pasos de adaptación (inner loop) que se realizan en cada tarea de validación. Durante estos pasos, el modelo se adapta utilizando el Support Set antes de evaluar en el Query Set.
        shots: es el número de muestras en el Support Set.
        device: es el dispositivo en el que se ejecuta el modelo (CPU o CUDA).
        mode (str, optional): Por defecto 'traditional'. Determina la forma en que se construyen las tareas de validación. En 'traditional', las tareas se forman por ventanas aleatorias de un único paciente. En 'patient_wise', las tareas se forman mezclando datos de diferentes pacientes, tanto en Support Set y como enQuery Set.

    Returns:
        tipo: La pérdida promedio en el Query Set después de la adaptación en cada tarea de validación. Esta métrica refleja la capacidad del modelo meta-entrenado para generalizar a nuevas tareas (pacientes) después de una rápida adaptación con pocos datos.
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
            
            if mode == 'traditional':
                x_task, y_task = x_batch[i], y_batch[i]
                x_support, y_support = x_task[:shots], y_task[:shots]
                x_query, y_query = x_task[shots:], y_task[shots:]
            else: 
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

def save_checkpoint(state, is_best, filename=LATEST_CKPT_PATH, best_filename=BEST_CKPT_PATH):
    """Guarda el estado del entrenamiento en un checkpoint. Si el modelo actual es el mejor hasta ahora, también guarda una copia como el mejor modelo.

    Args:
        state: es un diccionario que contiene el estado del entrenamiento, incluyendo la época actual, los pesos del modelo, el estado del optimizador, la mejor pérdida obtenida hasta ahora, el contador de paciencia para early stopping y la configuración de hiperparámetros utilizada.
        is_best (bool): indica si el modelo actual es el mejor hasta ahora.
        filename (opcional): Por defecto LATEST_CKPT_PATH. Es la ruta donde se guarda el checkpoint más reciente. Si is_best es True, también se guarda una copia en best_filename.
        best_filename (opcional): Por defecto BEST_CKPT_PATH. Ruta donde se guarda el mejor checkpoint. Parametrizable para que corridas distintas (p. ej. dentro de una búsqueda de hiperparámetros) no se pisen entre sí.
    """    
    torch.save(state, filename)
    if is_best:
        torch.save(state, best_filename)

def log_to_csv(epoch, train_loss, valid_loss, filepath=CSV_LOG_PATH):
    """Registra las métricas de entrenamiento y validación en un archivo CSV. Si el archivo no existe, se crea y se escribe la cabecera. Luego, se agrega una nueva fila con los datos de la época actual.

    Args:
        epoch: es el número de época actual del entrenamiento.
        train_loss: es la pérdida promedio obtenida en el conjunto de entrenamiento durante la época actual.
        valid_loss: es la pérdida promedio obtenida en el conjunto de validación durante la época actual.
        filepath (opcional): Por defecto CSV_LOG_PATH. Es la ruta del archivo CSV donde se registran las métricas. Si el archivo no existe, se crea y se escribe la cabecera. Luego, se agrega una nueva fila con los datos de la época actual.
    """    
    file_exists = os.path.isfile(filepath)
    with open(filepath, mode='a', newline='') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(['epoch', 'train_loss', 'valid_loss'])
        writer.writerow([epoch, train_loss, valid_loss])

def plot_from_csv(csv_path):
    """Lee las métricas de entrenamiento y validación desde un archivo CSV y genera un gráfico de las curvas de pérdida a lo largo de las épocas. El gráfico se guarda en el directorio de logs con un nombre descriptivo.

    Args:
        csv_path: es la ruta del archivo CSV que contiene las métricas de entrenamiento y validación. El archivo debe tener una estructura con columnas 'epoch', 'train_loss' y 'valid_loss'. La función lee estos datos, genera un gráfico de las curvas de pérdida a lo largo de las épocas y guarda el gráfico en el directorio de logs con un nombre descriptivo.
    """    
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
         N_patient_group=4, p_support=5, q_query=10, base_dataset=None, selected_patients=None, experiment_name=None):
    """Realiza el meta-entrenamiento del modelo utilizando el algoritmo MAML. El proceso incluye la construcción de tareas de entrenamiento y validación según el modo seleccionado, la adaptación rápida del modelo a cada tarea, la evaluación en el Query Set, el registro de métricas y la gestión de checkpoints. Al finalizar el entrenamiento, se generan gráficos de las curvas de aprendizaje. 

    Args:
        mode (str, optional): Por defecto 'patient_wise'. Determina la forma en que se construyen las tareas de validación. En 'traditional', las tareas se forman por ventanas aleatorias de un único paciente. En 'patient_wise', las tareas se forman mezclando datos de diferentes pacientes, tanto en Support Set y como en Query Set.
        shots (int, optional): Por defecto 5. Número de muestras en el Support Set para el modo 'traditional'. En el modo 'patient_wise', se utilizan los parámetros p_support y q_query para definir el número de muestras en Support Set y Query Set respectivamente.
        tasks_per_batch (int, optional): Por defecto 4. Número de tareas (episodios) que se procesan en cada batch durante el entrenamiento. Este parámetro afecta la estabilidad del entrenamiento y el uso de memoria.
        adapt_lr (float, optional): Por defecto 0.01. Tasa de aprendizaje utilizada durante la fase de adaptación rápida (inner loop) en cada tarea. Este hiperparámetro controla qué tan agresivamente el modelo se adapta a cada tarea específica durante la validación.
        meta_lr (float, optional): Por defecto 0.001. Tasa de aprendizaje utilizada para actualizar los pesos del modelo meta-entrenado (outer loop) después de evaluar en el Query Set. Este hiperparámetro controla la velocidad a la que el modelo meta-entrenado aprende a generalizar a nuevas tareas.
        adapt_steps (int, optional): Por defecto 5. Número de pasos de adaptación (inner loop) que se realizan en cada tarea de validación. Durante estos pasos, el modelo se adapta utilizando el Support Set antes de evaluar en el Query Set.
        seed (int, optional): Por defecto 42. Valor de la semilla para asegurar la reproducibilidad del entrenamiento. Se utiliza para fijar las semillas de random, numpy y torch.
        num_epochs (int, optional): Por defecto 500. Número máximo de épocas para el entrenamiento del modelo meta-entrenado. El entrenamiento puede detenerse antes si se activa el early stopping.
        patience (int, optional): Por defecto 20. Número de épocas sin mejora significativa en la pérdida de validación antes de activar el early stopping. Si la pérdida de validación no mejora en al menos min_delta durante este número de épocas, el entrenamiento se detiene para evitar sobreajuste.
        min_delta (opcional): Por defecto 1e-3. Valor mínimo de mejora en la pérdida de validación para considerar que el modelo ha mejorado. Si la mejora en la pérdida de validación es menor que este valor durante el período de paciencia, se considera que no hubo mejora significativa.
        N_patient_group (int, optional): Por defecto 4. Número de pacientes diferentes que se incluyen en cada tarea (episodio) en el modo 'patient_wise'. Este hiperparámetro controla la diversidad de pacientes en cada tarea, lo que puede afectar la capacidad del modelo para generalizar a nuevos pacientes.
        p_support (int, optional): Por defecto 5. Número de muestras en el Support Set para cada paciente en el modo 'patient_wise'. Este hiperparámetro controla la cantidad de datos de calibración disponibles para adaptar el modelo a cada paciente durante la validación.
        q_query (int, optional): Por defecto 10. Número de muestras en el Query Set para cada paciente en el modo 'patient_wise'. Este hiperparámetro controla la cantidad de datos sobre los cuales se evalúa la capacidad de generalización del modelo después de la adaptación.
        base_dataset (UCIDataset, optional): Por defecto None. Si se provee, se usa este dataset ya cargado en memoria en lugar de leerlo desde disco. Pensado para búsquedas de hiperparámetros que llaman a main() repetidas veces y no quieren recargar el dataset completo en cada corrida.
        selected_patients (list[int], optional): Por defecto None. Si se provee, restringe el universo de pacientes considerados a esta lista fija (en lugar de usar todos los pacientes del dataset), y se usa como base determinística para la partición train/val/test (dado el mismo seed, produce siempre la misma partición). Pensado para que distintas corridas de una búsqueda de hiperparámetros se comparen sobre la misma partición de pacientes.
        experiment_name (str, optional): Por defecto None. Si se provee, se usa para nombrar de forma única los archivos de checkpoint (latest/best) y el log CSV de esta corrida, evitando que corridas distintas (p. ej. dentro de una búsqueda de hiperparámetros) se sobrescriban entre sí. Si es None, se usan las rutas fijas del módulo (comportamiento original).

    Raises:
        ValueError: Si el valor de mode no es 'traditional' ni 'patient_wise', se lanza un error indicando que el modo debe ser uno de esos dos valores.
    """    
    if experiment_name is not None:
        latest_ckpt_path = os.path.join(CHECKPOINT_DIR, f'checkpoint_latest_{experiment_name}.pt')
        best_ckpt_path = os.path.join(CHECKPOINT_DIR, f'best_meta_model_{experiment_name}.pt')
        csv_log_path = os.path.join(LOG_DIR, f'training_log_{experiment_name}.csv')
    else:
        latest_ckpt_path = LATEST_CKPT_PATH
        best_ckpt_path = BEST_CKPT_PATH
        csv_log_path = CSV_LOG_PATH

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

    if base_dataset is not None:
        print("Usando dataset ya cargado en memoria (provisto externamente).")
        dataset_completo = base_dataset
    else:
        data_paths = [
            'data/processed/data_UCI/dataset_parte_1_por_picos.pt',
            'data/processed/data_UCI/dataset_parte_2_por_picos.pt',
            'data/processed/data_UCI/dataset_parte_3_por_picos.pt',
            'data/processed/data_UCI/dataset_parte_4_por_picos.pt',
        ]
        print("Cargando datasets...")
        dataset_completo = UCIDataset(data_paths)

    if selected_patients is not None:
        unique_patients = list(selected_patients)
    else:
        all_pids = torch.tensor([dataset_completo[i][2] for i in range(len(dataset_completo))])
        unique_patients = all_pids.unique().tolist()

    random.seed(seed)
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
    val_patients = unique_patients[n_train : n_train + n_val] 
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

    
    if os.path.exists(latest_ckpt_path):
        print(f"Checkpoint encontrado en {latest_ckpt_path}. Reanudando...")
        checkpoint = torch.load(latest_ckpt_path)
        maml.load_state_dict(checkpoint['model_state_dict']) 
        opt.load_state_dict(checkpoint['optimizer_state_dict']) 
        best_valid_loss = checkpoint['best_loss']
        patience_counter = checkpoint.get('patience_counter', 0)
        print(f"Reanudando desde época {start_epoch}. Mejor Loss anterior: {best_valid_loss:.4f}")
    else:
        print("Iniciando entrenamiento desde cero.")
        if os.path.exists(csv_log_path):
            os.remove(csv_log_path) 

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

        log_to_csv(epoch+1, epoch_train_loss, valid_loss, filepath=csv_log_path)

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
        }, is_best, filename=latest_ckpt_path, best_filename=best_ckpt_path)

        if patience_counter >= patience:
            print(f"\n[EARLY STOPPING] Se ha detenido el entrenamiento.")
            print(f"No hubo mejora de {min_delta} en las últimas {patience} épocas.")
            print(f"Mejor loss obtenido: {best_valid_loss:.4f}")
            break

    print("Entrenamiento completado.")

    plot_from_csv(csv_log_path)

    return {'best_loss': best_valid_loss, 'best_checkpoint_path': best_ckpt_path, 'csv_log_path': csv_log_path}

if __name__ == '__main__':
    main(mode='patient_wise', shots=10,tasks_per_batch=4, adapt_lr=0.005, meta_lr=0.001, adapt_steps=5, seed=42, num_epochs=500, patience=20, min_delta=1e-3,
         N_patient_group=2, p_support=10, q_query=20)        