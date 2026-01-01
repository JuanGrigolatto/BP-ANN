import torch
from src.data.data_chargers.Clase_UCIDataset import UCIDataset
import os
from torch.utils import data
from torch.utils.data import TensorDataset, random_split
from sklearn.model_selection import train_test_split
import numpy as np
from src.models.InceptionTime import InceptionTime
#from src.models.Modelo_conv import Modelo_Convolucional
from src.models.ConvolucionalV1 import Modelo_ConvolucionalV1
from src.models.ConvolucionalV1_2 import Modelo_ConvolucionalV1_2
from src.models.ConvolucionalV2 import Modelo_ConvolucionalV2
from tqdm.auto import tqdm 
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import ReduceLROnPlateau
import random
import src.utils.Tools.Tools as Tools
import pandas as pd


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # para multi-GPU

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def init_worker_fn(worker_id):
    """
    Función que el DataLoader ejecutará en cada nuevo proceso worker.
    """
    worker_info = torch.utils.data.get_worker_info()
    dataset = worker_info.dataset
    
    # Como usas random_split, 'dataset' es un objeto Subset.
    # Tenemos que bajar hasta encontrar el UCIDataset real.
    while hasattr(dataset, 'dataset'):
        dataset = dataset.dataset
    
    # Ahora sí, llamamos al método que abre los archivos
    if hasattr(dataset, 'worker_init'):
        dataset.worker_init()

def save_test_dataset(save_dir, prefix, data, labels, patients, indexs):
    """
    Guarda los datos de test en formato compatible con UCIDataset.
    
    Args:
        save_dir (str): Carpeta donde guardar los archivos.
        prefix (str): Prefijo de los archivos (ej: "test").
        data (np.ndarray): Señales, shape (N, 2, segment_length), dtype float32.
        labels (np.ndarray): Etiquetas, shape (N, 2), dtype float32.
        patients (np.ndarray): IDs de pacientes, shape (N,), dtype int64.
        indexs (np.ndarray): Índices globales, shape (N,), dtype int64.
    """

    os.makedirs(save_dir, exist_ok=True)

    num_samples, _, segment_length = data.shape

    # Asegurar dtypes correctos
    data = data.astype("float32")
    labels = labels.astype("float32")
    patients = patients.astype("int64")
    indexs = indexs.astype("int64")

    # Paths de los .dat
    data_path    = os.path.join(save_dir, f"{prefix}_data.dat")
    labels_path  = os.path.join(save_dir, f"{prefix}_labels.dat")
    patients_path= os.path.join(save_dir, f"{prefix}_patients.dat")
    indexs_path  = os.path.join(save_dir, f"{prefix}_indexs.dat")
    meta_path    = os.path.join(save_dir, f"{prefix}_meta.pt")

    # Guardar con memmap
    np.memmap(data_path, dtype="float32", mode="w+", shape=data.shape)[:] = data[:]
    np.memmap(labels_path, dtype="float32", mode="w+", shape=labels.shape)[:] = labels[:]
    np.memmap(patients_path, dtype="int64", mode="w+", shape=patients.shape)[:] = patients[:]
    np.memmap(indexs_path, dtype="int64", mode="w+", shape=indexs.shape)[:] = indexs[:]

    # Guardar metadata en un .pt
    meta = {
        "data_path": data_path,
        "labels_path": labels_path,
        "patients_path": patients_path,
        "indexs_path": indexs_path,
        "num_samples": num_samples,
        "segment_length": segment_length
    }
    torch.save(meta, meta_path)

    print(f" Dataset de test guardado en {save_dir}")
    print(f" Total muestras: {num_samples}, tamaño: {segment_length}")
    return meta_path

def get_patient_split_indices(pt_files, train_ratio=0.7, val_ratio=0.2, seed=42):
    """
    Genera índices globales para Train, Val y Test asegurando que 
    los pacientes NO se mezclen entre conjuntos.
    """
    global_indices_map = [] 
    current_offset = 0
    
    print("--- Iniciando División por Sujetos ---")
    
    
    for pt_file in pt_files:
        # Cargar metadata
        if not os.path.exists(pt_file):
            raise FileNotFoundError(f"No se encuentra el archivo de metadata: {pt_file}")
            
        meta = torch.load(pt_file)
        num_samples = meta['num_samples']
        
        patients_path = meta['patients_path'] 
        
        if not os.path.exists(patients_path):
             base_dir = os.path.dirname(pt_file)
             patients_path = os.path.join(base_dir, os.path.basename(patients_path))

        if not os.path.exists(patients_path):
             raise FileNotFoundError(f"No se encuentra el archivo de pacientes: {patients_path}")

        p_mmap = np.memmap(patients_path, dtype='int64', mode='r', shape=(num_samples,))
        
        ids = np.array(p_mmap) 
        indices = np.arange(current_offset, current_offset + num_samples)
        
        chunk_data = np.stack((ids, indices), axis=1)
        global_indices_map.append(chunk_data)
        
        current_offset += num_samples

    full_map = np.concatenate(global_indices_map, axis=0)
    unique_patients = np.unique(full_map[:, 0])
    
    print(f"Total pacientes únicos encontrados: {len(unique_patients)}")
    
    # 1. Separar Test
    test_ratio = 1.0 - train_ratio - val_ratio
   
    if test_ratio < 0: test_ratio = 0
    
    train_val_patients, test_patients = train_test_split(
        unique_patients, test_size=test_ratio, random_state=seed, shuffle=True
    )
    
    # 2. Separar Train y Val 
    if len(train_val_patients) > 0:
        val_relative = val_ratio / (train_ratio + val_ratio)
        train_patients, val_patients = train_test_split(
            train_val_patients, test_size=val_relative, random_state=seed, shuffle=True
        )
    else:
        train_patients = []
        val_patients = []
    
    print(f"Distribución de PACIENTES -> Train: {len(train_patients)}, Val: {len(val_patients)}, Test: {len(test_patients)}")
    
    mask_train = np.isin(full_map[:, 0], train_patients)
    mask_val = np.isin(full_map[:, 0], val_patients)
    mask_test = np.isin(full_map[:, 0], test_patients)
    
    train_indices = full_map[mask_train, 1]
    val_indices = full_map[mask_val, 1]
    test_indices = full_map[mask_test, 1]
    
    return train_indices, val_indices, test_indices

def main():
    # Configuración del dispositivo
    parameters = {
        'batch_size': 256,
        'shuffle': True,
        'num_workers': 2,
        'pin_memory': True,
        'persistent_workers': True, 
        'prefetch_factor': 2 ,   
        'worker_init_fn': init_worker_fn
    }
    test_params = {
        'batch_size': 256,
        'shuffle': False, 
        'num_workers': 0, 
        'pin_memory': False
    }
    set_seed(42)
    
    archivos = [
    'data/processed/data_UCI/dataset_parte_1_por_picos_global_norm.pt',
    'data/processed/data_UCI/dataset_parte_2_por_picos_global_norm.pt',
    'data/processed/data_UCI/dataset_parte_3_por_picos_global_norm.pt',
    'data/processed/data_UCI/dataset_parte_4_por_picos_global_norm.pt',
    ]

    dataset_completo = UCIDataset(archivos)


    train_idx, val_idx, test_idx = get_patient_split_indices(
        archivos, 
        train_ratio=0.7, 
        val_ratio=0.2, 
        seed=42
    )

    print(f"Muestras (Segmentos) -> Train: {len(train_idx)}, Val: {len(val_idx)}, Test: {len(test_idx)}")

    train_set = torch.utils.data.Subset(dataset_completo, train_idx)
    val_set = torch.utils.data.Subset(dataset_completo, val_idx)
    test_set = torch.utils.data.Subset(dataset_completo, test_idx)

    print(f"Train: {len(train_set)}, Val: {len(val_set)}, Test: {len(test_set)}")

    training_generator = torch.utils.data.DataLoader(train_set, **parameters)
    validation_generator = torch.utils.data.DataLoader(val_set, **parameters)
    test_generator = torch.utils.data.DataLoader(test_set, **test_params)

    all_data= []
    all_labels = []
    all_patients = []
    all_indexs = []    

    for x, y, pid, idx in test_generator:
        all_data.append(x.numpy())
        all_labels.append(y.numpy())
        all_patients.append(pid.numpy())
        all_indexs.append(idx.numpy())


    test_data = np.concatenate(all_data, axis=0)
    test_labels = np.concatenate(all_labels, axis=0)
    test_patients = np.concatenate(all_patients, axis=0)
    test_indexs = np.concatenate(all_indexs, axis=0)

    
    meta_test = save_test_dataset(
    save_dir='data/processed/data_UCI/test_set_por_pacientes_iso.pt',
    prefix="test",
    data=test_data,         # (N,2,L)
    labels=test_labels,     # (N,2)
    patients=test_patients, # (N,)
    indexs=test_indexs      # (N,)
    )
    print("data/processed/data_UCI/test_set_por_pacientes_iso.pt guardado")
    #Subset para testeo de modelos sin entrenamiento completo

    # Crear el modelo

    model=InceptionTime(c_in=2, c_out=2, seq_len=None, n_filters=32, depth = 6)
    #model=Modelo_Convolucional(in_channels=2,out_channels=2, long_signal=250)
    #model=Modelo_ConvolucionalV1(in_channels=2,out_channels=3, long_signal=500)
    #model=Modelo_ConvolucionalV1(in_channels=2,out_channels=2, long_signal=500)
    #model=Modelo_ConvolucionalV2(in_channels=2,out_channels=2, long_signal=500)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)  # Mueve el modelo a la GPU
    #optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay= 1e-4)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay= 1e-3)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, min_lr=1e-6, verbose=True)    
    criterion = torch.nn.MSELoss()  # MSELoss para regresión
    scaler = torch.amp.GradScaler('cuda')

    #retomar entrenamiento 
    start_epoch = 0
    log_path = 'graficas/training_log_Time32_ps.csv' 
    os.makedirs('graficas', exist_ok=True)
    checkpoint_path = "models/best_models/best_model_conv_Time32_200_epocas_picos_def_early8_ps.pt"
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = checkpoint.get("epoca", 0) + 1
        best_valid_loss = checkpoint.get("best_valid_loss", float('inf')) 
        print(f" Modelo cargado desde {checkpoint_path}, continuando en epoch {start_epoch}, best_valid_loss={best_valid_loss:.6f}")
    else:
        print("No se encontró un checkpoint, se comienza entrenamiento desde cero.")

    if os.path.exists(log_path):
        try:
            # Leemos el CSV. Asumimos columnas: epoch, train_loss, valid_loss
            df_log = pd.read_csv(log_path)
            print(f" Historial encontrado con {len(df_log)} registros.")
            
            # Llenamos el running_loss con lo que ya teniamos
            for _, row in df_log.iterrows():
                ep = int(row['epoch'])
                if ep < max_epochs:
                    running_loss[ep, 0] = row['train_loss']
                    running_loss[ep, 1] = row['valid_loss']
        except Exception as e:
            print(f" Error leyendo el log anterior: {e}. Se graficará solo lo nuevo.")
    
    # Si empezamos de cero absoluto, creamos el archivo con cabeceras
    if start_epoch == 0:
        with open(log_path, 'w') as f:
            f.write("epoch,train_loss,valid_loss\n")

    # ENTRENAMIENTO
    def train_one_step(batch, l1_lambda=1e-5):
        optimizer.zero_grad(set_to_none=True) # Reinicia los gradientes
        data, labels, _, _ = batch # Obtiene los datos y etiquetas

        #labels_sbp= labels[:,0].unsqueeze(1)
        #labels_dbp= labels[:,1].unsqueeze(1)
        #labels_pam = Tools.calcular_pam(labels_sbp, labels_dbp)

        #labels = torch.cat((labels_sbp, labels_dbp, labels_pam), dim=1)

        data, labels = data.to(device, non_blocking=True), labels.to(device, non_blocking=True) # Mueve los datos y etiquetas a la GPU
        
        #print("Rango de datos:", torch.min(data).item(), torch.max(data).item())

        #for name, param in model.named_parameters():
        #    print(f"{name}: mean={param.mean().item():.4f}, std={param.std().item():.4f}")
        with torch.amp.autocast('cuda'):
            preds = model.forward(data) # Realiza la predicción
            #print("Preds:", preds[0])
            #print("Labels:", labels[0])
            loss = criterion(preds, labels) # Calcula la pérdida
        """
        if torch.isnan(loss):
            print("¡Loss con NaN detectado! Abortando...")
            print("Preds:", preds[0])
            print("Labels:", labels[0])
            exit()
        """
        # Regularización L1 manual
        #l1_norm = sum(p.abs().sum() for p in model.parameters())
        #loss = loss + l1_lambda * l1_norm

        #loss.backward() # Calcula los gradientes mediante backpropagation
        scaler.scale(loss).backward()
        # Añade gradient clipping (busca evitar explosión de gradiente)
        #torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        #optimizer.step() # Actualiza los parámetros del modelo
        scaler.step(optimizer)
        scaler.update()
        return loss.item() # Devuelve la pérdida

    def evaluate_one_step(batch):
        with torch.no_grad():
            data, labels, _, _ = batch

            #labels_sbp= labels[:,0].unsqueeze(1)
            #labels_dbp= labels[:,1].unsqueeze(1)
            #labels_pam = Tools.calcular_pam(labels_sbp, labels_dbp)
            #labels = torch.cat((labels_sbp, labels_dbp, labels_pam), dim=1)

            data, labels = data.to(device), labels.to(device)
            preds = model.forward(data)
            loss = criterion(preds, labels)
            return loss.item()
    
    def train_one_epoch():    
        train_loss, valid_loss = 0.0, 0.0

        model.train()
      
        loop = tqdm(training_generator, leave=False, desc="Train Step")
        for batch in loop:
        #for batch in subset_loader:
            loss_batch = train_one_step(batch)
            train_loss += loss_batch
            loop.set_postfix(loss=loss_batch)
        
   
        model.eval()    
        for batch in validation_generator:
        #for batch in subset_loader:
            valid_loss += evaluate_one_step(batch)
       

        return train_loss/len(training_generator), valid_loss/len(validation_generator)
    
    max_epochs = 200
    best_valid_loss = float('inf') 
    running_loss = np.zeros(shape=(max_epochs, 2))
    min_delta = 0.0001  # mejora mínima requerida
    patience = 8
    patience_significativa = 8
    for epoch in tqdm(range(start_epoch, max_epochs)):

        train_l, valid_l = train_one_epoch()
        running_loss[epoch] = (train_l,valid_l)
        with open(log_path, 'a') as f:
            f.write(f"{epoch},{train_l:.6f},{valid_l:.6f}\n")    
        print(f"[Época {epoch}] Train Loss: {train_l:.6f} - Valid Loss: {valid_l:.6f}")
        improvement = best_valid_loss - valid_l

        if valid_l < best_valid_loss:
            best_valid_loss = valid_l
            no_improvement_count = 0  
            torch.save({
                'epoca': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': valid_l,
                'best_valid_loss': best_valid_loss 
            }, 'models/best_models/best_model_conv_Time32_200_epocas_picos_def_early8_ps.pt')
            print(f"Nuevo mejor modelo guardado (valid_loss = {valid_l:.6f})")

            
            if improvement > min_delta:
                no_significant_improvement_count = 0
            else:
                no_significant_improvement_count += 1
        else:
            no_improvement_count += 1
            no_significant_improvement_count += 1

        
        if no_improvement_count >= patience:
            print(f"Early stopping por falta de mejoras (últimas {patience} épocas).")
            break
        if no_significant_improvement_count >= patience_significativa:
            print(f"Early stopping por falta de mejoras > {min_delta:.4f} (últimas {patience_significativa} épocas).")
            break

        scheduler.step(valid_l)
    
    final_epoch_reached = epoch + 1

    fig, ax = plt.subplots(figsize=(7, 4), tight_layout=True)

    ax.plot(range(final_epoch_reached), running_loss[:final_epoch_reached, 0], label='Entrenamiento')
    ax.plot(range(final_epoch_reached), running_loss[:final_epoch_reached, 1], label='Validación')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.legend()
    plt.savefig('graficas/loss_curve.png')
    plt.show()

if __name__ == '__main__':
    main()
