"""
Módulo: train_model.py
Autor: Juan Marcos Grigolatto
Descripción: Script principal de entrenamiento para la red neuronal de estimación 
             de presión arterial. Gestiona la carga de datasets masivos (memmap), 
             el ciclo de entrenamiento/validación con precisión mixta (AMP), 
             el guardado dinámico de los subconjuntos de prueba, y la persistencia 
             de modelos mediante Checkpoints y Early Stopping.
"""
import torch
from src.data.data_chargers.Clase_UCIDataset import UCIDataset
import os
from torch.utils import data
from torch.utils.data import TensorDataset, random_split
import numpy as np
from src.models.InceptionTime import InceptionTime
from src.models.ConvolucionalV1 import Modelo_ConvolucionalV1
from src.models.ConvolucionalV2 import Modelo_ConvolucionalV2
from tqdm.auto import tqdm 
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import ReduceLROnPlateau
import random
import src.utils.Tools.Tools as Tools
import pandas as pd


def set_seed(seed=42):
    """_summary_ Establece la semilla para todas las fuentes de aleatoriedad en el entrenamiento, incluyendo random, numpy y torch, para asegurar la reproducibilidad de los resultados. Además, configura los backends de PyTorch para garantizar un comportamiento determinista durante el entrenamiento.

    Args:
        seed (int, optional): _description_. Por defecto 42. Número entero que se utilizará como semilla para todas las fuentes de aleatoriedad en el entrenamiento.
    """    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # para multi-GPU

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def init_worker_fn(worker_id):
    """_summary_ Función de inicialización para los workers del DataLoader que cargan los datos de entrenamiento. Esta función se encarga de acceder al dataset real (en caso de que se esté utilizando un Subset) y llamar a su método worker_init() para abrir los archivos memmap correspondientes a cada worker, asegurando que cada worker tenga acceso a los datos necesarios para cargar los batches durante el entrenamiento.

    Args:
        worker_id (_type_): _description_ Número entero que identifica al worker actual en el DataLoader, utilizado para acceder al dataset real y llamar a su método de inicialización.
    """    
    worker_info = torch.utils.data.get_worker_info()
    dataset = worker_info.dataset
    
    while hasattr(dataset, 'dataset'):
        dataset = dataset.dataset
    
    # Ahora sí, llamamos al método que abre los archivos
    if hasattr(dataset, 'worker_init'):
        dataset.worker_init()

def save_test_dataset(save_dir, prefix, data, labels, patients, indexs):
    """_summary_ Guarda el conjunto de datos de prueba en un formato eficiente utilizando memmap para los datos y etiquetas, y un archivo .pt para la metadata. La función crea los archivos necesarios para almacenar los datos de prueba, incluyendo las señales, las etiquetas, los IDs de pacientes y los índices, y luego guarda la metadata que contiene las rutas a estos archivos junto con información adicional como el número de muestras y la longitud de las señales. 

    Args:
        save_dir (_type_): _description_ Ruta del directorio donde se guardarán los archivos del conjunto de datos de prueba, incluyendo los archivos memmap para los datos y etiquetas, y el archivo .pt para la metadata.
        prefix (_type_): _description_  Prefijo que se utilizará para nombrar los archivos del conjunto de datos de prueba, permitiendo identificar fácilmente los archivos relacionados con este conjunto específico.
        data (_type_): _description_ Array o tensor que contiene las señales de prueba que se desean guardar, con una forma típica de (N, 2, L) donde N es el número de muestras, 2 es el número de canales (PPG y ECG), y L es la longitud de las señales.
        labels (_type_): _description_ Array o tensor que contiene las etiquetas de presión arterial (SBP y DBP) correspondientes a cada muestra de prueba, con una forma típica de (N, 2) donde N es el número de muestras y 2 corresponde a las dos etiquetas (SBP y DBP).
        patients (_type_): _description_ Array o tensor que contiene los IDs de pacientes correspondientes a cada muestra de prueba, con una forma típica de (N,) donde N es el número de muestras.
        indexs (_type_): _description_ Array o tensor que contiene los índices correspondientes a cada muestra de prueba, con una forma típica de (N,) donde N es el número de muestras.

    Returns:
        _type_: _description_ Retorna la ruta al archivo .pt que contiene la metadata del conjunto de datos de prueba, incluyendo las rutas a los archivos memmap de datos y etiquetas, y la información adicional sobre el número de muestras y la longitud de las señales.
    """

    os.makedirs(save_dir, exist_ok=True)

    num_samples, _, segment_length = data.shape

    data = data.astype("float32")
    labels = labels.astype("float32")
    patients = patients.astype("int64")
    indexs = indexs.astype("int64")

    data_path    = os.path.join(save_dir, f"{prefix}_data.dat")
    labels_path  = os.path.join(save_dir, f"{prefix}_labels.dat")
    patients_path= os.path.join(save_dir, f"{prefix}_patients.dat")
    indexs_path  = os.path.join(save_dir, f"{prefix}_indexs.dat")
    meta_path    = os.path.join(save_dir, f"{prefix}_meta.pt")

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

def main():
    """_summary_ Función principal de entrenamiento de red neuronal para presión arterial.

    Returns:
        _type_: _description_ Esta función no retorna ningún valor, pero ejecuta el proceso completo de entrenamiento de la red neuronal para estimación de presión arterial, incluyendo la carga de datos, la configuración del modelo, el ciclo de entrenamiento y validación, el guardado de modelos y logs, y la implementación de early stopping.
    """    
    # Configuración del dispositivo
    parameters = {
        'batch_size': 256,
        'shuffle': True,
        'num_workers': 2,
        'pin_memory': True,
        'persistent_workers': True, # Mantiene los workers vivos entre épocas
        'prefetch_factor': 2 ,   # Precarga 2 batches por worker
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

    total = len(dataset_completo)
    train_size = int(0.7 * total)
    val_size = int(0.2 * total)
    test_size = total - train_size - val_size

    # División aleatoria el dataset
    train_set, val_set, test_set = random_split(dataset_completo, [train_size, val_size, test_size])

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
    save_dir='data/processed/data_UCI/test_set_por_picos',
    prefix="test",
    data=test_data,         # (N,2,L)
    labels=test_labels,     # (N,2)
    patients=test_patients, # (N,)
    indexs=test_indexs      # (N,)
    )
    print("data/processed/data_UCI/test_set.pt guardado")
    
    # Creación del modelo
    model=InceptionTime(c_in=2, c_out=2, seq_len=None, n_filters=32, depth = 6)
    #model=Modelo_Convolucional(in_channels=2,out_channels=2, long_signal=250)
    #model=Modelo_ConvolucionalV1(in_channels=2,out_channels=3, long_signal=500)
    #model=Modelo_ConvolucionalV1(in_channels=2,out_channels=2, long_signal=500)
    #model=Modelo_ConvolucionalV2(in_channels=2,out_channels=2, long_signal=500)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)  # Mueve el modelo a la GPU

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay= 1e-4)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, min_lr=1e-6, verbose=True)    
    criterion = torch.nn.MSELoss()  # MSELoss para regresión
    scaler = torch.amp.GradScaler('cuda')

    start_epoch = 0
    log_path = 'graficas/training_log.csv' 
    os.makedirs('graficas', exist_ok=True)
    checkpoint_path = "models/best_models/best_model_conv_time32_200_epocas_picos_def_early8.pt"
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
            df_log = pd.read_csv(log_path)
            print(f" Historial encontrado con {len(df_log)} registros.")

            for _, row in df_log.iterrows():
                ep = int(row['epoch'])
                if ep < max_epochs:
                    running_loss[ep, 0] = row['train_loss']
                    running_loss[ep, 1] = row['valid_loss']
        except Exception as e:
            print(f" Error leyendo el log anterior: {e}. Se graficará solo lo nuevo.")
    
    if start_epoch == 0:
        with open(log_path, 'w') as f:
            f.write("epoch,train_loss,valid_loss\n")

    # ENTRENAMIENTO
    def train_one_step(batch, l1_lambda=1e-5):
        """_summary_ Realiza un paso de entrenamiento: procesa un batch de datos, calcula la pérdida, realiza backpropagation y actualiza los pesos del modelo.

        Args:
            batch (_type_): _description_ Batch de datos que contiene las señales de entrada, las etiquetas de presión arterial, los IDs de pacientes y los índices, que se utilizará para realizar un paso de entrenamiento en el modelo.
            l1_lambda (_type_, optional): _description_. Por defecto 1e-5. Coeficiente de regularización L1. 
        Returns:
            _type_: _description_ Retorna el valor de la pérdida calculada para el batch procesado, que se utilizará para monitorear el rendimiento del modelo durante el entrenamiento.
        """        
        optimizer.zero_grad(set_to_none=True) # Reinicia los gradientes
        data, labels, _, _ = batch # Obtiene los datos y etiquetas

        data, labels = data.to(device, non_blocking=True), labels.to(device, non_blocking=True) # Mueve los datos y etiquetas a la GPU
        
        with torch.amp.autocast('cuda'):
            preds = model.forward(data) # Realiza la predicción
            loss = criterion(preds, labels) # Calcula la pérdida

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        return loss.item() # Devuelve la pérdida

    def evaluate_one_step(batch):
        """_summary_ Realiza un paso de evaluación: procesa un batch de datos de validación, calcula la pérdida sin realizar backpropagation ni actualizar los pesos del modelo. 

        Args:
            batch (_type_): _description_ Batch de datos que contiene las señales de entrada, las etiquetas de presión arterial, los IDs de pacientes y los índices, que se utilizará para realizar un paso de evaluación en el modelo durante la fase de validación.

        Returns:
            _type_: _description_ Retorna el valor de la pérdida calculada para el batch procesado, que se utilizará para monitorear el rendimiento del modelo durante la validación.
        """        
        with torch.no_grad():
            data, labels, _, _ = batch
            data, labels = data.to(device), labels.to(device)
            preds = model.forward(data)
            loss = criterion(preds, labels)
            return loss.item()
    
    def train_one_epoch():    
        """_summary_ Realiza un ciclo completo de entrenamiento para una época, procesando todos los batches de entrenamiento y validación, calculando las pérdidas promedio para cada fase, y actualizando el scheduler de aprendizaje en función de la pérdida de validación.

        Returns:
            _type_: _description_ Retorna dos valores: train_loss, que es la pérdida promedio calculada para todos los batches de entrenamiento procesados durante la época, y valid_loss, que es la pérdida promedio calculada para todos los batches de validación procesados durante la época.
        """        
        train_loss, valid_loss = 0.0, 0.0

        model.train()
      
        loop = tqdm(training_generator, leave=False, desc="Train Step")
        for batch in loop:
            loss_batch = train_one_step(batch)
            train_loss += loss_batch
            loop.set_postfix(loss=loss_batch)
        
        model.eval()    
        for batch in validation_generator:
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
            }, 'models/best_models/best_model_conv_time32_200_epocas_picos_def_early8.pt')
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


    
     