"""
Módulo: train_model_ps.py
Autor: Juan Marcos Grigolatto
Descripción: Script de entrenamiento avanzado para la red neuronal InceptionTime.
             Este módulo implementa una división 
             estricta por pacientes (Patient-Subject Splitting) para evitar el 
             Data Leakage (fuga de datos). Garantiza que las señales de un mismo 
             individuo no se mezclen entre los conjuntos de Train, Validation y Test, 
             evaluando la verdadera capacidad de generalización del modelo.
"""
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
from src.models.ConvolucionalV2 import Modelo_ConvolucionalV2
from tqdm.auto import tqdm 
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import ReduceLROnPlateau
import random
import src.utils.Tools.Tools as Tools
import pandas as pd


def set_seed(seed=42):
    """_summary_ Establece la semilla para todas las operaciones aleatorias en Python, NumPy y PyTorch, asegurando la reproducibilidad de los experimentos. Esta función configura la semilla para el módulo random de Python, el generador de números aleatorios de NumPy, y los generadores de números aleatorios de PyTorch tanto para CPU como para GPU. Además, ajusta las configuraciones de cuDNN para garantizar resultados determinísticos en operaciones convolucionales.

    Args:
        seed (int, optional): _description_. Por defecto 42. La semilla que se utilizará para inicializar los generadores de números aleatorios en Python, NumPy y PyTorch, asegurando que los experimentos sean reproducibles.
    """    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # para multi-GPU

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def init_worker_fn(worker_id):
    """_summary_ Función de inicialización para los workers del DataLoader que utilizan UCIDataset. Esta función se encarga de obtener la instancia del dataset asociada al worker actual.
            
    Args:
        worker_id (_type_): _description_ Identificador del worker que se está inicializando. 
    """    
    worker_info = torch.utils.data.get_worker_info()
    dataset = worker_info.dataset
    
    while hasattr(dataset, 'dataset'):
        dataset = dataset.dataset
    
    if hasattr(dataset, 'worker_init'):
        dataset.worker_init()

def save_test_dataset(save_dir, prefix, data, labels, patients, indexs):
    """_summary_  Guarda los datos de test en formato compatible con UCIDataset, utilizando archivos .dat para las señales, etiquetas, IDs de pacientes e índices globales, y un archivo .pt para la metadata que contiene las rutas a los archivos y la información sobre el número de muestras y la longitud de los segmentos. Esta función asegura que los datos se guarden con los tipos de datos correctos y en la estructura esperada por UCIDataset, facilitando su posterior carga y uso para evaluación del modelo.

    Args:
        save_dir (_type_): _description_ Carpeta donde se guardarán los archivos generados para el conjunto de test, incluyendo los archivos .dat para las señales, etiquetas, IDs de pacientes e índices, así como el archivo .pt para la metadata.
        prefix (_type_): _description_ Prefijo para los archivos generados.
        data (_type_): _description_ Señales de entrada, shape (N, 2, segment_length), dtype float32.
        labels (_type_): _description_ Etiquetas de presión arterial, shape (N, 2), dtype float32.
        patients (_type_): _description_ IDs de pacientes, shape (N,), dtype int64.
        indexs (_type_): _description_ Índices globales, shape (N,), dtype int64.

    Returns:
        _type_: _description_ Ruta al archivo de metadata (.pt) que contiene la información sobre los archivos de datos y la estructura del conjunto de test guardado.
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

def get_patient_split_indices(pt_files, train_ratio=0.7, val_ratio=0.2, seed=42):
    """_summary_  Realiza una división de los datos basada en pacientes (Patient-Subject Splitting) para garantizar que las señales de un mismo individuo no se mezclen entre los conjuntos de Train, Validation y Test. Esta función carga la metadata de cada archivo .pt proporcionado, extrae los IDs de pacientes y sus índices globales, y luego utiliza la función train_test_split de scikit-learn para realizar la división estratificada por pacientes, asegurando que cada paciente completo se asigne a un solo conjunto. Finalmente, devuelve los índices globales correspondientes a cada conjunto (Train, Validation y Test) para su uso en la creación de los DataLoaders.

    Args:
        pt_files (_type_): _description_ Lista de rutas a los archivos .pt que contienen la metadata de los conjuntos de datos, incluyendo las rutas a los archivos de señales, etiquetas, IDs de pacientes e índices, así como la información sobre el número de muestras y la longitud de los segmentos. Estos archivos se utilizarán para extraer los IDs de pacientes y sus índices globales para realizar la división por pacientes.
        train_ratio (float, optional): _description_. Por defecto 0.7. Proporción del conjunto de datos que se asignará al conjunto de entrenamiento (Train) durante la división por pacientes. Este valor determina qué porcentaje de los pacientes únicos se incluirán en el conjunto de entrenamiento, mientras que el resto se dividirá entre los conjuntos de validación (Validation) y prueba (Test) según las proporciones especificadas.
        val_ratio (float, optional): _description_. Por defecto 0.2. Proporción del conjunto de datos que se asignará al conjunto de validación (Validation) durante la división por pacientes.
        seed (int, optional): _description_. Por defecto 42. Semilla para la generación de números aleatorios, asegurando resultados reproducibles.

    Returns:
        _type_: _description_ Retorna tres arrays de índices globales correspondientes a los conjuntos de entrenamiento (Train), validación (Validation) y prueba (Test), que se pueden utilizar para crear subconjuntos del dataset completo sin mezclar señales de un mismo paciente entre los conjuntos.
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
    
    test_ratio = 1.0 - train_ratio - val_ratio
   
    if test_ratio < 0: test_ratio = 0
    
    train_val_patients, test_patients = train_test_split(
        unique_patients, test_size=test_ratio, random_state=seed, shuffle=True
    )
    
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
    """_summary_  Función principal que configura el entorno de entrenamiento, carga los datos utilizando la división por pacientes, define el modelo, el optimizador, la función de pérdida y el scheduler, y luego ejecuta el ciclo de entrenamiento con early stopping basado en la mejora de la pérdida de validación. Además, guarda el mejor modelo encontrado durante el entrenamiento y genera una gráfica de la curva de pérdida para entrenamiento y validación.

    Returns:
        _type_: _description_   Esta función no retorna ningún valor, pero realiza todo el proceso de entrenamiento del modelo, incluyendo la configuración del entorno, la carga de datos, la definición del modelo y los componentes de entrenamiento, la ejecución del ciclo de entrenamiento con early stopping, y la generación de gráficos para visualizar el progreso del entrenamiento.
    """    
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

    model=InceptionTime(c_in=2, c_out=2, seq_len=None, n_filters=32, depth = 6)
    #model=Modelo_Convolucional(in_channels=2,out_channels=2, long_signal=250)
    #model=Modelo_ConvolucionalV1(in_channels=2,out_channels=3, long_signal=500)
    #model=Modelo_ConvolucionalV1(in_channels=2,out_channels=2, long_signal=500)
    #model=Modelo_ConvolucionalV2(in_channels=2,out_channels=2, long_signal=500)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)  # Mueve el modelo a la GPU
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay= 1e-3)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, min_lr=1e-6, verbose=True)    
    criterion = torch.nn.MSELoss()  # MSELoss para regresión
    scaler = torch.amp.GradScaler('cuda')

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
        """_summary_ Realiza un paso de entrenamiento: procesa un batch de datos, calcula la pérdida, realiza backpropagation y actualiza los pesos del modelo utilizando precisión mixta (AMP) para optimizar el rendimiento en GPU. La función reinicia los gradientes, mueve los datos y etiquetas a la GPU, realiza la predicción y el cálculo de la pérdida dentro del contexto de autocast para aprovechar la precisión mixta, y luego escala la pérdida para realizar backpropagation y actualizar los pesos del modelo.

        Args:
            batch (_type_): _description_ Batch de datos que contiene las señales de entrada, las etiquetas de presión arterial, los IDs de pacientes y los índices, que se utilizará para realizar un paso de entrenamiento en el modelo.
            l1_lambda (_type_, optional): _description_. Por defecto 1e-5. Coeficiente de regularización L1 

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
        """_summary_ Realiza un paso de evaluación: procesa un batch de datos de validación, calcula la pérdida sin realizar backpropagation ni actualizar los pesos del modelo. La función mueve los datos y etiquetas a la GPU, realiza la predicción y el cálculo de la pérdida.

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
