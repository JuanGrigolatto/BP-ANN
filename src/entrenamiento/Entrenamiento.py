import torch
from src.data.data_chargers.Clase_UCIDataset import UCIDataset
import os
from torch.utils import data
from torch.utils.data import TensorDataset, random_split
import numpy as np
#from src.models.InceptionTime import InceptionTime
#from src.models.Modelo_conv import Modelo_Convolucional
from src.models.ConvolucionalV1 import Modelo_ConvolucionalV1
#from src.models.ConvolucionalV2 import Modelo_ConvolucionalV2
from tqdm.auto import tqdm 
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import ReduceLROnPlateau
import random
import src.utils.Tools.Tools as Tools

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # para multi-GPU

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

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

def main():
    # Configuración del dispositivo
    parameters = {
        'batch_size': 64,
        'shuffle': True,
        'num_workers': 0,
        'pin_memory': False
    }
    set_seed(42)
    """
    print(os.path.exists('data_UCI/dataset_completo.pt'))
    dataset = UCIDataset('data_UCI/dataset_completo.pt')
    
    
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    
    training_set, validation_set = random_split(dataset, [train_size, val_size])
    
    training_generator = torch.utils.data.DataLoader(training_set, **parameters)
    validation_generator = torch.utils.data.DataLoader(validation_set, **parameters)
    """
    archivos = [
    'data/processed/data_UCI/dataset_parte_1_por_picos.pt',
    'data/processed/data_UCI/dataset_parte_2_por_picos.pt',
    'data/processed/data_UCI/dataset_parte_3_por_picos.pt',
    'data/processed/data_UCI/dataset_parte_4_por_picos.pt',
    ]

    dataset_completo = UCIDataset(archivos)

    total = len(dataset_completo)
    train_size = int(0.7 * total)
    val_size = int(0.2 * total)
    test_size = total - train_size - val_size

    # Dividir aleatoriamente el dataset
    train_set, val_set, test_set = random_split(dataset_completo, [train_size, val_size, test_size])


    print(f"Train: {len(train_set)}, Val: {len(val_set)}, Test: {len(test_set)}")

    training_generator = torch.utils.data.DataLoader(train_set, **parameters)
    validation_generator = torch.utils.data.DataLoader(val_set, **parameters)
    test_generator = torch.utils.data.DataLoader(test_set, **parameters)

    """
    test_signals = []
    test_labels = []
    test_IDs = []
    test_index = []

    for signals, labels, IDs, index in test_set:
        test_signals.append(signals)
        test_labels.append(labels)
        test_IDs.append(IDs)
        test_index.append(index)

    # listas a tensores
    test_signals = torch.stack(test_signals)
    test_labels = torch.stack(test_labels)
    test_IDs = torch.stack(test_IDs)
    test_index = torch.stack(test_index)
    
    # Guardado en un archivo .pt
    torch.save({'data': test_signals, 'labels': test_labels,'patient_ids': test_IDs,'index':test_index} ,'data_UCI/test_set_hanning.pt')
    print("data_UCI/test_set.pt guardado")
    """
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
    #Subset para testeo de modelos sin entrenamiento completo
    """
    subset_size = int(0.5 * len(dataset))  # 10%
    subset_indices = np.random.choice(len(dataset), subset_size, replace=False)
    subset = torch.utils.data.Subset(dataset, subset_indices)
    subset_loader = torch.utils.data.DataLoader(subset, **parameters)
    """

    # Crear el modelo

    #odel=InceptionTime(c_in=2, c_out=3, seq_len=None, n_filters=32)
    #model=Modelo_Convolucional(in_channels=2,out_channels=2, long_signal=250)
    model=Modelo_ConvolucionalV1(in_channels=2,out_channels=3, long_signal=500)
    #model=Modelo_ConvolucionalV2(in_channels=2,out_channels=2, long_signal=1250)
    # Añade esto después de crear el modelo

    


    """
    for layer in model.modules():
        if isinstance(layer, (torch.nn.Conv1d, torch.nn.Linear)):
            torch.nn.init.xavier_uniform_(layer.weight)
            torch.nn.init.zeros_(layer.bias)
    """   
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)  # Mueve el modelo a la GPU
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay= 1e-4)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, min_lr=1e-6, verbose=True)    
    criterion = torch.nn.MSELoss()  # MSELoss para regresión

    #retomar entrenamiento 
    start_epoch = 0
    checkpoint_path = "models/best_models/best_model_conv_v1_con_PAM_100_layerout.pt"
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = checkpoint.get("epoca", 0) + 1
        best_valid_loss = checkpoint.get("best_valid_loss", float('inf')) 
        print(f" Modelo cargado desde {checkpoint_path}, continuando en epoch {start_epoch}, best_valid_loss={best_valid_loss:.6f}")
    else:
        print("⚠ No se encontró un checkpoint, se comienza entrenamiento desde cero.")

    # ENTRENAMIENTO
    def train_one_step(batch):
        optimizer.zero_grad() # Reinicia los gradientes
        data, labels, _, _ = batch # Obtiene los datos y etiquetas

        labels_sbp= labels[:,0].unsqueeze(1)
        labels_dbp= labels[:,1].unsqueeze(1)
        labels_pam = Tools.calcular_pam(labels_sbp, labels_dbp)

        labels = torch.cat((labels_sbp, labels_dbp, labels_pam), dim=1)

        data, labels = data.to(device), labels.to(device) # Mueve los datos y etiquetas a la GPU
        
        #print("Rango de datos:", torch.min(data).item(), torch.max(data).item())

        #for name, param in model.named_parameters():
        #    print(f"{name}: mean={param.mean().item():.4f}, std={param.std().item():.4f}")

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
        loss.backward() # Calcula los gradientes mediante backpropagation
        # Añade gradient clipping (busca evitar explosión de gradiente)
        #torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step() # Actualiza los parámetros del modelo
        return loss.item() # Devuelve la pérdida

    def evaluate_one_step(batch):
        with torch.no_grad():
            data, labels, _, _ = batch

            labels_sbp= labels[:,0].unsqueeze(1)
            labels_dbp= labels[:,1].unsqueeze(1)
            labels_pam = Tools.calcular_pam(labels_sbp, labels_dbp)
            labels = torch.cat((labels_sbp, labels_dbp, labels_pam), dim=1)

            data, labels = data.to(device), labels.to(device)
            preds = model.forward(data)
            loss = criterion(preds, labels)
            return loss.item()
    
    def train_one_epoch():    
        train_loss, valid_loss = 0.0, 0.0

        model.train()
        for batch in training_generator:
        #for batch in subset_loader:
            train_loss += train_one_step(batch)
        

        model.eval()    
        for batch in validation_generator:
        #for batch in subset_loader:
            valid_loss += evaluate_one_step(batch)

        return train_loss/len(training_generator), valid_loss/len(validation_generator)
    
    max_epochs = 100
    best_valid_loss = float('inf') 
    running_loss = np.zeros(shape=(max_epochs, 2))
    min_delta = 0.00002  # mejora mínima requerida
    patience = 15
    patience_significativa = 15
  
    for epoch in tqdm(range(start_epoch, max_epochs)):
        train_l, valid_l = train_one_epoch()
        running_loss[epoch] = (train_l,valid_l)
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
            }, 'models/best_models/best_model_conv_v1_con_PAM_100_layerout.pt')
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

    fig, ax = plt.subplots(figsize=(7, 4), tight_layout=True)
    ax.plot(running_loss[:epoch, 0], label='Entrenamiento')
    ax.plot(running_loss[:epoch, 1], label='Validación')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.legend()
    plt.savefig('graficas/loss_curve.png')
    plt.show()

if __name__ == '__main__':
    main()


    
     