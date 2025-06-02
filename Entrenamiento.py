#%%
import torch
from Clase_UCIDataset import UCIDataset
import os
from torch.utils import data
import numpy as np
from Modelos.InceptionTime import InceptionTime
from tqdm.auto import tqdm 
import matplotlib.pyplot as plt
#%%
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")
print(device)
#%%

# Versión optimizada que precarga en RAM
class RAMDataset(data.Dataset):
    def __init__(self, uci_dataset):
        """Precarga en RAM los datos de un UCIDataset"""
        self.data = []
        self.labels = []
        
        print("Precargando datos en RAM...")
        for i in tqdm(range(len(uci_dataset)), desc="Cargando datos"):
            x, y = uci_dataset[i]
            self.data.append(x)
            self.labels.append(y)
        
        # Convertimos a tensores
        self.data = torch.stack(self.data)
        self.labels = torch.stack(self.labels)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        return self.data[index], self.labels[index]
    
parameters = {
    'batch_size': 256,
    'shuffle': True,
    'num_workers': 2,
    'pin_memory': True
}    
if __name__ == '__main__':
    # Obtener los IDs de los archivos
    data_dir = 'datos_UCI'
    all_IDs = [f[:-3] for f in os.listdir(data_dir) if f.endswith('.pt')]

    #  Aleatorizar los IDs
    np.random.shuffle(all_IDs)

    # Crear la partición (80% train, 20% validation)
    train_size = int(0.8 * len(all_IDs))
    partition = {
        'train': all_IDs[:train_size],
        'validation': all_IDs[train_size:]
    }

        # Crear datasets y precargar en RAM
    print("\nPrecargando conjunto de entrenamiento...")
    train_dataset = RAMDataset(UCIDataset(partition['train'], data_dir))
    
    print("\nPrecargando conjunto de validación...")
    val_dataset = RAMDataset(UCIDataset(partition['validation'], data_dir))

    # Crear DataLoaders
    train_loader = torch.utils.data.DataLoader(train_dataset, **parameters)
    val_loader = torch.utils.data.DataLoader(val_dataset, **parameters)

    # Verificación de uso de memoria
    print(f"\nMemoria usada por datos de entrenamiento: {train_dataset.data.element_size() * train_dataset.data.nelement() / (1024**2):.2f} MB")
    print(f"Memoria usada por datos de validación: {val_dataset.data.element_size() * val_dataset.data.nelement() / (1024**2):.2f} MB")



    """
    training_set = UCIDataset(partition['train'], data_dir=data_dir)	
    training_generator = torch.utils.data.DataLoader(training_set, **parameters)

    validation_set = UCIDataset(partition['validation'], data_dir=data_dir)
    validation_generator = torch.utils.data.DataLoader(validation_set, **parameters)
    """

    # Crear el modelo
    model=InceptionTime(c_in=2, c_out=2, seq_len=None, n_filters=32, nb_filters=None)
    model = model.to(device)  # Mueve el modelo a la GPU o CPU según corresponda
    # Se Define el optimizador y la función de pérdida
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = torch.nn.MSELoss()  # MSELoss para regresión
    #Parametros de entrenamiento
    global best_valid_loss
    best_valid_loss = float('inf')  
    max_epochs, best_valid_loss = 100, np.inf
    patience = 5
    epochs_no_improve = 0

    running_loss = np.zeros(shape=(max_epochs, 2))  # Para almacenar las pérdidas de entrenamiento y validación
    for epoch in tqdm(range(max_epochs), desc="Entrenamiento"):
        # Training
        train_loss, valid_loss = 0.0, 0.0
        for local_batch, local_labels in train_loader:
            optimizer.zero_grad() # Reset gradients
            # Transfer to GPU
            local_batch, local_labels = local_batch.to(device), local_labels.to(device)
            preds = model.forward(local_batch) # Realiza la predicción
            loss = criterion(preds, local_labels) # Calcula la pérdida
            loss.backward() # Calcula los gradientes mediante backpropagation
            optimizer.step() # Actualiza los parámetros del modelo
            train_loss += loss.item()  # Acumula la pérdida de entrenamiento

        with torch.set_grad_enabled(False):
            for local_batch, local_labels in val_loader:
                # Transfer to GPU
                local_batch, local_labels = local_batch.to(device), local_labels.to(device)
                preds = model.forward(local_batch)
                loss = criterion(preds, local_labels)
                valid_loss += loss.item()
                
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            epochs_no_improve = 0
            torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': valid_loss,
            'model_config': {'c_in': 2, 'c_out': 2, 'n_filters': 32}  # Parámetros de InceptionTime
            },'best_model.pt')
        else:
            epochs_no_improve += 1
            if epochs_no_improve == patience:
                print("Early Stopping!")
                break
        running_loss[epoch] = [train_loss / len(train_loader), valid_loss / len(val_loader)]
    
    fig, ax = plt.subplots(figsize=(7, 4), tight_layout=True)
    ax.plot(running_loss[:, 0], label='Entrenamiento')
    ax.plot(running_loss[:, 1], label='Validación')
    ax.set_title('Pérdida en Entrenamiento y Validación')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.legend()
    plt.savefig('loss_curve.png')
    plt.show()


    
     