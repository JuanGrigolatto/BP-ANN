
import torch
from Clase_UCIDataset import UCIDataset
import os
from torch.utils import data
import numpy as np
#from Modelos.InceptionTime import InceptionTime
from Modelos.Modelo_conv import Modelo_Convolucional
from tqdm.auto import tqdm 
import matplotlib.pyplot as plt

# Configuración del dispositivo
parameters = {
    'batch_size': 64,
    'shuffle': True,
    'num_workers': 0,
    'pin_memory': False
}

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

training_set = UCIDataset(partition['train'], data_dir=data_dir)	
training_generator = torch.utils.data.DataLoader(training_set, **parameters)

validation_set = UCIDataset(partition['validation'], data_dir=data_dir)
validation_generator = torch.utils.data.DataLoader(validation_set, **parameters)

# Crear el modelo

#model=InceptionTime(c_in=2, c_out=2, seq_len=None, n_filters=8, nb_filters=3)
model=Modelo_Convolucional(in_channels=2,out_channels=2, long_signal=250)   
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)  # Mueve el modelo a la GPU
optimizer = torch.optim.Adam(model.parameters(), lr=1e-6)
criterion = torch.nn.MSELoss()  # MSELoss para regresión

# ENTRENAMIENTO
def train_one_step(batch):
    optimizer.zero_grad() # Reinicia los gradientes
    data, labels = batch # Obtiene los datos y etiquetas
    data, labels = data.to(device), labels.to(device) # Mueve los datos y etiquetas a la GPU
    preds = model.forward(data) # Realiza la predicción
    print("Preds:", preds[0])
    print("Labels:", labels[0])
    loss = criterion(preds, labels) # Calcula la pérdida
    """
    if torch.isnan(loss):
        print("¡Loss con NaN detectado! Abortando...")
        print("Preds:", preds[0])
        print("Labels:", labels[0])
        exit()
    """
    loss.backward() # Calcula los gradientes mediante backpropagation
    optimizer.step() # Actualiza los parámetros del modelo
    return loss.item() # Devuelve la pérdida

def evaluate_one_step(batch):
    with torch.no_grad():
        data, labels = batch
        data, labels = data.to(device), labels.to(device)
        preds = model.forward(data)
        loss = criterion(preds, labels)
        return loss.item()
    
def train_one_epoch(epoch):    
    train_loss, valid_loss = 0.0, 0.0
       
    for batch in training_generator:
        train_loss += train_one_step(batch)    
    for batch in validation_generator:
        valid_loss += evaluate_one_step(batch)
        
    global best_valid_loss
    best_valid_loss = float('inf')     
    if valid_loss < best_valid_loss:
        best_valid_loss = valid_loss
        torch.save({'epoca': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': valid_loss}, 
                   'best_model.pt')

    return train_loss/len(training_generator), valid_loss/len(validation_generator)

max_epochs, best_valid_loss = 100, np.inf
running_loss = np.zeros(shape=(max_epochs, 2))
for epoch in tqdm(range(max_epochs)):
    running_loss[epoch] = train_one_epoch(epoch)

fig, ax = plt.subplots(figsize=(7, 4), tight_layout=True)
ax.plot(running_loss[:, 0], label='Entrenamiento')
ax.plot(running_loss[:, 1], label='Validación')
ax.set_xlabel('Epoch')
ax.set_ylabel('Loss')
ax.legend()
plt.show()