import torch
from Clase_UCIDataset import UCIDataset
import os
from torch.utils import data
from torch.utils.data import random_split
import numpy as np
#from Modelos.InceptionTime import InceptionTime
#from Modelos.Modelo_conv import Modelo_Convolucional
#from Modelos.ConvolucionalV1 import Modelo_ConvolucionalV1
from Modelos.ConvolucionalV2 import Modelo_ConvolucionalV2
from tqdm.auto import tqdm 
import matplotlib.pyplot as plt

def main():
    # Configuración del dispositivo
    parameters = {
        'batch_size': 256,
        'shuffle': True,
        'num_workers': 3,
        'pin_memory': True
    }

    print(os.path.exists('data_UCI/dataset_completo.pt'))
    dataset = UCIDataset('data_UCI/dataset_completo.pt')
    
    
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    
    training_set, validation_set = random_split(dataset, [train_size, val_size])
    
    training_generator = torch.utils.data.DataLoader(training_set, **parameters)
    validation_generator = torch.utils.data.DataLoader(validation_set, **parameters)
    
    #Subset para testeo de modelos sin entrenamiento completo
    """
    subset_size = int(0.5 * len(dataset))  # 10%
    subset_indices = np.random.choice(len(dataset), subset_size, replace=False)
    subset = torch.utils.data.Subset(dataset, subset_indices)
    subset_loader = torch.utils.data.DataLoader(subset, **parameters)
    """

    # Crear el modelo

    #model=InceptionTime(c_in=2, c_out=2, seq_len=None, n_filters=32)
    #model=Modelo_Convolucional(in_channels=2,out_channels=2, long_signal=250)
    #model=Modelo_ConvolucionalV1(in_channels=2,out_channels=2, long_signal=250)
    model=Modelo_ConvolucionalV2(in_channels=2,out_channels=2, long_signal=250)
    # Añade esto después de crear el modelo
    """
    for layer in model.modules():
        if isinstance(layer, (torch.nn.Conv1d, torch.nn.Linear)):
            torch.nn.init.xavier_uniform_(layer.weight)
            torch.nn.init.zeros_(layer.bias)
    """   
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)  # Mueve el modelo a la GPU
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = torch.nn.MSELoss()  # MSELoss para regresión

    # ENTRENAMIENTO
    def train_one_step(batch):
        optimizer.zero_grad() # Reinicia los gradientes
        data, labels = batch # Obtiene los datos y etiquetas
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
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step() # Actualiza los parámetros del modelo
        return loss.item() # Devuelve la pérdida

    def evaluate_one_step(batch):
        with torch.no_grad():
            data, labels = batch
            data, labels = data.to(device), labels.to(device)
            preds = model.forward(data)
            loss = criterion(preds, labels)
            return loss.item()
    
    def train_one_epoch(epoch, best_valid_loss):    
        train_loss, valid_loss = 0.0, 0.0

        model.train()
        for batch in training_generator:
        #for batch in subset_loader:
            train_loss += train_one_step(batch)

        model.eval()    
        for batch in validation_generator:
        #for batch in subset_loader:
            valid_loss += evaluate_one_step(batch)


        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save({'epoca': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'loss': valid_loss}, 
                    'best_model_conv_v2.pt')

        return train_loss/len(training_generator), valid_loss/len(validation_generator), best_valid_loss
        #return train_loss/len(subset_loader), valid_loss/len(subset_loader), best_valid_loss
    #global best_valid_loss
    max_epochs = 100
    best_valid_loss = float('inf') 
    running_loss = np.zeros(shape=(max_epochs, 2))

    for epoch in tqdm(range(max_epochs)):
        train_l, valid_l, best_valid_loss = train_one_epoch(epoch, best_valid_loss)
        running_loss[epoch] = (train_l,valid_l)

    fig, ax = plt.subplots(figsize=(7, 4), tight_layout=True)
    ax.plot(running_loss[:, 0], label='Entrenamiento')
    ax.plot(running_loss[:, 1], label='Validación')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.legend()
    plt.savefig('loss_curve.png')
    plt.show()

if __name__ == '__main__':
    main()