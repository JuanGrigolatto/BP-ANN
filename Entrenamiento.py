import torch
from Clase_UCIDataset import UCIDataset
import os
import numpy as np
from Modelos.InceptionTime import InceptionTime
from tqdm.auto import tqdm 
import matplotlib.pyplot as plt
#%%
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")
#%%
parameters = {
    'batch_size': 64,
    'shuffle': True,
    'num_workers': 0,
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

    # Conjuntos de datos y los DataLoaders
    training_set = UCIDataset(partition['train'], data_dir=data_dir)	
    training_generator = torch.utils.data.DataLoader(training_set, **parameters)

    validation_set = UCIDataset(partition['validation'], data_dir=data_dir)
    validation_generator = torch.utils.data.DataLoader(validation_set, **parameters)

    # Crear el modelo
    model=InceptionTime(c_in=2, c_out=2, seq_len=None, n_filters=32, nb_filters=None)
    model = model.to(device)  # Mueve el modelo a la GPU o CPU según corresponda
    # Se Define el optimizador y la función de pérdida
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = torch.nn.MSELoss()  # MSELoss para regresión

    global best_valid_loss
    best_valid_loss = float('inf')  
    max_epochs, best_valid_loss = 5, np.inf
    running_loss = np.zeros(shape=(max_epochs, 2))  # Para almacenar las pérdidas de entrenamiento y validación
    for epoch in tqdm(range(max_epochs)):
        # Training
        train_loss, valid_loss = 0.0, 0.0
        for local_batch, local_labels in training_generator:
            optimizer.zero_grad() # Reset gradients
            # Transfer to GPU
            local_batch, local_labels = local_batch.to(device), local_labels.to(device)
            preds = model.forward(local_batch) # Realiza la predicción
            loss = criterion(preds, local_labels) # Calcula la pérdida
            loss.backward() # Calcula los gradientes mediante backpropagation
            optimizer.step() # Actualiza los parámetros del modelo
            train_loss += loss.item()  # Acumula la pérdida de entrenamiento

        with torch.set_grad_enabled(False):
            for local_batch, local_labels in validation_generator:
                # Transfer to GPU
                local_batch, local_labels = local_batch.to(device), local_labels.to(device)
                preds = model.forward(local_batch)
                loss = criterion(preds, local_labels)
                valid_loss += loss.item()
                
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save({'epoca': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': valid_loss}, 
                   'best_model.pt')
        running_loss[epoch] = [train_loss / len(training_generator), valid_loss / len(validation_generator)]
    
    fig, ax = plt.subplots(figsize=(7, 4), tight_layout=True)
    ax.plot(running_loss[:, 0], label='Entrenamiento')
    ax.plot(running_loss[:, 1], label='Validación')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.legend()
    plt.show()


    
     