import torch
from src.data.data_chargers.Clase_UCIDataset import UCIDataset
import os
from torch.utils import data
from torch.utils.data import TensorDataset, random_split
import numpy as np
from src.models.ConvolucionalV1 import Modelo_ConvolucionalV1
from tqdm.auto import tqdm 
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import ReduceLROnPlateau
import random
from . import Entrenamiento


def train_one_step(batch, optimizer, model, criterion, device):
    optimizer.zero_grad() # Reinicia los gradientes
    data, labels, _, _ = batch # Obtiene los datos y etiquetas
    
    labels = labels[:,1].unsqueeze(1) # Usar solo la primera columna de etiquetas (SBP)

    data, labels = data.to(device), labels.to(device) # Mueve los datos y etiquetas a la GPU
    preds = model.forward(data) # Realiza la predicción
    loss = criterion(preds, labels) # Calcula la pérdida
    loss.backward() # Calcula los gradientes mediante backpropagation
    optimizer.step() # Actualiza los parámetros del modelo
    return loss.item() # Devuelve la pérdida

def evaluate_one_step(batch, model, criterion, device):
    with torch.no_grad():
        data, labels, _, _ = batch

        labels = labels[:,1].unsqueeze(1) # Usar solo la primera columna de etiquetas (SBP)

        data, labels = data.to(device), labels.to(device)
        preds = model.forward(data)
        loss = criterion(preds, labels)
        return loss.item()

def train_one_epoch(model, training_generator, validation_generator, optimizer=None, criterion=None, device='cpu'):    
    train_loss, valid_loss = 0.0, 0.0

    model.train()
    for batch in training_generator:
    #for batch in subset_loader:
        train_loss += train_one_step(batch, optimizer, model, criterion, device)
        

    model.eval()    
    for batch in validation_generator:
    #for batch in subset_loader:
        valid_loss += evaluate_one_step(batch, model, criterion, device)

    return train_loss/len(training_generator), valid_loss/len(validation_generator)

def main():
        
    parameters = {
        'batch_size': 256,
        'shuffle': True,
        'num_workers': 0,
        'pin_memory': False
    }
    Entrenamiento.set_seed(42)
    
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

    
    meta_test = Entrenamiento.save_test_dataset(
    save_dir='data/processed/data_UCI/test_set_por_picos',
    prefix="test",
    data=test_data,         # (N,2,L)
    labels=test_labels,     # (N,2)
    patients=test_patients, # (N,)
    indexs=test_indexs      # (N,)
    )
    print("data/processed/data_UCI/test_set_por_picos guardado")

    model = Modelo_ConvolucionalV1(in_channels=2,out_channels=1, long_signal=500)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)  # Mueve el modelo a la GPU
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay= 1e-4)
    criterion = torch.nn.MSELoss()  # MSELoss para regresión

    max_epochs = 100
    best_valid_loss = float('inf') 
    running_loss = np.zeros(shape=(max_epochs, 2))
    min_delta = 0.002  # mejora mínima requerida
    patience = 10
    patience_significativa = 5
  
    for epoch in tqdm(range(max_epochs)):
        train_l, valid_l = train_one_epoch(model, training_generator=training_generator, validation_generator=validation_generator, optimizer=optimizer, criterion=criterion, device=device)
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
            }, 'models/best_models/best_model_conv_v1_DBP.pt')
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

    