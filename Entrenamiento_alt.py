import torch
from Clase_UCIDataset import UCIDataset
import os
from torch.utils import data
from torch.utils.data import TensorDataset, random_split
import numpy as np
#from Modelos.InceptionTime import InceptionTime
#from Modelos.Modelo_conv import Modelo_Convolucional
#from Modelos.ConvolucionalV1 import Modelo_ConvolucionalV1
from Modelos.ConvolucionalV2 import Modelo_ConvolucionalV2
from tqdm.auto import tqdm 
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import ReduceLROnPlateau

def main():
    # Configuración del dispositivo
    parameters = {
        'batch_size': 256,
        'shuffle': True,
        'num_workers': 3,
        'pin_memory': True
    }
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
    'data_UCI/dataset_parte_1.pt',
    'data_UCI/dataset_parte_2.pt',
    'data_UCI/dataset_parte_3.pt',
    'data_UCI/dataset_parte_4.pt',
    ]

    todas_las_senales = []
    todas_las_etiquetas = []

    for archivo in archivos:
        data = torch.load(archivo)
        todas_las_senales.append(data['data'])
        todas_las_etiquetas.append(data['labels'])

    tensor_señales = torch.cat(todas_las_senales, dim=0)
    tensor_etiquetas = torch.cat(todas_las_etiquetas, dim=0)

    print(f"Datos cargados: señales {tensor_señales.shape}, etiquetas {tensor_etiquetas.shape}")

    dataset_completo = TensorDataset(tensor_señales, tensor_etiquetas)

    total = len(dataset_completo)
    train_size = int(0.7 * total)
    val_size = int(0.2 * total)
    test_size = total - train_size - val_size

    # Dividir aleatoriamente el dataset
    train_set, val_set, test_set = random_split(dataset_completo, [train_size, val_size, test_size])

    print(f"Train: {len(train_set)}, Val: {len(val_set)}, Test: {len(test_set)}")

    training_generator = torch.utils.data.DataLoader(train_set, **parameters)
    validation_generator = torch.utils.data.DataLoader(val_set, **parameters)

    test_signals = []
    test_labels = []

    for signals, labels in test_set:
        test_signals.append(signals)
        test_labels.append(labels)

    # Convertir listas a tensores
    test_signals = torch.stack(test_signals)
    test_labels = torch.stack(test_labels)

    # Guardar en un archivo .pt
    torch.save({'data': test_signals, 'labels': test_labels}, 'data_UCI/test_set.pt')
    
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
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, min_lr=1e-6, verbose=True)    
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
        #torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step() # Actualiza los parámetros del modelo
        return loss.item() # Devuelve la pérdida

    def evaluate_one_step(batch):
        with torch.no_grad():
            data, labels = batch
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
    min_delta = 0.0005  # mejora mínima requerida
    patience = 5
    patience_significativa = 5
  
    for epoch in tqdm(range(max_epochs)):
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
                'loss': valid_l
            }, 'best_model_conv_v2.pt')
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
    plt.savefig('loss_curve.png')
    plt.show()

if __name__ == '__main__':
    main()