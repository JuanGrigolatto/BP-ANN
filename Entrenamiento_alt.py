import torch
from Clase_UCIDataset import UCIDataset
import os
from torch.utils import data
from torch.utils.data import TensorDataset, random_split
import numpy as np
#from Modelos.InceptionTime import InceptionTime
#from Modelos.Modelo_conv import Modelo_Convolucional
from Modelos.ConvolucionalV1 import Modelo_ConvolucionalV1
#from Modelos.ConvolucionalV2 import Modelo_ConvolucionalV2
from tqdm.auto import tqdm 
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import ReduceLROnPlateau
import random

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # para multi-GPU

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def white_noise_torch(signal: torch.Tensor, snr_db = random.randint(10, 30)) -> torch.Tensor:
    # Potencia de la señal
    potencia_senal = torch.mean(signal ** 2)

    # Convertir SNR de dB a escala lineal
    snr_lineal = 10 ** (snr_db / 10)

    # Calcular la potencia del ruido deseada
    potencia_ruido = potencia_senal / snr_lineal

    # Generar ruido blanco gaussiano
    ruido = torch.randn_like(signal) * torch.sqrt(potencia_ruido)

    senal_con_ruido = signal + ruido

    return senal_con_ruido

def main():
    # Configuración del dispositivo
    parameters = {
        'batch_size': 256,
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
    'data_UCI/dataset_parte_1.pt',
    'data_UCI/dataset_parte_2.pt',
    'data_UCI/dataset_parte_3.pt',
    'data_UCI/dataset_parte_4.pt',
    ]

    dataset_completo = UCIDataset(archivos)

    total = len(dataset_completo)
    train_size = int(0.7 * total)
    val_size = int(0.2 * total)
    test_size = total - train_size - val_size


    # Dividir aleatoriamente el dataset
    train_set, val_set, test_set = random_split(dataset_completo, [train_size, val_size, test_size])

    # Guarda los índices originales del entrenamiento
    train_indices = train_set.indices  
    val_indices = val_set.indices
    test_indices = test_set.indices

    print(f"Train: {len(train_set)}, Val: {len(val_set)}, Test: {len(test_set)}")

    training_generator = torch.utils.data.DataLoader(train_set, **parameters)
    validation_generator = torch.utils.data.DataLoader(val_set, **parameters)

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
    torch.save({'data': test_signals, 'labels': test_labels,'patient_ids': test_IDs,'index':test_index} ,'data_UCI/test_set.pt')
    print("data_UCI/test_set.pt guardado")
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
    model=Modelo_ConvolucionalV1(in_channels=2,out_channels=2, long_signal=1250)
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
    criterion_train = torch.nn.MSELoss()  # MSELoss para regresión
    criterion_valid = torch.nn.MSELoss(reduction='none') #error por muestra
 
    # ENTRENAMIENTO
    def train_one_step(batch):
        optimizer.zero_grad() # Reinicia los gradientes
        data, labels, _, _ = batch # Obtiene los datos y etiquetas
        data, labels = data.to(device), labels.to(device) # Mueve los datos y etiquetas a la GPU
        
        #print("Rango de datos:", torch.min(data).item(), torch.max(data).item())

        #for name, param in model.named_parameters():
        #    print(f"{name}: mean={param.mean().item():.4f}, std={param.std().item():.4f}")

        preds = model.forward(data) # Realiza la predicción
        #print("Preds:", preds[0])
        #print("Labels:", labels[0])
        loss = criterion_train(preds, labels) # Calcula la pérdida
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
            data, labels, _, index= batch
            data, labels = data.to(device), labels.to(device)
            preds = model.forward(data)
            loss = criterion_valid(preds, labels) # [B, 2]
            sample_loss = loss.mean(dim=1)        # [B]
            return sample_loss, index
    
    def train_one_epoch():    
        train_loss, valid_loss = 0.0, 0.0

        all_errors = []
        all_indexs = []

        model.train()
        for batch in training_generator:
        #for batch in subset_loader:
            train_loss += train_one_step(batch)
        

        model.eval()    
        for batch in validation_generator:
       
            sample_loss, sample_index = evaluate_one_step(batch)
            valid_loss += sample_loss.mean().item()  # promedio del batch
            all_errors.extend(sample_loss.cpu().numpy())
            all_indexs.extend(sample_index.cpu().numpy())

        all_errors = np.array(all_errors)

        threshold = np.percentile(all_errors, 10)  # top 10% de errores
        
        indices_errores = []

        for idx, err in zip(all_indexs, all_errors):
            if err > threshold:
                indices_errores.append(idx)

        return train_loss/len(training_generator), valid_loss/len(validation_generator), indices_errores
    
    max_epochs = 100
    best_valid_loss = float('inf') 
    running_loss = np.zeros(shape=(max_epochs, 2))
    min_delta = 0.00002  # mejora mínima requerida
    patience = 5
    patience_significativa = 5
    max_augmentation = 1000
    errores_augmentados = set()
    total_augmented = 0

    for epoch in tqdm(range(max_epochs)):
        train_l, valid_l, indices_errores = train_one_epoch()
        running_loss[epoch] = (train_l,valid_l)

        
        print(f"[Época {epoch}] Train Loss: {train_l:.6f} - Valid Loss: {valid_l:.6f}")

        if len(indices_errores) > 0 and epoch > 5:  # Filtra errores que ya fueron augmentados
            nuevos_indices_errores = []
            for idx in indices_errores:
                if idx not in errores_augmentados:
                    nuevos_indices_errores.append(idx)
            
            #Se limita solo a 1000 errores
            indices_errores = nuevos_indices_errores[:max_augmentation] 

            print(f"Errores no repetidos seleccionados para augmentación: {len(indices_errores)}")

            if len(indices_errores) > 50:


                errores_augmentados.update(indices_errores)
                x_errores =  dataset_completo.data[indices_errores]
                y_errores = dataset_completo.labels[indices_errores]
                id_errores = dataset_completo.ID_patients[indices_errores]
                index_errores = dataset_completo.index_muestra[indices_errores]

                x_augmented = white_noise_torch(x_errores)
                y_augmented = y_errores
                id_augmented = id_errores
                index_augmented = index_errores 

                dataset_completo.data = torch.cat([dataset_completo.data, x_augmented], dim=0)
                dataset_completo.labels = torch.cat([dataset_completo.labels, y_augmented], dim=0)
                dataset_completo.ID_patients = torch.cat([dataset_completo.ID_patients, id_augmented], dim=0)
                dataset_completo.index_muestra= torch.cat([dataset_completo.index_muestra, index_augmented], dim=0)
                
                total_augmented += len(x_augmented)
                print(f"Total augmentado acumulado: {total_augmented}")
                #Se actualiza train_set para agregar las muestras aumentadas al final
            
                #Indices de muestras agregadas
                nuevos_indices = list(range(len(dataset_completo) - len(x_augmented), len(dataset_completo)))
                train_indices.extend(nuevos_indices)
            
                train_set = torch.utils.data.Subset(dataset_completo, train_indices)
                training_generator = torch.utils.data.DataLoader(train_set, **parameters)


                print(f"Se realiza data augmentation con {len(x_errores)} muestras")
            

        improvement = best_valid_loss - valid_l

        if valid_l < best_valid_loss:
            best_valid_loss = valid_l
            no_improvement_count = 0  
            torch.save({
                'epoca': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': valid_l
            }, 'Best_models/best_model_conv_v1_aug.pt')
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