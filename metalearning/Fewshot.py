import torch.utils
from src.data.data_chargers.Clase_UCIDataset import UCIDataset
from src.data.data_chargers.MetaDataset import TaskDataset
from src.models.Modelo_conv import Modelo_Convolucional
from src.models.ConvolucionalV1 import Modelo_ConvolucionalV1
import numpy as np
from torch import nn, optim
import matplotlib.pyplot as plt
#from src.models.InceptionTime import InceptionTime
import torch.utils.data as data
import torch
from src.data.data_chargers.Tuningndataset import TuningNDataset
import random
from sklearn.metrics import r2_score

def desnormalizar_zscore(norm_array, media, std):
    return norm_array * std + media

def desnormalizar_minmax(norm_array, min_val, max_val):
    return norm_array * (max_val - min_val) + min_val

def tuning(sample, optimizer, model, criterion, device):
        optimizer.zero_grad() # Reinicia los gradientes
        data, labels, *_ = sample # Obtiene los datos y etiquetas
        # Si vienen como listas (por batch_size=1), convertirlos a tensores
        if isinstance(data, list):
            data = data[0]
        if isinstance(labels, list):
            labels = labels[0]
            # Congelar todas las capas BatchNorm
        for layer in model.modules():
            if isinstance(layer, torch.nn.BatchNorm1d):
                layer.eval()
                layer.weight.requires_grad = False
                layer.bias.requires_grad = False

        data, labels = data.to(device), labels.to(device) # Mueve los datos y etiquetas a la GPU
        preds = model.forward(data) # Realiza la predicción
        loss = criterion(preds, labels) # Calcula la pérdida
        loss.backward() # Calcula los gradientes mediante backpropagation
        optimizer.step() # Actualiza los parámetros del modelo
        return loss.item() # Devuelve la pérdida

def evaluation(batch, model, criterion, device):
    with torch.no_grad():
        data, labels, *_ = batch
        # Si vienen como listas (por batch_size=1), convertirlos a tensores
        if isinstance(data, list):
            data = data[0]
        if isinstance(labels, list):
            labels = labels[0]
        data, labels = data.to(device), labels.to(device)
        preds = model.forward(data)
        loss = criterion(preds, labels)
    return preds, loss

def main(n_shots=5, num_tasks= 10000):
    SBP_MEAN = 134.02
    DBP_MEAN = 63.47
    SBP_STD = 22.75
    DBP_STD = 23.69
    
    test_data = torch.load('data/processed/data_UCI/few_shot_patient_data.pt')
    test_patient_ids = test_data['test_patient_ids']
    
    """
    data_dir = 'data_UCI/dataset_completo_prueba.pt'
    all_IDs = np.arange(0, num_tasks)
    """
    
    data_paths = [
        'data/processed/data_UCI/dataset_parte_1_por_picos.pt',
        'data/processed/data_UCI/dataset_parte_2_por_picos.pt',
        'data/processed/data_UCI/dataset_parte_3_por_picos.pt',
        'data/processed/data_UCI/dataset_parte_4_por_picos.pt'
    ]
    """
    all_data = []
    all_labels = []
    all_patient_ids = []

    for path in data_paths:
        dataset = torch.load(path)
        all_data.append(dataset['data'])
        all_labels.append(dataset['labels'])
        all_patient_ids.append(dataset['patient_ids'])

    merged_data = {
        'data': torch.cat(all_data, dim=0),
        'labels': torch.cat(all_labels, dim=0),
        'patient_ids': torch.cat(all_patient_ids, dim=0)
    }
    """
    dataset_completo = UCIDataset(data_paths)
    """
    unique_patients = merged_data['patient_ids'].unique().tolist()

    if len(unique_patients) >= num_tasks:
        list_IDs = random.sample(unique_patients, num_tasks)
    else:
        raise ValueError(f"Solo hay {len(unique_patients)} pacientes únicos, se solicita {num_tasks}")
    """

   
    taskset = TaskDataset(list_IDs=test_patient_ids, base_dataset=dataset_completo, num_shots=n_shots)
    #taskset = TaskDataset(list_IDs=list_IDs, data_dict=merged_data, num_shots=n_shots)
    id_patient_for_tuning =  random.choice(test_patient_ids)
    #id_patient_for_tuning =  random.choice(list_IDs)
    tuningset_for_train=TuningNDataset(taskset, id_patient_for_tuning, n_shots=n_shots, validation=False)
    tuningset_for_valid=TuningNDataset(taskset, id_patient_for_tuning, validation=True)

    print(f"ID paciente para fine tuning: {id_patient_for_tuning}")

    tuning_dataloader_TRAIN=torch.utils.data.DataLoader(tuningset_for_train, batch_size=1, shuffle=False)
    tuning_dataloader_VALID=torch.utils.data.DataLoader(tuningset_for_valid, batch_size=1, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    #Carga de modelo metaentrenado 
    #model=Modelo_Convolucional(in_channels=2,out_channels=2, long_signal=500)
    model=Modelo_ConvolucionalV1(in_channels=2,out_channels=2, long_signal=500)
    path_model='models/best_meta_models/best_meta_model_v1_100_epoch.pt'
    checkpoint = torch.load(path_model, map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Criterio de pérdida
    criterion = torch.nn.MSELoss()

    #Evaluación modelo previo a fine tuning
    model = model.to(device)  # Mueve el modelo a la GPU

    #Estrategia de evaluación previa al fine tuning

    # 1) Fine-tuning completo
    #for param in model.parameters():
    #    param.requires_grad = True

    # 2) Solo dense layers (Congelar capas de convolución)
    """
    for param in model.parameters():
        param.requires_grad = False
    for param in model.dense1.parameters():
        param.requires_grad = True
    for param in model.dense2.parameters():
        param.requires_grad = True
    """

    # 3) Congelar BatchNorm (Error con batch size = 1 si no se congela)
    """
    for m in model.modules():
        if isinstance(m, torch.nn.BatchNorm1d):
            m.eval()
            for param in m.parameters():
                param.requires_grad = False
    """
    #Fine tunning N-way K-shot    
    #optimizer = torch.optim.Adam(model.parameters(), lr=1e-3) 
    # Definicón del optimizador SOLO con los parámetros entrenables
    #optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    #preds_pre_fine_tuning = np.zeros((len(tuning_dataloader_VALID), n_shots, 2))
    #loss_pre_fine_tuning = np.zeros(len(tuning_dataloader_VALID))
    model.eval()    
    
    preds_pre_fine_tuning = []
    loss_pre_fine_tuning = []

    for batch in tuning_dataloader_VALID:
    #    preds_pre_fine_tuning[k] ,loss_pre_fine_tuning[k] = evaluation(batch, model, criterion, device)
        preds, loss = evaluation(batch, model, criterion, device)
        preds_pre_fine_tuning.extend(preds.detach().cpu().numpy())  
        loss_pre_fine_tuning.extend([loss.item()]*len(preds))

    preds_pre_fine_tuning = np.array(preds_pre_fine_tuning)
    loss_pre_fine_tuning = np.array(loss_pre_fine_tuning)
    
    labels = np.array([l.squeeze().cpu().numpy() for l in tuning_dataloader_VALID.dataset.labels])

    fig, ax = plt.subplots(figsize=(7, 4), tight_layout=True)
    ax.plot(loss_pre_fine_tuning, label='Loss pre fine tuning')
    ax.set_xlabel('number of sample')
    ax.set_ylabel('Loss')
    ax.legend()
    plt.savefig('metalearning/loss_pre_fine_tuning.png')
    plt.show()

    model.train()
    tuning_loss = np.zeros(shape=n_shots)
    for i, sample in enumerate(tuning_dataloader_TRAIN):
        tuning_loss[i] = tuning(sample,optimizer, model, criterion, device)
    
    torch.save({'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': tuning_loss[n_shots-1]}, 
                    'models/tuning_model.pt')

    fig, ax = plt.subplots(figsize=(7, 4), tight_layout=True)
    ax.plot(tuning_loss, label='Fine Tuning for 5 shots')
    ax.set_xlabel('number of shot')
    ax.set_ylabel('Loss')
    ax.legend()
    plt.savefig('metalearning/loss_curve_fine_tuning.png')
    plt.show()

    
    #Evaluación de modelo posterior al fine tuning

    #preds_post_fine_tuning = np.zeros((len(tuning_dataloader_VALID), n_shots, 2))
    #loss_post_fine_tuning = np.zeros(len(tuning_dataloader_VALID))

    preds_post_fine_tuning = []
    loss_post_fine_tuning = []

    model.eval()    
    for batch in tuning_dataloader_VALID:
        #preds_post_fine_tuning[j], loss_post_fine_tuning[j] = evaluation(batch, model, criterion, device)
        preds, loss = evaluation(batch, model, criterion, device)
        preds_post_fine_tuning.extend(preds.detach().cpu().numpy())
        loss_post_fine_tuning.extend([loss.item()]*len(preds))

    preds_post_fine_tuning = np.array(preds_post_fine_tuning)
    loss_post_fine_tuning = np.array(loss_post_fine_tuning)
    #Desnormalización
    """
    pred_pre_SBP_norm = desnormalizar_zscore(preds_pre_fine_tuning[:, 0], SBP_MEAN, SBP_STD)
    pred_pre_DBP_norm = desnormalizar_zscore(preds_pre_fine_tuning[:, 1], DBP_MEAN, DBP_STD)
    
    pred_post_SBP_norm = desnormalizar_zscore(preds_post_fine_tuning[:, 0], SBP_MEAN, SBP_STD)
    pred_post_DBP_norm = desnormalizar_zscore(preds_post_fine_tuning[:, 1], DBP_MEAN, DBP_STD)

    true_SBP_norm = desnormalizar_zscore(labels[:, 0], SBP_MEAN, SBP_STD)
    true_DBP_norm = desnormalizar_zscore(labels[:, 1], DBP_MEAN, DBP_STD)
    """
    pred_pre_flat = preds_pre_fine_tuning.reshape(-1, 2)
    pred_post_flat = preds_post_fine_tuning.reshape(-1, 2)
    labels_flat = np.array([l.squeeze().cpu().numpy() for l in tuning_dataloader_VALID.dataset.labels]).reshape(-1, 2)

    pred_pre_SBP_norm = desnormalizar_zscore(pred_pre_flat[:,0], SBP_MEAN, SBP_STD)
    pred_pre_DBP_norm = desnormalizar_zscore(pred_pre_flat[:,1], DBP_MEAN, DBP_STD)
    pred_post_SBP_norm = desnormalizar_zscore(pred_post_flat[:,0], SBP_MEAN, SBP_STD)
    pred_post_DBP_norm = desnormalizar_zscore(pred_post_flat[:,1], DBP_MEAN, DBP_STD)
    true_SBP_norm = desnormalizar_zscore(labels_flat[:,0], SBP_MEAN, SBP_STD)
    true_DBP_norm = desnormalizar_zscore(labels_flat[:,1], DBP_MEAN, DBP_STD)

    fig, ax = plt.subplots(figsize=(7, 4), tight_layout=True)
    ax.plot(loss_post_fine_tuning, label='Loss post fine tuning')
    ax.set_xlabel('number of sample')
    ax.set_ylabel('Loss')
    ax.legend()
    plt.savefig('metalearning/loss_post_fine_tuning.png')
    plt.show()

    # --- 1. Real vs Predicho ---
    fig, axs = plt.subplots(2, 2, figsize=(12, 5))
    axs[0][0].scatter(true_SBP_norm, pred_pre_SBP_norm, alpha=0.5)
    axs[0][0].plot([true_SBP_norm.min(), true_SBP_norm.max()],
            [true_SBP_norm.min(), true_SBP_norm.max()], 'r--')
    axs[0][0].set_title(f"SBP - Antes del Fine-Tuning")
    axs[0][0].set_xlabel("Valor verdadero")
    axs[0][0].set_ylabel("Predicción")
    axs[0][0].grid(True)

    axs[0][1].scatter(true_DBP_norm, pred_pre_DBP_norm, alpha=0.5)
    axs[0][1].plot([true_DBP_norm.min(), true_DBP_norm.max()],
            [true_DBP_norm.min(), true_DBP_norm.max()], 'r--')
    axs[0][1].set_title(f"DBP - Antes del Fine-Tuning")
    axs[0][1].set_xlabel("Valor verdadero")
    axs[0][1].set_ylabel("Predicción")
    axs[0][1].grid(True)

    axs[1][0].scatter(true_SBP_norm, pred_post_SBP_norm, alpha=0.5)
    axs[1][0].plot([true_SBP_norm.min(), true_SBP_norm.max()],
           [true_SBP_norm.min(), true_SBP_norm.max()], 'r--')
    axs[1][0].set_title(f"SBP - Despues del Fine-Tuning")
    axs[1][0].set_xlabel("Valor verdadero")
    axs[1][0].set_ylabel("Predicción")
    axs[1][0].grid(True)
    
    axs[1][1].scatter(true_DBP_norm, pred_post_DBP_norm, alpha=0.5)
    axs[1][1].plot([true_DBP_norm.min(), true_DBP_norm.max()],
           [true_DBP_norm.min(), true_DBP_norm.max()], 'r--')
    axs[1][1].set_title(f"DBP - Despues del Fine-Tuning")
    axs[1][1].set_xlabel("Valor verdadero")
    axs[1][1].set_ylabel("Predicción")
    axs[1][1].grid(True)

    plt.tight_layout()
    plt.show()

    # --- 2. Residuos ---
    residuals_pre_SBP = pred_pre_SBP_norm - true_SBP_norm
    residuals_pre_DBP = pred_pre_DBP_norm - true_DBP_norm
    residuals_post_SBP = pred_post_SBP_norm - true_SBP_norm
    residuals_post_DBP = pred_post_DBP_norm - true_DBP_norm

    fig, axs = plt.subplots(2, 2, figsize=(12, 5))
    axs[0][0].scatter(true_SBP_norm, residuals_pre_SBP, alpha=0.5)
    axs[0][0].axhline(0, color='red', linestyle='--')
    axs[0][0].set_title("Antes - Residuos - SBP")
    axs[0][0].set_xlabel("Valor verdadero")
    axs[0][0].set_ylabel("Error (Pred - Real)")
    axs[0][0].grid(True)

    axs[0][1].scatter(true_DBP_norm, residuals_pre_DBP, alpha=0.5)
    axs[0][1].axhline(0, color='red', linestyle='--')
    axs[0][1].set_title("Antes - Residuos - DBP")
    axs[0][1].set_xlabel("Valor verdadero")
    axs[0][1].set_ylabel("Error (Pred - Real)")
    axs[0][1].grid(True)
    
    axs[1][0].scatter(true_SBP_norm, residuals_post_SBP, alpha=0.5)
    axs[1][0].axhline(0, color='red', linestyle='--')
    axs[1][0].set_title("Después - Residuos -SBP")
    axs[1][0].set_xlabel("Valor verdadero")
    axs[1][0].set_ylabel("Error (Pred - Real)")
    axs[1][0].grid(True)

    axs[1][1].scatter(true_DBP_norm, residuals_post_DBP, alpha=0.5)
    axs[1][1].axhline(0, color='red', linestyle='--')
    axs[1][1].set_title("Después - Residuos - DBP") 
    axs[1][1].set_xlabel("Valor verdadero")
    axs[1][1].set_ylabel("Error (Pred - Real)")
    axs[1][1].grid(True)

    plt.tight_layout()
    plt.show()

    # --- 3. Histograma de errores ---
    fig, axs = plt.subplots(2, 2, figsize=(12, 5))
    axs[0][0].hist(residuals_pre_SBP, bins=50, edgecolor='black')
    axs[0][0].set_title("Antes - Histograma de errores -SBP")
    axs[0][0].set_xlabel("Error (Pred - Real)")
    axs[0][0].set_ylabel("Frecuencia")
    axs[0][0].grid(True)

    axs[0][1].hist(residuals_pre_DBP, bins=50, edgecolor='black')
    axs[0][1].set_title("Antes - Histograma de errores -DBP")
    axs[0][1].set_xlabel("Error (Pred - Real)")
    axs[0][1].set_ylabel("Frecuencia")
    axs[0][1].grid(True)

    axs[1][0].hist(residuals_post_SBP, bins=50, edgecolor='black')
    axs[1][0].set_title("Después - Histograma de errores -SBP")
    axs[1][0].set_xlabel("Error (Pred - Real)")
    axs[1][0].set_ylabel("Frecuencia")
    axs[1][0].grid(True)

    axs[1][1].hist(residuals_post_DBP, bins=50, edgecolor='black')
    axs[1][1].set_title("Después - Histograma de errores -DBP")
    axs[1][1].set_xlabel("Error (Pred - Real)")
    axs[1][1].set_ylabel("Frecuencia")
    axs[1][1].grid(True)

    plt.tight_layout()
    plt.show()
  
if __name__ == '__main__':
    main()
