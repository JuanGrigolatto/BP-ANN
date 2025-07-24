import torch.utils
from Clase_UCIDataset import UCIDataset
from MetaDataset import TaskDataset
#from Clase_UCIDataset import UCIDataset
from Modelos.Modelo_conv import Modelo_Convolucional
import numpy as np
from torch import nn, optim
import matplotlib.pyplot as plt
#from Modelos.InceptionTime import InceptionTime
import torch.utils.data as data
import torch
from Tuningndataset import TuningNDataset
import random
from sklearn.metrics import r2_score

def desnormalizar_zscore(norm_array, media, std):
    return norm_array * std + media

def desnormalizar_minmax(norm_array, min_val, max_val):
    return norm_array * (max_val - min_val) + min_val

def main(n_shots=10, num_tasks= 10000):
    SBP_MEAN = 134.02
    DBP_MEAN = 63.47
    SBP_STD = 22.75
    DBP_STD = 23.69
    
    test_data = torch.load('data_UCI/few_shot_patient_data.pt')
    test_patient_ids = test_data['test_patient_ids']
    
    """
    data_dir = 'data_UCI/dataset_completo_prueba.pt'
    all_IDs = np.arange(0, num_tasks)
    """
    
    data_paths = [
        'data_UCI/dataset_parte_1.pt',
        'data_UCI/dataset_parte_2.pt',
        'data_UCI/dataset_parte_3.pt',
        'data_UCI/dataset_parte_4.pt'
    ]

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
    unique_patients = merged_data['patient_ids'].unique().tolist()

    if len(unique_patients) >= num_tasks:
        list_IDs = random.sample(unique_patients, num_tasks)
    else:
        raise ValueError(f"Solo hay {len(unique_patients)} pacientes únicos, se solicita {num_tasks}")
    """

   
    taskset = TaskDataset(list_IDs=test_patient_ids, data_dict=merged_data, num_shots=n_shots)
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
    model=Modelo_Convolucional(in_channels=2,out_channels=2, long_signal=250)
    path_model='best_meta_model.pt'
    checkpoint = torch.load(path_model, map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Criterio de pérdida
    criterion = torch.nn.MSELoss()

    #Evaluación modelo previo a fine tuning
    model = model.to(device)  # Mueve el modelo a la GPU

    def evaluation(batch):
        with torch.no_grad():
            data, labels = batch
            data, labels = data.to(device), labels.to(device)
            preds = model.forward(data)
            loss = criterion(preds, labels)
        return preds.squeeze().cpu().numpy(), loss.item()
        
   
    loss_pre_fine_tuning = np.zeros(shape=len(tuning_dataloader_VALID))
    preds_pre_fine_tuning = np.zeros(shape=(len(tuning_dataloader_VALID), 2))
    model.eval()    
    for k, batch in enumerate(tuning_dataloader_VALID):
        preds_pre_fine_tuning[k] ,loss_pre_fine_tuning[k] = evaluation(batch)
        
    
    labels = np.array([l.squeeze().cpu().numpy() for l in tuning_dataloader_VALID.dataset.labels])

    fig, ax = plt.subplots(figsize=(7, 4), tight_layout=True)
    ax.plot(loss_pre_fine_tuning, label='Loss pre fine tuning')
    ax.set_xlabel('number of sample')
    ax.set_ylabel('Loss')
    ax.legend()
    plt.savefig('loss_pre_fine_tuning.png')
    plt.show()

    #Fine tunning N-way K-shot    
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)  
    def tuning(sample):
        optimizer.zero_grad(sample) # Reinicia los gradientes
        data, labels = sample # Obtiene los datos y etiquetas
        data, labels = data.to(device), labels.to(device) # Mueve los datos y etiquetas a la GPU
        preds = model.forward(data) # Realiza la predicción
        loss = criterion(preds, labels) # Calcula la pérdida
        loss.backward() # Calcula los gradientes mediante backpropagation
        optimizer.step() # Actualiza los parámetros del modelo
        return loss.item() # Devuelve la pérdida
    

    model.train()
    tuning_loss = np.zeros(shape=n_shots)
    for i, sample in enumerate(tuning_dataloader_TRAIN):
        tuning_loss[i] = tuning(sample)
    
    torch.save({'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': tuning_loss[n_shots-1]}, 
                    'tuning_model.pt')

    fig, ax = plt.subplots(figsize=(7, 4), tight_layout=True)
    ax.plot(tuning_loss, label='Fine Tuning for 5 shots')
    ax.set_xlabel('number of shot')
    ax.set_ylabel('Loss')
    ax.legend()
    plt.savefig('loss_curve_fine_tuning.png')
    plt.show()

    
    #Evaluación de modelo posterior al fine tuning

    loss_post_fine_tuning = np.zeros(shape=len(tuning_dataloader_VALID))
    preds_post_fine_tuning = np.zeros(shape=(len(tuning_dataloader_VALID),2))
    model.eval()    
    for j, batch in enumerate(tuning_dataloader_VALID):
        preds_post_fine_tuning[j], loss_post_fine_tuning[j] = evaluation(batch)

    #Desnormalización
    pred_pre_SBP_norm = desnormalizar_zscore(preds_pre_fine_tuning[:, 0], SBP_MEAN, SBP_STD)
    pred_pre_DBP_norm = desnormalizar_zscore(preds_pre_fine_tuning[:, 1], DBP_MEAN, DBP_STD)
    
    pred_post_SBP_norm = desnormalizar_zscore(preds_post_fine_tuning[:, 0], SBP_MEAN, SBP_STD)
    pred_post_DBP_norm = desnormalizar_zscore(preds_post_fine_tuning[:, 1], DBP_MEAN, DBP_STD)

    true_SBP_norm = desnormalizar_zscore(labels[:, 0], SBP_MEAN, SBP_STD)
    true_DBP_norm = desnormalizar_zscore(labels[:, 1], DBP_MEAN, DBP_STD)

    fig, ax = plt.subplots(figsize=(7, 4), tight_layout=True)
    ax.plot(loss_post_fine_tuning, label='Loss post fine tuning')
    ax.set_xlabel('number of sample')
    ax.set_ylabel('Loss')
    ax.legend()
    plt.savefig('loss_post_fine_tuning.png')
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
