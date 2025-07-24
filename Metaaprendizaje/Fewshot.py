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

def main(n_shots=5, num_tasks=3000, valid_batch=20):
    data_dir = 'data_UCI/dataset_completo_prueba.pt'
    all_IDs = np.arange(0, num_tasks)
    
    #Datasets y dataloaders
    #UCIset = UCIDataset(data_dir=data_dir)
    #subUCIset = torch.utils.data.Subset(UCIset, indices = [random.randint(1, 999) for _ in range(20)])
    #evaluation_generator = torch.utils.data.DataLoader(subUCIset, batch_size=1, shuffle=False)

    taskset= TaskDataset(all_IDs,data_dir=data_dir, num_shots=n_shots)
    id_patient_for_tuning =  random.randint(0, len(taskset))
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
        
    #labels = tuning_dataloader_VALID.dataset.labels
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

    fig, ax = plt.subplots(figsize=(7, 4), tight_layout=True)
    ax.plot(loss_post_fine_tuning, label='Loss post fine tuning')
    ax.set_xlabel('number of sample')
    ax.set_ylabel('Loss')
    ax.legend()
    plt.savefig('loss_post_fine_tuning.png')
    plt.show()

    # --- 1. Real vs Predicho ---
    fig, axs = plt.subplots(2, 2, figsize=(12, 5))
    axs[0][0].scatter(labels[:, 0], preds_pre_fine_tuning[:, 0], alpha=0.5)
    axs[0][0].plot([labels[:, 0].min(), labels[:, 0].max()],
            [labels[:, 0].min(), labels[:, 0].max()], 'r--')
    axs[0][0].set_title(f"SBP - Antes del Fine-Tuning")
    axs[0][0].set_xlabel("Valor verdadero")
    axs[0][0].set_ylabel("Predicción")
    axs[0][0].grid(True)

    axs[0][1].scatter(labels[:, 1], preds_pre_fine_tuning[:, 1], alpha=0.5)
    axs[0][1].plot([labels[:, 1].min(), labels[:, 1].max()],
            [labels[:, 1].min(), labels[:, 1].max()], 'r--')
    axs[0][1].set_title(f"DBP - Antes del Fine-Tuning")
    axs[0][1].set_xlabel("Valor verdadero")
    axs[0][1].set_ylabel("Predicción")
    axs[0][1].grid(True)

    axs[1][0].scatter(labels[:, 0], preds_post_fine_tuning[:, 0], alpha=0.5)
    axs[1][0].plot([labels[:, 0].min(), labels[:, 0].max()],
           [labels[:, 0].min(), labels[:, 0].max()], 'r--')
    axs[1][0].set_title(f"SBP - Despues del Fine-Tuning")
    axs[1][0].set_xlabel("Valor verdadero")
    axs[1][0].set_ylabel("Predicción")
    axs[1][0].grid(True)
    
    axs[1][1].scatter(labels[:, 1], preds_post_fine_tuning[:, 1], alpha=0.5)
    axs[1][1].plot([labels[:, 1].min(), labels[:, 1].max()],
           [labels[:, 1].min(), labels[:, 1].max()], 'r--')
    axs[1][1].set_title(f"DBP - Despues del Fine-Tuning")
    axs[1][1].set_xlabel("Valor verdadero")
    axs[1][1].set_ylabel("Predicción")
    axs[1][1].grid(True)

    plt.tight_layout()
    plt.show()

    # --- 2. Residuos ---
    residuals_pre_SBP = preds_pre_fine_tuning[:, 0] - labels[:, 0]
    residuals_pre_DBP = preds_pre_fine_tuning[:, 1] - labels[:, 1]
    residuals_post_SBP = preds_post_fine_tuning[:, 0] - labels[:, 0]
    residuals_post_DBP = preds_post_fine_tuning[:, 1] - labels[:, 1]

    fig, axs = plt.subplots(2, 2, figsize=(12, 5))
    axs[0][0].scatter(labels[:, 0], residuals_pre_SBP, alpha=0.5)
    axs[0][0].axhline(0, color='red', linestyle='--')
    axs[0][0].set_title("Antes - Residuos - SBP")
    axs[0][0].set_xlabel("Valor verdadero")
    axs[0][0].set_ylabel("Error (Pred - Real)")
    axs[0][0].grid(True)

    axs[0][1].scatter(labels[:, 1], residuals_pre_DBP, alpha=0.5)
    axs[0][1].axhline(0, color='red', linestyle='--')
    axs[0][1].set_title("Antes - Residuos - DBP")
    axs[0][1].set_xlabel("Valor verdadero")
    axs[0][1].set_ylabel("Error (Pred - Real)")
    axs[0][1].grid(True)
    
    axs[1][0].scatter(labels[:, 0], residuals_post_SBP, alpha=0.5)
    axs[1][0].axhline(0, color='red', linestyle='--')
    axs[1][0].set_title("Después - Residuos -SBP")
    axs[1][0].set_xlabel("Valor verdadero")
    axs[1][0].set_ylabel("Error (Pred - Real)")
    axs[1][0].grid(True)

    axs[1][1].scatter(labels[:, 1], residuals_post_DBP, alpha=0.5)
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
