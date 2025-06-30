from MetaDataset import TaskDataset
#from Clase_UCIDataset import UCIDataset
from Modelos.Modelo_conv import Modelo_Convolucional
import numpy as np
from torch import nn, optim
import matplotlib.pyplot as plt
#from Modelos.InceptionTime import InceptionTime
import torch.utils.data as data
import torch
from Singlepatientdataset import SinglePatientDataset


def main(shots=5, num_tasks=3000, valid_batch=20):
    data_dir = 'data_UCI/dataset_completo.pt'
    all_IDs = np.arange(0, num_tasks)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    #Dataset y dataloader
    tasksets = TaskDataset(all_IDs,data_dir=data_dir, num_shots=shots)
    finetuningset=SinglePatientDataset(tasksets, patient_id=0)  
    finetuningdataloader= torch.utils.data.Dataloader(finetuningset)
    dataloader = torch.utils.data.DataLoader(subset, batch_size=valid_batch, shuffle=True, drop_last=True)
    subset = torch.utils.data.Subset(tasksets, indices=list(range(20)))

    #Carga de modelo metaentrenado 
    model=Modelo_Convolucional(in_channels=2,out_channels=2, long_signal=250)
    path_model='best_meta_model.pt'
    checkpoint = torch.load(path_model, map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint['model_state_dict'])


    #Evaluación modelo previo a fine tuning
    model = model.to(device)  # Mueve el modelo a la GPU

    #Fine tunning N-way K-shot
    
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = torch.nn.MSELoss()  
    def train(batch):
        optimizer.zero_grad(batch) # Reinicia los gradientes
        data, labels = batch # Obtiene los datos y etiquetas
        data, labels = data.to(device), labels.to(device) # Mueve los datos y etiquetas a la GPU
        preds = model.forward(data) # Realiza la predicción
        loss = criterion(preds, labels) # Calcula la pérdida
        loss.backward() # Calcula los gradientes mediante backpropagation
        optimizer.step() # Actualiza los parámetros del modelo
        return loss.item() # Devuelve la pérdida
    
    model.train()
    for batch in training_generator:
        train_loss += train(batch)
        


    #Evaluación de modelo posterior al fine tuning


