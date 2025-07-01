import torch.utils
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

def main(n_shots=5, num_tasks=3000, valid_batch=20):
    data_dir = 'data_UCI/dataset_completo_prueba.pt'
    all_IDs = np.arange(0, num_tasks)
    
    #Datasets y dataloaders
    taskset= TaskDataset(all_IDs,data_dir=data_dir, num_shots=n_shots)
    id_patient_for_tuning =  random.randint(0, len(taskset))
    tuningset=TuningNDataset(taskset, id_patient_for_tuning)
    #Dataloader para fine tuning mediante n_shots
    tuning_dataloader=torch.utils.data.DataLoader(tuningset, batch_size=1, shuffle=False)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
    for i, sample in enumerate(tuning_dataloader):
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

if __name__ == '__main__':
    main()
