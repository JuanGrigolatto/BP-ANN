from MetaDataset import TaskDataset
#from Clase_UCIDataset import UCIDataset
from Modelos.Modelo_conv import Modelo_Convolucional
import numpy as np
import learn2learn as l2l
from torch import nn, optim
import matplotlib.pyplot as plt
#from Modelos.InceptionTime import InceptionTime
import torch.utils.data as data
import os

def main(shots=5, num_tasks=3000 ,tasks_per_batch=16, adapt_lr=0.01, meta_lr=0.001, adapt_steps=5,):
    data_dir = 'data_UCI/dataset_completo.pt'
    all_IDs = np.arange(0, 3000)    # IDs de pacientes, suponiendo que hay 3000 pacientes   
        
    # Crear el dataset y dataloader
    tasksets = TaskDataset(all_IDs,data_dir=data_dir, num_shots=shots)
    dataloader = data.DataLoader(tasksets, batch_size=tasks_per_batch, shuffle=True, drop_last=True)
    
    # Modelo y MAML
    #model= InceptionTime(c_in=2, c_out=2, seq_len=None, n_filters=32, nb_filters=None)
    model = Modelo_Convolucional(in_channels=2, out_channels=2, long_signal=250)
    maml = l2l.algorithms.MAML(model, lr=adapt_lr, first_order=False, allow_unused=True)
    opt = optim.Adam(maml.parameters(), meta_lr)
    lossfn = nn.MSELoss(reduction='mean')

    # Entrenamiento del modelo
    # Outer loop: iteraciones sobre lotes de tareas
    for iter, batch in enumerate(dataloader):
        meta_train_loss = 0.0
        x_batch, y_batch = batch # batch: lote de tareas, x_batch: entradas, y_batch: etiquetas
        effective_batch_size = x_batch.size(0)
        # Bucle sobre las tareas en el batch
        for i in range(effective_batch_size):
            learner = maml.clone()

            x_task = x_batch[i]
            y_task = y_batch[i]

            x_support = x_task[:shots]
            y_support = y_task[:shots]
            x_query   = x_task[shots:]
            y_query   = y_task[shots:]

            #train_inputs, train_targets = batch[0][i], batch[1][i]
            #x_support, y_support = train_inputs[::2], train_targets[::2]
            #x_query, y_query = train_inputs[1::2], train_targets[1::2]

            # Inner loop: adaptación del modelo a la tarea
            for _ in range(adapt_steps): #adapt_steps: cantidad de pasos de adaptación a la tarea
                support_preds = learner(x_support)
                support_loss = lossfn(support_preds, y_support)
                learner.adapt(support_loss)

            # Evaluación del modelo adaptado en el conjunto de consulta
            query_preds = learner(x_query)
            query_loss = lossfn(query_preds, y_query)
            meta_train_loss += query_loss

        meta_train_loss = meta_train_loss / effective_batch_size

        if iter % 200 == 0:
            print('Iteration:', iter, 'Meta Train Loss', meta_train_loss.item())

        opt.zero_grad()
        meta_train_loss.backward()
        opt.step()
        

if __name__ == '__main__':
    main()        