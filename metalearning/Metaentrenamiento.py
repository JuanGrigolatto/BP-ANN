from src.data.data_chargers.MetaDataset import TaskDataset
#from src.models.Modelo_conv import Modelo_Convolucional
#from src.models.InceptionTime import InceptionTime
from src.models.ConvolucionalV1 import Modelo_ConvolucionalV1
#from src.models.ConvolucionalV2 import Modelo_ConvolucionalV2
import numpy as np
import learn2learn as l2l
from torch import nn, optim
import matplotlib.pyplot as plt
import torch.utils.data as data
from tqdm.auto import tqdm 
import torch
import random

def main(shots=10, num_tasks=10000 ,tasks_per_batch=16, adapt_lr=0.01, meta_lr=0.001, adapt_steps=5, seed=42):
    """
    data_dir = 'data_UCI/dataset_completo.pt'
    all_IDs = np.arange(0, num_tasks)    # IDs de pacientes, suponiendo que hay 3000 pacientes   
        
    # Crear el dataset y dataloader
    tasksets = TaskDataset(all_IDs,data_dir=data_dir, num_shots=shots)
    dataloader = data.DataLoader(tasksets, batch_size=tasks_per_batch, shuffle=True, drop_last=True)
    """
    data_paths = [
        'data/processed/data_UCI/dataset_parte_1.pt',
        'data/processed/data_UCI/dataset_parte_2.pt',
        'data/processed/data_UCI/dataset_parte_3.pt',
        'data/processed/data_UCI/dataset_parte_4.pt'
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

    random.seed(seed)
    
    unique_patients = merged_data['patient_ids'].unique().tolist()
    random.shuffle(unique_patients)
    
    test_patients = unique_patients[:20]       # 20 pacientes para few-shot
    train_patients = unique_patients[20:]      # Resto para metaentrenamiento

    # Guardar lista de pacientes para evaluación few-shot
    torch.save({'test_patient_ids': test_patients}, 'data/processed/data_UCI/few_shot_patient_data.pt')

    if len(unique_patients) >= num_tasks:
        list_IDs = random.sample(train_patients, num_tasks)
    else:
        raise ValueError(f"Solo hay {len(train_patients)} pacientes únicos, se solicita {num_tasks}")
    
    tasksets = TaskDataset(list_IDs=list_IDs, data_dict=merged_data, num_shots=shots)
    dataloader = data.DataLoader(tasksets, batch_size=tasks_per_batch, shuffle=True, drop_last=True)

    # Modelo y MAML
    #model= InceptionTime(c_in=2, c_out=2, seq_len=None, n_filters=32, nb_filters=None)
    #model = Modelo_Convolucional(in_channels=2, out_channels=2, long_signal=250)
    model=Modelo_ConvolucionalV1(in_channels=2,out_channels=2, long_signal=1250)
    maml = l2l.algorithms.MAML(model, lr=adapt_lr, first_order=False, allow_unused=True)
    opt = optim.Adam(maml.parameters(), meta_lr)
    lossfn = nn.MSELoss(reduction='mean')

    running_meta_loss=np.zeros(shape=round((len(tasksets.valid_IDs)/tasks_per_batch)))
    best_valid_loss=float('inf') 
    # Entrenamiento del modelo
    # Outer loop: iteraciones sobre lotes de tareas
    for iter, batch in enumerate(tqdm(dataloader)):
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
        if meta_train_loss < best_valid_loss:
            best_valid_loss = meta_train_loss
            torch.save({'iteration': iter,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': opt.state_dict(),
                        'meta_loss': meta_train_loss}, 
                    'models/best_meta_models/best_meta_model_v1.pt')
            
        running_meta_loss[iter] = meta_train_loss.detach().numpy()

        #print("Iteration: ",  iter, 'Meta Train Loss: ', meta_train_loss.item())

        #if iter % 200 == 0:
        #    print('Iteration&: ', iter, 'Meta Train Loss&: ', meta_train_loss.item())

        opt.zero_grad()
        meta_train_loss.backward()
        opt.step()
    
    fig, ax = plt.subplots(figsize=(7, 4), tight_layout=True)
    ax.plot(running_meta_loss, label='Meta entrenamiento')
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Loss')
    ax.legend()
    plt.savefig('meta_loss_curve.png')
    plt.show()

if __name__ == '__main__':
    main()        