from src.data.data_chargers.MetaDataset import TaskDataset
from src.data.data_chargers.PatientWiseSet import PatientWiseDataset
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
from src.data.data_chargers.Clase_UCIDataset import UCIDataset

def main(num_tasks=10000 ,tasks_per_batch=4, adapt_lr=0.01, meta_lr=0.005, k_adapt_steps=10, seed=42, num_epochs=100, N_patient_group = 4, p_support = 5, q_query= 10, base_dataset=None, selected_patients=None):

    if base_dataset is None:
        data_paths = [
            'data/processed/data_UCI/dataset_parte_1_por_picos.pt',
            'data/processed/data_UCI/dataset_parte_2_por_picos.pt',
            'data/processed/data_UCI/dataset_parte_3_por_picos.pt',
            'data/processed/data_UCI/dataset_parte_4_por_picos.pt',
        ]
        dataset_completo = UCIDataset(data_paths)
    else:
        dataset_completo = base_dataset
    
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    """
    all_pids = [dataset_completo[i][2].item() for i in range(len(dataset_completo))]
    all_patients = list(set(all_pids))
    # Seleccionar 200 pacientes una sola vez 
    if selected_patients is None:
        if len(all_patients) < num_tasks:
            patient_ids = all_patients
        else:
            patient_ids = random.sample(all_patients, num_tasks)
        print(f"Usando {len(patient_ids)} pacientes para entrenamiento.")
    else:
        patient_ids = selected_patients
        print(f"Usando pacientes precargados: {len(patient_ids)}")

    """
    
    all_pids = torch.tensor([dataset_completo[i][2] for i in range(len(dataset_completo))])
    unique_patients = all_pids.unique().tolist()
    random.shuffle(unique_patients)

    # Contar cuántas muestras tiene cada paciente
    patient_to_indices = {pid: [] for pid in unique_patients}
    for i in range(len(dataset_completo)):
        _, _, pid, _ = dataset_completo[i]
        patient_to_indices[int(pid)].append(i)

    # Filtrar pacientes con suficientes muestras
    min_samples = p_support + q_query
    valid_IDs_global = [pid for pid, idxs in patient_to_indices.items() if len(idxs) >= min_samples]

    print(f"Pacientes totales: {len(unique_patients)}")
    print(f"Pacientes válidos (>= {min_samples} muestras): {len(valid_IDs_global)}")
    
    num_patients = len(valid_IDs_global)
    num_valid = int(0.1 * num_patients)
    valid_patients = valid_IDs_global[:num_valid]
    train_patients = valid_IDs_global[num_valid:]

    print(f"Pacientes totales: {num_patients}")
    print(f"Entrenamiento: {len(train_patients)}  |  Validación: {len(valid_patients)}")
    # Guardar lista de pacientes para evaluación few-shot
    torch.save({'test_patient_ids': valid_patients}, 'data/processed/data_UCI/few_shot_patient_data.pt')

    #list_IDs = train_patients  # Usar todos los pacientes de entrenamiento
    """
    # Limitar el número de tareas (pacientes) de entrenamiento
    if len(train_patients) > num_tasks:
        list_IDs = random.sample(train_patients, num_tasks)
        print(f"Usando solo {num_tasks} pacientes para entrenamiento.")
    else:
        list_IDs = train_patients
        print(f"Advertencia: solo {len(train_patients)} pacientes disponibles (menos que num_tasks={num_tasks})")    
    """
    list_IDs = train_patients
    print(f"Usando todos los {len(list_IDs)} pacientes disponibles para metaentrenamiento.")
    tasksets = PatientWiseDataset(list_IDs=list_IDs, base_dataset=dataset_completo, N_patients = N_patient_group, p_support=p_support, q_query=q_query)
    
    #tasksets = PatientWiseDataset(list_IDs=patient_ids, base_dataset=dataset_completo, N_patients = N_patient_group, p_support=p_support, q_query=q_query)
    dataloader = data.DataLoader(tasksets, batch_size=tasks_per_batch, shuffle=True, drop_last=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model=Modelo_ConvolucionalV1(in_channels=2,out_channels=2, long_signal=500).to(device)
    maml = l2l.algorithms.MAML(model, lr=adapt_lr, first_order=True, allow_unused=True)
    opt = optim.Adam(maml.parameters(), meta_lr)
    lossfn = nn.MSELoss(reduction='mean')

    running_meta_loss=[]# Perdida por iteración
    best_valid_loss=float('inf') 
    epoch_losses = [] #Perdida promedio por época

    for epoch in range(num_epochs):  
        print(f"\n--- Época {epoch+1}/{num_epochs} ---")
        epoch_loss_accum = 0.0   
        num_batches = 0
        # Outer loop: iteraciones sobre lotes de tareas
        for iter, batch in enumerate(tqdm(dataloader)):
            meta_train_loss = 0.0
            x_support, y_support, x_query, y_query = [t.to(device) for t in batch]

            effective_batch_size = x_support.size(0)  # cantidad de tareas (pacientes)

            for i in range(effective_batch_size):
                learner = maml.clone()  # copia diferenciable del modelo base

                # Extraer los subconjuntos de la tarea actual
                xs_spt, ys_spt = x_support[i], y_support[i]
                xs_qry, ys_qry = x_query[i], y_query[i]

                # ----- INNER LOOP -----
                for _ in range(k_adapt_steps):  # número de pasos de adaptación por tarea
                    support_preds = learner(xs_spt)
                    support_loss  = lossfn(support_preds, ys_spt)
                    learner.adapt(support_loss)

                # ----- OUTER LOOP -----
                query_preds = learner(xs_qry)
                query_loss  = lossfn(query_preds, ys_qry)
                meta_train_loss += query_loss

            # Promediar pérdidas de todas las tareas del batch
            meta_train_loss = meta_train_loss / effective_batch_size

            # Guardar mejor modelo (opcional)
            if meta_train_loss < best_valid_loss:
                best_valid_loss = meta_train_loss
                torch.save({
                    'epoch': epoch,
                    'iteration': iter,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': opt.state_dict(),
                    'meta_loss': meta_train_loss
                }, 'models/best_meta_models/best_meta_model_patientwise_2000_epoch.pt')

            running_meta_loss.append(meta_train_loss.item())

            # Actualización de parámetros meta (outer loop)
            opt.zero_grad()
            meta_train_loss.backward()
            opt.step()

            epoch_loss_accum += meta_train_loss.item()
            num_batches += 1

        # Promediar pérdida de la época
        epoch_avg_loss = epoch_loss_accum / num_batches
        epoch_losses.append(epoch_avg_loss)
        print(f"Época {epoch+1}: Pérdida promedio = {epoch_avg_loss:.4f}")

        #Graficar la pérdida de metaentrenamiento promedio por época   
    """
    fig, ax = plt.subplots(figsize=(7, 4), tight_layout=True)
    ax.plot(epoch_losses, marker='o', label='Meta entrenamiento (promedio por época)')
    ax.set_xlabel('Época')
    ax.set_ylabel('Loss promedio')
    ax.legend()
    plt.savefig('metalearning/meta_loss_curve.png')
    plt.show()
    """
if __name__ == '__main__':
    main()        