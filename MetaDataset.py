import torch.utils.data as data
import torch
import numpy as np
from collections import defaultdict

class TaskDataset(data.Dataset):
    def __init__(self,list_IDs,data_dir='data_UCI/dataset_completo.pt', num_shots=5):
        self.data_dir = data_dir
        self.num_shots = num_shots
        self.total_samples = 2 * num_shots
        self.list_IDs = list_IDs

        data = torch.load(data_dir)
        
        all_signals= data['data']  # (2, L)
        all_labels = data['labels']   # (2,)
        self.ID_patients = data['patient_ids']  # (N,)
        print(f"Datos cargados correctamente. Muestras: {len(all_signals)}")  

        self.patient_to_signals = defaultdict(list)
        self.patient_to_labels = defaultdict(list)

        
        for i in range(len(self.ID_patients)):
            pid = int(self.ID_patients[i])
            self.patient_to_signals[pid].append(all_signals[i])
            self.patient_to_labels[pid].append(all_labels[i])

        print(f"Señales agrupadas. Total pacientes únicos: {len(self.patient_to_signals)}")
        # Uso pacientes con cantidad suficiente de señales para 2* self.num_shots
        self.valid_IDs = [pid for pid in self.list_IDs if len(self.patient_to_signals[pid]) >= self.total_samples]
        print(f"{len(self.valid_IDs)} pacientes tienen al menos {self.total_samples} señales")


    def __len__(self):
        return len(self.valid_IDs) #Devuelve el número de pacientes en el dataset (Tareas)
    

    def __getitem__(self, index):
        pid = self.valid_IDs[index]

        signals = self.patient_to_signals[pid]
        labels = self.patient_to_labels[pid]
        
        if len(signals) < self.total_samples:
            raise ValueError(f"Paciente {pid} tiene solo {len(signals)} señales (< {self.total_samples})")

        indices = np.random.choice(len(signals), self.total_samples, replace=False)

        xs = torch.stack([signals[i] for i in indices])  # (2*num_shots, 2, 250)
        ys = torch.stack([labels[i] for i in indices])   # (2*num_shots, 2)

        return xs, ys