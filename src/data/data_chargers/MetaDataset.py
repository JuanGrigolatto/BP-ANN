import torch.utils.data as data
import torch
import numpy as np
from collections import defaultdict

class TaskDataset(data.Dataset):
    def __init__(self, list_IDs, base_dataset=None, num_shots=5):
        self.num_shots = num_shots
        self.total_samples = 2 * num_shots
        self.list_IDs = list_IDs
        
        if base_dataset is None:
            raise ValueError("Se debe proporcionar un dataset base como UCIDataset")
        
        self.base_dataset = base_dataset
        
        self.patient_to_indices = defaultdict(list)
        
        for i in range(len(base_dataset)):
            _, _, pid, _ = base_dataset[i]
            pid = int(pid)
            if pid in self.list_IDs:
                self.patient_to_indices[pid].append(i)

        print(f"Señales agrupadas. Total pacientes únicos: {len(self.patient_to_indices)}")
        # Uso pacientes con cantidad suficiente de señales para 2* self.num_shots
        self.valid_IDs = [pid for pid in self.list_IDs if len(self.patient_to_indices[pid]) >= self.total_samples]
        print(f"{len(self.valid_IDs)} pacientes tienen al menos {self.total_samples} señales")


    def __len__(self):
        return len(self.valid_IDs) #Devuelve el número de pacientes en el dataset (Tareas)
    

    def __getitem__(self, index):
        pid = self.valid_IDs[index]

        indices = np.random.choice(self.patient_to_indices[pid], self.total_samples, replace=False)

        xs = torch.stack([self.base_dataset[i][0] for i in indices])    # (2*num_shots, 2, 500)
        ys = torch.stack([self.base_dataset[i][1] for i in indices])   # (2*num_shots, 2)

        return xs, ys