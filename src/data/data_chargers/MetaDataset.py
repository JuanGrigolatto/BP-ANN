import torch.utils.data as data
import torch
import numpy as np
from collections import defaultdict

class TaskDataset(data.Dataset):
    #  parámetro 'gap' (brecha temporal) para que exista cambio de presión entre support y query
    def __init__(self, list_IDs, base_dataset=None, num_shots=5, gap=50):
        self.num_shots = num_shots
        self.total_samples_output = 2 * num_shots
        self.list_IDs = list_IDs
        self.gap = gap  
        
        if base_dataset is None:
            raise ValueError("Se debe proporcionar un dataset base como UCIDataset")
        
        self.base_dataset = base_dataset
        
        self.patient_to_indices = defaultdict(list)
        
        # Agrupamos los índices
        for i in range(len(base_dataset)):
            _, _, pid, _ = base_dataset[i]
            pid = int(pid)
            if pid in self.list_IDs:
                self.patient_to_indices[pid].append(i)

        # Asegurar orden temporal 
        for pid in self.patient_to_indices:
            self.patient_to_indices[pid].sort()

        print(f"Señales agrupadas. Total pacientes únicos: {len(self.patient_to_indices)}")
        
        # espacio para: [Support] + [Gap] + [Query]
        self.min_required = (2 * self.num_shots) + self.gap
        
        self.valid_IDs = [pid for pid in self.list_IDs if len(self.patient_to_indices[pid]) >= self.min_required]
        print(f"{len(self.valid_IDs)} pacientes tienen al menos {self.min_required} señales (Soporte + Gap {gap} + Consulta)")


    def __len__(self):
        return len(self.valid_IDs) 
    

    def __getitem__(self, index):
        pid = self.valid_IDs[index]
        all_indices = self.patient_to_indices[pid]
        
        # Lógica de Ventana Temporal con Gap 
        
        # Calculo punto de inicio máximo posible
        # Largo total - (bloque support + bloque gap + bloque query)
        max_start = len(all_indices) - (2 * self.num_shots) - self.gap
        
        if max_start > 0:
            # Elección un punto de inicio aleatorio en la línea de tiempo
            start_idx = np.random.randint(0, max_start)
            
            
            support_idxs = all_indices[start_idx : start_idx + self.num_shots]
            
            
            query_start = start_idx + self.num_shots + self.gap
            query_idxs = all_indices[query_start : query_start + self.num_shots]
            
            
            indices = np.concatenate([support_idxs, query_idxs])
            
        else:
            # aleatorio pero ordenado 
            indices = np.random.choice(all_indices, self.total_samples_output, replace=False)
            indices.sort() 

        # Cargar los tensores reales usando los índices seleccionados
        xs = torch.stack([self.base_dataset[i][0] for i in indices])    # (2*num_shots, 2, 500)
        ys = torch.stack([self.base_dataset[i][1] for i in indices])   # (2*num_shots, 2)

        return xs, ys