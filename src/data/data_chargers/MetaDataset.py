import torch
import numpy as np
import torch.utils.data as data
from collections import defaultdict

class TaskDataset(data.Dataset):
    """
    Dataset episódico para Meta-Learning (Few-Shot) intra-paciente.
    
    Construye tareas seleccionando un bloque de muestras de soporte (Support) 
    y un bloque de muestras de consulta (Query) del mismo paciente, asegurando 
    una brecha temporal (gap) entre ambos para evaluar la verdadera capacidad 
    de adaptación del modelo ante cambios de presión arterial a lo largo del tiempo.
    
    Args:
        list_IDs (list): Lista de IDs de pacientes habilitados para el muestreo.
        base_dataset (torch.utils.data.Dataset): Dataset original (ej. UCIDataset).
        num_shots (int): Cantidad de muestras para soporte y cantidad para consulta.
        gap (int): Separación mínima (en índices) entre el soporte y la consulta.
    """
    def __init__(self, list_IDs, base_dataset=None, num_shots=5, gap=50):
        
        self.num_shots = num_shots
        self.total_samples_output = 2 * num_shots
        self.list_IDs = list_IDs
        self.gap = gap  
        
        if base_dataset is None:
            raise ValueError("Se debe proporcionar un dataset base como UCIDataset")
        
        self.base_dataset = base_dataset
        
        self.patient_to_indices = defaultdict(list)
        
        #Este bucle llama a __getitem__ para cada índice del dataset base.
        for i in range(len(base_dataset)):
            _, _, pid, _ = base_dataset[i]
            pid = int(pid)
            if pid in self.list_IDs:
                self.patient_to_indices[pid].append(i)

        # Garantiza el orden cronológico estricto de los índices
        for pid in self.patient_to_indices:
            self.patient_to_indices[pid].sort()

        print(f"Señales agrupadas. Total pacientes únicos: {len(self.patient_to_indices)}")
        
        # Tamaño mínimo requerido: [Support] + [Gap] + [Query]
        self.min_required = (2 * self.num_shots) + self.gap
        
        # Filtra pacientes que no tienen suficientes datos temporales
        self.valid_IDs = [pid for pid in self.list_IDs if len(self.patient_to_indices[pid]) >= self.min_required]
        print(f"{len(self.valid_IDs)} pacientes tienen al menos {self.min_required} señales (Soporte + Gap {gap} + Consulta)")

    def __len__(self):
        """Devuelve la cantidad de tareas (pacientes válidos) disponibles."""
        return len(self.valid_IDs) 
    
    def __getitem__(self, index):
        """
        Genera un episodio (Support + Query) para un paciente específico.
        
        Returns:
            tuple: (xs, ys) tensores concatenados de tamaño (2 * num_shots, ...).
        """
        pid = self.valid_IDs[index]
        all_indices = self.patient_to_indices[pid]
        
        # Calculo punto de inicio máximo posible
        max_start = len(all_indices) - (2 * self.num_shots) - self.gap
        
        if max_start > 0:
            # Elección un punto de inicio aleatorio
            start_idx = np.random.randint(0, max_start)
            
            support_idxs = all_indices[start_idx : start_idx + self.num_shots]
            
            query_start = start_idx + self.num_shots + self.gap
            query_idxs = all_indices[query_start : query_start + self.num_shots]
            
            indices = np.concatenate([support_idxs, query_idxs])
            
        else:
            # Selección aleatoria preservando el orden temporal
            indices = np.random.choice(all_indices, self.total_samples_output, replace=False)
            indices.sort() 

        # Cargar los tensores reales usando los índices seleccionados
        xs = torch.stack([self.base_dataset[i][0] for i in indices])    
        ys = torch.stack([self.base_dataset[i][1] for i in indices])   

        return xs, ys