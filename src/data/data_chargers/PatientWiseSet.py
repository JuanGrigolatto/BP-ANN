import random
import torch
import numpy as np
import torch.utils.data as data
from collections import defaultdict

class PatientWiseDataset(data.Dataset):
    """
    Dataset experimental diseñado para tareas multifuente con muestreo aleatorio.
    
    NOTA DE DISEÑO EXPERIMENTAL (Ref: Jia et al., 2022): Construye tareas combinando 
    segmentos de múltiples pacientes en el conjunto de soporte y consulta para forzar 
    el aprendizaje de patrones poblacionales generales. 
    
    Args:
        list_IDs (list): Lista de IDs de pacientes habilitados.
        base_dataset (torch.utils.data.Dataset): Dataset original (ej. UCIDataset).
        N_patients (int): Cantidad de pacientes para soporte y cantidad para consulta (2N en total).
        p_support (int): Muestras extraídas por cada paciente de soporte.
        q_query (int): Muestras extraídas por cada paciente de consulta.
    """
    def __init__(self, list_IDs, base_dataset=None, N_patients=20, p_support = 5, q_query= 10):
        self.p_support = p_support
        self.q_query = q_query
        self.N = N_patients
        self.samples_per_patient = p_support + q_query
        self.list_IDs = list_IDs

        if base_dataset is None:
            raise ValueError("Se debe proporcionar un dataset base como UCIDataset")
        
        self.base_dataset = base_dataset

        self.patient_to_indices = defaultdict(list)

        # Agrupación de señales por paciente
        for i in range(len(base_dataset)):
            _, _, pid, _ = base_dataset[i]
            pid = int(pid)
            if pid in self.list_IDs:
                self.patient_to_indices[pid].append(i)
        
        print(f"Señales agrupadas. Total pacientes únicos: {len(self.patient_to_indices)}")

        # Filtro de pacientes con cantidad suficiente de señales
        self.valid_IDs = [pid for pid in self.list_IDs if len(self.patient_to_indices[pid]) >= self.samples_per_patient]
        print(f"{len(self.valid_IDs)} pacientes tienen al menos {self.samples_per_patient} señales")

    def __len__(self):
        """Devuelve el número de tareas posibles (limitado por combinaciones de 2N pacientes)."""
        return len(self.valid_IDs) // (2 * self.N)
  

    def __getitem__(self, index):
        """
        Genera un episodio extrayendo datos de soporte de N pacientes y datos 
        de consulta de OTROS N pacientes distintos (evaluación multifuente).
        """
        # Selección aleatoria de 2N pacientes válidos (N para soporte, N para consulta)
        selected_patients = random.sample(self.valid_IDs, 2 * self.N)
        support_patients = selected_patients[:self.N]
        query_patients   = selected_patients[self.N:]

        # Support set
        xs_spt, ys_spt = [], []
        for pid in support_patients:
            idxs = np.random.choice(self.patient_to_indices[pid], self.p_support, replace=False)
            for i in idxs:
                x, y, _, _ = self.base_dataset[i]
                xs_spt.append(x)
                ys_spt.append(y)

        # Query set
        xs_qry, ys_qry = [], []
        for pid in query_patients:
            idxs = np.random.choice(self.patient_to_indices[pid], self.q_query, replace=False)
            for i in idxs:
                x, y, _, _ = self.base_dataset[i]
                xs_qry.append(x)
                ys_qry.append(y)

        # Concatenar y convertir a tensores
        xs_spt = torch.stack(xs_spt)
        ys_spt = torch.stack(ys_spt)
        xs_qry = torch.stack(xs_qry)
        ys_qry = torch.stack(ys_qry)

        return xs_spt, ys_spt, xs_qry, ys_qry