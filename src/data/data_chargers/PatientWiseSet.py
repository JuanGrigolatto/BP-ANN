import torch.utils.data as data
import torch
import numpy as np
from collections import defaultdict
import random

class PatientWiseDataset(data.Dataset):
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

        for i in range(len(base_dataset)):
            _, _, pid, _ = base_dataset[i]
            pid = int(pid)
            if pid in self.list_IDs:
                self.patient_to_indices[pid].append(i)
        
        print(f"Señales agrupadas. Total pacientes únicos: {len(self.patient_to_indices)}")

        # Uso pacientes con cantidad suficiente de señales para self.samples_per_patient
        self.valid_IDs = [pid for pid in self.list_IDs if len(self.patient_to_indices[pid]) >= self.samples_per_patient]
        print(f"{len(self.valid_IDs)} pacientes tienen al menos {self.samples_per_patient} señales")

    def __len__(self):
        # Número de tareas posibles (limitado por combinaciones de pacientes)
        return len(self.valid_IDs) // (2 * self.N)
  

    def __getitem__(self, index):
        #Selección aleatoria 2N pacientes válidos
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