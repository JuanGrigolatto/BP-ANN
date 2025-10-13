import torch.utils.data as data
import numpy as np
from src.data.data_chargers.MetaDataset import TaskDataset

class TuningNDataset(data.Dataset):
    #Antes no estaba el import de TaskDataset y ponía 'TaskDataset' entre comillas
    def __init__(self, task_dataset: TaskDataset, patient_id: int, n_shots=5, validation=False):
        

        indices = task_dataset.patient_to_indices[patient_id][:n_shots]
        self.signals = [task_dataset.base_dataset[idx][0] for idx in indices]
        self.labels  = [task_dataset.base_dataset[idx][1] for idx in indices]

        if validation:
            # For validation, we use all signals from the patient
            n_shots = len(self.signals)
      
        self.total_samples = len(self.signals)
    
    def __len__(self):
        return self.total_samples
    
    def __getitem__(self, index):
        signals = self.signals[index]
        labels = self.labels[index]

        return signals, labels
    