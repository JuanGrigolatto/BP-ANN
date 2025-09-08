import torch.utils.data as data
import numpy as np
from src.data.data_chargers.MetaDataset import TaskDataset

class TuningNDataset(data.Dataset):
    #Antes no estaba el import de TaskDataset y ponía 'TaskDataset' entre comillas
    def __init__(self, task_dataset: TaskDataset, patient_id: int, n_shots=5, validation=False):
        
        if validation:
            # For validation, we use all signals from the patient
            self.signals = task_dataset.patient_to_signals[patient_id]
            self.labels = task_dataset.patient_to_labels[patient_id]
            n_shots = len(self.signals)
        else:
            self.signals = task_dataset.patient_to_signals[patient_id][:n_shots]
            self.labels = task_dataset.patient_to_labels[patient_id][:n_shots]

        self.total_samples = len(self.signals)
    
    def __len__(self):
        return self.total_samples
    
    def __getitem__(self, index):
        x = self.signals[index]
        y = self.labels[index]
        return x, y
    