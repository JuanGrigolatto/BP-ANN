import torch.utils.data as data
import numpy as np

class SinglePatientDataset(data.Dataset):
    def __init__(self, task_dataset: 'TaskDataset', patient_id: int):
        
        self.signals = task_dataset.patient_to_signals[patient_id]
        self.labels = task_dataset.patient_to_labels[patient_id]

        self.total_samples = len(self.signals)
    
    def __len__(self):
        return self.total_samples
    
    def __getitem__(self, index):
        x = self.signals[index]
        y = self.labels[index]
        return x, y