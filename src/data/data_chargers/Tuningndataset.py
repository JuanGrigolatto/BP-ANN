import torch.utils.data as data
from src.data.data_chargers.MetaDataset import TaskDataset

class TuningNDataset(data.Dataset):
    def __init__(self, task_dataset: TaskDataset, patient_id: int, n_shots=5, validation=False):
        # Extracción de todos los índices primero
        all_indices = task_dataset.patient_to_indices[patient_id]

        if not validation:
            # Entrenamiento (Support): Solo los primeros n_shots
            self.indices = all_indices[:n_shots]
        else:
            # Evaluación (Query): Todo el RESTO de los latidos después de los primeros n_shots
            # Esto garantiza que el modelo se evalúe con datos que NUNCA vio.
            self.indices = all_indices[n_shots:]

        self.signals = [task_dataset.base_dataset[idx][0] for idx in self.indices]
        self.labels  = [task_dataset.base_dataset[idx][1] for idx in self.indices]
      
        self.total_samples = len(self.signals)
    
    def __len__(self):
        return self.total_samples
    
    def __getitem__(self, index):
        signals = self.signals[index]
        labels = self.labels[index]

        return signals, labels
    