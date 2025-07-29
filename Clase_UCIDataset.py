import torch.utils.data as data
import torch
import os


class UCIDataset(data.Dataset):
    def __init__(self, file_list):
        self.data = []
        self.labels = []
        self.ID_patients = []
        self.index_muestra = []
    
        for file_path in file_list:
            data_dict = torch.load(file_path)
            self.data.append(data_dict['data'])
            self.labels.append(data_dict['labels'])
            self.ID_patients.append(data_dict['patient_ids'])
            self.index_muestra.append(data_dict['index'])
    
        self.data = torch.cat(self.data, dim=0)
        self.labels = torch.cat(self.labels, dim=0)
        self.ID_patients = torch.cat(self.ID_patients, dim=0)
        self.index_muestra = torch.cat(self.index_muestra, dim=0)

        print(f"Datos cargados correctamente. Muestras: {len(self.data)}")  # Debug
    
    def __len__(self):
        """Returns the total number of samples in the dataset."""
        return len(self.data)
    
    def __getitem__(self, index):
        """
        Generates one sample of data.
        Args:
            index (int): Index of the sample to retrieve.  
        ID = self.list_IDs[index]
        # Load data and get label
        file_path = f'{self.data_dir}/{ID}.pt'
        data = torch.load(file_path)
        x = data['signal'] # Tensor (2, longitud_segmento)
        y = data['label']  # Tensor (SBP, DBP)

        return x, y
        """
        return self.data[index], self.labels[index], self.ID_patients[index], self.index_muestra[index]



