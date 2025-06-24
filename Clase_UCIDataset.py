import torch.utils.data as data
import torch
import os


class UCIDataset(data.Dataset):
    def __init__(self, data_dir='data_UCI'):
        """
        Args:
            list_IDs (list): List of IDs for the dataset.
            labels (list): Corresponding labels for the dataset.
        """
        if not os.path.exists(data_dir):
            raise FileNotFoundError(f"¡Archivo no encontrado: {data_dir}")

        data_dict = torch.load(data_dir)
        self.data = data_dict['data']        # Tensor [N, 2, 250]
        self.labels = data_dict['labels']    # Tensor [N, 2]

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
        return self.data[index], self.labels[index]



