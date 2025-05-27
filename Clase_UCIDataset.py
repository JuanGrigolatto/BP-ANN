import torch.utils.data as data
import torch



class UCIDataset(data.Dataset):
    def __init__(self, list_IDs, data_dir='datos_UCI'):
        """
        Args:
            list_IDs (list): List of IDs for the dataset.
            labels (list): Corresponding labels for the dataset.
        """
        self.list_IDs = list_IDs
        self.data_dir = data_dir
    
    def __len__(self):
        """Returns the total number of samples in the dataset."""
        return len(self.list_IDs)
    
    def __getitem__(self, index):
        """
        Generates one sample of data.
        Args:
            index (int): Index of the sample to retrieve.  """
        ID = self.list_IDs[index]
        # Load data and get label
        file_path = f'{self.data_dir}/{ID}.pt'
        data = torch.load(file_path)
        x = data['signal'] # Tensor (2, longitud_segmento)
        y = data['label']  # Tensor (SBP, DBP)

        return x, y



