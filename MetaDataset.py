import torch.utils.data as data
import torch
import numpy as np

class TaskDataset(data.Dataset):
    def __init__(self, list_IDs ,data_dir='datos_UCI', num_shots=5, len_signal=625):
        self.list_IDs = list_IDs
        self.data_dir = data_dir
        self.num_shots = num_shots
        self.len_signal = len_signal
        


    def __len__(self):
        return len(self.list_IDs)
    
    def __getitem__(self, index):
        ID = self.list_IDs[index]
        # Load data and get label
        file_path = f'{self.data_dir}/{ID}.pt'
        data = torch.load(file_path)

        x = data['signal']  # (2, L)
        y = data['label']   # (2,)

        # Cortar 2*shots muestras aleatorias de la señal
        L = x.shape[1]

        
        
        window_size = 625  # 5 segundos
        total_samples = 2 * self.num_shots 

        if L < self.len_signal:
            raise ValueError(f"El archivo {ID} solo tiene {L} muestras (< {self.len_signal})")
        
        xs, ys = [], []
        for _ in range(total_samples):
            start = np.random.randint(0, L - window_size)
            x_win = x[:, start:start+window_size]
            xs.append(x_win)
            ys.append(y)

        xs = torch.stack(xs)  # (2*shots, 2, 625)
        ys = torch.stack(ys)  # (2*shots, 2)
        return xs, ys