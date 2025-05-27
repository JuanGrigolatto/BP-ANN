import torch
from Clase_UCIDataset import UCIDataset
import os
import numpy as np
#%%
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")
#%%
parameters = {
    'batch_size': 64,
    'shuffle': True,
    'num_workers': 0,
}    
max_epochs=5

# Obtener los IDs de los archivos
data_dir = 'datos_UCI'
all_IDs = [f[:-3] for f in os.listdir(data_dir) if f.endswith('.pt')]

#  Aleatorizar los IDs
np.random.shuffle(all_IDs)

# Crear la partición (80% train, 20% validation)
train_size = int(0.8 * len(all_IDs))
partition = {
    'train': all_IDs[:train_size],
    'validation': all_IDs[train_size:]
}

# Generators
training_set = UCIDataset(partition['train'], data_dir=data_dir)	
training_generator = torch.utils.data.DataLoader(training_set, **parameters)

validation_set = UCIDataset(partition['validation'], data_dir=data_dir)
validation_generator = torch.utils.data.DataLoader(validation_set, **parameters)

for epoch in range(max_epochs):
    # Training
    for local_batch, local_labels in training_generator:
        # Transfer to GPU
        local_batch, local_labels = local_batch.to(device), local_labels.to(device)

with torch.set_grad_enabled(False):
        for local_batch, local_labels in validation_generator:
            # Transfer to GPU
            local_batch, local_labels = local_batch.to(device), local_labels.to(device)