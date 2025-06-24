import torch
from Modelos.Modelo_conv import Modelo_Convolucional
from Clase_UCIDataset import UCIDataset
import os

def main():

    parameters = {
            'batch_size': 20,
            'shuffle': True,
            'num_workers': 2,
            'pin_memory': True
        }

    print(os.path.exists('data_UCI/dataset_completo_prueba.pt'))
    dataset = UCIDataset('data_UCI/dataset_completo_prueba.pt')
    print(len(dataset))
    dataset=dataset[:(20)]
    print(len(dataset[1]))
    dataloader = torch.utils.data.DataLoader(dataset)
    
    path_model='best_model.pt'

    model=Modelo_Convolucional(in_channels=2,out_channels=2, long_signal=250)
    
    checkpoint = torch.load(path_model)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

if __name__ == '__main__':
    main()
