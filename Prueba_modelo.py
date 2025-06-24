import torch
from Modelos.Modelo_conv import Modelo_Convolucional
from Clase_UCIDataset import UCIDataset
import os
from tqdm.auto import tqdm 

def main():

    parameters = {
            'batch_size': 20,
            'shuffle': True,
            'num_workers': 2,
            'pin_memory': True
        }

    print(os.path.exists('data_UCI/dataset_completo_prueba.pt'))
    dataset = UCIDataset('data_UCI/dataset_completo_prueba.pt')
    dataset=dataset[:(20)]
    dataloader = torch.utils.data.DataLoader(dataset, **parameters)
    
    path_model='best_model.pt'

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model=Modelo_Convolucional(in_channels=2,out_channels=2, long_signal=250)
    
    checkpoint = torch.load(path_model)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    model.to(device)
    bar = tqdm(dataloader['test'])
    acc = []
    with torch.no_grad():
        for batch in bar:
            data, labels = batch
            data, labels = data.to(device), labels.to(device)
            preds = model.forward(data)
            

if __name__ == '__main__':
    main()
