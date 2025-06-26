import torch
from Modelos.Modelo_conv import Modelo_Convolucional
from Clase_UCIDataset import UCIDataset
import os
from tqdm.auto import tqdm 
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
import numpy as np
import matplotlib.pyplot as plt

def main():

    parameters = {
            'batch_size': 20,
            'shuffle': True,
            'num_workers': 0,
            'pin_memory': True
        }

    print(os.path.exists('data_UCI/dataset_completo_prueba.pt'))
    dataset = UCIDataset('data_UCI/dataset_completo_prueba.pt')
    subset = torch.utils.data.Subset(dataset, indices=list(range(20)))
    dataloader = torch.utils.data.DataLoader(subset, **parameters)
    
    
    path_model='best_model.pt'

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model=Modelo_Convolucional(in_channels=2,out_channels=2, long_signal=250)
    
    checkpoint = torch.load(path_model, map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    model.to(device)
    bar = tqdm(dataloader)
    accmse = []
    accrmse = []
    accmae = []
    accr2= []
    with torch.no_grad():
        for batch in bar:
            data, labels = batch
            data, labels = data.to(device), labels.to(device)
            pred = model.forward(data)
            
            pred = pred.cpu().numpy()
            labels = labels.cpu().numpy()

            mse = mean_squared_error(labels, pred)
            accmse.append(mse)
            rmse = np.sqrt(mse)
            accrmse.append(rmse)
            mae = mean_absolute_error(labels, pred)
            accmae.append(mae)
            r2= r2_score(labels, pred)
            accr2.append(r2)
    
    print(f"\nMétricas promedio:")
    print(f"  MSE: {np.mean(accmse):.4f}")
    print(f" RMSE: {np.mean(accrmse):.4f}")
    print(f"  MAE: {np.mean(accmae):.4f}")
    print(f"   R2: {np.mean(accr2):.4f}")

if __name__ == '__main__':
    main()
