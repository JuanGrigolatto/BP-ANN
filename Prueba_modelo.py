import torch
from Modelos.Modelo_conv import Modelo_Convolucional


path='best_model.pt'

model=Modelo_Convolucional(in_channels=2,out_channels=2, long_signal=250)

checkpoint = torch.load(path)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()