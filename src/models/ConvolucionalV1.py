import torch
import torch.nn as nn
import torch.nn.functional as F

def calculate_output_dim(long_signal=250):
    # Secuencia exacta de operaciones como en el forward()
    dim = long_signal
    # Conv1 + ReLU + BN
    dim = dim - 2  # kernel_size=3, stride=1, padding=0
    # Conv2 + ReLU
    dim = dim - 2
    # Pool1
    dim = dim // 2
    # Conv3 + ReLU
    dim = dim - 2
    # Pool2
    dim = dim // 2
    # Conv4 + ReLU + BN
    dim = dim - 2
    # Pool3
    dim = dim // 2
    # 128 canales de la última conv
    return 128 * dim

class Modelo_ConvolucionalV1(nn.Module):
    def __init__(self, in_channels, out_channels, long_signal=250):
        super().__init__()
        # Bloque 1
        self.conv1 = nn.Conv1d(in_channels, 16, kernel_size=3,stride=1)
        #self.bn1 = nn.BatchNorm1d(16, momentum=0.5, eps=1e-4)
        self.bn1 = nn.BatchNorm1d(16)
        self.conv2 = nn.Conv1d(16, 64, kernel_size=3,stride=1)
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
        self.bn2 = nn.BatchNorm1d(64)
        self.conv3 = nn.Conv1d(64, 128, kernel_size=3,stride=1)
        self.bn3 = nn.BatchNorm1d(128)
        self.conv4 = nn.Conv1d(128, 128, kernel_size=3,stride=1)
        self.bn4 = nn.BatchNorm1d(128)
        self.flatten = nn.Flatten()
        dim_flatten=calculate_output_dim(long_signal)
        self.dense1= nn.Linear(dim_flatten, 150)
        self.dropout = nn.Dropout(0.5)
        self.bn5 = nn.BatchNorm1d(150)
        self.dense2= nn.Linear(150, 50)
        self.bn6 = nn.BatchNorm1d(50)
        #self.dense3= nn.Linear(50, 20)
        #self.dense4= nn.Linear(20, out_channels)
        self.dense3= nn.Linear(50, 20)
        self.dense4= nn.Linear(20, out_channels)


    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.pool(x)
        x = F.relu(self.bn4(self.conv4(x)))
        x = self.pool(x)
        x = self.flatten(x)
        x = F.elu(self.bn5(self.dense1(x)))
        x = self.dropout(x)
        x = F.elu(self.bn6(self.dense2(x)))
        x = self.dropout(x)
        x = F.elu(self.dense3(x))
        x= self.dense4(x)
        return x
    
    

