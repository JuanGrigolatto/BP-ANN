import torch
import torch.nn as nn
import torch.nn.functional as F

class Modelo_Convolucional(nn.Module):
    def __init__(self, in_channels, out_channels, long_signal=250):
        super().__init__()

        self.conv1=nn.Conv1d(in_channels=in_channels, out_channels=16, kernel_size=7, padding=3)
        self.bn1 = nn.BatchNorm1d(16)
        
        self.conv2 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm1d(32)

        self.conv3 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(64)

        self.pool = nn.AdaptiveAvgPool1d(1)  # reduce to (batch, channels, 1)
        self.fc = nn.Linear(64, out_channels)

    def forward (self, x):
        x=F.relu(self.bn1(self.conv1(x))) # (B, 16, L)
        x=F.relu(self.bn2(self.conv2(x))) # (B, 32, L)
        x=F.relu(self.bn3(self.conv3(x))) # (B, 64, L)
        x=self.pool(x)                 # (B, 64, 1)
        x=x.squeeze(-1)                # (B, 64)
        x = self.fc(x)                 # (B, 2) 
        return x