
import torch
import torch.nn as nn
import torch.nn.functional as F
"""
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
        
        self.relu = nn.ReLU()

    def forward (self, x):
        x=self.relu(self.bn1(self.conv1(x))) # (B, 16, L)
        x=self.relu(self.bn2(self.conv2(x))) # (B, 32, L)
        x=self.relu(self.bn3(self.conv3(x))) # (B, 64, L)
        x=self.pool(x)                 # (B, 64, 1)
        x=x.squeeze(-1)                # (B, 64)
        x = self.fc(x)                 # (B, 2) 
        return x
    """
class Modelo_Convolucional(nn.Module):
    def __init__(self, in_channels, out_channels, long_signal=250):
        super().__init__()
        
        # Bloque 1
        self.conv1 = nn.Conv1d(in_channels, 16, kernel_size=7, padding=3)
        self.bn1 = nn.BatchNorm1d(16, momentum=0.5, eps=1e-4)
        self.dropout1 = nn.Dropout(0.2)
        
        # Bloque 2
        self.conv2 = nn.Conv1d(16, 32, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm1d(32, momentum=0.5, eps=1e-4)
        self.dropout2 = nn.Dropout(0.3)
        
        # Bloque 3
        self.conv3 = nn.Conv1d(32, 64, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(64, momentum=0.5, eps=1e-4)
        self.dropout3 = nn.Dropout(0.4)
        
        # Capa final
        self.pool = nn.AdaptiveAvgPool1d(4)  # Cambiado de 1 a 4
        self.fc = nn.Linear(64*4, out_channels)
        
        # Inicialización cuidadosa
        self._initialize_weights()
        
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.dropout1(x)
        
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.dropout2(x)
        
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.dropout3(x)
        
        x = self.pool(x)
        x = x.view(x.size(0), -1)  # Aplanar manteniendo batch
        x = self.fc(x)
        return x

