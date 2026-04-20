"""
Módulo: Modelo_ConvolucionalV2.py
Autor: Juan Marcos Grigolatto
Descripción: Arquitectura de Red Neuronal Convolucional 1D (1D-CNN).
             A diferencia de la V1, este modelo presenta una arquitectura más ligera 
             y menos regularizada: reduce el uso de capas de Batch Normalization, 
             elimina el Dropout en el bloque denso y utiliza activaciones ReLU en lugar 
             de ELU para la regresión.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

def calculate_output_dim(long_signal=250):
    """
    Calcula dinámicamente la dimensión del tensor aplanado (Flatten) tras pasar
    por las capas convolucionales y de pooling. Permite adaptar la red a ventanas
    de señal de diferentes tamaños (ej. 250, 500 muestras).
    """
    dim = long_signal
    
    # Conv1 + ReLU + BN
    dim = dim - 2  # kernel_size=3, stride=1, padding=0
    # Conv2 + ReLU (Sin BN en esta versión)
    dim = dim - 2
    # Pool1
    dim = dim // 2

    # Conv3 + ReLU (Sin BN en esta versión)
    dim = dim - 2
    # Pool2
    dim = dim // 2

    # Conv4 + ReLU + BN
    dim = dim - 2
    # Pool3
    dim = dim // 2

    # 128 canales de la última conv
    return 128 * dim

class Modelo_ConvolucionalV2(nn.Module):
    """
    Variante de 1D-CNN para estimación de presión arterial.
    Diseñada con menor regularización explícita (sin Dropout) y activaciones ReLU.
    """
    def __init__(self, in_channels, out_channels, long_signal=250):
        super().__init__()

        # --- BLOQUE EXTRACTOR DE CARACTERÍSTICAS (FEATURE EXTRACTOR) ---
        self.conv1 = nn.Conv1d(in_channels, 16, kernel_size=3, stride=1)
        self.bn1 = nn.BatchNorm1d(16)

        self.conv2 = nn.Conv1d(16, 64, kernel_size=3, stride=1)
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)

        self.conv3 = nn.Conv1d(64, 128, kernel_size=3, stride=1)

        self.conv4 = nn.Conv1d(128, 128, kernel_size=3, stride=1)
        self.bn4 = nn.BatchNorm1d(128)

        # --- BLOQUE REGRESOR ---
        self.flatten = nn.Flatten()

        # Cálculo del tamaño de entrada para la primera capa lineal
        dim_flatten=calculate_output_dim(long_signal)

        self.dense1 = nn.Linear(dim_flatten, 150)
        self.dense2 = nn.Linear(150, 50)
        self.dense3= nn.Linear(50, 20)

        # Capa de salida lineal para regresión
        self.dense4 = nn.Linear(20, out_channels)

    def forward(self, x):
        """
        Flujo de procesamiento de los tensores (Forward Pass).
        """
        # Bloque Conv 1 & 2
        x= F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.conv2(x))
        x = self.pool(x)

        # Bloque Conv 3
        x = F.relu(self.conv3(x))
        x = self.pool(x)

        # Bloque Conv 4
        x = F.relu(self.bn4(self.conv4(x)))
        x = self.pool(x)

        # Aplanado
        x = self.flatten(x)

        # Bloque Denso (Regresión con activaciones ReLU)
        x = F.relu(self.dense1(x))
        x = F.relu(self.dense2(x))
        x = F.relu(self.dense3(x))

        # Salida sin activación
        x = self.dense4(x)
        
        return x
