"""
Módulo: Modelo_ConvolucionalV1.py
Autor: Juan Marcos Grigolatto
Descripción: Arquitectura de Red Neuronal Convolucional 1D (1D-CNN) para la 
             estimación de presión arterial. Utiliza capas convolucionales para 
             la extracción automática de características de las señales (PPG/ECG) 
             y capas densas con activación ELU para la regresión de los valores 
             de presión (Sistólica y Diastólica).
""" 
import torch
import torch.nn as nn
import torch.nn.functional as F

def calculate_output_dim(long_signal=250):
    """
    Calcula dinámicamente la dimensión del tensor aplanado (Flatten) tras pasar
    por las capas convolucionales y de pooling. Esto permite instanciar el modelo
    con ventanas de señal de distintas longitudes sin romper la primera capa Lineal.
    """
    dim = long_signal
    # Conv1 (kernel=3, sin padding)
    dim = dim - 2  
    # Conv2 (kernel=3, sin padding)
    dim = dim - 2
    # Pool1 (kernel=2, stride=2)
    dim = dim // 2

    # Conv3 (kernel=3, sin padding)
    dim = dim - 2
    # Pool2 (kernel=2, stride=2)
    dim = dim // 2

    # Conv4 (kernel=3, sin padding)
    dim = dim - 2
    # Pool3 (kernel=2, stride=2)
    dim = dim // 2
    
    # Multiplicado por los 128 canales de salida de la última convolución (Conv4)
    return 128 * dim

class Modelo_ConvolucionalV1(nn.Module):
    """
    Arquitectura Base 1D-CNN.
    - Extractor de características: 4 capas Conv1d, 3 capas MaxPool1d, Batch Normalization.
    - Regresor: 4 capas Lineales (Densas) con Dropout para regularización.
    - Activaciones: ReLU para convoluciones, ELU para capas densas (ideal en regresión 
      para evitar gradientes muertos).
    """
    def __init__(self, in_channels, out_channels, long_signal=250):
        super().__init__()
        
        # --- BLOQUE EXTRACTOR DE CARACTERÍSTICAS (FEATURE EXTRACTOR) ---
        self.conv1 = nn.Conv1d(in_channels, 16, kernel_size=3,stride=1)
        self.bn1 = nn.BatchNorm1d(16)

        self.conv2 = nn.Conv1d(16, 64, kernel_size=3,stride=1)
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
        self.bn2 = nn.BatchNorm1d(64)

        self.conv3 = nn.Conv1d(64, 128, kernel_size=3,stride=1)
        self.bn3 = nn.BatchNorm1d(128)

        self.conv4 = nn.Conv1d(128, 128, kernel_size=3,stride=1)
        self.bn4 = nn.BatchNorm1d(128)

        # --- BLOQUE REGRESOR ---
        self.flatten = nn.Flatten()

        # Cálculo dinámico de la entrada a la red neuronal densa
        dim_flatten=calculate_output_dim(long_signal)

        self.dense1= nn.Linear(dim_flatten, 150)
        self.dropout = nn.Dropout(0.5)
        self.bn5 = nn.BatchNorm1d(150)

        self.dense2= nn.Linear(150, 50)
        self.bn6 = nn.BatchNorm1d(50)

        self.dense3= nn.Linear(50, 20)

        # Capa de salida: emite 'out_channels' valores (ej. 2 para SBP y DBP)
        self.dense4= nn.Linear(20, out_channels)

    def forward(self, x):
        """ Define el flujo de procesamiento de los tensores (Forward Pass) a través de la red.
            - Entrada 'x': Tensor de forma (batch_size, in_channels, long_signal).
            - Salida: Tensor de forma (batch_size, out_channels) con las predicciones 
              de presión arterial.
        """
        # Bloque Conv 1 & 2 + Pool
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)

        # Bloque Conv 3 + Pool
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.pool(x)

        # Bloque Conv 4 + Pool
        x = F.relu(self.bn4(self.conv4(x)))
        x = self.pool(x)

        # Aplanado a 1D (Vector de características)
        x = self.flatten(x)

        # Capas Densas (Regresión)
        x = F.elu(self.bn5(self.dense1(x)))
        x = self.dropout(x)

        x = F.elu(self.bn6(self.dense2(x)))
        x = self.dropout(x)

        x = F.elu(self.dense3(x))


        x= self.dense4(x)
        
        return x
    
    

