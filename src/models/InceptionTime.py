"""
Módulo: InceptionTime.py
Autor: Juan Marcos Grigolatto
Descripción: Implementación de la arquitectura InceptionTime adaptada para la 
             estimación de presión arterial a partir de series temporales (ECG/PPG).
             Utiliza bloques Inception con múltiples campos receptivos en paralelo 
             y conexiones residuales (ResNet) para evitar el desvanecimiento del gradiente.
             Finaliza con Global Average Pooling (GAP) para reducir el sobreajuste.
"""
import torch
import torch.nn as nn
from fastcore.all import delegates

class Add(nn.Module):
    """Capa auxiliar para sumar la conexión residual al flujo principal del tensor."""
    def forward(self, x, y):
        return x + y

def ifnone(a, b):
    """Retorna 'b' si 'a' es None, útil para inicialización de parámetros por defecto."""
    return b if a is None else a

# --- ARQUITECTURA  ---

class InceptionModule(nn.Module):
    """
    Módulo Inception 1D base. 
    Aplica convoluciones de diferentes tamaños de kernel en paralelo para extraer 
    características temporales a múltiples escalas, y luego concatena los resultados.
    """
    def __init__(self, in_channels, bottleneck=True, n_filters=32, max_kernel_size=40):
        super(InceptionModule, self).__init__()

        # Cuello de botella (Bottleneck) con Conv 1x1 para reducir dimensionalidad
        # y costo computacional.
        self.use_bottleneck = bottleneck and in_channels > 1
        bottleneck_channels = n_filters if self.use_bottleneck else in_channels
        self.bottleneck_layer = nn.Conv1d(in_channels, n_filters, kernel_size=1, bias=False) if self.use_bottleneck else nn.Identity()
        
        # Cálculo de los tamaños de kernel (ej. 39, 19, 9) asegurando que sean impares
        # para mantener el padding simétrico.
        ks = [max_kernel_size // (2 ** i) for i in range(3)]
        ks = [k if k % 2 != 0 else k - 1 for k in ks]

        # Creación de las 3 ramas convolucionales paralelas
        self.convs = nn.ModuleList([
            nn.Conv1d(bottleneck_channels, n_filters, k, padding=(k-1)//2, bias=False) for k in ks
        ])

        # Rama paralela adicional con MaxPooling para invariancia traslacional
        self.maxconvpool = nn.Sequential(
            nn.MaxPool1d(3, stride=1, padding=1),
            nn.Conv1d(in_channels, n_filters, 1, bias=False)
        )

        # Normalización y Activación tras concatenar (3 ramas Conv + 1 rama Pool = n_filters * 4)
        self.batch_norm = nn.BatchNorm1d(n_filters * 4)
        self.activacion = nn.ReLU()

    def forward(self, x):
        input_tensor = x
        x = self.bottleneck_layer(x)

        # Ejecución en paralelo de las ramas convolucionales
        conv_outputs = [conv(x) for conv in self.convs]
        maxpool_output = self.maxconvpool(input_tensor)

        # Concatenación de todos los mapas de características en la dimensión de canales
        x = torch.cat(conv_outputs + [maxpool_output], dim=1)
        x = self.batch_norm(x)
        x = self.activacion(x)

        return x
    
@delegates(InceptionModule.__init__)
class InceptionBlock(nn.Module):
    """
    Agrupa múltiples InceptionModules en serie e incorpora conexiones residuales 
    (skip connections) cada 3 módulos, permitiendo entrenar redes más profundas 
    sin sufrir de desvanecimiento del gradiente.
    """
    def __init__(self, in_channels, n_filters=32, residual=True, depth=6, **kwargs):
        super(InceptionBlock, self).__init__()
        self.residual = residual
        self.depth = depth

        self.inception_modules = nn.ModuleList()
        self.shortcut= nn.ModuleList()
        self.add = Add()
        self.act = nn.ReLU()

        # Construcción de la profundidad de la red
        for d in range(depth):

            # El primer módulo recibe in_channels, los siguientes reciben la salida concatenada (n_filters * 4)
            self.inception_modules.append(InceptionModule(in_channels if d == 0 else n_filters * 4, n_filters=n_filters, bottleneck=True, **kwargs))
            
            # Se prepara una conexión residual cada 3 bloques
            if self.residual and d % 3 == 2: 
                n_in, n_out = in_channels if d == 2 else n_filters * 4, n_filters * 4
                # Si las dimensiones no coinciden, se usa una Conv 1x1 para proyectar; sino, un BatchNorm
                self.shortcut.append(nn.BatchNorm1d(n_in) if n_in == n_out else nn.Conv1d(n_in, n_out, 1))
        
    def forward(self, x):
        res = x
        for d, l in enumerate(range(self.depth)):
            x = self.inception_modules[d](x)

            # Aplicación de la conexión residual cada 3 capas
            if self.residual and d % 3 == 2: res = x = self.act(self.add(x, self.shortcut[d//3](res)))

        return x
    
@delegates(InceptionModule.__init__)
class InceptionTime(nn.Module):
    """
    Modelo Integrador Final InceptionTime.
    Compuesto por el bloque Inception profundo, seguido de Global Average Pooling 
    para extraer un único valor descriptivo por mapa de características, 
    y una capa Lineal para la predicción de valores continuos.
    """
    def __init__(self, c_in, c_out, seq_len=None, n_filters=32, nb_filters=None, **kwargs):
        super(InceptionTime, self).__init__()
        n_filters = ifnone(n_filters, nb_filters) 
        
        # Extractor de características
        self.inceptionblock = InceptionBlock(c_in, n_filters, **kwargs)

        # Global Average Pooling (GAP) reemplaza al clásico Flatten
        # Promedia la dimensión temporal
        self.gap = nn.AdaptiveAvgPool1d(1)

        # Capa de regresión final
        self.fc = nn.Linear(n_filters * 4, c_out)

    def forward(self, x):
        """
        Flujo de procesamiento de los tensores (Forward Pass).
        """

        # 1. Extracción de características espacio-temporales
        x = self.inceptionblock(x)

        # 2. Reducción de la dimensión temporal a 1 (GAP)
        x = self.gap(x)
        
        x = x.squeeze(-1) # Elimina la dimensión temporal

        # 3. Predicción lineal
        x = self.fc(x)

        return x