import random
import torch
import numpy as np
from torch.utils.data import random_split
from src.data.data_chargers.Clase_UCIDataset import UCIDataset
import matplotlib.pyplot as plt
import os
import src.utils.Tools.Tools as Tools

# ========================
# Configuración general
# ========================
SBP_MEAN = 134.02
DBP_MEAN = 63.47
SBP_STD = 22.75
DBP_STD = 23.69

def desnormalizar_zscore(norm_array, media, std):
    return norm_array * std + media

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)

# ========================
# Carga de datos
# ========================

archivos = [
    'data/processed/data_UCI/dataset_parte_1_por_picos.pt',
    'data/processed/data_UCI/dataset_parte_2_por_picos.pt',
    'data/processed/data_UCI/dataset_parte_3_por_picos.pt',
    'data/processed/data_UCI/dataset_parte_4_por_picos.pt',
]

dataset_completo = UCIDataset(archivos)

# División del dataset
total = len(dataset_completo)
train_size = int(0.7 * total)
val_size = int(0.2 * total)
test_size = total - train_size - val_size

train_set, val_set, test_set = random_split(dataset_completo, [train_size, val_size, test_size])
print(f"Train: {len(train_set)}, Val: {len(val_set)}, Test: {len(test_set)}")

# ========================
# Funciones auxiliares
# ========================

def obtener_etiquetas(dataset):
    sbp, dbp = [], []
    for i in range(len(dataset)):
        _, label, _, _ = dataset[i]
        label_sbp = desnormalizar_zscore(label[0].item(), SBP_MEAN, SBP_STD)
        label_dbp = desnormalizar_zscore(label[1].item(), DBP_MEAN, DBP_STD)
        sbp.append(label_sbp)
        dbp.append(label_dbp)
    return np.array(sbp), np.array(dbp)

def plot_hist_sbp_dbp(sbp, dbp, nombre_conjunto, color_sbp, color_dbp, save_dir="figures/"):
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.hist(sbp, bins=50, color=color_sbp, edgecolor='black', alpha=0.7)
    plt.title(f"SBP - {nombre_conjunto}")
    plt.xlabel("Presión sistólica (mmHg)")
    plt.ylabel("Frecuencia")
    plt.grid(alpha=0.3)

    plt.subplot(1, 2, 2)
    plt.hist(dbp, bins=50, color=color_dbp, edgecolor='black', alpha=0.7)
    plt.title(f"DBP - {nombre_conjunto}")
    plt.xlabel("Presión diastólica (mmHg)")
    plt.ylabel("Frecuencia")
    plt.grid(alpha=0.3)

    plt.tight_layout()

    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(os.path.join(save_dir, f"hist_{nombre_conjunto.lower()}.png"), dpi=300)
    plt.show()

# ========================
# Extracción y visualización
# ========================

sbp_train, dbp_train = obtener_etiquetas(train_set)
sbp_val, dbp_val = obtener_etiquetas(val_set)
sbp_test, dbp_test = obtener_etiquetas(test_set)

plot_hist_sbp_dbp(sbp_train, dbp_train, "Entrenamiento", "skyblue", "salmon")
plot_hist_sbp_dbp(sbp_val, dbp_val, "Validación", "lightgreen", "orange")
plot_hist_sbp_dbp(sbp_test, dbp_test, "Prueba", "plum", "tomato")