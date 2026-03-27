import random
import torch
import numpy as np
from src.data.data_chargers.Clase_UCIDataset import UCIDataset
from collections import defaultdict
import matplotlib.pyplot as plt
import os
from tqdm import tqdm

# ==========================================
# Configuración idéntica al Meta-Learning
# ==========================================
SBP_MEAN, SBP_STD = 134.02, 22.75
DBP_MEAN, DBP_STD = 63.47, 23.69
SEED = 42
SHOTS = 5
GAP = 50
MIN_REQUIRED = (2 * SHOTS) + GAP # 60 ventanas mínimas

def desnormalizar_zscore(norm_tensor, media, std):
    return norm_tensor * std + media

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def obtener_datos_por_pids(dataset, pids, mapping):
    sbp_list, dbp_list = [], []
    for pid in tqdm(pids, desc="Extrayendo etiquetas"):
        for idx in mapping[pid]:
            _, label, _, _ = dataset[idx]
            sbp_list.append(desnormalizar_zscore(label[0].item(), SBP_MEAN, SBP_STD))
            dbp_list.append(desnormalizar_zscore(label[1].item(), DBP_MEAN, DBP_STD))
    return np.array(sbp_list), np.array(dbp_list)

def main():
    set_seed(SEED)
    
    data_paths = [
        'data/processed/data_UCI/dataset_parte_1_por_picos.pt',
        'data/processed/data_UCI/dataset_parte_2_por_picos.pt',
        'data/processed/data_UCI/dataset_parte_3_por_picos.pt',
        'data/processed/data_UCI/dataset_parte_4_por_picos.pt',
    ]

    print("Cargando dataset base...")
    dataset_completo = UCIDataset(data_paths)

    # 1. Mapeo y Filtrado (Lógica MAML)
    temp_indices = defaultdict(list)
    for i in range(len(dataset_completo)):
        pid = int(dataset_completo[i][2])
        temp_indices[pid].append(i)

    # Solo pacientes que cumplen con la arquitectura de la tarea (60 muestras)
    valid_patients = [pid for pid in temp_indices.keys() if len(temp_indices[pid]) >= MIN_REQUIRED]
    random.shuffle(valid_patients)
    
    # 2. Split Estricto (70/15/15)
    n_train = int(len(valid_patients) * 0.70)
    n_val_end = int(len(valid_patients) * 0.85)
    
    train_pids = valid_patients[:n_train]
    val_pids = valid_patients[n_train : n_val_end]
    test_pids = valid_patients[n_val_end:]
    
    print(f"Pacientes Válidos (>=60 muestras): {len(valid_patients)}")
    print(f"Distribución: Train {len(train_pids)} | Val {len(val_pids)} | Test {len(test_pids)}")

    # 3. Extracción de datos
    sbp_train, dbp_train = obtener_datos_por_pids(dataset_completo, train_pids, temp_indices)
    sbp_val, dbp_val = obtener_datos_por_pids(dataset_completo, val_pids, temp_indices)
    sbp_test, dbp_test = obtener_datos_por_pids(dataset_completo, test_pids, temp_indices)

    # 4. Ploteo Final (Estética original preservada)
    fig, axes = plt.subplots(3, 2, figsize=(14, 16))
    
    config = [
        (sbp_train, dbp_train, "Meta-Entrenamiento", "skyblue", "salmon"),
        (sbp_val, dbp_val, "Meta-Validación", "lightgreen", "orange"),
        (sbp_test, dbp_test, "Meta-Prueba", "plum", "tomato")
    ]
    labels_abc = ["a)", "b)", "c)"]

    for i, (sbp, dbp, nombre, c1, c2) in enumerate(config):
        # SBP
        axes[i, 0].hist(sbp, bins=70, color=c1, edgecolor='black', alpha=0.7)
        axes[i, 0].set_title(f"SBP - {nombre}", fontsize=12)
        axes[i, 0].set_ylabel("Frecuencia")
        axes[i, 0].grid(alpha=0.3)
        
        # DBP
        axes[i, 1].hist(dbp, bins=70, color=c2, edgecolor='black', alpha=0.7)
        axes[i, 1].set_title(f"DBP - {nombre}", fontsize=12)
        axes[i, 1].grid(alpha=0.3)

        # Referencia a), b), c) (Posición original)
        axes[i, 0].text(-0.15, 1.15, labels_abc[i], transform=axes[i, 0].transAxes, 
                        fontsize=20, va='top', ha='right')

    plt.tight_layout(pad=4.0)
    os.makedirs("figures/", exist_ok=True)
    plt.savefig("figures/caracterizacion_meta_dataset.png", dpi=300, bbox_inches='tight')
    print("¡Gráfico generado! Imagen guardada en figures/caracterizacion_meta_dataset.png")

if __name__ == "__main__":
    main()