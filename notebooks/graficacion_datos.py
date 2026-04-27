"""
Módulo: graficacion_datos.py
Autor: Juan Marcos Grigolatto
Descripción: Script de  validación estadística sobre la totalidad de 
             la base de datos (sin filtros restrictivos de Meta-Learning). Realiza 
             un Patient-Subject Split (70/15/15) sobre todos los sujetos disponibles 
             y genera histogramas de distribución para las etiquetas de presión 
             (SBP y DBP). Su objetivo es establecer la "Línea Base" poblacional 
             para confirmar que no existen sesgos severos o fugas de datos 
             antes del entrenamiento de los modelos fundacionales.
"""
import random
import torch
import numpy as np
from torch.utils.data import DataLoader
from src.data.data_chargers.Clase_UCIDataset import UCIDataset
from collections import defaultdict
import matplotlib.pyplot as plt
import os
from tqdm import tqdm

SBP_MEAN, SBP_STD = 134.02, 22.75
DBP_MEAN, DBP_STD = 63.47, 23.69
SEED = 42

def desnormalizar_zscore(norm_tensor, media, std):
    """_summary_ Desnormaliza un tensor z-score a su valor original.

    Args:
        norm_tensor (_type_): _description_ Tensor normalizado (z-score).
        media (_type_): _description_ Media utilizada para la normalización.
        std (_type_): _description_ _descripcion_ Desviación estándar utilizada para la normalización.

    Returns:
        _type_: _description_ Tensor desnormalizado.
    """
    return norm_tensor * std + media

def set_seed(seed=42):
    """_summary_ Establece la semilla para reproducibilidad en random, numpy y torch.

    Args:
        seed (int, optional): _description_. Por defecto 42. Valor de la semilla a utilizar.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

set_seed(SEED)

def obtener_datos_por_pids(dataset, pids, mapping):
    """_summary_ Obtiene las etiquetas de SBP y DBP para una lista de pacientes especificados.

    Args:
        dataset (_type_): _description_ Dataset completo del cual se extraerán las etiquetas.
        pids (_type_): _description_ Lista de IDs de pacientes para los cuales se desean obtener las etiquetas.
        mapping (_type_): _description_ Diccionario que mapea cada PID a una lista de índices en el dataset donde ese PID está presente.

    Returns:
        _type_: _description_ Dos arrays de numpy: el primero con las etiquetas de SBP desnormalizadas y el segundo con las etiquetas de DBP desnormalizadas para los pacientes especificados.
    """
    
    sbp_list, dbp_list = [], []
    for pid in pids:
        for idx in mapping[pid]:
            _, label, _, _ = dataset[idx]
            sbp_list.append(desnormalizar_zscore(label[0].item(), SBP_MEAN, SBP_STD))
            dbp_list.append(desnormalizar_zscore(label[1].item(), DBP_MEAN, DBP_STD))
    return np.array(sbp_list), np.array(dbp_list)

def main():
    """_summary_ Función principal que ejecuta el proceso de carga del dataset, división por pacientes, extracción de etiquetas y generación de histogramas para SBP y DBP en los conjuntos de entrenamiento, validación y prueba.
    """
    data_paths = [
        'data/processed/data_UCI/dataset_parte_1_por_picos.pt',
        'data/processed/data_UCI/dataset_parte_2_por_picos.pt',
        'data/processed/data_UCI/dataset_parte_3_por_picos.pt',
        'data/processed/data_UCI/dataset_parte_4_por_picos.pt',
    ]

    print("Cargando dataset base completo...")
    dataset_completo = UCIDataset(data_paths)

    print("Caracterizando todos los registros por paciente...")
    temp_indices = defaultdict(list)
    for i in tqdm(range(len(dataset_completo)), desc="Procesando dataset"):
        pid = int(dataset_completo[i][2])
        temp_indices[pid].append(i)

    all_patients = list(temp_indices.keys())
    random.shuffle(all_patients) 
    
    n_train = int(len(all_patients) * 0.70)
    n_val = int(len(all_patients) * 0.15)
    
    train_pids = all_patients[:n_train]
    val_pids = all_patients[n_train : n_train + n_val]
    test_pids = all_patients[n_train + n_val:]
    
    print(f"Total de Pacientes Únicos: {len(all_patients)}")
    print(f"Total de Registros (ventanas): {len(dataset_completo)}")
    print(f"Distribución: Train {len(train_pids)} ptes | Val {len(val_pids)} ptes | Test {len(test_pids)} ptes")

    print("Extrayendo etiquetas desnormalizadas para histogramas...")
    sbp_train, dbp_train = obtener_datos_por_pids(dataset_completo, train_pids, temp_indices)
    sbp_val, dbp_val = obtener_datos_por_pids(dataset_completo, val_pids, temp_indices)
    sbp_test, dbp_test = obtener_datos_por_pids(dataset_completo, test_pids, temp_indices)

    fig, axes = plt.subplots(3, 2, figsize=(14, 16))
    
    config = [
        (sbp_train, dbp_train, "Entrenamiento", "skyblue", "salmon"),
        (sbp_val, dbp_val, "Validación", "lightgreen", "orange"),
        (sbp_test, dbp_test, "Prueba", "plum", "tomato")
    ]
    labels_abc = ["a)", "b)", "c)"]

    for i, (sbp, dbp, nombre, c1, c2) in enumerate(config):
        axes[i, 0].hist(sbp, bins=70, color=c1, edgecolor='black', alpha=0.7)
        axes[i, 0].set_title(f"SBP - {nombre}", fontsize=12)
        axes[i, 0].set_ylabel("Frecuencia")
        axes[i, 0].grid(alpha=0.3)
        
        axes[i, 1].hist(dbp, bins=70, color=c2, edgecolor='black', alpha=0.7)
        axes[i, 1].set_title(f"DBP - {nombre}", fontsize=12)
        axes[i, 1].grid(alpha=0.3)

        axes[i, 0].text(-0.15, 1.15, labels_abc[i], transform=axes[i, 0].transAxes, 
                        fontsize=20, va='top', ha='right')

    plt.tight_layout(pad=4.0)
    
    os.makedirs("figures/", exist_ok=True)
    plt.savefig("figures/caracterizacion_total_dataset.png", dpi=300, bbox_inches='tight')
    print("Imagen guardada en figures/caracterizacion_total_dataset.png")
    plt.show()

if __name__ == "__main__":
    main()