"""
Módulo: Graficacion_datos_meta.py
Autor: Juan Marcos Grigolatto
Descripción: Script de validación y caracterización poblacional para el paradigma 
             de Meta-Learning (MAML). Filtra a los sujetos de la base de datos 
             garantizando que posean el número mínimo de ventanas requeridas para 
             armar las tareas (Support + Query + Gap = 60 muestras). Aplica un 
             Patient-Subject Split estricto (70/15/15) y genera histogramas 
             para verificar que las distribuciones fisiológicas de presión 
             (SBP y DBP) se mantengan balanceadas y representativas en los 
             conjuntos de Meta-Entrenamiento, Meta-Validación y Meta-Prueba.
"""
import random
import torch
import numpy as np
from src.data.data_chargers.Clase_UCIDataset import UCIDataset
from collections import defaultdict
import matplotlib.pyplot as plt
import os
from tqdm import tqdm

SBP_MEAN, SBP_STD = 134.02, 22.75
DBP_MEAN, DBP_STD = 63.47, 23.69
SEED = 42
SHOTS = 5
GAP = 50
MIN_REQUIRED = (2 * SHOTS) + GAP 

def desnormalizar_zscore(norm_tensor, media, std):
    """_summary_ Desnormaliza un array normalizado utilizando la fórmula de normalización z-score. Esta función toma un array de valores normalizados (con media 0 y desviación estándar 1) y los convierte de nuevo a su escala original utilizando la media y desviación estándar proporcionadas.

    Args:
        norm_tensor (_type_): _description_ Tensor de valores normalizados, donde cada valor tiene media 0 y desviación estándar 1, resultado de aplicar la normalización z-score a los datos originales.
        media (_type_): _description_ Media de la escala original.
        std (_type_): _description_ Desviación estándar de la escala original.

    Returns:
        _type_: _description_ Tensor de valores desnormalizados, donde cada valor ha sido transformado de nuevo a su escala original.
    """
    return norm_tensor * std + media

def set_seed(seed=42):
    """_summary_ Establece la semilla para la reproducibilidad en múltiples librerías (random, numpy, torch). Esto asegura que los resultados sean consistentes entre ejecuciones, lo cual es crucial para experimentos científicos y comparaciones de modelos.

    Args:
        seed (int, optional): _description_. Por defecto 42. Valor comúnmente utilizado para la semilla, pero puede ser cualquier entero.
    """    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def obtener_datos_por_pids(dataset, pids, mapping):
    """_summary_ Dado un dataset completo, una lista de IDs de pacientes (pids) y un mapeo de IDs a índices en el dataset, esta función extrae los valores de SBP y DBP correspondientes a las muestras de los pacientes especificados. Los valores extraídos se desnormalizan utilizando las medias y desviaciones estándar predefinidas para SBP y DBP, y se devuelven como arrays de numpy.

    Args:
        dataset (_type_): _description_ Instancia del dataset completo que contiene todas las muestras y etiquetas.
        pids (_type_): _description_ Lista de IDs de pacientes que se desean extraer para un conjunto específico (train, val o test).
        mapping (_type_): _description_ Mapeo de IDs de pacientes a índices en el dataset.

    Returns:
        _type_: _description_ Dos arrays de numpy: el primero contiene los valores de SBP desnormalizados y el segundo contiene los valores de DBP desnormalizados, ambos correspondientes a las muestras de los pacientes especificados en pids.
    """    
    sbp_list, dbp_list = [], []
    for pid in tqdm(pids, desc="Extrayendo etiquetas"):
        for idx in mapping[pid]:
            _, label, _, _ = dataset[idx]
            sbp_list.append(desnormalizar_zscore(label[0].item(), SBP_MEAN, SBP_STD))
            dbp_list.append(desnormalizar_zscore(label[1].item(), DBP_MEAN, DBP_STD))
    return np.array(sbp_list), np.array(dbp_list)

def main():
    """_summary_ Función principal que ejecuta el proceso completo de carga del dataset, filtrado de pacientes, división en conjuntos de entrenamiento, validación y prueba, extracción de datos de SBP y DBP, y generación de histogramas para caracterizar la distribución de estas variables en cada conjunto. El resultado final es un gráfico guardado en la carpeta "figures/" que muestra la distribución de SBP y DBP para los conjuntos de Meta-Entrenamiento, Meta-Validación y Meta-Prueba.
    """    
    set_seed(SEED)
    
    data_paths = [
        'data/processed/data_UCI/dataset_parte_1_por_picos.pt',
        'data/processed/data_UCI/dataset_parte_2_por_picos.pt',
        'data/processed/data_UCI/dataset_parte_3_por_picos.pt',
        'data/processed/data_UCI/dataset_parte_4_por_picos.pt',
    ]

    print("Cargando dataset base...")
    dataset_completo = UCIDataset(data_paths)

    temp_indices = defaultdict(list)
    for i in range(len(dataset_completo)):
        pid = int(dataset_completo[i][2])
        temp_indices[pid].append(i)

    valid_patients = [pid for pid in temp_indices.keys() if len(temp_indices[pid]) >= MIN_REQUIRED]
    random.shuffle(valid_patients)
    
    n_train = int(len(valid_patients) * 0.70)
    n_val_end = int(len(valid_patients) * 0.85)
    
    train_pids = valid_patients[:n_train]
    val_pids = valid_patients[n_train : n_val_end]
    test_pids = valid_patients[n_val_end:]
    
    print(f"Pacientes Válidos (>=60 muestras): {len(valid_patients)}")
    print(f"Distribución: Train {len(train_pids)} | Val {len(val_pids)} | Test {len(test_pids)}")

    sbp_train, dbp_train = obtener_datos_por_pids(dataset_completo, train_pids, temp_indices)
    sbp_val, dbp_val = obtener_datos_por_pids(dataset_completo, val_pids, temp_indices)
    sbp_test, dbp_test = obtener_datos_por_pids(dataset_completo, test_pids, temp_indices)

    fig, axes = plt.subplots(3, 2, figsize=(14, 16))
    
    config = [
        (sbp_train, dbp_train, "Meta-Entrenamiento", "skyblue", "salmon"),
        (sbp_val, dbp_val, "Meta-Validación", "lightgreen", "orange"),
        (sbp_test, dbp_test, "Meta-Prueba", "plum", "tomato")
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
    plt.savefig("figures/caracterizacion_meta_dataset.png", dpi=300, bbox_inches='tight')
    print("¡Gráfico generado! Imagen guardada en figures/caracterizacion_meta_dataset.png")

if __name__ == "__main__":
    main()