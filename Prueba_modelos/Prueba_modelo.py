"""
Módulo: Analisis_error.py
Autor: Juan Marcos Grigolatto
Descripción: Script de evaluación final y clínica del modelo entrenado. 
             Calcula métricas estadísticas (MSE, RMSE, MAE, R2) y el estándar médico 
             AAMI/ISO para estimadores de presión arterial. Genera visualizaciones clave 
             como gráficos de Bland-Altman, diagramas de dispersión, distribución 
             de residuos y comparativas morfológicas de señales con alto/bajo error.
"""
import torch
from src.models.InceptionTime import InceptionTime
from src.models.ConvolucionalV1 import Modelo_ConvolucionalV1
from src.models.ConvolucionalV2 import Modelo_ConvolucionalV2
from src.data.data_chargers.Clase_UCIDataset import UCIDataset
import os
from tqdm.auto import tqdm 
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import pearsonr
import numpy as np
import matplotlib.pyplot as plt
import src.utils.Tools.Tools as Tools
import random

def set_seed(seed=42):
    """Establece la semilla para todas las operaciones aleatorias en Python, NumPy y PyTorch, asegurando la reproducibilidad de los experimentos. Esta función configura la semilla para el módulo random de Python, el generador de números aleatorios de NumPy, y los generadores de números aleatorios de PyTorch tanto para CPU como para GPU. Además, ajusta las configuraciones de cuDNN para garantizar resultados determinísticos en operaciones convolucionales. 

    Args:
        seed (int, optional): Por defecto 42.  La semilla que se utilizará para inicializar los generadores de números aleatorios en Python, NumPy y PyTorch, asegurando que los experimentos sean reproducibles.     
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

SBP_MIN, SBP_MAX = 70, 199.99
DBP_MIN, DBP_MAX = 50, 140
SBP_MEAN = 134.02
DBP_MEAN = 63.47
SBP_STD = 22.75
DBP_STD = 23.69
"""
ABP_MAX, ABP_MIN = 60, 180.00
"""
def aami_metrics(y_true, y_pred):
    """Calcula las métricas AAMI/ISO para un conjunto de predicciones y valores verdaderos. Esta función calcula el error medio y la desviación estándar del error entre las predicciones y los valores reales, utilizando la fórmula estándar de AAMI que se emplea para evaluar la precisión de los dispositivos de medición de presión arterial. 

    Args:
        y_true: Vector de valores verdaderos (real) de presión arterial, ya sea SBP o DBP, en unidades desnormalizadas (mmHg).
        y_pred: Vector de valores predichos (modelo) de presión arterial, ya sea SBP o DBP, en unidades desnormalizadas (mmHg).

    Returns:
        tipo: Tupla que contiene el error medio (mean_error) y la desviación estándar del error (std_error) entre las predicciones y los valores reales, ambos en unidades de mmHg. 
    """    
    errors = y_pred - y_true
    mean_error = np.mean(errors)
    std_error = np.std(errors, ddof=1)
    return mean_error, std_error

def desnormalizar_minmax(norm_array, min_val, max_val):
    """Desnormaliza un array normalizado utilizando la fórmula de normalización min-max. Esta función toma un array de valores normalizados (en el rango [0, 1]) y los convierte de nuevo a su escala original utilizando los valores mínimo y máximo proporcionados. 
    Args:
        norm_array: Array de valores normalizados, donde cada valor se encuentra en el rango [0, 1], resultado de aplicar la normalización min-max a los datos originales.
        min_val: Valor mínimo de la escala original.
        max_val: Valor máximo de la escala original.

    Returns:
        tipo: Array de valores desnormalizados, donde cada valor ha sido transformado de nuevo a su escala original. 
    """    
    return norm_array * (max_val - min_val) + min_val

def desnormalizar_zscore(norm_array, media, std):
    """Desnormaliza un array normalizado utilizando la fórmula de normalización z-score. Esta función toma un array de valores normalizados (con media 0 y desviación estándar 1) y los convierte de nuevo a su escala original utilizando la media y desviación estándar proporcionadas.

    Args:
        norm_array: Array de valores normalizados, donde cada valor tiene media 0 y desviación estándar 1, resultado de aplicar la normalización z-score a los datos originales.
        media: Media de la escala original.
        std: Desviación estándar de la escala original.

    Returns:
        tipo: Array de valores desnormalizados, donde cada valor ha sido transformado de nuevo a su escala original.
    """
    return norm_array * std + media

def bland_altman_graf(preds, labels, title, color="blue"):
    """Genera un gráfico de Bland-Altman para evaluar la concordancia entre las predicciones y los valores reales. Este gráfico muestra la diferencia entre las predicciones y los valores reales en función del promedio de ambos, permitiendo visualizar la presencia de sesgos sistemáticos y la dispersión de los errores. 

    Args:
        preds: Vector de valores predichos por el modelo, en unidades desnormalizadas (mmHg).
        labels: Vector de valores verdaderos (real) de presión arterial, ya sea SBP o DBP, en unidades desnormalizadas (mmHg).
        title: Título del gráfico.
        color (str, optional): Por defecto "blue". Color de los puntos en el gráfico de Bland-Altman. Puede ser cualquier color válido en Matplotlib, como "blue", "red", "green", etc. 
    """    
    differences = preds - labels
    averages = (preds + labels) / 2

    mean_diff = np.mean(differences)
    std_diff = np.std(differences)

    upper_limit = mean_diff + 1.96 * std_diff
    lower_limit = mean_diff - 1.96 * std_diff

    plt.figure(figsize=(8,5))
    plt.scatter(averages, differences, alpha=0.2, color=color, s=15)
    plt.axhline(mean_diff, color='red', linestyle='--', label=f'Media: {mean_diff:.2f}')
    plt.axhline(upper_limit, color='gray', linestyle='--', label=f'+1.96 SD: {upper_limit:.2f}')
    plt.axhline(lower_limit, color='gray', linestyle='--', label=f'-1.96 SD: {lower_limit:.2f}')
    plt.axhline(0, color='black', linewidth=1)
    plt.xlabel('Promedio (mmHg)')
    plt.ylabel('Diferencia (Pred - Real) (mmHg)')
    plt.title(f'Gráfico de Bland-Altman - {title}')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_comparacion_errores(dataset, indices_alto, indices_bajo, n=3):
    """Genera gráficos comparativos de las señales ECG y PPG para muestras con errores altos y bajos. Esta función toma un conjunto de índices correspondientes a muestras con errores altos y bajos, extrae las señales ECG y PPG de esas muestras, y las grafica en subplots para facilitar la comparación visual de las morfologías de las señales en ambos casos.

    Args:
        dataset: Conjunto de datos que contiene las señales ECG y PPG, así como las etiquetas y metadatos asociados. Se espera que el dataset tenga un formato compatible con PyTorch, donde cada muestra se puede acceder mediante un índice y devuelve las señales y etiquetas correspondientes.
        indices_alto: Lista de índices correspondientes a muestras con errores altos.
        indices_bajo: Lista de índices correspondientes a muestras con errores bajos.
        n (int, optional): Número de muestras a graficar en cada fila. Por defecto 3.
    """    
    fig, axes = plt.subplots(2, n, figsize=(4*n, 6))

    for i in range(n):
        idx = indices_alto[i]
        senales, _, _, _ = dataset[idx]
        ppg, ecg = senales[0].numpy(), senales[1].numpy()

        axes[0, i].plot(ecg, color="orange", label="ECG")
        axes[0, i].plot(ppg, color="blue", label="PPG")
        axes[0, i].set_title(f"Error alto (idx {idx})")

    for i in range(n):
        idx = indices_bajo[i]
        senales, _ , _, _ = dataset[idx]
        ppg, ecg = senales[0].numpy(), senales[1].numpy()

        axes[1, i].plot(ecg, color="orange", label="ECG")
        axes[1, i].plot(ppg, color="blue", label="PPG")
        axes[1, i].set_title(f"Error bajo (idx {idx})")

    plt.tight_layout()
    plt.show()

def main():
    """Función principal que realiza la evaluación del modelo entrenado en el conjunto de prueba. Esta función carga el modelo, prepara el conjunto de datos de prueba, realiza las predicciones, calcula las métricas de evaluación (MSE, RMSE, MAE, R2, AAMI), y genera visualizaciones como gráficos de Bland-Altman, diagramas de dispersión y comparativas morfológicas de señales con alto/bajo error. Además, guarda los errores de predicción en un archivo para análisis posterior.
    """    
    set_seed(42)

    parameters = {
        'batch_size': 256,
        'shuffle': False,
        'num_workers': 0,
        'pin_memory': False,
        'drop_last': False
    }

    print(os.path.exists('data/processed/data_UCI/test_set_por_pacientes_iso.pt/test_meta.pt'))
    dataset = UCIDataset(['data/processed/data_UCI/test_set_por_pacientes_iso.pt/test_meta.pt'])
    
    subset = torch.utils.data.Subset(dataset, indices=list(range(10000)))
    dataloader = torch.utils.data.DataLoader(subset, **parameters)

    all_labels = []
    for x, y, pid, idx in dataloader:
        all_labels.append(y)

    labels = torch.cat(all_labels, dim=0)

    print("min:", labels.min().item())
    print("max:", labels.max().item())
    print("mean:", labels.mean().item())
    print("std:", labels.std().item())
    print("Hay NaNs:", torch.isnan(labels).any().item())
    print("Hay Infs:", torch.isinf(labels).any().item())

    path_model = 'models/best_models/best_model_conv_Time32_200_epocas_picos_def_early8_ps.pt'

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = InceptionTime(c_in=2, c_out=2, seq_len=None, n_filters=32)
    #model=Modelo_ConvolucionalV1(in_channels=2, out_channels=2, long_signal=500)
    #model=Modelo_ConvolucionalV1_2(out_channels=2, long_signal=500)
    
    checkpoint = torch.load(path_model, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    model.to(device)

    bar = tqdm(dataloader)
    
    all_preds_desnorm = []
    all_labels_desnorm = []

    global_idx_counter = 0
    
    accmse, accrmse, accmae, accr2 = [], [], [], []
    acc_mean_error_sbp = []
    acc_std_error_sbp = []
    acc_mean_error_dbp = []
    acc_std_error_dbp = []
    all_preds = []
    all_labels = []

    n_error=0

    indices_errores = []
    valores_errores = []

    indices_error_alto = []
    indices_error_bajo = []

    with torch.no_grad():
        for batch in bar:
            data, labels, ID_paciente, indice_muestra = batch
  
            data, labels = data.to(device), labels.to(device)
            
            pred = model(data)
            

            pred = pred.cpu().numpy()
            labels = labels.cpu().numpy()

            pred_SBP = desnormalizar_zscore(pred[:, 0], SBP_MEAN, SBP_STD)
            pred_DBP = desnormalizar_zscore(pred[:, 1], DBP_MEAN, DBP_STD)
            true_SBP = desnormalizar_zscore(labels[:, 0], SBP_MEAN, SBP_STD)
            true_DBP = desnormalizar_zscore(labels[:, 1], DBP_MEAN, DBP_STD)

            batch_pred_desnorm = np.stack([pred_SBP, pred_DBP], axis=1)
            batch_labels_desnorm = np.stack([true_SBP, true_DBP], axis=1)

            all_preds_desnorm.append(batch_pred_desnorm)
            all_labels_desnorm.append(batch_labels_desnorm)

            for j in range(batch_pred_desnorm.shape[0]):
                pred_sbp_val, pred_dbp_val = batch_pred_desnorm[j]
                true_sbp_val, true_dbp_val = batch_labels_desnorm[j]
                
                error_sbp = abs(pred_sbp_val - true_sbp_val)
                error_dbp = abs(pred_dbp_val - true_dbp_val)
                
                valores_errores.append([error_sbp, error_dbp])
                
                indices_errores.append(indice_muestra[j].item()) 

                current_absolute_idx = global_idx_counter + j
                
                if error_sbp >= 10 or error_dbp >= 10:
                    indices_error_alto.append(current_absolute_idx)
                else:
                    indices_error_bajo.append(current_absolute_idx)
            
            global_idx_counter += batch_pred_desnorm.shape[0]

    all_preds = np.concatenate(all_preds_desnorm, axis=0)
    all_labels = np.concatenate(all_labels_desnorm, axis=0)
    
    print(f"Total muestras evaluadas: {all_preds.shape[0]}")

    mse = mean_squared_error(all_labels, all_preds)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(all_labels, all_preds)
    r2 = r2_score(all_labels, all_preds)
    
    r_sbp, _ = pearsonr(all_labels[:,0], all_preds[:,0])
    r_dbp, _ = pearsonr(all_labels[:,1], all_preds[:,1])

    mean_error_sbp, std_error_sbp = aami_metrics(all_labels[:,0], all_preds[:,0])
    mean_error_dbp, std_error_dbp = aami_metrics(all_labels[:,1], all_preds[:,1])

    print(f"\nMétricas GLOBALES (valores desnormalizados en mmHg):")
    print(f"  MSE: {mse:.4f}")
    print(f" RMSE: {rmse:.4f}")
    print(f"  MAE: {mae:.4f}")
    print(f"   R2: {r2:.4f}")
    print(f"\n   r SBP:  {r_sbp:.4f}")
    print(f"   r DBP:  {r_dbp:.4f}")
    print(f"\nMétricas AAMI Globales:")
    print(f"SBP - Mean Error: {mean_error_sbp:.2f} (+/- {std_error_sbp:.2f})")
    print(f"DBP - Mean Error: {mean_error_dbp:.2f} (+/- {std_error_dbp:.2f})")
    
    print(f"Muestras con error alto: {len(indices_error_alto)}")
    plot_comparacion_errores(subset, indices_error_alto, indices_error_bajo, n=7)
    
    print(f"Muestras con error alto: {len(indices_error_alto)}")
    print(f"Muestras con error bajo: {len(indices_error_bajo)}")
    
    print(f"numero de ventanas con errores: {n_error}")

    errores = {
        'valores': valores_errores,
        'indices': indices_errores
    }

    np.savez('data/processed/Errores_predicción', **errores)
    
    plt.figure(figsize=(8, 5))
    plt.scatter(all_labels[:, 0], all_preds[:, 0], alpha=0.2, label="SBP", color='#1f77b4',s=15)
    plt.scatter(all_labels[:, 1], all_preds[:, 1], alpha=0.2, label="DBP", color='#ff7f0e',s=15)
    plt.plot([all_labels.min(), all_labels.max()],
             [all_labels.min(), all_labels.max()], 'r--')
    plt.xlabel("Valor verdadero (mmHg)")
    plt.ylabel("Predicción (mmHg)")
    plt.title(f"R² SBP={r2_score(all_labels[:,0], all_preds[:,0]):.2f} | "
              f"DBP={r2_score(all_labels[:,1], all_preds[:,1]):.2f}")
    plt.legend()
    plt.grid(True)
    plt.show()

    residuals_sbp = all_preds[:, 0] - all_labels[:, 0]
    residuals_dbp = all_preds[:, 1] - all_labels[:, 1]

    plt.figure(figsize=(8, 4))
    plt.scatter(all_labels[:, 0], residuals_sbp, alpha=0.2, label="SBP", color='#1f77b4', s=15)
    plt.scatter(all_labels[:, 1], residuals_dbp, alpha=0.2, label="DBP", color='#ff7f0e', s=15)
    plt.axhline(0, color='red', linestyle='--')
    plt.xlabel("Valor verdadero (mmHg)")
    plt.ylabel("Error (Pred - Real) (mmHg)")
    plt.title("Errores de predicción (residuos)")
    plt.legend()
    plt.grid(True)
    plt.show()

    plt.figure(figsize=(8, 5))
    plt.hist(residuals_sbp, bins=50, alpha=0.4, label="SBP", color='#1f77b4', edgecolor="black")
    plt.hist(residuals_dbp, bins=50, alpha=0.4, label="DBP", color='#ff7f0e', edgecolor="black")
    plt.title("Distribución de errores")
    plt.xlabel("Error (mmHg)")
    plt.ylabel("Frecuencia")
    plt.legend()
    plt.grid(True)
    plt.show()

    bland_altman_graf(all_preds[:,0], all_labels[:,0], title="SBP", color='#1f77b4')
    bland_altman_graf(all_preds[:,1], all_labels[:,1], title="DBP", color='#ff7f0e')

if __name__ == '__main__':
    main()
