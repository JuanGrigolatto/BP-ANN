"""
Módulo: Fewshot.py
Autor: Juan Marcos Grigolatto
Descripción: Script central para la fase de adaptación rápida intra-paciente 
             (Few-Shot Fine-Tuning / Delta Learning). Toma el modelo fundacional 
             pre-entrenado mediante MAML y lo calibra a la fisiología específica 
             de pacientes no vistos, utilizando un conjunto mínimo de latidos 
             (Support Set). Evalúa y compara el error de estimación de presión 
             arterial (SBP y DBP) antes y después de la adaptación sobre los 
             latidos futuros (Query Set), cuantificando la ganancia de precisión 
             y la capacidad de personalización del modelo para uso clínico.
"""
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import random
from sklearn.metrics import mean_absolute_error, mean_squared_error

from src.data.data_chargers.Clase_UCIDataset import UCIDataset
from src.data.data_chargers.MetaDataset import TaskDataset
from src.models.ConvolucionalV1 import Modelo_ConvolucionalV1
from src.data.data_chargers.Tuningndataset import TuningNDataset

def promedio_metricas(m_list):
    """_summary_ Realiza el promedio de una lista de métricas, donde cada métrica es una tupla (mae, rmse, bias, std).  

    Args:
        m_list (_type_): _description_ Lista de tuplas con métricas por paciente.

    Returns:
        _type_: _description_ Tupla con el promedio de cada métrica (mae, rmse, bias, std) a nivel global.
    """    
    return np.mean(m_list, axis=0)

def calcular_metricas(y_true, y_pred):
    """_summary_ Calcula las métricas de error (MAE, RMSE) y las métricas clínicas ISO (Bias y STD) para un conjunto de predicciones vs valores reales.

    Args:
        y_true (_type_): _description_ Valores reales de presión arterial (SBP o DBP) para un paciente específico.
        y_pred (_type_): _description_ Valores predichos por el modelo para ese mismo paciente, después de la adaptación few-shot.

    Returns:
        _type_: _description_ Tupla con las métricas calculadas: (MAE, RMSE, Bias, STD), donde:
            - MAE: Error absoluto medio entre predicciones y valores reales.
            - RMSE: Raíz del error cuadrático medio, que penaliza más los errores grandes.
            - Bias: Promedio de los errores (pred - real), que indica si el modelo tiende a sobreestimar o subestimar.
            - STD: Desviación estándar de los errores, que refleja la variabilidad o consistencia de las predicciones.
    """    
    errores = y_pred - y_true
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))    
    
    # Métricas Clínicas (ISO 81060-2)
    bias = np.mean(errores)   
    std = np.std(errores)     
    
    # Retorna 4 valores: 0:MAE, 1:RMSE, 2:BIAS, 3:STD
    return mae, rmse, bias, std

def desnormalizar_zscore(norm_array, media, std):
    """_summary_ Desnormaliza un array que fue normalizado con z-score, utilizando la media y desviación estándar globales del entrenamiento. Esto es crucial para interpretar las predicciones en unidades reales (mmHg) y calcular métricas clínicas significativas.

    Args:
        norm_array (_type_): _description_ Array de predicciones normalizadas (z-score) que se desea desnormalizar.
        media (_type_): _description_ Media global utilizada para la normalización z-score durante el entrenamiento (por ejemplo, media de SBP o DBP en el dataset de entrenamiento).
        std (_type_): _description_ Desviación estándar global utilizada para la normalización z-score durante el entrenamiento (por ejemplo, desviación estándar de SBP o DBP en el dataset de entrenamiento).

    Returns:
        _type_: _description_  Array de predicciones desnormalizadas, en las mismas unidades que los valores reales (mmHg), listo para evaluación clínica y cálculo de métricas.
    """    
    return norm_array * std + media

def tuning(sample, optimizer, model, criterion, device, bias_norm=None):
    """_summary_ Realiza un paso de fine-tuning (adaptación) del modelo para un paciente específico, utilizando un solo batch de datos de soporte (Support Set). Si se proporciona bias_norm, el modelo se entrenará para predecir "deltas" respecto a ese bias, lo que puede mejorar la estabilidad y rapidez de la adaptación en escenarios few-shot.

    Args:
        sample (_type_): _description_ Batch de datos de soporte para un paciente específico, que incluye las señales de entrada y las etiquetas correspondientes (SBP y DBP normalizados).
        optimizer (_type_): _description_ Optimizador para actualizar los parámetros del modelo.
        model (_type_): _description_ Modelo a adaptar.
        criterion (_type_): _description_ Función de pérdida para calcular el error.
        device (_type_): _description_ Dispositivo en el que se ejecuta el modelo (CPU o GPU).
        bias_norm (_type_, optional): _description_ Bias normalizado para calcular deltas. Por defecto None.

    Returns:
        _type_: _description_ Valor de la pérdida después de realizar el paso de adaptación con el batch dado. 
    """    
    optimizer.zero_grad() 
    data, labels, *_ = sample 
    
    if isinstance(data, list): data = data[0]
    if isinstance(labels, list): labels = labels[0]
        
    for layer in model.modules():
        if isinstance(layer, torch.nn.BatchNorm1d):
            layer.eval()
            layer.weight.requires_grad = False
            layer.bias.requires_grad = False

    data, labels = data.to(device), labels.to(device) 
    
    if bias_norm is not None:
        bias_tensor = torch.tensor(bias_norm, dtype=torch.float32, device=device)
        labels = labels - bias_tensor

    preds = model.forward(data) 
    loss = criterion(preds, labels) 
    loss.backward() 
    optimizer.step() 
    return loss.item()

def evaluation(batch, model, criterion, device):
    """_summary_  Realiza la evaluación del modelo en un batch de datos de validación (Query Set) para un paciente específico, calculando las predicciones y la pérdida correspondiente. Esta función se utiliza tanto antes como después del fine-tuning para cuantificar la mejora en la precisión de las predicciones.

    Args:
        batch (_type_): _description_  Batch de datos de evaluación.
        model (_type_): _description_ Modelo a evaluar.
        criterion (_type_): _description_ Función de pérdida para calcular el error.
        device (_type_): _description_ Dispositivo en el que se ejecuta el modelo (CPU o GPU).

    Returns:
        _type_: _description_ Tupla con las predicciones del modelo para el batch dado y el valor de la pérdida calculada, ambos necesarios para el análisis posterior de métricas y visualizaciones.
    """    
    with torch.no_grad():
        data, labels, *_ = batch
        if isinstance(data, list): data = data[0]
        if isinstance(labels, list): labels = labels[0]
        data, labels = data.to(device), labels.to(device)
        preds = model.forward(data)
        loss = criterion(preds, labels)
    return preds, loss

def graficar_resultados_pacientes(true_means, pred_means, maes_post, maes_pre=None, titulo="Por Paciente"):
    """_summary_ Genera un conjunto de gráficos para visualizar los resultados de la adaptación few-shot a nivel de paciente, incluyendo:
    a) Gráfico de dispersión de valores reales vs predichos, destacando pacientes con valores extremos.
    b) Gráfico de Bland-Altman para analizar el bias y la variabilidad de las predicciones.
    c) Gráfico de dispersión comparando MAE pre y post adaptación, resaltando mejoras y fallas.
    d) Histograma de MAE post-adaptación con anotación del porcentaje de pacientes
    
    Args:
        true_means (_type_): _description_ Lista o array con los valores reales promedio de presión arterial (SBP o DBP) para cada paciente, calculados a partir de los latidos del Query Set.
        pred_means (_type_): _description_ Lista o array con los valores predichos promedio de presión arterial para cada paciente, después de la adaptación few-shot.
        maes_post (_type_): _description_ Lista o array con los valores de MAE post-adaptación para cada paciente, que reflejan la precisión final del modelo después de la personalización.
        maes_pre (_type_, optional): _description_ Lista o array con los valores de MAE pre-adaptación para cada paciente, que reflejan la precisión inicial del modelo antes de la personalización. Defaults to None.
        titulo (str, optional): _description_ Título del conjunto de gráficos. Por defecto "Por Paciente".
    """   
    true_means = np.array(true_means)
    pred_means = np.array(pred_means)
    maes_post = np.array(maes_post) 
    if maes_pre is not None: maes_pre = np.array(maes_pre)

    bias_per_patient = pred_means - true_means
    mean_bias = np.mean(bias_per_patient)
    std_bias = np.std(bias_per_patient)

    mean_pop = np.mean(true_means)
    std_pop = np.std(true_means)
    umbral_std = 1.5 
    
    is_extreme = (true_means < (mean_pop - umbral_std * std_pop)) | \
                 (true_means > (mean_pop + umbral_std * std_pop))
    
    n_extremos = np.sum(is_extreme)
    n_normales = len(true_means) - n_extremos

    fig, axs = plt.subplots(2, 2, figsize=(16, 12)) 

    axs[0, 0].scatter(true_means[~is_extreme], pred_means[~is_extreme], alpha=0.5, s=15, c='royalblue', label='Rango Medio')
    axs[0, 0].scatter(true_means[is_extreme], pred_means[is_extreme], alpha=0.6, s=20, c='crimson', marker='x', label='Extremos (>1.5$\sigma$)')
    
    min_val = min(true_means.min(), pred_means.min())
    max_val = max(true_means.max(), pred_means.max())
    axs[0, 0].plot([min_val, max_val], [min_val, max_val], 'k--', lw=2, label='Ideal') 
    axs[0, 0].set_xlabel('Valor real (mmHg)')
    axs[0, 0].set_ylabel('Predicción (mmHg)')
    axs[0, 0].grid(True, alpha=0.3)
    axs[0, 0].legend()

    means = (true_means + pred_means) / 2
    axs[0, 1].scatter(means, bias_per_patient, alpha=0.5, s=15, c='purple', edgecolors='k', linewidth=0.3)
    axs[0, 1].axhline(mean_bias, color='k', ls='-', lw=2, label=f'Bias: {mean_bias:.2f}')
    axs[0, 1].axhline(mean_bias + 1.96 * std_bias, color='r', ls='--', label=f'Lim: ±{1.96*std_bias:.1f}')
    axs[0, 1].axhline(mean_bias - 1.96 * std_bias, color='r', ls='--')
    axs[0, 1].set_xlabel('Promedio (mmHg)')
    axs[0, 1].set_ylabel('Diferencia (Pred - Real) (mmHg)')
    axs[0, 1].legend(loc='upper right')
    axs[0, 1].grid(True, alpha=0.3)

    if maes_pre is not None:
        axs[1, 0].scatter(maes_pre[~is_extreme], maes_post[~is_extreme], 
                          alpha=0.5, s=20, c='green', edgecolors='none', label=f'Rango Medio (n={n_normales})')
        axs[1, 0].scatter(maes_pre[is_extreme], maes_post[is_extreme], 
                          alpha=0.8, s=30, c='crimson', edgecolors='k', marker='^', label=f'Extremos (n={n_extremos})')

        limite = max(maes_pre.max(), maes_post.max()) + 2
        axs[1, 0].plot([0, limite], [0, limite], 'k--', lw=2, label='Sin Cambios')
        axs[1, 0].fill_between([0, limite], 0, [0, limite], color='green', alpha=0.05, label='Zona de Mejora')
        axs[1, 0].set_xlabel('MAE Inicial (Pre-Adaptación)')
        axs[1, 0].set_ylabel('MAE Final (Post-Adaptación)')
        axs[1, 0].set_xlim(0, limite)
        axs[1, 0].set_ylim(0, limite)
        axs[1, 0].legend()
        axs[1, 0].grid(True, alpha=0.3)
    else:
        axs[1, 0].text(0.5, 0.5, "Datos Pre no disponibles", ha='center', transform=axs[1, 0].transAxes)

    axs[1, 1].hist(maes_post, bins=30, color='orange', edgecolor='black', alpha=0.7)
    axs[1, 1].axvline(5, color='red', linestyle='dashed', linewidth=2, label='Umbral 5 mmHg')
    
    total_p = len(maes_post)
    dentro_5 = np.sum(maes_post <= 5)
    porcentaje = (dentro_5 / total_p) * 100
    
    texto_stats = f"Pacientes con MAE < 5mmHg:\n{dentro_5}/{total_p} ({porcentaje:.1f}%)"
    axs[1, 1].text(0.95, 0.85, texto_stats, transform=axs[1, 1].transAxes, 
                   fontsize=10, verticalalignment='top', horizontalalignment='right',
                   bbox=dict(boxstyle="round", facecolor="white", alpha=0.9))

    axs[1, 1].set_xlabel('MAE Final por Paciente (mmHg)')
    axs[1, 1].set_ylabel('Frecuencia')
    axs[1, 1].legend()
    
    label_style = dict(fontsize=18, fontweight='bold', color='black', 
                       va='top', ha='left',  # Alineación vertical top, horizontal left
                       bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=2))

    axs[0, 0].text(0.02, 0.96, 'a)', transform=axs[0, 0].transAxes, **label_style)
    axs[0, 1].text(0.02, 0.96, 'b)', transform=axs[0, 1].transAxes, **label_style)
    axs[1, 0].text(0.02, 0.96, 'c)', transform=axs[1, 0].transAxes, **label_style)
    axs[1, 1].text(0.02, 0.96, 'd)', transform=axs[1, 1].transAxes, **label_style)

    plt.tight_layout()
    plt.savefig(f'metalearning/pacientes_{titulo}_final_tradicional_4.png', dpi=300)
    print(f"Gráfico guardado: metalearning/pacientes_{titulo}_final_tradicional_4.png")

def main(n_shots=5, base_lr = 5e-3, base_dataset=None, test_patient_ids=None, is_delta_model=False):
    """_summary_ Función principal para ejecutar la evaluación few-shot fine-tuning del modelo de presión arterial. Carga el modelo pre-entrenado, prepara los datos de soporte y consulta para cada paciente, realiza la adaptación del modelo utilizando un número limitado de latidos (n_shots) y evalúa la mejora en la precisión de las predicciones antes y después de la adaptación. Además, calcula métricas globales y por paciente, y genera visualizaciones para analizar los resultados.

    Args:
        n_shots (int, optional): _description_. Por defecto 5. Indica el número de muestras de soporte (Support Set) utilizados para la adaptación few-shot de cada paciente. 
        base_lr (_type_, optional): _description_. Por defecto 5e-3. Tasa de aprendizaje utilizada durante el proceso de fine-tuning para adaptar el modelo a cada paciente específico.
        base_dataset (_type_, optional): _description_. Por defecto None. Dataset base que contiene los datos completos, necesario si se desea especificar un conjunto de pacientes de prueba personalizado. Si es None, se cargará el dataset completo desde las rutas predefinidas.
        test_patient_ids (_type_, optional): _description_. Por defecto None. Lista de IDs de pacientes reservados para la evaluación few-shot. Si es None, se cargarán los IDs desde un archivo preprocesado específico. Si se proporciona una lista personalizada, se utilizará esa lista para la evaluación.
        is_delta_model (bool, optional): _description_. Por defecto False. Indica si el modelo está configurado para predecir "deltas" respecto a un bias específico del paciente, lo que puede mejorar la estabilidad y rapidez de la adaptación en escenarios few-shot. Si es True, durante el fine-tuning se restará el bias normalizado de las etiquetas, y durante la evaluación se sumará nuevamente para obtener las predicciones finales en unidades reales.

    Returns:
        _type_: _description_ Diccionario con los resultados globales de la evaluación few-shot, incluyendo métricas de error (MAE, RMSE), métricas clínicas (Bias, STD), tasa de mejora por paciente, y un resumen detallado por paciente con sus respectivas métricas pre y post adaptación.
    """    
    SBP_MEAN = 134.02
    DBP_MEAN = 63.47
    SBP_STD = 22.75
    DBP_STD = 23.69
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if base_dataset is None:
        data_paths = [
        'data/processed/data_UCI/dataset_parte_1_por_picos.pt',
        'data/processed/data_UCI/dataset_parte_2_por_picos.pt',
        'data/processed/data_UCI/dataset_parte_3_por_picos.pt',
        'data/processed/data_UCI/dataset_parte_4_por_picos.pt'
        ]
        dataset_completo = UCIDataset(data_paths)
    else:
        dataset_completo = base_dataset

    if test_patient_ids is None:
        test_data = torch.load('data/processed/data_UCI/few_shot_patient_data.pt')
        test_patient_ids = test_data['test_patient_ids']

        print(f"Evaluando sobre {len(test_patient_ids)} pacientes reservados.")
    else: 
        test_patient_ids = test_patient_ids

    model = Modelo_ConvolucionalV1(in_channels=2, out_channels=2, long_signal=500)
    path_model = 'models/checkpoints/best_meta_model_v1.pt'
    
    print(f"Cargando modelo desde {path_model}...")
    checkpoint = torch.load(path_model, map_location=device, weights_only=False) 
    state_dict = checkpoint['model_state_dict']

    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('module.'):
            new_state_dict[k[7:]] = v
        else:
            new_state_dict[k] = v
  
    model.load_state_dict(new_state_dict)
    base_weights = model.state_dict()
    criterion = torch.nn.MSELoss()
    model = model.to(device) 

    taskset = TaskDataset(list_IDs=test_patient_ids, base_dataset=dataset_completo, num_shots=n_shots)

    global_errors_pre_SBP = []
    global_errors_post_SBP = []
    global_errors_pre_DBP = []
    global_errors_post_DBP = []
    global_metrics_pre_SBP, global_metrics_post_SBP = [], []
    global_metrics_pre_DBP, global_metrics_post_DBP = [], []

    means_true_sbp = []
    means_pred_sbp = []
    maes_sbp = [] 
    maes_pre_sbp = []

    means_true_dbp = []
    means_pred_dbp = []
    maes_dbp = []
    maes_pre_dbp = []

    mejoraron_sbp = 0
    empeoraron_sbp = 0
    mejoraron_dbp = 0
    empeoraron_dbp = 0

    resultados_por_paciente = []

    print(f"\nIniciando evaluación Few-Shot en {len(taskset.list_IDs)} pacientes...")

    for i in range(len(taskset.list_IDs)):
        id_paciente = taskset.list_IDs[i]
        
        model.load_state_dict(base_weights)
        for param in model.parameters():
            param.requires_grad = False
            
        for name, param in model.named_parameters():
            if 'dense' in name:
                param.requires_grad = True
                
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=base_lr)

        id_patient_for_tuning = taskset.list_IDs[i]
  
        tuningset_for_train = TuningNDataset(taskset, id_patient_for_tuning, n_shots=n_shots, validation=False)
        tuningset_for_valid = TuningNDataset(taskset, id_patient_for_tuning, validation=True)

        tuning_loader_TRAIN = torch.utils.data.DataLoader(tuningset_for_train, batch_size=1, shuffle=False)
        tuning_loader_VALID = torch.utils.data.DataLoader(tuningset_for_valid, batch_size=1, shuffle=False)

        model.eval()
        preds_pre, loss_pre = [], []
        for batch in tuning_loader_VALID:
            p, l = evaluation(batch, model, criterion, device)
            preds_pre.extend(p.detach().cpu().numpy())  
            loss_pre.extend([l.item()]*len(p))
        preds_pre = np.array(preds_pre)

        labels_soporte_norm = [lbls.squeeze().cpu().numpy() for _, lbls, *_ in tuning_loader_TRAIN]
        labels_soporte_norm = np.array(labels_soporte_norm).reshape(-1, 2)
        bias_norm = np.mean(labels_soporte_norm, axis=0) # Shape: (2,)
        
        model.train()
        tuning_loss = np.zeros(shape=n_shots)
        for shot_idx, sample in enumerate(tuning_loader_TRAIN):
            tuning_loss[shot_idx] = tuning(sample, optimizer, model, criterion, device, bias_norm if is_delta_model else None)
    
        model.eval()
        preds_post, loss_post = [], []
        for batch in tuning_loader_VALID:
            p, l = evaluation(batch, model, criterion, device)
            preds_post.extend(p.detach().cpu().numpy())
            loss_post.extend([l.item()]*len(p))
        preds_post = np.array(preds_post)

        pred_pre_flat = preds_pre.reshape(-1, 2)
        pred_post_flat = preds_post.reshape(-1, 2)
        labels_flat = np.array([l.squeeze().cpu().numpy() for l in tuning_loader_VALID.dataset.labels]).reshape(-1, 2)

        if is_delta_model:
            pred_pre_flat += bias_norm
            pred_post_flat += bias_norm

        pred_pre_SBP = desnormalizar_zscore(pred_pre_flat[:,0], SBP_MEAN, SBP_STD)
        pred_pre_DBP = desnormalizar_zscore(pred_pre_flat[:,1], DBP_MEAN, DBP_STD)
        
        pred_post_SBP = desnormalizar_zscore(pred_post_flat[:,0], SBP_MEAN, SBP_STD)
        pred_post_DBP = desnormalizar_zscore(pred_post_flat[:,1], DBP_MEAN, DBP_STD)
        
        true_SBP = desnormalizar_zscore(labels_flat[:,0], SBP_MEAN, SBP_STD)
        true_DBP = desnormalizar_zscore(labels_flat[:,1], DBP_MEAN, DBP_STD)

        global_errors_pre_SBP.extend(pred_pre_SBP - true_SBP)
        global_errors_post_SBP.extend(pred_post_SBP - true_SBP)
        global_errors_pre_DBP.extend(pred_pre_DBP - true_DBP)
        global_errors_post_DBP.extend(pred_post_DBP - true_DBP)
        
        m_pre_sbp = calcular_metricas(true_SBP, pred_pre_SBP)
        m_post_sbp = calcular_metricas(true_SBP, pred_post_SBP)
        m_pre_dbp = calcular_metricas(true_DBP, pred_pre_DBP)
        m_post_dbp = calcular_metricas(true_DBP, pred_post_DBP)

        global_metrics_pre_SBP.append(m_pre_sbp)
        global_metrics_post_SBP.append(m_post_sbp)
        global_metrics_pre_DBP.append(m_pre_dbp)
        global_metrics_post_DBP.append(m_post_dbp)

        means_true_sbp.append(np.mean(true_SBP))
        means_pred_sbp.append(np.mean(pred_post_SBP))
        maes_sbp.append(m_post_sbp[0]) # MAE del paciente
        maes_pre_sbp.append(m_pre_sbp[0])

        means_true_dbp.append(np.mean(true_DBP))
        means_pred_dbp.append(np.mean(pred_post_DBP))
        maes_dbp.append(m_post_dbp[0])
        maes_pre_dbp.append(m_pre_dbp[0])

        if m_post_sbp[0] < m_pre_sbp[0]: mejoraron_sbp += 1
        else: empeoraron_sbp += 1

        if m_post_dbp[0] < m_pre_dbp[0]: mejoraron_dbp += 1
        else: empeoraron_dbp += 1

        resultados_por_paciente.append({
            "paciente": int(id_paciente),
            'mae_pre_sbp': float(m_pre_sbp[0]),
            'mae_post_sbp': float(m_post_sbp[0]),
            'rmse_pre_sbp': float(m_pre_sbp[1]),
            'rmse_post_sbp': float(m_post_sbp[1]),
            'iso_bias_pre_sbp': float(m_pre_sbp[2]),
            'iso_bias_post_sbp': float(m_post_sbp[2]), 
            'iso_std_pre_sbp': float(m_pre_sbp[3]),
            'iso_std_post_sbp': float(m_post_sbp[3]),
            'mae_pre_dbp': float(m_pre_dbp[0]),
            'mae_post_dbp': float(m_post_dbp[0]),
            'rmse_pre_dbp': float(m_pre_dbp[1]),
            'rmse_post_dbp': float(m_post_dbp[1]),
            'iso_bias_pre_dbp': float(m_pre_dbp[2]),
            'iso_bias_post_dbp': float(m_post_dbp[2]), 
            'iso_std_pre_dbp': float(m_pre_dbp[3]),
            'iso_std_post_dbp': float(m_post_dbp[3]),
        })

        print(f"Paciente {id_patient_for_tuning} | SBP MAE: {m_pre_sbp[0]:.2f}->{m_post_sbp[0]:.2f} | RMSE: {m_pre_sbp[1]:.2f}->{m_post_sbp[1]:.2f} | ISO: {m_pre_sbp[2]:.2f}±{m_pre_sbp[3]:.2f} -> {m_post_sbp[2]:.2f}±{m_post_sbp[3]:.2f}")

    err_pre_sbp_all = np.array(global_errors_pre_SBP)
    err_post_sbp_all = np.array(global_errors_post_SBP)
    err_pre_dbp_all = np.array(global_errors_pre_DBP)
    err_post_dbp_all = np.array(global_errors_post_DBP)
    
    mae_pre_sbp_global = np.mean(np.abs(err_pre_sbp_all))
    mae_post_sbp_global = np.mean(np.abs(err_post_sbp_all))
    mae_pre_dbp_global = np.mean(np.abs(err_pre_dbp_all))
    mae_post_dbp_global = np.mean(np.abs(err_post_dbp_all))

    rmse_post_sbp_global = np.sqrt(np.mean(np.square(err_post_sbp_all)))
    rmse_post_dbp_global = np.sqrt(np.mean(np.square(err_post_dbp_all)))

    std_post_sbp_global = np.std(err_post_sbp_all)
    std_post_dbp_global = np.std(err_post_dbp_all)

    bias_post_sbp_global = np.mean(err_post_sbp_all)
    bias_post_dbp_global = np.mean(err_post_dbp_all)
    
    total = len(taskset.list_IDs)
    tasa_mejora_sbp = mejoraron_sbp / total
    tasa_mejora_dbp = mejoraron_dbp / total
    
    print("\n" + "="*60)
    print("       RESULTADOS FINALES GLOBAL (Calculados sobre todos los latidos)")
    print("="*60)
    
    print("\n--- SISTÓLICA (SBP) ---")
    print(f"Ingeniería (MAE)   : {mae_post_sbp_global:.2f} mmHg")
    print(f"Ingeniería (RMSE)  : {rmse_post_sbp_global:.2f} mmHg")
    print(f"RESUMEN ISO FINAL  : {bias_post_sbp_global:.2f} ± {std_post_sbp_global:.2f} mmHg (Meta: <= 5 ± 8)")
    
    check_rmse = np.sqrt(bias_post_sbp_global**2 + std_post_sbp_global**2)
    print(f"  [Chequeo: sqrt(Bias^2 + STD^2) = {check_rmse:.2f} vs RMSE = {rmse_post_sbp_global:.2f}] -> ¡Cuadran!")

    print("\n--- DIASTÓLICA (DBP) ---")
    print(f"Ingeniería (MAE)   : {mae_post_dbp_global:.2f} mmHg")
    print(f"Ingeniería (RMSE)  : {rmse_post_dbp_global:.2f} mmHg")
    print(f"RESUMEN ISO FINAL  : {bias_post_dbp_global:.2f} ± {std_post_dbp_global:.2f} mmHg (Meta: <= 5 ± 8)")

    print("\n--- CONSISTENCIA ---")
    print(f"Tasa Mejora SBP: {(tasa_mejora_sbp)*100:.1f}% ({mejoraron_sbp}/{total})")
    print(f"Tasa Mejora DBP: {(tasa_mejora_dbp)*100:.1f}% ({mejoraron_dbp}/{total})")
    
    graficar_resultados_pacientes(means_true_sbp, means_pred_sbp, maes_sbp, maes_pre_sbp, titulo="SBP")
    graficar_resultados_pacientes(means_true_dbp, means_pred_dbp, maes_dbp, maes_pre_dbp, titulo="DBP")
     
    resultados = {
        "mae_pre_sbp": mae_pre_sbp_global,
        "mae_post_sbp": mae_post_sbp_global,
        "rmse_post_sbp": rmse_post_sbp_global, 
        "std_post_sbp": std_post_sbp_global,    
        "tasa_mejora_sbp": tasa_mejora_sbp,
        "mejoraron sbp": mejoraron_sbp,     
        "empeoraron sbp": empeoraron_sbp,   
        
        "mae_pre_dbp": mae_pre_dbp_global,
        "mae_post_dbp": mae_post_dbp_global,
        "rmse_post_dbp": rmse_post_dbp_global, 
        "std_post_dbp": std_post_dbp_global,    
        "tasa_mejora_dbp": tasa_mejora_dbp,
        "mejoraron dbp": mejoraron_dbp,
        "empeoraron dbp": empeoraron_dbp,

        "resultados_por_paciente": resultados_por_paciente
    }
    return resultados

if __name__ == '__main__':
    main(is_delta_model=False)