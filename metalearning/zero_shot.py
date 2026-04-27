"""
Módulo: Zero_shot.py
Autor: Juan Marcos Grigolatto
Descripción: Script de validación del metamodelo con pesos congelados (Zero-Shot). 
             Evalúa el rendimiento metamodelo general de estimación de presión arterial sin ningún
             tipo de ajuste previo sobre pacientes no vistos. Permite seleccionar 
             mediante configuración si se testea el modelo tradicional o  Delta Learning.
"""
import os
import random
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error
import torch
import torch.utils.data as data
from src.data.data_chargers.Clase_UCIDataset import UCIDataset
from src.data.data_chargers.MetaDataset import TaskDataset
from src.data.data_chargers.Intrapatientset import Intrapatientset
from src.models.ConvolucionalV1 import Modelo_ConvolucionalV1

def calcular_metricas_avanzadas(y_true, y_pred):
    """_summary_ Calcula métricas avanzadas de error: MAE, RMSE, Bias y Desviación Estándar de los errores.

    Args:
        y_true (_type_): _description_ Valores reales de presión arterial (mmHg)
        y_pred (_type_): _description_ Valores predichos por el modelo (mmHg)

    Returns:
        _type_: _description_ Tupla con las métricas: (MAE, RMSE, Bias, SD)
    """    
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    errores = y_pred - y_true
    mae = np.mean(np.abs(errores))
    rmse = np.sqrt(np.mean(errores**2))
    bias = np.mean(errores)
    sd = np.std(errores)    
    return mae, rmse, bias, sd

def desnormalizar_zscore(norm_array, media, std):
    """_summary_ Desnormaliza un array normalizado con Z-score a sus valores originales en mmHg.

    Args:
        norm_array (_type_): _description_ Array normalizado (Z-score)
        media (_type_): _description_ Media utilizada para la normalización (SBP_MEAN o DBP_MEAN)
        std (_type_): _description_ Desviación estándar utilizada para la normalización (SBP_STD o DBP_STD)

    Returns:
        _type_: _description_ Valores desnormalizados en mmHg
    """    
    return norm_array * std + media

def evaluation(batch, model, device):
    """_summary_ Realiza la inferencia del modelo sobre un batch de datos, sin actualizar los pesos (modo evaluación).

    Args:
        batch (_type_): _description_ Batch de datos que contiene señales y etiquetas (SBP, DBP) normalizadas.
        model (_type_): _description_ Modelo de red neuronal previamente entrenado y cargado.
        device (_type_): _description_ Dispositivo de cómputo (CPU o GPU) donde se realizará la inferencia.

    Returns:
        _type_: _description_ Predicciones del modelo para el batch de entrada, en formato tensor (normalizado).
    """    
    with torch.no_grad():
        batch_data, labels, *_ = batch
        if isinstance(batch_data, list): batch_data = batch_data[0]
        batch_data = batch_data.to(device)
        preds = model.forward(batch_data)
    return preds

def main(n_shots=5):
    """_summary_ Función principal que ejecuta la evaluación zero-shot del modelo sobre pacientes no vistos. Permite configurar el número de shots y el tipo de modelo (Delta Learning o Tradicional) mediante variables internas.

    Args:
        n_shots (int, optional): _description_. Por defecto 5. Número de muestras (shots) utilizadas para la evaluación intrapatient, aunque el modelo no se ajusta con ellas (zero-shot), se muestran en las gráficas para referencia.
    """    
    IS_DELTA_MODEL = False 
    
    PACIENTES_OBJETIVO = [101, 2041, 8423, 1126] 
    
    if IS_DELTA_MODEL:
        NOMBRE_EXPERIMENTO = "ZEROSHOT_patient_HIBRIDO"
        PATH_MODELO = 'models/checkpoints/best_meta_DELTA_LEARNING_refine_alpha50.pt'
    else:
        NOMBRE_EXPERIMENTO = "ZEROSHOT_TRADICIONAL_tradicional"
        PATH_MODELO = 'models/checkpoints/best_meta_model_v1.pt'

    print(f"--- INICIANDO EVALUACIÓN ZERO-SHOT ---")
    print(f"Modo Delta Learning: {IS_DELTA_MODEL}")
    print(f"Modelo: {PATH_MODELO}")

    SEED = 42
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
    if torch.cuda.is_available(): torch.cuda.manual_seed(SEED)

    save_dir_graficas = f"resultados_intrapatient/{NOMBRE_EXPERIMENTO}"
    os.makedirs(save_dir_graficas, exist_ok=True)
    
    SBP_MEAN, SBP_STD = 134.02, 22.75
    DBP_MEAN, DBP_STD = 63.47, 23.69
    
    # Carga de datos
    data_paths = [
        f'data/processed/data_UCI/dataset_parte_{i}_por_picos.pt' for i in range(1, 5)
    ]
    dataset_completo = UCIDataset(data_paths)

    # Carga del Modelo
    model = Modelo_ConvolucionalV1(in_channels=2, out_channels=2, long_signal=500)
    checkpoint = torch.load(PATH_MODELO, map_location=torch.device('cpu'), weights_only=False)
    state_dict = checkpoint['model_state_dict']
    new_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    model.load_state_dict(new_state_dict, strict=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    taskset = TaskDataset(list_IDs=PACIENTES_OBJETIVO, base_dataset=dataset_completo, num_shots=n_shots)
    mapa_indices_pacientes = taskset.patient_to_indices
    pacientes_seleccionados = [pid for pid in PACIENTES_OBJETIVO if pid in mapa_indices_pacientes]

    for id_paciente in pacientes_seleccionados:
        print(f"\n >> EVALUANDO PACIENTE: {id_paciente}")
        
        dataset_paciente = Intrapatientset(
            patient_id=id_paciente, base_dataset=dataset_completo,
            patient_to_indices_map=mapa_indices_pacientes
        )
        loader_paciente = torch.utils.data.DataLoader(dataset_paciente, batch_size=n_shots, shuffle=False)

        historial_sbp = {'real': [], 'pred': []}
        historial_dbp = {'real': [], 'pred': []}
        
        bias_norm = None

        for i, (batch_signals, batch_labels) in enumerate(loader_paciente):
            
            if i == 0 and IS_DELTA_MODEL:
                bias_norm = batch_labels.mean(dim=0, keepdim=True).numpy()

            preds_raw = evaluation((batch_signals, batch_labels), model, device)
            preds_numpy = preds_raw.detach().cpu().numpy()

            if IS_DELTA_MODEL:
                preds_final = preds_numpy + bias_norm
            else:
                preds_final = preds_numpy

            p_sbp = desnormalizar_zscore(preds_final[:, 0], SBP_MEAN, SBP_STD)
            p_dbp = desnormalizar_zscore(preds_final[:, 1], DBP_MEAN, DBP_STD)
            t_sbp = desnormalizar_zscore(batch_labels[:, 0].numpy(), SBP_MEAN, SBP_STD)
            t_dbp = desnormalizar_zscore(batch_labels[:, 1].numpy(), DBP_MEAN, DBP_STD)

            historial_sbp['real'].extend(t_sbp); historial_sbp['pred'].extend(p_sbp)
            historial_dbp['real'].extend(t_dbp); historial_dbp['pred'].extend(p_dbp)

        mae_s, rmse_s, bias_s, sd_s = calcular_metricas_avanzadas(historial_sbp['real'], historial_sbp['pred'])
        
        plt.rcParams.update({'font.size': 12, 'font.family': 'serif'})
        fig, axs = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
        x_axis = range(len(historial_sbp['real']))
        
        color_linea = 'tab:green' if IS_DELTA_MODEL else 'tab:red'
        etiqueta_mod = "MAPA Híbrido (Delta)" if IS_DELTA_MODEL else "Patient-wise (Tradicional)"

        axs[0].plot(x_axis, historial_sbp['real'], color='black', linewidth=1.5, label='Invasiva (Real)', alpha=0.8)
        axs[0].plot(x_axis, historial_sbp['pred'], color=color_linea, linestyle='--', linewidth=2, label=etiqueta_mod, alpha=0.9)
        if IS_DELTA_MODEL:
            axs[0].axvline(x=n_shots, color='green', linestyle=':', linewidth=2, label='Calibración Aritmética')
        
        axs[0].set_title(f"Paciente {id_paciente} - SBP Zero-Shot (RMSE: {rmse_s:.2f} mmHg)", fontweight='bold')
        axs[0].set_ylabel("Presión (mmHg)")
        axs[0].legend(loc='upper right')
        axs[0].grid(True, linestyle='--', alpha=0.5)

        axs[1].plot(x_axis, historial_dbp['real'], color='black', linewidth=1.5, label='Invasiva (Real)', alpha=0.8)
        axs[1].plot(x_axis, historial_dbp['pred'], color='tab:blue', linestyle='--', linewidth=2, label=etiqueta_mod, alpha=0.9)
        if IS_DELTA_MODEL:
            axs[1].axvline(x=n_shots, color='green', linestyle=':', linewidth=2)
            
        axs[1].set_xlabel("Latidos")
        axs[1].set_ylabel("Presión (mmHg)")
        axs[1].grid(True, linestyle='--', alpha=0.5)
        
        plt.tight_layout()
        save_name = f"zeroshot_{'delta' if IS_DELTA_MODEL else 'trad'}_{id_paciente}.png"
        plt.savefig(os.path.join(save_dir_graficas, save_name), dpi=300, bbox_inches='tight')
        plt.close()

    print(f"\n--- EVALUACIÓN FINALIZADA. Gráficas en: {save_dir_graficas} ---")

if __name__ == '__main__':
    main()