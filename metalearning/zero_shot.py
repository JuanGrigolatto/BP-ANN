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

# ==============================================================================
# --- FUNCIONES AUXILIARES ---
# ==============================================================================

def calcular_metricas_avanzadas(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    errores = y_pred - y_true
    mae = np.mean(np.abs(errores))
    rmse = np.sqrt(np.mean(errores**2))
    bias = np.mean(errores)
    sd = np.std(errores)    
    return mae, rmse, bias, sd

def desnormalizar_zscore(norm_array, media, std):
    return norm_array * std + media

def evaluation(batch, model, device):
    with torch.no_grad():
        batch_data, labels, *_ = batch
        if isinstance(batch_data, list): batch_data = batch_data[0]
        batch_data = batch_data.to(device)
        preds = model.forward(batch_data)
    return preds

# ==============================================================================
# --- SCRIPT PRINCIPAL ---
# ==============================================================================

def main(n_shots=5):
    
    # ---------------------------------------------------------
    # CONFIGURACIÓN DEL EXPERIMENTO
    # ---------------------------------------------------------
    IS_DELTA_MODEL = False # <--- CAMBIAR AQUÍ: True (Híbrido) | False (Patient-wise Tradicional)
    
    PACIENTES_OBJETIVO = [101, 2041, 8423, 1126] 
    
    if IS_DELTA_MODEL:
        NOMBRE_EXPERIMENTO = "ZEROSHOT_patient_HIBRIDO"
        PATH_MODELO = 'models/checkpoints/best_meta_DELTA_LEARNING_refine_alpha50.pt'
    else:
        NOMBRE_EXPERIMENTO = "ZEROSHOT_TRADICIONAL_tradicional"
        # Usamos tu modelo de Patient-wise que no hace delta
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
        
        # Para el modelo Delta, necesitamos el anclaje inicial
        bias_norm = None

        for i, (batch_signals, batch_labels) in enumerate(loader_paciente):
            
            # 1. Si es modelo Delta, capturamos el Bias en el primer lote
            if i == 0 and IS_DELTA_MODEL:
                bias_norm = batch_labels.mean(dim=0, keepdim=True).numpy()

            # 2. Inferencia Zero-Shot (Pesos congelados)
            preds_raw = evaluation((batch_signals, batch_labels), model, device)
            preds_numpy = preds_raw.detach().cpu().numpy()

            # 3. Reconstrucción según el tipo de modelo
            if IS_DELTA_MODEL:
                # Delta + Offset Aritmético
                preds_final = preds_numpy + bias_norm
            else:
                # Predicción directa (Patient-wise tradicional)
                preds_final = preds_numpy

            # Desnormalización a mmHg
            p_sbp = desnormalizar_zscore(preds_final[:, 0], SBP_MEAN, SBP_STD)
            p_dbp = desnormalizar_zscore(preds_final[:, 1], DBP_MEAN, DBP_STD)
            t_sbp = desnormalizar_zscore(batch_labels[:, 0].numpy(), SBP_MEAN, SBP_STD)
            t_dbp = desnormalizar_zscore(batch_labels[:, 1].numpy(), DBP_MEAN, DBP_STD)

            historial_sbp['real'].extend(t_sbp); historial_sbp['pred'].extend(p_sbp)
            historial_dbp['real'].extend(t_dbp); historial_dbp['pred'].extend(p_dbp)

        # --- Gráficas Homogéneas Calidad Tesis ---
        mae_s, rmse_s, bias_s, sd_s = calcular_metricas_avanzadas(historial_sbp['real'], historial_sbp['pred'])
        
        plt.rcParams.update({'font.size': 12, 'font.family': 'serif'})
        fig, axs = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
        x_axis = range(len(historial_sbp['real']))
        
        color_linea = 'tab:green' if IS_DELTA_MODEL else 'tab:red'
        etiqueta_mod = "MAPA Híbrido (Delta)" if IS_DELTA_MODEL else "Patient-wise (Tradicional)"

        # SBP
        axs[0].plot(x_axis, historial_sbp['real'], color='black', linewidth=1.5, label='Invasiva (Real)', alpha=0.8)
        axs[0].plot(x_axis, historial_sbp['pred'], color=color_linea, linestyle='--', linewidth=2, label=etiqueta_mod, alpha=0.9)
        if IS_DELTA_MODEL:
            axs[0].axvline(x=n_shots, color='green', linestyle=':', linewidth=2, label='Calibración Aritmética')
        
        axs[0].set_title(f"Paciente {id_paciente} - SBP Zero-Shot (RMSE: {rmse_s:.2f} mmHg)", fontweight='bold')
        axs[0].set_ylabel("Presión (mmHg)")
        axs[0].legend(loc='upper right')
        axs[0].grid(True, linestyle='--', alpha=0.5)

        # DBP
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