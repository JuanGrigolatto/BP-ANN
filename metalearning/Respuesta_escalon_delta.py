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
# --- FUNCIONES AUXILIARES Y DE FEW-SHOT ---
# ==============================================================================

def calcular_metricas_avanzadas(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    errores = y_pred - y_true
    errores_abs = np.abs(errores)
    
    mae = np.mean(errores_abs)
    mse = np.mean(errores**2)
    rmse = np.sqrt(mse) 
    bias = np.mean(errores) 
    sd = np.std(errores)    
    return mae, rmse, bias, sd

def desnormalizar_zscore(norm_array, media, std):
    return norm_array * std + media

def tuning(sample, optimizer, model, criterion, device, bias_norm=None):
    # Dejamos la función por si se necesita para otros experimentos,
    # pero NO se usará en esta prueba dinámica para evitar el colapso de pesos.
    optimizer.zero_grad() 
    batch_data, labels, *_ = sample 
    
    if isinstance(batch_data, list): batch_data = batch_data[0]
    if isinstance(labels, list): labels = labels[0]
        
    for layer in model.modules():
        if isinstance(layer, torch.nn.BatchNorm1d):
            layer.eval()
            layer.weight.requires_grad = False
            layer.bias.requires_grad = False

    batch_data, labels = batch_data.to(device), labels.to(device) 
    
    if bias_norm is not None:
        bias_tensor = torch.tensor(bias_norm, dtype=torch.float32, device=device)
        labels = labels - bias_tensor

    preds = model.forward(batch_data) 
    loss = criterion(preds, labels) 
    loss.backward() 
    optimizer.step() 
    return loss.item()

def evaluation(batch, model, criterion, device):
    with torch.no_grad():
        batch_data, labels, *_ = batch
        if isinstance(batch_data, list): batch_data = batch_data[0]
        if isinstance(labels, list): labels = labels[0]
        batch_data, labels = batch_data.to(device), labels.to(device)
        preds = model.forward(batch_data)
        loss = criterion(preds, labels)
    return preds, loss

# ==============================================================================
# --- SCRIPT PRINCIPAL ---
# ==============================================================================

def main(n_shots=5, n_epochs=5, lr=5e-3, MIN_SEÑALES_REQUERIDAS=500):
    
    # --- CONFIGURACIÓN DE LA PRUEBA DE ESTRÉS ---
    MINUTOS_DESEADOS = 60  
    PACIENTES_OBJETIVO = [101, 2041, 8423, 1126] 
    NOMBRE_EXPERIMENTO = "PRUEBA_ESCALON_SIN_COLAPSO"
    PATH_MODELO = 'models/checkpoints/best_meta_DELTA_LEARNING_refine_alpha50.pt'

    print(f"--- INICIANDO PRUEBA DE RESPUESTA INTRA-PACIENTE (Cálculo Aritmético de Offset) ---")
    print(f"Modelo: {PATH_MODELO}")
    print(f"Pacientes Objetivo: {PACIENTES_OBJETIVO}")

    SEED = 42
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
    if torch.cuda.is_available(): torch.cuda.manual_seed(SEED)

    save_dir_graficas = f"resultados_intrapatient/{NOMBRE_EXPERIMENTO}"
    os.makedirs(save_dir_graficas, exist_ok=True)
    
    SBP_MEAN, SBP_STD = 134.02, 22.75
    DBP_MEAN, DBP_STD = 63.47, 23.69
    
    SEGUNDOS_POR_LOTE = (500 / 125) * n_shots 
    intervalo_ajuste = int((MINUTOS_DESEADOS * 60) / SEGUNDOS_POR_LOTE)
    print(f"Intervalo de inflado forzado cada {intervalo_ajuste} lotes.")

    # Carga de datos
    data_paths = [
        'data/processed/data_UCI/dataset_parte_1_por_picos.pt',
        'data/processed/data_UCI/dataset_parte_2_por_picos.pt',
        'data/processed/data_UCI/dataset_parte_3_por_picos.pt',
        'data/processed/data_UCI/dataset_parte_4_por_picos.pt'
    ]
    dataset_completo = UCIDataset(data_paths)

    # Carga del Modelo
    model = Modelo_ConvolucionalV1(in_channels=2, out_channels=2, long_signal=500)
    if not os.path.exists(PATH_MODELO):
        print(f"¡ERROR! No encuentro el modelo en {PATH_MODELO}")
        return

    checkpoint = torch.load(PATH_MODELO, map_location=torch.device('cpu'), weights_only=False)
    state_dict = checkpoint['model_state_dict']
    new_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    model.load_state_dict(new_state_dict, strict=False)

    criterion = torch.nn.MSELoss() 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)  

    taskset = TaskDataset(list_IDs=PACIENTES_OBJETIVO, base_dataset=dataset_completo, num_shots=n_shots)
    mapa_indices_pacientes = taskset.patient_to_indices

    pacientes_seleccionados = [pid for pid in PACIENTES_OBJETIVO if pid in mapa_indices_pacientes]
    print(f"Pacientes encontrados y listos para procesar: {pacientes_seleccionados}")

    for id_paciente in pacientes_seleccionados:
        print(f"\n >> PROCESANDO PACIENTE: {id_paciente}")
        
        # Reset del modelo al estado base
        model.load_state_dict(new_state_dict, strict=False)
        
        dataset_paciente_completo = Intrapatientset(
            patient_id=id_paciente,
            base_dataset=dataset_completo,
            patient_to_indices_map=mapa_indices_pacientes
        )

        loader_paciente = torch.utils.data.DataLoader(
            dataset_paciente_completo, batch_size=n_shots, shuffle=False, num_workers=0, drop_last=False
        )

        historial_sbp = {'real': [], 'pred': []}
        historial_dbp = {'real': [], 'pred': []}
        puntos_de_ajuste_x = []
        
        # Variable crítica: Memoria del manguito
        bias_norm = None

        for i, (batch_signals, batch_labels) in enumerate(loader_paciente):
            batch_data = (batch_signals, batch_labels)

            # =================================================================
            # 1. CALIBRACIÓN ARITMÉTICA (Simulación del inflado del manguito)
            # =================================================================
            if i == 0 or (i + 1) % intervalo_ajuste == 0:
                
                # Extraemos el nivel basal de estos latidos reales
                labels_numpy = batch_labels.numpy()
                bias_norm = np.mean(labels_numpy, axis=0) # [bias_sbp, bias_dbp]
                
                # ¡SE ELIMINÓ EL ENTRENAMIENTO AQUÍ!
                # La red ya aprendió a leer morfologías en la fase de meta-entrenamiento.
                # Actualizar pesos con 5 latidos idénticos colapsaba la red a 0.
                
                # Marcador para el gráfico (se grafica en número de latidos, no lotes)
                puntos_de_ajuste_x.append((i) * n_shots)

            # =================================================================
            # 2. MONITOREO CONTINUO (Inferencia Zero-Shot de pesos)
            # =================================================================
            model.eval()
            preds, _ = evaluation(batch_data, model, criterion, device) 
            preds_numpy = preds.detach().cpu().numpy()
            
            # COMPENSACIÓN: A la variación que predijo la red (Delta), 
            # le sumamos el "Offset" del manguito.
            if bias_norm is not None:
                preds_numpy += bias_norm

            # Desnormalizamos para volver a mmHg
            pred_sbp = desnormalizar_zscore(preds_numpy[:, 0], SBP_MEAN, SBP_STD)
            pred_dbp = desnormalizar_zscore(preds_numpy[:, 1], DBP_MEAN, DBP_STD)
            true_sbp = desnormalizar_zscore(batch_labels[:, 0].numpy(), SBP_MEAN, SBP_STD)
            true_dbp = desnormalizar_zscore(batch_labels[:, 1].numpy(), DBP_MEAN, DBP_STD)

            historial_sbp['real'].extend(true_sbp); historial_sbp['pred'].extend(pred_sbp)
            historial_dbp['real'].extend(true_dbp); historial_dbp['pred'].extend(pred_dbp)

        # --- Métricas y Gráficas ---
        mae_s, rmse_s, bias_s, std_s = calcular_metricas_avanzadas(historial_sbp['real'], historial_sbp['pred'])
        mae_d, rmse_d, bias_d, std_d = calcular_metricas_avanzadas(historial_dbp['real'], historial_dbp['pred'])

        print(f"   [SBP] RMSE: {rmse_s:.2f} | Bias: {bias_s:.2f}")
        print(f"   [DBP] RMSE: {rmse_d:.2f} | Bias: {bias_d:.2f}")
        
        # Plotting
        fig, axs = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
        x_axis = range(len(historial_sbp['real']))
        
        # SBP
        axs[0].plot(x_axis, historial_sbp['real'], 'k', label='Real (Invasiva)', alpha=0.7)
        axs[0].plot(x_axis, historial_sbp['pred'], 'r--', label='Estimado (MAPA Híbrido)', alpha=0.9)
        for x_pos in puntos_de_ajuste_x:
            axs[0].axvline(x=x_pos, color='green', linestyle='-', linewidth=2, alpha=0.7, label='Calibración (Manguito)' if x_pos == 0 else "")
        axs[0].set_title(f"Paciente {id_paciente} - SBP Continua (RMSE Global: {rmse_s:.2f} mmHg)")
        axs[0].set_ylabel("Presión (mmHg)")
        axs[0].legend(loc='upper right')
        axs[0].grid(True, linestyle='--', alpha=0.5)

        # DBP
        axs[1].plot(x_axis, historial_dbp['real'], 'k', label='Real (Invasiva)', alpha=0.7)
        axs[1].plot(x_axis, historial_dbp['pred'], 'b--', label='Estimado (MAPA Híbrido)', alpha=0.9)
        for x_pos in puntos_de_ajuste_x:
            axs[1].axvline(x=x_pos, color='green', linestyle='-', linewidth=2, alpha=0.7)
        axs[1].set_title(f"Paciente {id_paciente} - DBP Continua (RMSE Global: {rmse_d:.2f} mmHg)")
        axs[1].set_xlabel("Latido Cardíaco N°")
        axs[1].set_ylabel("Presión (mmHg)")
        axs[1].legend(loc='upper right')
        axs[1].grid(True, linestyle='--', alpha=0.5)
        
        plt.tight_layout()
        save_path = os.path.join(save_dir_graficas, f"step_response_hibrido_{id_paciente}.png")
        plt.savefig(save_path, dpi=300)
        plt.close()
        print(f"   Gráfica guardada: {save_path}")

    print("\n--- PRUEBA DINÁMICA FINALIZADA ---")

if __name__ == '__main__':
    main()