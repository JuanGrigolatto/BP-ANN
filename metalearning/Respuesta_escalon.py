import torch.utils
from src.data.data_chargers.Clase_UCIDataset import UCIDataset
from src.data.data_chargers.MetaDataset import TaskDataset
from src.models.ConvolucionalV1 import Modelo_ConvolucionalV1
import numpy as np
import torch.utils.data as data
import torch
import random
import metalearning.Fewshot as Fewshot
from src.data.data_chargers.Intrapatientset import Intrapatientset
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
import os

# --- FUNCIONES AUXILIARES ---
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

def main(n_shots=5, n_epochs=5, lr=5e-3, MIN_SEÑALES_REQUERIDAS=500):
    
    # --- CONFIGURACIÓN DE LA PRUEBA DE ESTRÉS ---
    MINUTOS_DESEADOS = 60  # <--- ALTO para evitar re-ajustes y ver la respuesta pura
    PACIENTES_OBJETIVO = [101, 2041, 8423, 1126] # <--- LOS "DIFÍCILES"
    NOMBRE_EXPERIMENTO = "PRUEBA_ESCALON"
    
    # Ruta del modelo REFINADO (Fase 2)
    PATH_MODELO = 'models/checkpoints/best_meta_DELTA_LEARNING_refine_alpha50.pt'

    print(f"--- INICIANDO PRUEBA DE RESPUESTA AL ESCALÓN ---")
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
    print(f"Intervalo de ajuste forzado cada {intervalo_ajuste} lotes (prácticamente nunca).")

    # Carga de datos
    test_data = torch.load('data/processed/data_UCI/few_shot_patient_data.pt')
    test_patient_ids = test_data['test_patient_ids']
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

    checkpoint = torch.load(PATH_MODELO, map_location=torch.device('cpu'))
    state_dict = checkpoint['model_state_dict']
    new_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    model.load_state_dict(new_state_dict, strict=False)

    criterion = torch.nn.MSELoss() # Usamos MSE para el tuning (mejor para calibrar valores)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)  

    taskset = TaskDataset(list_IDs=PACIENTES_OBJETIVO, base_dataset=dataset_completo, num_shots=n_shots)
    mapa_indices_pacientes = taskset.patient_to_indices

    # Verificar disponibilidad de los pacientes objetivo
    pacientes_seleccionados = [pid for pid in PACIENTES_OBJETIVO if pid in mapa_indices_pacientes]
    print(f"Pacientes encontrados y listos para procesar: {pacientes_seleccionados}")

    resultados_finales_experimento = {}

    for id_paciente in pacientes_seleccionados:
        print(f" >> PROCESANDO PACIENTE: {id_paciente}")
        
        # Reset del modelo al estado base para cada paciente
        model.load_state_dict(new_state_dict, strict=False)
        
        # Body Freezing (Igual que antes)
        for param in model.parameters(): param.requires_grad = False
        for name, param in model.named_parameters():
            if 'dense' in name or 'conv4' in name: param.requires_grad = True

        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
        
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

        for i, (batch_signals, batch_labels) in enumerate(loader_paciente):
            batch_data = (batch_signals, batch_labels)

            # 1. EVALUAR
            model.eval()
            preds, _ = Fewshot.evaluation(batch_data, model, criterion, device) 
            
            # Desnormalizar y guardar
            pred_sbp = Fewshot.desnormalizar_zscore(preds[:, 0].detach().cpu().numpy(), SBP_MEAN, SBP_STD)
            pred_dbp = Fewshot.desnormalizar_zscore(preds[:, 1].detach().cpu().numpy(), DBP_MEAN, DBP_STD)
            true_sbp = Fewshot.desnormalizar_zscore(batch_labels[:, 0].numpy(), SBP_MEAN, SBP_STD)
            true_dbp = Fewshot.desnormalizar_zscore(batch_labels[:, 1].numpy(), DBP_MEAN, DBP_STD)

            historial_sbp['real'].extend(true_sbp); historial_sbp['pred'].extend(pred_sbp)
            historial_dbp['real'].extend(true_dbp); historial_dbp['pred'].extend(pred_dbp)

            # 2. ADAPTAR (Solo al principio o cada MUCHO tiempo)
            if i == 0 or (i + 1) % intervalo_ajuste == 0:
                model.train()
                for _ in range(n_epochs): 
                    _ = Fewshot.tuning(batch_data, optimizer, model, criterion, device)
                puntos_de_ajuste_x.append((i + 1) * n_shots)

        # Métricas y Gráficas
        mae_s, rmse_s, bias_s, std_s = calcular_metricas_avanzadas(historial_sbp['real'], historial_sbp['pred'])
        mae_d, rmse_d, bias_d, std_d = calcular_metricas_avanzadas(historial_dbp['real'], historial_dbp['pred'])

        print(f"   [SBP] RMSE: {rmse_s:.2f} | Bias: {bias_s:.2f}")
        
        # Plotting simplificado para ver el escalón
        fig, axs = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
        x_axis = range(len(historial_sbp['real']))
        
        # SBP
        axs[0].plot(x_axis, historial_sbp['real'], 'k', label='Real', alpha=0.7)
        axs[0].plot(x_axis, historial_sbp['pred'], 'r--', label='Estimado', alpha=0.9)
        for x_pos in puntos_de_ajuste_x:
            axs[0].axvline(x=x_pos, color='g', linestyle=':', alpha=0.5)
        axs[0].set_title(f"Paciente {id_paciente} - SBP (RMSE: {rmse_s:.2f})")
        axs[0].legend()
        axs[0].grid(True, linestyle='--', alpha=0.5)

        # DBP
        axs[1].plot(x_axis, historial_dbp['real'], 'k', label='Real', alpha=0.7)
        axs[1].plot(x_axis, historial_dbp['pred'], 'b--', label='Estimado', alpha=0.9)
        axs[1].set_title(f"Paciente {id_paciente} - DBP (RMSE: {rmse_d:.2f})")
        axs[1].grid(True, linestyle='--', alpha=0.5)
        
        plt.tight_layout()
        save_path = os.path.join(save_dir_graficas, f"step_response_{id_paciente}.png")
        plt.savefig(save_path)
        plt.close()
        print(f"   Gráfica guardada: {save_path}")

    print("\n--- PRUEBA FINALIZADA ---")

if __name__ == '__main__':
    main()