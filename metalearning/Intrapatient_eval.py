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
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt
import os

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
def main(n_shots=5, n_epochs=5, lr = 5e-3, MIN_SEÑALES_REQUERIDAS = 1000, NUM_PACIENTES_A_PROBAR = 10, MINUTOS_DESEADOS = 15):

    SEED = 42
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(SEED)

    save_dir_graficas = "resultados_intrapatient/graficas_adaptacion_intrapatient_DELTA_LEARNING_fast_anneal_k100_alpha90"
    os.makedirs(save_dir_graficas, exist_ok=True)
    
    SBP_MEAN, SBP_STD = 134.02, 22.75
    DBP_MEAN, DBP_STD = 63.47, 23.69

    
    SEGUNDOS_POR_LOTE = (500 / 125) * n_shots   # 500 muestras a 125 Hz

    intervalo_ajuste = int((MINUTOS_DESEADOS * 60) / SEGUNDOS_POR_LOTE)

    print(f"Intervalo de ajuste cada {intervalo_ajuste} lotes.")

    test_data = torch.load('data/processed/data_UCI/few_shot_patient_data.pt')

    test_patient_ids = test_data['test_patient_ids']

    data_paths = [
        'data/processed/data_UCI/dataset_parte_1_por_picos.pt',
        'data/processed/data_UCI/dataset_parte_2_por_picos.pt',
        'data/processed/data_UCI/dataset_parte_3_por_picos.pt',
        'data/processed/data_UCI/dataset_parte_4_por_picos.pt'
    ]

    dataset_completo = UCIDataset(data_paths)

    print(" Datos cargados.")

    model=Modelo_ConvolucionalV1(in_channels=2,out_channels=2, long_signal=500)
    path_model='models/checkpoints/best_meta_DELTA_LEARNING_refine_alpha50.pt'
    print(f"Cargando pesos desde {path_model}...")
    checkpoint = torch.load(path_model, map_location=torch.device('cpu'))
    
    state_dict = checkpoint['model_state_dict']
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('module.'):
            name = k[7:] # Quitar 'module.'
        else:
            name = k
        new_state_dict[name] = v
    
    # Cargamos el diccionario LIMPIO
    model.load_state_dict(new_state_dict, strict=False)

    criterion = torch.nn.MSELoss()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)  

    taskset = TaskDataset(list_IDs=test_patient_ids, base_dataset=dataset_completo, num_shots=n_shots)

    mapa_indices_pacientes = taskset.patient_to_indices

    # Filtrar pacientes que cumplen con el mínimo de señales
    pacientes_elegibles = []
    for pid, indices in mapa_indices_pacientes.items():
        if len(indices) >= MIN_SEÑALES_REQUERIDAS:
            pacientes_elegibles.append(pid)

    print(f"Pacientes elegibles (>= {MIN_SEÑALES_REQUERIDAS} señales): {len(pacientes_elegibles)}")

    # Seleccionar aleatoriamente N pacientes de la lista elegible
    if len(pacientes_elegibles) < NUM_PACIENTES_A_PROBAR:
        print(f"Advertencia: Se pidieron {NUM_PACIENTES_A_PROBAR} pacientes, pero solo {len(pacientes_elegibles)} son elegibles. Usando {len(pacientes_elegibles)}.")
        pacientes_seleccionados = pacientes_elegibles
    else:
        pacientes_seleccionados = random.sample(pacientes_elegibles, NUM_PACIENTES_A_PROBAR)

    print(f"Pacientes seleccionados para el experimento: {pacientes_seleccionados}")

    resultados_finales_experimento = {}

    for id_paciente in pacientes_seleccionados:
        print(f"PROCESANDO PACIENTE: {id_paciente}")
        
        model.load_state_dict(new_state_dict, strict=False)
        """
        #  Se congelan todas las capas primero
        for param in model.parameters():
            param.requires_grad = False
            
        # Descongelar SOLO las capas 'dense' 
        capas_activas = []
        for name, param in model.named_parameters():
            if 'dense' in name: 
                param.requires_grad = True
                capas_activas.append(name)
            if 'conv4' in name:
                param.requires_grad = True
                capas_activas.append(name)

        if not capas_activas:
            print("No se descongeló nada")
        else:
            print(f"  Body Freezing activado. Capas entrenables: {capas_activas[0]} ... {capas_activas[-1]}")
        
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
        """
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        try:
            dataset_paciente_completo = Intrapatientset(
            patient_id=id_paciente,
            base_dataset=dataset_completo,
            patient_to_indices_map=mapa_indices_pacientes
            )

            if len(dataset_paciente_completo) < n_shots:
                print(f"Error: Paciente {id_paciente} tiene {len(dataset_paciente_completo)} señales, menos que n_shots={n_shots}. Saltando.")
                continue

            loader_paciente_N_shots = torch.utils.data.DataLoader(
                dataset_paciente_completo,
                batch_size=n_shots,
                shuffle=False, 
                num_workers=0,
                drop_last=False
            )
    
            print(f"Paciente {id_paciente}: Iniciando adaptación online...")

            # Listas para métricas
            resultados_mae_sbp = []
            resultados_mae_dbp = []

            ### CAMBIO 6: Historiales separados para SBP y DBP
            historial_sbp = {'real': [], 'pred': []}
            historial_dbp = {'real': [], 'pred': []}

            puntos_de_ajuste_x = []

            for i, (batch_signals, batch_labels) in enumerate(loader_paciente_N_shots):
                
                # Tupla para pasar a funciones Fewshot
                batch_data = (batch_signals, batch_labels)

                # --- 1. EVALUAR (Usando Fewshot.evaluation) ---
                model.eval()
                # Fewshot.evaluation retorna (preds, loss)
                preds, _ = Fewshot.evaluation(batch_data, model, criterion, device) 
                
                # --- PROCESAMIENTO DUAL (Separar canales) ---
                # Asumimos: Col 0 = SBP, Col 1 = DBP
                pred_sbp_raw = preds[:, 0].detach().cpu().numpy()
                pred_dbp_raw = preds[:, 1].detach().cpu().numpy()
                
                true_sbp_raw = batch_labels[:, 0].numpy()
                true_dbp_raw = batch_labels[:, 1].numpy()

                # Desnormalizar
                pred_sbp = Fewshot.desnormalizar_zscore(pred_sbp_raw, SBP_MEAN, SBP_STD)
                pred_dbp = Fewshot.desnormalizar_zscore(pred_dbp_raw, DBP_MEAN, DBP_STD)
                
                true_sbp = Fewshot.desnormalizar_zscore(true_sbp_raw, SBP_MEAN, SBP_STD)
                true_dbp = Fewshot.desnormalizar_zscore(true_dbp_raw, DBP_MEAN, DBP_STD)

                mae_lote_sbp = mean_absolute_error(true_sbp, pred_sbp)
                mae_lote_dbp = mean_absolute_error(true_dbp, pred_dbp)
                
                resultados_mae_sbp.append(mae_lote_sbp)
                resultados_mae_dbp.append(mae_lote_dbp)

                historial_sbp['real'].extend(true_sbp)
                historial_sbp['pred'].extend(pred_sbp)
                historial_dbp['real'].extend(true_dbp)
                historial_dbp['pred'].extend(pred_dbp)

                if i == 0 or (i + 1) % intervalo_ajuste == 0:
                    model.train()
                    for _ in range(n_epochs): 
                        _ = Fewshot.tuning(batch_data, optimizer, model, criterion, device)
                    muestra_actual = (i + 1) * n_shots
                    puntos_de_ajuste_x.append(muestra_actual)

            resultados_finales_experimento[id_paciente] = {
                'mae_sbp': resultados_mae_sbp,
                'mae_dbp': resultados_mae_dbp
            }


            mae_s, rmse_s, bias_s, std_s = calcular_metricas_avanzadas(historial_sbp['real'], historial_sbp['pred'])
            mae_d, rmse_d, bias_d, std_d = calcular_metricas_avanzadas(historial_dbp['real'], historial_dbp['pred'])

            print(f"--- Resultados Paciente {id_paciente} ---")
            print(f"[SBP] RMSE: {rmse_s:.2f} | Bias: {bias_s:.2f} | SD: {std_s:.2f}")
            print(f"[DBP] RMSE: {rmse_d:.2f} | Bias: {bias_d:.2f} | SD: {std_d:.2f}")

            try:
              
                fig, axs = plt.subplots(2, 2, figsize=(15, 10), sharex=False)
                
               
                x_axis = range(len(historial_sbp['real']))
                axs[0, 0].plot(x_axis, historial_sbp['real'], label='Real', color='black', alpha=0.7)
                axs[0, 0].plot(x_axis, historial_sbp['pred'], label='Estimado', color='tab:red', ls='--', alpha=0.9)
                for idx, x_pos in enumerate(puntos_de_ajuste_x):
                    # Solo poner label en la primera para no saturar la leyenda
                    label = "Ajuste (Tuning)" if idx == 0 else None
                    if x_pos < len(x_axis): # Verificar límites
                        axs[0, 0].axvline(x=x_pos, color='green', linestyle=':', alpha=0.6, linewidth=1.5, label=label)
                axs[0, 0].set_title(f"SBP Tracking (RMSE: {rmse_s:.2f})")
                axs[0, 0].legend(loc='upper right')
                axs[0, 0].set_xlim(0, len(x_axis))  
                axs[0, 0].set_xlabel("Muestras (Tiempo)") 
                axs[0, 0].grid(True, linestyle='--', alpha=0.5)
                
                
                axs[1, 0].plot(resultados_mae_sbp, marker='o', color='tab:red', markersize=3)
                axs[1, 0].set_ylabel("MAE SBP [mmHg]")
                axs[1, 0].set_xlabel("Lotes de Adaptación")
                axs[1, 0].grid(True, linestyle='--')
                axs[1, 0].set_xlim(0, len(resultados_mae_sbp))
                

                
                axs[0, 1].plot(x_axis, historial_dbp['real'], label='Real', color='black', alpha=0.7)
                axs[0, 1].plot(x_axis, historial_dbp['pred'], label='Estimado', color='tab:blue', ls='--', alpha=0.9)
                for idx, x_pos in enumerate(puntos_de_ajuste_x):
                    if x_pos < len(x_axis):
                        axs[0, 1].axvline(x=x_pos, color='green', linestyle=':', alpha=0.6, linewidth=1.5)
                axs[0, 1].set_title(f"DBP Tracking (RMSE: {rmse_d:.2f})")
                axs[0, 1].legend(loc='upper right')
                axs[0, 1].grid(True, linestyle='--', alpha=0.5)
                axs[0, 1].set_xlim(0, len(x_axis)) 
                axs[0, 1].set_xlabel("Muestras (Tiempo)")
                
                
                axs[1, 1].plot(resultados_mae_dbp, marker='o', color='tab:blue', markersize=3)
                axs[1, 1].set_ylabel("MAE DBP [mmHg]")
                axs[1, 1].set_xlabel("Lotes de Adaptación")
                axs[1, 1].grid(True, linestyle='--')
                axs[1, 1].set_xlim(0, len(resultados_mae_dbp))

                plt.tight_layout()
                save_path_grafica = os.path.join(save_dir_graficas, f"dual_adapt_{id_paciente}.png")
                plt.savefig(save_path_grafica)
                plt.close() 
                print(f"Gráfica guardada en: {save_path_grafica}")

            except Exception as e:
                print(f"Error al generar/guardar la gráfica: {e}")
        except Exception as e:
            print(f"Error procesando paciente {id_paciente}: {e}")

    print("\n\n--- EXPERIMENTO FINALIZADO ---")
    return resultados_finales_experimento


if __name__ == '__main__':
    main()
