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

def main(n_shots=5, n_epochs=5, lr = 1e-3, MIN_SEÑALES_REQUERIDAS = 1000, NUM_PACIENTES_A_PROBAR = 10):

    SEED = 42
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(SEED)

    #save_dir_graficas = "resultados_intrapatient/graficas_calibracion_estatica_dual"
    save_dir_graficas = "resultados_intrapatient/no_meta_model"
    os.makedirs(save_dir_graficas, exist_ok=True)
    
    SBP_MEAN, SBP_STD = 134.02, 22.75
    DBP_MEAN, DBP_STD = 63.47, 23.69

    #test_data = torch.load('data/processed/data_UCI/few_shot_patient_data.pt')
    test_data = torch.load('data/processed/data_UCI/test_set_por_picos.pt')
    #test_patient_ids = test_data['test_patient_ids']

    ruta_pacientes = 'data/processed/data_UCI/test_set_por_picos/test_patients.dat'
    n_muestras = test_data['num_samples']
    if not os.path.exists(ruta_pacientes):
        print(f"⚠️ No encontro el .dat en {ruta_pacientes}")
        ruta_pacientes = 'data/processed/data_UCI/test_set_por_picos_patients.npy'
    
    if not os.path.exists(ruta_pacientes):
        print("No encontró el archivo de pacientes.")
        print("Archivos en 'data/processed/data_UCI/':")
        print(os.listdir('data/processed/data_UCI/'))
        raise FileNotFoundError("Revisa la ruta del archivo de pacientes")

    print(f"Cargando pacientes desde: {ruta_pacientes}")
    
    pacientes_array = np.memmap(ruta_pacientes, dtype='int64', mode='r', shape=(n_muestras,))
    test_patient_ids = pacientes_array

    data_paths = [
        'data/processed/data_UCI/dataset_parte_1_por_picos.pt',
        'data/processed/data_UCI/dataset_parte_2_por_picos.pt',
        'data/processed/data_UCI/dataset_parte_3_por_picos.pt',
        'data/processed/data_UCI/dataset_parte_4_por_picos.pt'
    ]

    dataset_completo = UCIDataset(data_paths)
    print(" Datos cargados.")

    model = Modelo_ConvolucionalV1(in_channels=2, out_channels=2, long_signal=500)
    #path_model = 'models/checkpoints/best_meta_model_patientwise_s10q20_adapt5_PG2.pt'
    path_model = 'models/best_models/best_model_conv_v1_200_epocas_picos_def_early8_ps.pt'
    print(f"Cargando pesos desde {path_model}...")
    checkpoint = torch.load(path_model, map_location=torch.device('cpu'))
    state_dict = checkpoint['model_state_dict']
    
    # --- LIMPIEZA DE PESOS (Solo se hace una vez aquí) ---
    new_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    
    # Guardamos los pesos LIMPIOS en base_weights
    base_weights = new_state_dict 
    
    # Cargamos al modelo para probar que funciona
    model.load_state_dict(base_weights, strict=False)

    criterion = torch.nn.MSELoss()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)  

    taskset = TaskDataset(list_IDs=test_patient_ids, base_dataset=dataset_completo, num_shots=n_shots)

    mapa_indices_pacientes = taskset.patient_to_indices

    pacientes_elegibles = []
    for pid, indices in mapa_indices_pacientes.items():
        if len(indices) >= MIN_SEÑALES_REQUERIDAS:
            pacientes_elegibles.append(pid)

    print(f"Pacientes elegibles (>= {MIN_SEÑALES_REQUERIDAS} señales): {len(pacientes_elegibles)}")

    if len(pacientes_elegibles) < NUM_PACIENTES_A_PROBAR:
        pacientes_seleccionados = pacientes_elegibles
    else:
        pacientes_seleccionados = random.sample(pacientes_elegibles, NUM_PACIENTES_A_PROBAR)

    print(f"Pacientes seleccionados: {pacientes_seleccionados}")

    resultados_finales_experimento = {}

    for id_paciente in pacientes_seleccionados:
        print(f"PROCESANDO PACIENTE: {id_paciente}")
        
        model.load_state_dict(base_weights, strict=False)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        try:
            dataset_paciente_completo = Intrapatientset(
                patient_id=id_paciente,
                base_dataset=dataset_completo,
                patient_to_indices_map=mapa_indices_pacientes
            )

            loader_paciente_N_shots = torch.utils.data.DataLoader(
                dataset_paciente_completo,
                batch_size=n_shots,
                shuffle=False,
                num_workers=0,
                drop_last=False
            )
    
            print(f"Paciente {id_paciente}: {len(dataset_paciente_completo)} señales.")


            """
            data_iter = iter(loader_paciente_N_shots)
            
            
            try:
                support_signals, support_labels = next(data_iter)
            except StopIteration:
                print("No se pudieron cargar datos.")
                continue
                    
            print(f" > Calibrando estáticamente con {n_shots} señales ({n_epochs} épocas)...")
            
            model.train()
            batch_calibration = (support_signals, support_labels)
            for _ in range(n_epochs):
                _ = Fewshot.tuning(batch_calibration, optimizer, model, criterion, device)
            
            print(" > Calibración completada. Iniciando Test continuo (sin re-entrenar)...")
            """
            
            model.eval() 

            hist_sbp = {'real': [], 'pred': []}
            hist_dbp = {'real': [], 'pred': []}
            mae_list_sbp = []
            mae_list_dbp = []
        
            for i, (batch_signals, batch_labels) in enumerate(loader_paciente_N_shots):
            #for i, (batch_signals, batch_labels) in enumerate(data_iter):
                
                # Evaluar (Predicción)
                preds, _ = Fewshot.evaluation((batch_signals, batch_labels), model, criterion, device)
                
                # Desnormalizar SBP (col 0) y DBP (col 1)
                pred_sbp = Fewshot.desnormalizar_zscore(preds[:, 0].cpu().numpy(), SBP_MEAN, SBP_STD)
                pred_dbp = Fewshot.desnormalizar_zscore(preds[:, 1].cpu().numpy(), DBP_MEAN, DBP_STD)
                true_sbp = Fewshot.desnormalizar_zscore(batch_labels[:, 0].numpy(), SBP_MEAN, SBP_STD)
                true_dbp = Fewshot.desnormalizar_zscore(batch_labels[:, 1].numpy(), DBP_MEAN, DBP_STD)

                # Guardar métricas por lote
                mae_list_sbp.append(mean_absolute_error(true_sbp, pred_sbp))
                mae_list_dbp.append(mean_absolute_error(true_dbp, pred_dbp))

                # Historial continuo
                hist_sbp['real'].extend(true_sbp); hist_sbp['pred'].extend(pred_sbp)
                hist_dbp['real'].extend(true_dbp); hist_dbp['pred'].extend(pred_dbp)

            # Cálculo de métricas globales
            mae_s, rmse_s, bias_s, std_s = calcular_metricas_avanzadas(hist_sbp['real'], hist_sbp['pred'])
            mae_d, rmse_d, bias_d, std_d = calcular_metricas_avanzadas(hist_dbp['real'], hist_dbp['pred'])

            print(f"   [RESULTADO TEST] SBP RMSE: {rmse_s:.2f} | DBP RMSE: {rmse_d:.2f}")
            
            resultados_finales_experimento[id_paciente] = {'sbp': mae_list_sbp, 'dbp': mae_list_dbp}

            try:
                ### CAMBIO 4: Plot 2x2 con sharex=False y set_xlim para quitar espacio blanco
                fig, axs = plt.subplots(2, 2, figsize=(15, 10), sharex=False)
                
                # --- SBP ---
                # Tracking
                x_track_s = range(len(hist_sbp['real']))
                axs[0, 0].plot(x_track_s, hist_sbp['real'], 'k', alpha=0.7, label='Real')
                axs[0, 0].plot(x_track_s, hist_sbp['pred'], 'r--', alpha=0.9, label='Estimado')
                axs[0, 0].set_title(f"SBP Tracking (Static Calib) - RMSE: {rmse_s:.2f}")
                axs[0, 0].legend(loc='upper right')
                axs[0, 0].grid(True, linestyle='--', alpha=0.5)
                axs[0, 0].set_xlim(0, len(x_track_s)) # Ajuste exacto al eje X
                
                # MAE Evolution
                axs[1, 0].plot(mae_list_sbp, 'r-o', markersize=3)
                axs[1, 0].axhline(y=mae_s, color='k', linestyle=':', label=f'Media: {mae_s:.2f}')
                axs[1, 0].set_ylabel("MAE SBP [mmHg]")
                axs[1, 0].set_xlabel("Lotes de Test")
                axs[1, 0].grid(True, linestyle='--')
                axs[1, 0].set_xlim(0, len(mae_list_sbp)) # Ajuste exacto al eje X
                axs[1, 0].legend()

                # --- DBP ---
                # Tracking
                x_track_d = range(len(hist_dbp['real']))
                axs[0, 1].plot(x_track_d, hist_dbp['real'], 'k', alpha=0.7, label='Real')
                axs[0, 1].plot(x_track_d, hist_dbp['pred'], 'b--', alpha=0.9, label='Estimado')
                axs[0, 1].set_title(f"DBP Tracking (Static Calib) - RMSE: {rmse_d:.2f}")
                axs[0, 1].legend(loc='upper right')
                axs[0, 1].grid(True, linestyle='--', alpha=0.5)
                axs[0, 1].set_xlim(0, len(x_track_d)) # Ajuste exacto al eje X

                # MAE Evolution
                axs[1, 1].plot(mae_list_dbp, 'b-o', markersize=3)
                axs[1, 1].axhline(y=mae_d, color='k', linestyle=':', label=f'Media: {mae_d:.2f}')
                axs[1, 1].set_ylabel("MAE DBP [mmHg]")
                axs[1, 1].set_xlabel("Lotes de Test")
                axs[1, 1].grid(True, linestyle='--')
                axs[1, 1].set_xlim(0, len(mae_list_dbp)) # Ajuste exacto al eje X
                axs[1, 1].legend()

                plt.tight_layout()
                save_path = os.path.join(save_dir_graficas, f"calib_dual_{id_paciente}.png")
                plt.savefig(save_path)
                plt.close()
                print(f"   Gráfica guardada: {save_path}")

            except Exception as e:
                print(f"Error graficando: {e}")

        except Exception as e:
            print(f"Error procesando paciente {id_paciente}: {e}")

    print("\n\n--- EXPERIMENTO DE CALIBRACIÓN ESTÁTICA FINALIZADO ---")
    return resultados_finales_experimento

if __name__ == '__main__':
    main()