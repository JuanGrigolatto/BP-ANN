import torch.utils
from src.data.data_chargers.Clase_UCIDataset import UCIDataset
from src.data.data_chargers.MetaDataset import TaskDataset
from src.models.ConvolucionalV1 import Modelo_ConvolucionalV1
import numpy as np
import torch.utils.data as data
import torch
import random
import metalearning.Fewshot_single as Fewshot
from src.data.data_chargers.Intrapatientset import Intrapatientset
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt
import os

def main(n_shots=5, n_epochs=5, lr = 1e-3, MIN_SEÑALES_REQUERIDAS = 200, NUM_PACIENTES_A_PROBAR = 10, tipo_presion='DBP'):

    SEED = 42
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(SEED)

    save_dir_graficas = "resultados_intrapatient/graficas_calibracion_estatica_dbp"
    os.makedirs(save_dir_graficas, exist_ok=True)
    
    if tipo_presion == "SBP":
        MEAN = 134.02
        STD = 22.75
    else:
        MEAN = 63.47
        STD = 23.69

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

    model = Modelo_ConvolucionalV1(in_channels=2, out_channels=1, long_signal=500)
    path_model = 'models/best_meta_models/best_meta_model_dbp_patientwise.pt'
    checkpoint = torch.load(path_model, map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint['model_state_dict'])

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
        
        model.load_state_dict(checkpoint['model_state_dict']) 
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

            # --- LOGICA MODIFICADA ---
            
            # Convertimos el loader en un iterador para sacar el primer lote manualmente
            data_iter = iter(loader_paciente_N_shots)
            
            # CALIBRACIÓN 
            #
            try:
                support_signals, support_labels = next(data_iter)
            except StopIteration:
                print("No se pudieron cargar datos.")
                continue

            print(f"Calibrando modelo con el primer lote ({n_shots} señales) por {n_epochs} épocas...")
            
            model.train()
            final_cal_loss = 0.0
            for _ in range(n_epochs):
                final_cal_loss = Fewshot.tuning(
                    (support_signals, support_labels), 
                    optimizer, model, criterion, device, tipo_presion=tipo_presion
                )
            print(f"Calibración terminada. Loss final de ajuste: {final_cal_loss:.4f}")

            # EVALUACIÓN 
            
            resultados_mae_test = []
            model.eval() 
        
            with torch.no_grad(): 
                for i, (batch_signals, batch_labels) in enumerate(data_iter):
                    
                    preds, loss, etiqueta = Fewshot.evaluation(
                        (batch_signals, batch_labels), 
                        model, criterion, device, tipo_presion=tipo_presion
                    )
                
                    preds_desnorm = Fewshot.desnormalizar_zscore(preds.detach().cpu().numpy().reshape(-1), MEAN, STD)
                    true_desnorm = Fewshot.desnormalizar_zscore(etiqueta.detach().cpu().numpy().reshape(-1), MEAN, STD)
                    
                    mae_lote = mean_absolute_error(true_desnorm, preds_desnorm)
                    resultados_mae_test.append(mae_lote)

            resultados_finales_experimento[id_paciente] = resultados_mae_test
            
            mae_promedio = np.mean(resultados_mae_test)
            print(f"MAE Promedio post-calibración: {mae_promedio:.3f} mmHg")

            # GRAFICAR
            try:
                plt.figure(figsize=(12, 6))
                plt.plot(resultados_mae_test, marker='o', linestyle='-', markersize=4, label='MAE Test (Post-Calibración)')
                plt.axhline(y=mae_promedio, color='r', linestyle='--', label=f'Promedio ({mae_promedio:.2f})')
                
                plt.title(f"Desempeño Estático - Paciente {id_paciente}\n(Calibrado con 1 lote de {n_shots} señales)")
                plt.xlabel(f"Lotes de Test (secuenciales)")
                plt.ylabel(f"MAE [{tipo_presion} mmHg]")
                plt.legend()
                plt.grid(True, which='both', linestyle='--', linewidth=0.5)
                plt.tight_layout()
                
                save_path_grafica = os.path.join(save_dir_graficas, f"calibracion_paciente_{id_paciente}.png")
                plt.savefig(save_path_grafica)
                plt.close()
                print(f"Gráfica guardada en: {save_path_grafica}")
            except Exception as e:
                print(f"Error gráfica: {e}")

        except Exception as e:
            print(f"Error procesando paciente {id_paciente}: {e}")

    print("\n\n EXPERIMENTO FINALIZADO")
    return resultados_finales_experimento

if __name__ == '__main__':
    main()