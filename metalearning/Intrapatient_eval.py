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

    save_dir_graficas = "resultados_intrapatient/graficas_adaptacion_dbp"
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

    model=Modelo_ConvolucionalV1(in_channels=2,out_channels=1, long_signal=500)
    path_model='models/best_meta_models/best_meta_model_sbp_patientwise.pt'
    checkpoint = torch.load(path_model, map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint['model_state_dict'])

    #base_weights = model.state_dict()

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
        model.load_state_dict(checkpoint['model_state_dict']) 
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
    
            print(f"Paciente {id_paciente}: {len(dataset_paciente_completo)} señales cargadas en {len(loader_paciente_N_shots)} lotes de ~{n_shots}.")

            resultados_pre_ajuste_mae = []
            """
            for epoch in range(1): # Solo para demostración
                for i, (batch_signals, batch_labels) in enumerate(loader_paciente_N_shots):
            
                    # batch_signals tendrá forma: [N_SHOTS, Canales, Longitud] (ej. [16, 2, 500])
                    # batch_labels tendrá forma: [N_SHOTS, 2] (ej. [16, 2])
            
                    print(f"  Lote {i+1}: Señales shape={batch_signals.shape}, Labels shape={batch_labels.shape}")
            
                    # Aquí iría tu lógica de fine-tuning con este lote...
                    # loss = tuning((batch_signals, batch_labels), ...)

        except ValueError as e:
            print(e)"""


            for i, (batch_signals, batch_labels) in enumerate(loader_paciente_N_shots):
                
                model.eval()
                with torch.no_grad():
                    preds, loss, etiqueta = Fewshot.evaluation((batch_signals, batch_labels), model, criterion, device, tipo_presion=tipo_presion)
                
                preds_desnorm = Fewshot.desnormalizar_zscore(preds.detach().cpu().numpy().reshape(-1), MEAN, STD)
                true_desnorm = Fewshot.desnormalizar_zscore(etiqueta.detach().cpu().numpy().reshape(-1), MEAN, STD)
                
                mae_lote = mean_absolute_error(true_desnorm, preds_desnorm)
                resultados_pre_ajuste_mae.append(mae_lote)

                print(f"  Lote {i+1}/{len(loader_paciente_N_shots)} | MAE (pre-ajuste): {mae_lote:.3f} mmHg")

                # 2. AJUSTAR (entrenar el modelo DESPUÉS de evaluar)
                # Ahora 'n_epochs' es un bucle INTERNO
                model.train()
                batch_loss = 0.0
                for _ in range(n_epochs): # Entrena N veces en ESTE lote
                    batch_loss = Fewshot.tuning((batch_signals, batch_labels), optimizer, model, criterion, device, tipo_presion=tipo_presion)

                print(f"    ...Lote {i+1} (Última loss: {batch_loss:.4f})")

            print(f"\n Adaptación Progresiva Completa (Paciente {id_paciente})")
            print("Evolución del MAE (medido en cada lote *antes* de entrenar):")
            print([round(mae, 2) for mae in resultados_pre_ajuste_mae])
            resultados_finales_experimento[id_paciente] = resultados_pre_ajuste_mae


            try:
                plt.figure(figsize=(12, 6))
                plt.plot(resultados_pre_ajuste_mae, marker='o', linestyle='-', markersize=4)
                plt.title(f"Adaptación Progresiva - Paciente {id_paciente} (LR={lr}, N_Shots={n_shots})")
                plt.xlabel(f"Lote de Adaptación (N={n_shots})")
                plt.ylabel(f"MAE (pre-ajuste) [{tipo_presion} mmHg]")
                plt.grid(True, which='both', linestyle='--', linewidth=0.5)
                plt.tight_layout()
                
                save_path_grafica = os.path.join(save_dir_graficas, f"adaptacion_paciente_{id_paciente}.png")
                plt.savefig(save_path_grafica)
                plt.close() # Cierra la figura para liberar memoria
                print(f"Gráfica de adaptación guardada en: {save_path_grafica}")
            except Exception as e:
                print(f"Error al generar/guardar la gráfica: {e}")


        except Exception as e:
            print(f"Error procesando paciente {id_paciente}: {e}")

    print("\n\n--- EXPERIMENTO FINALIZADO ---")
    print("Resultados guardados en 'resultados_finales_experimento'")
    return resultados_finales_experimento



    """
            for epoch in range(n_epochs):
                print(f"\n--- ÉPOCA {epoch + 1}/{n_epochs} (Paciente {id_paciente}) ---")
                
                # Iteramos sobre los lotes de datos secuenciales
                for i, (batch_signals, batch_labels) in enumerate(loader_paciente_N_shots):
                    
                    # 1. VALIDAR (evaluar el modelo ANTES de entrenar)
                    model.eval()
                    with torch.no_grad():
                        preds, loss, etiqueta = Fewshot.evaluation((batch_signals, batch_labels), model, criterion, device, tipo_presion=tipo_presion)
                    
                    # Calcular métricas para este lote
                    preds_desnorm = Fewshot.desnormalizar_zscore(preds.detach().cpu().numpy().reshape(-1), MEAN, STD)
                    true_desnorm = Fewshot.desnormalizar_zscore(etiqueta.detach().cpu().numpy().reshape(-1), MEAN, STD)
                    
                    # Usamos sklearn directamente para el MAE del lote
                    mae_lote = mean_absolute_error(true_desnorm, preds_desnorm)
                    resultados_pre_ajuste_mae.append(mae_lote)

                    print(f"  Época {epoch+1}, Lote {i+1}/{len(loader_paciente_N_shots)} | MAE (pre-ajuste): {mae_lote:.3f} mmHg")

                    # 2. AJUSTAR (entrenar el modelo DESPUÉS de evaluar)
                    model.train()
                    batch_loss = Fewshot.tuning((batch_signals, batch_labels), optimizer, model, criterion, device, tipo_presion=tipo_presion)

            print(f"\n--- Adaptación Progresiva Completa (Paciente {id_paciente}) ---")
            print("Evolución del MAE (medido en cada lote *antes* de entrenar):")
            print([round(mae, 2) for mae in resultados_pre_ajuste_mae])
            resultados_finales_experimento[id_paciente] = resultados_pre_ajuste_mae

        except Exception as e:
            print(f"Error procesando paciente {id_paciente}: {e}")
            
    print("\n\n--- EXPERIMENTO FINALIZADO ---")
    print("Resultados guardados en 'resultados_finales_experimento'")
    return resultados_finales_experimento
    """
if __name__ == '__main__':
    main()
