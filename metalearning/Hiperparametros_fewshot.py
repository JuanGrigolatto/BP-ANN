import os
import csv
import json
import traceback
from datetime import datetime
import torch
from src.data.data_chargers.Clase_UCIDataset import UCIDataset
from metalearning.Fewshot import main

RESULTS_DIR = "resultados_hiper_fewshot"
os.makedirs(RESULTS_DIR, exist_ok=True)

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

#Hiperparámetros a probar
learning_rates = [1e-5, 1e-4, 5e-4, 1e-3, 5e-3, 1e-2, 5e-2]
num_shots = 5 

# Archivo de resultados con timestamp
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
results_file = os.path.join(RESULTS_DIR, f"grid_fewshot_{timestamp}.csv")

with open(results_file, mode='w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow([
    "experiment", "learning_rate", "shots", 
    "mae_pre_sbp", "mae_post_sbp", "delta_mae_sbp",
    "mae_pre_dbp", "mae_post_dbp", "delta_mae_dbp", 
    "tasa_mejora_sbp", "tasa_mejora_dbp", "mejoraron_sbp",
    "mejoraron_dbp", "empeoraron_sbp", "empeoraron_dbp"
])

todos_resultados_por_paciente = []
for i, lr in enumerate(learning_rates, start=1):
    print(f"\n=== Experimento {i}/{len(learning_rates)} ===")
    print(f"Learning Rate: {lr:.5f}, Shots: {num_shots}")

    try:
        # Ejecutar el experimento llamando a Fewshot.main()
        resultados = main(
            n_shots=num_shots,
            base_lr=lr,
            base_dataset=dataset_completo,
            test_patient_ids=test_patient_ids            
        )

        mae_pre_sbp = resultados["mae_pre_sbp"]
        mae_post_sbp = resultados["mae_post_sbp"]
        delta_sbp = mae_post_sbp - mae_pre_sbp
        mejoraron_sbp = resultados["mejoraron sbp"] 
        mejoraron_dbp = resultados["mejoraron dbp"] 
        empeoraron_sbp = resultados["empeoraron sbp"]
        empeoraron_dbp = resultados["empeoraron dbp"]
        tasa_mejora_sbp = resultados["tasa_mejora_sbp"]
        tasa_mejora_dbp = resultados["tasa_mejora_dbp"]
        mae_pre_dbp = resultados["mae_pre_dbp"]
        mae_post_dbp = resultados["mae_post_dbp"]
        delta_dbp = mae_post_dbp - mae_pre_dbp
        
        resultados_por_paciente = resultados["resultados_por_paciente"]
        todos_resultados_por_paciente.append({
            "learning_rate": lr,
            "n_shots": num_shots,
            "pacientes": resultados_por_paciente
        })

        with open(results_file, mode='a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([i, lr, num_shots, mae_pre_sbp, mae_post_sbp, delta_sbp, mae_pre_dbp, mae_post_dbp, delta_dbp, tasa_mejora_sbp, tasa_mejora_dbp, mejoraron_sbp, mejoraron_dbp, empeoraron_sbp, empeoraron_dbp])
        
    except Exception as e:
        print(f" Error en experimento {i}: {e}")
        traceback.print_exc()

json_file = os.path.join(RESULTS_DIR, f"todos_resultados_por_paciente_{timestamp}.json")
with open(json_file, "w") as f:
    json.dump(todos_resultados_por_paciente, f, indent=4)

print(f"\nTodos los experimentos completados. Resultados en: {results_file}")

