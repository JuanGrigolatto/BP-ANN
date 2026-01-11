import os
import csv
import json
import traceback
from datetime import datetime
import torch
from src.data.data_chargers.Clase_UCIDataset import UCIDataset
from metalearning.Fewshot_single import main
import random
import numpy as np

def set_all_seeds(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

RESULTS_DIR = "resultados_hiper_fewshot"
os.makedirs(RESULTS_DIR, exist_ok=True)

# === Carga de datos ===
test_data = torch.load('data/processed/data_UCI/few_shot_patient_data.pt')
test_patient_ids = test_data['test_patient_ids']

data_paths = [
    'data/processed/data_UCI/dataset_parte_1_por_picos.pt',
    'data/processed/data_UCI/dataset_parte_2_por_picos.pt',
    'data/processed/data_UCI/dataset_parte_3_por_picos.pt',
    'data/processed/data_UCI/dataset_parte_4_por_picos.pt'
]

dataset_completo = UCIDataset(data_paths)
print(" Datos cargados correctamente.")

# === Hiperparámetros a probar ===
learning_rates = [1e-5, 1e-4, 5e-4, 1e-3, 5e-3, 1e-2, 5e-2]
num_shots = 5 
SEED_FIJA = 42

# === Archivo de resultados ===
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
results_file = os.path.join(RESULTS_DIR, f"grid_fewshot_{timestamp}.csv")

with open(results_file, mode='w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow([
        "experiment", "learning_rate", "shots", 
        "mae_pre", "mae_post", "delta_mae",
        "tasa_mejora", "mejoraron", "empeoraron"
    ])

todos_resultados_por_paciente = []

# === Loop de experimentos ===
for i, lr in enumerate(learning_rates, start=1):
    print(f"\n=== Experimento {i}/{len(learning_rates)} ===")
    print(f"Learning Rate: {lr:.5f}, Shots: {num_shots}")
    print(f" Reseteando semilla a {SEED_FIJA} para reproducibilidad...")
    set_all_seeds(SEED_FIJA)
    try:
        resultados = main(
            n_shots=num_shots,
            base_lr=lr,
            base_dataset=dataset_completo,
            test_patient_ids=test_patient_ids
        )

        # Validación de resultado
        if not resultados or "mae_pre" not in resultados or "mae_post" not in resultados:
            print(f"⚠️ Resultado inválido en experimento {i}, se omite.")
            continue

        mae_pre = resultados["mae_pre"]
        mae_post = resultados["mae_post"]
        delta = mae_post - mae_pre
        mejoraron = resultados["mejoraron"]
        empeoraron = resultados["empeoraron"]
        tasa_mejora = resultados["tasa_mejora"]

        resultados_por_paciente = resultados["resultados_por_paciente"]
        todos_resultados_por_paciente.append({
            "learning_rate": lr,
            "n_shots": num_shots,
            "pacientes": resultados_por_paciente
        })

        # Guardar fila en CSV
        with open(results_file, mode='a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([i, lr, num_shots, mae_pre, mae_post, delta, tasa_mejora, mejoraron, empeoraron])

    except Exception as e:
        print(f" Error en experimento {i}: {e}")
        traceback.print_exc()

# === Guardar resultados por paciente ===
json_file = os.path.join(RESULTS_DIR, f"todos_resultados_por_paciente_{timestamp}.json")
with open(json_file, "w") as f:
    json.dump(todos_resultados_por_paciente, f, indent=4)

print(f"\n Todos los experimentos completados. Resultados en: {results_file}")