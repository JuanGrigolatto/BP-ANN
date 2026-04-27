"""
Módulo: Busqueda_Hiperparametros.py
Autor: Juan Marcos Grigolatto
Descripción: Implementación del experimento para la búsqueda exhaustiva de hiperparámetros 
             aplicada al algoritmo MAML en modo Patient-wise. Itera sobre múltiples 
             combinaciones de tasas de aprendizaje (meta y adaptativas), pasos de 
             adaptación (k_steps) y tamaños de grupo. Registra los resultados de 
             cada experimento en formato JSON y CSV, y genera gráficos comparativos 
             para justificar empíricamente la selección de la configuración óptima 
             del modelo fundacional.
"""
import itertools
import torch
import matplotlib.pyplot as plt
import numpy as np
import time
from metalearning.Patient_wise import main as patientwise_main
import csv
import os
import json
from datetime import datetime
from src.data.data_chargers.Clase_UCIDataset import UCIDataset
import random

def experiment_patientwise():
    """_summary_   Realiza una búsqueda exhaustiva de hiperparámetros para el algoritmo MAML. 
    """    
    # === Crear carpeta de resultados ===
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    results_dir = os.path.join("results", f"exp_{timestamp}")
    os.makedirs(results_dir, exist_ok=True)
    print(f"\n Carpeta de resultados: {results_dir}")
    
    # === Cargar dataset una sola vez ===
    print("Cargando dataset UCI en memoria...")
    data_paths = [
        'data/processed/data_UCI/dataset_parte_1_por_picos.pt',
        'data/processed/data_UCI/dataset_parte_2_por_picos.pt',
        'data/processed/data_UCI/dataset_parte_3_por_picos.pt',
        'data/processed/data_UCI/dataset_parte_4_por_picos.pt',
    ]

    dataset_completo = UCIDataset(data_paths)

    # === Seleccionar 200 pacientes una sola vez ===
    random.seed(42)
    all_pids = [dataset_completo[i][2].item() for i in range(len(dataset_completo))]
    all_patients = list(set(all_pids))
    selected_patients = random.sample(all_patients, 200)
    print(f" Seleccionados {len(selected_patients)} pacientes para todos los experimentos.")

    print(" Dataset cargado correctamente.")

    adapt_lrs = [0.001, 0.005, 0.01, 0.02]
    meta_lrs = [0.0001,0.0005, 0.001, 0.005]
    k_steps = [1, 3, 5, 10]
    N_patient_groups = [4, 6, 10, 20]
    query_sets = [10]
    suport_sets = [5]

    hyperparam_combinations = list(itertools.product(adapt_lrs, meta_lrs, k_steps, N_patient_groups, query_sets, suport_sets))

    results = []

    for i, (adapt_lr, meta_lr, k_adapt_steps, N_group, query_set, suport_set) in enumerate(hyperparam_combinations, 1):
        print(f"\n=== Experimento {i}/{len(hyperparam_combinations)} ===")
        print(f"adapt_lr={adapt_lr}, meta_lr={meta_lr}, k_adapt_steps={k_adapt_steps}, N_group={N_group}, query_sets={query_set}, suport_sets={suport_set}")

        start_time = time.time()

        try:
            patientwise_main(
                num_tasks=200,
                tasks_per_batch=4,
                adapt_lr=adapt_lr,
                meta_lr=meta_lr,
                k_adapt_steps=k_adapt_steps,
                seed=42,
                num_epochs=5,
                N_patient_group=N_group,
                p_support=suport_set,
                q_query=query_set,
                base_dataset=dataset_completo,
                selected_patients=selected_patients  
            )
            elapsed_time = time.time() - start_time

            loss_file = "models/best_meta_models/best_meta_model_patientwise.pt"
            #results.append((adapt_lr, meta_lr, k_adapt_steps, N_group, "Completado", elapsed_time))
            
            if os.path.exists(loss_file):
                losses = torch.load(loss_file)
                loss_final = float(losses['meta_loss']) if 'meta_loss' in losses else np.nan
            else:
                loss_final = np.nan

            result = {
                'adapt_lr': adapt_lr,
                'meta_lr': meta_lr,
                'k_adapt_steps': k_adapt_steps,
                'N_patient_group': N_group,
                'query_set': query_set,
                'suport_set': suport_set,
                'loss': loss_final,
                'elapsed_time_min': round(elapsed_time / 60, 2)
            }

            results.append(result)
            print(f" Experimento completado en {elapsed_time/60:.2f} min | Meta Loss: {loss_final:.4f}")

        except Exception as e:
            print(f" Error en experimento {i}: {e}")
            results.append({
                'adapt_lr': adapt_lr,
                'meta_lr': meta_lr,
                'k_adapt_steps': k_adapt_steps,
                'N_patient_group': N_group,
                'query_set': query_set,
                'suport_set': suport_set,
                'loss': np.inf,
                'elapsed_time_min': 0
            })



    # === Guardar resultados ===
    results = sorted(results, key=lambda x: x['loss'])
    json_path = os.path.join(results_dir, "results.json")
    csv_path = os.path.join(results_dir, "results.csv")

    with open(json_path, "w") as f:
        json.dump(results, f, indent=4)

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)

    print(f"\n Resultados guardados en:\n- {json_path}\n- {csv_path}")
    
    # === Graficar resultados ===
    losses = [r['loss'] for r in results if np.isfinite(r['loss'])]
    labels = [f"a={r['adapt_lr']}, m={r['meta_lr']}, k={r['k_adapt_steps']}, G={r['N_patient_group']}" for r in results if np.isfinite(r['loss'])]
    
    plt.figure(figsize=(10, 5))
    plt.bar(range(len(losses)), losses, color='skyblue')
    plt.xticks(range(len(losses)), labels, rotation=60, ha='right')
    plt.ylabel("Meta Loss Final")
    plt.title("Comparación de hiperparámetros - MAML Patient-wise")
    plt.tight_layout()
    
    plot_path = os.path.join(results_dir, "hparam_comparison.png")
    plt.savefig(plot_path)
    plt.close()

    print(f" Gráfico guardado en: {plot_path}")

    # === Mejor configuración ===
    best = results[0]
    print("\n Mejor configuración encontrada:")
    for k, v in best.items():
        print(f"  {k}: {v}")

if __name__ == "__main__":
    experiment_patientwise()


