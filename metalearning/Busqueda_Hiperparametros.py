import itertools
import torch
import matplotlib.pyplot as plt
import numpy as np
import time
from metalearning.Patient_wise import main as patientwise_main

def experiment_patientwise():

    adapt_lrs = [0.001, 0.005, 0.01, 0.02]
    meta_lrs = [0.0001,0.0005, 0.001, 0.005]
    k_steps = [1, 3, 5, 10]
    N_patient_groups = [4, 6, 10, 20]

    hyperparam_combinations = list(itertools.product(adapt_lrs, meta_lrs, k_steps, N_patient_groups))

    results = []

    for i, (adapt_lr, meta_lr, k_adapt_steps, N_group) in enumerate(hyperparam_combinations, 1):
        print(f"\n=== Experimento {i}/{len(hyperparam_combinations)} ===")
        print(f"adapt_lr={adapt_lr}, meta_lr={meta_lr}, k_adapt_steps={k_adapt_steps}, N_group={N_group}")

        start_time = time.time()

        try:
            patientwise_main(
                shots=5,
                num_tasks=1000,
                tasks_per_batch=8,
                adapt_lr=adapt_lr,
                meta_lr=meta_lr,
                k_adapt_steps=k_adapt_steps,
                seed=42,
                num_epochs=10,
                N_patient_group=N_group,
                p_support=10,
                q_query=20
            )
            elapsed_time = time.time() - start_time
            print(f"Experimento completado en {elapsed_time/60:.2f} minutos")
            results.append((adapt_lr, meta_lr, k_adapt_steps, N_group, "Completado", elapsed_time))
            
            losses = torch.load('models/best_meta_models/best_meta_model_patientwise.pt')
            loss_final = float(losses['meta_loss'])
            results.append({
                'adapt_lr': adapt_lr,
                'meta_lr': meta_lr,
                'k_adapt_steps': k_adapt_steps,
                'N_patient_group': N_group,
                'loss': loss_final
            })

        except Exception as e:
            print(f" Error en experimento {i}: {e}")
            results.append({
                'adapt_lr': adapt_lr,
                'meta_lr': meta_lr,
                'k_adapt_steps': k_adapt_steps,
                'N_patient_group': N_group,
                'loss': np.inf
            })


    # Ordena por menor pérdida
    results = sorted(results, key=lambda x: x['loss'])
    
    print("\n=== Resultados ordenados por menor pérdida ===")
    for r in results:
        print(f"Loss={r['loss']:.4f} | adapt_lr={r['adapt_lr']}, meta_lr={r['meta_lr']}, "
              f"k_adapt_steps={r['k_adapt_steps']}, N_group={r['N_patient_group']}")
    
    
    plt.figure(figsize=(8,4))
    losses = [r['loss'] for r in results if np.isfinite(r['loss'])]
    labels = [f"adapt={r['adapt_lr']}, meta={r['meta_lr']}, k={r['k_adapt_steps']}" for r in results if np.isfinite(r['loss'])]
    plt.bar(range(len(losses)), losses)
    plt.xticks(range(len(losses)), labels, rotation=45, ha='right')
    plt.ylabel("Meta Loss Final")
    plt.title("Comparación de hiperparámetros - MAML Patient-wise")
    plt.tight_layout()
    plt.savefig("metalearning/hparam_comparison.png")
    plt.show()

    # Mejor configuración
    best = results[0]
    print(f"\n Mejor configuración encontrada:")
    print(best)

if __name__ == "__main__":
    experiment_patientwise()           


