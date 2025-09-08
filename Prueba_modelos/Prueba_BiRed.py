import torch
from src.models.ConvolucionalV1 import Modelo_ConvolucionalV1
from src.data.data_chargers.Clase_UCIDataset import UCIDataset
import os
from tqdm.auto import tqdm 
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np
import matplotlib.pyplot as plt

# Valores reales de Min y Max que usaste al normalizar SBP y DBP

SBP_MIN, SBP_MAX = 70, 199.99
DBP_MIN, DBP_MAX = 50, 140
SBP_MEAN = 134.02
DBP_MEAN = 63.47
SBP_STD = 22.75
DBP_STD = 23.69
"""
ABP_MAX, ABP_MIN = 60, 180.00
"""
def aami_metrics(y_true, y_pred):
    errors = y_pred - y_true
    mean_error = np.mean(errors)
    std_error = np.std(errors, ddof=1)
    return mean_error, std_error

def desnormalizar_minmax(norm_array, min_val, max_val):
    return norm_array * (max_val - min_val) + min_val

def desnormalizar_zscore(norm_array, media, std):
    return norm_array * std + media

def bland_altman_graf(preds, labels, title):
    differences = preds - labels
    averages = (preds + labels) / 2

    mean_diff = np.mean(differences)
    std_diff = np.std(differences)

    upper_limit = mean_diff + 1.96 * std_diff
    lower_limit = mean_diff - 1.96 * std_diff

    plt.figure(figsize=(8,5))
    plt.scatter(averages, differences, alpha=0.5)
    plt.axhline(mean_diff, color='red', linestyle='--', label=f'Media: {mean_diff:.2f}')
    plt.axhline(upper_limit, color='gray', linestyle='--', label=f'+1.96 SD: {upper_limit:.2f}')
    plt.axhline(lower_limit, color='gray', linestyle='--', label=f'-1.96 SD: {lower_limit:.2f}')
    plt.axhline(0, color='black', linewidth=1)
    plt.xlabel('Promedio (mmHg)')
    plt.ylabel('Diferencia (Pred - Real) (mmHg)')
    plt.title(f'Gráfico de Bland-Altman - {title}')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_comparacion_errores(dataset, indices_alto, indices_bajo, n=3):
    """
    dataset: tu UCIDataset o similar
    indices_alto: lista de indices con error alto
    indices_bajo: lista de indices con error bajo
    n: cantidad de ejemplos a graficar de cada grupo
    """
    fig, axes = plt.subplots(2, n, figsize=(4*n, 6))

    # Graficar n ejemplos con error alto
    for i in range(n):
        idx = indices_alto[i]
        senales, _, _, _ = dataset[idx]
        ppg, ecg = senales[0].numpy(), senales[1].numpy()

        axes[0, i].plot(ecg, color="orange", label="ECG")
        axes[0, i].plot(ppg, color="blue", label="PPG")
        axes[0, i].set_title(f"Error alto (idx {idx})")

    # Graficar n ejemplos con error bajo
    for i in range(n):
        idx = indices_bajo[i]
        senales, _ , _, _ = dataset[idx]
        ppg, ecg = senales[0].numpy(), senales[1].numpy()

        axes[1, i].plot(ecg, color="orange", label="ECG")
        axes[1, i].plot(ppg, color="blue", label="PPG")
        axes[1, i].set_title(f"Error bajo (idx {idx})")

    plt.tight_layout()
    plt.show()

def main():
    parameters = {
        'batch_size': 256,
        'shuffle': True,
        'num_workers': 0,
        'pin_memory': False
    }

    print(os.path.exists('data/processed/data_UCI/test_set_por_picos/test_meta.pt"'))
    dataset = UCIDataset(['data/processed/data_UCI/test_set_por_picos/test_meta.pt'])
    
    subset = torch.utils.data.Subset(dataset, indices=list(range(10000)))
    dataloader = torch.utils.data.DataLoader(subset, **parameters)

    all_labels = []
    for x, y, pid, idx in dataloader:
        all_labels.append(y)

    labels = torch.cat(all_labels, dim=0)
    

    print("min:", labels.min().item())
    print("max:", labels.max().item())
    print("mean:", labels.mean().item())
    print("std:", labels.std().item())
    print("Hay NaNs:", torch.isnan(labels).any().item())
    print("Hay Infs:", torch.isinf(labels).any().item())
    
    path_model = 'models/best_models/best_model_time32_picos.pt'

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Modelo_ConvolucionalV1(in_channels=2,out_channels=1, long_signal=500)
    
    checkpoint = torch.load(path_model, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    model.to(device)

    bar = tqdm(dataloader)
    accmse, accrmse, accmae, accr2 = [], [], [], []
    acc_mean_error_sbp = []
    acc_std_error_sbp = []
    #acc_mean_error_dbp = []
    #acc_std_error_dbp = []
    all_preds = []
    all_labels = []
    
    n_error=0

    indices_errores = []
    valores_errores = []

    indices_error_alto = []
    indices_error_bajo = []

    with torch.no_grad():
        for batch in bar:
            data, labels, ID_paciente, indice_muestra = batch
            
            labels = labels[:,0].unsqueeze(1) 

            data, labels = data.to(device), labels.to(device)
            
            pred = model(data)
            
            # Conversión a NumPy
            pred = pred.cpu().numpy()
            labels = labels.cpu().numpy()

            """
            # Desnormalización
            pred_SBP = desnormalizar_minmax(pred[:, 0], SBP_MIN, SBP_MAX)
            pred_DBP = desnormalizar_minmax(pred[:, 1], DBP_MIN, DBP_MAX)
            true_SBP = desnormalizar_minmax(labels[:, 0], SBP_MIN, SBP_MAX)
            true_DBP = desnormalizar_minmax(labels[:, 1], DBP_MIN, DBP_MAX)
            """
            """
            pred_SBP = desnormalizar_zscore(pred[:, 0], SBP_MEAN, SBP_STD)
            pred_DBP = desnormalizar_zscore(pred[:, 1], DBP_MEAN, DBP_STD)
            true_SBP = desnormalizar_zscore(labels[:, 0], SBP_MEAN, SBP_STD)
            true_DBP = desnormalizar_zscore(labels[:, 1], DBP_MEAN, DBP_STD)
        
            pred_desnorm = np.stack([pred_SBP, pred_DBP], axis=1)
            labels_desnorm = np.stack([true_SBP, true_DBP], axis=1)
            """
            pred = desnormalizar_zscore(pred, SBP_MEAN, SBP_STD)
            true = desnormalizar_zscore(labels, SBP_MEAN, SBP_STD)

            mse = mean_squared_error(true, pred)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(true, pred)
            r2 = r2_score(true, pred)

            mean_error_SBP, std_error_SBP = aami_metrics(true, pred)
            #mean_error_DBP, std_error_DBP = aami_metrics(true_DBP, pred_DBP)        

            accmse.append(mse)
            accrmse.append(rmse)
            accmae.append(mae)
            accr2.append(r2)

            acc_mean_error_sbp.append(mean_error_SBP)
            acc_std_error_sbp.append(std_error_SBP)
            #acc_mean_error_dbp.append(mean_error_DBP)
            #acc_std_error_dbp.append(std_error_DBP)

            all_preds.append(pred)
            all_labels.append(true)
            
            #for j in range(pred_desnorm.shape[0]):
            for j in range(len(pred)):
                """
                pred_sbp, pred_dbp = pred_desnorm[0][j]
                true_sbp, true_dbp = labels_desnorm[0][j]
                error_sbp = abs(pred_sbp - true_sbp)
                """
                pred_x = pred[j]
                true_x = true[j]
                error_x = abs(pred_x - true_x)
                #error_dbp = abs(pred_dbp - true_dbp)
                
                valores_errores.append(error_x)
                indices_errores.append(indice_muestra[j].item())
                if error_x >= 10: 
                    indices_error_alto.append(j)
                else:
                    indices_error_bajo.append(j)
    
                # Umbral configurable para detección de error alto
            """
                if error_sbp > 20 or error_dbp > 20:
                    n_error = n_error + 1
                    indices_errores.append(indice_muestra)
            """
            """
                    print(f"\n Error alto detectado:")
                    print(f"  Predicción SBP/DBP: {pred_sbp:.2f} / {pred_dbp:.2f}")
                    print(f"  Real      SBP/DBP: {true_sbp:.2f} / {true_dbp:.2f}")
                    print(f"  Error SBP: {error_sbp:.2f}, Error DBP: {error_dbp:.2f}")
                    print(f"  n: {j}")


                    # Graficar la señal cruda de entrada
                    senal = data[j].cpu().numpy()  # [2, 250]
                    ppg = senal[0]
                    ecg = senal[1]

                    plt.figure(figsize=(10, 4))
                    plt.subplot(2, 1, 1)
                    plt.plot(ecg, label='ECG', color='orange')
                    plt.ylabel('Amplitud')
                    plt.title('ECG')

                    plt.subplot(2, 1, 2)
                    plt.plot(ppg, label='PPG', color='blue')
                    plt.xlabel('Muestras')
                    plt.ylabel('Amplitud')
                    plt.title('PPG')

                    plt.tight_layout()
                    plt.show()
        """
        """
                if error_sbp < 10 and error_dbp < 10:
                    n_error = n_error + 1
                    indices_errores.append(indice_muestra)
        """
        """    
                    print(f"\n Error bajo detectado:")
                    print(f"  Predicción SBP/DBP: {pred_sbp:.2f} / {pred_dbp:.2f}")
                    print(f"  Real      SBP/DBP: {true_sbp:.2f} / {true_dbp:.2f}")
                    print(f"  Error SBP: {error_sbp:.2f}, Error DBP: {error_dbp:.2f}")
                    print(f"  n: {j}")
                   

                    # Graficar la señal cruda de entrada
                    senal = data[j].cpu().numpy()  # [2, 250]
                    ppg = senal[0]
                    ecg = senal[1]

                    plt.figure(figsize=(10, 4))
                    plt.subplot(2, 1, 1)
                    plt.plot(ecg, label='ECG', color='orange')
                    plt.ylabel('Amplitud')
                    plt.title('ECG')

                    plt.subplot(2, 1, 2)
                    plt.plot(ppg, label='PPG', color='blue')
                    plt.xlabel('Muestras')
                    plt.ylabel('Amplitud')
                    plt.title('PPG')

                    plt.tight_layout()
                    plt.show()
        """
    print(f"Muestras con error alto: {len(indices_error_alto)}")
    print(f"Muestras con error bajo: {len(indices_error_bajo)}")
            
    plot_comparacion_errores(subset, indices_error_alto, indices_error_bajo, n=7)
    
    print(f"numero de ventanas con errores: {n_error}")

    errores = {
        'valores': valores_errores,
        'indices': indices_errores
    }

    np.savez('data/processed/Errores_predicción', **errores)
    
    # Unimos todos los resultados para gráficos finales
    all_preds = np.concatenate(all_preds, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)

    print(f"\nMétricas promedio (valores desnormalizados en mmHg):")
    print(f"  MSE: {np.mean(accmse):.4f}")
    print(f" RMSE: {np.mean(accrmse):.4f}")
    print(f"  MAE: {np.mean(accmae):.4f}")
    print(f"   R2: {np.mean(accr2):.4f}")
    print(f"\nPromedio de métricas AAMI (valores desnormalizados en mmHg):")
    print(f"SBP - Mean Error: {np.mean(acc_mean_error_sbp):.2f}")
    print(f"SBP - Std Error:  {np.mean(acc_std_error_sbp):.2f}")
    #print(f"DBP - Mean Error: {np.mean(acc_mean_error_dbp):.2f}")
    #print(f"DBP - Std Error:  {np.mean(acc_std_error_dbp):.2f}")

    # Gráfico real vs predicho
    plt.figure(figsize=(8, 5))
    plt.scatter(all_labels, all_preds, alpha=0.5)
    plt.plot([all_labels.min(), all_labels.max()], [all_labels.min(), all_labels.max()], 'r--')
    plt.xlabel("Valor verdadero (mmHg)")
    plt.ylabel("Predicción (mmHg)")
    plt.title(f"R² = {r2_score(all_labels, all_preds):.2f}")
    plt.grid(True)
    plt.show()

    # Gráfico de residuos
    residuals = all_preds - all_labels
    plt.figure(figsize=(8, 4))
    plt.scatter(all_labels, residuals, alpha=0.5)
    plt.axhline(0, color='red', linestyle='--')
    plt.xlabel("Valor verdadero (mmHg)")
    plt.ylabel("Error (Pred - Real) (mmHg)")
    plt.title("Errores de predicción (residuos)")
    plt.grid(True)
    plt.show()

    # Histograma de errores
    plt.hist(residuals.flatten(), bins=50, edgecolor='black')
    plt.title("Distribución de errores")
    plt.xlabel("Error (mmHg)")
    plt.ylabel("Frecuencia")
    plt.grid(True)
    plt.show()

    #Gráfico Bland Altman
    bland_altman_graf(all_preds, all_labels, title="Modelo Conv V1 S - Picos")

if __name__ == '__main__':
    main()