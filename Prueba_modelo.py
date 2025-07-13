import torch
#from Modelos.Modelo_conv import Modelo_Convolucional
from Modelos.InceptionTime import InceptionTime
from Modelos.ConvolucionalV1 import Modelo_ConvolucionalV1
from Modelos.ConvolucionalV2 import Modelo_ConvolucionalV2
from Clase_UCIDataset import UCIDataset
import os
from tqdm.auto import tqdm 
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np
import matplotlib.pyplot as plt

# Valores reales de Min y Max que usaste al normalizar SBP y DBP
"""
SBP_MIN, SBP_MAX = 80, 199.78
DBP_MIN, DBP_MAX = 50, 192.09
"""
ABP_MAX, ABP_MIN = 50, 199.99

def desnormalizar_minmax(norm_array, min_val, max_val):
    return norm_array * (max_val - min_val) + min_val

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


def main():
    parameters = {
        'batch_size': 50,
        'shuffle': True,
        'num_workers': 0,
        'pin_memory': True
    }

    print(os.path.exists('data_UCI/test_set.pt'))
    dataset = UCIDataset('data_UCI/test_set.pt')
    
    subset = torch.utils.data.Subset(dataset, indices=list(range(100)))
    dataloader = torch.utils.data.DataLoader(subset, **parameters)
    
    #dataloader = torch.utils.data.DataLoader(dataset, **parameters)

    path_model = 'best_model_conv_v2.pt'

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #model = InceptionTime(c_in=2, c_out=2, seq_len=None, n_filters=32)
    model=Modelo_ConvolucionalV2(in_channels=2,out_channels=2, long_signal=250)
    
    checkpoint = torch.load(path_model, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    model.to(device)

    bar = tqdm(dataloader)
    accmse, accrmse, accmae, accr2 = [], [], [], []
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in bar:
            data, labels = batch
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
            pred_SBP = desnormalizar_minmax(pred[:, 0], ABP_MIN, ABP_MAX)
            pred_DBP = desnormalizar_minmax(pred[:, 1], ABP_MIN, ABP_MAX)
            true_SBP = desnormalizar_minmax(labels[:, 0], ABP_MIN, ABP_MAX)
            true_DBP = desnormalizar_minmax(labels[:, 1], ABP_MIN, ABP_MAX)
            # Juntamos nuevamente para cálculo de métricas y gráficos
            pred_desnorm = np.stack([pred_SBP, pred_DBP], axis=1)
            labels_desnorm = np.stack([true_SBP, true_DBP], axis=1)

            mse = mean_squared_error(labels_desnorm, pred_desnorm)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(labels_desnorm, pred_desnorm)
            r2 = r2_score(labels_desnorm, pred_desnorm)

            accmse.append(mse)
            accrmse.append(rmse)
            accmae.append(mae)
            accr2.append(r2)

            all_preds.append(pred_desnorm)
            all_labels.append(labels_desnorm)

    # Unimos todos los resultados para gráficos finales
    all_preds = np.concatenate(all_preds, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)

    print(f"\nMétricas promedio (valores desnormalizados en mmHg):")
    print(f"  MSE: {np.mean(accmse):.4f}")
    print(f" RMSE: {np.mean(accrmse):.4f}")
    print(f"  MAE: {np.mean(accmae):.4f}")
    print(f"   R2: {np.mean(accr2):.4f}")

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
    bland_altman_graf(all_preds, all_labels, title="Modelo convolucional V1")
if __name__ == '__main__':
    main()
