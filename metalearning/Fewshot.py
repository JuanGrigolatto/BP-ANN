import torch.utils
from src.data.data_chargers.Clase_UCIDataset import UCIDataset
from src.data.data_chargers.MetaDataset import TaskDataset
from src.models.Modelo_conv import Modelo_Convolucional
from src.models.ConvolucionalV1 import Modelo_ConvolucionalV1
import numpy as np
from torch import device, nn, optim
import matplotlib.pyplot as plt
import torch.utils.data as data
import torch
from src.data.data_chargers.Tuningndataset import TuningNDataset
import random
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def promedio_metricas(m_list):
    return np.mean(m_list, axis=0)

def calcular_metricas(y_true, y_pred):
    errores = y_pred - y_true
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))    
    
    # Métricas Clínicas (ISO 81060-2)
    bias = np.mean(errores)   
    std = np.std(errores)     
    
    # Retorna 4 valores: 0:MAE, 1:RMSE, 2:BIAS, 3:STD
    return mae, rmse, bias, std

def desnormalizar_zscore(norm_array, media, std):
    return norm_array * std + media

def tuning(sample, optimizer, model, criterion, device):
    optimizer.zero_grad() 
    data, labels, *_ = sample 
    
    if isinstance(data, list): data = data[0]
    if isinstance(labels, list): labels = labels[0]
        
    # Congelar capas BatchNorm para estabilidad en few-shot
    for layer in model.modules():
        if isinstance(layer, torch.nn.BatchNorm1d):
            layer.eval()
            layer.weight.requires_grad = False
            layer.bias.requires_grad = False

    data, labels = data.to(device), labels.to(device) 
    preds = model.forward(data) 
    loss = criterion(preds, labels) 
    loss.backward() 
    optimizer.step() 
    return loss.item() 

def evaluation(batch, model, criterion, device):
    with torch.no_grad():
        data, labels, *_ = batch
        if isinstance(data, list): data = data[0]
        if isinstance(labels, list): labels = labels[0]
        data, labels = data.to(device), labels.to(device)
        preds = model.forward(data)
        loss = criterion(preds, labels)
    return preds, loss

def graficar_resultados_pacientes(true_means, pred_means, maes, titulo="Por Paciente"):
    """
    Grafica 1 punto por paciente (Promedio Real vs Promedio Predicho).
    Ayuda a ver el desempeño poblacional sin el ruido de cada latido.
    """
    # Convertir a numpy por seguridad
    true_means = np.array(true_means)
    pred_means = np.array(pred_means)
    maes = np.array(maes)
    
    bias_per_patient = pred_means - true_means
    mean_bias = np.mean(bias_per_patient)
    std_bias = np.std(bias_per_patient)

    fig, axs = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(f'Análisis inter-instancia (N={len(true_means)}): {titulo}', fontsize=16)

    # --- 1. Scatter: Promedio Real vs Promedio Predicho ---
    # ¿El modelo detecta pacientes hipertensos?
    axs[0].scatter(true_means, pred_means, alpha=0.5, s=15, c='blue', edgecolors='k', linewidth=0.5)
    
    # Línea ideal
    min_val = min(true_means.min(), pred_means.min())
    max_val = max(true_means.max(), pred_means.max())
    axs[0].plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Ideal')
    
    axs[0].set_title('Regresión: Promedios por Paciente')
    axs[0].set_xlabel('Promedio Real (mmHg)')
    axs[0].set_ylabel('Promedio Predicho (mmHg)')
    axs[0].grid(True, alpha=0.3)
    axs[0].legend()

    # --- 2. Bland-Altman de Promedios ---
    # ¿El error depende de si el paciente es hipertenso?
    means = (true_means + pred_means) / 2
    axs[1].scatter(means, bias_per_patient, alpha=0.5, s=15, c='purple', edgecolors='k', linewidth=0.5)
    
    axs[1].axhline(mean_bias, color='k', ls='-', lw=2, label=f'Bias Global: {mean_bias:.2f}')
    axs[1].axhline(mean_bias + 1.96 * std_bias, color='r', ls='--', label=f'±1.96 SD')
    axs[1].axhline(mean_bias - 1.96 * std_bias, color='r', ls='--')
    
    axs[1].set_title('Bland-Altman (Por Paciente)')
    axs[1].set_xlabel('Presión Arterial Media del Paciente (mmHg)')
    axs[1].set_ylabel('Bias del Paciente (Pred - Real)')
    axs[1].legend()
    axs[1].grid(True, alpha=0.3)

    # --- 3. Histograma de MAE por Paciente ---
    # ¿Cuántos pacientes tienen un error inaceptable?
    axs[2].hist(maes, bins=30, color='orange', edgecolor='black', alpha=0.7)
    axs[2].axvline(5, color='red', linestyle='dashed', linewidth=2, label='Umbral 5 mmHg')
    axs[2].set_title('Distribución de MAE por Paciente')
    axs[2].set_xlabel('MAE del Paciente (mmHg)')
    axs[2].set_ylabel('Cantidad de Pacientes')
    axs[2].legend()

    plt.tight_layout()
    plt.savefig(f'metalearning/pacientes_{titulo}.png', dpi=300)
    print(f"Gráfico guardado: metalearning/pacientes_{titulo}.png")

def main(n_shots=5, base_lr = 5e-3, base_dataset=None, test_patient_ids=None):
    SBP_MEAN = 134.02
    DBP_MEAN = 63.47
    SBP_STD = 22.75
    DBP_STD = 23.69
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- Carga de Datos ---
    if base_dataset is None:
        data_paths = [
        'data/processed/data_UCI/dataset_parte_1_por_picos.pt',
        'data/processed/data_UCI/dataset_parte_2_por_picos.pt',
        'data/processed/data_UCI/dataset_parte_3_por_picos.pt',
        'data/processed/data_UCI/dataset_parte_4_por_picos.pt'
        ]
        dataset_completo = UCIDataset(data_paths)
    else:
        dataset_completo = base_dataset

    if test_patient_ids is None:
        # weights_only=False para evitar warnings
        test_data = torch.load('data/processed/data_UCI/few_shot_patient_data.pt', weights_only=False)
        test_patient_ids = test_data['test_patient_ids']
    else: 
        test_patient_ids = test_patient_ids

    # --- Carga de Modelo ---
    model = Modelo_ConvolucionalV1(in_channels=2, out_channels=2, long_signal=500)
    path_model = 'models/checkpoints/best_meta_model_v1.pt'
    
    print(f"Cargando modelo desde {path_model}...")
    checkpoint = torch.load(path_model, map_location=device, weights_only=False) 
    state_dict = checkpoint['model_state_dict']

    # Limpieza de prefijos 'module.'
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('module.'):
            new_state_dict[k[7:]] = v
        else:
            new_state_dict[k] = v
  
    model.load_state_dict(new_state_dict)
    base_weights = model.state_dict()
    criterion = torch.nn.MSELoss()
    model = model.to(device) 

    taskset = TaskDataset(list_IDs=test_patient_ids, base_dataset=dataset_completo, num_shots=n_shots)

    # Listas globales
    global_metrics_pre_SBP, global_metrics_post_SBP = [], []
    global_metrics_pre_DBP, global_metrics_post_DBP = [], []

    means_true_sbp = []
    means_pred_sbp = []
    maes_sbp = [] 

    means_true_dbp = []
    means_pred_dbp = []
    maes_dbp = []

    mejoraron_sbp = 0
    empeoraron_sbp = 0
    mejoraron_dbp = 0
    empeoraron_dbp = 0

    resultados_por_paciente = []

    print(f"\nIniciando evaluación Few-Shot en {len(taskset.list_IDs)} pacientes...")

    for i in range(len(taskset.list_IDs)):
        id_paciente = taskset.list_IDs[i]
        
        # Reiniciar modelo y optimizador
        model.load_state_dict(base_weights)
        optimizer = torch.optim.Adam(model.parameters(), lr=base_lr)
        id_patient_for_tuning = taskset.list_IDs[i]
  
        tuningset_for_train = TuningNDataset(taskset, id_patient_for_tuning, n_shots=n_shots, validation=False)
        tuningset_for_valid = TuningNDataset(taskset, id_patient_for_tuning, validation=True)

        tuning_loader_TRAIN = torch.utils.data.DataLoader(tuningset_for_train, batch_size=1, shuffle=False)
        tuning_loader_VALID = torch.utils.data.DataLoader(tuningset_for_valid, batch_size=1, shuffle=False)

        # 1. Evaluación PRE Fine-Tuning
        model.eval()
        preds_pre, loss_pre = [], []
        for batch in tuning_loader_VALID:
            p, l = evaluation(batch, model, criterion, device)
            preds_pre.extend(p.detach().cpu().numpy())  
            loss_pre.extend([l.item()]*len(p))
        preds_pre = np.array(preds_pre)

        # 2. Fine-Tuning
        model.train()
        tuning_loss = np.zeros(shape=n_shots)
        for shot_idx, sample in enumerate(tuning_loader_TRAIN):
            tuning_loss[shot_idx] = tuning(sample, optimizer, model, criterion, device)
    
        # 3. Evaluación POST Fine-Tuning
        model.eval()
        preds_post, loss_post = [], []
        for batch in tuning_loader_VALID:
            p, l = evaluation(batch, model, criterion, device)
            preds_post.extend(p.detach().cpu().numpy())
            loss_post.extend([l.item()]*len(p))
        preds_post = np.array(preds_post)

        # Desnormalización
        pred_pre_flat = preds_pre.reshape(-1, 2)
        pred_post_flat = preds_post.reshape(-1, 2)
        labels_flat = np.array([l.squeeze().cpu().numpy() for l in tuning_loader_VALID.dataset.labels]).reshape(-1, 2)

        pred_pre_SBP = desnormalizar_zscore(pred_pre_flat[:,0], SBP_MEAN, SBP_STD)
        pred_pre_DBP = desnormalizar_zscore(pred_pre_flat[:,1], DBP_MEAN, DBP_STD)
        pred_post_SBP = desnormalizar_zscore(pred_post_flat[:,0], SBP_MEAN, SBP_STD)
        pred_post_DBP = desnormalizar_zscore(pred_post_flat[:,1], DBP_MEAN, DBP_STD)
        true_SBP = desnormalizar_zscore(labels_flat[:,0], SBP_MEAN, SBP_STD)
        true_DBP = desnormalizar_zscore(labels_flat[:,1], DBP_MEAN, DBP_STD)
        
        # Métricas (mae, rmse, bias, std)
        m_pre_sbp = calcular_metricas(true_SBP, pred_pre_SBP)
        m_post_sbp = calcular_metricas(true_SBP, pred_post_SBP)
        m_pre_dbp = calcular_metricas(true_DBP, pred_pre_DBP)
        m_post_dbp = calcular_metricas(true_DBP, pred_post_DBP)

        global_metrics_pre_SBP.append(m_pre_sbp)
        global_metrics_post_SBP.append(m_post_sbp)
        global_metrics_pre_DBP.append(m_pre_dbp)
        global_metrics_post_DBP.append(m_post_dbp)

        means_true_sbp.append(np.mean(true_SBP))
        means_pred_sbp.append(np.mean(pred_post_SBP))
        maes_sbp.append(m_post_sbp[0]) # MAE del paciente

        means_true_dbp.append(np.mean(true_DBP))
        means_pred_dbp.append(np.mean(pred_post_DBP))
        maes_dbp.append(m_post_dbp[0])  

        if m_post_sbp[0] < m_pre_sbp[0]: mejoraron_sbp += 1
        else: empeoraron_sbp += 1

        if m_post_dbp[0] < m_pre_dbp[0]: mejoraron_dbp += 1
        else: empeoraron_dbp += 1

        # Guardar historial completo por paciente
        resultados_por_paciente.append({
            "paciente": id_paciente,
            # SBP
            'mae_pre_sbp': float(m_pre_sbp[0]),
            'mae_post_sbp': float(m_post_sbp[0]),
            'rmse_pre_sbp': float(m_pre_sbp[1]),
            'rmse_post_sbp': float(m_post_sbp[1]),
            'iso_bias_pre_sbp': float(m_pre_sbp[2]),
            'iso_bias_post_sbp': float(m_post_sbp[2]), 
            'iso_std_pre_sbp': float(m_pre_sbp[3]),
            'iso_std_post_sbp': float(m_post_sbp[3]),
            # DBP
            'mae_pre_dbp': float(m_pre_dbp[0]),
            'mae_post_dbp': float(m_post_dbp[0]),
            'rmse_pre_dbp': float(m_pre_dbp[1]),
            'rmse_post_dbp': float(m_post_dbp[1]),
            'iso_bias_pre_dbp': float(m_pre_dbp[2]),
            'iso_bias_post_dbp': float(m_post_dbp[2]), 
            'iso_std_pre_dbp': float(m_pre_dbp[3]),
            'iso_std_post_dbp': float(m_post_dbp[3]),
        })

        print(f"Paciente {id_patient_for_tuning} | SBP MAE: {m_pre_sbp[0]:.2f}->{m_post_sbp[0]:.2f} | RMSE: {m_pre_sbp[1]:.2f}->{m_post_sbp[1]:.2f} | ISO: {m_pre_sbp[2]:.2f}±{m_pre_sbp[3]:.2f} -> {m_post_sbp[2]:.2f}±{m_post_sbp[3]:.2f}")

    # --- REPORTE GLOBAL ---
    avg_pre_SBP = promedio_metricas(global_metrics_pre_SBP)
    avg_post_SBP = promedio_metricas(global_metrics_post_SBP)
    avg_pre_DBP = promedio_metricas(global_metrics_pre_DBP)
    avg_post_DBP = promedio_metricas(global_metrics_post_DBP)
    
    total = len(taskset.list_IDs)
    tasa_mejora_sbp = mejoraron_sbp / total
    tasa_mejora_dbp = mejoraron_dbp / total

    print("\n" + "="*60)
    print("       RESULTADOS FINALES GLOBAL (Promedio Pacientes)")
    print("="*60)
    
    # SBP REPORT
    print("\n--- SISTÓLICA (SBP) ---")
    print(f"Ingeniería (MAE)   : {avg_pre_SBP[0]:.2f} -> {avg_post_SBP[0]:.2f} mmHg")
    print(f"Ingeniería (RMSE)  : {avg_pre_SBP[1]:.2f} -> {avg_post_SBP[1]:.2f} mmHg")
    print(f"Clínica (ISO Bias) : {avg_pre_SBP[2]:.2f} -> {avg_post_SBP[2]:.2f} mmHg")
    print(f"Clínica (ISO STD)  : {avg_pre_SBP[3]:.2f} -> {avg_post_SBP[3]:.2f} mmHg")
    print(f"RESUMEN ISO FINAL  : {avg_post_SBP[2]:.2f} ± {avg_post_SBP[3]:.2f} mmHg (Meta: <= 5 ± 8)")

    # DBP REPORT
    print("\n--- DIASTÓLICA (DBP) ---")
    print(f"Ingeniería (MAE)   : {avg_pre_DBP[0]:.2f} -> {avg_post_DBP[0]:.2f} mmHg")
    print(f"Ingeniería (RMSE)  : {avg_pre_DBP[1]:.2f} -> {avg_post_DBP[1]:.2f} mmHg")
    print(f"RESUMEN ISO FINAL  : {avg_post_DBP[2]:.2f} ± {avg_post_DBP[3]:.2f} mmHg (Meta: <= 5 ± 8)")

    print("\n--- CONSISTENCIA ---")
    print(f"Tasa Mejora SBP: {(tasa_mejora_sbp)*100:.1f}% ({mejoraron_sbp}/{total})")
    print(f"Tasa Mejora DBP: {(tasa_mejora_dbp)*100:.1f}% ({mejoraron_dbp}/{total})")
    
    graficar_resultados_pacientes(means_true_sbp, means_pred_sbp, maes_sbp, titulo="SBP")
    graficar_resultados_pacientes(means_true_dbp, means_pred_dbp, maes_dbp, titulo="DBP")

    # DICCIONARIO COMPLETO (PRE Y POST PARA COMPARAR)
    resultados = {
        # SBP PRE
        "mae_pre_sbp": float(avg_pre_SBP[0]),
        "rmse_pre_sbp": float(avg_pre_SBP[1]),
        "iso_bias_pre_sbp": float(avg_pre_SBP[2]),
        "iso_std_pre_sbp": float(avg_pre_SBP[3]),
        # SBP POST
        "mae_post_sbp": float(avg_post_SBP[0]),
        "rmse_post_sbp": float(avg_post_SBP[1]),
        "iso_bias_post_sbp": float(avg_post_SBP[2]),
        "iso_std_post_sbp": float(avg_post_SBP[3]),
        
        # DBP PRE
        "mae_pre_dbp": float(avg_pre_DBP[0]),
        "rmse_pre_dbp": float(avg_pre_DBP[1]),
        "iso_bias_pre_dbp": float(avg_pre_DBP[2]),
        "iso_std_pre_dbp": float(avg_pre_DBP[3]),
        # DBP POST
        "mae_post_dbp": float(avg_post_DBP[0]),
        "rmse_post_dbp": float(avg_post_DBP[1]),
        "iso_bias_post_dbp": float(avg_post_DBP[2]),
        "iso_std_post_dbp": float(avg_post_DBP[3]),

        # Métricas de Mejora
        "tasa_mejora_sbp": tasa_mejora_sbp,
        "tasa_mejora_dbp": tasa_mejora_dbp,
        "resultados_por_paciente": resultados_por_paciente
    }
    return resultados

if __name__ == '__main__':
    main()