"""
Módulo: Respuesta_escalon.py
Autor: Juan Marcos Grigolatto
Descripción: Simulación de uso clínico en entorno real (Monitoreo Continuo). 
             Evalúa la "respuesta al escalón" del modelo tras aplicar un Ajuste 
             Inicial Único (Few-Shot Fine-Tuning) utilizando solo las primeras 
             N muestras del paciente. Congela las capas convolucionales (extracción 
             de características) y estadísticas de normalización (BatchNorm) para 
             evitar el sobreajuste del modelo, actualizando únicamente los pesos del 
             regresor final. 
"""
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
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
import os

def calcular_metricas_avanzadas(y_true, y_pred):
    """_summary_ Calcula métricas avanzadas de error: MAE, RMSE, Bias y Desviación Estándar de los errores.

    Args: 
        y_true (_type_): _description_ Valores reales de presión arterial (mmHg)
        y_pred (_type_): _description_ Valores predichos por el modelo (mmHg)

    Returns:
        _type_: _description_ Tupla con las métricas: (MAE, RMSE, Bias, SD)
    """    
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

def tuning(sample, optimizer, model, criterion, device):
    """_summary_ Realiza un paso de ajuste (tuning) del modelo sobre un batch de datos, actualizando únicamente los pesos del regresor final, mientras mantiene congeladas las capas convolucionales y estadísticas de normalización.

    Args:
        sample (_type_): _description_ Batch de datos que contiene señales y etiquetas (SBP, DBP) normalizadas.
        optimizer (_type_): _description_ Optimizador utilizado para actualizar los pesos del regresor final.
        model (_type_): _description_ Modelo de red neuronal previamente entrenado y cargado.
        criterion (_type_): _description_ Función de pérdida utilizada para calcular el error.
        device (_type_): _description_ Dispositivo de cómputo (CPU o GPU) donde se realizará el ajuste.

    Returns:
        _type_: _description_ Pérdida calculada durante el ajuste.
    """    
    optimizer.zero_grad() 
    data, labels, *_ = sample 
    
    if isinstance(data, list): data = data[0]
    if isinstance(labels, list): labels = labels[0]
        
    for layer in model.modules():
        if isinstance(layer, torch.nn.BatchNorm1d) or isinstance(layer, torch.nn.BatchNorm2d):
            layer.eval()

    data, labels = data.to(device), labels.to(device) 

    preds = model.forward(data) 
    loss = criterion(preds, labels) 
    loss.backward() 
    optimizer.step() 
    return loss.item()

def main(n_shots=5, n_epochs=5, lr=5e-3, MIN_SEÑALES_REQUERIDAS=500):
    """_summary_ Función principal que ejecuta la simulación de respuesta al escalón con ajuste inicial único. Permite configurar el número de shots, épocas de ajuste, tasa de aprendizaje y el mínimo de señales requeridas para evaluar la respuesta.

    Args:
        n_shots (int, optional): _description_. Por defecto 5. Número de muestras (shots) utilizadas para la evaluación intrapatient, aunque el modelo no se ajusta con ellas (zero-shot), se muestran en las gráficas para referencia. 
        n_epochs (int, optional): _description_. Por defecto 5. Número de épocas de ajuste (fine-tuning) realizadas sobre el paciente utilizando solo las primeras N muestras (shots).
        lr (_type_, optional): _description_. Por defecto 5e-3. Tasa de aprendizaje utilizada para el optimizador durante el ajuste inicial.
    """    
    
    USE_DELTA_LEARNING = False 
    
    PACIENTES_OBJETIVO = [101, 2041, 8423, 1126] 
    
    if USE_DELTA_LEARNING:
        NOMBRE_EXPERIMENTO = "PRUEBA_AJUSTE_UNICO_DELTA"
        PATH_MODELO = 'models/checkpoints/best_meta_DELTA_LEARNING_refine_alpha50.pt'
    else:
        NOMBRE_EXPERIMENTO = "PRUEBA_AJUSTE_UNICO_tradicional"
        PATH_MODELO = 'models/checkpoints/best_meta_model_v1.pt' 

    print(f"--- INICIANDO PRUEBA CON AJUSTE INICIAL ÚNICO (5 SHOTS) ---")
    print(f"Modo Delta Learning: {USE_DELTA_LEARNING}")
    print(f"Modelo: {PATH_MODELO}")
    
    SEED = 42
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
    if torch.cuda.is_available(): torch.cuda.manual_seed(SEED)

    save_dir_graficas = f"resultados_intrapatient/{NOMBRE_EXPERIMENTO}"
    os.makedirs(save_dir_graficas, exist_ok=True)
    
    SBP_MEAN, SBP_STD = 134.02, 22.75
    DBP_MEAN, DBP_STD = 63.47, 23.69

    test_data = torch.load('data/processed/data_UCI/few_shot_patient_data.pt', weights_only=False)
    test_patient_ids = test_data['test_patient_ids']
    data_paths = [
        'data/processed/data_UCI/dataset_parte_1_por_picos.pt',
        'data/processed/data_UCI/dataset_parte_2_por_picos.pt',
        'data/processed/data_UCI/dataset_parte_3_por_picos.pt',
        'data/processed/data_UCI/dataset_parte_4_por_picos.pt'
    ]
    dataset_completo = UCIDataset(data_paths)

    model = Modelo_ConvolucionalV1(in_channels=2, out_channels=2, long_signal=500)
    if not os.path.exists(PATH_MODELO):
        print(f"¡ERROR! No encuentro el modelo en {PATH_MODELO}")
        return
        
    checkpoint = torch.load(PATH_MODELO, map_location=torch.device('cpu'), weights_only=False)
    state_dict = checkpoint['model_state_dict']
    new_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    model.load_state_dict(new_state_dict, strict=False)

    criterion = torch.nn.MSELoss() 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)  

    taskset = TaskDataset(list_IDs=PACIENTES_OBJETIVO, base_dataset=dataset_completo, num_shots=n_shots)
    mapa_indices_pacientes = taskset.patient_to_indices

    pacientes_seleccionados = [pid for pid in PACIENTES_OBJETIVO if pid in mapa_indices_pacientes]

    for id_paciente in pacientes_seleccionados:
        print(f"\n >> PROCESANDO PACIENTE: {id_paciente}")
        
        model.load_state_dict(new_state_dict, strict=False)
        
        for param in model.parameters(): 
            param.requires_grad = False  
            
        for name, param in model.named_parameters():
            if 'dense' in name: 
                param.requires_grad = True 

        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
        
        dataset_paciente_completo = Intrapatientset(
            patient_id=id_paciente,
            base_dataset=dataset_completo,
            patient_to_indices_map=mapa_indices_pacientes
        )

        loader_paciente = torch.utils.data.DataLoader(
            dataset_paciente_completo, batch_size=n_shots, shuffle=False, num_workers=0, drop_last=False
        )

        historial_sbp = {'real': [], 'pred': []}
        historial_dbp = {'real': [], 'pred': []}
        
        bias_tensor_guardado = None

        for i, (batch_signals, batch_labels) in enumerate(loader_paciente):
            
            if i == 0:
                if USE_DELTA_LEARNING:
                    bias_tensor_guardado = batch_labels.mean(dim=0, keepdim=True)
                    labels_for_tuning = batch_labels - bias_tensor_guardado
                else:
                    labels_for_tuning = batch_labels

                batch_data_tune = (batch_signals, labels_for_tuning)

                model.train()
                for _ in range(n_epochs): 
                    _ = Fewshot.tuning(batch_data_tune, optimizer, model, criterion, device)
                
                print("    -> Ajuste inicial completado.")

            if USE_DELTA_LEARNING:
                labels_for_eval = batch_labels - bias_tensor_guardado
            else:
                labels_for_eval = batch_labels
                
            batch_data_eval = (batch_signals, labels_for_eval)

            model.eval()
            preds_model, _ = Fewshot.evaluation(batch_data_eval, model, criterion, device) 
            
            if USE_DELTA_LEARNING:
                preds_absolutas = preds_model.detach().cpu() + bias_tensor_guardado
            else:
                preds_absolutas = preds_model.detach().cpu()
            
            pred_sbp = Fewshot.desnormalizar_zscore(preds_absolutas[:, 0].numpy(), SBP_MEAN, SBP_STD)
            pred_dbp = Fewshot.desnormalizar_zscore(preds_absolutas[:, 1].numpy(), DBP_MEAN, DBP_STD)
            true_sbp = Fewshot.desnormalizar_zscore(batch_labels[:, 0].numpy(), SBP_MEAN, SBP_STD)
            true_dbp = Fewshot.desnormalizar_zscore(batch_labels[:, 1].numpy(), DBP_MEAN, DBP_STD)

            historial_sbp['real'].extend(true_sbp); historial_sbp['pred'].extend(pred_sbp)
            historial_dbp['real'].extend(true_dbp); historial_dbp['pred'].extend(pred_dbp)

        mae_s, rmse_s, bias_s, std_s = calcular_metricas_avanzadas(historial_sbp['real'], historial_sbp['pred'])
        mae_d, rmse_d, bias_d, std_d = calcular_metricas_avanzadas(historial_dbp['real'], historial_dbp['pred'])

        print(f"   [SBP] RMSE: {rmse_s:.2f} | Bias: {bias_s:.2f} | SD: {std_s:.2f}")
        
        modo_str = "Delta" if USE_DELTA_LEARNING else "Absoluto"
        
        plt.rcParams.update({'font.size': 12, 'font.family': 'serif'})
        fig, axs = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
        x_axis = range(len(historial_sbp['real']))
        
        axs[0].plot(x_axis, historial_sbp['real'], color='black', linewidth=1.5, label='Invasiva de Referencia (ABP)', alpha=0.8)
        axs[0].plot(x_axis, historial_sbp['pred'], color='tab:orange', linestyle='--', linewidth=2, label=f'Estimación ({modo_str})', alpha=0.9)
        axs[0].axvline(x=n_shots, color='red', linestyle=':', linewidth=2.5, alpha=0.8, label='Ajuste')
        
        axs[0].set_title(f"Respuesta dinámica con Ajuste Inicial Único ({modo_str}) - SBP", fontweight='bold')
        axs[0].set_ylabel("Presión Sistólica (mmHg)", fontweight='bold')
        axs[0].legend(loc='upper right', framealpha=0.9)
        axs[0].grid(True, linestyle='--', alpha=0.5)

        axs[1].plot(x_axis, historial_dbp['real'], color='black', linewidth=1.5, label='Invasiva de Referencia (ABP)', alpha=0.8)
        axs[1].plot(x_axis, historial_dbp['pred'], color='tab:cyan', linestyle='--', linewidth=2, label=f'Estimación ({modo_str})', alpha=0.9)
        axs[1].axvline(x=n_shots, color='red', linestyle=':', linewidth=2.5, alpha=0.8)
        
        axs[1].set_title(f"Respuesta dinámica con Ajuste Inicial Único ({modo_str}) - DBP", fontweight='bold')
        axs[1].set_xlabel("Muestras", fontweight='bold')
        axs[1].set_ylabel("Presión Diastólica (mmHg)", fontweight='bold')
        axs[1].legend(loc='upper right', framealpha=0.9)
        axs[1].grid(True, linestyle='--', alpha=0.5)
        
        plt.tight_layout()
        save_path = os.path.join(save_dir_graficas, f"step_response_{modo_str.lower()}_{id_paciente}.png")
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"   Gráfica guardada: {save_path}")

    print("\n--- PRUEBA FINALIZADA ---")

if __name__ == '__main__':
    main()