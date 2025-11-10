import torch.utils
from src.data.data_chargers.Clase_UCIDataset import UCIDataset
from src.data.data_chargers.MetaDataset import TaskDataset
from src.models.Modelo_conv import Modelo_Convolucional
from src.models.ConvolucionalV1 import Modelo_ConvolucionalV1
import numpy as np
from torch import nn, optim
import matplotlib.pyplot as plt
#from src.models.InceptionTime import InceptionTime
import torch.utils.data as data
import torch
from src.data.data_chargers.Tuningndataset import TuningNDataset
import random
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error, mean_squared_error

def promedio_metricas(m_list):
    return np.mean(m_list, axis=0)

def calcular_metricas(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    bias = np.mean(y_pred - y_true)  # Sesgo sistemático promedio
    r2 = r2_score(y_true, y_pred)
    return mae, rmse, bias, r2

def desnormalizar_zscore(norm_array, media, std):
    return norm_array * std + media

def desnormalizar_minmax(norm_array, min_val, max_val):
    return norm_array * (max_val - min_val) + min_val

def tuning(sample, optimizer, model, criterion, device, tipo_presion="SBP"):
        optimizer.zero_grad() # Reinicia los gradientes
        data, labels, *_ = sample # Obtiene los datos y etiquetas
        if tipo_presion == "SBP":
            etiqueta = sample[:, 0].unsqueeze(1)
        else:
            etiqueta = sample[:, 1].unsqueeze(1)        

        # Si vienen como listas (por batch_size=1), convertirlos a tensores
        if isinstance(data, list):
            data = data[0]
        if isinstance(etiqueta, list):
            labels = etiqueta[0]
            # Congelar todas las capas BatchNorm
        for layer in model.modules():
            if isinstance(layer, torch.nn.BatchNorm1d):
                layer.eval()
                layer.weight.requires_grad = False
                layer.bias.requires_grad = False

        data, labels = data.to(device), labels.to(device) # Mueve los datos y etiquetas a la GPU
        preds = model.forward(data) # Realiza la predicción
        loss = criterion(preds, labels) # Calcula la pérdida
        loss.backward() # Calcula los gradientes mediante backpropagation
        optimizer.step() # Actualiza los parámetros del modelo
        return loss.item() # Devuelve la pérdida

def evaluation(batch, model, criterion, device, tipo_presion="SBP"):  
    with torch.no_grad():
        data, labels, *_ = batch
        if tipo_presion == "SBP":
            etiqueta = labels[:, 0].unsqueeze(1)
        else:
            etiqueta = labels[:, 1].unsqueeze(1)

        # Si vienen como listas (por batch_size=1), convertirlos a tensores
        if isinstance(data, list):
            data = data[0]
        if isinstance(etiqueta, list):
            labels = etiqueta[0]
        data, labels = data.to(device), labels.to(device)
        preds = model.forward(data)
        loss = criterion(preds, labels)
    return preds, loss

def main(n_shots=5, base_lr = 1e-6, base_dataset=None, test_patient_ids=None, tipo_presion="SBP"):
    SBP_MEAN = 134.02
    DBP_MEAN = 63.47
    SBP_STD = 22.75
    DBP_STD = 23.69
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
        test_data = torch.load('data/processed/data_UCI/few_shot_patient_data.pt')
        test_patient_ids = test_data['test_patient_ids']
    
    else: 
        test_patient_ids = test_patient_ids

    #Carga de modelo metaentrenado 
    #model=Modelo_Convolucional(in_channels=2,out_channels=2, long_signal=500)
    model=Modelo_ConvolucionalV1(in_channels=2,out_channels=1, long_signal=500)
    path_model='models/best_meta_models/best_meta_model_sbp_patientwise.pt'
    checkpoint = torch.load(path_model, map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint['model_state_dict'])

    base_weights = model.state_dict()  # Guarda los pesos iniciales del modelo

    # Criterio de pérdida
    criterion = torch.nn.MSELoss()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #Evaluación modelo previo a fine tuning
    model = model.to(device)  # Mueve el modelo a la GPU

    
    #optimizer = torch.optim.Adam(model.parameters(), lr=1e-3) 
    # Definicón del optimizador SOLO con los parámetros entrenables
    #optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4)
    #optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    taskset = TaskDataset(list_IDs=test_patient_ids, base_dataset=dataset_completo, num_shots=n_shots)
    #id_patient_for_tuning =  random.choice(test_patient_ids)
    


    global_metrics_pre, global_metrics_post = [], []

    mejoraron = 0
    empeoraron = 0

    resultados_por_paciente = []

    for i in range(len(taskset.list_IDs)):
        id_paciente = taskset.list_IDs[i]
        preds_post_fine_tuning = []
        loss_post_fine_tuning = []

        preds_pre_fine_tuning = []
        loss_pre_fine_tuning = []
        
        model.load_state_dict(base_weights)  # Reinicia los pesos del modelo antes de cada fine-tuning

        optimizer = torch.optim.Adam(model.parameters(), lr=base_lr)
        id_patient_for_tuning =  taskset.list_IDs[i]

  
        tuningset_for_train=TuningNDataset(taskset, id_patient_for_tuning, n_shots=n_shots, validation=False)
        tuningset_for_valid=TuningNDataset(taskset, id_patient_for_tuning, validation=True)

        #print(f"ID paciente para fine tuning: {id_patient_for_tuning}")
        print(f"\n=== Fine-Tuning Paciente {id_patient_for_tuning} ({i+1}/{len(taskset.list_IDs)}) ===")

        tuning_dataloader_TRAIN=torch.utils.data.DataLoader(tuningset_for_train, batch_size=1, shuffle=False)
        tuning_dataloader_VALID=torch.utils.data.DataLoader(tuningset_for_valid, batch_size=1, shuffle=False)

        #Estrategia de evaluación previa al fine tuning

        # 1) Fine-tuning completo
        #for param in model.parameters():
        #    param.requires_grad = True

        # 2) Solo dense layers (Congelar capas de convolución)
        """
        for param in model.parameters():
            param.requires_grad = False
        for param in model.dense1.parameters():
            param.requires_grad = True
        for param in model.dense2.parameters():
            param.requires_grad = True
        """

        # 3) Congelar BatchNorm (Error con batch size = 1 si no se congela)
        """
        for m in model.modules():
            if isinstance(m, torch.nn.BatchNorm1d):
                m.eval()
                for param in m.parameters():
                    param.requires_grad = False
        """
        #Fine tunning N-way K-shot    
    
        model.eval()    
    
        for batch in tuning_dataloader_VALID:
        #    preds_pre_fine_tuning[k] ,loss_pre_fine_tuning[k] = evaluation(batch, model, criterion, device)
            preds, loss = evaluation(batch, model, criterion, device, tipo_presion=tipo_presion)
            preds_pre_fine_tuning.extend(preds.detach().cpu().numpy())  
            loss_pre_fine_tuning.extend([loss.item()]*len(preds))

        preds_pre_fine_tuning = np.array(preds_pre_fine_tuning)
        loss_pre_fine_tuning = np.array(loss_pre_fine_tuning)
    
        labels = np.array([l.squeeze().cpu().numpy() for l in tuning_dataloader_VALID.dataset.labels])
        """
        fig, ax = plt.subplots(figsize=(7, 4), tight_layout=True)
        ax.plot(loss_pre_fine_tuning, label='Loss pre fine tuning')
        ax.set_xlabel('number of sample')
        ax.set_ylabel('Loss')
        ax.legend()
        plt.savefig('metalearning/loss_pre_fine_tuning.png')
        plt.show()
        """
        model.train()
        tuning_loss = np.zeros(shape=n_shots)

        for shot_idx, sample in enumerate(tuning_dataloader_TRAIN):
            
            tuning_loss[shot_idx] = tuning(sample, optimizer, model, criterion, device, tipo_presion=tipo_presion)
    
        torch.save({'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'loss': tuning_loss[n_shots-1]}, 
                        'models/tuning_model.pt')
        """
        fig, ax = plt.subplots(figsize=(7, 4), tight_layout=True)
        ax.plot(tuning_loss, label='Fine Tuning for 5 shots')
        ax.set_xlabel('number of shot')
        ax.set_ylabel('Loss')
        ax.legend()
        plt.savefig('metalearning/loss_curve_fine_tuning.png')
        plt.show()
        """
    
        #Evaluación de modelo posterior al fine tuning

        #preds_post_fine_tuning = np.zeros((len(tuning_dataloader_VALID), n_shots, 2))
        #loss_post_fine_tuning = np.zeros(len(tuning_dataloader_VALID))

        model.eval()    
        for batch in tuning_dataloader_VALID:
            #preds_post_fine_tuning[j], loss_post_fine_tuning[j] = evaluation(batch, model, criterion, device)
            preds, loss = evaluation(batch, model, criterion, device, tipo_presion=tipo_presion)
            preds_post_fine_tuning.extend(preds.detach().cpu().numpy())
            loss_post_fine_tuning.extend([loss.item()]*len(preds))

        preds_post_fine_tuning = np.array(preds_post_fine_tuning)
        loss_post_fine_tuning = np.array(loss_post_fine_tuning)
        labels = labels.reshape(-1)
        #Desnormalización
        """
        pred_pre_SBP_norm = desnormalizar_zscore(preds_pre_fine_tuning[:, 0], SBP_MEAN, SBP_STD)
        pred_pre_DBP_norm = desnormalizar_zscore(preds_pre_fine_tuning[:, 1], DBP_MEAN, DBP_STD)
    
        pred_post_SBP_norm = desnormalizar_zscore(preds_post_fine_tuning[:, 0], SBP_MEAN, SBP_STD)
        pred_post_DBP_norm = desnormalizar_zscore(preds_post_fine_tuning[:, 1], DBP_MEAN, DBP_STD)

        true_SBP_norm = desnormalizar_zscore(labels[:, 0], SBP_MEAN, SBP_STD)
        true_DBP_norm = desnormalizar_zscore(labels[:, 1], DBP_MEAN, DBP_STD)
        """
        pred_pre = desnormalizar_zscore(preds_pre_fine_tuning.reshape(-1), SBP_MEAN, SBP_STD)
        pred_post = desnormalizar_zscore(preds_post_fine_tuning.reshape(-1), SBP_MEAN, SBP_STD)
        true_vals = desnormalizar_zscore(labels, SBP_MEAN, SBP_STD)

    
        """
        fig, ax = plt.subplots(figsize=(7, 4), tight_layout=True)
        ax.plot(loss_post_fine_tuning, label='Loss post fine tuning')
        ax.set_xlabel('number of sample')
        ax.set_ylabel('Loss')
        ax.legend()
        plt.savefig('metalearning/loss_post_fine_tuning.png')
        plt.show()

        # --- 1. Real vs Predicho ---
        fig, axs = plt.subplots(2, 2, figsize=(12, 5))
        axs[0][0].scatter(true_SBP_norm, pred_pre_SBP_norm, alpha=0.5)
        axs[0][0].plot([true_SBP_norm.min(), true_SBP_norm.max()],
                [true_SBP_norm.min(), true_SBP_norm.max()], 'r--')
        axs[0][0].set_title(f"SBP - Antes del Fine-Tuning")
        axs[0][0].set_xlabel("Valor verdadero")
        axs[0][0].set_ylabel("Predicción")
        axs[0][0].grid(True)

        axs[0][1].scatter(true_DBP_norm, pred_pre_DBP_norm, alpha=0.5)
        axs[0][1].plot([true_DBP_norm.min(), true_DBP_norm.max()],
                [true_DBP_norm.min(), true_DBP_norm.max()], 'r--')
        axs[0][1].set_title(f"DBP - Antes del Fine-Tuning")
        axs[0][1].set_xlabel("Valor verdadero")
        axs[0][1].set_ylabel("Predicción")
        axs[0][1].grid(True)

        axs[1][0].scatter(true_SBP_norm, pred_post_SBP_norm, alpha=0.5)
        axs[1][0].plot([true_SBP_norm.min(), true_SBP_norm.max()],
            [true_SBP_norm.min(), true_SBP_norm.max()], 'r--')
        axs[1][0].set_title(f"SBP - Despues del Fine-Tuning")
        axs[1][0].set_xlabel("Valor verdadero")
        axs[1][0].set_ylabel("Predicción")
        axs[1][0].grid(True)
    
        axs[1][1].scatter(true_DBP_norm, pred_post_DBP_norm, alpha=0.5)
        axs[1][1].plot([true_DBP_norm.min(), true_DBP_norm.max()],
             [true_DBP_norm.min(), true_DBP_norm.max()], 'r--')
        axs[1][1].set_title(f"DBP - Despues del Fine-Tuning")
        axs[1][1].set_xlabel("Valor verdadero")
        axs[1][1].set_ylabel("Predicción")
        axs[1][1].grid(True)

        plt.tight_layout()
        plt.show()

        # --- 2. Residuos ---
        residuals_pre_SBP = pred_pre_SBP_norm - true_SBP_norm
        residuals_pre_DBP = pred_pre_DBP_norm - true_DBP_norm
        residuals_post_SBP = pred_post_SBP_norm - true_SBP_norm
        residuals_post_DBP = pred_post_DBP_norm - true_DBP_norm

        fig, axs = plt.subplots(2, 2, figsize=(12, 5))
        axs[0][0].scatter(true_SBP_norm, residuals_pre_SBP, alpha=0.5)
        axs[0][0].axhline(0, color='red', linestyle='--')
        axs[0][0].set_title("Antes - Residuos - SBP")
        axs[0][0].set_xlabel("Valor verdadero")
        axs[0][0].set_ylabel("Error (Pred - Real)")
        axs[0][0].grid(True)

        axs[0][1].scatter(true_DBP_norm, residuals_pre_DBP, alpha=0.5)
        axs[0][1].axhline(0, color='red', linestyle='--')
        axs[0][1].set_title("Antes - Residuos - DBP")
        axs[0][1].set_xlabel("Valor verdadero")
        axs[0][1].set_ylabel("Error (Pred - Real)")
        axs[0][1].grid(True)
    
        axs[1][0].scatter(true_SBP_norm, residuals_post_SBP, alpha=0.5)
        axs[1][0].axhline(0, color='red', linestyle='--')
        axs[1][0].set_title("Después - Residuos -SBP")
        axs[1][0].set_xlabel("Valor verdadero")
        axs[1][0].set_ylabel("Error (Pred - Real)")
        axs[1][0].grid(True)

        axs[1][1].scatter(true_DBP_norm, residuals_post_DBP, alpha=0.5)
        axs[1][1].axhline(0, color='red', linestyle='--')
        axs[1][1].set_title("Después - Residuos - DBP") 
        axs[1][1].set_xlabel("Valor verdadero")
        axs[1][1].set_ylabel("Error (Pred - Real)")
        axs[1][1].grid(True)

        plt.tight_layout()
        plt.show()

        # --- 3. Histograma de errores ---
        fig, axs = plt.subplots(2, 2, figsize=(12, 5))
        axs[0][0].hist(residuals_pre_SBP, bins=50, edgecolor='black')
        axs[0][0].set_title("Antes - Histograma de errores -SBP")
        axs[0][0].set_xlabel("Error (Pred - Real)")
        axs[0][0].set_ylabel("Frecuencia")
        axs[0][0].grid(True)

        axs[0][1].hist(residuals_pre_DBP, bins=50, edgecolor='black')
        axs[0][1].set_title("Antes - Histograma de errores -DBP")
        axs[0][1].set_xlabel("Error (Pred - Real)")
        axs[0][1].set_ylabel("Frecuencia")
        axs[0][1].grid(True)

        axs[1][0].hist(residuals_post_SBP, bins=50, edgecolor='black')
        axs[1][0].set_title("Después - Histograma de errores -SBP")
        axs[1][0].set_xlabel("Error (Pred - Real)")
        axs[1][0].set_ylabel("Frecuencia")
        axs[1][0].grid(True)

        axs[1][1].hist(residuals_post_DBP, bins=50, edgecolor='black')
        axs[1][1].set_title("Después - Histograma de errores -DBP")
        axs[1][1].set_xlabel("Error (Pred - Real)")
        axs[1][1].set_ylabel("Frecuencia")
        axs[1][1].grid(True)

        plt.tight_layout()
        plt.show()
        """

        metrics_pre = calcular_metricas(true_vals, pred_pre)
        metrics_post = calcular_metricas(true_vals, pred_post)

        global_metrics_pre.append(metrics_pre)
        global_metrics_post.append(metrics_post)

        mae_pre = metrics_pre[0]
        mae_post = metrics_post[0]
    
        if mae_post < mae_pre:
            mejoraron += 1
        else:
            empeoraron += 1

        resultados_por_paciente.append({
            "paciente": id_paciente,
            'patient_id': int(id_patient_for_tuning),
            'mae_pre': float(mae_pre),
            'mae_post': float(mae_post)
        })

        print(f"\n=== Paciente {id_patient_for_tuning} ===")
        print(" Antes / Después")
        print(f"MAE: {metrics_pre[0]:.2f} → {metrics_post[0]:.2f}")
        print(f"RMSE: {metrics_pre[1]:.2f} → {metrics_post[1]:.2f}")
        print(f"R²: {metrics_pre[3]:.3f} → {metrics_post[3]:.3f}")
        """
        if i == len(taskset.list_IDs) - 1:
            residuals_pre = pred_pre_SBP - true_SBP
            residuals_post = pred_post_SBP - true_SBP
            plt.figure(figsize=(6,4))
            plt.scatter(true_SBP, residuals_pre, alpha=0.5, label="Pre-FSL")
            plt.scatter(true_SBP, residuals_post, alpha=0.5, label="Post-FSL")
            plt.axhline(0, color='r', linestyle='--')
            plt.title(f"Residuos SBP - Paciente {id_patient_for_tuning}")
            plt.xlabel("SBP real (mmHg)")
            plt.ylabel("Error (Pred - Real)")
            plt.legend()
            plt.show()
        """

    avg_pre = promedio_metricas(global_metrics_pre)
    avg_post = promedio_metricas(global_metrics_post)

    print("\n===============================")
    print("=== MÉTRICAS GLOBALES (PROMEDIO ENTRE PACIENTES) ===")
    print("===============================")
    print(f" MAE: {avg_pre[0]:.2f} → {avg_post[0]:.2f}  | RMSE: {avg_pre[1]:.2f} → {avg_post[1]:.2f}  | R²: {avg_pre[3]:.3f} → {avg_post[3]:.3f}")
    

    # Estimación global de mejora porcentual
    mejora_global = ((avg_pre - avg_post) / np.abs(avg_pre)) * 100
   

    print("\n=== Mejora global Few-Shot (%) ===")
    print(f"MAE: {mejora_global[0]:.2f}%, RMSE: {mejora_global[1]:.2f}%, R²: {mejora_global[3]:.2f}%")
    

    total_pacientes = len(taskset.list_IDs)
    tasa_mejora = (mejoraron / total_pacientes) * 100

    print("\n=== Pacientes: mejora (MAE) ===")
    print(f"Evaluados: {total_pacientes}")
    print(f" Mejoraron: {mejoraron}, Empeoraron: {empeoraron}, Tasa mejora: {tasa_mejora:.2f}%")
   

    mejoras_mae = [r['mae_pre'] - r['mae_post'] for r in resultados_por_paciente]
   
    print(f"\nMejora MAE promedio por paciente: {np.mean(mejoras_mae):.3f} mmHg")


    print("\n=== Pacientes que NO mejoraron con Few-Shot (MAE) ===")

    # Pacientes que empeoraron o no cambiaron (SBP)
    no_mejoran = [r for r in resultados_por_paciente if r['mae_post'] >= r['mae_pre']]
   

    if len(no_mejoran) == 0 and len(no_mejoran) == 0:
        print("Todos los pacientes mejoraron")
    else:
        if len(no_mejoran) > 0:
            print(f"\nSBP (Pacientes: {len(no_mejoran)})")
            for r in no_mejoran:
                diff = r['mae_post'] - r['mae_pre']
                print(f" - Paciente {r['paciente']}: MAE_pre={r['mae_pre']:.2f}, MAE_post={r['mae_post']:.2f}, Δ={diff:+.2f}")
        
    resultados = {
        "mae_pre": float(avg_pre[0]),
        "mae_post": float(avg_post[0]),
        "mejoraron": mejoraron,
        "empeoraron": empeoraron,
        "tasa_mejora": tasa_mejora,
        "resultados_por_paciente": resultados_por_paciente
    }
    return resultados

if __name__ == '__main__':
    main()