import os
import random
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.utils.data as data
from sklearn.metrics import mean_absolute_error, mean_squared_error

from src.data.data_chargers.Clase_UCIDataset import UCIDataset
from src.data.data_chargers.MetaDataset import TaskDataset
from src.data.data_chargers.Intrapatientset import Intrapatientset
from src.models.ConvolucionalV1 import Modelo_ConvolucionalV1
import metalearning.Fewshot as Fewshot

# --- FUNCIONES AUXILIARES ---
def calcular_metricas_avanzadas(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    errores = y_pred - y_true
    mae = np.mean(np.abs(errores))
    rmse = np.sqrt(np.mean(errores**2))
    bias = np.mean(errores)
    sd = np.std(errores)    
    return mae, rmse, bias, sd

def main(n_shots=5, n_epochs=5, lr=5e-3, MINUTOS_DESEADOS=15, NUM_PACIENTES_TOTAL=10):
    
    # ---------------------------------------------------------
    # CONFIGURACIÓN DEL EXPERIMENTO
    # ---------------------------------------------------------
    IS_DELTA_MODEL = False  # True: Meta-Delta | False: Patient-wise Tradicional
    PATH_IDS_TEST = 'data/processed/data_UCI/few_shot_patient_data.pt'
    PACIENTES_INTERES = [101, 2041, 8423, 1126]
    
    if IS_DELTA_MODEL:
        NOMBRE_EXPERIMENTO = f"PERIODICO_PARTIAL_DELTA_{MINUTOS_DESEADOS}min"
        PATH_MODELO = 'models/checkpoints/best_meta_DELTA_LEARNING_refine_alpha50.pt'
    else:
        NOMBRE_EXPERIMENTO = f"PERIODICO_PARTIAL_TRAD_{MINUTOS_DESEADOS}min_tradiciona"
        PATH_MODELO = 'models/checkpoints/best_meta_model_v1.pt'

    print(f"--- INICIANDO MONITOREO DINÁMICO (Ajuste cada {MINUTOS_DESEADOS} min) ---")

    SEED = 42
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
    if torch.cuda.is_available(): torch.cuda.manual_seed(SEED)

    save_dir = f"resultados_intrapatient/{NOMBRE_EXPERIMENTO}"
    os.makedirs(save_dir, exist_ok=True)
    
    SBP_MEAN, SBP_STD = 134.02, 22.75
    DBP_MEAN, DBP_STD = 63.47, 23.69
    
    SEGUNDOS_POR_LOTE = (500 / 125) * n_shots 
    intervalo_ajuste = int((MINUTOS_DESEADOS * 60) / SEGUNDOS_POR_LOTE)

    # 1. Carga de datos base
    data_paths = [f'data/processed/data_UCI/dataset_parte_{i}_por_picos.pt' for i in range(1, 5)]
    dataset_completo = UCIDataset(data_paths)

    # 2. Carga de IDs de Test Reales (Evita el TypeError)
    if not os.path.exists(PATH_IDS_TEST):
        print(f"ERROR: No se encuentra el archivo de IDs en {PATH_IDS_TEST}")
        return
    
    test_data = torch.load(PATH_IDS_TEST, weights_only=False)
    ids_disponibles_test = test_data['test_patient_ids'] if isinstance(test_data, dict) else test_data
    
    # IMPORTANTE: Convertimos a lista para que TaskDataset pueda iterar
    ids_list = list(ids_disponibles_test)

    # 3. Carga del Modelo
    model = Modelo_ConvolucionalV1(in_channels=2, out_channels=2, long_signal=500)
    checkpoint = torch.load(PATH_MODELO, map_location='cpu', weights_only=False)
    state_dict = checkpoint['model_state_dict']
    new_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    model.load_state_dict(new_state_dict, strict=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    criterion = torch.nn.MSELoss()

    # 4. Selección de Pacientes
    # Pasamos la lista de IDs para evitar el error 'NoneType' is not iterable
    taskset = TaskDataset(list_IDs=ids_list, base_dataset=dataset_completo, num_shots=n_shots)
    mapa_indices = taskset.patient_to_indices
    
    pacientes_finales = [p for p in PACIENTES_INTERES if p in mapa_indices]
    otros_candidatos = [p for p in ids_list if p not in pacientes_finales and p in mapa_indices]
    
    n_faltantes = NUM_PACIENTES_TOTAL - len(pacientes_finales)
    if n_faltantes > 0 and len(otros_candidatos) > 0:
        pacientes_finales.extend(random.sample(otros_candidatos, min(len(otros_candidatos), n_faltantes)))

    print(f"Pacientes seleccionados (Test): {pacientes_finales}")

    # 5. Bucle de Procesamiento
    for id_paciente in pacientes_finales:
        print(f"\n >> PROCESANDO PACIENTE: {id_paciente}")
        model.load_state_dict(new_state_dict, strict=False)
        
        # Partial Tuning: Congelar Convolucionales, liberar Dense
        for param in model.parameters(): param.requires_grad = False
        for name, param in model.named_parameters():
            if 'dense' in name: param.requires_grad = True

        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
        
        dataset_paciente = Intrapatientset(id_paciente, dataset_completo, mapa_indices)
        loader = torch.utils.data.DataLoader(dataset_paciente, batch_size=n_shots, shuffle=False)

        historial_sbp = {'real': [], 'pred': []}
        historial_dbp = {'real': [], 'pred': []}
        puntos_ajuste = []
        bias_norm = None

        for i, (batch_signals, batch_labels) in enumerate(loader):
            # Ajuste periódico
            if i == 0 or (i + 1) % intervalo_ajuste == 0:
                model.train()
                if IS_DELTA_MODEL:
                    bias_norm = batch_labels.mean(dim=0, keepdim=True)
                    labels_tuning = batch_labels - bias_norm
                else:
                    labels_tuning = batch_labels
                
                for _ in range(n_epochs):
                    _ = Fewshot.tuning((batch_signals, labels_tuning), optimizer, model, criterion, device)
                puntos_ajuste.append(i * n_shots)

            model.eval()
            preds_raw, _ = Fewshot.evaluation((batch_signals, batch_labels), model, criterion, device)
            preds_np = preds_raw.detach().cpu().numpy()

            if IS_DELTA_MODEL and bias_norm is not None:
                preds_final = preds_np + bias_norm.numpy()
            else:
                preds_final = preds_np

            p_s = Fewshot.desnormalizar_zscore(preds_final[:, 0], SBP_MEAN, SBP_STD)
            p_d = Fewshot.desnormalizar_zscore(preds_final[:, 1], DBP_MEAN, DBP_STD)
            t_s = Fewshot.desnormalizar_zscore(batch_labels[:, 0].numpy(), SBP_MEAN, SBP_STD)
            t_d = Fewshot.desnormalizar_zscore(batch_labels[:, 1].numpy(), DBP_MEAN, DBP_STD)

            historial_sbp['real'].extend(t_s); historial_sbp['pred'].extend(p_s)
            historial_dbp['real'].extend(t_d); historial_dbp['pred'].extend(p_d)

        # Gráficas homogeneizadas
        mae_s, rmse_s, bias_s, sd_s = calcular_metricas_avanzadas(historial_sbp['real'], historial_sbp['pred'])
        plt.rcParams.update({'font.size': 12, 'font.family': 'serif'})
        fig, axs = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
        x_axis = range(len(historial_sbp['real']))
        color_p = 'tab:red' if IS_DELTA_MODEL else 'tab:orange'
        
        axs[0].plot(x_axis, historial_sbp['real'], 'k', alpha=0.6, label='Real (ABP)')
        axs[0].plot(x_axis, historial_sbp['pred'], color=color_p, linestyle='--', label='Estimado')
        for p in puntos_ajuste: axs[0].axvline(x=p, color='green', linestyle=':', alpha=0.5, linewidth=1.5)
        axs[0].set_title(f"Paciente {id_paciente} - SBP (Ajuste cada {MINUTOS_DESEADOS} min)")
        axs[0].set_ylabel("mmHg"); axs[0].legend(loc='upper right'); axs[0].grid(True, alpha=0.3)

        axs[1].plot(x_axis, historial_dbp['real'], 'k', alpha=0.6)
        axs[1].plot(x_axis, historial_dbp['pred'], color='tab:blue', linestyle='--')
        for p in puntos_ajuste: axs[1].axvline(x=p, color='green', linestyle=':', alpha=0.5, linewidth=1.5)
        axs[1].set_xlabel("Latidos"); axs[1].set_ylabel("mmHg"); axs[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f"peri_partial_{id_paciente}.png"), dpi=300)
        plt.close()

    print(f"\n--- EXPERIMENTO FINALIZADO EN {save_dir} ---")

if __name__ == '__main__':
    main()