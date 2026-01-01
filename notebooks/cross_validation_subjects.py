import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import KFold
import numpy as np
from src.models.ConvolucionalV1 import Modelo_ConvolucionalV1
from src.data.data_chargers.Clase_UCIDataset import UCIDataset
import pandas as pd
import os
from tqdm import tqdm

def compute_metrics(y_true, y_pred):
    """
    Devuelve métricas como floats porque la red predice solo DBP (salida única).
    Acepta tensores torch con shape (N,1) o (N,) y convierte a 1D numpy.
    """
    y_true = y_true.cpu().numpy()
    y_pred = y_pred.cpu().numpy()

    # Si vienen como (N,1) -> pasar a (N,)
    if y_true.ndim == 2 and y_true.shape[1] == 1:
        y_true = y_true.squeeze(axis=1)
    if y_pred.ndim == 2 and y_pred.shape[1] == 1:
        y_pred = y_pred.squeeze(axis=1)

    # Ahora y_true y y_pred son vectores 1D
    mse = float(np.mean((y_true - y_pred) ** 2))
    rmse = float(np.sqrt(mse))
    mae = float(np.mean(np.abs(y_true - y_pred)))

    # R^2 (protección si ss_tot == 0)
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    if ss_tot == 0:
        r2 = 0.0
    else:
        r2 = float(1 - ss_res / ss_tot)

    return {
        "MSE": mse,
        "RMSE": rmse,
        "MAE": mae,
        "R2": r2
    }

def compute_metrics_two_labels(y_true, y_pred):
    """
    Calcula métricas por cada salida (SBP y DBP) y global promedio.
    y_true y y_pred deben tener shape (N,2).
    """
    y_true = y_true.cpu().numpy()
    y_pred = y_pred.cpu().numpy()

    metrics = {}
    nombres = ["SBP", "DBP"]

    for i, nombre in enumerate(nombres):
        yt = y_true[:, i]
        yp = y_pred[:, i]

        mse = float(np.mean((yt - yp) ** 2))
        rmse = float(np.sqrt(mse))
        mae = float(np.mean(np.abs(yt - yp)))

        ss_res = np.sum((yt - yp) ** 2)
        ss_tot = np.sum((yt - np.mean(yt)) ** 2)
        r2 = float(1 - ss_res / ss_tot) if ss_tot != 0 else 0.0

        metrics[f"MSE_{nombre}"] = mse
        metrics[f"RMSE_{nombre}"] = rmse
        metrics[f"MAE_{nombre}"] = mae
        metrics[f"R2_{nombre}"] = r2

    # Promedio global (opcional)
    metrics["MAE_mean"] = np.mean([metrics["MAE_SBP"], metrics["MAE_DBP"]])
    metrics["RMSE_mean"] = np.mean([metrics["RMSE_SBP"], metrics["RMSE_DBP"]])
    metrics["R2_mean"] = np.mean([metrics["R2_SBP"], metrics["R2_DBP"]])

    return metrics

def get_patient_map(pt_files):
    """
    Lee los metadatos y devuelve un array (N, 2) donde:
    Columna 0: ID del paciente
    Columna 1: Índice global en el dataset
    """
    global_indices_map = [] 
    current_offset = 0
    
    print("--- Cargando mapa de Pacientes ---")
    
    for pt_file in pt_files:
        if not os.path.exists(pt_file):
            raise FileNotFoundError(f"Falta archivo: {pt_file}")
            
        meta = torch.load(pt_file)
        num_samples = meta['num_samples']
        patients_path = meta['patients_path'] 
        
        # Corrección de path si es relativo o se movió
        if not os.path.exists(patients_path):
             base_dir = os.path.dirname(pt_file)
             patients_path = os.path.join(base_dir, os.path.basename(patients_path))

        if not os.path.exists(patients_path):
             raise FileNotFoundError(f"Falta archivo de pacientes: {patients_path}")

        # Lectura eficiente con memmap
        p_mmap = np.memmap(patients_path, dtype='int64', mode='r', shape=(num_samples,))
        ids = np.array(p_mmap) 
        indices = np.arange(current_offset, current_offset + num_samples)
        
        chunk_data = np.stack((ids, indices), axis=1)
        global_indices_map.append(chunk_data)
        
        current_offset += num_samples

    full_map = np.concatenate(global_indices_map, axis=0)
    return full_map
# ----------------------------
# Configuración
# ----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batch_size = 256
epochs = 10
k_folds = 5
learning_rate = 1e-3

# ----------------------------
# Dataset
# ----------------------------
archivos = [
    'data/processed/data_UCI/dataset_parte_1_por_picos.pt',
    'data/processed/data_UCI/dataset_parte_2_por_picos.pt',
    'data/processed/data_UCI/dataset_parte_3_por_picos.pt',
    'data/processed/data_UCI/dataset_parte_4_por_picos.pt',
]

dataset_completo = UCIDataset(archivos)

# ----------------------------
# Criterio de pérdida
# ----------------------------
criterion = nn.MSELoss()

# ----------------------------
# Validación cruzada
# ----------------------------
kfold = KFold(n_splits=k_folds, shuffle=True, random_state=42)

fold_results = []
# 1. OBTENER EL MAPA DE PACIENTES
full_map = get_patient_map(archivos) 
# full_map[:, 0] son los IDs de pacientes
# full_map[:, 1] son los índices globales (0, 1, 2... N)

unique_patients = np.unique(full_map[:, 0])
print(f"Total de pacientes únicos para CV: {len(unique_patients)}")
# Iteramos sobre los PACIENTES, no sobre el dataset directamente
for fold, (train_patients_idx, val_patients_idx) in enumerate(tqdm(kfold.split(unique_patients), total=k_folds)):
    
    # Obtenemos los IDs reales de los pacientes para este fold
    train_patients = unique_patients[train_patients_idx]
    val_patients = unique_patients[val_patients_idx]
    
    # -------------------------------------------------------
    # CRÍTICO: Convertir lista de pacientes a índices de muestras
    # -------------------------------------------------------
    # Buscamos en full_map qué filas corresponden a los pacientes de train
    mask_train = np.isin(full_map[:, 0], train_patients)
    mask_val = np.isin(full_map[:, 0], val_patients)
    
    # Extraemos los índices globales
    train_indices = full_map[mask_train, 1]
    val_indices = full_map[mask_val, 1]
    
    # Control de sanidad
    intersect = np.intersect1d(train_indices, val_indices)
    assert len(intersect) == 0, "ERROR: Hay cruce de datos entre train y val!"
    
    print(f"\nFold {fold+1}: {len(train_patients)} pacientes en Train ({len(train_indices)} muestras), {len(val_patients)} pacientes en Val ({len(val_indices)} muestras)")

    # Creamos los Subsets y Loaders
    train_subset = Subset(dataset_completo, train_indices)
    val_subset = Subset(dataset_completo, val_indices)

    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False)
    # Modelo nuevo en cada fold (salida = 1 para DBP)
    model = Modelo_ConvolucionalV1(in_channels=2, out_channels=2, long_signal=500)
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    best_val_loss = float("inf")
    best_metrics = None
    
    for epoch in tqdm(range(epochs), desc=f"Fold {fold+1}/{k_folds}", leave=False):
        model.train()
        train_loss = 0.0
        for batch in train_loader:
            signals, labels, _, _ = batch
            # seleccionás DBP y dejás shape (N,1)
            #labels = labels[:,0].unsqueeze(1)
            labels = labels[:, :2]

            signals, labels = signals.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(signals)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        # Validación
        model.eval()
        val_loss = 0.0
        all_preds, all_labels = [], []
        with torch.no_grad():
            for batch in val_loader:
                signals, labels, _, _ = batch
                #labels = labels[:,0].unsqueeze(1)
                labels = labels[:, :2]

                signals, labels = signals.to(device), labels.to(device)
                
                outputs = model(signals)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

                all_preds.append(outputs)
                all_labels.append(labels)

        train_loss /= len(train_loader)
        val_loss /= len(val_loader)

        all_preds = torch.cat(all_preds)
        all_labels = torch.cat(all_labels)
        #metrics = compute_metrics(all_labels, all_preds)
        metrics = compute_metrics_two_labels(all_labels, all_preds)
        # metrics['MAE'] ya es float si la red es single-output
        """
        print(f"Epoch {epoch+1}/{epochs} | "
              f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
              f"Val MAE [DBP={metrics['MAE']:.2f}]")
        """
    
        print(f"Epoch {epoch+1}/{epochs} | "
              f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
              f"MAE [SBP={metrics['MAE_SBP']:.2f}, DBP={metrics['MAE_DBP']:.2f}]")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_metrics = metrics

    fold_results.append(best_metrics)
    #print(f"Fold {fold+1} - Mejor MAE: DBP={best_metrics['MAE']:.2f}")
    print(f"Fold {fold+1} - Mejor MAE: SBP={best_metrics['MAE_SBP']:.2f}, DBP={best_metrics['MAE_DBP']:.2f}")

# ----------------------------
# Resultados finales
# ----------------------------
print("\n Resultados por fold:")
"""
resultados_dict = {
    "fold": [],
    "MSE": [],
    "RMSE": [],
    "MAE": [],
    "R2": []
}
"""
resultados_dict = {
    "fold": [],
    "MAE_SBP": [], "RMSE_SBP": [], "R2_SBP": [],
    "MAE_DBP": [], "RMSE_DBP": [], "R2_DBP": [],
    "MAE_mean": [], "RMSE_mean": [], "R2_mean": []
}

"""
for i, m in enumerate(fold_results):
    print(f"Fold {i+1}: "
          f"MAE [DBP={m['MAE']:.2f}], "
          f"RMSE [DBP={m['RMSE']:.2f}], "
          f"R² [DBP={m['R2']:.3f}]")
    
    resultados_dict["fold"].append(i+1)
    resultados_dict["MSE"].append(m["MSE"])
    resultados_dict["RMSE"].append(m["RMSE"])
    resultados_dict["MAE"].append(m["MAE"])
    resultados_dict["R2"].append(m["R2"])

# Promedio sobre folds
avg_mae = np.mean([m["MAE"] for m in fold_results])
avg_rmse = np.mean([m["RMSE"] for m in fold_results])
avg_r2 = np.mean([m["R2"] for m in fold_results])

print("\n🔎 Promedio en todos los folds:")
print(f"MAE  -> DBP={avg_mae:.2f}")
print(f"RMSE -> DBP={avg_rmse:.2f}")
print(f"R²   -> DBP={avg_r2:.3f}")
"""
for i, m in enumerate(fold_results):
    print(f"Fold {i+1}: "
          f"MAE [SBP={m['MAE_SBP']:.2f}, DBP={m['MAE_DBP']:.2f}] | "
          f"RMSE [SBP={m['RMSE_SBP']:.2f}, DBP={m['RMSE_DBP']:.2f}]")

    resultados_dict["fold"].append(i+1)
    resultados_dict["MAE_SBP"].append(m["MAE_SBP"])
    resultados_dict["RMSE_SBP"].append(m["RMSE_SBP"])
    resultados_dict["R2_SBP"].append(m["R2_SBP"])
    resultados_dict["MAE_DBP"].append(m["MAE_DBP"])
    resultados_dict["RMSE_DBP"].append(m["RMSE_DBP"])
    resultados_dict["R2_DBP"].append(m["R2_DBP"])
    resultados_dict["MAE_mean"].append(m["MAE_mean"])
    resultados_dict["RMSE_mean"].append(m["RMSE_mean"])
    resultados_dict["R2_mean"].append(m["R2_mean"])

# Promedio sobre folds
print("\n🔎 Promedio en todos los folds:")
for k in ["MAE_SBP", "MAE_DBP", "RMSE_SBP", "RMSE_DBP", "R2_SBP", "R2_DBP"]:
    vals = [m[k] for m in fold_results]
    print(f"{k}: mean={np.mean(vals):.2f}, std={np.std(vals):.2f}")

# Guardar a CSV
df = pd.DataFrame(resultados_dict)
"""
df.loc["mean"] = ["-", np.mean(df["MSE"]), np.mean(df["RMSE"]), np.mean(df["MAE"]), np.mean(df["R2"])]
df.loc["std"] = ["-", np.std(df["MSE"]), np.std(df["RMSE"]), np.std(df["MAE"]), np.std(df["R2"])]
"""
df.loc["mean"] = ["-"] + [np.mean(df[c]) for c in df.columns if c != "fold"]
df.loc["std"] = ["-"] + [np.std(df[c]) for c in df.columns if c != "fold"]

os.makedirs("results", exist_ok=True)
csv_path = os.path.abspath(os.path.join("results", "cross_validation_metrics_sbp_dbp_PS.csv"))
df.to_csv(csv_path, index=True)

print(f"\n Métricas guardadas en '{csv_path}'")