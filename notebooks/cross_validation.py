import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import KFold
import numpy as np
from src.models.ConvolucionalV1 import Modelo_ConvolucionalV1
from src.data.data_chargers.Clase_UCIDataset import UCIDataset

def compute_metrics(y_true, y_pred):
    y_true = y_true.cpu().numpy()
    y_pred = y_pred.cpu().numpy()

    mse = np.mean((y_true - y_pred) ** 2, axis=0)  
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(y_true - y_pred), axis=0)

    # R^2 por componente
    ss_res = np.sum((y_true - y_pred) ** 2, axis=0)
    ss_tot = np.sum((y_true - np.mean(y_true, axis=0)) ** 2, axis=0)
    r2 = 1 - ss_res / ss_tot

    return {
        "MSE": mse,
        "RMSE": rmse,
        "MAE": mae,
        "R2": r2
    }

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

for fold, (train_ids, val_ids) in enumerate(kfold.split(dataset_completo)):
    print(f"\n🔹 Fold {fold+1}/{k_folds}")

    train_subset = Subset(dataset_completo, train_ids)
    val_subset = Subset(dataset_completo, val_ids)

    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False)

    # Modelo nuevo en cada fold
    model=Modelo_ConvolucionalV1(in_channels=2,out_channels=1, long_signal=500)
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    best_val_loss = float("inf")
    best_metrics = None
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for batch in train_loader:
            signals, labels, _, _ = batch
            
            labels = labels[:,1].unsqueeze(1) 

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
                
                labels = labels[:,1].unsqueeze(1)

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
        metrics = compute_metrics(all_labels, all_preds)

        print(f"Epoch {epoch+1}/{epochs} | "
              f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
              f"Val MAE [DBP={metrics['MAE']:.2f}]")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_metrics = metrics

    fold_results.append(best_metrics)
    print(f"Fold {fold+1} - Mejor MAE: DBP={best_metrics['MAE']:.2f}")

# ----------------------------
# Resultados finales
# ----------------------------
print("\n Resultados por fold:")
for i, m in enumerate(fold_results):
    print(f"Fold {i+1}: "
          f"MAE [DBP={m['MAE']:.2f}], "
          f"RMSE [DBP={m['RMSE']:.2f}], "
          f"R² [DBP={m['R2']:.3f}]")

# Promedio sobre folds
avg_mae = np.mean([m["MAE"] for m in fold_results], axis=0)
avg_rmse = np.mean([m["RMSE"] for m in fold_results], axis=0)
avg_r2 = np.mean([m["R2"] for m in fold_results], axis=0)

print("\n🔎 Promedio en todos los folds:")
print(f"MAE  -> DBP={avg_mae:.2f}")
print(f"RMSE -> DBP={avg_rmse:.2f}")
print(f"R²   -> DBP={avg_r2:.3f}")