"""
import argparse
import os
import math
from typing import Optional, Tuple
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
import matplotlib
matplotlib.use("Agg")  # para que funcione sin pantalla
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score, pairwise_distances
from Modelos.ConvolucionalV1 import Modelo_ConvolucionalV1
from Clase_UCIDataset import UCIDataset

try:
    import umap
    HAS_UMAP = True
except Exception:
    HAS_UMAP = False

print(f"UMAP: {HAS_UMAP}")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Device: {DEVICE}")



archivos = [
    'data_UCI/dataset_parte_1_por_picos.pt',
    'data_UCI/dataset_parte_2_por_picos.pt',
    'data_UCI/dataset_parte_3_por_picos.pt',
    'data_UCI/dataset_parte_4_por_picos.pt',
    ]

dataset = UCIDataset(archivos)



dataLoader = torch.utils.data.DataLoader(dataset, batch_size = 256, shuffle = False)

#Carga de modelo

model = Modelo_ConvolucionalV1(in_channels=2,out_channels=2, long_signal=500)
path_model = 'best_model_conv_v1_picos.pt'
checkpoint = torch.load(path_model, map_location=DEVICE)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()
model.to(DEVICE)
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from Clase_UCIDataset import UCIDataset
from torch.utils import data
import torch
import tqdm
# X shape: (N, 2, 250)
# errors shape: (N,) con MAE o abs(pred - label)

def procesamiento_dimensional(dataloader, dual=False):
    all_data = []
    all_indices = []

    for datos, _, _, indices in dataloader:
        all_data.append(datos.numpy())      # (batch, 2, 250)
        all_indices.append(indices.numpy())

    all_data = np.concatenate(all_data, axis=0)   # (N, 2, 250)
    all_indices = np.concatenate(all_indices, axis=0)
    print(f"Total de señales cargadas: {all_data.shape[0]}")

    # Filtrar las señales que corresponden a los índices con error
    print("Filtrando las señales con errores...")
    indices_con_error_set = set(errores["indices"])
    indices_a_mantener = np.isin(all_indices, list(indices_con_error_set))

    X_filtrado = all_data[indices_a_mantener]      # (N_filtrado, 2, 250)
    valores_error_filtrados = errores["valores"][indices_a_mantener]

    print(f"Señales filtradas para PCA/t-SNE: {X_filtrado.shape[0]}")

    if dual:
        # Separar canales
        ppg = X_filtrado[:, 0, :]   # (N_filtrado, 250)
        ecg = X_filtrado[:, 1, :]   # (N_filtrado, 250)
        """
        pca = PCA()  
        pca.fit(ecg)

        explained_var = pca.explained_variance_ratio_     # varianza explicada por cada componente
        cumulative_var = np.cumsum(explained_var)         # varianza acumulada

        print("Varianza explicada acumulada:")
        print(cumulative_var[:30])  # mostramos los primeros 20

        # Número de componentes para 95%
        n_components_95 = np.argmax(cumulative_var >= 0.95) + 1
        print(f"Se necesitan {n_components_95} componentes para llegar al 95% de la varianza.")
        """

        print("Aplicando PCA...")
        X_pca_ppg = PCA(n_components=24, random_state=42).fit_transform(ppg)
        X_pca_ecg = PCA(n_components=118, random_state=42).fit_transform(ecg)

        print("Aplicando t-SNE...")
        X_emb_ppg = TSNE(n_components=2, perplexity=30, random_state=42, n_jobs=-1).fit_transform(X_pca_ppg)
        X_emb_ecg = TSNE(n_components=2, perplexity=30, random_state=42, n_jobs=-1).fit_transform(X_pca_ecg)

        X_emb = [X_emb_ppg, X_emb_ecg]

    else:

        # PCA a 50 componentes sobre PPG+ECG juntos
        print("Aplicando PCA...")
        
        X_pca = PCA(n_components=119, random_state=42).fit_transform(X_filtrado.reshape(X_filtrado.shape[0], -1))
        """
        pca = PCA()  
        pca.fit(X_filtrado.reshape(X_filtrado.shape[0], -1))

        explained_var = pca.explained_variance_ratio_     # varianza explicada por cada componente
        cumulative_var = np.cumsum(explained_var)         # varianza acumulada

        print("Varianza explicada acumulada:")
        print(cumulative_var[:30])  # mostramos los primeros 20

        # Número de componentes para 95%
        n_components_95 = np.argmax(cumulative_var >= 0.95) + 1
        print(f"Se necesitan {n_components_95} componentes para llegar al 95% de la varianza.")
        """
    
        print("Aplicando t-SNE...")
        X_emb = TSNE(n_components=2, perplexity=30, random_state=42, n_jobs=-1).fit_transform(X_pca)
    
    
    return X_emb, valores_error_filtrados


archivos = [
    'data_UCI/test_set_por_picos/test_meta.pt'
    ]

dataset = UCIDataset(archivos)

errores = np.load('Errores_predicción.npz')
subset = torch.utils.data.Subset(dataset, indices=list(range(5000)))
dataloader = torch.utils.data.DataLoader(subset, batch_size = 256, shuffle = False)

X_emb, valores_error_filtrados = procesamiento_dimensional(dataloader)
X_emb_dual, valores_error_filtrados2 =procesamiento_dimensional(dataloader, dual=True)

X_emb_dual_ppg, X_emb_dual_ecg = X_emb_dual 
valores_error_filtrados = np.array(valores_error_filtrados)


plt.figure(figsize=(8,6))
sc = plt.scatter(X_emb[:,0], X_emb[:,1], c=valores_error_filtrados[:,0], cmap="coolwarm", alpha=0.6)
plt.colorbar(sc, label="Error sbp (mmHg)")
plt.title("Mapa de similitud de señales PPG+ECG")
plt.xlabel("Dim 1")
plt.ylabel("Dim 2")
plt.show()

plt.figure(figsize=(8,6))
sc = plt.scatter(X_emb[:,0], X_emb[:,1], c=valores_error_filtrados[:, 1], cmap="coolwarm", alpha=0.6)
plt.colorbar(sc, label="Error dbp (mmHg)")
plt.title("Mapa de similitud de señales PPG+ECG")
plt.xlabel("Dim 1")
plt.ylabel("Dim 2")
plt.show()

plt.figure(figsize=(8,6))
sc = plt.scatter(X_emb_dual_ppg[:,0], X_emb_dual_ppg[:,1], c=valores_error_filtrados2[:,0], cmap="coolwarm", alpha=0.6)
plt.colorbar(sc, label="Error sbp (mmHg)")
plt.title("Mapa de señales PPG")
plt.xlabel("Dim 1")
plt.ylabel("Dim 2")
plt.show()

plt.figure(figsize=(8,6))
sc = plt.scatter(X_emb_dual_ppg[:,0], X_emb_dual_ppg[:,1], c=valores_error_filtrados2[:,1], cmap="coolwarm", alpha=0.6)
plt.colorbar(sc, label="Error dbp (mmHg)")
plt.title("Mapa de señales PPG")
plt.xlabel("Dim 1")
plt.ylabel("Dim 2")
plt.show()

plt.figure(figsize=(8,6))
sc = plt.scatter(X_emb_dual_ecg[:,0], X_emb_dual_ecg[:,1], c=valores_error_filtrados2[:,0], cmap="coolwarm", alpha=0.6)
plt.colorbar(sc, label="Error sbp (mmHg)")
plt.title("Mapa de señales ECG")
plt.xlabel("Dim 1")
plt.ylabel("Dim 2")
plt.show()

plt.figure(figsize=(8,6))
sc = plt.scatter(X_emb_dual_ecg[:,0], X_emb_dual_ecg[:,1], c=valores_error_filtrados2[:,1], cmap="coolwarm", alpha=0.6)
plt.colorbar(sc, label="Error dbp (mmHg)")
plt.title("Mapa de señales ECG")
plt.xlabel("Dim 1")
plt.ylabel("Dim 2")
plt.show()






