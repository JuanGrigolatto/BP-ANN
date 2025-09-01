import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from scipy.stats import wasserstein_distance
import Tools
import random

def get_95_component_variance(signals):
    """
    Calcula el número de componentes principales necesarias para explicar al menos el 95% de la varianza.
    """

    pca = PCA()  
    pca.fit(signals.reshape(signals.shape[0], -1))

    explained_var = pca.explained_variance_ratio_     # varianza explicada por cada componente
    cumulative_var = np.cumsum(explained_var)         # varianza acumulada

    print("Varianza explicada acumulada:")
    print(cumulative_var[:30])  # mostramos los primeros 20

    # Número de componentes para 95%
    n_components_95 = np.argmax(cumulative_var >= 0.95) + 1
    print(f"Se necesitan {n_components_95} componentes para llegar al 95% de la varianza.")

    return n_components_95

def intrapatient_variability_all(desnorm_SBP, desnorm_DBP, patients):
    """Calcula la variabilidad intra-paciente (std de SBP y DBP) por paciente"""
    unique_patients = np.unique(patients)   #
    sbp_std_list, dbp_std_list = [], []

    for pid in unique_patients:
        idx = patients == pid
        if np.sum(idx) > 1:  # solo pacientes con más de 1 muestra
            sbp = desnorm_SBP[idx]
            dbp = desnorm_DBP[idx]
            sbp_std_list.append(np.std(sbp))
            dbp_std_list.append(np.std(dbp))
        else:
            sbp_std_list.append(0.0)  # o np.nan si preferís
            dbp_std_list.append(0.0)

    return unique_patients, np.array(sbp_std_list), np.array(dbp_std_list)

def intrapatient_pca(patient_id, signals, patients, n_samples=500):
    idx = patients == patient_id
    patient_signals = signals[idx]

    if len(patient_signals) > n_samples:
        subset_idx = np.random.choice(len(patient_signals), size=n_samples, replace=False)
        patient_signals = patient_signals[subset_idx]

    X = patient_signals.reshape(patient_signals.shape[0], -1)

    # --- PCA completo para calcular varianza acumulada ---
    pca_full = PCA()
    pca_full.fit(X)
    cumulative_var = np.cumsum(pca_full.explained_variance_ratio_)
    n_components_95 = np.argmax(cumulative_var >= 0.95) + 1
    print(f" Para paciente {patient_id}: se necesitan {n_components_95} componentes para el 95% de la varianza")

    # --- PCA a 2D para visualización ---
    pca_2d = PCA(n_components=n_components_95)
    X_pca = pca_2d.fit_transform(X)

    return X_pca

#Cargar y combinar los datos de los cuatro archivos .pt
data_paths = [
        'data_UCI/dataset_parte_1.pt',
        'data_UCI/dataset_parte_2.pt',
        'data_UCI/dataset_parte_3.pt',
        'data_UCI/dataset_parte_4.pt'
    ]

all_data = []
all_labels = []
all_patient_ids = []

for path in data_paths:
    dataset = torch.load(path)
    all_data.append(dataset['data'])
    all_labels.append(dataset['labels'])
    all_patient_ids.append(dataset['patient_ids'])

merged_data = {
        'data': torch.cat(all_data, dim=0),
        'labels': torch.cat(all_labels, dim=0),
        'patient_ids': torch.cat(all_patient_ids, dim=0)
    }

#print (f"ID_pacientes: {merged_data['patient_ids']}")    
unique_patients = merged_data['patient_ids'].unique().tolist()
#print(f"Número de pacientes únicos: {len(unique_patients)}")
#print(f"Pacientes únicos: {unique_patients}")
# 5000 muestras aleatorias del dataset

total_samples = len(merged_data['data'])
subset_idx = np.random.choice(total_samples, size=5000)
subset_signals = merged_data['data'][subset_idx].numpy().reshape(5000, -1)  # Convertir a numpy y aplanar
subset_labels = merged_data['labels'][subset_idx].numpy()
subset_patients = merged_data['patient_ids'][subset_idx].numpy()

n_components_95 = get_95_component_variance(subset_signals)

# Reducir dimensionalidad

pca = PCA(n_components=n_components_95)
pca_result = pca.fit_transform(subset_signals)

# Graficar con colores por paciente

plt.figure(figsize=(8,6))
scatter = plt.scatter(pca_result[:,0], pca_result[:,1], c=subset_patients, cmap="tab20", s=10, alpha=0.7)
plt.colorbar(scatter, label="Patient ID")
plt.title("Distribución interpaciente (PCA de señales - 5000 muestras aleatorias)")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.show()
    

# Calcular el promedio de señales por paciente
patient_means = []
patient_ids = []

for patient_id in unique_patients:
    # Filtrar señales de este paciente
    patient_mask = (merged_data['patient_ids'] == patient_id)
    patient_signals = merged_data['data'][patient_mask].numpy()
    
    # Calcular promedio de las señales del paciente
    mean_signal = np.mean(patient_signals, axis=0)
    patient_means.append(mean_signal)
    patient_ids.append(patient_id)

patient_means = np.array(patient_means)
patient_ids = np.array(patient_ids)

# Reducir dimensionalidad del promedio por paciente

n_components_95 = get_95_component_variance(patient_means)
pca = PCA(n_components=n_components_95)
pca_result = pca.fit_transform(patient_means.reshape(len(patient_means), -1))

# Graficar - cada punto es un paciente
plt.figure(figsize=(8,6))
scatter = plt.scatter(pca_result[:,0], pca_result[:,1], c=patient_ids, cmap="tab20", s=50, alpha=0.8)
plt.colorbar(scatter, label="Patient ID")
plt.title("Distribución de PACIENTES (PCA de promedios de señales)")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.show()

SBP_MEAN = 134.02
DBP_MEAN = 63.47
SBP_STD = 22.75
DBP_STD = 23.69

desnorm_SBP = Tools.desnormalizar_zscore(merged_data['labels'][:, 0].numpy(), SBP_MEAN, SBP_STD)
desnorm_DBP = Tools.desnormalizar_zscore(merged_data['labels'][:, 1].numpy(), DBP_MEAN, DBP_STD)


unique_patients_p,sbp_std, dbp_std = intrapatient_variability_all(desnorm_SBP=desnorm_SBP, desnorm_DBP=desnorm_DBP, patients=merged_data['patient_ids'].numpy())

# === Plot SBP intra-paciente ===
plt.figure(figsize=(12,5))
plt.bar(unique_patients_p, sbp_std, alpha=0.7)
plt.xlabel("Patient ID")
plt.ylabel("STD SBP")
plt.title("Variabilidad intra-paciente (SBP)")
plt.show()

# === Plot DBP intra-paciente ===
plt.figure(figsize=(12,5))
plt.bar(unique_patients_p, dbp_std, alpha=0.7, color="orange")
plt.xlabel("Patient ID")
plt.ylabel("STD DBP")
plt.title("Variabilidad intra-paciente (DBP)")
plt.show()

# === Scatter comparando ambos ===
plt.figure(figsize=(6,6))
plt.scatter(sbp_std, dbp_std, alpha=0.7, c="#00000022")
plt.xlabel("STD SBP")
plt.ylabel("STD DBP")
plt.title("Comparación de variabilidad intra-paciente (SBP vs DBP)")
plt.grid(True)
plt.show()

num_patient = random.choice(merged_data['patient_ids'].numpy())
intrapatient_pca = intrapatient_pca(num_patient, merged_data['data'].numpy(), merged_data['patient_ids'].numpy(), n_samples=500)

plt.figure(figsize=(6,6))
plt.scatter(intrapatient_pca[:,0], intrapatient_pca[:,1], alpha=0.6, s=20)
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.title(f"Distribución de señales (PCA 2D) - Paciente {num_patient}")
plt.grid(True)
plt.show()

