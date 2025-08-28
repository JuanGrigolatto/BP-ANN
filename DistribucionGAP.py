import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from scipy.stats import wasserstein_distance


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

    
unique_patients = merged_data['patient_ids'].unique().tolist()

# 5000 muestras aleatorias del dataset
total_samples = len(merged_data['data'])
subset_idx = np.random.choice(total_samples, size=5000)
subset_signals = merged_data['data'][subset_idx].numpy().reshape(5000, -1)  # Convertir a numpy y aplanar
subset_labels = merged_data['labels'][subset_idx].numpy()
subset_patients = merged_data['patient_ids'][subset_idx].numpy()
"""
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
"""      
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
