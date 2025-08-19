#%% Visualización de datos guardados con save_partial_file
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from Clase_UCIDataset import UCIDataset
from torch.utils.data import DataLoader

def visualizar_datos(nombre_archivo, sample_index=0):
    
    SBP_MEAN = 134.02
    DBP_MEAN = 63.47
    SBP_STD = 22.75
    DBP_STD = 23.69
    # Ruta al archivo de metadatos
    meta_path = os.path.join("data_UCI", nombre_archivo)
    meta = torch.load(meta_path)

    # Cargar memmaps en modo solo lectura
    data_mmap = np.memmap(meta["data_path"], dtype="float32", mode="r", 
                          shape=(meta["num_samples"], 2, meta["segment_length"]))
    labels_mmap = np.memmap(meta["labels_path"], dtype="float32", mode="r", 
                            shape=(meta["num_samples"], 2))
    patients_mmap = np.memmap(meta["patients_path"], dtype="int64", mode="r", 
                              shape=(meta["num_samples"],))
    indexs_mmap = np.memmap(meta["indexs_path"], dtype="int64", mode="r", 
                            shape=(meta["num_samples"],))

    # Extraer muestra seleccionada
  
    signal = data_mmap[sample_index]
    ppg = signal[0]
    ecg = signal[1]
    sbp, dbp = labels_mmap[sample_index]
    patient_id = patients_mmap[sample_index]
    sample_id = indexs_mmap[sample_index]

    sbp = sbp * SBP_STD + SBP_MEAN
    dbp = dbp * DBP_STD + DBP_MEAN

    # Graficar
    fig, axs = plt.subplots(2, 1, figsize=(12, 6), sharex=True)

    axs[0].plot(ppg, color="tab:blue", label="PPG")
    axs[0].set_title(f"PPG - Paciente {patient_id}, Muestra {sample_id}")
    axs[0].legend()

    axs[1].plot(ecg, color="tab:orange", label="ECG")
    axs[1].set_title(f"ECG | SBP={sbp:.1f} mmHg, DBP={dbp:.1f} mmHg")
    axs[1].legend()

    plt.tight_layout()
    plt.show()

    # Inicializar min y max
    sbp_min, sbp_max = float("inf"), float("-inf")
    dbp_min, dbp_max = float("inf"), float("-inf")

    # Recorrer todas las etiquetas
    for sbp_norm, dbp_norm in labels_mmap:
        sbp = sbp_norm * SBP_STD + SBP_MEAN
        dbp = dbp_norm * DBP_STD + DBP_MEAN

        if sbp < sbp_min: sbp_min = sbp
        if sbp > sbp_max: sbp_max = sbp
        if dbp < dbp_min: dbp_min = dbp
        if dbp > dbp_max: dbp_max = dbp

    print(f"SBP -> min: {sbp_min:.2f}, max: {sbp_max:.2f}")
    print(f"DBP -> min: {dbp_min:.2f}, max: {dbp_max:.2f}")



if __name__ == "__main__":
    visualizar_datos("test_set_por_picos/test_meta.pt", sample_index=910)

# %%
"""
def visualizar_uci(dataset, sample_index=0, desnormalizar=False, SBP_MEAN=120, SBP_STD=15, DBP_MEAN=80, DBP_STD=10):
   
    
    x, y, pid, idx = dataset[sample_index]

    # Señales
    ppg = x[0].numpy()
    ecg = x[1].numpy()
    
    # Etiquetas
    sbp, dbp = y.numpy()
    if desnormalizar:
        sbp = sbp * SBP_STD + SBP_MEAN
        dbp = dbp * DBP_STD + DBP_MEAN

    # Graficar
    fig, axs = plt.subplots(2, 1, figsize=(12, 6), sharex=True)

    axs[0].plot(ppg, color="tab:blue", label="PPG")
    axs[0].set_title(f"PPG - Paciente {pid.item()}, Muestra {idx.item()}")
    axs[0].legend()

    axs[1].plot(ecg, color="tab:orange", label="ECG")
    axs[1].set_title(f"ECG | SBP={sbp:.1f} mmHg, DBP={dbp:.1f} mmHg")
    axs[1].legend()

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
   
    file_list = ["data_UCI/dataset_parte_1_hanning.pt", "data_UCI/dataset_parte_2_hanning.pt"]

    dataset = UCIDataset(file_list)
    
    visualizar_uci(dataset, sample_index=100, desnormalizar=True)
"""