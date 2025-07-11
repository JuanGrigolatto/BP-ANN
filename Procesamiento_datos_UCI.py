# -*- coding: utf-8 -*-
"""
Created on Mon Feb 10 12:52:58 2025

@author: juang
"""
#%% Librerias
import h5py
import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.signal import find_peaks
import torch
import gc
#%% Definición de funciones

def split_into_windows(signal, fs, t_window, overlap):
    window_size = fs * t_window  # Tamaño de la ventana en muestras
    step = int(window_size * (1 - overlap))  # Paso entre ventanas
    windows = []
    for i in range(0, len(signal) - window_size + 1, step):
        windows.append(signal[i:i + window_size])
    return np.array(windows)

def min_max_normalization(signal, global_max, global_min):
    return (signal - global_min) / (global_max - global_min)

def signal_segmentation(fs, t_window, ppg_signal, abp_signal, ecg_signal):
    ppg_signal_segmented=[]
    abp_signal_segmented=[]
    ecg_signal_segmented=[]

    #Dividir en ventanas
    for i in range(len(ppg_signal)):  
        ppg_signal_segmented.append(split_into_windows(ppg_signal[i], fs, t_window, overlap=0.5))
        abp_signal_segmented.append(split_into_windows(abp_signal[i], fs, t_window, overlap=0.5))
        ecg_signal_segmented.append(split_into_windows(ecg_signal[i], fs, t_window, overlap=0.5))

    return ppg_signal_segmented, abp_signal_segmented, ecg_signal_segmented

def signal_normalization(ppg_signals, abp_signals, ecg_signals, ppg_max, ppg_min, ecg_max, ecg_min, abp_max, abp_min):
    ppg_normalized=[]
    abp_normalized=[]
    ecg_normalized=[]
    
    for i in range(len(ppg_signals)):
        ppg_normalized.append(min_max_normalization(ppg_signals[i], ppg_max, ppg_min))
        abp_normalized.append(min_max_normalization(abp_signals[i], abp_max, abp_min))
        ecg_normalized.append(min_max_normalization(ecg_signals[i], ecg_max, ecg_min))
    return ppg_normalized, abp_normalized, ecg_normalized

def pressure_normalization(sbp, dbp, sbp_min, sbp_max, dbp_min, dbp_max):
    sbp_norm = (sbp - sbp_min) / (sbp_max - sbp_min)
    dbp_norm = (dbp - dbp_min) / (dbp_max - dbp_min)
    return sbp_norm, dbp_norm

def pressure_desnormalization(sbp_norm, dbp_norm, sbp_min, sbp_max, dbp_min, dbp_max):
    sbp = sbp_norm * (sbp_max - sbp_min) + sbp_min
    dbp = dbp_norm * (dbp_max - dbp_min) + dbp_min
    return sbp, dbp

def get_abp_labels(abp_signals):
  
    DISTANCIA_MINIMA_SBP = 40          # Distancia mínima entre picos SBP (en muestras)
    DISTANCIA_MINIMA_DBP = 40
    matriz_picos_sistolicos=[]
    matriz_picos_diastolicos=[]
    matriz_presiones_sistolicas=[]
    matriz_presiones_diastolicas=[]

    for i in range(len(abp_signals)):
        picos_sistolicos=[]
        picos_diastolicos=[]
        presiones_sistolicas = []
        presiones_diastolicas = []
        for j in range(len(abp_signals[i])):
            # Encontrar picos en la señal ABP
            peakss,_ = find_peaks(abp_signals[i][j], height=80, prominence= 15 ,distance=DISTANCIA_MINIMA_SBP) 
            peaksd,_ = find_peaks(-(abp_signals)[i][j], height=-132, prominence= 15, distance=DISTANCIA_MINIMA_DBP)
            picos_sistolicos.append(peakss)
            picos_diastolicos.append(peaksd)
            if (len(peakss)>0):
                ps = np.max(abp_signals[i][j][peakss])
            else:
                ps = np.nan
        
            if (len(peaksd)>0):
                pd = np.min(abp_signals[i][j][peaksd])
            else:
                pd = np.nan
            presiones_sistolicas.append(ps)
            presiones_diastolicas.append(pd)
        
        matriz_presiones_sistolicas.append(presiones_sistolicas)
        matriz_presiones_diastolicas.append(presiones_diastolicas)
        
        matriz_picos_sistolicos.append(picos_sistolicos)
        matriz_picos_diastolicos.append(picos_diastolicos)

    return matriz_presiones_sistolicas, matriz_presiones_diastolicas, matriz_picos_sistolicos, matriz_picos_diastolicos

def labels_normalization(matriz_presiones_sistolicas, matriz_presiones_diastolicas):

    # Aplanar y convertir a arrays de numpy
    todas_sbp = np.array([x for sublista in matriz_presiones_sistolicas for x in sublista if not np.isnan(x)])
    todas_dbp = np.array([x for sublista in matriz_presiones_diastolicas for x in sublista if not np.isnan(x)])

    SBP_MIN = np.min(todas_sbp)
    SBP_MAX = np.max(todas_sbp)
    DBP_MIN = np.min(todas_dbp)
    DBP_MAX = np.max(todas_dbp)

    print(f"SBP_MIN={SBP_MIN:.2f}, SBP_MAX={SBP_MAX:.2f}")
    print(f"DBP_MIN={DBP_MIN:.2f}, DBP_MAX={DBP_MAX:.2f}")
    matriz_presiones_sistolicas_norm=[]
    matriz_presiones_diastolicas_norm=[]
    for i in range(len(matriz_presiones_sistolicas)):
        list_sbp=[]
        list_dbp=[]  
        for j in range(len(matriz_presiones_sistolicas[i])):

            ps_norm,pd_norm=pressure_normalization(matriz_presiones_sistolicas[i][j], matriz_presiones_diastolicas[i][j],
                                     SBP_MIN, SBP_MAX,DBP_MIN,DBP_MAX)
            list_sbp.append(ps_norm)
            list_dbp.append(pd_norm)    
        matriz_presiones_sistolicas_norm.append(list_sbp)
        matriz_presiones_diastolicas_norm.append(list_dbp)
    return matriz_presiones_sistolicas_norm, matriz_presiones_diastolicas_norm

def save_partial_file(ppg_signals, ecg_signals, sbp_labels, dbp_labels, patient_id_inicial, nombre_archivo):
    output_dir = 'data_UCI'
    os.makedirs(output_dir, exist_ok=True)

    num_total = sum(len(ppg) for ppg in ppg_signals)
    long_segmento = len(ppg_signals[0][0])  

    # Rutas de archivos temporales únicos por parte
    data_path = os.path.join(output_dir, f'{nombre_archivo}_data.dat')
    labels_path = os.path.join(output_dir, f'{nombre_archivo}_labels.dat')
    patients_path = os.path.join(output_dir, f'{nombre_archivo}_patients.dat')

    # Crear archivos memmap
    data_mmap = np.memmap(data_path, dtype='float32', mode='w+', shape=(num_total, 2, long_segmento))
    labels_mmap = np.memmap(labels_path, dtype='float32', mode='w+', shape=(num_total, 2))
    patients_mmap = np.memmap(patients_path, dtype='int64', mode='w+', shape=(num_total,))

    index = 0
    for paciente_id, (ppg_segmentos, ecg_segmentos, sbp_segmentos, dbp_segmentos) in enumerate(zip(ppg_signals, ecg_signals, sbp_labels, dbp_labels)):
        for ppg, ecg, sbp, dbp in zip(ppg_segmentos, ecg_segmentos, sbp_segmentos, dbp_segmentos):

            if np.isnan(ppg).any() or np.isnan(ecg).any() or np.isnan(sbp) or np.isnan(dbp):
                continue

            data_mmap[index, 0] = ppg
            data_mmap[index, 1] = ecg
            labels_mmap[index] = [sbp, dbp]
            patients_mmap[index] = paciente_id + patient_id_inicial
            index += 1

    # Recortar arrays al número real de muestras válidas
    data_tensor = torch.from_numpy(np.array(data_mmap[:index]))
    labels_tensor = torch.from_numpy(np.array(labels_mmap[:index]))
    patients_tensor = torch.from_numpy(np.array(patients_mmap[:index]))

    # Guardar en archivo .pt
    torch.save({
        'data': data_tensor,
        'labels': labels_tensor,
        'patient_ids': patients_tensor
    }, os.path.join(output_dir, f'{nombre_archivo}.pt'))

    print(f"{nombre_archivo}.pt guardado con {index} muestras.")
#%% Carga de señales de Part_1.mat

archivo_datos1 = "datos/Part_1.mat"

with h5py.File(archivo_datos1, 'r') as f:
    
    claves=list(f.keys())
    print("claves: ",claves)
    dataset1=f['Part_1']
    print(dataset1.shape)
    datos_numpy = np.array(dataset1)
    
    datos_extraidos = []
    for i in range(dataset1.shape[0]):  
        ref = dataset1[i, 0]  # Obtener la referencia HDF5
        datos_extraidos.append(np.array(f[ref]))  # Convertir el objeto referenciado en un array NumPy

    datos_extraidos = np.array(datos_extraidos, dtype=object) 

    ppg_signal=[]
    abp_signal=[]
    ecg_signal=[]
    for i in range(datos_extraidos.shape[0]):
        ppg_signal.append(datos_extraidos[i][:, 0])
        abp_signal.append(datos_extraidos[i][:, 1])
        ecg_signal.append(datos_extraidos[i][:, 2])       
#%% Procesamiento de archivo parte 1
fs=125

window_duration=2

all_ppg_values = np.concatenate(ppg_signal)
all_ecg_values = np.concatenate(ecg_signal)
all_abp_values = np.concatenate(abp_signal)

PPG_MIN, PPG_MAX = np.min(all_ppg_values), np.max(all_ppg_values)
ECG_MIN, ECG_MAX = np.min(all_ecg_values), np.max(all_ecg_values)
ABP_MIN, ABP_MAX = np.min(all_abp_values), np.max(all_abp_values)

print(f"PPG: min={PPG_MIN:.2f}, max={PPG_MAX:.2f}")
print(f"ECG: min={ECG_MIN:.2f}, max={ECG_MAX:.2f}")
print(f"ABP: min={ABP_MIN:.2f}, max={ABP_MAX:.2f}")

ppg_segmented, abp_segmented, ecg_segmented= signal_segmentation(fs, window_duration, ppg_signal, 
                                                                 abp_signal, ecg_signal)

ppg_normalized, abp_normalized, ecg_normalized= signal_normalization(ppg_segmented, abp_segmented, ecg_segmented,
                                                                     PPG_MAX,PPG_MIN,ECG_MAX,ECG_MIN,ABP_MAX,ABP_MIN)

presiones_sistolicas, presiones_diastolicas, indices_sistolicos, indices_diastolicos = get_abp_labels(abp_segmented)

presiones_sistolicas_norm, presiones_diastolicas_norm= labels_normalization(presiones_sistolicas, presiones_diastolicas)
#%% Ploteo de señales    
plt.figure(figsize=(10, 4))
plt.plot(ppg_signal[1000][:1000])  # Mostrar los primeros 1000 puntos
plt.title("Señal PPG - Primer Registro")
plt.xlabel("Muestras")
plt.ylabel("Amplitud")
plt.show()

plt.figure(figsize=(10, 4))
plt.plot(abp_signal[1000][:1000])  # Mostrar los primeros 1000 puntos
plt.title("Señal ABP - Primer Registro")
plt.xlabel("Muestras")
plt.ylabel("Amplitud")
plt.show()
    
plt.figure(figsize=(10, 4))
plt.plot(ecg_signal[1000][:1000])  # Mostrar los primeros 1000 puntos
plt.title("Señal ECG - Primer Registro")
plt.xlabel("Muestras")
plt.ylabel("Amplitud")
plt.show()
#%% Ploteo de segmentación
plt.figure(figsize=(10, 4))
plt.plot(ppg_segmented[1000][0])  # Mostrar los primeros 1000 puntos
plt.title("Recorte de 2 seg de señal de PPG")
plt.xlabel("Muestras")
plt.ylabel("Amplitud")
plt.show()

plt.figure(figsize=(10, 4))
plt.plot(abp_segmented[1000][0])  # Mostrar los primeros 1000 puntos
plt.title("Recorte de 5 seg de señal de ABP")
plt.xlabel("Muestras")
plt.ylabel("Amplitud")
plt.show()

plt.figure(figsize=(10, 4))
plt.plot(ecg_segmented[1000][0])  # Mostrar los primeros 1000 puntos
plt.title("Recorte de 5 seg de señal de ECG")
plt.xlabel("Muestras")
plt.ylabel("Amplitud")
plt.show()
#%% Relación de señales en el recorte
plt.figure(figsize=(10, 4))
plt.plot(ppg_normalized[1100][0], color= "blue", label="PPG")
plt.plot(abp_normalized[1100][0], color= "green", label="ABP")
plt.plot(ecg_normalized[1100][0], color= "red", label="ECG") 
plt.title("Relación entre señales en el tiempo")
plt.xlabel("Muestras")
plt.ylabel("Amplitud")
plt.show()
#%%
plt.figure(figsize=(10, 4))
prueba= (abp_segmented[1000][0])
plt.plot((prueba))  # Mostrar los primeros 1000 puntos
plt.title("Recorte de 5 seg de señal de ABP")
plt.xlabel("Muestras")
plt.ylabel("Amplitud")
plt.show()
#%%
paciente_id = 2637
ventana_id = 0
senal_abp = abp_normalized[paciente_id][ventana_id]

# Extraé los índices de picos sistólicos y diastólicos ya calculados
ind_sistolicos = indices_sistolicos[paciente_id][ventana_id]
ind_diastolicos= indices_diastolicos[paciente_id][ventana_id]

# Graficá
plt.figure(figsize=(10, 4))
plt.plot(senal_abp, label='Señal ABP Normalizada')
plt.plot(ind_sistolicos, senal_abp[ind_sistolicos], 'ro', label='Picos Sistólicos')
plt.plot(ind_diastolicos, senal_abp[ind_diastolicos], 'bo', label='Picos Diastólicos')
plt.xlabel('Muestras')
plt.ylabel('Amplitud')
plt.title(f'Paciente {paciente_id} - Ventana {ventana_id}')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
#%% Generación de archivo parte 1 .pt y liberación de memoria RAM
save_partial_file(ppg_normalized, ecg_normalized, presiones_sistolicas_norm, presiones_diastolicas_norm, 0, 'dataset_parte_1')
del ppg_signal
del abp_signal
del ecg_signal
del ppg_segmented
del abp_segmented
del ecg_segmented
del ppg_normalized
del abp_normalized
del ecg_normalized
del presiones_diastolicas
del presiones_sistolicas
del presiones_diastolicas_norm
del presiones_sistolicas_norm
gc.collect()
#%% Carga de señales de Part_2.mat
archivo_datos2 = "datos/Part_2.mat"
with h5py.File(archivo_datos2, 'r') as f:
    
    claves=list(f.keys())
    print("claves: ",claves)
    dataset2=f['Part_2']
    print(dataset2.shape)
    datos_numpy = np.array(dataset2)
    
    datos_extraidos2 = []
    for i in range(dataset2.shape[0]):  
        ref = dataset2[i, 0]  # Obtener la referencia HDF5
        datos_extraidos2.append(np.array(f[ref]))  # Convertir el objeto referenciado en un array NumPy

    datos_extraidos2 = np.array(datos_extraidos2, dtype=object)

    ppg_signal2=[]
    abp_signal2=[]
    ecg_signal2=[]

    for i in range(datos_extraidos2.shape[0]):
        ppg_signal2.append(datos_extraidos2[i][:, 0])
        abp_signal2.append(datos_extraidos2[i][:, 1])
        ecg_signal2.append(datos_extraidos2[i][:, 2])

#%% Procesamiento de archivo parte 2
fs=125

window_duration=2

ppg_segmented2, abp_segmented2, ecg_segmented2= signal_segmentation(fs, window_duration, ppg_signal2, 
                                                                 abp_signal2, ecg_signal2)
ppg_normalized2, abp_normalized2, ecg_normalized2= signal_normalization(ppg_segmented2, abp_segmented2, ecg_segmented2)

presiones_sistolicas2, presiones_diastolicas2 = get_abp_labels(abp_normalized2)

presiones_sistolicas_norm2, presiones_diastolicas_norm2= labels_normalization(presiones_sistolicas2, presiones_diastolicas2)
#%% Generación de archivo parte 2 .pt y liberación de memoria RAM
save_partial_file(ppg_normalized2, ecg_normalized2, presiones_sistolicas_norm2, presiones_diastolicas_norm2, 3000, 'dataset_parte_2')
del ppg_signal2
del abp_signal2
del ecg_signal2
del ppg_segmented2
del abp_segmented2
del ecg_segmented2
del ppg_normalized2
del abp_normalized2
del ecg_normalized2
del presiones_diastolicas2
del presiones_sistolicas2
del presiones_diastolicas_norm2
del presiones_sistolicas_norm2
gc.collect()
#%% #%% Carga de señales de Part_3.mat
archivo_datos3 = "datos/Part_3.mat"
with h5py.File(archivo_datos3, 'r') as f:
    
    claves=list(f.keys())
    print("claves: ",claves)
    dataset3=f['Part_3']
    print(dataset3.shape)
    datos_numpy = np.array(dataset3)
    
    datos_extraidos3 = []
    for i in range(dataset3.shape[0]):  
        ref = dataset3[i, 0]  # Obtener la referencia HDF5
        datos_extraidos3.append(np.array(f[ref]))  # Convertir el objeto referenciado en un array NumPy

    datos_extraidos3 = np.array(datos_extraidos3, dtype=object)

    ppg_signal3=[]
    abp_signal3=[]
    ecg_signal3=[]

    for i in range(datos_extraidos3.shape[0]):
        ppg_signal3.append(datos_extraidos3[i][:, 0])
        abp_signal3.append(datos_extraidos3[i][:, 1])
        ecg_signal3.append(datos_extraidos3[i][:, 2])
#%% Procesamiento de archivo parte 3
fs=125

window_duration=2

ppg_segmented3, abp_segmented3, ecg_segmented3= signal_segmentation(fs, window_duration, ppg_signal3, 
                                                                 abp_signal3, ecg_signal3)
ppg_normalized3, abp_normalized3, ecg_normalized3= signal_normalization(ppg_segmented3, abp_segmented3, ecg_segmented3)

presiones_sistolicas3, presiones_diastolicas3 = get_abp_labels(abp_normalized3)

presiones_sistolicas_norm3, presiones_diastolicas_norm3= labels_normalization(presiones_sistolicas3, presiones_diastolicas3)
#%% Generación de archivo parte 3 .pt y liberación de memoria RAM
save_partial_file(ppg_normalized3, ecg_normalized3, presiones_sistolicas_norm3, presiones_diastolicas_norm3, 6000, 'dataset_parte_3')  
del ppg_signal3
del abp_signal3
del ecg_signal3
del ppg_segmented3
del abp_segmented3
del ecg_segmented3
del ppg_normalized3
del abp_normalized3
del ecg_normalized3
del presiones_diastolicas3
del presiones_sistolicas3
del presiones_diastolicas_norm3
del presiones_sistolicas_norm3
gc.collect()      
#%% #%% Carga de señales de Part_4.mat
archivo_datos4 = "datos/Part_4.mat"
with h5py.File(archivo_datos4, 'r') as f:
    
    claves=list(f.keys())
    print("claves: ",claves)
    dataset4=f['Part_4']
    print(dataset4.shape)
    datos_numpy = np.array(dataset4)
    
    datos_extraidos4 = []
    for i in range(dataset4.shape[0]):  
        ref = dataset4[i, 0]  # Obtener la referencia HDF5
        datos_extraidos4.append(np.array(f[ref]))  # Convertir el objeto referenciado en un array NumPy

    datos_extraidos4 = np.array(datos_extraidos4, dtype=object)

    ppg_signal4=[]
    abp_signal4=[]
    ecg_signal4=[]

    for i in range(datos_extraidos4.shape[0]):
        ppg_signal4.append(datos_extraidos4[i][:, 0])
        abp_signal4.append(datos_extraidos4[i][:, 1])
        ecg_signal4.append(datos_extraidos4[i][:, 2])
#%% Procesamiento de archivo parte 4
fs=125

window_duration=2

ppg_segmented4, abp_segmented4, ecg_segmented4= signal_segmentation(fs, window_duration, ppg_signal4, 
                                                                 abp_signal4, ecg_signal4)
ppg_normalized4, abp_normalized4, ecg_normalized4= signal_normalization(ppg_segmented4, abp_segmented4, ecg_segmented4)

presiones_sistolicas4, presiones_diastolicas4 = get_abp_labels(abp_normalized4)

presiones_sistolicas_norm4, presiones_diastolicas_norm4= labels_normalization(presiones_sistolicas4, presiones_diastolicas4)
#%% Generación de archivo parte 4 .pt y liberación de memoria RAM
save_partial_file(ppg_normalized4, ecg_normalized4, presiones_sistolicas_norm4, presiones_diastolicas_norm4, 9000, 'dataset_parte_4')
del ppg_signal4
del abp_signal4
del ecg_signal4
del ppg_segmented4
del abp_segmented4
del ecg_segmented4
del ppg_normalized4
del abp_normalized4
del ecg_normalized4
del presiones_diastolicas4
del presiones_sistolicas4
del presiones_diastolicas_norm4
del presiones_sistolicas_norm4
gc.collect()
# %%  Mostrar señales de los archivos .pt
"""
import torch
import matplotlib.pyplot as plt
import numpy as np

# Cargar el dataset
ruta = 'data_UCI/dataset_completo_prueba.pt'
data_dict = torch.load(ruta)

data = data_dict['data']           # (N, 2, long_segmento)
labels = data_dict['labels']       # (N, 2)
patient_ids = data_dict['patient_ids']  # (N,)

print(f"Dataset cargado con {data.shape[0]} muestras.")

def visualizar_batch(start_idx=0, batch_size=10):
    end_idx = min(start_idx + batch_size, data.shape[0])
    num_muestras = end_idx - start_idx

    fig, axs = plt.subplots(num_muestras, 1, figsize=(12, 2*num_muestras), sharex=True)

    if num_muestras == 1:
        axs = [axs]

    for i, idx in enumerate(range(start_idx, end_idx)):
        ppg = data[idx, 0].numpy()
        ecg = data[idx, 1].numpy()
        sbp, dbp = labels[idx].tolist()
        pid = patient_ids[idx].item()

        axs[i].plot(ppg, label='PPG', color='blue', alpha=0.7)
        axs[i].plot(ecg, label='ECG', color='orange', alpha=0.7)
        axs[i].set_title(f"Muestra #{idx} - Paciente {pid} - SBP: {sbp:.1f} / DBP: {dbp:.1f}")
        axs[i].legend(loc='upper right')

    plt.xlabel("Tiempo (muestras)")
    plt.tight_layout()
    plt.show()

visualizar_batch(start_idx=300000)
""" 
# %% Histograma SBP Y DBP (señales recortadas)
"""
# Ruta del archivo con las etiquetas
ruta_dataset = 'data_UCI/dataset_completo_prueba.pt' 

# Cargar datos
data_dict = torch.load(ruta_dataset)
labels = data_dict['labels']  

labels_np = labels.numpy()
sbp = labels_np[:, 0]
dbp = labels_np[:, 1]

# Desnormalización
sbp = sbp * (SBP_MAX - SBP_MIN) + SBP_MIN
dbp = dbp * (DBP_MAX - DBP_MIN) + DBP_MIN

# Rango de presiones arteriales
bins_sbp = np.arange(SBP_MIN, SBP_MAX, 5)
bins_dbp = np.arange(DBP_MIN, DBP_MAX, 5)

plt.figure(figsize=(14, 6))

# Histograma SBP
plt.subplot(1, 2, 1)
plt.hist(sbp, bins=bins_sbp, color='crimson', edgecolor='black', alpha=0.75)
plt.title("Distribución de SBP (Presión Sistólica)", fontsize=14)
plt.xlabel("mmHg", fontsize=12)
plt.ylabel("Frecuencia", fontsize=12)
plt.axvline(np.mean(sbp), color='black', linestyle='--', label=f"Media: {np.mean(sbp):.1f}")
plt.legend()

# Histograma DBP
plt.subplot(1, 2, 2)
plt.hist(dbp, bins=bins_dbp, color='steelblue', edgecolor='black', alpha=0.75)
plt.title("Distribución de DBP (Presión Diastólica)", fontsize=14)
plt.xlabel("mmHg", fontsize=12)
plt.axvline(np.mean(dbp), color='black', linestyle='--', label=f"Media: {np.mean(dbp):.1f}")
plt.legend()

plt.tight_layout()
plt.show()
"""