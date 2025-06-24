# -*- coding: utf-8 -*-
"""
Created on Mon Feb 10 12:52:58 2025

@author: juang
"""
#%% Librerias
import h5py
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks



#%% Carga de señales para entrenamiento

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

#%% Carga de señales para prueba
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


#%% Ploteo de señales    
plt.figure(figsize=(10, 4))
plt.plot(ppg_signal[2000][:1000])  # Mostrar los primeros 1000 puntos
plt.title("Señal PPG - Primer Registro")
plt.xlabel("Muestras")
plt.ylabel("Amplitud")
plt.show()

plt.figure(figsize=(10, 4))
plt.plot(abp_signal[2000][:1000])  # Mostrar los primeros 1000 puntos
plt.title("Señal ABP - Primer Registro")
plt.xlabel("Muestras")
plt.ylabel("Amplitud")
plt.show()
    
plt.figure(figsize=(10, 4))
plt.plot(ecg_signal[2000][:1000])  # Mostrar los primeros 1000 puntos
plt.title("Señal ECG - Primer Registro")
plt.xlabel("Muestras")
plt.ylabel("Amplitud")
plt.show()
    
#%% Segmentación de señales
fs = 125  # Frecuencia de muestreo (Hz)
window_duration = 2  # Duración de la ventana (segundos)
window_size = fs * window_duration  # Tamaño de la ventana en muestras

# Función dividir la señal en ventanas 
def split_into_windows(signal, window_size, overlap):
    step = int(window_size * (1 - overlap))  # Paso entre ventanas
    windows = []
    for i in range(0, len(signal) - window_size + 1, step):
        windows.append(signal[i:i + window_size])
    return np.array(windows)

def min_max_normalization(signal):
    return (signal - np.min(signal)) / (np.max(signal) - np.min(signal))
#%% Segmentación señales entrenamiento
ppg_signal_segmented=[]
abp_signal_segmented=[]
ecg_signal_segmented=[]

#Dividir en ventanas
for i in range(datos_extraidos.shape[0]):
    
    ppg_signal_segmented.append(split_into_windows(ppg_signal[i], window_size, overlap=0.5))
    abp_signal_segmented.append(split_into_windows(abp_signal[i], window_size, overlap=0.5))
    ecg_signal_segmented.append(split_into_windows(ecg_signal[i], window_size, overlap=0.5))

#%% Segmentación señales prueba
ppg_signal_segmented2=[]
abp_signal_segmented2=[]
ecg_signal_segmented2=[]

for j in range(datos_extraidos2.shape[0]):

    ppg_signal_segmented2.append(split_into_windows(ppg_signal2[j], window_size, overlap=0.5))
    abp_signal_segmented2.append(split_into_windows(abp_signal2[j], window_size, overlap=0.5))
    ecg_signal_segmented2.append(split_into_windows(ecg_signal2[j], window_size, overlap=0.5))
    

#%% Normalización de señales entrenamiento

ppg_normalized=[]
abp_normalized=[]
ecg_normalized=[]

for i in range(len(abp_signal_segmented)):
    
    ppg_normalized.append(min_max_normalization(ppg_signal_segmented[i]))
    abp_normalized.append(min_max_normalization(abp_signal_segmented[i]))
    ecg_normalized.append(min_max_normalization(ecg_signal_segmented[i]))

#%% Normalización de señales prueba
ppg_normalized2=[]
abp_normalized2=[]
ecg_normalized2=[]

for j in range(len(abp_signal_segmented2)):

    ppg_normalized2.append(min_max_normalization(ppg_signal_segmented2[j]))
    abp_normalized2.append(min_max_normalization(abp_signal_segmented2[j]))
    ecg_normalized2.append(min_max_normalization(ecg_signal_segmented2[j]))

#%% longitud de las señales segmentadas
print(len(ppg_signal_segmented))
print(len(abp_signal_segmented))
print(len(ecg_signal_segmented))   
for i in range(len(ppg_signal_segmented)):
    print(f"PPG: {len(ppg_signal_segmented[i])}")
    #for j in range(len(ppg_signal_segmented[i])):
        #print(f"PPG: {len(ppg_signal_segmented[i][j])}")  
        
#%% Ploteo de segmentación
plt.figure(figsize=(10, 4))
plt.plot(ppg_signal_segmented[1578][0])  # Mostrar los primeros 1000 puntos
plt.title("Recorte de 2 seg de señal de PPG")
plt.xlabel("Muestras")
plt.ylabel("Amplitud")
plt.show()

plt.figure(figsize=(10, 4))
plt.plot(abp_signal_segmented[1578][0])  # Mostrar los primeros 1000 puntos
plt.title("Recorte de 5 seg de señal de ABP")
plt.xlabel("Muestras")
plt.ylabel("Amplitud")
plt.show()

plt.figure(figsize=(10, 4))
plt.plot(ecg_signal_segmented[1578][0])  # Mostrar los primeros 1000 puntos
plt.title("Recorte de 5 seg de señal de ECG")
plt.xlabel("Muestras")
plt.ylabel("Amplitud")
plt.show()


# Relación de señales en el recorte
plt.figure(figsize=(10, 4))
plt.plot(ppg_normalized[287][0], color= "blue", label="PPG")
plt.plot(abp_normalized[287][0], color= "green", label="ABP")
plt.plot(ecg_normalized[287][0], color= "red", label="ECG") 
plt.title("Relación entre señales en el tiempo")
plt.xlabel("Muestras")
plt.ylabel("Amplitud")
plt.show()

#%% Detección de PS y PD en señal ABP

FRECUENCIA_CARDIACA_MIN = 40  # LPM (latidos por minuto)
FRECUENCIA_CARDIACA_MAX = 180  # LPM
AMPLITUD_SISTOLICA_MIN = 80    # Valor mínimo para un pico sistólico válido
# AMPLITUD_SISTOLICA_MAX = 180   # Valor máximo para un pico sistólico válido
AMPLITUD_DIASTOLICA_MIN = 30   # Valor mínimo para un pico diastólico válido
# AMPLITUD_DIASTOLICA_MAX = 110   # Valor máximo para un pico diastólico válido
DISTANCIA_MINIMA = 50          # Distancia mínima entre picos (en muestras)

def normalizar_presiones(sbp, dbp, sbp_min, sbp_max, dbp_min, dbp_max):
    sbp_norm = (sbp - sbp_min) / (sbp_max - sbp_min)
    dbp_norm = (dbp - dbp_min) / (dbp_max - dbp_min)
    return sbp_norm, dbp_norm

def desnormalizar_presiones(sbp_norm, dbp_norm, sbp_min, sbp_max, dbp_min, dbp_max):
    sbp = sbp_norm * (sbp_max - sbp_min) + sbp_min
    dbp = dbp_norm * (dbp_max - dbp_min) + dbp_min
    return ps, pd
#%% Normalización señales entrenamiento
matriz_picos_sistolicos=[]
matriz_picos_diastolicos=[]
matriz_presiones_sistolicas=[]
matriz_presiones_diastolicas=[]
matriz_presiones_sistolicas_norm=[]
matriz_presiones_diastolicas_norm=[]

for i in range(len(abp_signal_segmented)):
    picos_sistolicos=[]
    picos_diastolicos=[]
    presiones_sistolicas = []
    presiones_diastolicas = []
    for j in range(len(abp_signal_segmented[i])):
        # Encontrar picos en la señal ABP
        peakss,_ = find_peaks(abp_signal_segmented[i][j], height=AMPLITUD_SISTOLICA_MIN, distance=DISTANCIA_MINIMA) 
        peaksd,_ = find_peaks(-(abp_signal_segmented)[i][j], distance=DISTANCIA_MINIMA)
        picos_sistolicos.append(peakss)
        picos_diastolicos.append(peaksd)
        # Filtrar picos sistólicos (eliminar los no deseados)
        # señal_abp = abp_signal_segmented[i][j]
        # peakss = [p for p in peakss if AMPLITUD_SISTOLICA_MIN <= señal_abp[p] <= AMPLITUD_SISTOLICA_MAX]

        # Filtrar picos diastólicos (eliminar los no deseados)
        # peaksd = [p for p in peaksd if -(AMPLITUD_DIASTOLICA_MIN) <= -señal_abp[p] <= -(AMPLITUD_DIASTOLICA_MAX)]
        # Estimación de las presiones sistolicas y diastolicas promedios de la señal de 10 seg 
        if (len(peakss)>0):
            ps = np.mean(abp_signal_segmented[i][j][peakss])
        else:
            ps = np.nan
        
        if (len(peaksd)>0):
            pd = np.mean(abp_signal_segmented[i][j][peaksd])
        else:
            pd = np.nan
        presiones_sistolicas.append(ps)
        presiones_diastolicas.append(pd)
        
    matriz_presiones_sistolicas.append(presiones_sistolicas)
    matriz_presiones_diastolicas.append(presiones_diastolicas)
        
    matriz_picos_sistolicos.append(picos_sistolicos)
    matriz_picos_diastolicos.append(picos_diastolicos)

    matriz_presiones_sistolicas_norm=matriz_presiones_sistolicas
    matriz_presiones_diastolicas_norm=matriz_presiones_diastolicas
# Aplanar y convertir a arrays de numpy
todas_sbp = np.array([x for sublista in matriz_presiones_sistolicas for x in sublista if not np.isnan(x)])
todas_dbp = np.array([x for sublista in matriz_presiones_diastolicas for x in sublista if not np.isnan(x)])

SBP_MIN = np.min(todas_sbp)
SBP_MAX = np.max(todas_sbp)
DBP_MIN = np.min(todas_dbp)
DBP_MAX = np.max(todas_dbp)

print(f"SBP_MIN={SBP_MIN:.2f}, SBP_MAX={SBP_MAX:.2f}")
print(f"DBP_MIN={DBP_MIN:.2f}, DBP_MAX={DBP_MAX:.2f}")

for i in range(len(matriz_presiones_sistolicas)):  
    for j in range(len(matriz_presiones_sistolicas[i])):

        ps_norm,pd_norm=normalizar_presiones(matriz_presiones_sistolicas[i][j], matriz_presiones_diastolicas[i][j],
                                     SBP_MIN, SBP_MAX,DBP_MIN,DBP_MAX)
        matriz_presiones_sistolicas_norm[i][j]=ps_norm
        matriz_presiones_diastolicas_norm[i][j]=pd_norm
#%% Normalización señales prueba

matriz_picos_sistolicos2=[]
matriz_picos_diastolicos2=[]
matriz_presiones_sistolicas2=[]
matriz_presiones_diastolicas2=[]
matriz_presiones_sistolicas_norm2=[]
matriz_presiones_diastolicas_norm2=[]

for i in range(len(abp_signal_segmented2)):
    picos_sistolicos2=[]
    picos_diastolicos2=[]
    presiones_sistolicas2 = []
    presiones_diastolicas2 = []
    for j in range(len(abp_signal_segmented2[i])):
        # Encontrar picos en la señal ABP
        peakss2,_ = find_peaks(abp_signal_segmented2[i][j], height=AMPLITUD_SISTOLICA_MIN, distance=DISTANCIA_MINIMA) 
        peaksd2,_ = find_peaks(-(abp_signal_segmented2)[i][j], distance=DISTANCIA_MINIMA)
        picos_sistolicos2.append(peakss2)
        picos_diastolicos2.append(peaksd2)
        # Filtrar picos sistólicos (eliminar los no deseados)
        # señal_abp = abp_signal_segmented[i][j]
        # peakss = [p for p in peakss if AMPLITUD_SISTOLICA_MIN <= señal_abp[p] <= AMPLITUD_SISTOLICA_MAX]

        # Filtrar picos diastólicos (eliminar los no deseados)
        # peaksd = [p for p in peaksd if -(AMPLITUD_DIASTOLICA_MIN) <= -señal_abp[p] <= -(AMPLITUD_DIASTOLICA_MAX)]
        # Estimación de las presiones sistolicas y diastolicas promedios de la señal de 10 seg 
        if (len(peakss2)>0):
            ps = np.mean(abp_signal_segmented2[i][j][peakss2])
        else:
            ps = np.nan
        
        if (len(peaksd2)>0):
            pd = np.mean(abp_signal_segmented2[i][j][peaksd2])
        else:
            pd = np.nan
        presiones_sistolicas2.append(ps)
        presiones_diastolicas2.append(pd)
        
    matriz_presiones_sistolicas2.append(presiones_sistolicas2)
    matriz_presiones_diastolicas2.append(presiones_diastolicas2)
        
    matriz_picos_sistolicos2.append(picos_sistolicos2)
    matriz_picos_diastolicos2.append(picos_diastolicos2)

    matriz_presiones_sistolicas_norm2=matriz_presiones_sistolicas2
    matriz_presiones_diastolicas_norm2=matriz_presiones_diastolicas2
# Aplanar y convertir a arrays de numpy
todas_sbp2 = np.array([x for sublista in matriz_presiones_sistolicas2 for x in sublista if not np.isnan(x)])
todas_dbp2 = np.array([x for sublista in matriz_presiones_diastolicas2 for x in sublista if not np.isnan(x)])

SBP_MIN2 = np.min(todas_sbp2)
SBP_MAX2 = np.max(todas_sbp2)
DBP_MIN2 = np.min(todas_dbp2)
DBP_MAX2 = np.max(todas_dbp2)

print(f"SBP_MIN={SBP_MIN2:.2f}, SBP_MAX={SBP_MAX2:.2f}")
print(f"DBP_MIN={DBP_MIN2:.2f}, DBP_MAX={DBP_MAX2:.2f}")

for i in range(len(matriz_presiones_sistolicas2)):  
    for j in range(len(matriz_presiones_sistolicas2[i])):

        ps_norm,pd_norm=normalizar_presiones(matriz_presiones_sistolicas2[i][j], matriz_presiones_diastolicas2[i][j],
                                     SBP_MIN2, SBP_MAX2,DBP_MIN2,DBP_MAX2)
        matriz_presiones_sistolicas_norm2[i][j]=ps_norm
        matriz_presiones_diastolicas_norm2[i][j]=pd_norm


#%% Ploteo Detección de picos sistolicos y diastolicos
"""
m3000=2012
for i in range(len(abp_signal_segmented[m3000])):
    n10s=i
    plt.figure(figsize=(10, 4))
    plt.plot(abp_signal_segmented[m3000][n10s])  # Mostrar los primeros 1000 puntos
    plt.plot(matriz_picos_sistolicos[m3000][n10s], abp_signal_segmented[m3000][n10s][matriz_picos_sistolicos[m3000][n10s]], "x")
    plt.plot(matriz_picos_diastolicos[m3000][n10s], abp_signal_segmented[m3000][n10s][matriz_picos_diastolicos[m3000][n10s]], "x")
    plt.axhline(y=matriz_presiones_sistolicas[m3000][n10s], color='r', linestyle='--', label=f'Presión sistólica promedio: {matriz_presiones_sistolicas[m3000][n10s]:.2f}')
    plt.axhline(y=matriz_presiones_diastolicas[m3000][n10s], color='g', linestyle='--', label=f'Presión diastólica promedio: {matriz_presiones_diastolicas[m3000][n10s]:.2f}')
    plt.text(0.5, matriz_presiones_sistolicas[m3000][n10s] + 9, f'Sistólica: {matriz_presiones_sistolicas[m3000][n10s]:.2f}', color='r', fontsize=10, ha='center')
    plt.text(0.5, matriz_presiones_diastolicas[m3000][n10s] - 12, f'Diastólica: {matriz_presiones_diastolicas[m3000][n10s]:.2f}', color='g', fontsize=10, ha='center')
    plt.title("Detección de picos sistolicos y diastolicos en ABP")
    plt.xlabel("Muestras")
    plt.ylabel("Amplitud")
    plt.show()
"""
#%% Formación de dataset y etiquetas normalizadas para entrenamiento y evaluacion


# Guardar datos en formato .pt 
"""
import os
import torch

output_dir = 'datos_UCI'
os.makedirs(output_dir, exist_ok=True)

index = 0

for paciente_id, (ppg_list, ecg_list, sbp_list, dbp_list) in enumerate(zip(ppg_signal_segmented, ecg_signal_segmented, matriz_presiones_sistolicas, matriz_presiones_diastolicas)):
    for seg_id, (ppg, ecg, sbp, dbp) in enumerate(zip(ppg_list, ecg_list, sbp_list, dbp_list)):
        ppg = np.asarray(ppg)
        ecg = np.asarray(ecg)

        # Concatenar señales en un tensor (2, long_segmento)
        signal = np.stack([ppg, ecg], axis=0)
        signal_tensor = torch.tensor(signal, dtype=torch.float32)

        # Guardar etiquetas como tensor (SBP, DBP)
        label_tensor = torch.tensor([sbp, dbp], dtype=torch.float32)

        # Identificador de paciente
        patient_tensor = torch.tensor(paciente_id, dtype=torch.long)

        # Crear contenedor
        data = {
            'signal': signal_tensor,
            'label': label_tensor,
            'patient_id': patient_tensor
        }

        # Guardar archivo .pt
        torch.save(data, os.path.join(output_dir, f'{index:05d}.pt'))
        index += 1

print(f"Guardados {index} archivos .pt en {output_dir}")
"""
import os
import torch
import numpy as np

output_dir = 'data_UCI'
os.makedirs(output_dir, exist_ok=True)

all_signals = []
all_labels = []
all_patient_ids = []

for paciente_id, (ppg_list, ecg_list, sbp_list, dbp_list) in enumerate(zip(ppg_normalized, ecg_normalized, matriz_presiones_sistolicas_norm, matriz_presiones_diastolicas_norm)):
    for seg_id, (ppg, ecg, sbp, dbp) in enumerate(zip(ppg_list, ecg_list, sbp_list, dbp_list)):
        ppg = np.asarray(ppg)
        ecg = np.asarray(ecg)

        # Verificar si hay NaN en señales o etiquetas
        if np.isnan(ppg).any() or np.isnan(ecg).any() or np.isnan(sbp) or np.isnan(dbp):
            continue

        # Concatenar señales (2, long_segmento)
        signal = np.stack([ppg, ecg], axis=0)
        signal_tensor = torch.tensor(signal, dtype=torch.float32)
        label_tensor = torch.tensor([sbp, dbp], dtype=torch.float32)
        patient_tensor = torch.tensor(paciente_id, dtype=torch.long)

        all_signals.append(signal_tensor)
        all_labels.append(label_tensor)
        all_patient_ids.append(patient_tensor)

# Convertir listas a tensores
data_tensor = torch.stack(all_signals)          # shape: (N, 2, long_segmento)
labels_tensor = torch.stack(all_labels)         # shape: (N, 2)
patient_ids_tensor = torch.stack(all_patient_ids)  # shape: (N,)

# Guardar todo en un único archivo .pt
torch.save({
    'data': data_tensor,
    'labels': labels_tensor,
    'patient_ids': patient_ids_tensor
}, os.path.join(output_dir, 'dataset_completo.pt'))

print(f"Guardado dataset unificado: {data_tensor.shape[0]} muestras.")

# %% Formación de dataset para prueba desnormalizados

import os
import torch
import numpy as np

output_dir = 'data_UCI'
os.makedirs(output_dir, exist_ok=True)

all_signals2 = []
all_labels2 = []
all_patient_ids2 = []

for paciente_id2, (ppg_list2, ecg_list2, sbp_list2, dbp_list2) in enumerate(zip(ppg_signal_segmented2, ecg_signal_segmented2, matriz_presiones_sistolicas2, matriz_presiones_diastolicas2)):
    for seg_id, (ppg2, ecg2, sbp2, dbp2) in enumerate(zip(ppg_list2, ecg_list2, sbp_list2, dbp_list2)):
        ppg2 = np.asarray(ppg2)
        ecg2 = np.asarray(ecg2)

        # Verificar si hay NaN en señales o etiquetas
        if np.isnan(ppg2).any() or np.isnan(ecg2).any() or np.isnan(sbp2) or np.isnan(dbp2):
            continue

        # Concatenar señales (2, long_segmento)
        signal2 = np.stack([ppg2, ecg2], axis=0)
        signal_tensor2 = torch.tensor(signal2, dtype=torch.float32)
        label_tensor2 = torch.tensor([sbp2, dbp2], dtype=torch.float32)
        patient_tensor2 = torch.tensor(paciente_id2, dtype=torch.long)

        all_signals2.append(signal_tensor2)
        all_labels2.append(label_tensor2)
        all_patient_ids2.append(patient_tensor2)

# Convertir listas a tensores
data_tensor2 = torch.stack(all_signals2)          # shape: (N, 2, long_segmento)
labels_tensor2 = torch.stack(all_labels2)         # shape: (N, 2)
patient_ids_tensor2 = torch.stack(all_patient_ids2)  # shape: (N,)

# Guardar todo en un único archivo .pt
torch.save({
    'data': data_tensor2,
    'labels': labels_tensor2,
    'patient_ids': patient_ids_tensor2
}, os.path.join(output_dir, 'dataset_completo_prueba.pt'))

print(f"Guardado dataset unificado: {data_tensor2.shape[0]} muestras.")
# %% mostrar datos guardados
"""
import os
import torch
import matplotlib.pyplot as plt

# Configuración
output_dir = 'datos_UCI'
num_files_to_inspect = 20  # Cantidad de archivos a visualizar
start_num_batch = 1  # Número de lote inicial para inspeccionar
# Obtener lista ordenada de archivos
files = sorted([f for f in os.listdir(output_dir) if f.endswith('.pt')])

for i, filename in enumerate(files[start_num_batch:(start_num_batch+num_files_to_inspect)]):
    filepath = os.path.join(output_dir, filename)
    data = torch.load(filepath)
    
    print(f"\n=== Archivo {i+1}/{num_files_to_inspect}: {filename} ===")
    print(f"Paciente ID: {data['patient_id'].item()}")
    print(f"Etiquetas (SBP, DBP): {data['label'].numpy()}")
    print(f"Dimensión señal (canales, longitud): {data['signal'].shape}")
   

    # Visualización de señales
    #plt.figure(figsize=(12, 4))
    #plt.plot(data['signal'][0], label='PPG', color='red', alpha=0.7)
    #plt.plot(data['signal'][1], label='ECG', color='blue', alpha=0.7)
    #plt.title(f"Señal {filename}\nPaciente {data['patient_id'].item()} | SBP: {data['label'][0].item()} | DBP: {data['label'][1].item()}")
    #plt.xlabel("Muestras")
    #plt.ylabel("Amplitud")
    #plt.legend()
    #plt.tight_layout()
    #plt.show()
    """
# %% Ploteo distribución de presiones sistólicas y diastólicas
""" Este código no plotea, se queda ejecutando indefinidamente
output_dir = 'datos_UCI'

# Cargar todos los datos
sbp_values = []
dbp_values = []

for filename in os.listdir(output_dir):
    if filename.endswith('.pt'):
        data = torch.load(os.path.join(output_dir, filename))
        sbp_values.append(data['label'][0].item())
        dbp_values.append(data['label'][1].item())

# Convertir a arrays de numpy
sbp = np.array(sbp_values)
dbp = np.array(dbp_values)

# Configurar los bins (personaliza según tus datos)
bin_ranges = np.arange(50, 221, 10)  # De 50 a 220 mmHg en pasos de 10

# Crear figura
plt.figure(figsize=(14, 6))

# Gráfico de SBP
plt.subplot(1, 2, 1)
plt.hist(sbp, bins=bin_ranges, color='#E74C3C', edgecolor='black', alpha=0.8)
plt.title('Distribución de Presión Sistólica (SBP)', fontsize=14, pad=20)
plt.xlabel('mmHg', fontsize=12)
plt.ylabel('Frecuencia', fontsize=12)
plt.axvline(x=np.mean(sbp), color='black', linestyle='--', label=f'Media: {np.mean(sbp):.1f}')
plt.legend()

# Gráfico de DBP
plt.subplot(1, 2, 2)
plt.hist(dbp, bins=bin_ranges, color='#3498DB', edgecolor='black', alpha=0.8)
plt.title('Distribución de Presión Diastólica (DBP)', fontsize=14, pad=20)
plt.xlabel('mmHg', fontsize=12)
plt.axvline(x=np.mean(dbp), color='black', linestyle='--', label=f'Media: {np.mean(dbp):.1f}')
plt.legend()

# Ajustes finales
plt.suptitle('Distribución de Valores de Presión Arterial', fontsize=16, y=1.02)
plt.tight_layout()

plt.show()"""
# %%  Mostrar señales de los archivos .pt

import torch
import os
import matplotlib.pyplot as plt

# Parámetros configurables
input_dir = 'datos_UCI'
batch_size = 5          # Cantidad de archivos a mostrar por batch
batch_index = 0         # Cambiá este índice para ver otros batch

# Obtener lista ordenada de archivos
files = sorted(os.listdir(input_dir))
total_batches = len(files) // batch_size + int(len(files) % batch_size != 0)

# Calcular rango de archivos a mostrar en este batch
start = batch_index * batch_size
end = min(start + batch_size, len(files))
batch_files = files[start:end]

print(f"Mostrando batch {batch_index+1}/{total_batches} con archivos {start} a {end-1}")

# Mostrar señales del batch
for filename in batch_files:
    filepath = os.path.join(input_dir, filename)
    data = torch.load(filepath)

    signal = data['signal']       # (2, long)
    label = data['label']         # (SBP, DBP)
    patient_id = data['patient_id']

    ppg = signal[0].numpy()
    ecg = signal[1].numpy()

    # Graficar señales
    plt.figure(figsize=(10, 4))
    plt.subplot(2, 1, 1)
    plt.plot(ppg)
    plt.title(f'PPG - File: {filename} - Paciente: {patient_id.item()} - SBP: {label[0].item():.1f} / DBP: {label[1].item():.1f}')
    plt.ylabel('Amplitud')

    plt.subplot(2, 1, 2)
    plt.plot(ecg)
    plt.title('ECG')
    plt.ylabel('Amplitud')
    plt.xlabel('Tiempo (muestras)')

    plt.tight_layout()
    plt.show()


