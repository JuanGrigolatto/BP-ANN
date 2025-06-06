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



#%% Carga de señales

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

ppg_signal_segmented=[]
abp_signal_segmented=[]
ecg_signal_segmented=[]

# Función dividir la señal en ventanas 
def split_into_windows(signal, window_size, overlap):
    step = int(window_size * (1 - overlap))  # Paso entre ventanas
    windows = []
    for i in range(0, len(signal) - window_size + 1, step):
        windows.append(signal[i:i + window_size])
    return np.array(windows)

def min_max_normalization(signal):
    return (signal - np.min(signal)) / (np.max(signal) - np.min(signal))

#Dividir en ventanas
for i in range(datos_extraidos.shape[0]):
    
    ppg_signal_segmented.append(split_into_windows(ppg_signal[i], window_size, overlap=0.5))
    abp_signal_segmented.append(split_into_windows(abp_signal[i], window_size, overlap=0.5))
    ecg_signal_segmented.append(split_into_windows(ecg_signal[i], window_size, overlap=0.5))

#%% Normalización de señales

ppg_normalized=[]
abp_normalized=[]
ecg_normalized=[]
for i in range(len(abp_signal_segmented)):
    
    ppg_normalized.append(min_max_normalization(ppg_signal_segmented[i]))
    abp_normalized.append(min_max_normalization(abp_signal_segmented[i]))
    ecg_normalized.append(min_max_normalization(ecg_signal_segmented[i]))

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


matriz_picos_sistolicos=[]
matriz_picos_diastolicos=[]
matriz_presiones_sistolicas=[]
matriz_presiones_diastolicas=[]

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
#%% Formación de dataset y etiquetas


# Guardar datos en formato .pt para PyTorch !!!
import os
import torch

output_dir = 'datos_UCI'
os.makedirs(output_dir, exist_ok=True)

index = 0

for paciente_id, (ppg_list, ecg_list, sbp_list, dbp_list) in enumerate(zip(ppg_normalized, ecg_normalized, matriz_presiones_sistolicas, matriz_presiones_diastolicas)):
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
    
# %% mostrar datos guardados
"""
import os
import torch
import matplotlib.pyplot as plt

# Configuración
output_dir = 'datos_UCI'
num_files_to_inspect = 20  # Cantidad de archivos a visualizar
start_num_batch = 70000  # Número de lote inicial para inspeccionar
# Obtener lista ordenada de archivos
files = sorted([f for f in os.listdir(output_dir) if f.endswith('.pt')])

for i, filename in enumerate(files[start_num_batch:(start_num_batch+num_files_to_inspect)]):
    filepath = os.path.join(output_dir, filename)
    data = torch.load(filepath)
    
    print(f"\n=== Archivo {i+1}/{num_files_to_inspect}: {filename} ===")
    print(f"Paciente ID: {data['patient_id'].item()}")
    print(f"Etiquetas (SBP, DBP): {data['label'].numpy()}")
    print(f"Dimensión señal (canales, longitud): {data['signal'].shape}")
   """ 

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
# %%
