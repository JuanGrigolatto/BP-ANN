# -*- coding: utf-8 -*-
"""
Created on Mon Feb 10 12:52:58 2025

@author: juang
"""

import h5py
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

# Carga de señales

archivo_datos1 = "Part_1.mat"

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
    
    plt.figure(figsize=(10, 4))
    plt.plot(ppg_signal[2000][:250])  # Mostrar los primeros 1000 puntos
    plt.title("Señal PPG - Primer Registro")
    plt.xlabel("Muestras")
    plt.ylabel("Amplitud")
    plt.show()

    plt.figure(figsize=(10, 4))
    plt.plot(abp_signal[2000][:250])  # Mostrar los primeros 1000 puntos
    plt.title("Señal ABP - Primer Registro")
    plt.xlabel("Muestras")
    plt.ylabel("Amplitud")
    plt.show()
    
    plt.figure(figsize=(10, 4))
    plt.plot(ecg_signal[2000][:250])  # Mostrar los primeros 1000 puntos
    plt.title("Señal ECG - Primer Registro")
    plt.xlabel("Muestras")
    plt.ylabel("Amplitud")
    plt.show()
    
# Segmentación
fs = 125  # Frecuencia de muestreo (Hz)
window_duration = 2  # Duración de la ventana (segundos)
window_size = fs * window_duration  # Tamaño de la ventana en muestras

ppg_signal_segmented=[]
abp_signal_segmented=[]

# Función dividir la señal en ventanas 
# def split_into_windows(signal, window_size, overlap=0.5):
#     step = int(window_size * (1 - overlap))  # Paso entre ventanas
#     windows = []
#     for i in range(0, len(signal) - window_size + 1, step):
#         windows.append(signal[i:i + window_size])
#     return np.array(windows)



# Dividir en ventanas
# for i in range(datos_extraidos.shape[0]):
    
#     ppg_signal_segmented.append(split_into_windows(ppg_signal[i], window_size, overlap=0.5))
#     abp_signal_segmented.append(split_into_windows(ppg_signal[i], window_size, overlap=0.5))
# # Ver el resultado
# print(len(ppg_signal_segmented))  # (N ventanas, x muestras)

windows=[]
for i in range(datos_extraidos.shape[0]):
    # Encontrar picos en la señal PPG
    peaks, _ = find_peaks(ppg_signal[i], height=0.01, distance=fs)  # Ajusta los parámetros según tu señal

    # Segmentar la señal alineando las ventanas con los picos
    
    for j in range(len(peaks) - 1):
        start = peaks[j] - int(0.2 * fs)  # Comenzar 0.3 segundos antes del pico
        end = peaks[j] + int(0.8 * fs)    # Terminar 0.7 segundos después del pico
        if end < len(ppg_signal[i]):  # Evitar índices fuera de rango
            windows.append(ppg_signal[i][start:end])
             



plt.figure(figsize=(10, 4))
plt.plot(windows[40000])  # Mostrar los primeros 1000 puntos
plt.title("Pico PPG")
plt.xlabel("Muestras")
plt.ylabel("Amplitud")
plt.show()
