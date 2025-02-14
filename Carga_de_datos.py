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
window_duration = 5  # Duración de la ventana (segundos)
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
# windows=[]
# for i in range(datos_extraidos.shape[0]):
#     # Encontrar picos en la señal PPG
#     peaks, _ = find_peaks(ppg_signal[i], height=0.01, distance=fs)  # Ajusta los parámetros según tu señal

#     # Segmentar la señal alineando las ventanas con los picos
    
#     for j in range(len(peaks) - 1):
#         start = peaks[j] - int(0.2 * fs)  # Comenzar 0.3 segundos antes del pico
#         end = peaks[j] + int(0.8 * fs)    # Terminar 0.7 segundos después del pico
#         if end < len(ppg_signal[i]):  # Evitar índices fuera de rango
#             windows.append(ppg_signal[i][start:end])



#%% Normalización de señales

ppg_normalized=[]
abp_normalized=[]
ecg_normalized=[]
for i in range(len(abp_signal_segmented)):
    
    ppg_normalized.append(min_max_normalization(ppg_signal_segmented[i]))
    abp_normalized.append(min_max_normalization(abp_signal_segmented[i]))
    ecg_normalized.append(min_max_normalization(ecg_signal_segmented[i]))
        
        
#%% Ploteo de segmentación
plt.figure(figsize=(10, 4))
plt.plot(ppg_signal_segmented[1578][0])  # Mostrar los primeros 1000 puntos
plt.title("Recorte de 10 seg de señal de PPG")
plt.xlabel("Muestras")
plt.ylabel("Amplitud")
plt.show()

plt.figure(figsize=(10, 4))
plt.plot(abp_signal_segmented[1578][0])  # Mostrar los primeros 1000 puntos
plt.title("Recorte de 10 seg de señal de ABP")
plt.xlabel("Muestras")
plt.ylabel("Amplitud")
plt.show()

plt.figure(figsize=(10, 4))
plt.plot(ecg_signal_segmented[1578][0])  # Mostrar los primeros 1000 puntos
plt.title("Recorte de 10 seg de señal de ECG")
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

# #Detección picos diastolicos

# #se invierte la señal

# abp_signal_segmented_invertida=-(abp_signal_segmented[69][0])

# plt.figure(figsize=(10, 4))
# plt.plot(abp_signal_segmented_invertida)  # Mostrar los primeros 1000 puntos
# plt.title("ABP invertida")
# plt.xlabel("Muestras")
# plt.ylabel("Amplitud")
# plt.show()

# peaksd, amp_peaksd= find_peaks(abp_signal_segmented_invertida, distance=60)  

# plt.figure(figsize=(10, 4))
# plt.plot(abp_signal_segmented_invertida)  # Mostrar los primeros 1000 puntos
# plt.plot(peaksd, abp_signal_segmented_invertida[peaksd], "x")
# plt.title("Detección de picos diastolicos en ABP invertida")
# plt.xlabel("Muestras")
# plt.ylabel("Amplitud")
# plt.show()

# plt.figure(figsize=(10, 4))
# plt.plot(abp_signal_segmented[69][0])  # Mostrar los primeros 1000 puntos
# plt.plot(peaksd, abp_signal_segmented[69][0][peaksd], "x")
# plt.title("Detección de picos diastolicos en ABP")
# plt.xlabel("Muestras")
# plt.ylabel("Amplitud")
# plt.show()



    