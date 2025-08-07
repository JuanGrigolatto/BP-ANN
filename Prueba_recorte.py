#%% Importación de librerías
import h5py
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import heartpy as hp


#%% Extracción de datos desde archivo HDF5
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

#%% filtrado señales PPG y ECG
def filtrar_ppg(senial_ppg):

    orden = 4
    frec_sup = 21
    frec_inf = 0.5

    frecs_corte = [frec_inf, frec_sup]
    b, a = signal.butter(orden, frecs_corte, 'bandpass', fs = 125)

    ppg_filtrada = signal.filtfilt(b,a, senial_ppg)

     # Línea de base estimada con media móvil
    baseline = np.convolve(ppg_filtrada, np.ones(125)/125, mode='same')

    # Señal sin línea de base
    ppg_filtrada = ppg_filtrada - baseline 
    
    return ppg_filtrada

def filtrar_ecg(senial_ecg):
    orden = 4
    frec_sup = 40
    frec_inf = 0.5

    frecs_corte = [frec_inf, frec_sup]
    b, a = signal.butter(orden, frecs_corte, 'bandpass', fs = 125)

    ecg_filtrada = signal.filtfilt(b,a, senial_ecg)

    # Línea de base estimada con media móvil
    baseline = np.convolve(ecg_filtrada, np.ones(125)/125, mode='same')

    # Señal sin línea de base
    ecg_filtrada = ecg_filtrada - baseline

    return ecg_filtrada

for i in range(len(ppg_signal)):
    ppg_signal[i] = filtrar_ppg(ppg_signal[i])

for i in range(len(ecg_signal)):
    ecg_signal[i] = filtrar_ecg(ecg_signal[i])


#%% Normalización de señales PPG y ECG
def z_score_normalization(signal, epsilon=1e-8):
    signal = np.array(signal)
    mean = np.mean(signal)
    std = np.std(signal)
    return (signal - mean) / (std + epsilon)

for i in range(len(ppg_signal)):
    ppg_signal[i] = z_score_normalization(ppg_signal[i])

for i in range(len(ecg_signal)):
    ecg_signal[i] = z_score_normalization(ecg_signal[i])
#%% Detección de picos en PPG y ECG
def detectar_picos_ppg(ppg, fs=125):
    
    max_val = np.max(ppg)
    min_val = np.min(ppg)
    rango = max_val - min_val

    prominence = 0.1 * rango
    height = min_val + 0.5 * rango
    distancia_min = int(0.4 * fs)

    # Detectar picos
    peaks, _ = signal.find_peaks(ppg, height=height, prominence=prominence, distance=distancia_min)
    
    return peaks

def detectar_picos_ecg(ecg, fs=125, hr_min=40, hr_max=180):
    max_val = np.max(ecg)
    min_val = np.min(ecg)
    rango = max_val - min_val

    prominence = 0.2 * rango              
    height = min_val + 0.6 * rango        

    distancia_min = int(fs * 60 / hr_max)

    peaks, _ = signal.find_peaks(ecg, height=height, prominence=prominence, distance=distancia_min)
    return peaks

#%% Graficación de señales PPG con picos detectados

picos_ppg = []
for i in range(len(ppg_signal)):
    picos = detectar_picos_ppg(ppg_signal[i])
    picos_ppg.append(picos)

plt.figure(figsize=(12, 6))
for i in range(3):
    # Filtrar picos que estén dentro de las primeras 3000 muestras
    picos_filtrados = [p for p in picos_ppg[i] if p < 3000]
    plt.subplot(3, 1, i + 1)
    plt.plot(ppg_signal[i][:3000], label='Señal de PPG', color='blue')
    plt.plot(picos_filtrados, ppg_signal[i][:3000][picos_filtrados], 'ro', label='Picos Detectados')
    plt.title(f'Señal de PPG {i + 1}')
    plt.xlabel('muestras')
    plt.ylabel('Amplitud')
    plt.legend()

#%% Graficación de señales ECG con picos detectados
picos_ecg = []
for i in range(len(ecg_signal)):
    picos = detectar_picos_ecg(ecg_signal[i])
    picos_ecg.append(picos)

plt.figure(figsize=(12, 6))
for i in range(3):
    # Filtrar picos que estén dentro de las primeras 3000 muestras
    picos_filtrados = [p for p in picos_ecg[i] if p < 3000]
    plt.subplot(3, 1, i + 1)
    plt.plot(ecg_signal[i][:3000], label='Señal de ECG', color='blue')
    plt.plot(picos_filtrados, ecg_signal[i][:3000][picos_filtrados], 'ro', label='Picos Detectados')
    plt.title(f'Señal de ECG {i + 1}')
    plt.xlabel('muestras')
    plt.ylabel('Amplitud')
    plt.legend()

# %%
