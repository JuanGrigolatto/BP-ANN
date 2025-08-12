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
    #baseline = np.convolve(ppg_filtrada, np.ones(125)/125, mode='same')
    
    #Filtro de media 
    baseline = np.mean(ppg_filtrada)

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
    #baseline = np.convolve(ecg_filtrada, np.ones(125)/125, mode='same')

    #Filtro de media 
    baseline = np.mean(ecg_filtrada)

    # Señal sin línea de base
    ecg_filtrada = ecg_filtrada - baseline

    return ecg_filtrada

def filtrado_para_deteccion_Q(senial_ecg):
    orden = 4
    frec_sup = 15
    frec_inf = 5
    frecs_corte = [frec_inf, frec_sup]

    b, a = signal.butter(orden, frecs_corte, 'bandpass', fs = 125)

    ecg_filtrada = signal.filtfilt(b,a, senial_ecg)
    
    return ecg_filtrada


for i in range(len(ppg_signal)):
    ppg_signal[i] = filtrar_ppg(ppg_signal[i])

for i in range(len(ecg_signal)):
    ecg_signal[i] = filtrar_ecg(ecg_signal[i])
    ecg_signal[i] = filtrado_para_deteccion_Q(ecg_signal[i])


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
    height = min_val + 0.4 * rango        

    distancia_min = int(fs * 60 / hr_max)

    peaks, _ = signal.find_peaks(ecg, height=height, prominence=prominence, distance=distancia_min)
    return peaks

#%% Graficación de señales PPG con picos detectados

picos_ppg = []
for i in range(len(ppg_signal)):
    picos = detectar_picos_ppg(ppg_signal[i])
    picos_ppg.append(picos)
inicio=60
plt.figure(figsize=(12, 6))    
for i in range(10):
    # Filtrar picos que estén dentro de las primeras 3000 muestras
    
    picos_filtrados = [p for p in picos_ppg[i+inicio] if p < 3000]
    plt.subplot(10, 1, i + 1)
    plt.plot(ppg_signal[i+inicio][:3000], label='Señal de PPG', color='blue')
    plt.plot(picos_filtrados, ppg_signal[i+inicio][:3000][picos_filtrados], 'ro', label='Picos Detectados')
    plt.title(f'Señal de PPG {i + 1}')
    plt.xlabel('muestras')
    plt.ylabel('Amplitud')
    plt.legend()

#%% Graficación de señales ECG con picos detectados
picos_ecg = []
for i in range(len(ecg_signal)):
    picos = detectar_picos_ecg(ecg_signal[i])
    picos_ecg.append(picos)
inicio=80


plt.figure(figsize=(12, 6))
for i in range(10):
    # Filtrar picos que estén dentro de las primeras 3000 muestras
    picos_filtrados = [p for p in picos_ecg[i+inicio] if p < 500]
    plt.subplot(10, 1, i + 1)
    plt.plot(ecg_signal[i+inicio][:500], label='Señal de ECG', color='blue')
    plt.plot(picos_filtrados, ecg_signal[i+inicio][:500][picos_filtrados], 'ro', label='Picos Detectados')
    plt.title(f'Señal de ECG {i + 1}')
    plt.xlabel('muestras')
    plt.ylabel('Amplitud')
    plt.legend()

# %% Definición y graficación de ventanas
t_duration = 2  # Duración en segundos de cada segmento
fs = 125  # Frecuencia de muestreo
window_width = t_duration * fs  # Ancho fijo por muestra

# Definición de ventanas 
hamming_window = signal.windows.hamming(window_width)
hanning_window = signal.windows.hann(window_width)
blackman_window = signal.windows.blackman(window_width)
rectangular_window = np.ones(window_width)

plt.plot(hamming_window, label="Hamming")
plt.plot(hanning_window, label="Hanning")
plt.plot(blackman_window, label="Blackman")
plt.plot(rectangular_window, label="Rectangular")
plt.legend()
plt.title("Morfología ventanas")
plt.show()

# %% Recorte de pulsos PPG y ECG a longitud fija

def recortar_por_ventanas_fijas(senial, fs = 125, window ='Hamming', t_duration = 10, overlap = 0.7):
    window_width = t_duration * fs
    
    if window == 'Hamming':
        ventana = signal.windows.hamming(window_width)
    elif window == 'Hanning':
        ventana = signal.windows.hann(window_width)
    elif window == 'Blackman':
        ventana = signal.windows.blackman(window_width)
    elif window == 'Rectangular':
        ventana = np.ones(window_width)

    step = int(window_width * (1 - overlap))

    segments = []
    for i in range(0, len(senial) - window_width + 1, step):
        frame = senial[i:i+window_width]    # cortar segmento
        frame_win = frame * ventana  # aplicar ventana
        segments.append(frame_win)

    return segments

ppg_signal_hamming = recortar_por_ventanas_fijas(ppg_signal[0], window='Hamming', t_duration=2, overlap=0.7)
ppg_signal_hanning = recortar_por_ventanas_fijas(ppg_signal[0], window='Hanning', t_duration=2, overlap=0.7)
ppg_signal_blackman = recortar_por_ventanas_fijas(ppg_signal[0], window='Blackman', t_duration=2, overlap=0.7)
ppg_signal_rectangular = recortar_por_ventanas_fijas(ppg_signal[0], window='Rectangular', t_duration=2, overlap=0.7)

plt.figure(figsize=(12, 6))
plt.subplot(4, 1, 1)
plt.plot(ppg_signal_hamming[0], label='Señal de PPG hamming', color='blue')
plt.legend()
plt.tight_layout()
plt.subplot(4, 1, 2)
plt.plot(ppg_signal_hanning[0], label='Señal de PPG hanning', color='blue')
plt.legend()
plt.tight_layout()
plt.subplot(4, 1, 3)
plt.plot(ppg_signal_blackman[0], label='Señal de PPG blackman', color='blue')
plt.legend()
plt.tight_layout()
plt.subplot(4, 1, 4)
plt.plot(ppg_signal_rectangular[0], label='Señal de PPG rectangular', color='blue')
plt.legend()
plt.tight_layout()
plt.show()

ecg_signal_hamming = recortar_por_ventanas_fijas(ecg_signal[0], window='Hamming', t_duration=10, overlap=0.7)
ecg_signal_hanning = recortar_por_ventanas_fijas(ecg_signal[0], window='Hanning', t_duration=10, overlap=0.7)
ecg_signal_blackman = recortar_por_ventanas_fijas(ecg_signal[0], window='Blackman', t_duration=10, overlap=0.7)
ecg_signal_rectangular = recortar_por_ventanas_fijas(ecg_signal[0], window='Rectangular', t_duration=10, overlap=0.7)

plt.figure(figsize=(12, 6))
plt.subplot(4, 1, 1)
plt.plot(ecg_signal_hamming[0], label='Señal de PPG hamming', color='blue')
plt.legend()
plt.tight_layout()
plt.subplot(4, 1, 2)
plt.plot(ecg_signal_hanning[0], label='Señal de PPG hanning', color='blue')
plt.legend()
plt.tight_layout()
plt.subplot(4, 1, 3)
plt.plot(ecg_signal_blackman[0], label='Señal de PPG blackman', color='blue')
plt.legend()
plt.tight_layout()
plt.subplot(4, 1, 4)
plt.plot(ecg_signal_rectangular[0], label='Señal de PPG rectangular', color='blue')
plt.legend()
plt.tight_layout()
plt.show()
# %% Recorte de pulsos PPG y ECG a longitud variable 
def recortar_por_picos(seniales, list_peaks, overlap_peaks=4):
    all_segments = []
    all_starts = []
    all_stops = []
    
    for senial, peaks in zip(seniales, list_peaks):
        i=0
        segments = []
        starts = []
        stops = []        
        while i < (len(peaks) - 4):
        
            intervalo_rr_first = peaks[i+1] - peaks[i]
            intervalo_rr_last = peaks[i+4] - peaks[i+3] 
        
            pre_qrs=int(0.25 * intervalo_rr_first)   # para incluir onda P 
            post_qrs=int(0.45 *intervalo_rr_last)   # para incluir onda T

            start = max(0, peaks[i] - pre_qrs)
            stop  = min(len(senial), peaks[i+4] + post_qrs)

            window = senial[start:stop]

            starts.append(start)
            stops.append(stop)
            segments.append(window)

            avance = max(1, overlap_peaks)  # avance mínimo de 1 pico
            i += avance

        all_segments.append(segments)
        all_starts.append(starts)
        all_stops.append(stops)

    return all_segments, all_starts, all_stops

#Segmentación de señales
ecg_signal_segmented, starts_ecg, stops_ecg = recortar_por_picos(ecg_signal, picos_ecg, 0)

#%% Graficación de recortes

plt.figure(figsize=(12, 6))
for k in range(10):
    plt.subplot(10, 1, k + 1)
    plt.plot(ecg_signal_segmented[1740][k], label='Señal de ECG', color='blue')
    plt.title(f'Señal de ECG segmentada {k + 1}')
    plt.xlabel('muestras')
    plt.ylabel('Amplitud')
    plt.legend()

# %%
