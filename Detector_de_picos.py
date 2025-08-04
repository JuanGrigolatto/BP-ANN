import heartpy as hp
import matplotlib.pyplot as plt
import torch
import numpy as np
from scipy.signal import argrelextrema, resample
from prueba_seniales_ruido import white_noise_torch
from scipy import signal
"""
archivo = 'data_UCI/dataset_parte_1.pt'
seniales = []
data = torch.load(archivo)
seniales = data['data']
print(seniales.shape)

ppg = seniales[:, 0, :250].numpy()
ecg = seniales[:, 1, :250].numpy() 


plt.figure(figsize=(12,4))
plt.plot(ppg[0], label='PPG', color='blue')
plt.show()

plt.figure(figsize=(12,4))
plt.plot(ecg[0], label='ECG', color='red')
plt.show()


# Detectar picos en la señal 
wd, m = hp.process(ppg[0], sample_rate = 125.0)

plt.figure(figsize=(12,4))

hp.plotter(wd, m)

plt.show()

for measure in m.keys():
    print('%s: %f' %(measure, m[measure]))
"""


def filtrar_picos_por_distancia(peaks, min_distance = 40):

    picos_filtrados = [peaks[0]]
    
    for p in peaks[1:]:
        if p - picos_filtrados[-1] >= min_distance:
            picos_filtrados.append(p)
    
    return picos_filtrados

def extraer_pulso_fixed_length(signal, fs=125, ancho_pulso=100):
    # Procesar con heartpy para obtener picos sistólicos
   
    try:
        wd, m = hp.process(signal, fs)
        bpm = m['bpm']
        print(m.keys())
        peaks = wd['peaklist']
    except:
        return None  
    
    if len(peaks) < 2:
        return None
    
    if bpm > 0: 
        rr_s = 60 / bpm
    else: 
        rr_s=0.8 

    window_pre = int(0.4 * fs *rr_s)
    foots = []

    peaks = filtrar_picos_por_distancia(peaks) 
    # Detección de foots respecto a los picos
    for pico in peaks:
        # max evita que sea negativo el valor
        ini = max(0, pico - window_pre)
        sub = signal[ini:pico]
        if len(sub) > 0:
            foots.append(ini + np.argmin(sub))
    
    # Verifica que haya al menos un pulso completo
    if len(foots) < 2:
        return None

    pulsos_segmented = []
    
    for i in range(len(foots)-1):
        inicio, fin = foots[i], foots[i+1]
        pulso= signal[inicio:fin]

        # Padding o recorte
        largo_actual = len(pulso)
        if largo_actual >= ancho_pulso:
            pulso_seg = pulso[:ancho_pulso]
        else:
            pulso_seg = np.zeros(ancho_pulso)
            pulso_seg[:largo_actual] = pulso

        pulsos_segmented.append(pulso_seg)

    return pulsos_segmented, peaks, foots

# === Cargar una muestra de tu dataset
archivo = 'data_UCI/dataset_parte_2.pt'
data = torch.load(archivo)
ppg_muestras = data['data'][:, 0, :]  # Todas las muestras, canal PPG

i=1047

#ppg_muestras_ruido = white_noise_torch(ppg_muestras[i], 10)

pulsos_segmented, peaks, foots = extraer_pulso_fixed_length(ppg_muestras[i].numpy())

if pulsos_segmented is not None:
    plt.figure(figsize=(12, 5))
    plt.plot(ppg_muestras[i].numpy(), label='Señal PPG', color='blue')
    plt.plot(peaks, ppg_muestras[i][peaks].numpy(), 'ro', label='Picos')
    plt.plot(foots, ppg_muestras[i][foots].numpy(), 'go', label='Foots')
    #plt.plot(ppg_muestras_ruido.numpy(), label='Señal PPG', color='blue')
    #plt.plot(peaks, ppg_muestras_ruido[peaks].numpy(), 'ro', label='Picos')
    #plt.plot(foots, ppg_muestras_ruido[foots].numpy(), 'go', label='Foots')
    plt.xlabel('Muestras')
    plt.ylabel('Amplitud')
    plt.title('Detección de Picos y Foots en Señal de ppg')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    """
    for j in range(len(pulsos_segmented)):
        plt.figure(figsize=(12, 5))
        plt.plot(pulsos_segmented[j], label=f'pulso {j} de muestra {i}', color='blue')
        plt.xlabel('Muestras')
        plt.ylabel('Amplitud')
        plt.title(f'pulso {j} de muestra {i}')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()
    """
else:
    print("Todo lo que pudo fallar lo hizo")

# filtrado 
orden = 4
frec_sup = 21
frec_inf = 0.5

frecs_corte = [frec_inf, frec_sup]
b, a = signal.butter(orden, frecs_corte, 'bandpass', fs = 125)

w, h = signal.freqz(b,a, worN=8000, fs=125)

plt.semilogx(w, 20 * np.log10(abs(h)))
plt.title('Butterworth filter frequency response')
plt.xlabel('Frequency [rad/s]')
plt.ylabel('Amplitude [dB]')
plt.margins(0, 0.1)
plt.grid(which='both', axis='both')
plt.axvline(frec_inf, color='green') # cutoff inf frequency
plt.axvline(frec_sup, color='green') # cutoff sup frequency
plt.show()

ppg_filtrada = signal.filtfilt(b,a, ppg_muestras[i].numpy())

plt.figure(1)
plt.subplot(2, 1, 1)
plt.plot(ppg_muestras[i].numpy(), label='Señal PPG', color='blue')
plt.plot(ppg_filtrada, label='Señal PPG', color='red')
plt.xlabel('Muestras')
plt.ylabel('Amplitud')
plt.title('Señal sin filtrar')
plt.legend()
plt.grid(True)
plt.tight_layout()

#sos = signal.butter(orden, frecs_corte, 'bandpass', fs = 125, output='sos')

plt.subplot(2, 1, 2)
plt.plot(ppg_filtrada, label='Señal PPG', color='blue')
plt.xlabel('Muestras')
plt.ylabel('Amplitud')
plt.title('Señal filtrada')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


