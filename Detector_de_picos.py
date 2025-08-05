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
def filtrar_ppg(senial_ppg):

    orden = 4
    frec_sup = 21
    frec_inf = 0.5

    frecs_corte = [frec_inf, frec_sup]
    b, a = signal.butter(orden, frecs_corte, 'bandpass', fs = 125)

    w, h = signal.freqz(b,a, worN=8000, fs=125)

    ppg_filtrada = signal.filtfilt(b,a, senial_ppg)

    """
    plt.subplot(2, 1, 2)
    plt.plot(ppg_filtrada, label='Señal PPG', color='blue')
    plt.xlabel('Muestras')
    plt.ylabel('Amplitud')
    plt.title('Señal filtrada')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    """
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

    w, h = signal.freqz(b,a, worN=8000, fs=125)

    ecg_filtrada = signal.filtfilt(b,a, senial_ecg)

    # Línea de base estimada con media móvil
    baseline = np.convolve(ecg_filtrada, np.ones(125)/125, mode='same')

    # Señal sin línea de base
    ecg_filtrada = ecg_filtrada - baseline

    return ecg_filtrada

def filtrar_picos_por_distancia(peaks, min_distance = 40):

    picos_filtrados = [peaks[0]]
    
    for p in peaks[1:]:
        if p - picos_filtrados[-1] >= min_distance:
            picos_filtrados.append(p)
    
    return picos_filtrados

def extraer_pulso_ecg_fixed_length(signal, fs=125, ancho_pulso=100):
        wd, m = hp.process(signal, fs)
        peaks = wd['peaklist']
        print(wd.keys())

        intervalos_rr = wd['RR_list']  # Lista de intervalos RR en milisegundos

        frec_cardiaca = m['bpm'] / 60  # Frecuencia cardiaca en Hz

        intervalos_rr = intervalos_rr / 1000  # Convertir a segundos
        intervalos_rr = intervalos_rr * fs  # Convertir a muestras

        print(f"Intervalos RR: {intervalos_rr}")
        print(f"Frecuencia cardiaca: {frec_cardiaca:.2f} Hz")
        print(f"Picos detectados ECG: {len(peaks)}")

        if len(peaks) < 2:
            return None

        #Recorte de la señal
        segmentos = []
        foots_ecg = []
        for i in range(len(peaks)-1):
            
            #ignora el primer y ultimo pico
            if i>0 and i < len(peaks):
                pre_qrs=int(0.25 * intervalos_rr[i])   # para incluir onda P
                post_qrs=int(0.45 * intervalos_rr[i])  # para incluir onda T

                inicio = max(0, peaks[i] - pre_qrs)
                fin = min(len(signal), peaks[i] + post_qrs)
            

                foots_ecg.append(inicio)
                foots_ecg.append(fin)
                onda_ecg = signal[inicio:fin]

                largo_actual = len(onda_ecg)
                if largo_actual >= ancho_pulso:
                    onda_seg = onda_ecg[:ancho_pulso]
            
                else:
                    onda_seg = np.zeros(ancho_pulso)
                    onda_seg[:largo_actual] = onda_ecg

                segmentos.append(onda_seg)

        return segmentos, peaks, foots_ecg

    
def extraer_pulso_ppg_fixed_length(signal, fs=125, ancho_pulso=100):
    # Procesar con heartpy para obtener picos sistólicos
   
    try:
        wd, m = hp.process(signal, fs)
        wd_f, _ = hp.process(-signal, fs)  # Señal invertida para detectar foots
        #bpm = m['bpm']
        peaks = wd['peaklist']
        foots = wd_f['peaklist']
    except:
        return None  
    
    if len(peaks) < 2:
        return None
    """"
    if bpm > 0: 
        rr_s = 60 / bpm
    else: 
        rr_s=0.8 

    window_pre = int(0.4 * fs *rr_s)
    foots = []
    """
    peaks = filtrar_picos_por_distancia(peaks)
    foots = filtrar_picos_por_distancia(foots) 
    """
    # Detección de foots respecto a los picos
    for pico in peaks:
        # max evita que sea negativo el valor
        ini = max(0, pico - window_pre)
        sub = signal[ini:pico]
        if len(sub) > 0:
            foots.append(ini + np.argmin(sub))
    """
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
ecg_muestras = data['data'][:, 1, :]  # Todas las muestras, canal ECG

i=10001

#ppg_muestras_ruido = white_noise_torch(ppg_muestras[i], 10)
ppg_filtrada = filtrar_ppg(ppg_muestras[i].numpy())
ecg_filtrada = filtrar_ecg(ecg_muestras[i].numpy())
pulsos_segmented_ppg, peaks_ppg, foots_ppg = extraer_pulso_ppg_fixed_length(ppg_filtrada)
pulsos_segmented_ecg, peaks_ecg, foots_ecg= extraer_pulso_ecg_fixed_length(ecg_filtrada) 

if pulsos_segmented_ppg is not None:
    plt.figure(figsize=(12, 5))
    plt.plot(ppg_filtrada, label='Señal PPG', color='blue')
    plt.plot(peaks_ppg, ppg_filtrada[peaks_ppg], 'ro', label='Picos')
    plt.plot(foots_ppg, ppg_filtrada[foots_ppg], 'go', label='Foots')
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
    print(f"Se detectaron {len(pulsos_segmented_ppg)} pulsos en la señal.")
    print(f"Picos detectados: {len(peaks_ppg)}, Foots detectados: {len(foots_ppg)}")
else:
    print("Todo lo que pudo fallar lo hizo")


if pulsos_segmented_ecg is not None:
    plt.figure(figsize=(12, 5))
    plt.plot(ecg_filtrada, label='Señal ECG', color='blue')
    plt.plot(peaks_ecg, ecg_filtrada[peaks_ecg], 'ro', label='Picos')
    plt.plot(foots_ecg, ecg_filtrada[foots_ecg], 'go', label='Foots')
    #plt.plot(ppg_muestras_ruido.numpy(), label='Señal PPG', color='blue')
    #plt.plot(peaks, ppg_muestras_ruido[peaks].numpy(), 'ro', label='Picos')
    #plt.plot(foots, ppg_muestras_ruido[foots].numpy(), 'go', label='Foots')
    plt.xlabel('Muestras')
    plt.ylabel('Amplitud')
    plt.title('Detección de Picos y Foots en Señal de ecg')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    print(f"Se detectaron {len(pulsos_segmented_ecg)} pulsos en la señal.")
    print(f"Picos detectados: {len(peaks_ecg)}, Foots detectados: {len(foots_ecg)}")
else:
    print("Todo lo que pudo fallar lo hizo")


