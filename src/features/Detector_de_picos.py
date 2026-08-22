"""
Módulo: Detector_de_picos.py
Autor: Juan Marcos Grigolatto
Descripción: Acondicionamiento de señales biomédicas (ECG y PPG) para la 
             estimación de presión arterial. Incluye filtrado digital, 
             eliminación de línea de base, detección de puntos fiduciarios 
             (picos y valles) y segmentación de pulsos en longitud fija.
"""

# --- IMPORTACIONES---
import heartpy as hp
import matplotlib.pyplot as plt
import torch
import numpy as np
from scipy.signal import argrelextrema, resample
from scipy import signal

# --- DEFINICIÓN DE FUNCIONES DE FILTRADO ---
def filtrar_ppg(senial_ppg):
    """
    Acondiciona la señal de Fotopletismografía (PPG).
    Aplica un filtro pasabanda Butterworth y elimina la deriva de la línea de base.
    """
    # Parámetros del filtro pasabanda (0.5 a 21 Hz)
    orden = 4
    frec_sup = 21
    frec_inf = 0.5

    frecs_corte = [frec_inf, frec_sup]
    b, a = signal.butter(orden, frecs_corte, 'bandpass', fs = 125)
    w, h = signal.freqz(b,a, worN=8000, fs=125)

    # Filtrado bidireccional para evitar desfase
    ppg_filtrada = signal.filtfilt(b,a, senial_ppg)

    # Estimación de la línea de base mediante media móvil (ventana de 1 seg)
    baseline = np.convolve(ppg_filtrada, np.ones(125)/125, mode='same')

    # Remoción de la línea de base
    ppg_filtrada = ppg_filtrada - baseline 
    
    return ppg_filtrada

def filtrar_ecg(senial_ecg):
    """
    Acondiciona la señal de Electrocardiograma (ECG).
    Aplica un filtro pasabanda Butterworth y elimina la deriva de la línea de base.
    """
    # Parámetros del filtro pasabanda adaptado para QRS (0.5 a 40 Hz)
    orden = 4
    frec_sup = 40
    frec_inf = 0.5

    frecs_corte = [frec_inf, frec_sup]
    b, a = signal.butter(orden, frecs_corte, 'bandpass', fs = 125)
    w, h = signal.freqz(b,a, worN=8000, fs=125)

    # Filtrado bidireccional 
    ecg_filtrada = signal.filtfilt(b,a, senial_ecg)

    # Estimación y remoción de la línea de base
    baseline = np.convolve(ecg_filtrada, np.ones(125)/125, mode='same')
    ecg_filtrada = ecg_filtrada - baseline

    return ecg_filtrada

# --- DEFINICIÓN DE FUNCIONES DE CONTROL DE DETECCIÓN DE PUNTOS FIDUCIARIOS ---
def filtrar_picos_por_distancia(peaks, min_distance = 40):
    """
    Filtra detecciones múltiples o espurias asegurando una distancia mínima 
    (en muestras) entre picos consecutivos.
    """
    picos_filtrados = [peaks[0]]
    
    for p in peaks[1:]:
        if p - picos_filtrados[-1] >= min_distance:
            picos_filtrados.append(p)
    
    return picos_filtrados

def filtrar_picos_en_extremos(signal, peaks, blanking=100):
    """
    Descarta los picos detectados demasiado cerca de los bordes del array
    para evitar la extracción de pulsos incompletos.
    """
    picos_filtrados = []
    for pico in peaks:
        if pico >= blanking and pico <= len(signal) - blanking:
            picos_filtrados.append(pico)
    return picos_filtrados

# --- DEFINICIÓN DE FUNCIONES DE SEGMENTACIÓN ---

def extraer_pulso_ecg_fixed_length(signal, fs=125, ancho_pulso=100):
    """
    Detecta ondas R en el ECG y extrae segmentos de longitud fija alrededor 
    del complejo QRS basados en el intervalo RR dinámico.
    """
    wd, m = hp.process(signal, fs)
    peaks = wd['peaklist']
    print(wd.keys())

    # Cálculos de frecuencia cardíaca e intervalos RR
    intervalos_rr = wd['RR_list'] 
    frec_cardiaca = m['bpm'] / 60  
    intervalos_rr = intervalos_rr / 1000  # Conversión a segundos
    intervalos_rr = intervalos_rr * fs  # Conversión a muestras

    print(f"Intervalos RR: {intervalos_rr}")
    print(f"Frecuencia cardiaca: {frec_cardiaca:.2f} Hz")
    print(f"Picos detectados ECG: {len(peaks)}")

    if len(peaks) < 2:
        return None

    segmentos = []
    foots_ecg = []
    # Recorte de la señal latido a latido
    for i in range(len(peaks)-1):
        # Se ignora el primer y último pico
        if i > 0 and i < len(peaks):
            # Ventana dinámica basada en el RR actual
            pre_qrs=int(0.25 * intervalos_rr[i])   # Captura onda P
            post_qrs=int(0.45 * intervalos_rr[i])  # Captura onda T

            inicio = max(0, peaks[i] - pre_qrs)
            fin = min(len(signal), peaks[i] + post_qrs)
            
            foots_ecg.append(inicio)
            foots_ecg.append(fin)
            onda_ecg = signal[inicio:fin]

            # Estandarización de longitud (truncamiento o zero-padding)
            largo_actual = len(onda_ecg)
            if largo_actual >= ancho_pulso:
                onda_seg = onda_ecg[:ancho_pulso]
            
            else:
                onda_seg = np.zeros(ancho_pulso)
                onda_seg[:largo_actual] = onda_ecg

            segmentos.append(onda_seg)

    return segmentos, peaks, foots_ecg

    
def extraer_pulso_ppg_fixed_length(signal, fs=125, ancho_pulso=500, num_picos=5):
    """
    Detecta picos sistólicos y valles (foots) en la señal PPG, y extrae 
    pulsos completos (valle a valle) estandarizados a una longitud fija.
    """
    try:
        # Detección de picos en señal original e invertida (para foots)
        wd, m = hp.process(signal, fs)
        wd_f, _ = hp.process(-signal, fs)  
        peaks = wd['peaklist']
        foots = wd_f['peaklist']
    except:
        return None  
    
    if len(peaks) < 2:
        return None

    # Eliminación de detecciones muy cercanas
    peaks = filtrar_picos_por_distancia(peaks)
    foots = filtrar_picos_por_distancia(foots) 

    # Verificación de pulsos mínimos para segmentar
    if len(foots) < 2:
        return None
    blanking = 100
    pulsos_segmented = []

    # Segmentación valle a valle
    for i in range(len(foots)-1):
        inicio, fin = foots[i], foots[i+1]
        pulso= signal[inicio:fin]

        # Estandarización de longitud (truncamiento o zero-padding)
        largo_actual = len(pulso)
        if largo_actual >= ancho_pulso:
            pulso_seg = pulso[:ancho_pulso]
        else:
            pulso_seg = np.zeros(ancho_pulso)
            pulso_seg[:largo_actual] = pulso

        pulsos_segmented.append(pulso_seg)
    
    return pulsos_segmented, peaks, foots



if __name__ == "__main__":
    # --- CARGA DE DATOS Y EJECUCIÓN PRINCIPAL ---

    # Carga de tensores preprocesados (Dataset UCI)
    archivo = 'data/processed/data_UCI/dataset_parte_2.pt'
    data = torch.load(archivo)

    # Extracción de canales PPG y ECG
    ppg_muestras = data['data'][:, 0, :]  
    ecg_muestras = data['data'][:, 1, :]  

    # Selección de muestra de prueba
    i=0

    # Ejecución del pipeline de filtrado
    ppg_filtrada = filtrar_ppg(ppg_muestras[i].numpy())
    ecg_filtrada = filtrar_ecg(ecg_muestras[i].numpy())

    # Extracción y segmentación
    pulsos_segmented_ppg, peaks_ppg, foots_ppg = extraer_pulso_ppg_fixed_length(ppg_filtrada)
    pulsos_segmented_ecg, peaks_ecg, foots_ecg= extraer_pulso_ecg_fixed_length(ecg_filtrada) 

    # --- RESULTADOS Y VISUALIZACIÓN ---
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


