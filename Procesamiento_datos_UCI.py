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
import random
from scipy.stats import skew
from scipy.stats import kurtosis
from scipy.stats import entropy

#%% Definición de funciones
"""
PPG_MAX=4.00
PPG_MIN=0.00
ECG_MAX=7.88
ECG_MIN=-7.50
ABP_MAX = 199.99
ABP_MIN= 50.00
"""
SBP_MEAN = 134.02
DBP_MEAN = 63.47
SBP_STD = 22.75
DBP_STD = 23.69
def split_into_windows(signal, fs, t_window, overlap):
    window_size = fs * t_window  # Tamaño de la ventana en muestras
    step = int(window_size * (1 - overlap))  # Paso entre ventanas
    windows = []
    for i in range(0, len(signal) - window_size + 1, step):
        windows.append(signal[i:i + window_size])
    return np.array(windows)

def min_max_normalization(signal, max, min):
    signal = np.array(signal)
    return (signal - min) / (max - min)

def z_score_normalization(signal, epsilon=1e-8):
    signal = np.array(signal)
    mean = np.mean(signal)
    std = np.std(signal)
    return (signal - mean) / (std + epsilon)

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

def signal_normalization(ppg_signals, abp_signals, ecg_signals):
    ppg_normalized = []
    abp_normalized = []
    ecg_normalized = []

    for i in range(len(ppg_signals)):  # Recorre pacientes
        ppg_paciente = []
        abp_paciente = []
        ecg_paciente = []

        for j in range(len(ppg_signals[i])):  # Recorre ventanas del paciente i
            ppg = ppg_signals[i][j]
            abp = abp_signals[i][j]
            ecg = ecg_signals[i][j]

            #ppg_n = min_max_normalization(ppg, np.max(ppg), np.min(ppg))
            #abp_n = min_max_normalization(abp, np.max(abp), np.min(abp))
            #ecg_n = min_max_normalization(ecg, np.max(ecg), np.min(ecg))

            ppg_n = z_score_normalization(ppg)
            abp_n = z_score_normalization(abp)
            ecg_n = z_score_normalization(ecg)

            ppg_paciente.append(ppg_n)
            abp_paciente.append(abp_n)
            ecg_paciente.append(ecg_n)

        ppg_normalized.append(ppg_paciente)
        abp_normalized.append(abp_paciente)
        ecg_normalized.append(ecg_paciente)

    return ppg_normalized, abp_normalized, ecg_normalized

#def pressure_normalization(sbp, dbp, sbp_min, sbp_max, dbp_min, dbp_max):
#    sbp_norm = (sbp - sbp_min) / (sbp_max - sbp_min)
#    dbp_norm = (dbp - dbp_min) / (dbp_max - dbp_min)
#    return sbp_norm, dbp_norm

def pressure_normalization_z_score(sbp, dbp, sbp_mean, sbp_std, dbp_mean, dbp_std):
    sbp_norm = (sbp - sbp_mean) / sbp_std
    dbp_norm = (dbp - dbp_mean) / dbp_std
    return sbp_norm, dbp_norm

#def pressure_desnormalization(sbp_norm, dbp_norm, sbp_min, sbp_max, dbp_min, dbp_max):
#    sbp = sbp_norm * (sbp_max - sbp_min) + sbp_min
#    dbp = dbp_norm * (dbp_max - dbp_min) + dbp_min
#    return sbp, dbp

def max_min_pressures(sbp,dbp):
    
    SBP_MIN = np.min(sbp)
    SBP_MAX = np.max(sbp)
    DBP_MIN = np.min(dbp)
    DBP_MAX = np.max(dbp)

    print(f"SBP_MIN={SBP_MIN:.2f}, SBP_MAX={SBP_MAX:.2f}")
    print(f"DBP_MIN={DBP_MIN:.2f}, DBP_MAX={DBP_MAX:.2f}")
    
    return SBP_MIN, SBP_MAX, DBP_MIN, DBP_MAX 

def signal_stats_analisis(signal):
     
    skewness_value = skew(signal)
    kurtosis_value = kurtosis(signal)
    #entropy_value = entropy(signal)
     
    #Entropía de Shannon a partir de histograma
    
    hist, _ = np.histogram(signal, bins=50, density=True)
    p = hist + 1e-12
    p = p / np.sum(p)
    entropy_value = entropy(p)
    

    #return skewness_value, kurtosis_value, entropy_value
    return skewness_value, kurtosis_value, entropy_value

def white_noise(signal, snr_db=20):
    # Potencia de la señal
    potencia_senal = np.mean(senal ** 2)

    # Convertir SNR de dB a escala lineal
    snr_lineal = 10 ** (snr_db / 10)

    # Calcular la potencia del ruido deseada
    potencia_ruido = potencia_senal / snr_lineal

    # Generar ruido blanco gaussiano
    ruido = np.random.normal(0, np.sqrt(potencia_ruido), senal.shape)

    senal_con_ruido = senal + ruido

    return senal_con_ruido


def get_abp_labels(abp_signals):
    matriz_picos_sistolicos = []
    matriz_picos_diastolicos = []
    matriz_presiones_sistolicas = []
    matriz_presiones_diastolicas = []

    for i in range(len(abp_signals)):
        picos_sistolicos = []
        picos_diastolicos = []
        presiones_sistolicas = []
        presiones_diastolicas = []

        for j in range(len(abp_signals[i])):
            señal = abp_signals[i][j]
            max_val = np.max(señal)
            min_val = np.min(señal)
            rango = max_val - min_val

            prominence = 0.2 * rango
            height_sbp = min_val + 0.6 * rango
            height_dbp = min_val + 0.2 * rango
            distancia_min = int(0.3 * fs)

            # Detectar picos
            peakss, _ = find_peaks(señal, height=height_sbp, prominence=prominence, distance=distancia_min)
            peaksd, _ = find_peaks(-señal, height=-height_dbp, prominence=prominence, distance=distancia_min)

            picos_sistolicos.append(peakss)
            picos_diastolicos.append(peaksd)

    
            if len(peakss) > 0 and len(peaksd) > 0:
                ps_val = np.max(señal[peakss])
                pd_val = np.min(señal[peaksd])

                if ps_val > pd_val:  # Solo si tiene sentido fisiológico
                    ps = ps_val
                    pd = pd_val
                else:
                    print(f"Señal inválida (SBP ≤ DBP) en paciente {i}, ventana {j}: SBP={ps_val:.2f}, DBP={pd_val:.2f}")
                    ps = np.nan
                    pd = np.nan
            else:
                # Demasiado pocos picos, probablemente ruido
                ps = np.nan
                pd = np.nan

            presiones_sistolicas.append(ps)
            presiones_diastolicas.append(pd)

        matriz_presiones_sistolicas.append(presiones_sistolicas)
        matriz_presiones_diastolicas.append(presiones_diastolicas)
        matriz_picos_sistolicos.append(picos_sistolicos)
        matriz_picos_diastolicos.append(picos_diastolicos)

    return (
        matriz_presiones_sistolicas,
        matriz_presiones_diastolicas,
        matriz_picos_sistolicos,
        matriz_picos_diastolicos,
    )

def labels_normalization(matriz_presiones_sistolicas, matriz_presiones_diastolicas, SBP_MEAN, SBP_STD, DBP_MEAN, DBP_STD):

    # Aplanar y convertir a arrays de numpy
    """
    todas_sbp = np.array([x for sublista in matriz_presiones_sistolicas for x in sublista if not np.isnan(x)])
    todas_dbp = np.array([x for sublista in matriz_presiones_diastolicas for x in sublista if not np.isnan(x)])

    SBP_MIN = np.min(todas_sbp)
    SBP_MAX = np.max(todas_sbp)
    DBP_MIN = np.min(todas_dbp)
    DBP_MAX = np.max(todas_dbp)
    SBP_MEAN = np.mean(todas_sbp)
    DBP_MEAN = np.mean(todas_dbp)
    SBP_STD = np.std(todas_sbp)
    DBP_STD = np.std(todas_dbp)


    print(f"SBP_MIN={SBP_MIN:.2f}, SBP_MAX={SBP_MAX:.2f}")
    print(f"DBP_MIN={DBP_MIN:.2f}, DBP_MAX={DBP_MAX:.2f}")
    print(f"SBP_MEAN={SBP_MEAN:.2f}, SBP_STD={SBP_STD:.2f}")
    print(f"DBP_MEAN={DBP_MEAN:.2f}, DBP_STD={DBP_STD:.2f}")
    """
    matriz_presiones_sistolicas_norm=[]
    matriz_presiones_diastolicas_norm=[]
    for i in range(len(matriz_presiones_sistolicas)):
        list_sbp=[]
        list_dbp=[]  
        for j in range(len(matriz_presiones_sistolicas[i])):

            #ps_norm,pd_norm=pressure_normalization(matriz_presiones_sistolicas[i][j], matriz_presiones_diastolicas[i][j],
            #                         SBP_MIN, SBP_MAX,DBP_MIN,DBP_MAX)
            
            ps_norm,pd_norm=pressure_normalization_z_score(matriz_presiones_sistolicas[i][j], matriz_presiones_diastolicas[i][j],
                                                            SBP_MEAN, SBP_STD, DBP_MEAN, DBP_STD)
            
            if not np.isnan(ps_norm) and not np.isnan(pd_norm):
                list_sbp.append(ps_norm)
                list_dbp.append(pd_norm)
            else:
                print(f"Etiqueta inválida normalizada: SBP={ps_norm}, DBP={pd_norm}")    
        matriz_presiones_sistolicas_norm.append(list_sbp)
        matriz_presiones_diastolicas_norm.append(list_dbp)
    return matriz_presiones_sistolicas_norm, matriz_presiones_diastolicas_norm

def delete_signals_no_peaks(ppg, abp, ecg, presiones_sbp, presiones_dbp, indices_sistolicos, indices_diastolicos):
    ppg_filtradas = []
    abp_filtradas = []
    ecg_filtradas = []
    sbp_filtradas = []
    dbp_filtradas = []
    sistolicos_filtrados = []
    diastolicos_filtrados = []

    total_eliminadas_nopicos = 0

    for i in range(len(ppg)):
        ppg_paciente = []
        abp_paciente = []
        ecg_paciente = []
        sbp_paciente = []
        dbp_paciente = []
        sist_paciente = []
        diast_paciente = []

        for j in range(len(ppg[i])):
            sbp = presiones_sbp[i][j]
            dbp = presiones_dbp[i][j]

            if not (np.isnan(sbp) or np.isnan(dbp) or sbp > 200 or sbp < 70 or 
                   dbp > 140 or dbp < 50):
                ppg_paciente.append(ppg[i][j])
                abp_paciente.append(abp[i][j])
                ecg_paciente.append(ecg[i][j])
                sbp_paciente.append(sbp)
                dbp_paciente.append(dbp)
                sist_paciente.append(indices_sistolicos[i][j])
                diast_paciente.append(indices_diastolicos[i][j])
            else:
                total_eliminadas_nopicos += 1
               
        if len(ppg_paciente) > 0:
            ppg_filtradas.append(ppg_paciente)
            abp_filtradas.append(abp_paciente)
            ecg_filtradas.append(ecg_paciente)
            sbp_filtradas.append(sbp_paciente)
            dbp_filtradas.append(dbp_paciente)
            sistolicos_filtrados.append(sist_paciente)
            diastolicos_filtrados.append(diast_paciente)

    print(f"Ventanas eliminadas por no tener picos detectados: {total_eliminadas_nopicos}")
    return (
        ppg_filtradas,
        abp_filtradas,
        ecg_filtradas,
        sbp_filtradas,
        dbp_filtradas,
        sistolicos_filtrados,
        diastolicos_filtrados
    )



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

window_duration = 10
"""
all_ppg_values = np.concatenate(ppg_signal)
all_ecg_values = np.concatenate(ecg_signal)
all_abp_values = np.concatenate(abp_signal)

PPG_MIN, PPG_MAX = np.min(all_ppg_values), np.max(all_ppg_values)
ECG_MIN, ECG_MAX = np.min(all_ecg_values), np.max(all_ecg_values)
ABP_MIN, ABP_MAX = np.min(all_abp_values), np.max(all_abp_values)

print(f"PPG: min={PPG_MIN:.2f}, max={PPG_MAX:.2f}")
print(f"ECG: min={ECG_MIN:.2f}, max={ECG_MAX:.2f}")
print(f"ABP: min={ABP_MIN:.2f}, max={ABP_MAX:.2f}")
"""
ppg_segmented, abp_segmented, ecg_segmented= signal_segmentation(fs, window_duration, ppg_signal, 
                                                                 abp_signal, ecg_signal)


presiones_sistolicas, presiones_diastolicas, indices_sistolicos, indices_diastolicos = get_abp_labels(abp_segmented)

ppg_depurada, abp_depurada, ecg_depurada, sbp_depurada, dbp_depurada, indices_sist_dep, indices_dias_dep = delete_signals_no_peaks(ppg_segmented, abp_segmented, ecg_segmented, presiones_sistolicas, 
                                                                                                    presiones_diastolicas, indices_sistolicos, indices_diastolicos)

todas_sbp1 = np.array([x for sublista in presiones_sistolicas for x in sublista if not np.isnan(x)])
todas_dbp1 = np.array([x for sublista in presiones_diastolicas for x in sublista if not np.isnan(x)])

SBP_MEAN1 = np.mean(todas_sbp1)
DBP_MEAN1 = np.mean(todas_dbp1)
SBP_STD1 = np.std(todas_sbp1)
DBP_STD1 = np.std(todas_dbp1)

#presiones_sistolicas_norm, presiones_diastolicas_norm= labels_normalization(sbp_depurada, dbp_depurada)

ppg_normalized, abp_normalized, ecg_normalized= signal_normalization(ppg_depurada, abp_depurada, ecg_depurada)
#%%
del ppg_segmented, abp_segmented, ecg_segmented
del presiones_sistolicas, presiones_diastolicas, indices_sistolicos, indices_diastolicos
del ppg_depurada, abp_depurada, ecg_depurada, sbp_depurada, dbp_depurada, indices_sist_dep, indices_dias_dep
del todas_sbp1, todas_dbp1
del ppg_normalized, abp_normalized, ecg_normalized
#%%
num_muestras=10
for i in range(num_muestras):

    ind_diastolicos=[]
    ind_sistolicos=[]

    paciente_id = random.randint(0, 2000)
    ventana_id = random.randint(0, len(abp_normalized[paciente_id])-1 )
    senal_abp = abp_normalized[paciente_id][ventana_id]

    # Extraé los índices de picos sistólicos y diastólicos ya calculados
    ind_sistolicos = indices_sist_dep[paciente_id][ventana_id]
    ind_diastolicos= indices_dias_dep[paciente_id][ventana_id]

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
#save_partial_file(ppg_normalized, ecg_normalized, presiones_sistolicas_norm, presiones_diastolicas_norm, 0, 'dataset_parte_1')
#%%
"""
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
"""
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

window_duration=10
"""
all_ppg_values2 = np.concatenate(ppg_signal2)
all_ecg_values2 = np.concatenate(ecg_signal2)
all_abp_values2 = np.concatenate(abp_signal2)

PPG_MIN, PPG_MAX = np.min(all_ppg_values2), np.max(all_ppg_values2)
ECG_MIN, ECG_MAX = np.min(all_ecg_values2), np.max(all_ecg_values2)
ABP_MIN, ABP_MAX = np.min(all_abp_values2), np.max(all_abp_values2)

print(f"PPG2: min={PPG_MIN:.2f}, max={PPG_MAX:.2f}")
print(f"ECG2: min={ECG_MIN:.2f}, max={ECG_MAX:.2f}")
print(f"ABP2: min={ABP_MIN:.2f}, max={ABP_MAX:.2f}")
"""
ppg_segmented2, abp_segmented2, ecg_segmented2= signal_segmentation(fs, window_duration, ppg_signal2, 
                                                                 abp_signal2, ecg_signal2)


presiones_sistolicas2, presiones_diastolicas2, indices_sistolicos2, indices_diastolicos2 = get_abp_labels(abp_segmented2)

ppg_depurada2, abp_depurada2, ecg_depurada2, sbp_depurada2, dbp_depurada2, indices_sist_dep2, indices_dias_dep2 = delete_signals_no_peaks(ppg_segmented2, abp_segmented2, ecg_segmented2, presiones_sistolicas2, 
                                                                                                    presiones_diastolicas2, indices_sistolicos2, indices_diastolicos2)
todas_sbp2 = np.array([x for sublista in presiones_sistolicas2 for x in sublista if not np.isnan(x)])
todas_dbp2 = np.array([x for sublista in presiones_diastolicas2 for x in sublista if not np.isnan(x)])

SBP_MEAN2 = np.mean(todas_sbp2)
DBP_MEAN2 = np.mean(todas_dbp2)
SBP_STD2 = np.std(todas_sbp2)
DBP_STD2 = np.std(todas_dbp2)

#presiones_sistolicas_norm2, presiones_diastolicas_norm2= labels_normalization(sbp_depurada2, dbp_depurada2)

ppg_normalized2, abp_normalized2, ecg_normalized2= signal_normalization(ppg_depurada2, abp_depurada2, ecg_depurada2)

#%%
del ppg_segmented2, abp_segmented2, ecg_segmented2
del presiones_sistolicas2, presiones_diastolicas2, indices_sistolicos2, indices_diastolicos2
del ppg_depurada2, abp_depurada2, ecg_depurada2, sbp_depurada2, dbp_depurada2, indices_sist_dep2, indices_dias_dep2
del todas_sbp2, todas_dbp2
del ppg_normalized2, abp_normalized2, ecg_normalized2
#%%
"""
num_muestras=10
for i in range(num_muestras):
    ind_diastolicos2=[]
    ind_sistolicos2=[]

    paciente_id = random.randint(0, 2999)
    ventana_id = random.randint(0, len(abp_normalized2[paciente_id])-1 )
    senal_abp = abp_normalized2[paciente_id][ventana_id]

    # Extraé los índices de picos sistólicos y diastólicos ya calculados
    ind_sistolicos2 = indices_sist_dep2[paciente_id][ventana_id]
    ind_diastolicos2= indices_dias_dep2[paciente_id][ventana_id]

    # Graficá
    plt.figure(figsize=(10, 4))
    plt.plot(senal_abp, label='Señal ABP Normalizada')
    plt.plot(ind_sistolicos2, senal_abp[ind_sistolicos2], 'ro', label='Picos Sistólicos')
    plt.plot(ind_diastolicos2, senal_abp[ind_diastolicos2], 'bo', label='Picos Diastólicos')
    plt.xlabel('Muestras')
    plt.ylabel('Amplitud')
    plt.title(f'Paciente {paciente_id} - Ventana {ventana_id}')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
"""
#%% Generación de archivo parte 2 .pt y liberación de memoria RAM
#save_partial_file(ppg_normalized2, ecg_normalized2, presiones_sistolicas_norm2, presiones_diastolicas_norm2, 3000, 'dataset_parte_2')
#%%
"""
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
"""
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

window_duration=10
"""
all_ppg_values3 = np.concatenate(ppg_signal3)
all_ecg_values3 = np.concatenate(ecg_signal3)
all_abp_values3 = np.concatenate(abp_signal3)

PPG_MIN, PPG_MAX = np.min(all_ppg_values3), np.max(all_ppg_values3)
ECG_MIN, ECG_MAX = np.min(all_ecg_values3), np.max(all_ecg_values3)
ABP_MIN, ABP_MAX = np.min(all_abp_values3), np.max(all_abp_values3)

print(f"PPG3: min={PPG_MIN:.2f}, max={PPG_MAX:.2f}")
print(f"ECG3: min={ECG_MIN:.2f}, max={ECG_MAX:.2f}")
print(f"ABP3: min={ABP_MIN:.2f}, max={ABP_MAX:.2f}")
"""
ppg_segmented3, abp_segmented3, ecg_segmented3= signal_segmentation(fs, window_duration, ppg_signal3, 
                                                                 abp_signal3, ecg_signal3)


presiones_sistolicas3, presiones_diastolicas3, indices_sistolicos3, indices_diastolicos3 = get_abp_labels(abp_segmented3)

ppg_depurada3, abp_depurada3, ecg_depurada3, sbp_depurada3, dbp_depurada3, indices_sist_dep3, indices_dias_dep3 = delete_signals_no_peaks(ppg_segmented3, abp_segmented3, ecg_segmented3, presiones_sistolicas3, 
                                                                                                    presiones_diastolicas3, indices_sistolicos3, indices_diastolicos3)

todas_sbp3 = np.array([x for sublista in presiones_sistolicas3 for x in sublista if not np.isnan(x)])
todas_dbp3 = np.array([x for sublista in presiones_diastolicas3 for x in sublista if not np.isnan(x)])

SBP_MEAN3 = np.mean(todas_sbp3)
DBP_MEAN3 = np.mean(todas_dbp3)
SBP_STD3 = np.std(todas_sbp3)
DBP_STD3 = np.std(todas_dbp3)
#presiones_sistolicas_norm3, presiones_diastolicas_norm3= labels_normalization(sbp_depurada3, dbp_depurada3)

ppg_normalized3, abp_normalized3, ecg_normalized3= signal_normalization(ppg_depurada3, abp_depurada3, ecg_depurada3)
#%%
del ppg_segmented3, abp_segmented3, ecg_segmented3
del presiones_sistolicas3, presiones_diastolicas3, indices_sistolicos3, indices_diastolicos3
del ppg_depurada3, abp_depurada3, ecg_depurada3, sbp_depurada3, dbp_depurada3, indices_sist_dep3, indices_dias_dep3
del todas_sbp3, todas_dbp3
del ppg_normalized3, abp_normalized3, ecg_normalized3
#%%
"""
num_muestras=10
for i in range(num_muestras):
    ind_diastolicos3=[]
    ind_sistolicos3=[]

    paciente_id = random.randint(0, 2999)
    ventana_id = random.randint(0, len(abp_normalized3[paciente_id])-1 )
    senal_abp = abp_normalized3[paciente_id][ventana_id]

    # Extraé los índices de picos sistólicos y diastólicos ya calculados
    ind_sistolicos3 = indices_sist_dep3[paciente_id][ventana_id]
    ind_diastolicos3= indices_dias_dep3[paciente_id][ventana_id]

    # Graficá
    plt.figure(figsize=(10, 4))
    plt.plot(senal_abp, label='Señal ABP Normalizada')
    plt.plot(ind_sistolicos3, senal_abp[ind_sistolicos3], 'ro', label='Picos Sistólicos')
    plt.plot(ind_diastolicos3, senal_abp[ind_diastolicos3], 'bo', label='Picos Diastólicos')
    plt.xlabel('Muestras')
    plt.ylabel('Amplitud')
    plt.title(f'Paciente {paciente_id} - Ventana {ventana_id}')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    """
#%% Generación de archivo parte 3 .pt y liberación de memoria RAM
#save_partial_file(ppg_normalized3, ecg_normalized3, presiones_sistolicas_norm3, presiones_diastolicas_norm3, 6000, 'dataset_parte_3')  
#%%
"""
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
"""
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

window_duration=10
"""
all_ppg_values4 = np.concatenate(ppg_signal4)
all_ecg_values4 = np.concatenate(ecg_signal4)
all_abp_values4 = np.concatenate(abp_signal4)

PPG_MIN, PPG_MAX = np.min(all_ppg_values4), np.max(all_ppg_values4)
ECG_MIN, ECG_MAX = np.min(all_ecg_values4), np.max(all_ecg_values4)
ABP_MIN, ABP_MAX = np.min(all_abp_values4), np.max(all_abp_values4)

print(f"PPG4: min={PPG_MIN:.2f}, max={PPG_MAX:.2f}")
print(f"ECG4: min={ECG_MIN:.2f}, max={ECG_MAX:.2f}")
print(f"ABP4: min={ABP_MIN:.2f}, max={ABP_MAX:.2f}")
"""
ppg_segmented4, abp_segmented4, ecg_segmented4= signal_segmentation(fs, window_duration, ppg_signal4, 
                                                                 abp_signal4, ecg_signal4)


presiones_sistolicas4, presiones_diastolicas4, indices_sistolicos4, indices_diastolicos4 = get_abp_labels(abp_segmented4)

ppg_depurada4, abp_depurada4, ecg_depurada4, sbp_depurada4, dbp_depurada4, indices_sist_dep4, indices_dias_dep4 = delete_signals_no_peaks(ppg_segmented4, abp_segmented4, ecg_segmented4, presiones_sistolicas4, 
                                                                                                    presiones_diastolicas4, indices_sistolicos4, indices_diastolicos4)

todas_sbp4 = np.array([x for sublista in presiones_sistolicas4 for x in sublista if not np.isnan(x)])
todas_dbp4 = np.array([x for sublista in presiones_diastolicas4 for x in sublista if not np.isnan(x)])

SBP_MEAN4 = np.mean(todas_sbp4)
DBP_MEAN4 = np.mean(todas_dbp4)
SBP_STD4 = np.std(todas_sbp4)
DBP_STD4 = np.std(todas_dbp4)

#presiones_sistolicas_norm4, presiones_diastolicas_norm4= labels_normalization(sbp_depurada4, dbp_depurada4)

ppg_normalized4, abp_normalized4, ecg_normalized4= signal_normalization(ppg_depurada4, abp_depurada4, ecg_depurada4)
#%%
del ppg_segmented4, abp_segmented4, ecg_segmented4
del presiones_sistolicas4, presiones_diastolicas4, indices_sistolicos4, indices_diastolicos4
del ppg_depurada4, abp_depurada4, ecg_depurada4, sbp_depurada4, dbp_depurada4, indices_sist_dep4, indices_dias_dep4
del todas_sbp4, todas_dbp4
del ppg_normalized4, abp_normalized4, ecg_normalized4
#%%
"""
num_muestras=10
for i in range(num_muestras):
    ind_diastolicos4=[]
    ind_sistolicos4=[]

    paciente_id = random.randint(0, 2000)
    ventana_id = random.randint(0, len(abp_normalized4[paciente_id])-1 )
    senal_abp = abp_normalized4[paciente_id][ventana_id]

    # Extraé los índices de picos sistólicos y diastólicos ya calculados
    ind_sistolicos4 = indices_sist_dep4[paciente_id][ventana_id]
    ind_diastolicos4= indices_dias_dep4[paciente_id][ventana_id]

    # Graficá
    plt.figure(figsize=(10, 4))
    plt.plot(senal_abp, label='Señal ABP Normalizada')
    plt.plot(ind_sistolicos4, senal_abp[ind_sistolicos4], 'ro', label='Picos Sistólicos')
    plt.plot(ind_diastolicos4, senal_abp[ind_diastolicos4], 'bo', label='Picos Diastólicos')
    plt.xlabel('Muestras')
    plt.ylabel('Amplitud')
    plt.title(f'Paciente {paciente_id} - Ventana {ventana_id}')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    """
#%% Ploteo de señales
   
plt.figure(figsize=(10, 4))
plt.plot(ppg_signal4[1000][:1000])  # Mostrar los primeros 1000 puntos
plt.title("Señal PPG - Primer Registro")
plt.xlabel("Muestras")
plt.ylabel("Amplitud")
plt.show()

plt.figure(figsize=(10, 4))
plt.plot(abp_signal4[1000][:1000])  # Mostrar los primeros 1000 puntos
plt.title("Señal ABP - Primer Registro")
plt.xlabel("Muestras")
plt.ylabel("Amplitud")
plt.show()
    
plt.figure(figsize=(10, 4))
plt.plot(ecg_signal4[1000][:1000])  # Mostrar los primeros 1000 puntos
plt.title("Señal ECG - Primer Registro")
plt.xlabel("Muestras")
plt.ylabel("Amplitud")
plt.show()
#%% Ploteo de segmentación
plt.figure(figsize=(10, 4))
plt.plot(ppg_segmented4[1000][0])  # Mostrar los primeros 1000 puntos
plt.title("Recorte de 2 seg de señal de PPG")
plt.xlabel("Muestras")
plt.ylabel("Amplitud")
plt.show()

plt.figure(figsize=(10, 4))
plt.plot(abp_segmented4[1000][0])  # Mostrar los primeros 1000 puntos
plt.title("Recorte de 5 seg de señal de ABP")
plt.xlabel("Muestras")
plt.ylabel("Amplitud")
plt.show()

plt.figure(figsize=(10, 4))
plt.plot(ecg_segmented4[1000][0])  # Mostrar los primeros 1000 puntos
plt.title("Recorte de 5 seg de señal de ECG")
plt.xlabel("Muestras")
plt.ylabel("Amplitud")
plt.show()
#%% Relación de señales en el recorte
plt.figure(figsize=(10, 4))
plt.plot(ecg_normalized[1940][0], color= "blue", label="PPG")
#plt.plot(abp_normalized4[1100][0], color= "green", label="ABP")
#plt.plot(ecg_normalized[1940][0], color= "red", label="ECG") 
plt.title("Relación entre señales en el tiempo")
plt.xlabel("Muestras")
plt.ylabel("Amplitud")
plt.show()
print(f"Min: {np.min(ppg_normalized[1800][0])}, Max: {np.max(ppg_normalized[1940][0])}")
#%%
plt.figure(figsize=(10, 4))
prueba= (ppg_normalized[1000][0])
plt.plot((prueba))  # Mostrar los primeros 1000 puntos
plt.title("Recorte de 5 seg de señal de ABP")
plt.xlabel("Muestras")
plt.ylabel("Amplitud")
plt.show()

#%%

paciente_id = 1000
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
                                                        
#%% Generación de archivo parte 4 .pt y liberación de memoria RAM
#save_partial_file(ppg_normalized4, ecg_normalized4, presiones_sistolicas_norm4, presiones_diastolicas_norm4, 9000, 'dataset_parte_4')
#%%
"""
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
"""
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
def plot_histogram_pressures(sbp, dbp, bins_sbp, bins_dbp, title):
    # Histograma SBP
    plt.subplot(1, 2, 1)
    plt.hist(sbp, bins=bins_sbp, color='crimson', edgecolor='black', alpha=0.75)
    plt.title(f"Distribución de SBP (Presión Sistólica) - {title}", fontsize=14)
    plt.xlabel("mmHg", fontsize=12)
    plt.ylabel("Frecuencia", fontsize=12)
    plt.axvline(np.mean(sbp), color='black', linestyle='--', label=f"Media: {np.mean(sbp):.1f}")
    plt.legend()

    # Histograma DBP
    plt.subplot(1, 2, 2)
    plt.hist(dbp, bins=bins_dbp, color='steelblue', edgecolor='black', alpha=0.75)
    plt.title(f"Distribución de DBP (Presión Diastólica) - {title}", fontsize=14)
    plt.xlabel("mmHg", fontsize=12)
    plt.axvline(np.mean(dbp), color='black', linestyle='--', label=f"Media: {np.mean(dbp):.1f}")
    plt.legend()

    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(14, 6))

def Histograma(SBP_MIN, SBP_MAX, DBP_MIN, DBP_MAX, ruta_dataset, title):
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

    plot_histogram_pressures(sbp, dbp, bins_sbp, bins_dbp, title)
    
def box_plot(SBP_MIN, SBP_MAX, DBP_MIN, DBP_MAX, ruta_dataset, title):
    data_dict = torch.load(ruta_dataset)
    labels = data_dict['labels']  

    labels_np = labels.numpy()
    sbp = labels_np[:, 0]
    dbp = labels_np[:, 1]

    # Desnormalización
    sbp = sbp * (SBP_MAX - SBP_MIN) + SBP_MIN
    dbp = dbp * (DBP_MAX - DBP_MIN) + DBP_MIN

    plt.figure(figsize=(10, 4))
    plt.boxplot([sbp, dbp], labels=["SBP", "DBP"])
    plt.title(f"Diagrama de bigote para presiones - {title}", fontsize=14)
    plt.legend()
    plt.tight_layout()
    plt.show()  
        

datasets_info = [
    {
        "ruta": "data_UCI/dataset_parte_1.pt",
        "SBP_MIN": 55.98, "SBP_MAX": 199.95,
        "DBP_MIN": 50.00, "DBP_MAX": 190.30
    },
    {
        "ruta": "data_UCI/dataset_parte_2.pt",
        "SBP_MIN": 61.97, "SBP_MAX": 199.95,
        "DBP_MIN": 50.00, "DBP_MAX": 191.26
    },
    {
        "ruta": "data_UCI/dataset_parte_3.pt",
        "SBP_MIN": 60.91, "SBP_MAX": 199.99,
        "DBP_MIN": 50.00, "DBP_MAX": 182.93
    },
    {
        "ruta": "data_UCI/dataset_parte_4.pt",
        "SBP_MIN": 59.84, "SBP_MAX": 199.99,
        "DBP_MIN": 50.00, "DBP_MAX": 178.92
    }
]

# Histograma de cada dataset con sus respectivos parámetros
for info in datasets_info:
    ruta = info["ruta"]
    title = os.path.basename(ruta).replace('.pt', '')
    Histograma(
        SBP_MIN=info["SBP_MIN"],
        SBP_MAX=info["SBP_MAX"],
        DBP_MIN=info["DBP_MIN"],
        DBP_MAX=info["DBP_MAX"],
        ruta_dataset=ruta,
        title=title
    )
    box_plot(SBP_MIN=info["SBP_MIN"],
        SBP_MAX=info["SBP_MAX"],
        DBP_MIN=info["DBP_MIN"],
        DBP_MAX=info["DBP_MAX"],
        ruta_dataset=ruta,
        title=title)
    
# %%
import torch
import matplotlib.pyplot as plt

# Ruta al archivo .pt
ruta = 'data_UCI/dataset_parte_1.pt'

# Cargar el dataset
data_dict = torch.load(ruta)
data = data_dict['data']           # (N, 2, long_segmento)
labels = data_dict['labels']       # (N, 2)
patient_ids = data_dict['patient_ids']  # (N,)

print(f"Dataset cargado con {data.shape[0]} muestras.")

def visualizar_batch(start_idx=0, batch_size=10):
    end_idx = min(start_idx + batch_size, data.shape[0])
    num_muestras = end_idx - start_idx

    fig, axs = plt.subplots(num_muestras, 1, figsize=(12, 2.2 * num_muestras), sharex=True)

    if num_muestras == 1:
        axs = [axs]

    for i, idx in enumerate(range(start_idx, end_idx)):
        ppg = data[idx, 0].numpy()
        ecg = data[idx, 1].numpy()
        sbp, dbp = labels[idx].tolist()
        pid = patient_ids[idx].item()

        axs[i].plot(ppg, label='PPG', color='blue', alpha=0.7)
        axs[i].plot(ecg, label='ECG', color='orange', alpha=0.7)
        axs[i].set_title(f"Muestra #{idx} - Paciente {pid} - SBP: {sbp:.3f} / DBP: {dbp:.3f} (normalizados)")
        axs[i].legend(loc='upper right')
        axs[i].grid(True)

    plt.xlabel("Muestras")
    plt.tight_layout()
    plt.show()

# Llamada de ejemplo
visualizar_batch(start_idx=14774, batch_size=10)

# %%
def recorrer_anormales(abp, sbp_labels, dbp_labels, picos_sistolicos, picos_diastolicos,
                       rango_sbp=(80, 200), rango_dbp=(40, 130)):
    """
    Recorre una por una las señales con etiquetas fuera de rango y permite inspeccionarlas visualmente.
    """
    print(" Recorriendo señales anómalas interactivamente...")

    errores = []

    for i in range(len(sbp_labels)):  # paciente
        for j in range(len(sbp_labels[i])):  # ventana
            sbp = sbp_labels[i][j]
            dbp = dbp_labels[i][j]

            if (
                sbp < rango_sbp[0] or sbp > rango_sbp[1] or
                dbp < rango_dbp[0] or dbp > rango_dbp[1] or
                sbp <= dbp
            ):
                errores.append((i, j, round(sbp, 1), round(dbp, 1)))

    print(f" Ventanas con etiquetas anormales encontradas: {len(errores)}")
    if not errores:
        return

    print(" Presioná Enter para avanzar, o 'q' + Enter para salir.\n")

    for (i, j, sbp_val, dbp_val) in errores:
        abp_signal = abp[i][j]
        sist = picos_sistolicos[i][j]
        diast = picos_diastolicos[i][j]

        plt.figure(figsize=(8, 3))
        plt.plot(abp_signal, label='ABP')
        plt.plot(sist, abp_signal[sist], 'ro', label='Picos sistólicos')
        plt.plot(diast, abp_signal[diast], 'go', label='Picos diastólicos')
        plt.title(f"Paciente {i} | Ventana {j} | SBP = {sbp_val} mmHg, DBP = {dbp_val} mmHg")
        plt.xlabel("Muestras")
        plt.ylabel("Presión (normalizada)")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

        user_input = input("➡️ Siguiente (Enter) | Salir (q + Enter): ")
        if user_input.lower() == 'q':
            break

recorrer_anormales(abp_depurada3, sbp_depurada3, dbp_depurada3, 
                   indices_sist_dep3, indices_dias_dep3)

# %%
import numpy as np
import matplotlib.pyplot as plt

def analizar_lote_senales(lista_senales, nombre="abp"):
    """Analiza skewness, kurtosis y entropía de múltiples señales usando signal_stats_analisis."""
    skewness_values = []
    kurtosis_values = []
    entropy_values = []

    for signal in lista_senales:
        skew_val, kurt_val, ent_val = signal_stats_analisis(signal)
        skewness_values.append(skew_val)
        kurtosis_values.append(kurt_val)
        entropy_values.append(ent_val)

    skewness_values = np.array(skewness_values)
    kurtosis_values = np.array(kurtosis_values)
    entropy_values = np.array(entropy_values)

    # Mostrar estadísticas
    print(f"\n--- Estadísticas globales para {nombre} ---")
    for label, valores in zip(["Skewness", "Kurtosis", "Entropía"],
                              [skewness_values, kurtosis_values, entropy_values]):
        print(f"{label}:")
        print(f"  Media         = {np.mean(valores):.3f}")
        print(f"  Desvío Std    = {np.std(valores):.3f}")
        print(f"  Percentil 5   = {np.percentile(valores, 5):.3f}")
        print(f"  Percentil 95  = {np.percentile(valores, 95):.3f}\n")

    # Graficar histogramas
    plt.figure(figsize=(15, 4))
    
    plt.subplot(1, 3, 1)
    plt.hist(skewness_values, bins=50, color='lightblue', edgecolor='black')
    plt.title(f'{nombre} - Skewness')
    plt.xlabel("Valor"); plt.ylabel("Frecuencia")

    plt.subplot(1, 3, 2)
    plt.hist(kurtosis_values, bins=50, color='lightcoral', edgecolor='black')
    plt.title(f'{nombre} - Kurtosis')
    plt.xlabel("Valor"); plt.ylabel("Frecuencia")

    plt.subplot(1, 3, 3)
    plt.hist(entropy_values, bins=50, color='lightgreen', edgecolor='black')
    plt.title(f'{nombre} - Entropía')
    plt.xlabel("Valor"); plt.ylabel("Frecuencia")

    plt.tight_layout()
    plt.show()

    return skewness_values, kurtosis_values, entropy_values

#patient_number=1234
#patient_number = 743
patient_number = 1747
skew_val, kurt_val, entropy_val = analizar_lote_senales(abp_normalized4[patient_number])
# %%
#paciente_id = 1234
#paciente_id = 743
paciente_id = 1747
ventana_id = 0

senal= abp_depurada4[paciente_id]

for i in range(len(abp_depurada[paciente_id])):
    
    skew_val, kurt_val, entropy_val = signal_stats_analisis (abp_depurada[paciente_id][i])
    print(f"number: {i}")
    print(f"skew: {skew_val}")
    print(f"kurt: {kurt_val}")
    print(f"entropy: {entropy_val}")
    plt.figure(figsize=(10, 4))
    plt.plot(abp_depurada[paciente_id][i])
    plt.xlabel('Muestras')
    plt.ylabel('Amplitud')
    plt.title(f'Paciente {paciente_id} - Ventana {ventana_id}')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()   
# %%

def analizar_dataset_completo(lista_senales):
    from scipy.stats import skew, kurtosis, entropy
    import numpy as np
    import matplotlib.pyplot as plt

    skewness_values = []
    kurtosis_values = []
    entropy_values = []

    for senial in lista_senales:
        if senial is None or np.all(np.isnan(senial)) or np.std(senial) < 1e-3:
            continue

        #hist, _ = np.histogram(senial, bins=20, density=True)
        #hist += 1e-10  # evitar log(0)

        skewness_values.append(skew(senial))
        kurtosis_values.append(kurtosis(senial))
        entropy_values.append(entropy(senial))

    skewness_values = np.array(skewness_values)
    kurtosis_values = np.array(kurtosis_values)
    entropy_values = np.array(entropy_values)

    valid_mask = np.isfinite(skewness_values)

    if not np.any(valid_mask):
        print("No hay datos válidos.")
        return np.nan, np.nan, np.nan

    plt.figure(figsize=(15, 4))

    plt.subplot(1, 3, 1)
    plt.hist(skewness_values[valid_mask], bins=50, color='lightblue', edgecolor='black')
    plt.title("Skewness global")

    plt.subplot(1, 3, 2)
    plt.hist(kurtosis_values[valid_mask], bins=50, color='salmon', edgecolor='black')
    plt.title("Kurtosis global")

    plt.subplot(1, 3, 3)
    plt.hist(entropy_values[valid_mask], bins=50, color='lightgreen', edgecolor='black')
    plt.title("Entropía global")

    plt.tight_layout()
    plt.show()

    return (
        np.mean(skewness_values[valid_mask]),
        np.mean(kurtosis_values[valid_mask]),
        np.mean(entropy_values[valid_mask])
    )

skew_global, kurt_global, entropy_global = analizar_dataset_completo(abp_normalized3)
print(f"skew_global: {skew_global}")
print(f"kurt_global: {kurt_global}")
print(f"entropy_global: {entropy_global}")

# %%
paciente_id = 58
n_signal_no_valid=0
for i in range(len(abp_depurada[paciente_id])):
    skew_val, kurt_val, entropy_val = signal_stats_analisis (abp_depurada[paciente_id][i])
    
    if(skew_val > 2 or skew_val < -2 or kurt_val > 1.5 or kurt_val < -1.5 
       or entropy_val < 3 or entropy_val > 4):
        print(f"number: {i}")
        print(f"skew: {skew_val}")
        print(f"kurt: {kurt_val}")
        print(f"entropy: {entropy_val}")
        plt.figure(figsize=(10, 4))
        plt.plot(abp_depurada[paciente_id][i])
        plt.xlabel('Muestras')
        plt.ylabel('Amplitud')
        plt.title(f'Paciente {paciente_id} - Ventana {ventana_id}')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()
        n_signal_no_valid = n_signal_no_valid + 1

if n_signal_no_valid == 0:
    print("no hay señales que cumplan con los requisitos")

# %%
n_signal_no_valid=0
for i in range(len(abp_depurada)):
    for j in range(len(abp_depurada[i])):
        skew_val, kurt_val, entropy_val = signal_stats_analisis (abp_depurada[i][j])
    
        if(kurt_val < 2 or kurt_val > 6 or entropy_val < 2 or entropy_val > 4.5 or skew_val > 0.3 or skew_val < 1.5):
            print(f"number: {i}")
            print(f"skew: {skew_val}")
            print(f"kurt: {kurt_val}")
            print(f"entropy: {entropy_val}")
            plt.figure(figsize=(10, 4))
            plt.plot(abp_depurada[paciente_id][i])
            plt.xlabel('Muestras')
            plt.ylabel('Amplitud')
            plt.title(f'Paciente {paciente_id} - Ventana {ventana_id}')
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plt.show()
            n_signal_no_valid = n_signal_no_valid + 1

if n_signal_no_valid == 0:
    print("no hay señales que cumplan con los requisitos")

#%%
"""
SBP_MEAN= (SBP_MEAN1+SBP_MEAN2+SBP_MEAN3+SBP_MEAN4)/4
DBP_MEAN= (DBP_MEAN1+DBP_MEAN2+DBP_MEAN3+DBP_MEAN4)/4
SBP_STD= (SBP_STD1+SBP_STD2+SBP_STD3+SBP_STD4)/4
DBP_STD=(DBP_STD1+DBP_MEAN2+DBP_STD3+DBP_STD4)/4

print(f"SBP_MEAN={SBP_MEAN:.2f}, SBP_STD={SBP_STD:.2f}")
print(f"DBP_MEAN={DBP_MEAN:.2f}, DBP_STD={DBP_STD:.2f}")
"""
#%%
presiones_sistolicas_norm, presiones_diastolicas_norm= labels_normalization(sbp_depurada, dbp_depurada, SBP_MEAN, SBP_STD, DBP_MEAN, DBP_STD)
#save_partial_file(ppg_normalized, ecg_normalized, presiones_sistolicas_norm, presiones_diastolicas_norm, 0, 'dataset_parte_1')
#%%
presiones_sistolicas_norm2, presiones_diastolicas_norm2= labels_normalization(sbp_depurada2, dbp_depurada2, SBP_MEAN, SBP_STD, DBP_MEAN, DBP_STD)
#save_partial_file(ppg_normalized2, ecg_normalized2, presiones_sistolicas_norm2, presiones_diastolicas_norm2, 3000, 'dataset_parte_2')
#%%
presiones_sistolicas_norm3, presiones_diastolicas_norm3= labels_normalization(sbp_depurada3, dbp_depurada3, SBP_MEAN, SBP_STD, DBP_MEAN, DBP_STD)
#save_partial_file(ppg_normalized3, ecg_normalized3, presiones_sistolicas_norm3, presiones_diastolicas_norm3, 6000, 'dataset_parte_3')
#%%
presiones_sistolicas_norm4, presiones_diastolicas_norm4= labels_normalization(sbp_depurada4, dbp_depurada4, SBP_MEAN, SBP_STD, DBP_MEAN, DBP_STD)
#save_partial_file(ppg_normalized4, ecg_normalized4, presiones_sistolicas_norm4, presiones_diastolicas_norm4, 9000, 'dataset_parte_4')



