"""
Módulo: Procesamiento_por_picos.py
Autor: Juan Marcos Grigolatto
Descripción: Script principal de procesamiento para la generación de la base de datos 
             de entrenamiento. Carga archivos crudos (.mat), aplica filtrado, 
             sincroniza y segmenta las señales de PPG, ECG y ABP latido a latido 
             basándose en los picos R del ECG. Extrae etiquetas de presión 
             (Sistólica/Diastólica) y aplica normalización global.
"""

#--- IMPORTACIONES---
from src.utils.Tools import Tools
import matplotlib.pyplot as plt

#--- CARGA DE SEÑALES ---
# Extracción de las matrices de datos crudos de PPG, ABP y ECG desde el archivo .mat
ppg_signal, abp_signal, ecg_signal = Tools.leer_archivos_mat("data/raw/datos/Part_4.mat") 

#--- FILTRADO Y ACONDICIONAMIENTO DE SEÑALES ---
ppg_signal_filtrada = []
ecg_signal_filtrada = []
ecg_signal_filtrada_Q = []
abp_signal_filtrada = []

# Filtrado secuencial por paciente/registro para eliminar ruido y deriva de línea de base
for i in range(len(ppg_signal)):
    ppg_signal_filtrada.append(Tools.filtrar_ppg(ppg_signal[i]))
    ecg_signal_filtrada.append(Tools.filtrar_ecg(ecg_signal[i]))
    abp_signal_filtrada.append(Tools.filtrar_abp(abp_signal[i]))

# Acondicionamiento específico del ECG para maximizar la tasa de acierto en la detección del complejo QRS
for j in range(len(ecg_signal_filtrada)):
    ecg_signal_filtrada_Q.append(Tools.filtrado_para_deteccion_Q(ecg_signal_filtrada[j]))

#--- DETECCIÓN DE PICOS FIDUCIARIOS ---
# Identificación de máximos locales/picos en cada señal acondicionada
picos_ppg = []
for l in range(len(ppg_signal_filtrada)):
    picos = Tools.detectar_picos_ppg(ppg_signal_filtrada[l])
    picos_ppg.append(picos)

picos_ecg = []
for m in range(len(ecg_signal_filtrada_Q)):
    picos = Tools.detectar_picos_ecg(ecg_signal_filtrada_Q[m])
    picos_ecg.append(picos)

picos_abp = []
for n in range(len(abp_signal)):
    picos = Tools.detectar_picos_abp(abp_signal_filtrada[n])
    picos_abp.append(picos)

#--- SEGMENTACIÓN SINCRONIZADA POR PICOS ---
all_segments_ppg = []
all_segments_abp = []
all_segments_ecg = []

# Se utiliza el pico R del ECG como ancla temporal absoluta.
# Recorta ventanas de longitud fija (500 muestras) garantizando que las ondas 
# de PPG y ABP correspondan exactamente al mismo ciclo cardíaco que el ECG.
for h in range(len(ecg_signal_filtrada)):
    segmentos_ppg, segmentos_abp, segmentos_ecg, _ , _ = Tools.recortar_por_picos_sincronizado(ppg_signal_filtrada[h], abp_signal_filtrada[h], ecg_signal_filtrada[h], picos_ecg[h], 0, lenght_segment=500)
    all_segments_ppg.append(segmentos_ppg)
    all_segments_abp.append(segmentos_abp)
    all_segments_ecg.append(segmentos_ecg)

#--- EXTRACCIÓN DE ETIQUETAS Y LIMPIEZA DE DATOS ---
# 1. Extracción de las etiquetas objetivo (Ground Truth) desde la señal invasiva ABP
presiones_sistolicas, presiones_diastolicas, indices_sistolicos, indices_diastolicos = Tools.get_abp_labels(all_segments_abp, fs=125)

# 2. Control de calidad: Eliminación de segmentos corruptos o donde no se detectaron picos consistentes
ppg_dep, abp_dep, ecg_dep, sbp_dep, dbp_dep, sistolicos_dep, diastolicos_dep = Tools.delete_signals_no_peaks(all_segments_ppg, all_segments_abp, all_segments_ecg, presiones_sistolicas, presiones_diastolicas, indices_sistolicos, indices_diastolicos)

# Verificación de rangos para asegurar la validez fisiológica de las etiquetas extraídas
sbp_min, sbp_max, dbp_min, dbp_max = Tools.max_min_pressures(sbp_dep, dbp_dep)

print(f"Presión sistólica mínima: {sbp_min}, máxima: {sbp_max}")
print(f"Presión diastólica mínima: {dbp_min}, máxima: {dbp_max}")

#--- NORMALIZACIÓN DE SEÑALES Y ETIQUETAS ---
# Parámetros estadísticos poblacionales para la normalización Z-score.
SBP_MEAN = 134.02
DBP_MEAN = 63.47
SBP_STD = 22.75
DBP_STD = 23.69

# Normalización Z-score de los tensores de entrada (features) a nivel global
ppg_norm, abp_norm, ecg_norm, param_norm = Tools.signal_normalization_global(ppg_dep, abp_dep, ecg_dep)
print(f"Parámetros de normalización de señales de entrada: {param_norm}")

# Normalización de las etiquetas objetivo (targets)
presiones_sistolicas_norm, presiones_diastolicas_norm= Tools.labels_normalization(sbp_dep, dbp_dep, SBP_MEAN, SBP_STD, DBP_MEAN, DBP_STD)

#--- ENSAMBLADO Y GUARDADO DEL DATASET FINAL ---
index = Tools.save_partial_file(ppg_norm, ecg_norm, presiones_sistolicas_norm, presiones_diastolicas_norm, 8975, 2922776, 'dataset_parte_4_por_picos_global_norm')

num_pacientesIDs = Tools.get_num_patientsIDs(ppg_norm)

print((f"indice final del archivo : {index}")) 
print(f"Número de pacientes para archivo : {num_pacientesIDs}")
