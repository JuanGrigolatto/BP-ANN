#%% Importación de librerias

from src.utils.Tools import Tools
import matplotlib.pyplot as plt

#%%Carga de señales

ppg_signal, abp_signal, ecg_signal = Tools.leer_archivos_mat("data/raw/datos/Part_4.mat") 

#%%Filtrado de señales

ppg_signal_filtrada = []
ecg_signal_filtrada = []
ecg_signal_filtrada_Q = []
abp_signal_filtrada = []

for i in range(len(ppg_signal)):
    ppg_signal_filtrada.append(Tools.filtrar_ppg(ppg_signal[i]))
    ecg_signal_filtrada.append(Tools.filtrar_ecg(ecg_signal[i]))
    abp_signal_filtrada.append(Tools.filtrar_abp(abp_signal[i]))

for j in range(len(ecg_signal_filtrada)):
    ecg_signal_filtrada_Q.append(Tools.filtrado_para_deteccion_Q(ecg_signal_filtrada[j]))


#%% Detección de picos

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
#%%
"""
inicio=2078
plt.figure(figsize=(12, 6))    
for i in range(10):
    # Filtrar picos que estén dentro de las primeras 3000 muestras
    
    picos_filtrados = [p for p in picos_abp[i+inicio] if p < 3000]
    plt.subplot(10, 1, i + 1)
    plt.plot(abp_signal_filtrada[i+inicio][:3000], label='Señal de PPG', color='blue')
    plt.plot(picos_filtrados, abp_signal_filtrada[i+inicio][:3000][picos_filtrados], 'ro', label='Picos Detectados')
    plt.title(f'Señal de PPG {i + 1}')
    plt.xlabel('muestras')
    plt.ylabel('Amplitud')
    plt.legend()
"""
#%% Segmentación por picos
all_segments_ppg = []
all_segments_abp = []
all_segments_ecg = []

for h in range(len(ecg_signal_filtrada)):
    segmentos_ppg, segmentos_abp, segmentos_ecg, _ , _ = Tools.recortar_por_picos_sincronizado(ppg_signal_filtrada[h], abp_signal_filtrada[h], ecg_signal_filtrada[h], picos_ecg[h], 0, lenght_segment=500)
    all_segments_ppg.append(segmentos_ppg)
    all_segments_abp.append(segmentos_abp)
    all_segments_ecg.append(segmentos_ecg)
#%%
"""
inicio=0
plt.figure(figsize=(12, 6))    
for i in range(20):
    # Filtrar picos que estén dentro de las primeras 3000 muestras
    plt.subplot(20, 1, i + 1)
    plt.plot(all_segments_abp[1041][i+inicio], label='Señal de PPG', color='blue')
    plt.title(f'Señal de PPG {i + 1}')
    plt.xlabel('muestras')
    plt.ylabel('Amplitud')
    plt.legend()
"""
#%% Obtención de etiquetas

presiones_sistolicas, presiones_diastolicas, indices_sistolicos, indices_diastolicos = Tools.get_abp_labels(all_segments_abp, fs=125)

ppg_dep, abp_dep, ecg_dep, sbp_dep, dbp_dep, sistolicos_dep, diastolicos_dep = Tools.delete_signals_no_peaks(all_segments_ppg, all_segments_abp, all_segments_ecg, presiones_sistolicas, presiones_diastolicas, indices_sistolicos, indices_diastolicos)

sbp_min, sbp_max, dbp_min, dbp_max = Tools.max_min_pressures(sbp_dep, dbp_dep)

print(f"Presión sistólica mínima: {sbp_min}, máxima: {sbp_max}")
print(f"Presión diastólica mínima: {dbp_min}, máxima: {dbp_max}")
#%% Normalización
"""
for k in range(len(ppg_signal_filtrada)):

    ppg_norm.append(Tools.z_score_normalization(ppg_dep[k]))
    abp_norm.append(Tools.z_score_normalization(abp_dep[k]))
    ecg_norm.append(Tools.z_score_normalization(ecg_dep[k]))
"""
SBP_MEAN = 134.02
DBP_MEAN = 63.47
SBP_STD = 22.75
DBP_STD = 23.69

ppg_norm, abp_norm, ecg_norm = Tools.signal_normalization(ppg_dep, abp_dep, ecg_dep)

presiones_sistolicas_norm, presiones_diastolicas_norm= Tools.labels_normalization(sbp_dep, dbp_dep, SBP_MEAN, SBP_STD, DBP_MEAN, DBP_STD)

#%% 
"""
inicio=0
plt.figure(figsize=(12, 6))    
for i in range(20):
    # Filtrar picos que estén dentro de las primeras 3000 muestras
    plt.subplot(20, 1, i + 1)
    plt.plot(ppg_norm[1041][i+inicio], label='Señal de PPG', color='blue')
    plt.title(f'Señal de PPG {i + 1}')
    plt.xlabel('muestras')
    plt.ylabel('Amplitud')
    plt.legend()
"""
#%% Guardado de indice final de paciente

index = Tools.save_partial_file(ppg_norm, ecg_norm, presiones_sistolicas_norm, presiones_diastolicas_norm, 8975, 2922776, 'dataset_parte_4_por_picos')

num_pacientesIDs = Tools.get_num_patientsIDs(ppg_norm)

print((f"indice final del archivo : {index}")) 
print(f"Número de pacientes para archivo : {num_pacientesIDs}")
