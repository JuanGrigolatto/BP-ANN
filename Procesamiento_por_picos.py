#%% Importación de librerias

import Tools
import matplotlib.pyplot as plt

#%%Carga de señales

ppg_signal, abp_signal, ecg_signal = Tools.leer_archivos_mat("datos/Part_1.mat") 

#%%Filtrado de señales

ppg_signal_filtrada = []
ecg_signal_filtrada = []
ecg_signal_filtrada_Q = []

for i in range(len(ppg_signal)):
    ppg_signal_filtrada.append(Tools.filtrar_ppg(ppg_signal[i]))
    ecg_signal_filtrada.append(Tools.filtrar_ecg(ecg_signal[i]))

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
    picos = Tools.detectar_picos_abp(abp_signal[n])
    picos_abp.append(picos)
"""
inicio=60
plt.figure(figsize=(12, 6))    
for i in range(10):
    # Filtrar picos que estén dentro de las primeras 3000 muestras
    
    picos_filtrados = [p for p in picos_ecg[i+inicio] if p < 3000]
    plt.subplot(10, 1, i + 1)
    plt.plot(ecg_signal_filtrada[i+inicio][:3000], label='Señal de PPG', color='blue')
    plt.plot(picos_filtrados, ecg_signal_filtrada[i+inicio][:3000][picos_filtrados], 'ro', label='Picos Detectados')
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
    segmentos_ppg, segmentos_abp, segmentos_ecg, _ , _ = Tools.recortar_por_picos_sincronizado(ppg_signal_filtrada[h], abp_signal[h], ecg_signal_filtrada[h], picos_ecg[h], 0)
    all_segments_ppg.append(segmentos_ppg)
    all_segments_abp.append(segmentos_abp)
    all_segments_ecg.append(segmentos_ecg)

#%%Normalización

ppg_norm = []
abp_norm = []
ecg_norm_q = []
ecg_norm = [] 
for k in range(len(ppg_signal_filtrada)):

    ppg_norm.append(Tools.z_score_normalization(ppg_signal_filtrada[k]))
    abp_norm.append(Tools.z_score_normalization(abp_signal[k]))
    ecg_norm_q.append(Tools.z_score_normalization(ecg_signal_filtrada_Q[k]))
    ecg_norm.append(Tools.z_score_normalization(ecg_signal_filtrada[k]))


#%% Normalización y obtención de etiquetas
SBP_MEAN = 134.02
DBP_MEAN = 63.47
SBP_STD = 22.75
DBP_STD = 23.69

presiones_sistolicas, presiones_diastolicas, indices_sistolicos, indices_diastolicos = Tools.get_abp_labels(all_segments_abp, fs=125)

presiones_sistolicas_norm, presiones_diastolicas_norm= Tools.labels_normalization(presiones_sistolicas, presiones_diastolicas, SBP_MEAN, SBP_STD, DBP_MEAN, DBP_STD)

# %%
