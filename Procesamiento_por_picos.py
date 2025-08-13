#%% Importación de librerias

import Tools

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
    ecg_signal_filtrada_Q.append(Tools.filtrado_para_deteccion_Q(ecg_signal_filtrada))

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

#%%Detección de picos

picos_ppg = []
for l in range(len(ppg_norm)):
    picos = Tools.detectar_picos_ppg(ppg_norm[l])
    picos_ppg.append(picos)

picos_ecg = []
for m in range(len(ecg_norm_q)):
    picos = Tools.detectar_picos_ecg(ecg_norm_q[m])
    picos_ecg.append(picos)

