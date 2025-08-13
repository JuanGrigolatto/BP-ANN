#%% Importación de librerias
import Tools
import matplotlib.pyplot as plt
import random
#%% Procesamiento 

#%%Carga de señales
ppg_signal, abp_signal, ecg_signal = Tools.leer_archivos_mat("datos/Part_4.mat") 
print(f"Cantidad de señales cargadas: {len(ppg_signal)}, {len(ecg_signal)}, {len(abp_signal)}")
"""
indice_paciente = 0 
plt.figure(figsize=(12, 8))

plt.subplot(3, 1, 1)
plt.plot(ppg_signal[indice_paciente][:1000], color='red')
plt.title(f'PPG - Paciente {indice_paciente}')
plt.ylabel('Amplitud')

plt.subplot(3, 1, 2)
plt.plot(abp_signal[indice_paciente][:1000], color='blue')
plt.title(f'ABP - Paciente {indice_paciente}')
plt.ylabel('Presión (mmHg)')

plt.subplot(3, 1, 3)
plt.plot(ecg_signal[indice_paciente][:1000], color='green')
plt.title(f'ECG - Paciente {indice_paciente}')
plt.xlabel('Muestras')
plt.ylabel('Amplitud')

plt.tight_layout()
plt.show()
"""
#%%Filtrado de señales
ppg_signal_filtrada = []
ecg_signal_filtrada = []

for i in range(len(ppg_signal)):
    ppg_signal_filtrada.append(Tools.filtrar_ppg(ppg_signal[i]))
    ecg_signal_filtrada.append(Tools.filtrar_ecg(ecg_signal[i]))
"""
indice_paciente = 0 
plt.figure(figsize=(12, 8))

plt.subplot(3, 1, 1)
plt.plot(ppg_signal_filtrada[indice_paciente][:1000], color='red')
plt.title(f'PPG - Paciente {indice_paciente}')
plt.ylabel('Amplitud')

plt.subplot(3, 1, 2)
plt.plot(abp_signal[indice_paciente][:1000], color='blue')
plt.title(f'ABP - Paciente {indice_paciente}')
plt.ylabel('Presión (mmHg)')

plt.subplot(3, 1, 3)
plt.plot(ecg_signal_filtrada[indice_paciente][:1000], color='green')
plt.title(f'ECG - Paciente {indice_paciente}')
plt.xlabel('Muestras')
plt.ylabel('Amplitud')

plt.tight_layout()
plt.show()
"""
print(f"Cantidad de señales post filtrado: {len(ppg_signal_filtrada)}, {len(ecg_signal_filtrada)}")
#%%Ventaneo con ventanas de 10 seg y Hanning

ppg_segmentada = []
ecg_segmentada = []
abp_segmentada = []
for j in range(len(ppg_signal_filtrada)):
    ppg_segmentada.append(Tools.recortar_por_ventanas_no_cuadradas(ppg_signal_filtrada[j], window = 'Hanning', t_duration=4, overlap=0.7)) 
    ecg_segmentada.append(Tools.recortar_por_ventanas_no_cuadradas(ecg_signal_filtrada[j], window = 'Hanning', t_duration=4, overlap=0.7)) 
    abp_segmentada.append(Tools.recortar_por_ventanas_cuadradas(abp_signal[j], fs=125 ,t_window=4, overlap=0.7))

print(f"Cantidad de ventanas: {len(ppg_segmentada)}, {len(ecg_segmentada)}")
"""
indice_paciente = 0 
plt.figure(figsize=(12, 8))

plt.subplot(3, 1, 1)
plt.plot(ppg_segmentada[indice_paciente][0], color='red')
plt.title(f'PPG - Paciente {indice_paciente}')
plt.ylabel('Amplitud')

plt.subplot(3, 1, 2)
plt.plot(abp_segmentada[indice_paciente][0], color='blue')
plt.title(f'ABP - Paciente {indice_paciente}')
plt.ylabel('Presión (mmHg)')

plt.subplot(3, 1, 3)
plt.plot(ecg_segmentada[indice_paciente][0], color='green')
plt.title(f'ECG - Paciente {indice_paciente}')
plt.xlabel('Muestras')
plt.ylabel('Amplitud')

plt.tight_layout()
plt.show()
"""
#%%Obtención de etiquetas
presiones_sistolicas, presiones_diastolicas, indices_sistolicos, indices_diastolicos = Tools.get_abp_labels(abp_segmentada, fs=125)

#Eliminación de señales sin picos de presión detectados
ppg_depurada, abp_depurada, ecg_depurada, sbp_depurada, dbp_depurada, indices_sist_dep, indices_dias_dep = Tools.delete_signals_no_peaks(ppg_segmentada, abp_segmentada, ecg_segmentada, presiones_sistolicas, 
                                                                                                    presiones_diastolicas, indices_sistolicos, indices_diastolicos)

#Normalización de señales por ventana
ppg_norm, abp_norm, ecg_norm = Tools.signal_normalization(ppg_depurada, abp_depurada, ecg_depurada)
print(f"Cantidad de ventanas normalizadas: {len(ppg_segmentada)}, {len(ecg_segmentada)} ")
"""
num_muestras=10
for i in range(num_muestras):

    ind_diastolicos=[]
    ind_sistolicos=[]

    paciente_id = random.randint(0, 2000)
    ventana_id = random.randint(0, len(abp_norm[paciente_id])-1 )
    senal_abp = abp_norm[paciente_id][ventana_id]

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
"""
#Normalización de etiquetas

SBP_MEAN = 134.02
DBP_MEAN = 63.47
SBP_STD = 22.75
DBP_STD = 23.69

#Normalización de presiones
presiones_sistolicas_norm, presiones_diastolicas_norm= Tools.labels_normalization(sbp_depurada, dbp_depurada, SBP_MEAN, SBP_STD, DBP_MEAN, DBP_STD)

#Guardado de indice final de paciente 
index = Tools.save_partial_file(ppg_depurada, ecg_depurada, presiones_sistolicas_norm, presiones_diastolicas_norm, 8994,  1635159, 'dataset_parte_4_hanning')

num_pacientesIDs = Tools.get_num_patientsIDs(ppg_depurada)

print((f"indice final del archivo : {index}")) 
print(f"Número de pacientes para archivo : {num_pacientesIDs}")