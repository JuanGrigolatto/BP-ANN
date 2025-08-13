import numpy as np
from scipy import signal
from scipy.stats import skew
from scipy.stats import kurtosis
from scipy.stats import entropy
from scipy.signal import find_peaks
import os
import torch
import h5py

#Normalización

def min_max_normalization(signal, max, min):
    signal = np.array(signal)
    return (signal - min) / (max - min)

def z_score_normalization(signal, epsilon=1e-8):
    signal = np.array(signal)
    mean = np.mean(signal)
    std = np.std(signal)
    return (signal - mean) / (std + epsilon)

def pressure_normalization_z_score(sbp, dbp, sbp_mean, sbp_std, dbp_mean, dbp_std):
    sbp_norm = (sbp - sbp_mean) / sbp_std
    dbp_norm = (dbp - dbp_mean) / dbp_std
    return sbp_norm, dbp_norm

def labels_normalization(matriz_presiones_sistolicas, matriz_presiones_diastolicas, SBP_MEAN, SBP_STD, DBP_MEAN, DBP_STD):
 
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

def signal_normalization(ppg_signals, abp_signals, ecg_signals):
    ppg_normalized = []
    ecg_normalized = []
    abp_normalized = []
    for i in range(len(ppg_signals)):  # Recorre pacientes
        ppg_paciente = []
        ecg_paciente = []
        abp_paciente = []

        for j in range(len(ppg_signals[i])):  # Recorre ventanas del paciente i
            ppg = ppg_signals[i][j]
            ecg = ecg_signals[i][j]
            abp = abp_signals[i][j]
            #ppg_n = min_max_normalization(ppg, np.max(ppg), np.min(ppg))
            #abp_n = min_max_normalization(abp, np.max(abp), np.min(abp))
            #ecg_n = min_max_normalization(ecg, np.max(ecg), np.min(ecg))

            ppg_n = z_score_normalization(ppg)
            ecg_n = z_score_normalization(ecg)
            abp_n = z_score_normalization(abp)

            ppg_paciente.append(ppg_n)
            ecg_paciente.append(ecg_n)
            abp_paciente.append(abp_n)

        ppg_normalized.append(ppg_paciente)
        ecg_normalized.append(ecg_paciente)
        abp_normalized.append(abp_paciente)
    return ppg_normalized, abp_normalized, ecg_normalized

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
#Filtrado

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

# Detección de picos en PPG y ECG

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

def detectar_picos_abp(abp, fs=125):
    max_val = np.max(abp)
    min_val = np.min(abp)
    rango = max_val - min_val

    prominence = 0.2 * rango
    height_sbp = min_val + 0.6 * rango
    height_dbp = min_val + 0.2 * rango
    distancia_min = int(0.3 * fs)

    peaks, _ = find_peaks(abp, height=height_sbp, prominence=prominence, distance=distancia_min)

    return peaks

# Ventaneo
def recortar_por_ventanas_cuadradas(signal, fs, t_window, overlap):
    window_size = fs * t_window  # Tamaño de la ventana en muestras
    step = int(window_size * (1 - overlap))  # Paso entre ventanas
    windows = []
    for i in range(0, len(signal) - window_size + 1, step):
        windows.append(signal[i:i + window_size])
    return np.array(windows)

def recortar_por_ventanas_no_cuadradas(senial, fs = 125, window ='Hamming', t_duration = 10, overlap = 0.7):
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

def adjust_window(win, max_len):
    if len(win) < max_len:
        return np.pad(win, (0, max_len - len(win)), mode='constant')
    elif len(win) > max_len:
        center = len(win)//2
        start_cut = max(0, center - max_len//2)
        return win[start_cut:start_cut + max_len]
    else:
        return win

def recortar_por_picos_sincronizado(ppg, abp, ecg, peaks, overlap_peaks=4, lenght_segment=500):
    
    segments_ppg = []
    segments_abp = []
    segments_ecg = []
    
    starts = []
    stops = []
    
    i = 0

    while i < (len(peaks) - 4):
        
            intervalo_rr_first = peaks[i+1] - peaks[i]
            intervalo_rr_last = peaks[i+4] - peaks[i+3] 
        
            pre_qrs=int(0.25 * intervalo_rr_first)   # para incluir onda P 
            post_qrs=int(0.45 *intervalo_rr_last)   # para incluir onda T

            start = max(0, peaks[i] - pre_qrs)
            stop  = min(len(ecg), peaks[i+4] + post_qrs)

            window_ecg = ecg[start:stop]
            window_ppg = ppg[start:stop]
            window_abp = abp[start:stop]

            window_ecg = adjust_window(window_ecg, lenght_segment)
            window_ppg = adjust_window(window_ppg, lenght_segment)
            window_abp = adjust_window(window_abp, lenght_segment)

            segments_ecg.append(window_ecg)
            segments_ppg.append(window_ppg)
            segments_abp.append(window_abp)    

            starts.append(start)
            stops.append(stop)
            # Avanzar al siguiente pico
            avance = max(1, overlap_peaks)  # avance mínimo de 1 pico
            i += avance
            
    return segments_ppg, segments_abp, segments_ecg, starts, stops


# Analisis estadistico de señales

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

# Obtener etiquetas de presión

def get_abp_labels(abp_signals, fs):
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

#Eliminar señales sin picos

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
#Leer archivos .mat

def leer_archivos_mat(ruta_archivo):
    with h5py.File(ruta_archivo, 'r') as f:
        claves = list(f.keys())
        print("Claves encontradas:", claves)

        # Asumimos que el dataset principal está en la primera clave
        dataset = f[claves[1]]

        # Convertir referencias internas a arrays
        datos_extraidos = []
        for i in range(dataset.shape[0]):
            ref = dataset[i, 0]  # referencia HDF5
            datos_extraidos.append(np.array(f[ref]))

        datos_extraidos = np.array(datos_extraidos, dtype=object)

        
        ppg_signal = [registro[:, 0] for registro in datos_extraidos]
        abp_signal = [registro[:, 1] for registro in datos_extraidos]
        ecg_signal = [registro[:, 2] for registro in datos_extraidos]

    return ppg_signal, abp_signal, ecg_signal

#Guardar datos en archivo .pt

def save_partial_file(ppg_signals, ecg_signals, sbp_labels, dbp_labels, patient_id_inicial, index_inicial, nombre_archivo):
    output_dir = 'data_UCI'
    os.makedirs(output_dir, exist_ok=True)

    num_total = sum(len(ppg) for ppg in ppg_signals)
    long_segmento = len(ppg_signals[0][0])  

    # Rutas de archivos temporales únicos por parte
    data_path = os.path.join(output_dir, f'{nombre_archivo}_data.dat')
    labels_path = os.path.join(output_dir, f'{nombre_archivo}_labels.dat')
    patients_path = os.path.join(output_dir, f'{nombre_archivo}_patients.dat')
    indexs_path = os.path.join(output_dir, f'{nombre_archivo}_indexs.dat')

    # Crear archivos memmap
    data_mmap = np.memmap(data_path, dtype='float32', mode='w+', shape=(num_total, 2, long_segmento))
    labels_mmap = np.memmap(labels_path, dtype='float32', mode='w+', shape=(num_total, 2))
    patients_mmap = np.memmap(patients_path, dtype='int64', mode='w+', shape=(num_total,))
    indexs_mmap = np.memmap(indexs_path, dtype='int64', mode='w+', shape=(num_total,))

    index = 0
    for paciente_id, (ppg_segmentos, ecg_segmentos, sbp_segmentos, dbp_segmentos) in enumerate(zip(ppg_signals, ecg_signals, sbp_labels, dbp_labels)):
        for ppg, ecg, sbp, dbp in zip(ppg_segmentos, ecg_segmentos, sbp_segmentos, dbp_segmentos):

            if np.isnan(ppg).any() or np.isnan(ecg).any() or np.isnan(sbp) or np.isnan(dbp):
                continue

            data_mmap[index, 0] = ppg
            data_mmap[index, 1] = ecg
            labels_mmap[index] = [sbp, dbp]
            patients_mmap[index] = paciente_id + patient_id_inicial
            indexs_mmap[index] = index + index_inicial
            index += 1

    # Recortar arrays al número real de muestras válidas
    data_tensor = torch.from_numpy(np.array(data_mmap[:index]))
    labels_tensor = torch.from_numpy(np.array(labels_mmap[:index]))
    patients_tensor = torch.from_numpy(np.array(patients_mmap[:index]))
    indexs_tensor = torch.from_numpy(np.array(indexs_mmap[:index]))

    # Guardar en archivo .pt
    torch.save({
        'data': data_tensor,
        'labels': labels_tensor,
        'patient_ids': patients_tensor,
        'index': indexs_tensor
        }, os.path.join(output_dir, f'{nombre_archivo}.pt'))

    print(f"{nombre_archivo}.pt guardado con {index} muestras.")

    return index

def get_num_patientsIDs(signals):
    n_ids=0
    for i in range(len(signals)):
        n_ids=n_ids+1
    return n_ids