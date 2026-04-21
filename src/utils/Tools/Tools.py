"""
Módulo: Tools.py
Autor: Juan Marcos Grigolatto
Descripción: Librería central de utilidades para el procesamiento de señales 
             biomédicas. Contiene las funciones core para la carga de datos (HDF5),
             filtrado digital, detección de picos fiduciarios, segmentación, 
             normalización y almacenamiento eficiente en disco mediante mapeo 
             de memoria (memmap) para el entrenamiento de modelos de Deep Learning.
"""
# --- IMPORTACIONES ---
import numpy as np
from scipy import signal
from scipy.stats import skew
from scipy.stats import kurtosis
from scipy.stats import entropy
from scipy.signal import find_peaks
import os
import torch
import h5py
import matplotlib.pyplot as plt

# --- ANÁLISIS DE ETIQUETAS Y PRESIONES ---

def max_min_pressures(sbp,dbp):
    """_summary_ Calcula los valores mínimos y máximos de presión sistólica (SBP) y diastólica (DBP) a partir de las matrices de etiquetas. 

    Args:
        sbp (_type_): _description_ Lista de listas con las presiones sistólicas (SBP) extraídas de las señales ABP invasivas.
        dbp (_type_): _description_ Lista de listas con las presiones diastólicas (DBP) extraídas de las señales ABP invasivas.

    Returns:
        _type_: _description_ Retorna los valores mínimos y máximos de presión sistólica (SBP) y diastólica (DBP) a partir de las matrices de etiquetas.
    """
    sbp = np.array([val for sublist in sbp for val in sublist])
    dbp = np.array([val for sublist in dbp for val in sublist])

    SBP_MIN = np.min(sbp)
    SBP_MAX = np.max(sbp)
    DBP_MIN = np.min(dbp)
    DBP_MAX = np.max(dbp)

    print(f"SBP_MIN={SBP_MIN:.2f}, SBP_MAX={SBP_MAX:.2f}")
    print(f"DBP_MIN={DBP_MIN:.2f}, DBP_MAX={DBP_MAX:.2f}")
    
    return SBP_MIN, SBP_MAX, DBP_MIN, DBP_MAX 

def min_max_normalization(signal, max, min):
    """_summary_ Aplica normalización Min-Max a la señal utilizando los valores mínimo y máximo de presión sistólica (SBP) y diastólica (DBP) para escalar los valores al rango [0, 1].

    Args:
        signal (_type_): _description_ Lista o array con la señal a normalizar.
        max (_type_): _description_ Valor máximo de la señal.
        min (_type_): _description_ Valor mínimo de la señal.

    Returns:
        _type_: _description_ Retorna la señal normalizada utilizando la fórmula de normalización Min-Max, escalando los valores al rango [0, 1].
    """
    signal = np.array(signal)
    return (signal - min) / (max - min)

def z_score_normalization(signal, epsilon=1e-8):
    """_summary_ Aplica normalización Z-score a la señal utilizando su media y desviación estándar para centrarla en cero y escalarla a unidades de desviación estándar. Se añade un pequeño valor epsilon para evitar divisiones por cero en caso de que la desviación estándar sea muy pequeña o nula.

    Args:
        signal (_type_): _description_ Lista o array con la señal a normalizar.
        epsilon (_type_, optional): _description_. Por defecto 1e-8. Valor pequeño añadido al denominador para evitar divisiones por cero en caso de desviación estándar muy pequeña o nula.

    Returns:
        _type_: _description_ Retorna la señal normalizada utilizando la fórmula de normalización Z-score, centrando los valores en cero y escalándolos a unidades de desviación estándar. Se añade un pequeño valor epsilon para evitar divisiones por cero en caso de que la desviación estándar sea muy pequeña o nula.
    """
    signal = np.array(signal)
    mean = np.mean(signal)
    std = np.std(signal)
    return (signal - mean) / (std + epsilon)

def pressure_normalization_z_score(sbp, dbp, sbp_mean, sbp_std, dbp_mean, dbp_std):
    """_summary_ Aplica normalización Z-score a las etiquetas de presión sistólica (SBP) y diastólica (DBP) utilizando sus medias y desviaciones estándar globales para centrar los valores en cero y escalarlos a unidades de desviación estándar. Se añaden pequeños valores epsilon para evitar divisiones por cero en caso de que las desviaciones estándar sean muy pequeñas o nulas.

    Args:
        sbp (_type_): _description_ Valor de presión sistólica (SBP) a normalizar. 
        dbp (_type_): _description_ Valor de presión diastólica (DBP) a normalizar.
        sbp_mean (_type_): _description_ Media global de las presiones sistólicas (SBP) calculada a partir de las etiquetas del dataset.
        sbp_std (_type_): _description_ Desviación estándar global de las presiones sistólicas (SBP) calculada a partir de las etiquetas del dataset.
        dbp_mean (_type_): _description_ Media global de las presiones diastólicas (DBP) calculada a partir de las etiquetas del dataset.
        dbp_std (_type_): _description_ Desviación estándar global de las presiones diastólicas (DBP) calculada a partir de las etiquetas del dataset.

    Returns:
        _type_: _description_ Retorna los valores normalizados de presión sistólica (SBP) y diastólica (DBP) utilizando la fórmula de normalización Z-score, centrando los valores en cero y escalándolos a unidades de desviación estándar. Se añaden pequeños valores epsilon para evitar divisiones por cero en caso de que las desviaciones estándar sean muy pequeñas o nulas.
    """    """"""
    sbp_norm = (sbp - sbp_mean) / sbp_std
    dbp_norm = (dbp - dbp_mean) / dbp_std
    return sbp_norm, dbp_norm

def desnormalizar_zscore(norm_array, media, std):
    """_summary_ Revierte la normalización Z-score para obtener los valores originales de presión sistólica (SBP) y diastólica (DBP) a partir de los valores normalizados, utilizando las medias y desviaciones estándar globales para escalar y centrar los valores de vuelta a su rango original.

    Args:
        norm_array (_type_): _description_ Array o lista con los valores normalizados de presión sistólica (SBP) o diastólica (DBP) que se desean desnormalizar.
        media (_type_): _description_ Media global de las presiones sistólicas (SBP) o diastólicas (DBP) calculada a partir de las etiquetas del dataset.
        std (_type_): _description_ Desviación estándar global de las presiones sistólicas (SBP) o diastólicas (DBP) calculada a partir de las etiquetas del dataset.

    Returns:
        _type_: _description_ Retorna los valores desnormalizados de presión sistólica (SBP) o diastólica (DBP) utilizando la fórmula de desnormalización Z-score, escalando y centrando los valores de vuelta a su rango original.
    """   
    return norm_array * std + media

def labels_normalization(matriz_presiones_sistolicas, matriz_presiones_diastolicas, SBP_MEAN, SBP_STD, DBP_MEAN, DBP_STD):
    """_summary_ Aplica normalización Z-score a las matrices de etiquetas de presión sistólica (SBP) y diastólica (DBP) utilizando sus medias y desviaciones estándar globales para centrar los valores en cero y escalarlos a unidades de desviación estándar. 

    Args:
        matriz_presiones_sistolicas (_type_): _description_ Lista de listas con las presiones sistólicas (SBP) extraídas de las señales ABP invasivas, que se desean normalizar.
        matriz_presiones_diastolicas (_type_): _description_  Lista de listas con las presiones diastólicas (DBP) extraídas de las señales ABP invasivas, que se desean normalizar.
        SBP_MEAN (_type_): _description_ Media global de las presiones sistólicas (SBP) calculada a partir de las etiquetas del dataset, que se utilizará para centrar los valores de SBP en la normalización Z-score.
        SBP_STD (_type_): _description_ Desviación estándar global de las presiones sistólicas (SBP) calculada a partir de las etiquetas del dataset, que se utilizará para escalar los valores de SBP a unidades de desviación estándar en la normalización Z-score.
        DBP_MEAN (_type_): _description_ Media global de las presiones diastólicas (DBP) calculada a partir de las etiquetas del dataset, que se utilizará para centrar los valores de DBP en la normalización Z-score.
        DBP_STD (_type_): _description_ Desviación estándar global de las presiones diastólicas (DBP) calculada a partir de las etiquetas del dataset, que se utilizará para escalar los valores de DBP a unidades de desviación estándar en la normalización Z-score.

    Returns:
        _type_: _description_ Retorna las matrices de etiquetas normalizadas de presión sistólica (SBP) y diastólica (DBP) utilizando la fórmula de normalización Z-score, centrando los valores en cero y escalándolos a unidades de desviación estándar. La función también filtra y descarta cualquier etiqueta que resulte en valores NaN o Inf tras la normalización, asegurando que solo se mantengan etiquetas válidas para el entrenamiento del modelo.
    """    
    matriz_presiones_sistolicas_norm=[]
    matriz_presiones_diastolicas_norm=[]
    for i in range(len(matriz_presiones_sistolicas)):
        list_sbp=[]
        list_dbp=[]  
        for j in range(len(matriz_presiones_sistolicas[i])):          
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
    """Aplica normalización Z-score local (por ventana) y filtra segmentos con valores NaN/Inf."""
    ppg_normalized = []
    ecg_normalized = []
    abp_normalized = []

    total_ventanas = 0
    ventanas_eliminadas = 0
    for i in range(len(ppg_signals)):  # Recorre pacientes
        ppg_paciente = []
        ecg_paciente = []
        abp_paciente = []

        for j in range(len(ppg_signals[i])):  # Recorre ventanas del paciente i
            total_ventanas += 1
            ppg = ppg_signals[i][j]
            ecg = ecg_signals[i][j]
            abp = abp_signals[i][j]
            if(np.isnan(ppg).any() or np.isnan(ecg).any() or np.isnan(abp).any() or \
               np.isinf(ppg).any() or np.isinf(ecg).any() or np.isinf(abp).any()):
                ventanas_eliminadas += 1
                continue

            ppg_n = z_score_normalization(ppg)
            ecg_n = z_score_normalization(ecg)
            abp_n = z_score_normalization(abp)

            if (np.isnan(ppg_n).any() or np.isnan(ecg_n).any() or np.isnan(abp_n).any()):
                    ventanas_eliminadas += 1
                    continue

            ppg_paciente.append(ppg_n)
            ecg_paciente.append(ecg_n)
            abp_paciente.append(abp_n)

        if ppg_paciente:
            ppg_normalized.append(ppg_paciente)
            ecg_normalized.append(ecg_paciente)
            abp_normalized.append(abp_paciente)

    print(f"Ventanas eliminadas por NaN/inf: {ventanas_eliminadas}")

    return ppg_normalized, abp_normalized, ecg_normalized

def signal_normalization_global(ppg_signals, abp_signals, ecg_signals):
    """_summary_  Aplica normalización Z-score global a las señales de fotopletismografía (PPG), electrocardiograma (ECG) y presión arterial (ABP) utilizando la media y desviación estándar calculadas a partir de todas las ventanas del dataset para cada tipo de señal. La función también filtra y descarta cualquier ventana que contenga valores NaN o Inf antes de calcular los parámetros de normalización, asegurando que solo se utilicen datos válidos para el cálculo de la media y desviación estándar globales. Finalmente, se aplica la normalización Z-score a cada ventana utilizando los parámetros globales calculados, y se devuelve la lista de señales normalizadas junto con los parámetros utilizados para la normalización.

    Args:
        ppg_signals (_type_): _description_ Lista de listas con las señales de fotopletismografía (PPG) segmentadas en ventanas, organizadas por paciente.
        abp_signals (_type_): _description_ Lista de listas con las señales de presión arterial (ABP) segmentadas en ventanas, organizadas por paciente.
        ecg_signals (_type_): _description_ Lista de listas con las señales de electrocardiograma (ECG) segmentadas en ventanas, organizadas por paciente.

    Returns:
        _type_: _description_ Retorna las señales normalizadas utilizando normalización Z-score global, calculando la media y desviación estándar a partir de todas las ventanas del dataset para cada tipo de señal (PPG, ECG, ABP). La función también filtra y descarta cualquier ventana que contenga valores NaN o Inf antes de calcular los parámetros de normalización, asegurando que solo se utilicen datos válidos para el cálculo de la media y desviación estándar globales. Finalmente, se aplica la normalización Z-score a cada ventana utilizando los parámetros globales calculados, y se devuelve la lista de señales normalizadas junto con los parámetros utilizados para la normalización.
    """    

    ppg_normalized = []
    ecg_normalized = []
    abp_normalized = []

    total_ventanas = 0
    ventanas_eliminadas = 0

    ppg_all = []
    ecg_all = []
    abp_all = []

    for i in range(len(ppg_signals)):
        for j in range(len(ppg_signals[i])):
            ppg = ppg_signals[i][j]
            ecg = ecg_signals[i][j]
            abp = abp_signals[i][j]

            if (np.isnan(ppg).any() or np.isnan(ecg).any() or np.isnan(abp).any() or
                np.isinf(ppg).any() or np.isinf(ecg).any() or np.isinf(abp).any()):
                continue

            ppg_all.append(ppg)
            ecg_all.append(ecg)
            abp_all.append(abp)

    ppg_all = np.concatenate(ppg_all)
    ecg_all = np.concatenate(ecg_all)
    abp_all = np.concatenate(abp_all)

    ppg_mean, ppg_std = np.mean(ppg_all), np.std(ppg_all)
    ecg_mean, ecg_std = np.mean(ecg_all), np.std(ecg_all)
    abp_mean, abp_std = np.mean(abp_all), np.std(abp_all)

    print(f"[NORMALIZACIÓN GLOBAL - Archivo actual]")
    print(f"Medias:  PPG={ppg_mean:.4f}, ECG={ecg_mean:.4f}, ABP={abp_mean:.4f}")
    print(f"Desvíos: PPG={ppg_std:.4f}, ECG={ecg_std:.4f}, ABP={abp_std:.4f}")

    for i in range(len(ppg_signals)):
        ppg_paciente = []
        ecg_paciente = []
        abp_paciente = []

        for j in range(len(ppg_signals[i])):
            total_ventanas += 1
            ppg = ppg_signals[i][j]
            ecg = ecg_signals[i][j]
            abp = abp_signals[i][j]

            if (np.isnan(ppg).any() or np.isnan(ecg).any() or np.isnan(abp).any() or
                np.isinf(ppg).any() or np.isinf(ecg).any() or np.isinf(abp).any()):
                ventanas_eliminadas += 1
                continue

            # Aplicar z-score global
            ppg_n = (ppg - ppg_mean) / ppg_std
            ecg_n = (ecg - ecg_mean) / ecg_std
            abp_n = (abp - abp_mean) / abp_std

            if (np.isnan(ppg_n).any() or np.isnan(ecg_n).any() or np.isnan(abp_n).any()):
                ventanas_eliminadas += 1
                continue

            ppg_paciente.append(ppg_n)
            ecg_paciente.append(ecg_n)
            abp_paciente.append(abp_n)

        if ppg_paciente:
            ppg_normalized.append(ppg_paciente)
            ecg_normalized.append(ecg_paciente)
            abp_normalized.append(abp_paciente)

    print(f"Ventanas eliminadas por NaN/inf: {ventanas_eliminadas}/{total_ventanas}")

    # Devuelve las señales normalizadas y los parámetros usados 
    return ppg_normalized, abp_normalized, ecg_normalized, (ppg_mean, ppg_std, ecg_mean, ecg_std, abp_mean, abp_std)

def calcular_pam(sbp: torch.Tensor, dbp: torch.Tensor) -> torch.Tensor:
    """_summary_ Calcula la presión arterial media (PAM) a partir de los valores de presión sistólica (SBP) y diastólica (DBP) utilizando la fórmula PAM = (SBP + 2*DBP) / 3.

    Args:
        sbp (torch.Tensor): _description_ Tensor con los valores de presión sistólica (SBP) para los cuales se desea calcular la presión arterial media (PAM).
        dbp (torch.Tensor): _description_ Tensor con los valores de presión diastólica (DBP) para los cuales se desea calcular la presión arterial media (PAM).

    Returns:
        torch.Tensor: _description_ Tensor con los valores de presión arterial media (PAM) calculados.
    """    
    return (sbp + (2*dbp)) / 3

def get_pam_labels(matriz_presiones_sistolicas, matriz_presiones_diastolicas):
    """_summary_ Calcula la presión arterial media (PAM) a partir de las matrices de etiquetas de presión sistólica (SBP) y diastólica (DBP). 

    Args:
        matriz_presiones_sistolicas (_type_): _description_  Lista de listas con las presiones sistólicas (SBP) extraídas de las señales ABP invasivas, que se utilizarán para calcular la presión arterial media (PAM).
        matriz_presiones_diastolicas (_type_): _description_ Lista de listas con las presiones diastólicas (DBP) extraídas de las señales ABP invasivas, que se utilizarán para calcular la presión arterial media (PAM).

    Returns:
        _type_: _description_ Retorna una matriz de etiquetas de presión arterial media (PAM) calculada a partir de las matrices de etiquetas de presión sistólica (SBP) y diastólica (DBP).
    """    
    matriz_presiones_media = []
    for i in range(len(matriz_presiones_sistolicas)):
        presiones_media = []
        for j in range(len(matriz_presiones_sistolicas[i])):
            sbp = matriz_presiones_sistolicas[i][j]
            dbp = matriz_presiones_diastolicas[i][j]
            if not np.isnan(sbp) and not np.isnan(dbp):
                pam = calcular_pam(sbp, dbp)
                presiones_media.append(pam)
            else:
                presiones_media.append(np.nan)
        matriz_presiones_media.append(presiones_media)
    return matriz_presiones_media

# --- Filtrado ---

def filtrar_ppg(senial_ppg):
    """_summary_ Aplica filtro pasabanda Butterworth (0.5-21 Hz) y sustrae la media para la señal de fotopletismografía (PPG).
    Args:
        senial_ppg (_type_): _description_ Lista o array con la señal de fotopletismografía (PPG) que se desea filtrar.

    Returns:
        _type_: _description_ Retorna la señal de fotopletismografía (PPG) filtrada.
    """    
    orden = 4
    orden = 4
    frec_sup = 21
    frec_inf = 0.5

    frecs_corte = [frec_inf, frec_sup]
    b, a = signal.butter(orden, frecs_corte, 'bandpass', fs = 125)

    ppg_filtrada = signal.filtfilt(b,a, senial_ppg)
    
    #Filtro de media 
    baseline = np.mean(ppg_filtrada)

    # Señal sin línea de base
    ppg_filtrada = ppg_filtrada - baseline 

    return ppg_filtrada

def filtrar_ecg(senial_ecg):
    """_summary_ Aplica filtro pasabanda Butterworth (0.5-40 Hz) y sustrae la media para la señal ECG.
    Args:
        senial_ecg (_type_): _description_ Lista o array con la señal ECG que se desea filtrar.

    Returns:
        _type_: _description_ Retorna la señal ECG filtrada.
    """
    orden = 4
    frec_sup = 40
    frec_inf = 0.5

    frecs_corte = [frec_inf, frec_sup]
    b, a = signal.butter(orden, frecs_corte, 'bandpass', fs = 125)

    ecg_filtrada = signal.filtfilt(b,a, senial_ecg)

    #Filtro de media 
    baseline = np.mean(ecg_filtrada)

    # Señal sin línea de base
    ecg_filtrada = ecg_filtrada - baseline

    return ecg_filtrada

def filtrado_para_deteccion_Q(senial_ecg):
    """_summary_ Aplica filtro pasabanda Butterworth (5-15 Hz) para mejorar la detección de ondas Q en la señal ECG.
    Args:
        senial_ecg (_type_): _description_ Lista o array con la señal ECG que se desea filtrar para mejorar la detección de ondas Q.

    Returns:
        _type_: _description_ Retorna la señal ECG filtrada.
    """    
    orden = 4
    frec_sup = 15
    frec_inf = 5
    frecs_corte = [frec_inf, frec_sup]

    b, a = signal.butter(orden, frecs_corte, 'bandpass', fs = 125)

    ecg_filtrada = signal.filtfilt(b,a, senial_ecg)

    return ecg_filtrada

def filtrar_abp(senial_abp):
    """_summary_ Aplica filtro pasa bajos Butterworth (21 Hz) para la señal de presión arterial invasiva (ABP).

    Args:
        senial_abp (_type_): _description_ Lista o array con la señal de presión arterial invasiva (ABP) que se desea filtrar.

    Returns:
        _type_: _description_ Retorna la señal de presión arterial invasiva (ABP) filtrada.
    """    
    orden = 4
    frec_corte= 21
   
    b, a = signal.butter(orden, frec_corte, 'lowpass', fs = 125)

    ecg_filtrada = signal.filtfilt(b,a, senial_abp)

    return ecg_filtrada

# --- Detección de picos en PPG y ECG ---

def detectar_picos_ppg(ppg, fs=125):
    """_summary_ Detecta los picos en la señal de fotopletismografía (PPG) utilizando el método de prominencia y altura relativa para identificar los picos fiduciarios correspondientes a los latidos cardíacos. La función ajusta dinámicamente los parámetros de detección de picos en función del rango de amplitud de la señal PPG, estableciendo un umbral de prominencia y altura relativa para garantizar una detección robusta incluso en presencia de ruido o variabilidad en la señal. Además, se establece una distancia mínima entre picos para evitar la detección de falsos positivos debido a artefactos o ruido de alta frecuencia.

    Args:
        ppg (_type_): _description_ Lista o array con la señal de fotopletismografía (PPG) en la que se desea detectar los picos correspondientes a los latidos cardíacos.
        fs (int, optional): _description_. Por defecto 125. Frecuencia de muestreo de la señal PPG, que se utiliza para calcular la distancia mínima entre picos en función de la duración esperada de un ciclo cardíaco.

    Returns:
        _type_: _description_ Retorna un array con los índices de los picos detectados en la señal de fotopletismografía (PPG) que corresponden a los latidos cardíacos.
    """    
    
    max_val = np.max(ppg)
    min_val = np.min(ppg)
    rango = max_val - min_val

    prominence = 0.1 * rango
    height = min_val + 0.5 * rango
    distancia_min = int(0.4 * fs)

    # Detección de picos
    peaks, _ = signal.find_peaks(ppg, height=height, prominence=prominence, distance=distancia_min)
    
    return peaks

def detectar_picos_ecg(ecg, fs=125, hr_min=40, hr_max=180):
    """_summary_ Detecta ondas R en la señal de ECG basado en la frecuencia cardíaca máxima y prominencia.

    Args:
        ecg (_type_): _description_ Lista o array con la señal de ECG en la que se desea detectar las ondas R.
        fs (int, optional): _description_. Por defecto 125. Frecuencia de muestreo de la señal ECG.
        hr_min (int, optional): _description_. Por defecto 40. Frecuencia cardíaca mínima esperada.
        hr_max (int, optional): _description_. Por defecto 180. Frecuencia cardíaca máxima esperada.

    Returns:
        _type_: _description_ Retorna un array con los índices de las ondas R detectadas en la señal de ECG.
    """
    max_val = np.max(ecg)
    min_val = np.min(ecg)
    rango = max_val - min_val

    prominence = 0.2 * rango              
    height = min_val + 0.4 * rango        

    distancia_min = int(fs * 60 / hr_max)

    peaks, _ = signal.find_peaks(ecg, height=height, prominence=prominence, distance=distancia_min)
    return peaks

def detectar_picos_abp(abp, fs=125):
    """_summary_ Detecta los picos en la señal de presión arterial invasiva (ABP) utilizando el método de prominencia y altura relativa para identificar los picos fiduciarios correspondientes a los latidos cardíacos. La función ajusta dinámicamente los parámetros de detección de picos en función del rango de amplitud de la señal ABP, estableciendo un umbral de prominencia y altura relativa para garantizar una detección robusta incluso en presencia de ruido o variabilidad en la señal. Además, se establece una distancia mínima entre picos para evitar la detección de falsos positivos debido a artefactos o ruido de alta frecuencia.

    Args:
        abp (_type_): _description_ Lista o array con la señal de presión arterial invasiva (ABP) en la que se desea detectar los picos correspondientes a los latidos cardíacos.
        fs (int, optional): _description_. Por defecto 125. Frecuencia de muestreo de la señal ABP.

    Returns:
        _type_: _description_ Retorna un array con los índices de los picos detectados en la señal de presión arterial invasiva (ABP) que corresponden a los latidos cardíacos.
    """    
    max_val = np.max(abp)
    min_val = np.min(abp)
    max_val = np.max(abp)
    min_val = np.min(abp)
    rango = max_val - min_val

    prominence = 0.2 * rango
    height_sbp = min_val + 0.6 * rango
    height_dbp = min_val + 0.2 * rango
    distancia_min = int(0.3 * fs)

    peaks, _ = find_peaks(abp, height=height_sbp, prominence=prominence, distance=distancia_min)

    return peaks

# --- Ventaneo ---

def recortar_por_ventanas_cuadradas(signal, fs, t_window, overlap):
    """_summary_ Segmenta la señal en ventanas cuadradas de duración fija (t_window) con un porcentaje de solapamiento (overlap) especificado. La función calcula el tamaño de la ventana en muestras a partir de la frecuencia de muestreo (fs) y la duración de la ventana, y luego recorta la señal en segmentos utilizando un bucle que avanza por la señal en pasos determinados por el tamaño de la ventana y el porcentaje de solapamiento. El resultado es un array de ventanas segmentadas que pueden ser utilizadas para análisis posteriores o entrenamiento de modelos de aprendizaje automático.

    Args:
        signal (_type_): _description_ Lista o array con la señal que se desea segmentar en ventanas cuadradas.
        fs (_type_): _description_ Frecuencia de muestreo de la señal, que se utiliza para calcular el tamaño de la ventana en muestras a partir de la duración de la ventana en segundos (t_window).
        t_window (_type_): _description_ Duración de la ventana en segundos.
        overlap (_type_): _description_ Porcentaje de solapamiento entre ventanas.

    Returns:
        _type_: _description_ Retorna un array de ventanas segmentadas a partir de la señal original, donde cada ventana tiene una duración fija (t_window) y un porcentaje de solapamiento (overlap) especificado.
    """    
    window_size = fs * t_window  # Tamaño de la ventana en muestras
    step = int(window_size * (1 - overlap))  # Paso entre ventanas
    windows = []
    for i in range(0, len(signal) - window_size + 1, step):
        windows.append(signal[i:i + window_size])
    return np.array(windows)

def recortar_por_ventanas_no_cuadradas(senial, fs = 125, window ='Hamming', t_duration = 10, overlap = 0.7):
    """_summary_ Segmenta la señal en ventanas no cuadradas de duración fija (t_duration) con un porcentaje de solapamiento (overlap) especificado, aplicando una ventana de tipo especificado (window) a cada segmento para reducir los efectos de borde. La función calcula el tamaño de la ventana en muestras a partir de la frecuencia de muestreo (fs) y la duración de la ventana, y luego recorta la señal en segmentos utilizando un bucle que avanza por la señal en pasos determinados por el tamaño de la ventana y el porcentaje de solapamiento. Para cada segmento recortado, se aplica una ventana del tipo especificado (Hamming, Hanning, Blackman o Rectangular) para suavizar los bordes del segmento y reducir los efectos de discontinuidad. El resultado es una lista de segmentos segmentados y suavizados que pueden ser utilizados para análisis posteriores o entrenamiento de modelos de aprendizaje automático.

    Args:
        senial (_type_): _description_ Lista o array con la señal que se desea segmentar en ventanas no cuadradas.
        fs (int, optional): _description_. Por defecto 125. Frecuencia de muestreo de la señal, que se utiliza para calcular el tamaño de la ventana en muestras a partir de la duración de la ventana en segundos (t_duration). 
        window (str, optional): _description_. Por defecto 'Hamming'. Tipo de ventana a aplicar a cada segmento recortado para reducir los efectos de borde. Las opciones disponibles son 'Hamming', 'Hanning', 'Blackman' y 'Rectangular'.
        overlap (float, optional): _description_. Por defecto 0.7. Porcentaje de solapamiento entre ventanas.

    Returns:
        _type_: _description_ Retorna una lista de segmentos segmentados y suavizados a partir de la señal original, donde cada segmento tiene una duración fija (t_duration) y un porcentaje de solapamiento (overlap) especificado, y se ha aplicado una ventana del tipo especificado (window) para reducir los efectos de borde.
    """    
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
    """_summary_ Segmenta la señal en ventanas no cuadradas centradas en los picos detectados, utilizando un número específico de picos para definir el inicio y el final de cada segmento. La función recorta segmentos de la señal que comienzan un número determinado de picos antes del pico actual y terminan un número determinado de picos después del pico actual, asegurando que cada segmento capture un ciclo cardíaco completo. El parámetro overlap_peaks permite controlar el solapamiento entre segmentos consecutivos, avanzando por la señal en pasos determinados por el número de picos especificado. El resultado es una lista de segmentos segmentados a partir de la señal original, donde cada segmento está centrado en un pico detectado y captura un ciclo cardíaco completo.

    Args:
        seniales (_type_): _description_ Lista de listas con las señales que se desean segmentar en ventanas no cuadradas centradas en los picos detectados, organizadas por paciente.
        list_peaks (_type_): _description_ Lista de listas con los índices de los picos detectados en cada señal, organizados por paciente, que se utilizarán para definir el inicio y el final de cada segmento recortado.
        overlap_peaks (int, optional): _description_. Por defecto 4. Número de picos antes y después del pico actual que se utilizarán para definir el inicio y el final de cada segmento recortado, respectivamente. Este parámetro controla el solapamiento entre segmentos consecutivos, avanzando por la señal en pasos determinados por el número de picos especificado.

    Returns:
        _type_: _description_
    """    
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
    """_summary_ Ajusta una ventana de señal a una longitud específica (max_len) mediante padding o recorte centrado. Si la ventana es más corta que max_len, se agrega padding a ambos lados para mantenerla centrada. Si la ventana es más larga que max_len, se recorta desde el centro para conservar la parte central de la señal. Si la ventana ya tiene la longitud deseada, se devuelve sin cambios.

    Args:
        win (_type_): _description_ Lista o array con la ventana de señal que se desea ajustar a una longitud específica (max_len).
        max_len (_type_): _description_ Longitud deseada para la ventana de señal.

    Returns:
        _type_: _description_ Retorna la ventana de señal ajustada a la longitud específica (max_len) mediante padding o recorte centrado, dependiendo de si la ventana original es más corta o más larga que max_len. Si la ventana ya tiene la longitud deseada, se devuelve sin cambios.
    """    
    diff = max_len - len(win)
    if diff > 0:
        # Pad a izquierda y derecha para mantener centrado
        pad_left = diff // 2
        pad_right = diff - pad_left
        return np.pad(win, (pad_left, pad_right), mode='constant')
    elif diff < 0:
        # Recortar desde el centro
        center = len(win) // 2
        start_cut = center - max_len // 2
        return win[start_cut:start_cut + max_len]
    else:
        return win
    
def recortar_por_picos_sincronizado(ppg, abp, ecg, peaks, overlap_peaks=4, lenght_segment=500):
    """_summary_ Segmenta las señales de fotopletismografía (PPG), presión arterial invasiva (ABP) y electrocardiograma (ECG) en ventanas no cuadradas centradas en los picos detectados en la señal ECG, utilizando un número específico de picos para definir el inicio y el final de cada segmento. La función recorta segmentos de las señales que comienzan un número determinado de picos antes del pico actual y terminan un número determinado de picos después del pico actual, asegurando que cada segmento capture un ciclo cardíaco completo. El parámetro overlap_peaks permite controlar el solapamiento entre segmentos consecutivos, avanzando por las señales en pasos determinados por el número de picos especificado. Además, se ajusta cada segmento a una longitud específica (lenght_segment) mediante padding o recorte centrado para garantizar que todas las ventanas tengan la misma longitud, lo que facilita su uso para análisis posteriores o entrenamiento de modelos de aprendizaje automático.

    Args:
        ppg (_type_): _description_ Lista o array con la señal de fotopletismografía (PPG) que se desea segmentar en ventanas no cuadradas centradas en los picos detectados en la señal ECG.
        abp (_type_): _description_ Lista o array con la señal de presión arterial invasiva (ABP) que se desea segmentar en ventanas no cuadradas centradas en los picos detectados en la señal ECG.
        ecg (_type_): _description_ Lista o array con la señal de electrocardiograma (ECG) que se desea segmentar en ventanas no cuadradas centradas en los picos detectados en la señal ECG.
        peaks (_type_): _description_ Lista o array con los índices de los picos detectados en la señal ECG, que se utilizarán para definir el inicio y el final de cada segmento recortado.
        overlap_peaks (int, optional): _description_ Número de picos que se solapan entre segmentos consecutivos. Por defecto 4.
        lenght_segment (int, optional): _description_ Longitud deseada para cada segmento de señal. Por defecto 500.

    Returns:
        _type_: _description_ Retorna tres listas de segmentos segmentados a partir de las señales de fotopletismografía (PPG), presión arterial invasiva (ABP) y electrocardiograma (ECG), donde cada segmento está centrado en un pico detectado en la señal ECG y captura un ciclo cardíaco completo. 
    """    
    segments_ppg = []
    segments_abp = []
    segments_ecg = []
    
    starts = []
    stops = []
    
    i = 0

    while i < (len(peaks) - 4):
            
        ventana_muestras = lenght_segment
        start = peaks[i] - ventana_muestras // 2
        stop = start + ventana_muestras

        if start < 0:
            start = 0
            stop = ventana_muestras

        if stop > len(ecg):
            stop = len(ecg)
            start = stop - ventana_muestras
            if start < 0:  # si la señal es más corta que la ventana
                start = 0

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
        avance = max(1, overlap_peaks) 
        i += avance
            
    return segments_ppg, segments_abp, segments_ecg, starts, stops


# --- ANÁLISIS ESTADÍSTICO DE SEÑALES ---

def signal_stats_analisis(signal):
     
    skewness_value = skew(signal)
    kurtosis_value = kurtosis(signal)
    #entropy_value = entropy(signal)
     
    # Entropía de Shannon a partir de histograma
    
    hist, _ = np.histogram(signal, bins=50, density=True)
    p = hist + 1e-12
    p = p / np.sum(p)
    entropy_value = entropy(p)
    

    #return skewness_value, kurtosis_value, entropy_value
    return skewness_value, kurtosis_value, entropy_value

# --- EXTRACCIÓN DE ETIQUETAS Y LIMPIEZA ---

def get_abp_labels(abp_signals, fs):
    """_summary_ Extrae las etiquetas de presión sistólica (SBP) y diastólica (DBP) de las señales de presión arterial invasiva (ABP) utilizando la detección de picos para identificar los picos sistólicos y diastólicos en cada ventana de señal. La función recorre cada ventana de señal ABP, detecta los picos correspondientes a los latidos cardíacos utilizando el método de prominencia y altura relativa, y luego extrae los valores de presión sistólica (SBP) y diastólica (DBP) asociados a esos picos. Se aplican criterios fisiológicos para validar las etiquetas extraídas, asegurando que solo se consideren válidas aquellas ventanas donde la presión sistólica sea mayor que la presión diastólica. Las etiquetas extraídas se organizan en matrices separadas para SBP y DBP, junto con las matrices de índices de picos sistólicos y diastólicos correspondientes.

    Args:
        abp_signals (_type_): _description_ Lista de listas con las señales de presión arterial invasiva (ABP) organizadas por paciente y ventana, de las cuales se desea extraer las etiquetas de presión sistólica (SBP) y diastólica (DBP) utilizando la detección de picos.
        fs (_type_): _description_  Frecuencia de muestreo de las señales ABP, que se utiliza para calcular la distancia mínima entre picos en función de la duración esperada de un ciclo cardíaco.

    Returns:
        _type_: _description_ Retorna cuatro matrices: matriz_presiones_sistolicas, matriz_presiones_diastolicas, matriz_picos_sistolicos y matriz_picos_diastolicos. La matriz_presiones_sistolicas contiene las etiquetas de presión sistólica (SBP) extraídas de las señales ABP, la matriz_presiones_diastolicas contiene las etiquetas de presión diastólica (DBP) extraídas de las señales ABP, la matriz_picos_sistolicos contiene los índices de los picos sistólicos detectados en cada ventana de señal ABP, y la matriz_picos_diastolicos contiene los índices de los picos diastólicos detectados en cada ventana de señal ABP.
    """    
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
            senial = abp_signals[i][j]
            max_val = np.max(senial)
            min_val = np.min(senial)
            rango = max_val - min_val

            prominence = 0.2 * rango
            height_sbp = min_val + 0.6 * rango
            height_dbp = min_val + 0.2 * rango
            distancia_min = int(0.3 * fs)

            # Detectar picos
            peakss, _ = find_peaks(senial, height=height_sbp, prominence=prominence, distance=distancia_min)
            peaksd, _ = find_peaks(-senial, height=-height_dbp, prominence=prominence, distance=distancia_min)

            picos_sistolicos.append(peakss)
            picos_diastolicos.append(peaksd)
                
            #ps = np.max(senial[peakss])
            #pd = np.min(senial[peaksd])

            if len(peakss) > 0 and len(peaksd) > 0:
                ps_val = np.max(senial[peakss])
                pd_val = np.min(senial[peaksd])

                if ps_val > pd_val:  # Solo si tiene sentido fisiológico
                    ps = ps_val
                    pd = pd_val
                else:
                    print(f"Señal inválida (SBP ≤ DBP) en paciente {i}, ventana {j}: SBP={ps_val:.2f}, DBP={pd_val:.2f}")
                    ps = np.nan
                    pd = np.nan
            else:
                # pocos picos, probablemente ruido
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

def delete_signals_no_peaks(ppg, abp, ecg, presiones_sbp, presiones_dbp, indices_sistolicos, indices_diastolicos):
    """_summary_ Elimina las ventanas de señal que no tienen picos detectados en la señal de presión arterial invasiva (ABP) o que tienen etiquetas de presión sistólica (SBP) y diastólica (DBP) no válidas. La función recorre cada ventana de señal, verificando si se han detectado picos válidos en la señal ABP y si las etiquetas de presión sistólica y diastólica cumplen con criterios fisiológicos. Si una ventana no cumple con estos criterios, se elimina de las matrices de señales y etiquetas correspondientes. Al final del proceso, se devuelve un conjunto filtrado de señales y etiquetas que solo incluye aquellas ventanas que tienen picos válidos y etiquetas de presión arterial coherentes.

    Args:
        ppg (_type_): _description_ Lista de listas con las señales de fotopletismografía (PPG) organizadas por paciente y ventana, que se filtrarán para eliminar aquellas ventanas que no tienen picos detectados en la señal de presión arterial invasiva (ABP) o que tienen etiquetas de presión sistólica (SBP) y diastólica (DBP) no válidas.
        abp (_type_): _description_ Lista de listas con las señales de presión arterial invasiva (ABP) organizadas por paciente y ventana, que se utilizarán para extraer las etiquetas de presión sistólica (SBP) y diastólica (DBP).
        ecg (_type_): _description_ Lista de listas con las señales de electrocardiograma (ECG) organizadas por paciente y ventana, que se utilizarán para el análisis cardíaco.
        presiones_sbp (_type_): _description_ Matriz con las etiquetas de presión sistólica (SBP) extraídas de las señales ABP.
        presiones_dbp (_type_): _description_ Matriz con las etiquetas de presión diastólica (DBP) extraídas de las señales ABP.
        indices_sistolicos (_type_): _description_ Matriz con los índices de los picos sistólicos detectados en cada ventana de señal ABP.
        indices_diastolicos (_type_): _description_ Matriz con los índices de los picos diastólicos detectados en cada ventana de señal ABP.

    Returns:
        _type_: _description_ Retorna un conjunto filtrado de señales y etiquetas que solo incluye aquellas ventanas que tienen picos válidos en la señal de presión arterial invasiva (ABP) y etiquetas de presión sistólica (SBP) y diastólica (DBP) coherentes, eliminando aquellas ventanas que no cumplen con estos criterios.
    """    
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

# --- CARGA Y ALMACENAMIENTO DE DATOS ---

def leer_archivos_mat(ruta_archivo):
    """_summary_ Lee un archivo MAT utilizando la biblioteca h5py, extrae las señales de fotopletismografía (PPG), presión arterial invasiva (ABP) y electrocardiograma (ECG) organizadas en un formato específico, y devuelve estas señales como listas de arrays. La función abre el archivo MAT en modo lectura, identifica las claves presentes en el archivo para localizar el dataset que contiene las referencias a las señales, y luego recorre cada referencia para extraer las señales correspondientes a PPG, ABP y ECG. Las señales extraídas se almacenan en listas separadas y se devuelven al final del proceso.

    Args:
        ruta_archivo (_type_): _description_ Ruta del archivo MAT que se desea leer y del cual se extraerán las señales de fotopletismografía (PPG), presión arterial invasiva (ABP) y electrocardiograma (ECG).

    Returns:
        _type_: _description_ Retorna tres listas de arrays: ppg_signal, abp_signal y ecg_signal. La lista ppg_signal contiene las señales de fotopletismografía (PPG) extraídas del archivo MAT, la lista abp_signal contiene las señales de presión arterial invasiva (ABP) extraídas del archivo MAT, y la lista ecg_signal contiene las señales de electrocardiograma (ECG) extraídas del archivo MAT.
    """    
    with h5py.File(ruta_archivo, 'r') as f:
        claves = list(f.keys())
        print("Claves encontradas:", claves)

        dataset = f[claves[1]]

        datos_extraidos = []
        for i in range(dataset.shape[0]):
            ref = dataset[i, 0]  # referencia HDF5
            datos_extraidos.append(np.array(f[ref]))

        datos_extraidos = np.array(datos_extraidos, dtype=object)
        
        ppg_signal = [registro[:, 0] for registro in datos_extraidos]
        abp_signal = [registro[:, 1] for registro in datos_extraidos]
        ecg_signal = [registro[:, 2] for registro in datos_extraidos]

    return ppg_signal, abp_signal, ecg_signal

def save_partial_file(ppg_signals, ecg_signals, sbp_labels, dbp_labels,
                      patient_id_inicial, index_inicial, nombre_archivo):
    """_summary_  Guarda las señales de fotopletismografía (PPG) y electrocardiograma (ECG), junto con las etiquetas de presión sistólica (SBP) y diastólica (DBP), en archivos separados utilizando la técnica de memoria mapeada (memmap) para manejar grandes volúmenes de datos sin cargar todo en memoria. La función crea un directorio de salida si no existe, calcula el número total de muestras y la longitud de cada segmento, y luego guarda las señales y etiquetas en archivos memmap separados para datos, etiquetas, IDs de pacientes e índices. Finalmente, se guarda un archivo .pt con las rutas a los archivos memmap y la información sobre el número de muestras y la longitud de los segmentos.

    Args:
        ppg_signals (_type_): _description_ Lista de listas con las señales de fotopletismografía (PPG) organizadas por paciente y ventana, que se guardarán en archivos separados utilizando la técnica de memoria mapeada (memmap).
        ecg_signals (_type_): _description_ Lista de listas con las señales de electrocardiograma (ECG) organizadas por paciente y ventana, que se guardarán en archivos separados utilizando la técnica de memoria mapeada (memmap).
        sbp_labels (_type_): _description_ Lista de listas con las etiquetas de presión sistólica (SBP) organizadas por paciente y ventana, que se guardarán en archivos separados utilizando la técnica de memoria mapeada (memmap).
        dbp_labels (_type_): _description_ Lista de listas con las etiquetas de presión diastólica (DBP) organizadas por paciente y ventana, que se guardarán en archivos separados utilizando la técnica de memoria mapeada (memmap).
        patient_id_inicial (_type_): _description_ ID inicial del paciente para el archivo de salida.
        index_inicial (_type_): _description_ Índice inicial para el archivo de salida.
        nombre_archivo (_type_): _description_ Nombre del archivo de salida.

    Returns:
        _type_: _description_ Retorna el número de muestras guardadas en los archivos memmap, que corresponde a la cantidad de ventanas de señal válidas que se han procesado y almacenado en los archivos separados utilizando la técnica de memoria mapeada (memmap).
    """    
    output_dir = 'data/processed/data_UCI'
    os.makedirs(output_dir, exist_ok=True)

    num_total = sum(len(ppg) for ppg in ppg_signals)
    long_segmento = len(ppg_signals[0][0])  

    data_path = os.path.join(output_dir, f'{nombre_archivo}_data.dat')
    labels_path = os.path.join(output_dir, f'{nombre_archivo}_labels.dat')
    patients_path = os.path.join(output_dir, f'{nombre_archivo}_patients.dat')
    indexs_path = os.path.join(output_dir, f'{nombre_archivo}_indexs.dat')

    data_mmap = np.memmap(data_path, dtype='float32', mode='w+', shape=(num_total, 2, long_segmento))
    labels_mmap = np.memmap(labels_path, dtype='float32', mode='w+', shape=(num_total, 2))
    patients_mmap = np.memmap(patients_path, dtype='int64', mode='w+', shape=(num_total,))
    indexs_mmap = np.memmap(indexs_path, dtype='int64', mode='w+', shape=(num_total,))

    index = 0
    for paciente_id, (ppg_segmentos, ecg_segmentos, sbp_segmentos, dbp_segmentos) in enumerate(
        zip(ppg_signals, ecg_signals, sbp_labels, dbp_labels)
    ):
        for ppg, ecg, sbp, dbp in zip(ppg_segmentos, ecg_segmentos, sbp_segmentos, dbp_segmentos):

            if np.isnan(ppg).any() or np.isnan(ecg).any() or np.isnan(sbp) or np.isnan(dbp):
                continue

            data_mmap[index, 0] = ppg
            data_mmap[index, 1] = ecg
            labels_mmap[index] = [sbp, dbp]
            patients_mmap[index] = paciente_id + patient_id_inicial
            indexs_mmap[index] = index + index_inicial
            index += 1

    data_mmap.flush()
    labels_mmap.flush()
    patients_mmap.flush()
    indexs_mmap.flush()

    torch.save({
        'data_path': data_path,
        'labels_path': labels_path,
        'patients_path': patients_path,
        'indexs_path': indexs_path,
        'num_samples': index,
        'segment_length': long_segmento
    }, os.path.join(output_dir, f'{nombre_archivo}.pt'))

    print(f"{nombre_archivo}.pt guardado con {index} muestras (memmap en disco).")
    return index

def get_num_patientsIDs(signals):
    """ Devuelve la cantidad de IDs de pacientes únicos en el dataset
    Args:
        signals (_type_): Lista de señales segmentadas por paciente, donde cada elemento corresponde a un paciente y contiene sus ventanas segmentadas.

    Returns:
        _type_: _description_ retorna la cantidad de IDs de pacientes únicos en el dataset, que se corresponde con la cantidad de elementos en la lista de señales.
    """
    n_ids=0
    for i in range(len(signals)):
        n_ids=n_ids+1
    return n_ids