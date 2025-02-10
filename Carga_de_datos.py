# -*- coding: utf-8 -*-
"""
Created on Mon Feb 10 12:52:58 2025

@author: juang
"""

import h5py
import numpy as np
import matplotlib.pyplot as plt

archivo_datos1 = "Part_1.mat"

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
    
    # plt.figure(figsize=(10, 4))
    # plt.plot(ppg_signal[2000][:1000])  # Mostrar los primeros 1000 puntos
    # plt.title("Señal PPG - Primer Registro")
    # plt.xlabel("Muestras")
    # plt.ylabel("Amplitud")
    # plt.show()

    # plt.figure(figsize=(10, 4))
    # plt.plot(abp_signal[2000][:1000])  # Mostrar los primeros 1000 puntos
    # plt.title("Señal ABP - Primer Registro")
    # plt.xlabel("Muestras")
    # plt.ylabel("Amplitud")
    # plt.show()
    
    # plt.figure(figsize=(10, 4))
    # plt.plot(ecg_signal[2000][:1000])  # Mostrar los primeros 1000 puntos
    # plt.title("Señal ECG - Primer Registro")
    # plt.xlabel("Muestras")
    # plt.ylabel("Amplitud")
    # plt.show()
    
    
  