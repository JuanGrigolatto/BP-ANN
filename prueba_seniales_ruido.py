import numpy as np
import torch
import matplotlib.pyplot as plt
import random
import os

def white_noise_torch(signal: torch.Tensor, snr_db: float = 20) -> torch.Tensor:
    # Potencia de la señal
    potencia_senal = torch.mean(signal ** 2)

    # Convertir SNR de dB a escala lineal
    snr_lineal = 10 ** (snr_db / 10)

    # Calcular la potencia del ruido deseada
    potencia_ruido = potencia_senal / snr_lineal

    # Generar ruido blanco gaussiano
    ruido = torch.randn_like(signal) * torch.sqrt(potencia_ruido)

    senal_con_ruido = signal + ruido

    return senal_con_ruido

def aplicar_ruido_por_partes():
    archivos = [
        'data_UCI/dataset_parte_1.pt',
        'data_UCI/dataset_parte_2.pt',
        'data_UCI/dataset_parte_3.pt',
        'data_UCI/dataset_parte_4.pt',
    ]

    indices_errores = np.load('indices_errores.npy')
    print("indices errores: ", indices_errores)

    for archivo in archivos:
        data = torch.load(archivo)
        senales = data['data']
        etiquetas = data['labels']
        IDs = data['patient_ids']
        indices_globales = data['index']

        # Convertir a numpy para filtrado
        indices_globales_np = indices_globales.numpy()

        # Filtrar cuáles de los errores pertenecen a este archivo
        mascara = np.isin(indices_globales_np, indices_errores)
        indices_locales = np.where(mascara)[0]  # Posiciones dentro de este archivo

        if len(indices_locales) > 0:
            senales[indices_locales] = white_noise_torch(
                senales[indices_locales],
                snr_db=20
            )

        # Guardar nuevo archivo 
        base_name = os.path.splitext(os.path.basename(archivo))[0]
        nuevo_nombre = f"data_UCI/{base_name}_ruido.pt"

        torch.save({
            'data': senales,
            'labels': etiquetas,
            'patient_ids': IDs,
            'index': indices_globales
        }, nuevo_nombre)

        print(f"Guardado: {nuevo_nombre}")

def main():
    
    #path = 'data_UCI/dataset_parte_1.pt'
    
    """
    signals = data['data']
    labels = data['labels']
    patient_ids = data['patient_ids']
    indices_muestras= data['index'] 
    """



    """
    signals_con_ruido = []
    for signal in signals:
        signal_con_ruido = white_noise(signal.numpy(), snr_db=20)
        signals_con_ruido.append(signal_con_ruido)
    
    # Selección de las señales a plotear
    signals_con_ruido = np.array(signals_con_ruido)
    """
    """
    total_signals = signals_con_ruido.shape[0]
    selected_indices = random.sample(range(total_signals), k=10000)  # sin reemplazo

    signals_subset = signals_con_ruido[selected_indices]
    labels_subset = labels[selected_indices]
    ids_subset = patient_ids[selected_indices]
    """
    """
    #signals_for_plot = signals_con_ruido[:10]  # Selecciona las primeras 10 señales

    #Ploteo de señales con ruido (aleatorio)
    signals_for_plot = random.choices(senales, k=5) # Selecciona 5 señales aleatorias   
    signals_for_plot = signals_for_plot.numpy()
  
    for  signal in signals_for_plot:
           
        senial = signal
        ppg = senial[0]
        ecg = senial[1]

        plt.figure(figsize=(10, 4))
        plt.subplot(2, 1, 1)
        plt.plot(ecg, label='ECG', color='orange')
        plt.ylabel('Amplitud')
        plt.title('ECG')

        plt.subplot(2, 1, 2)
        plt.plot(ppg, label='PPG', color='blue')
        plt.xlabel('Muestras')
        plt.ylabel('Amplitud')
        plt.title('PPG')

        plt.tight_layout()
        plt.show()
"""
if __name__ == '__main__':
    aplicar_ruido_por_partes()