import numpy as np
import torch
import matplotlib.pyplot as plt
import random

def white_noise(signal, snr_db=20):
    # Potencia de la señal
    potencia_senal = np.mean(signal ** 2)

    # Convertir SNR de dB a escala lineal
    snr_lineal = 10 ** (snr_db / 10)

    # Calcular la potencia del ruido deseada
    potencia_ruido = potencia_senal / snr_lineal

    # Generar ruido blanco gaussiano
    ruido = np.random.normal(0, np.sqrt(potencia_ruido), signal.shape)

    senal_con_ruido = signal + ruido

    return senal_con_ruido

def main():
    path = 'data_UCI/dataset_parte_1.pt'

    data = torch.load(path)

    signals = data['data']
    labels = data['labels']
    patient_ids = data['patient_ids']
    
    print (f"Señales cargadas: {signals.shape}")

    # Aplicación de ruido blanco a las señales
    signals_con_ruido = []
    for signal in signals:
        signal_con_ruido = white_noise(signal.numpy(), snr_db=20)
        signals_con_ruido.append(signal_con_ruido)

    # Selección de las señales a plotear
    signals_con_ruido = np.array(signals_con_ruido)
    """
    total_signals = signals_con_ruido.shape[0]
    selected_indices = random.sample(range(total_signals), k=10000)  # sin reemplazo

    signals_subset = signals_con_ruido[selected_indices]
    labels_subset = labels[selected_indices]
    ids_subset = patient_ids[selected_indices]
    """
    # Guardado en archivo .pt
    save_path = 'data_UCI/dataset_parte_1_ruido.pt'
    torch.save({
        'data': torch.tensor(signals_con_ruido, dtype=torch.float32),
        'labels': labels,
        'patient_ids': patient_ids
    }, save_path)
    print(f"Señales con ruido guardadas en {save_path}")

    signals_con_ruido_set = random.choices(signals_con_ruido, k=10000)  # Selecciona 10,000 señales aleatorias

    #signals_for_plot = signals_con_ruido[:10]  # Selecciona las primeras 10 señales
    signals_for_plot = random.choices(signals_con_ruido, k=5) # Selecciona 5 señales aleatorias   
  
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

if __name__ == '__main__':
    main()