import heartpy as hp
import matplotlib.pyplot as plt
import torch

archivo = 'data_UCI/dataset_parte_1.pt'
seniales = []
data = torch.load(archivo)
seniales = data['data']
print(seniales.shape)

ppg = seniales[:, 0, :250].numpy()
ecg = seniales[:, 1, :250].numpy() 

""""
plt.figure(figsize=(12,4))
plt.plot(ppg[0], label='PPG', color='blue')
plt.show()

plt.figure(figsize=(12,4))
plt.plot(ecg[0], label='ECG', color='red')
plt.show()
"""

# Detectar picos en la señal 
wd, m = hp.process(ppg[0], sample_rate = 125.0)

plt.figure(figsize=(12,4))

hp.plotter(wd, m)

plt.show()

for measure in m.keys():
    print('%s: %f' %(measure, m[measure]))

#Segmentación 