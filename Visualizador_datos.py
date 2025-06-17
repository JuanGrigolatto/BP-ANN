import torch 

import os
import torch
import matplotlib.pyplot as plt

# Configuración
output_dir = 'datos_UCI'
num_files_to_inspect = 20  # Cantidad de archivos a visualizar
start_num_batch = 1 # Número de lote inicial para inspeccionar
# Obtener lista ordenada de archivos
files = sorted([f for f in os.listdir(output_dir) if f.endswith('.pt')])

for i, filename in enumerate(files[start_num_batch:(start_num_batch+num_files_to_inspect)]):
    filepath = os.path.join(output_dir, filename)
    data = torch.load(filepath)
    
    print(f"\n=== Archivo {i+1}/{num_files_to_inspect}: {filename} ===")
    print(f"Paciente ID: {data['patient_id'].item()}")
    print(f"Etiquetas (SBP, DBP): {data['label'].numpy()}")
    print(f"Dimensión señal (canales, longitud): {data['signal'].shape}")