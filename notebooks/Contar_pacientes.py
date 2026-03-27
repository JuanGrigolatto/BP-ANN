# -*- coding: utf-8 -*-
"""
Script para cargar datos procesados por picos y contar pacientes
"""

import torch
import numpy as np
import os

# Rutas de los archivos
data_paths = [
    'data/processed/data_UCI/dataset_parte_1_por_picos.pt',
    'data/processed/data_UCI/dataset_parte_2_por_picos.pt',
    'data/processed/data_UCI/dataset_parte_3_por_picos.pt',
    'data/processed/data_UCI/dataset_parte_4_por_picos.pt',
]

def contar_pacientes_en_archivos(data_paths):
    """
    Carga los archivos .pt y cuenta pacientes únicos
    """
    
    total_instancias = 0
    todos_los_pacientes = set()
    
    print("=" * 60)
    print("CARGANDO DATOS POR PICOS")
    print("=" * 60)
    
    for idx, ruta in enumerate(data_paths, 1):
        if not os.path.exists(ruta):
            print(f"��� Archivo no encontrado: {ruta}")
            continue
        
        try:
            # Cargar metadata
            meta = torch.load(ruta, map_location="cpu")
            
            # Información del archivo
            num_samples = meta['num_samples']
            segment_length = meta['segment_length']
            
            print(f"\n📁 Parte {idx}: {os.path.basename(ruta)}")
            print(f"   • Instancias: {num_samples}")
            print(f"   • Largo de segmento: {segment_length}")
            
            # Cargar IDs de pacientes
            patients_path = meta['patients_path']
            if os.path.exists(patients_path):
                patients_array = np.memmap(patients_path, dtype='int64', mode='r', 
                                          shape=(num_samples,))
                pacientes_unicos_parte = set(patients_array)
                todos_los_pacientes.update(pacientes_unicos_parte)
                
                print(f"   • Pacientes únicos en esta parte: {len(pacientes_unicos_parte)}")
                print(f"   • IDs de pacientes: {sorted(pacientes_unicos_parte)}")
            else:
                print(f"   ⚠️ Archivo de pacientes no encontrado: {patients_path}")
            
            total_instancias += num_samples
            
        except Exception as e:
            print(f"❌ Error cargando {ruta}: {e}")
    
    # RESUMEN FINAL
    print("\n" + "=" * 60)
    print("RESUMEN FINAL")
    print("=" * 60)
    print(f"✓ Total de INSTANCIAS (segmentos): {total_instancias}")
    print(f"✓ Total de PACIENTES ÚNICOS: {len(todos_los_pacientes)}")
    print(f"✓ IDs de pacientes: {sorted(todos_los_pacientes)}")
    print("=" * 60)
    
    return total_instancias, todos_los_pacientes

# Ejecutar
if __name__ == '__main__':
    total_inst, pacientes = contar_pacientes_en_archivos(data_paths)