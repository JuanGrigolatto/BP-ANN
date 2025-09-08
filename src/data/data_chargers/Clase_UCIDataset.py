import torch
import torch.utils.data as data
import numpy as np
import os
"""
class UCIDataset(data.Dataset):
    def __init__(self, file_list):
        
        #file_list: lista de rutas a archivos .pt que contienen:
        #    - 'data': tensor (num_muestras, canales, long_signal)
        #    - 'labels': tensor (num_muestras, 2)
        #    - 'patient_ids': tensor (num_muestras,)
        #    - 'index': tensor (num_muestras,)
        
        self.file_list = file_list
        self.datasets = []  # aquí guardaremos cada archivo cargado en RAM una sola vez
        self.index_map = [] # lista de (file_idx, sample_idx)

        for file_idx, file_path in enumerate(file_list):
            
            data_dict = torch.load(file_path, map_location='cpu')
            self.datasets.append(data_dict)  # guardamos el archivo en memoria (no concatenamos)

            num_samples = data_dict['data'].shape[0]
            for sample_idx in range(num_samples):
                self.index_map.append((file_idx, sample_idx))

        print(f"Datos indexados correctamente. Muestras totales: {len(self.index_map)}")
        print(f"Archivos cargados: {len(self.datasets)}")

    def __len__(self):
        return len(self.index_map)

    def __getitem__(self, index):
        file_idx, sample_idx = self.index_map[index]
        data_dict = self.datasets[file_idx]

        x = data_dict['data'][sample_idx]
        y = data_dict['labels'][sample_idx]
        pid = data_dict['patient_ids'][sample_idx]
        idx = data_dict['index'][sample_idx]

        return x, y, pid, idx

"""

class UCIDataset(data.Dataset):
    def __init__(self, file_list):
        
        #file_list: lista de rutas a archivos .pt que contienen solo metadatos:
        #    - 'data_path', 'labels_path', 'patients_path', 'indexs_path'
        #    - 'num_samples'
        #    - 'segment_length'
        #Los datos reales están en .dat (np.memmap).
        
        self.file_list = file_list
        self.datasets = []   # guardamos diccionario con memmaps y metadata
        self.index_map = []  # lista de (file_idx, sample_idx)

        for file_idx, file_path in enumerate(file_list):
            meta = torch.load(file_path, map_location="cpu")

            # Abrir memmaps en modo lectura
            data_mmap = np.memmap(meta['data_path'], dtype='float32', mode='r',
                                  shape=(meta['num_samples'], 2, meta['segment_length']))
            labels_mmap = np.memmap(meta['labels_path'], dtype='float32', mode='r',
                                    shape=(meta['num_samples'], 2))
            patients_mmap = np.memmap(meta['patients_path'], dtype='int64', mode='r',
                                      shape=(meta['num_samples'],))
            indexs_mmap = np.memmap(meta['indexs_path'], dtype='int64', mode='r',
                                    shape=(meta['num_samples'],))

            self.datasets.append({
                "data": data_mmap,
                "labels": labels_mmap,
                "patients": patients_mmap,
                "indexs": indexs_mmap,
                "num_samples": meta['num_samples']
            })

            # Generamos el mapa de índices global
            for sample_idx in range(meta['num_samples']):
                self.index_map.append((file_idx, sample_idx))

        print(f"Datos indexados correctamente. Muestras totales: {len(self.index_map)}")
        print(f"Archivos cargados: {len(self.datasets)}")

    def __len__(self):
        return len(self.index_map)

    def __getitem__(self, index):
        file_idx, sample_idx = self.index_map[index]
        dataset = self.datasets[file_idx]

        # Acceso directo al memmap
        x = torch.from_numpy(dataset["data"][sample_idx].copy())
        y = torch.from_numpy(dataset["labels"][sample_idx].copy())
        pid = torch.tensor(dataset["patients"][sample_idx])      # paciente_id
        idx = torch.tensor(dataset["indexs"][sample_idx])        # index global

        return x, y, pid, idx


