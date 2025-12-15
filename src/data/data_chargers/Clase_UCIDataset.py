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
        self.file_list = file_list
        self.datasets_meta = [] 
        self.index_map = []
        
        # IMPORTANTE: Inicialmente cerrado. 
        # Esto evita que Windows intente serializar archivos abiertos y crashee.
        self.open_files = None 
        
        # Ajusta esta ruta base según donde esté tu script respecto a la raíz del proyecto
        ruta_base = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))

        for file_idx, file_path in enumerate(file_list):
            ruta_completa_pt = os.path.join(ruta_base, file_path)
            
            # Cargar metadata en CPU es seguro y evita errores de CUDA en multiproceso
            meta = torch.load(ruta_completa_pt, map_location="cpu")
            
            data_path = os.path.join(ruta_base, meta['data_path'])
            labels_path = os.path.join(ruta_base, meta['labels_path'])
            patients_path = os.path.join(ruta_base, meta['patients_path'])
            indexs_path = os.path.join(ruta_base, meta['indexs_path'])

            self.datasets_meta.append({
                "data_path": data_path,
                "labels_path": labels_path,
                "patients_path": patients_path,
                "indexs_path": indexs_path,
                "num_samples": meta['num_samples'],
                "segment_length": meta['segment_length']
            })
            
            # Indexado global
            for sample_idx in range(meta['num_samples']):
                self.index_map.append((file_idx, sample_idx))

        print(f"Datos indexados correctamente. Muestras totales: {len(self.index_map)}")

    # --- MAGIC METHODS PARA ARREGLAR PICKLE EN WINDOWS ---
    def __getstate__(self):
        """
        Se ejecuta automáticamente al enviar el dataset a los workers.
        Eliminamos los punteros de archivos abiertos del estado a serializar.
        """
        state = self.__dict__.copy()
        state['open_files'] = None 
        return state

    def __setstate__(self, state):
        """Restaurar el estado en el worker (llega con open_files=None)."""
        self.__dict__.update(state)
    # -----------------------------------------------------

    def worker_init(self):
        """
        Abre los archivos memmap en modo lectura persistente.
        Se llama dentro del worker_init_fn o lazy load.
        """
        self.open_files = []
        for meta in self.datasets_meta:
            # mode='r' permite lectura concurrente sin bloqueos
            self.open_files.append({
                "data": np.memmap(meta["data_path"], dtype='float32', mode='r', 
                                  shape=(meta["num_samples"], 2, meta["segment_length"])),
                "labels": np.memmap(meta["labels_path"], dtype='float32', mode='r', 
                                    shape=(meta["num_samples"], 2)),
                "patients": np.memmap(meta["patients_path"], dtype='int64', mode='r', 
                                      shape=(meta["num_samples"],)),
                "indexs": np.memmap(meta["indexs_path"], dtype='int64', mode='r', 
                                    shape=(meta["num_samples"],))
            })

    def __len__(self):
        return len(self.index_map)
    
    def __getitem__(self, index):
        # Fallback de seguridad: Si intentas acceder (ej: en main) y no están abiertos,
        # los abre al vuelo.
        if self.open_files is None:
            self.worker_init()

        file_idx, sample_idx = self.index_map[index]
        files = self.open_files[file_idx]

        # IMPORTANTE: .copy() es vital. 
        # Desvincula la memoria del archivo mapeado para que PyTorch pueda hacer lo suyo
        # sin errores de buffer o concurrencia.
        x = torch.from_numpy(files["data"][sample_idx].copy())
        y = torch.from_numpy(files["labels"][sample_idx].copy())
        pid = torch.tensor(files["patients"][sample_idx])
        idx = torch.tensor(files["indexs"][sample_idx])
        
        return x, y, pid, idx