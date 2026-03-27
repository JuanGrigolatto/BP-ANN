import torch
import torch.utils.data as data
import numpy as np
import os

class UCIDataset(data.Dataset):
    """
    Dataset personalizado para cargar datos de señales de la base UCI usando memory mapping (memmap).
    
    Esta clase está diseñada para manejar datasets más grandes que la memoria RAM disponible,
    cargando los datos bajo demanda desde el disco. Implementa métodos seguros para 
    multiprocesamiento en Windows (__getstate__, __setstate__).
    
    Args:
        file_list (list): Lista de rutas relativas a los archivos de metadata (.pt).
    """
    def __init__(self, file_list):
        self.file_list = file_list
        self.datasets_meta = [] 
        self.index_map = []

        # Inicialmente cerrado para evitar que Windows intente serializar archivos abiertos.
        self.open_files = None 
        
        # Ruta base relativa a este script (ajustar si cambia la estructura de carpetas)
        ruta_base = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))

        for file_idx, file_path in enumerate(file_list):
            ruta_completa_pt = os.path.join(ruta_base, file_path)
            
            # Cargar metadata en CPU para evitar problemas de inicialización de CUDA
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

            # Indexado global para acceder a cualquier muestra como si fuera un solo arreglo
            for sample_idx in range(meta['num_samples']):
                self.index_map.append((file_idx, sample_idx))

        print(f"Datos indexados correctamente. Muestras totales: {len(self.index_map)}")

    def __getstate__(self):
        """
        Define el estado a serializar al enviar el dataset a los workers (PyTorch DataLoader).
        Elimina los punteros de archivos abiertos para evitar errores de Pickle en Windows.
        """
        state = self.__dict__.copy()
        state['open_files'] = None 
        return state

    def __setstate__(self, state):
        """Restaurar el estado en el worker"""
        self.__dict__.update(state)

    def worker_init(self):
        """
        Abre los archivos binarios utilizando numpy memmap en modo solo lectura ('r').
        Esto permite lectura concurrente por múltiples workers sin bloqueos.
        """
        self.open_files = []
        for meta in self.datasets_meta:
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
        """Devuelve el número total de muestras en el dataset concatenado."""
        return len(self.index_map)
    
    def __getitem__(self, index):
        """
        Recupera una muestra específica del disco de forma segura.
        
        Args:
            index (int): Índice global de la muestra.
            
        Returns:
            tuple: (x, y, pid, idx) correspondientes a la señal, etiquetas, ID del paciente y su índice.
        """
        # Lazy loading: si los archivos no están abiertos (ej. worker principal), abrirlos.
        if self.open_files is None:
            self.worker_init()

        file_idx, sample_idx = self.index_map[index]
        files = self.open_files[file_idx]
        # El método .copy() carga explícitamente el chunk en RAM desvinculándolo del archivo memmap
        x = torch.from_numpy(files["data"][sample_idx].copy())
        y = torch.from_numpy(files["labels"][sample_idx].copy())
        pid = torch.tensor(files["patients"][sample_idx])
        idx = torch.tensor(files["indexs"][sample_idx])
        
        return x, y, pid, idx