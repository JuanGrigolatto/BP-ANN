import torch.utils.data as data

class Intrapatientset(data.Dataset):
    """
    Dataset derivado que aísla todas las señales de un único paciente.
    
    Permite acceder únicamente a los índices que corresponden a un 
    paciente específico. Ideal para tareas de evaluación intra-paciente o fine-tuning.
    
    Args:
        patient_id (int): Identificador único del paciente.
        base_dataset (torch.utils.data.Dataset): Dataset global que contiene todas las muestras.
        patient_to_indices_map (dict): Diccionario que mapea {patient_id: [lista_de_indices_globales]}.
    """
    def __init__(self, patient_id, base_dataset, patient_to_indices_map):

        self.base_dataset = base_dataset
               
        if patient_id not in patient_to_indices_map:
            raise ValueError(f"ID de paciente {patient_id} no encontrado en el mapa.")
            
        self.signal_indices = patient_to_indices_map[patient_id]
        
        #print(f"Cargado PatientSignalDataset para Paciente {patient_id} con {len(self.signal_indices)} señales.")

    def __len__(self):
        """Devuelve la cantidad de muestras disponibles para este paciente."""
        return len(self.signal_indices)

    def __getitem__(self, index):
        """
        Recupera una muestra específica del paciente.
        
        Args:
            index (int): Índice local (relativo a este paciente, de 0 a len-1, de muestra).
            
        Returns:
            tuple: (signal, label) correspondientes a la muestra solicitada.
        """
        # 1. Mapeo de índice local del paciente al índice global del base_dataset
        global_index = self.signal_indices[index]
        
        # 2. El base_dataset devuelve (x, y, pid, idx). Solo desempaquetamos señal y etiqueta.
        signal, label, _, _ = self.base_dataset[global_index]
        
        return signal, label