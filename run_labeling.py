import sys
import os
from pathlib import Path

# Asegurar que el repo esté en sys.path
repo_root = Path(__file__).resolve().parent
sys.path.insert(0, str(repo_root))

import numpy as np

try:
    from src.utils.Tools.Tools import leer_archivos_mat
    from validar_picos import extract_random_windows, interactive_labeling
except Exception as e:
    print('Error al importar módulos necesarios:', e)
    raise

MAT_DIR = r"C:\PA-ANN\BP-ANN-clean\BP-ANN\data\raw\datos"

def main():
    mat_dir = Path(MAT_DIR)
    if not mat_dir.exists():
        raise FileNotFoundError(f"No existe directorio: {mat_dir}")

    ppg_all = []
    files = sorted([p for p in mat_dir.iterdir() if p.suffix.lower() == '.mat'])
    print('Archivos encontrados:', files)
    for f in files:
        print('Cargando', f)
        ppg, abp, ecg = leer_archivos_mat(str(f))
        print(f'  Señales extraídas: ppg={len(ppg)}, abp={len(abp)}, ecg={len(ecg)}')
        # ppg is list of 1D arrays
        ppg_all.extend(ppg)

    print('Total ventanas PPG recolectadas:', len(ppg_all))
    if len(ppg_all) == 0:
        raise RuntimeError('No se encontraron ventanas PPG en los archivos .mat')

    # intentar convertir a ndarray 2D, ajustando cada ventana a 500 muestras usando adjust_window si es necesario
    try:
        from src.utils.Tools.Tools import adjust_window
    except Exception:
        def adjust_window(win, max_len):
            import numpy as _np
            win = _np.asarray(win).squeeze()
            diff = max_len - len(win)
            if diff > 0:
                pad_left = diff // 2
                pad_right = diff - pad_left
                return _np.pad(win, (pad_left, pad_right), mode='constant')
            elif diff < 0:
                center = len(win) // 2
                start_cut = center - max_len // 2
                return win[start_cut:start_cut + max_len]
            else:
                return win

    fixed_windows = []
    for idx, x in enumerate(ppg_all):
        arr_x = np.asarray(x).squeeze()
        arr_x = adjust_window(arr_x, 500)
        fixed_windows.append(arr_x)

    windows_arr = np.stack(fixed_windows, axis=0)

    print('Array de ventanas shape=', windows_arr.shape)

    sampled_windows, sampled_indices = extract_random_windows(windows_arr, n_samples=250, seed=42, channel='ppg')
    print('Muestreo completado, ventanas muestreadas:', sampled_windows.shape)

    # Guardar backup por si la GUI falla
    np.save('sampled_windows_backup.npy', sampled_windows)
    np.save('sampled_indices_backup.npy', sampled_indices)
    print('Backups guardados: sampled_windows_backup.npy, sampled_indices_backup.npy')

    # Lanzar etiquetado interactivo
    interactive_labeling(sampled_windows, sampled_indices, output_json='ground_truth_picos.json', resume=True)

if __name__ == '__main__':
    main()
