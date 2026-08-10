"""
Script de validación estadística de detección de picos
- Importa la función de detección de picos del repo (detectar_picos_ppg)
- Extrae aleatoriamente 250 ventanas de entrada (cada ventana tiene 500 muestras, fs=125)
- Modo 'label': interfaz interactiva con matplotlib.ginput() para etiquetar picos (Ground Truth)
    Guarda progresivamente en ground_truth_picos.json y mantiene sample_indices.json para reanudar
- Modo 'eval': compara detecciones automáticas contra Ground Truth con tolerancia ±6 muestras
    Calcula TP, FP, FN, Sensibilidad (Recall), Precisión (PPV) y F1-score

Uso (ejemplo):
  python validar_picos.py label --input windows.npy
  python validar_picos.py eval --input windows.npy

Notas:
- El argumento --input debe ser un archivo .npy, .npz (clave 'arr_0' o 'windows'), o un .pkl que contenga
  una lista/array de ventanas shape (N,500) o lista de listas. Si no se provee, el usuario puede cargar
  las ventanas en memoria y llamar a las funciones desde otro script.
- Interactividad: al etiquetar, haga clic sobre los picos en la figura y luego presione Enter cuando termine.
"""

import os
import json
import argparse
import numpy as np
import random
import matplotlib.pyplot as plt
from datetime import datetime
from typing import List, Tuple

# Importar la función de detección de picos del repositorio
try:
    from src.utils.Tools.Tools import detectar_picos_ppg
except Exception as e:
    # Fallback: informar y re-raise para que el usuario lo vea
    raise ImportError(f"No se pudo importar detectar_picos_ppg desde src.utils.Tools.Tools: {e}")


def extract_random_windows(windows_or_dataset, n_samples: int = 250, seed: int = 42, channel: str = 'ppg', channel_idx: int = None) -> Tuple[np.ndarray, np.ndarray]:
    """Extrae aleatoriamente n_samples ventanas desde un array de ventanas o desde un objeto tipo Dataset.

    - Si se pasa un ndarray 2D (N, window_len) se toma directamente.
    - Si se pasa un objeto Dataset (implementa __len__ y __getitem__), se muestrean índices y se extraen las ventanas usando dataset[i].
    - Se puede indicar el canal a extraer ('ppg', 'abp', 'ecg') o un índice de canal si el elemento devuelto por dataset[i]
      contiene múltiples señales.

    Returns:
        sampled_windows: np.ndarray shape (m, window_len)
        sampled_indices: np.ndarray shape (m,) índices respecto al dataset plano (0..N-1)
    """
    def _extract_signal_from_item(item):
        # heurísticas para extraer la señal de interés desde dataset[i]
        import numpy as _np
        # si el item es ndarray 1D -> asumir que es la señal
        if isinstance(item, _np.ndarray):
            if item.ndim == 1:
                return item
            # si es (C, L) o (L, C)
            if item.ndim == 2:
                if item.shape[0] in (1, 3) and item.shape[1] != 500:
                    # asumir (C, L)
                    idx_map = {'ppg': 0, 'abp': 1, 'ecg': 2}
                    idx = channel_idx if channel_idx is not None else idx_map.get(channel, 0)
                    return _np.asarray(item[idx])
                if item.shape[1] in (1, 3) and item.shape[0] != 500:
                    # asumir (L, C)
                    idx_map = {'ppg': 0, 'abp': 1, 'ecg': 2}
                    idx = channel_idx if channel_idx is not None else idx_map.get(channel, 0)
                    return _np.asarray(item[:, idx])
                # si una de las dimensiones es 500, devolver la dimensión correcta
                if item.shape[0] == 500:
                    return item
                if item.shape[1] == 500:
                    return item[:, 0] if item.ndim == 2 else item
        # si es lista/tupla
        if isinstance(item, (list, tuple)):
            if channel_idx is not None and 0 <= channel_idx < len(item):
                return _np.asarray(item[channel_idx])
            # heurística: si hay 3 elementos, asumir orden (ppg, abp, ecg)
            if len(item) == 3:
                idx_map = {'ppg': 0, 'abp': 1, 'ecg': 2}
                idx = idx_map.get(channel, 0)
                return _np.asarray(item[idx])
            # si hay 1 elemento anidado
            if len(item) == 1:
                return _np.asarray(item[0])
        # si es dict
        if isinstance(item, dict):
            if channel in item:
                return _np.asarray(item[channel])
            # buscar claves comunes
            for key in ('ppg', 'abp', 'ecg', 'signal', 'signals'):
                if key in item:
                    val = item[key]
                    if isinstance(val, (list, tuple)) and len(val) == 3:
                        idx_map = {'ppg': 0, 'abp': 1, 'ecg': 2}
                        idx = idx_map.get(channel, 0)
                        return _np.asarray(val[idx])
                    return _np.asarray(val)
        # último recurso: intentar convertir a ndarray plano
        try:
            return _np.asarray(item)
        except Exception:
            raise ValueError('No se pudo extraer la señal desde el elemento del dataset. Proporcione channel_idx si corresponde.')

    # Si es ndarray 2D plano
    if isinstance(windows_or_dataset, np.ndarray) and windows_or_dataset.ndim == 2:
        arr = windows_or_dataset
        N = arr.shape[0]
        rng = random.Random(seed)
        n = min(n_samples, N)
        sampled_indices = np.array(rng.sample(range(N), n), dtype=int)
        sampled_windows = arr[sampled_indices]
        return sampled_windows, sampled_indices

    # Si es objeto dataset-like
    if hasattr(windows_or_dataset, '__len__') and hasattr(windows_or_dataset, '__getitem__'):
        dataset = windows_or_dataset
        N = len(dataset)
        rng = random.Random(seed)
        n = min(n_samples, N)
        sampled_indices = np.array(rng.sample(range(N), n), dtype=int)
        sampled_list = []
        for idx in sampled_indices:
            item = dataset[idx]
            sig = _extract_signal_from_item(item)
            sig = np.asarray(sig).squeeze()
            # asegurar 1D
            if sig.ndim != 1:
                sig = sig.reshape(-1)
            sampled_list.append(sig)
        sampled_windows = np.stack(sampled_list, axis=0)
        return sampled_windows, sampled_indices

    # Si es otra estructura iterable (lista de ventanas)
    # intentar aplanar
    try:
        arr = np.concatenate([np.array(x) for x in windows_or_dataset])
        if arr.ndim == 2:
            N = arr.shape[0]
            rng = random.Random(seed)
            n = min(n_samples, N)
            sampled_indices = np.array(rng.sample(range(N), n), dtype=int)
            sampled_windows = arr[sampled_indices]
            return sampled_windows, sampled_indices
    except Exception:
        pass

    raise ValueError('Entrada no reconocida para extracción de ventanas. Pase un ndarray 2D o un objeto Dataset con __len__ y __getitem__.')


def interactive_labeling(windows: np.ndarray, sampled_indices: np.ndarray, output_json: str = 'ground_truth_picos.json', resume: bool = True):
    """Interfaz para etiquetar manualmente picos en cada ventana usando matplotlib.ginput().

    Guarda resultados incrementalmente en output_json. Si resume=True y el archivo existe, reanuda.

    Parameters:
        windows: np.ndarray shape (M, window_len)
        sampled_indices: np.ndarray with original indices (used for tracking)
        output_json: path to save ground truth
        resume: if True, resume from existing file
    """
    if len(windows) == 0:
        print("No hay ventanas para etiquetar.")
        return

    # Load existing gt if present
    gt = {}
    if resume and os.path.exists(output_json):
        try:
            with open(output_json, 'r', encoding='utf-8') as f:
                gt = json.load(f)
            print(f"Se cargó ground truth existente con {len(gt)} ventanas etiquetadas. Se reanudará.")
        except Exception as e:
            print(f"No se pudo leer {output_json}: {e}. Se iniciará un archivo nuevo.")
            gt = {}

    total = len(windows)
    for i in range(total):
        orig_idx = int(sampled_indices[i])
        if str(orig_idx) in gt:
            print(f"Ventana {i+1}/{total} (orig idx {orig_idx}) ya etiquetada. Saltando.")
            continue

        sig = windows[i]
        fig, ax = plt.subplots(figsize=(10, 3))
        ax.plot(np.arange(len(sig)), sig, '-k')
        ax.set_title(f"Ventana {i+1}/{total} - Índice original {orig_idx} - Click en picos reales; luego presione Enter")
        ax.set_xlabel('Muestra')
        ax.set_ylabel('Amplitud')
        ax.grid(True)

        print(f"Etiquetando ventana {i+1}/{total} (original index {orig_idx}). Haga clic en los picos y presione Enter cuando termine.")
        # Mostrar y recoger clicks (n=0 permite cualquier número hasta Enter)
        plt.show(block=False)
        try:
            clicks = plt.ginput(n=0, timeout=0)
        except Exception as e:
            print(f"ginput falló: {e}. Intentando una reintento breve.")
            clicks = plt.ginput(n=0, timeout=0)
        plt.close(fig)

        # Convertir coordenadas x a índices de muestra
        indices = []
        for (x, y) in clicks:
            idx = int(round(x))
            idx = max(0, min(idx, len(sig)-1))
            indices.append(idx)

        indices = sorted(list(set(indices)))  # ordenar y eliminar duplicados

        # Guardar entry usando original index como clave para facilidad de reanudado
        gt[str(orig_idx)] = {
            'window_len': int(len(sig)),
            'peaks': indices,
            'labeled_at': datetime.utcnow().isoformat() + 'Z'
        }

        # Guardado incremental: escribir archivo temporal y renombrar
        tmp_path = output_json + '.tmp'
        try:
            with open(tmp_path, 'w', encoding='utf-8') as f:
                json.dump(gt, f, indent=2, ensure_ascii=False)
            os.replace(tmp_path, output_json)
            print(f"Guardado ground truth parcial para ventana original {orig_idx} (picos: {indices}).")
        except Exception as e:
            print(f"Error al guardar ground truth: {e}")

    print(f"Etiquetado completado. Ground truth guardado en: {output_json} (total ventanas etiquetadas: {len(gt)})")


def evaluate_detection(windows: np.ndarray, sampled_indices: np.ndarray, gt_json: str = 'ground_truth_picos.json', tol_samples: int = 6):
    """Evalúa la detección automática contra el ground truth guardado.

    Criterio: un pico detectado es TP si está a <= tol_samples del pico real más cercano.

    Parameters:
        windows: np.ndarray shape (M, window_len)
        sampled_indices: np.ndarray of original indices matching windows order
        gt_json: path to ground truth JSON (usa keys=original indices)
        tol_samples: integer tolerance en muestras (ej. 6 muestras = 50 ms @125 Hz)
    """
    if not os.path.exists(gt_json):
        raise FileNotFoundError(f"Ground truth file {gt_json} no encontrado. Ejecute el modo 'label' primero.")

    with open(gt_json, 'r', encoding='utf-8') as f:
        gt = json.load(f)

    total_TP = 0
    total_FP = 0
    total_FN = 0

    for i in range(len(windows)):
        orig_idx = str(int(sampled_indices[i]))
        if orig_idx not in gt:
            print(f"Advertencia: ventana original {orig_idx} no encontrada en ground truth. Será ignorada.")
            continue

        sig = windows[i]
        gt_peaks = np.array(gt[orig_idx]['peaks'], dtype=int)

        # Detección automática
        try:
            detected = np.array(detectar_picos_ppg(sig, fs=125), dtype=int)
        except Exception as e:
            print(f"Error al ejecutar detectar_picos_ppg en ventana {orig_idx}: {e}")
            detected = np.array([], dtype=int)

        # Matching: marcar pares únicos. Para cada GT, buscar detectados dentro de tol; asignar el detectado más cercano
        matched_detected = set()
        matched_gt = set()

        for j, g in enumerate(gt_peaks):
            # buscar detectados dentro de [g - tol, g + tol]
            if detected.size == 0:
                continue
            diffs = np.abs(detected - g)
            within = np.where(diffs <= tol_samples)[0]
            if within.size > 0:
                # elegir el detectado más cercano
                k = within[np.argmin(diffs[within])]
                # si ese detectado no está ya emparejado
                if int(k) not in matched_detected:
                    matched_detected.add(int(k))
                    matched_gt.add(int(j))

        TP = len(matched_detected)
        FP = max(0, detected.size - TP)
        FN = max(0, gt_peaks.size - TP)

        total_TP += TP
        total_FP += FP
        total_FN += FN

    # Métricas agregadas
    recall = total_TP / (total_TP + total_FN) if (total_TP + total_FN) > 0 else 0.0
    precision = total_TP / (total_TP + total_FP) if (total_TP + total_FP) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    print("\n--- Resultados de evaluación ---")
    print(f"Verdaderos Positivos (TP): {total_TP}")
    print(f"Falsos Positivos (FP): {total_FP}")
    print(f"Falsos Negativos (FN): {total_FN}")
    print(f"Sensibilidad (Recall): {recall:.4f}")
    print(f"Precisión (PPV): {precision:.4f}")
    print(f"F1-Score: {f1:.4f}")

    return {
        'TP': int(total_TP),
        'FP': int(total_FP),
        'FN': int(total_FN),
        'Recall': float(recall),
        'Precision': float(precision),
        'F1': float(f1)
    }


# Utilidades para cargar archivos comunes (.npy, .npz, .pkl)
def load_windows_from_file(path: str):
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    ext = os.path.splitext(path)[1].lower()
    if ext == '.npy':
        arr = np.load(path, allow_pickle=True)
        return arr
    elif ext == '.npz':
        data = np.load(path, allow_pickle=True)
        # heurística: buscar key 'windows' o 'arr_0' o la primer entrada
        if 'windows' in data:
            return data['windows']
        elif 'arr_0' in data:
            return data['arr_0']
        else:
            # devolver primer elemento
            keys = list(data.keys())
            return data[keys[0]]
    elif ext in ('.pkl', '.pickle'):
        import pickle
        with open(path, 'rb') as f:
            obj = pickle.load(f)
        return obj
    else:
        raise ValueError('Formato de archivo no soportado. Use .npy, .npz o .pkl')


def main():
    parser = argparse.ArgumentParser(description='Validación estadística de detección de picos (Ground Truth y evaluación).')
    parser.add_argument('mode', choices=['label', 'eval'], help="'label' para crear/completar Ground Truth, 'eval' para evaluar detección")
    parser.add_argument('--input', '-i', help='Archivo con ventanas (npy/npz/pkl). Cada ventana debe tener 500 muestras. Si no se usa archivo puede importar y llamar a las funciones desde Python pasando la instancia de Dataset (recomendado).', required=True)
    parser.add_argument('--n', type=int, default=250, help='Número de ventanas aleatorias para extraer y etiquetar (por defecto 250)')
    parser.add_argument('--seed', type=int, default=42, help='Semilla para muestreo aleatorio (reproducibilidad)')
    parser.add_argument('--gt', type=str, default='ground_truth_picos.json', help='Ruta del JSON de ground truth')
    parser.add_argument('--tol', type=int, default=6, help='Tolerancia en muestras para considerar TP (por defecto 6 muestras = 50 ms a 125Hz)')
    parser.add_argument('--channel', type=str, choices=['ppg','abp','ecg'], default='ppg', help='Canal a extraer desde dataset si dataset[i] devuelve múltiples señales (default ppg)')
    parser.add_argument('--channel-idx', type=int, default=None, help='Índice de canal a usar si dataset[i] devuelve una tupla/lista de señales (opcional, anula --channel)')

    args = parser.parse_args()

    # Intentar cargar input como archivo; si es una ruta inexistente, el usuario probablemente llamará a las funciones desde Python con un Dataset
    data = None
    if os.path.exists(args.input):
        data = load_windows_from_file(args.input)
    else:
        raise FileNotFoundError(f"El archivo {args.input} no existe. Para usar un Dataset en memoria, importe el módulo y llame a extract_random_windows desde Python con su instancia de Dataset.")

    # Extraer aplanando si es necesario
    windows_flat = None
    if isinstance(data, np.ndarray) and data.ndim == 2:
        windows_flat = data
    else:
        # intentar aplanar estructura tipo pacientes->ventanas
        try:
            # handles list of lists or ndarray of arrays
            windows_flat = np.concatenate([np.array(x) for x in data])
        except Exception:
            windows_flat = np.array(data)

    # Validación básica de tamaño de ventana
    if windows_flat.ndim != 2:
        raise ValueError('No se pudo interpretar las ventanas como un arreglo 2D (N, window_len)')
    if windows_flat.shape[1] != 500:
        print(f"Advertencia: las ventanas tienen longitud {windows_flat.shape[1]} (se esperaba 500). Continuando de todas formas.")

    # Archivo para indices de muestra usados en el muestreo (permite reanudar con las mismas ventanas)
    sample_indices_path = 'sample_indices.json'

    if args.mode == 'label':
        # Si ya existe sample_indices.json, cargarlo para mantener consistencia
        if os.path.exists(sample_indices_path):
            with open(sample_indices_path, 'r', encoding='utf-8') as f:
                sample_info = json.load(f)
            if str(args.n) in sample_info and str(args.seed) in sample_info:
                samp = np.array(sample_info.get('indices', []), dtype=int)
                print(f"Se encontró {sample_indices_path}. Usando índices muestreados previamente ({len(samp)} ventanas).")
                sampled_windows = windows_flat[samp]
                sampled_indices = samp
            else:
                sampled_windows, sampled_indices = extract_random_windows(windows_flat, n_samples=args.n, seed=args.seed, channel=args.channel, channel_idx=args.channel_idx)
                with open(sample_indices_path, 'w', encoding='utf-8') as f:
                    json.dump({'n': int(args.n), 'seed': int(args.seed), 'indices': sampled_indices.tolist()}, f, indent=2)
        else:
            sampled_windows, sampled_indices = extract_random_windows(windows_flat, n_samples=args.n, seed=args.seed, channel=args.channel, channel_idx=args.channel_idx)
            with open(sample_indices_path, 'w', encoding='utf-8') as f:
                json.dump({'n': int(args.n), 'seed': int(args.seed), 'indices': sampled_indices.tolist()}, f, indent=2)

        # Lanzar la interface interactiva
        interactive_labeling(sampled_windows, sampled_indices, output_json=args.gt, resume=True)

    elif args.mode == 'eval':
        # Cargar sample indices
        if not os.path.exists(sample_indices_path):
            raise FileNotFoundError(f"No se encontró {sample_indices_path}. Asegúrese de haber ejecutado el modo 'label' primero.")
        with open(sample_indices_path, 'r', encoding='utf-8') as f:
            sample_info = json.load(f)
        sampled_indices = np.array(sample_info['indices'], dtype=int)
        sampled_windows = windows_flat[sampled_indices]

        evaluate_detection(sampled_windows, sampled_indices, gt_json=args.gt, tol_samples=args.tol)


if __name__ == '__main__':
    # Identity per developer instruction
    print('Soy un asistente AI usando Copilot CLI runtime en VS Code.')
    main()
