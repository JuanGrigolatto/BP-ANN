"""
Prepare VitalDB AAMI Test Subset for validation using existing filter and detector.
Does NOT perform matching or final validation. Saves intermediate detection outputs.

Behavior:
- Loads Subset.Signals from the provided .mat (v7.3/HDF5) file
- Assumes ABP is channel index 2 (0-based) in Signals[samples, channels, segments]
- Uses non-overlapping windows: window_size=500 samples (4 s at fs=125)
  starting at 0 with step=500 (remainder samples at end of 1250 may be ignored)
- Calls filtrar_abp() and detectar_picos_abp() from Tools.tools (or src.utils.Tools.Tools)
- Saves per-window detection outputs to validation/results/intermediate/ as a JSON summary and a pickle of raw detector outputs

Usage:
    python validation/prepare_vital_for_validation.py --data_mat "C:\path\to\VitalDB_AAMI_Test_Subset.mat"

"""

import os
import sys
import argparse
import json
import pickle
from pathlib import Path

import h5py
import numpy as np

# Ensure repo root is in sys.path so local modules can be imported
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

# Try importing the user's filter/detector from known locations
try:
    # wrapper created to allow import Tools.tools
    from Tools.tools import filtrar_abp, detectar_picos_abp
except Exception:
    try:
        # fallback to direct package path
        from src.utils.Tools.Tools import filtrar_abp, detectar_picos_abp
    except Exception:
        # Final fallback: import by file path (robust even if packages lack __init__)
        import importlib.util
        tools_path = ROOT / 'src' / 'utils' / 'Tools' / 'Tools.py'
        if not tools_path.exists():
            print('ERROR: Could not find Tools.py at expected location:', tools_path)
            raise FileNotFoundError(tools_path)
        spec = importlib.util.spec_from_file_location('project_tools', str(tools_path))
        project_tools = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(project_tools)
        try:
            filtrar_abp = project_tools.filtrar_abp
            detectar_picos_abp = project_tools.detectar_picos_abp
        except AttributeError as e:
            print('ERROR: Tools.py does not expose expected functions:', e)
            raise


def robust_call_filtrar(abp_window, fs=125):
    try:
        return filtrar_abp(abp_window)
    except TypeError:
        # maybe signature requires fs
        return filtrar_abp(abp_window, fs)


def robust_call_detector(filtered_window, fs=125):
    # Call detector and return raw output; do not interpret here beyond basic normalization
    try:
        return detectar_picos_abp(filtered_window, fs=fs)
    except TypeError:
        try:
            return detectar_picos_abp(filtered_window)
        except Exception as e:
            # re-raise with context
            raise


def main(args):
    p = Path(args.data_mat)
    if not p.exists():
        raise FileNotFoundError(p)

    out_dir = Path('validation') / 'results' / 'intermediate'
    out_dir.mkdir(parents=True, exist_ok=True)

    print('Loading', p)
    with h5py.File(str(p), 'r') as f:
        if 'Subset' not in f:
            raise KeyError('Subset group not found in MAT file')
        sub = f['Subset']
        if 'Signals' not in sub:
            raise KeyError('Signals dataset not found in Subset')
        Signals = np.array(sub['Signals'])
        SBP = np.array(sub['SBP']).reshape(-1) if 'SBP' in sub else None
        DBP = np.array(sub['DBP']).reshape(-1) if 'DBP' in sub else None

    samples, channels, n_segments = Signals.shape
    print('Signals shape (samples, channels, segments):', Signals.shape)
    print('n_segments:', n_segments)

    abp_chan = 2
    fs = args.fs
    window_size = args.window_size  # 500
    step = args.step  # 500

    summary = []
    raw_store = {}

    total_windows = 0
    total_detected_peaks = 0

    for seg in range(n_segments):
        abp = Signals[:, abp_chan, seg].astype(float)
        seg_sbp = float(SBP[seg]) if SBP is not None else None
        seg_dbp = float(DBP[seg]) if DBP is not None else None

        windows = []
        starts = list(range(0, samples - window_size + 1, step))
        if len(starts) == 0:
            print(f'Segment {seg} shorter than window_size; skipping')
            continue

        for w_idx, start in enumerate(starts):
            end = start + window_size
            win = abp[start:end]
            # Filter
            filtered = robust_call_filtrar(win, fs=fs)
            # Detect
            det_raw = robust_call_detector(filtered, fs=fs)

            # Try to normalize detector output metadata
            # We don't force a specific interpretation; store raw and also any indices if we can find them
            sbp_indices_rel = None
            dbp_indices_rel = None
            sbp_values = None
            dbp_values = None

            # If detector returned a dict, look for common keys
            if isinstance(det_raw, dict):
                for k in det_raw.keys():
                    lk = k.lower()
                    if 'sbp' in lk and ('idx' in lk or 'index' in lk or 'loc' in lk):
                        sbp_indices_rel = np.array(det_raw[k]).astype(int)
                    if 'dbp' in lk and ('idx' in lk or 'index' in lk or 'loc' in lk):
                        dbp_indices_rel = np.array(det_raw[k]).astype(int)
            # If returned tuple/list, try to unpack
            elif isinstance(det_raw, (list, tuple)):
                if len(det_raw) >= 1:
                    try:
                        sbp_indices_rel = np.array(det_raw[0]).astype(int)
                    except Exception:
                        pass
                if len(det_raw) >= 2:
                    try:
                        dbp_indices_rel = np.array(det_raw[1]).astype(int)
                    except Exception:
                        pass
            # If ndarray, assume indices
            elif isinstance(det_raw, np.ndarray):
                try:
                    sbp_indices_rel = det_raw.astype(int)
                except Exception:
                    pass

            # If we have relative indices, compute absolute indices within segment
            sbp_indices_abs = None
            dbp_indices_abs = None
            if sbp_indices_rel is not None:
                sbp_indices_abs = (sbp_indices_rel + start).tolist()
                sbp_values = [float(filtered[int(i)]) for i in sbp_indices_rel]
            if dbp_indices_rel is not None:
                dbp_indices_abs = (dbp_indices_rel + start).tolist()
                dbp_values = [float(filtered[int(i)]) for i in dbp_indices_rel]

            windows.append({
                'segment': int(seg),
                'window_index': int(w_idx),
                'start_sample': int(start),
                'end_sample': int(end),
                'n_sbp_detected': int(len(sbp_indices_rel)) if sbp_indices_rel is not None else None,
                'n_dbp_detected': int(len(dbp_indices_rel)) if dbp_indices_rel is not None else None,
                'sbp_indices_abs': sbp_indices_abs,
                'dbp_indices_abs': dbp_indices_abs,
                'sbp_values': sbp_values,
                'dbp_values': dbp_values,
                'segment_sbp_ref': seg_sbp,
                'segment_dbp_ref': seg_dbp,
            })

            # store raw detector output per window key
            raw_key = f'seg{seg:04d}_w{w_idx:02d}'
            raw_store[raw_key] = det_raw

            total_windows += 1
            if sbp_indices_rel is not None:
                total_detected_peaks += len(sbp_indices_rel)

        # append segment summary
        summary.append({
            'segment': int(seg),
            'n_windows': len(windows),
            'windows': windows,
        })

    # Save summary JSON
    summary_path = out_dir / 'preparation_summary.json'
    with open(summary_path, 'w') as fh:
        json.dump({'fs': fs, 'window_size': window_size, 'step': step, 'segments_processed': len(summary),
                   'total_windows': total_windows, 'total_detected_peaks': total_detected_peaks,
                   'segments': summary}, fh, indent=2)

    # Save raw detector outputs as pickle (may be large)
    raw_path = out_dir / 'raw_detector_outputs.pkl'
    with open(raw_path, 'wb') as fh:
        pickle.dump(raw_store, fh)

    print('Saved preparation summary to', summary_path)
    print('Saved raw detector outputs to', raw_path)
    print('Processed segments:', len(summary), 'Total windows:', total_windows, 'Total SBP detections (where available):', total_detected_peaks)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_mat', required=False,
                        default=r'C:\PA-ANN\BP-ANN-clean\data\raw\VitalDB_AAMI_Test_Subset\VitalDB_AAMI_Test_Subset.mat')
    parser.add_argument('--fs', type=int, default=125)
    parser.add_argument('--window_size', type=int, default=500)
    parser.add_argument('--step', type=int, default=500)
    args = parser.parse_args()
    main(args)
