"""
Full validation over all segments in VitalDB_AAMI_Test_Subset.mat
Follows exactly the procedure specified by the user.
Saves:
 - validation/results/VitalDB_AAMI/results.csv (one row per segment)
 - validation/results/VitalDB_AAMI/summary.csv (coverage + metrics)
 - validation/results/VitalDB_AAMI/bland_altman_crudo.png (Bland-Altman plots full scale)
 - validation/results/VitalDB_AAMI/bland_altman_acotado.png (Bland-Altman plots zoomed to +/- 30 mmHg)

No temporal validation is performed. Uses filtrar_abp and get_abp_labels from Tools.py without modification.
"""

from pathlib import Path
import h5py
import numpy as np
import math
import statistics
import csv
import sys
import argparse
from scipy.signal import find_peaks
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

try:
    from src.utils.Tools.Tools import filtrar_abp
except Exception:
    import importlib.util
    tools_path = ROOT / 'src' / 'utils' / 'Tools' / 'Tools.py'
    spec = importlib.util.spec_from_file_location('project_tools', str(tools_path))
    project_tools = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(project_tools)
    filtrar_abp = project_tools.filtrar_abp


def calcular_labels_promedio_10s(abp_filtrada, fs):
    """
    Calcula SBP y DBP promedio de todos los latidos detectados
    en una ventana completa de 10 segundos.

    Para cada latido:
        - SBP: valor de la señal en el pico sistólico.
        - DBP: valor de la señal en el valle diastólico.

    Finalmente se calcula el promedio de los valores detectados.
    """

    max_val = np.max(abp_filtrada)
    min_val = np.min(abp_filtrada)
    rango = max_val - min_val

    prominence = 0.2 * rango
    height_sbp = min_val + 0.6 * rango
    height_dbp = min_val + 0.2 * rango
    distancia_min = int(0.3 * fs)

    # Detectar picos sistólicos
    picos_sbp, _ = find_peaks(
        abp_filtrada,
        height=height_sbp,
        prominence=prominence,
        distance=distancia_min
    )

    # Detectar valles diastólicos
    picos_dbp, _ = find_peaks(
        -abp_filtrada,
        height=-height_dbp,
        prominence=prominence,
        distance=distancia_min
    )

    # Valores beat-to-beat
    valores_sbp = abp_filtrada[picos_sbp]
    valores_dbp = abp_filtrada[picos_dbp]

    # Debe existir al menos un SBP y un DBP
    if len(valores_sbp) > 0 and len(valores_dbp) > 0:

        # Promedio de todos los latidos detectados
        sbp_promedio = float(np.mean(valores_sbp))
        dbp_promedio = float(np.mean(valores_dbp))

        # Criterio fisiológico
        if sbp_promedio <= dbp_promedio:
            sbp_promedio = np.nan
            dbp_promedio = np.nan

    else:
        sbp_promedio = np.nan
        dbp_promedio = np.nan

    return (
        sbp_promedio,
        dbp_promedio,
        picos_sbp,
        picos_dbp
    )

_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_DATA_MAT = _ROOT / "data" / "raw" / "VitalDB_AAMI_Test_Subset" / "VitalDB_AAMI_Test_Subset.mat"

_parser = argparse.ArgumentParser(
    description="Validación completa de picos ABP (SBP/DBP) sobre el subset AAMI de VitalDB."
)
_parser.add_argument(
    "--data_mat",
    type=str,
    default=str(_DEFAULT_DATA_MAT),
    help=f"Ruta al .mat de VitalDB_AAMI_Test_Subset. Por defecto: {_DEFAULT_DATA_MAT}",
)
_args, _ = _parser.parse_known_args()
DATA_MAT = Path(_args.data_mat)
OUT_DIR = Path('validation') / 'results' / 'VitalDB_AAMI'
OUT_DIR.mkdir(parents=True, exist_ok=True)

fs = 125

if not DATA_MAT.exists():
    raise FileNotFoundError(
        f"No se encontró el archivo VitalDB en '{DATA_MAT}'.\n"
        "Pasá la ruta correcta con --data_mat o ubicalo en "
        "data/raw/VitalDB_AAMI_Test_Subset/VitalDB_AAMI_Test_Subset.mat"
    )

# Load data
with h5py.File(str(DATA_MAT), 'r') as f:
    sub = f['Subset']
    Signals = np.array(sub['Signals'])
    SBP = np.array(sub['SBP']).reshape(-1)
    DBP = np.array(sub['DBP']).reshape(-1)

samples, channels, total_segments = Signals.shape
print('Loaded Signals shape:', Signals.shape)
# Calcular etiquetas sobre la ventana completa de 10 segundos
labels_calculadas = []

for seg in range(total_segments):

    abp = Signals[:, 2, seg].astype(float)

    # Filtrado de la señal completa de 10 s
    abp_filtrada = filtrar_abp(abp)

    # Detección beat-to-beat y promedio
    sbp_pred, dbp_pred, picos_sbp, valles_dbp = \
        calcular_labels_promedio_10s(abp_filtrada, fs)

    labels_calculadas.append({
        'SBP_pred': sbp_pred,
        'DBP_pred': dbp_pred,
        'picos_sbp': picos_sbp,
        'valles_dbp': valles_dbp
    })

# Prepare per-segment records
rows = []

for i in range(total_segments):
    ref_sbp = float(SBP[i])
    ref_dbp = float(DBP[i])

    sbp_pred = labels_calculadas[i]['SBP_pred']
    dbp_pred = labels_calculadas[i]['DBP_pred']

    # Validación de la predicción
    sbp_valido = (
        not math.isnan(sbp_pred)
        and not math.isinf(sbp_pred)
    )

    dbp_valido = (
        not math.isnan(dbp_pred)
        and not math.isinf(dbp_pred)
    )

    # Cantidad de ventanas válidas.
    # En esta versión hay una única ventana de 10 s.
    cnt_valid_sbp = 1 if sbp_valido else 0
    cnt_valid_dbp = 1 if dbp_valido else 0

    seg_valid_sbp = 1 if sbp_valido else 0
    seg_valid_dbp = 1 if dbp_valido else 0

    rows.append({
        'segment': i,
        'SBP_ref': ref_sbp,
        'SBP_pred': sbp_pred,
        'DBP_ref': ref_dbp,
        'DBP_pred': dbp_pred,

        'cantidad_latidos_SBP': len(
            labels_calculadas[i]['picos_sbp']
        ),
        'cantidad_latidos_DBP': len(
            labels_calculadas[i]['valles_dbp']
        ),

        'cantidad_ventanas_validas_SBP': cnt_valid_sbp,
        'cantidad_ventanas_validas_DBP': cnt_valid_dbp,

        'segmento_valido_SBP': seg_valid_sbp,
        'segmento_valido_DBP': seg_valid_dbp,
    })

# Save results.csv
results_path = OUT_DIR / 'results.csv'
with open(results_path, 'w', newline='') as fh:
    fieldnames = [
    'segment',
    'SBP_ref',
    'SBP_pred',
    'DBP_ref',
    'DBP_pred',
    'cantidad_latidos_SBP',
    'cantidad_latidos_DBP',
    'cantidad_ventanas_validas_SBP',
    'cantidad_ventanas_validas_DBP',
    'segmento_valido_SBP',
    'segmento_valido_DBP'
]
    writer = csv.DictWriter(fh, fieldnames=fieldnames)
    writer.writeheader()
    for r in rows:
        writer.writerow(r)

# Coverage and metrics
total_segments = total_segments
segments_1_valid_sbp = sum(
    1 for r in rows
    if r['cantidad_ventanas_validas_SBP'] == 1
)

segments_0_valid_sbp = sum(
    1 for r in rows
    if r['cantidad_ventanas_validas_SBP'] == 0
)

segments_with_at_least1_sbp = total_segments - segments_0_valid_sbp

coverage_sbp = (
    segments_with_at_least1_sbp / total_segments * 100.0
)

segments_1_valid_dbp = sum(
    1 for r in rows
    if r['cantidad_ventanas_validas_DBP'] == 1
)

segments_0_valid_dbp = sum(
    1 for r in rows
    if r['cantidad_ventanas_validas_DBP'] == 0
)

segments_with_at_least1_dbp = total_segments - segments_0_valid_dbp

coverage_dbp = (
    segments_with_at_least1_dbp / total_segments * 100.0
)

# Metrics: consider segments where SBP_pred is finite
sbp_refs_all = [r['SBP_ref'] for r in rows]
sbp_preds_all = [r['SBP_pred'] for r in rows]

pairs_sbp = [(ref,pred) for ref,pred in zip(sbp_refs_all, sbp_preds_all) if (not math.isnan(pred)) and (not math.isnan(ref))]
errors_sbp = [pred - ref for ref,pred in pairs_sbp]
valid_refs_sbp = [ref for ref,pred in pairs_sbp]
valid_preds_sbp = [pred for ref,pred in pairs_sbp]

if len(errors_sbp)>0:
    MAE_sbp = sum(abs(e) for e in errors_sbp)/len(errors_sbp)
    RMSE_sbp = math.sqrt(sum(e*e for e in errors_sbp)/len(errors_sbp))
    Bias_sbp = sum(errors_sbp)/len(errors_sbp)
    SD_sbp = statistics.pstdev(errors_sbp) if len(errors_sbp)>1 else 0.0
    r_sbp, _ = pearsonr(valid_refs_sbp, valid_preds_sbp) if len(errors_sbp)>1 else (float('nan'), float('nan'))
else:
    MAE_sbp=RMSE_sbp=Bias_sbp=SD_sbp=r_sbp=float('nan')

# DBP
dbp_refs_all = [r['DBP_ref'] for r in rows]
dbp_preds_all = [r['DBP_pred'] for r in rows]

pairs_dbp = [(ref,pred) for ref,pred in zip(dbp_refs_all, dbp_preds_all) if (not math.isnan(pred)) and (not math.isnan(ref))]
errors_dbp = [pred - ref for ref,pred in pairs_dbp]
valid_refs_dbp = [ref for ref,pred in pairs_dbp]
valid_preds_dbp = [pred for ref,pred in pairs_dbp]

if len(errors_dbp)>0:
    MAE_dbp = sum(abs(e) for e in errors_dbp)/len(errors_dbp)
    RMSE_dbp = math.sqrt(sum(e*e for e in errors_dbp)/len(errors_dbp))
    Bias_dbp = sum(errors_dbp)/len(errors_dbp)
    SD_dbp = statistics.pstdev(errors_dbp) if len(errors_dbp)>1 else 0.0
    r_dbp, _ = pearsonr(valid_refs_dbp, valid_preds_dbp) if len(errors_dbp)>1 else (float('nan'), float('nan'))
else:
    MAE_dbp=RMSE_dbp=Bias_dbp=SD_dbp=r_dbp=float('nan')

# Save summary.csv
summary_path = OUT_DIR / 'summary.csv'
with open(summary_path, 'w', newline='') as fh:
    writer = csv.writer(fh)
    writer.writerow(['total_segments', total_segments])
    writer.writerow(['segments_valid_1_SBP', segments_1_valid_sbp])
    writer.writerow(['segments_valid_0_SBP', segments_0_valid_sbp])
    writer.writerow(['coverage_SBP_percent', coverage_sbp])
    writer.writerow(['MAE_SBP', MAE_sbp])
    writer.writerow(['RMSE_SBP', RMSE_sbp])
    writer.writerow(['Bias_SBP', Bias_sbp])
    writer.writerow(['SD_SBP', SD_sbp])
    writer.writerow(['Pearson_r_SBP', r_sbp])
    writer.writerow([])
    writer.writerow(['segments_valid_1_DBP', segments_1_valid_dbp])
    writer.writerow(['segments_valid_0_DBP', segments_0_valid_dbp])
    writer.writerow(['coverage_DBP_percent', coverage_dbp])
    writer.writerow(['MAE_DBP', MAE_dbp])
    writer.writerow(['RMSE_DBP', RMSE_dbp])
    writer.writerow(['Bias_DBP', Bias_dbp])
    writer.writerow(['SD_DBP', SD_dbp])
    writer.writerow(['Pearson_r_DBP', r_dbp])

# --- BLAND-ALTMAN PLOT GENERATION ---
def plot_bland_altman(refs, preds, errors, bias, sd, title, ax, y_limits=None):
    means = (np.array(refs) + np.array(preds)) / 2.0
    errs = np.array(errors)
    
    # Scatter plot
    ax.scatter(means, errs, alpha=0.5, color='tab:blue', edgecolors='none', s=20)
    
    # Reference lines (Bias y Límites de Acuerdo del 95%)
    ax.axhline(bias, color='red', linestyle='-', lw=2, label=f'Sesgo Medio ({bias:.2f})')
    ax.axhline(bias + 1.96 * sd, color='black', linestyle='--', lw=1.5, label=f'+1.96 DE ({bias + 1.96 * sd:.2f})')
    ax.axhline(bias - 1.96 * sd, color='black', linestyle='--', lw=1.5, label=f'-1.96 DE ({bias - 1.96 * sd:.2f})')
    ax.axhline(0, color='gray', linestyle=':', lw=1)
    
    # Optional axis scaling
    if y_limits is not None:
        ax.set_ylim(y_limits)
    
    # Scientific formatting
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.set_xlabel('Presión Media: (Predicción + Referencia) / 2 [mmHg]', fontsize=10)
    ax.set_ylabel('Diferencia: Predicción - Referencia [mmHg]', fontsize=10)
    ax.legend(loc='upper right', fontsize=9, framealpha=0.9)
    ax.grid(True, alpha=0.3)

# 1. Generación de Gráfico Crudo (Autoescalado)
fig_raw, (ax1_raw, ax2_raw) = plt.subplots(1, 2, figsize=(14, 6))

if len(errors_sbp) > 0:
    plot_bland_altman(valid_refs_sbp, valid_preds_sbp, errors_sbp, Bias_sbp, SD_sbp, 
                      'Análisis de Bland-Altman: Presión Arterial Sistólica (SBP)', ax1_raw)
if len(errors_dbp) > 0:
    plot_bland_altman(valid_refs_dbp, valid_preds_dbp, errors_dbp, Bias_dbp, SD_dbp, 
                      'Análisis de Bland-Altman: Presión Arterial Diastólica (DBP)', ax2_raw)

plt.tight_layout()
bland_altman_raw_path = OUT_DIR / 'bland_altman_crudo.png'
plt.savefig(bland_altman_raw_path, dpi=300)
plt.close(fig_raw)

# 2. Generación de Gráfico Acotado (Zoom al rango clínico)
fig_zoom, (ax1_zoom, ax2_zoom) = plt.subplots(1, 2, figsize=(14, 6))

if len(errors_sbp) > 0:
    plot_bland_altman(valid_refs_sbp, valid_preds_sbp, errors_sbp, Bias_sbp, SD_sbp, 
                      'Análisis de Bland-Altman Acotado: SBP', ax1_zoom, y_limits=(-30, 30))
if len(errors_dbp) > 0:
    plot_bland_altman(valid_refs_dbp, valid_preds_dbp, errors_dbp, Bias_dbp, SD_dbp, 
                      'Análisis de Bland-Altman Acotado: DBP', ax2_zoom, y_limits=(-30, 30))

# Nota al pie indicando el ajuste metodológico
fig_zoom.text(0.5, 0.01, 
              'Nota: El eje de ordenadas ha sido restringido a ±30 mmHg para optimizar la visualización de la varianza estructural.\n'
              'Los estadísticos reportados en la leyenda incluyen todos los segmentos procesados, contemplando anomalías extremas.', 
              ha='center', fontsize=9, style='italic', color='dimgray')

plt.tight_layout(rect=[0, 0.05, 1, 1]) # Ajusta el layout para que la nota no se solape
bland_altman_zoom_path = OUT_DIR / 'bland_altman_acotado.png'
plt.savefig(bland_altman_zoom_path, dpi=300)
plt.close(fig_zoom)
# ------------------------------------

# Top 10 segments by abs error SBP and DBP
abs_errors_sbp = []
for idx,(ref,pred) in enumerate(zip(sbp_refs_all,sbp_preds_all)):
    if not math.isnan(pred) and not math.isnan(ref):
        abs_errors_sbp.append((idx, abs(pred-ref), pred-ref))
abs_errors_sbp.sort(key=lambda x: x[1], reverse=True)

abs_errors_dbp = []
for idx,(ref,pred) in enumerate(zip(dbp_refs_all,dbp_preds_all)):
    if not math.isnan(pred) and not math.isnan(ref):
        abs_errors_dbp.append((idx, abs(pred-ref), pred-ref))
abs_errors_dbp.sort(key=lambda x: x[1], reverse=True)

# Print requested summary
print('\nValidation complete. Summary:')
print('1. Cantidad total de segmentos:', total_segments)
print('2. Cantidad de segmentos válidos para SBP (>=1 ventana válida):', segments_with_at_least1_sbp)
print('3. Cobertura SBP (%): {:.2f}'.format(coverage_sbp))
print('4. MAE SBP: {:.3f}, RMSE SBP: {:.3f}, Bias SBP: {:.3f}, SD error SBP: {:.3f}, r: {:.3f}'.format(MAE_sbp, RMSE_sbp, Bias_sbp, SD_sbp, r_sbp))
print('5. Cantidad de segmentos válidos para DBP (>=1 ventana válida):', segments_with_at_least1_dbp)
print('6. Cobertura DBP (%): {:.2f}'.format(coverage_dbp))
print('7. MAE DBP: {:.3f}, RMSE DBP: {:.3f}, Bias DBP: {:.3f}, SD error DBP: {:.3f}, r: {:.3f}'.format(MAE_dbp, RMSE_dbp, Bias_dbp, SD_dbp, r_dbp))
print('8. Segmentos con 0 ventanas válidas SBP/DBP: SBP 0:', segments_0_valid_sbp, ', DBP 0:', segments_0_valid_dbp)
print('   Segmentos con 1 ventana válida SBP/DBP: SBP 1:', segments_1_valid_sbp, ', DBP 1:', segments_1_valid_dbp)

print('\n9. Top 10 segmentos con mayor error absoluto en SBP:')
for t in abs_errors_sbp[:10]:
    idx, abs_err, err = t
    print(f'  Segment {idx}: abs_err={abs_err:.3f}, error={err:+.3f}, SBP_ref={sbp_refs_all[idx]:.3f}, SBP_pred={sbp_preds_all[idx]:.3f}')

print('\n10. Top 10 segmentos con mayor error absoluto en DBP:')
for t in abs_errors_dbp[:10]:
    idx, abs_err, err = t
    print(f'  Segment {idx}: abs_err={abs_err:.3f}, error={err:+.3f}, DBP_ref={dbp_refs_all[idx]:.3f}, DBP_pred={dbp_preds_all[idx]:.3f}')

print('\nSaved results to', results_path)
print('Saved summary to', summary_path)
print('Saved full scale Bland-Altman plots to', bland_altman_raw_path)
print('Saved zoomed Bland-Altman plots to', bland_altman_zoom_path)