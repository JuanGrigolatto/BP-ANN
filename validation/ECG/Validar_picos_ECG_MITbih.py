import os
import glob
import wfdb
import traceback
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import resample_poly
import src.utils.Tools.Tools as tools  #[cite: 1]

def evaluar_registro(ruta_base, registro_id, ventana_ms=50):
    """
    Carga, remuestrea (FIR), filtra y evalúa un único registro.
    Retorna métricas, errores y el canal utilizado.
    """
    ruta_registro = os.path.join(ruta_base, str(registro_id))
    record = wfdb.rdrecord(ruta_registro)
    annotation = wfdb.rdann(ruta_registro, 'atr')
    
    fs_original = record.fs  
    fs_nueva = 125           
    
    canal_usado = 'MLII'
    warning_msg = None
    if 'MLII' in record.sig_name:
        canal_idx = record.sig_name.index('MLII')
    else:
        canal_idx = 0  
        canal_usado = record.sig_name[canal_idx]
        warning_msg = f"Registro {registro_id}: MLII no encontrada. Usando {canal_usado}."
        
    senal_original = record.p_signal[:, canal_idx]
    
    # Símbolos estándar AAMI + latidos raros/ectópicos para detección total
    simbolos_validos = ['N', 'L', 'R', 'V', 'A', 'F', 'j', 'S', 'E', '/', 'a', 'J', 'e', 'f', 'Q']
    indices_validos = [i for i, sym in enumerate(annotation.symbol) if sym in simbolos_validos]
    picos_ref_originales = annotation.sample[indices_validos]
    
    # Remuestreo FIR 
    up, down = 25, 72
    senal_125 = resample_poly(senal_original, up, down)
    picos_ref_125 = np.round(picos_ref_originales * up / down).astype(int)
    
    # Acondicionamiento y Detección
    senal_filtrada = tools.filtrado_para_deteccion_Q(senal_125) #[cite: 1]
    picos_detectados = tools.detectar_picos_ecg(senal_filtrada, fs=fs_nueva) #[cite: 1]
    
    # Two-Pointer Sweep Matching 
    tolerancia_muestras = int(np.round((ventana_ms / 1000.0) * fs_nueva))
    
    tp, fp, fn = 0, 0, 0
    i_ref, i_det = 0, 0
    errores_temporales_ms = []
    
    while i_ref < len(picos_ref_125) and i_det < len(picos_detectados):
        ref = picos_ref_125[i_ref]
        det = picos_detectados[i_det]
        distancia = det - ref
        
        if abs(distancia) <= tolerancia_muestras:
            tp += 1
            errores_temporales_ms.append((distancia / fs_nueva) * 1000)
            i_ref += 1
            i_det += 1
        elif det < ref - tolerancia_muestras:
            fp += 1
            i_det += 1
        else:
            fn += 1
            i_ref += 1
            
    fn += len(picos_ref_125) - i_ref
    fp += len(picos_detectados) - i_det
    
    return tp, fp, fn, errores_temporales_ms, warning_msg, canal_usado

if __name__ == '__main__':
    ruta_datos = r"C:\PA-ANN\BP-ANN-clean\data\raw\mit-bih-arrhythmia-database-1.0.0"
    
    archivos_hea = glob.glob(os.path.join(ruta_datos, '*.hea'))
    registros = sorted([os.path.basename(f).replace('.hea', '') for f in archivos_hea])
    
    print(f"Iniciando procesamiento masivo y estricto de {len(registros)} registros...")
    
    tp_global, fp_global, fn_global = 0, 0, 0
    errores_globales_ms = []
    indices_globales_latido = []  # Nuevo Eje X para el hexbin
    latido_counter = 0
    
    lista_resultados = []
    warnings_log = []
    
    for reg in registros:
        try:
            tp, fp, fn, errores_ms, warning, canal = evaluar_registro(ruta_datos, reg, ventana_ms=50)
            
            if warning:
                warnings_log.append(warning)
                
            tp_global += tp
            fp_global += fp
            fn_global += fn
            
            # Asignar índice cronológico global a los errores de este registro
            errores_globales_ms.extend(errores_ms)
            indices_globales_latido.extend(range(latido_counter, latido_counter + len(errores_ms)))
            latido_counter += len(errores_ms)
            
            # Métricas locales para el DataFrame
            sens_local = tp / (tp + fn) if (tp + fn) > 0 else 0
            prec_local = tp / (tp + fp) if (tp + fp) > 0 else 0
            me_local = np.mean(errores_ms) if errores_ms else 0
            sd_local = np.std(errores_ms) if errores_ms else 0
            
            lista_resultados.append({
                'Registro': reg,
                'Canal': canal,
                'TP': tp,
                'FP': fp,
                'FN': fn,
                'Sensibilidad': round(sens_local, 4),
                'Precision': round(prec_local, 4),
                'ME_ms': round(me_local, 2),
                'SD_ms': round(sd_local, 2)
            })
            
            print(f"Registro {reg} procesado -> TP: {tp} | FP: {fp} | FN: {fn} | Sens: {sens_local:.3f}")
            
        except Exception as e:
            print(f"  [!] Error crítico procesando el registro {reg}:")
            traceback.print_exc()

    # --- EXPORTAR TABLA DE RESULTADOS (CSV) ---
    df_resultados = pd.DataFrame(lista_resultados)
    df_resultados.to_csv("resultados_por_registro_mitbih.csv", index=False)
    print("\n[✔] Tabla detallada por registro guardada en 'resultados_por_registro_mitbih.csv'")

    # --- IMPRIMIR WARNINGS ---
    if warnings_log:
        print("\n--- REPORTE DE DERIVACIONES NO ESTÁNDAR ---")
        for w in warnings_log:
            print(w)

    # --- MÉTRICAS ESTADÍSTICAS GLOBALES ---
    total_gt = tp_global + fn_global
    sensibilidad = tp_global / total_gt if total_gt > 0 else 0
    precision = tp_global / (tp_global + fp_global) if (tp_global + fp_global) > 0 else 0
    error_medio = np.mean(errores_globales_ms) if errores_globales_ms else 0
    std_error = np.std(errores_globales_ms) if errores_globales_ms else 0
    perc_2_5 = np.percentile(errores_globales_ms, 2.5) if errores_globales_ms else 0
    perc_97_5 = np.percentile(errores_globales_ms, 97.5) if errores_globales_ms else 0
    
    print("\n" + "=" * 50)
    print("RESULTADOS GLOBALES AUDITADOS (MIT-BIH)")
    print("=" * 50)
    print(f"Total Latidos Válidos (GT): {total_gt}")
    print(f"Sensibilidad: {sensibilidad:.4f}")
    print(f"Precisión (VPP): {precision:.4f}")
    print(f"Error Medio (Sesgo): {error_medio:.2f} ms")
    print(f"Desviación Estándar Paramétrica: ±{std_error:.2f} ms")
    print(f"Intervalo Empírico 95% (Percentiles): [{perc_2_5:.2f} ms,  {perc_97_5:.2f} ms]")
    print("=" * 50)

    # --- 1. GRÁFICO DE BLAND-ALTMAN GLOBAL (HEXBIN CORREGIDO) ---
    plt.figure(figsize=(12, 6))
    
    # Eje X: Índice de Latido Global (evita superposición de tiempos)
    hb = plt.hexbin(indices_globales_latido, errores_globales_ms, gridsize=100, cmap='Blues', mincnt=1, bins='log')
    cb = plt.colorbar(hb)
    cb.set_label('Cantidad de latidos (escala log)')
    
    plt.axhline(error_medio, color='red', linewidth=2, label=f'Sesgo (ME): {error_medio:.2f} ms')
    
    limite_superior = error_medio + 1.96 * std_error
    limite_inferior = error_medio - 1.96 * std_error
    plt.axhline(limite_superior, color='orange', linestyle='--', linewidth=2, label=f'+1.96 SD: {limite_superior:.2f} ms')
    plt.axhline(limite_inferior, color='orange', linestyle='--', linewidth=2, label=f'-1.96 SD: {limite_inferior:.2f} ms')
    
    plt.title("Bland-Altman Global (Hexbin): Error de Localización (ECG a 125 Hz)")
    plt.xlabel("Índice de Latido Global (Orden de procesamiento)")
    plt.ylabel("Error de localización (ms)")
    plt.legend(loc='upper right')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig("bland_altman_mitbih_hexbin.png", dpi=300)
    plt.show()
    plt.close() 

    # --- 2. HISTOGRAMA DE ERRORES TEMPORALES ---
    plt.figure(figsize=(10, 5))
    bordes_bins = np.arange(-52, 53, 8) 
    plt.hist(errores_globales_ms, bins=bordes_bins, color='#1f77b4', edgecolor='black', rwidth=0.85, alpha=0.7)
    plt.xticks(np.arange(-48, 49, 8))
    
    total_latidos = len(errores_globales_ms)
    exactos = sum(1 for e in errores_globales_ms if abs(e) <= 8)
    porcentaje_exactos = (exactos / total_latidos) * 100 if total_latidos > 0 else 0
    
    plt.title(f"Histograma de Errores de Localización Temporal (Acierto <= 1 muestra: {porcentaje_exactos:.2f}%)")
    plt.xlabel("Error temporal (ms)")
    plt.ylabel("Cantidad de latidos (Frecuencia)")
    plt.grid(True, linestyle='--', alpha=0.6, axis='y') 
    plt.tight_layout()
    plt.savefig("histograma_errores_mitbih.png", dpi=300)
    plt.show()
    plt.close()