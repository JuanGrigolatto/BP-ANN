# Documentación de la estructura del proyecto

> Este documento fue verificado línea por línea contra el árbol real del repositorio
> (no es un plan aspiracional). Si movés o renombrás un archivo, actualizá esta sección.

## Jerarquía de directorios

```
BP-ANN/
│
├── 📄 README.md                          # Documentación principal del proyecto
├── 📄 LICENSE                            # Licencia Apache 2.0
├── 📄 STRUCTURE.md                       # Este archivo
├── 📄 requirements.txt                   # Dependencias de Python
├── 📄 setup.py                           # Configuración del paquete
│
├── 📁 src/                               # Paquete principal de código fuente
│   │
│   ├── 📁 features/                      # Procesamiento de señales e ingeniería de características
│   │   │
│   │   ├── 📁 data_processing/
│   │   │   └── Procesamiento_por_picos.py    # Segmentación latido a latido (basada en picos):
│   │   │                                     # carga .mat crudos, filtra, sincroniza y
│   │   │                                     # segmenta PPG/ECG/ABP latido a latido,
│   │   │                                     # extrae etiquetas SBP/DBP y normaliza.
│   │   │
│   │   └── Detector_de_picos.py          # Algoritmo de detección de picos para señales PPG/ECG
│   │
│   ├── 📁 data/                          # Carga de datos y gestión de datasets
│   │   │
│   │   └── 📁 data_chargers/             # Clases de Dataset de PyTorch
│   │       ├── Clase_UCIDataset.py       # UCIDataset: dataset base memmap-backed (soporta datasets más grandes que la RAM)
│   │       ├── MetaDataset.py            # TaskDataset: episodios MAML intra-paciente (Support/Query con gap temporal)
│   │       ├── PatientWiseSet.py         # PatientWiseDataset: episodios cross-patient (Support y Query de pacientes distintos)
│   │       ├── Tuningndataset.py         # TuningNDataset: subconjunto por paciente para fine-tuning few-shot
│   │       └── Intrapatientset.py        # Intrapatientset: todas las muestras secuenciales de UN paciente (evaluación clínica / recalibración)
│   │
│   ├── 📁 models/                        # Arquitecturas de red neuronal
│   │   ├── ConvolucionalV1.py            # 1D-CNN con activaciones ELU, BatchNorm y Dropout
│   │   ├── ConvolucionalV2.py            # 1D-CNN liviana con regularización reducida
│   │   └── InceptionTime.py             # InceptionTime: bloques multi-escala y conexiones residuales
│   │
│   ├── 📁 utils/                         # Utilidades generales
│   │   └── 📁 Tools/
│   │       └── Tools.py                  # Funciones compartidas: lectura HDF5/.mat, filtrado digital,
│   │                                     # detección de picos (ECG/ABP), segmentación y normalización
│   │
│   └── 📁 entrenamiento/                 # Entrenamiento supervisado estándar
│       ├── Entrenamiento.py              # Entrenamiento con split aleatorio 70/20/10 + AMP + Early Stopping
│       └── Entrenamiento_patient_subject.py  # Entrenamiento con split estricto por paciente
│
├── 📁 metalearning/                      # Pipeline de meta-aprendizaje
│   │
│   ├── Metaentrenamiento.py              # Meta-entrenamiento MAML (modos 'traditional' y 'patient_wise').
│   │                                     # main() acepta opcionalmente base_dataset/selected_patients/
│   │                                     # experiment_name para ser reutilizado por scripts de búsqueda
│   │                                     # de hiperparámetros sin recargar el dataset ni pisar checkpoints.
│   ├── Metaentrenamiento_delta.py        # Meta-entrenamiento con Delta Learning (loss híbrida MSE+Pearson)
│   ├── Fewshot.py                        # Fine-tuning few-shot sobre pacientes de test; MAE/RMSE pre/post adaptación
│   ├── zero_shot.py                      # Evaluación zero-shot (sin ninguna adaptación) del meta-modelo
│   ├── Respuesta_escalon.py              # Calibración única al inicio (congela conv, actualiza solo el regresor denso)
│   ├── Intrapatient_eval.py              # Recalibración periódica durante monitoreo prolongado
│   ├── Busqueda_Hiperparametros.py       # Grid search de hiperparámetros MAML (usa Metaentrenamiento.main())
│   ├── Hiperparametros_fewshot.py        # Grid search de learning rates para la fase de adaptación few-shot
│   │
│   ├── 📁 logs/                          # Logs de pérdida por experimento (CSV) + curvas (PNG)
│   └── 📁 Gráficos_ resultados_ metalearning/  # Gráficos de resultados guardados (Convencional / Patient-wise)
│
├── 📁 models/                            # Pesos de modelo persistidos (pesado, ver nota más abajo)
│   ├── 📁 best_models/                   # Checkpoints del mejor modelo supervisado por experimento
│   ├── 📁 best_meta_models/              # Checkpoints del mejor meta-modelo por experimento
│   └── 📁 checkpoints/                   # Checkpoints "latest" y "best" de cada corrida (resumable)
│
├── 📁 notebooks/                         # Scripts de visualización y análisis
│   ├── graficacion_datos.py              # Caracterización del dataset (split aleatorio)
│   ├── Graficacion_datos_meta.py         # Caracterización del dataset para meta-learning
│   ├── Graficacion_perdida.py            # Curvas de pérdida de meta-entrenamiento
│   └── Graficacion_combinada.py          # Visualización de resultados de búsqueda de hiperparámetros
│
├── 📁 Prueba_modelos/                    # Inferencia y pruebas standalone
│   └── Prueba_modelo.py                  # Carga un checkpoint y corre inferencia sobre datos de test;
│                                         # exporta métricas de error a data/processed/ (ver nota más abajo)
│
├── 📁 validation/                        # Validación algorítmica independiente del pipeline de señales
│   │
│   ├── 📁 ECG/                           # Validación del detector de picos QRS sobre MIT-BIH Arrhythmia DB
│   │   ├── Validar_picos_ECG_MITbih2.py  # Script vigente: remuestreo FIR a 125 Hz, filtrado, detección,
│   │   │                                 # matching por barrido de dos punteros (tolerancia ±50 ms),
│   │   │                                 # exporta CSV por registro + gráficos (histograma y error vs. RR).
│   │   ├── Validar_picos_ECG_MITbih.py   # Versión preliminar/legada de lo anterior (sin CSV por registro
│   │   │                                 # ni gráfico de error vs. RR). Se conserva a modo de referencia histórica.
│   │   ├── resultados_por_registro_mitbih.csv   # Salida: métricas (TP/FP/FN, sensibilidad, precisión) por registro
│   │   ├── error_vs_rr_mitbih.png               # Salida: densidad de error de localización vs. intervalo RR previo
│   │   └── histograma_errores_mitbih.png        # Salida: histograma de errores temporales de detección
│   │
│   └── 📁 ABP/                           # Validación del detector de picos ABP sobre VitalDB (AAMI)
│       ├── prepare_vital_for_validation.py      # Carga el .mat de VitalDB, detecta picos por ventana,
│       │                                        # guarda salidas intermedias (JSON + pickle)
│       ├── run_full_vital_validation.py         # Validación completa: SBP/DBP por segmento, Bland-Altman
│       │                                        # (crudo y acotado ±30 mmHg), CSV de resultados y resumen
│       ├── test_data/synthetic_abp.mat          # Señal ABP sintética de prueba
│       └── results/                             # Salidas: VitalDB_AAMI/ (CSV, Bland-Altman, top-10 errores
│                                                 # con sus gráficos por segmento), intermediate/, csv/
│
└── 📁 data/                              # Almacenamiento de datos (NO versionado en su mayoría, ver nota)
    ├── 📁 interim/
    │   └── patients_temp.dat             # Archivo temporal de procesamiento (memmap intermedio)
    └── 📁 processed/
        ├── Errores_predicción.npy / .npz # Salida de Prueba_modelo.py: errores (SBP,DBP) por muestra de test
        └── indices_errores.npy           # Salida de Prueba_modelo.py: índices globales asociados a esos errores
```

---

## ⚠️ Nota importante sobre los datos

`data/` está excluido por `.gitignore`; los 4 archivos que aparecen en `data/interim/` y
`data/processed/` quedaron versionados de forma puntual (forzados con `git add`) y son
**salidas** de una corrida anterior de `Prueba_modelos/Prueba_modelo.py`, no el dataset de
entrenamiento.

**El dataset procesado real que consumen los scripts de entrenamiento/meta-entrenamiento
(`data/processed/data_UCI/dataset_parte_1_por_picos.pt` … `_4_por_picos.pt`,
`few_shot_patient_data.pt`, `test_set_por_pacientes_iso/test_meta.pt`) NO está incluido en
el repositorio** (es demasiado pesado para versionar). Para regenerarlo:

1. Descargar el dataset crudo *Cuff-Less Blood Pressure Estimation* (UCI/MIMIC II).
2. Ubicarlo en `data/raw/` según la ruta que espera `Procesamiento_por_picos.py`.
3. Correr `src/features/data_processing/Procesamiento_por_picos.py` para generar los `.pt`/`.dat` procesados.
4. Recién entonces, los scripts de `src/entrenamiento/` y `metalearning/` pueden ejecutarse.

Este paso todavía no está automatizado con argumentos de línea de comandos (las rutas están
hardcodeadas dentro de cada script) — es el próximo punto a resolver en la fase de "probar
que los scripts corran".

---

## Descripción de los módulos

### `src/models/`

| Archivo | Arquitectura | Caso de uso |
|---|---|---|
| `ConvolucionalV1.py` | 4× Conv1d → 4× Linear, ELU, BN, Dropout(0.5) | Modelo principal para meta-entrenamiento y few-shot |
| `ConvolucionalV2.py` | 4× Conv1d → 4× Linear, ReLU, BN reducido | Comparación baseline liviana |
| `InceptionTime.py` | Bloques Inception + residual + GAP | Extracción de características temporales multi-escala |

Todos los modelos aceptan tensores de entrada de forma `(batch, channels=2, signal_length)` y devuelven `(batch, 2)` para la regresión simultánea de SBP y DBP.

---

### `src/data/data_chargers/`

| Archivo | Clase | Propósito |
|---|---|---|
| `Clase_UCIDataset.py` | `UCIDataset` | Dataset base respaldado por memmap; soporta datasets más grandes que la RAM disponible |
| `MetaDataset.py` | `TaskDataset` | Construye episodios MAML intra-paciente: Support/Query del mismo paciente con brecha temporal |
| `PatientWiseSet.py` | `PatientWiseDataset` | Construye episodios cross-patient: Support y Query mezclando distintos pacientes |
| `Tuningndataset.py` | `TuningNDataset` | Subconjunto por paciente para fine-tuning few-shot (primeros n_shots = soporte, resto = evaluación) |
| `Intrapatientset.py` | `Intrapatientset` | Todas las muestras secuenciales de un único paciente, para evaluación clínica intra-paciente |

---

### `metalearning/`

| Archivo | Propósito |
|---|---|
| `Metaentrenamiento.py` | Loop principal de MAML. Admite modos `traditional` (tarea aleatoria) y `patient_wise` (cross-patient). `main()` también acepta `base_dataset`, `selected_patients` y `experiment_name` para ser reutilizado desde la búsqueda de hiperparámetros |
| `Metaentrenamiento_delta.py` | MAML con Delta Learning: predice deltas de PA respecto de la media del support set; loss híbrida; annealing de pasos |
| `Fewshot.py` | Aplica fine-tuning K-shot sobre pacientes de test; calcula MAE, RMSE y tasa de mejora pre/post |
| `zero_shot.py` | Corre el meta-modelo congelado sobre pacientes no vistos; sin ninguna adaptación |
| `Respuesta_escalon.py` | Calibración única en t=0 (congela capas conv, actualiza solo el regresor denso); sigue el desempeño durante toda la sesión |
| `Intrapatient_eval.py` | Recalibra cada X minutos; grafica el seguimiento de SBP/DBP y el MAE a lo largo del tiempo por paciente |
| `Busqueda_Hiperparametros.py` | Grid search exhaustivo sobre `adapt_lr × meta_lr × k_steps × N_patient_group`; carga el dataset y selecciona los pacientes una sola vez, reutiliza `Metaentrenamiento.main()` por combinación; guarda JSON + CSV + gráfico comparativo |
| `Hiperparametros_fewshot.py` | Barre tasas de aprendizaje de adaptación; reporta la tasa de mejora poblacional (% de pacientes que mejoran post-adaptación) |

---

### `validation/`

| Archivo | Propósito |
|---|---|
| `ECG/Validar_picos_ECG_MITbih2.py` | Validación del detector QRS propio contra las anotaciones de MIT-BIH Arrhythmia DB. Remuestrea a 125 Hz (FIR), filtra, detecta, empareja por barrido de dos punteros (tolerancia configurable, default ±50 ms), y reporta sensibilidad/precisión/sesgo/SD global y por registro. Ruta del dataset configurable vía `--data_path` (default: `data/raw/mit-bih-arrhythmia-database-1.0.0/`); corre como script directo o como módulo (`python -m validation.ECG.Validar_picos_ECG_MITbih2`) |
| `ECG/Validar_picos_ECG_MITbih.py` | Versión preliminar del script anterior; se mantiene como referencia histórica del proceso de auditoría |
| `ABP/prepare_vital_for_validation.py` | Prepara detecciones intermedias de picos ABP sobre el subset AAMI de VitalDB. Ruta configurable vía `--data_mat` |
| `ABP/run_full_vital_validation.py` | Validación completa de SBP/DBP estimados vs. referencia sobre VitalDB, con análisis Bland-Altman. Ruta configurable vía `--data_mat` (default: `data/raw/VitalDB_AAMI_Test_Subset/VitalDB_AAMI_Test_Subset.mat`) |

---

## Flujo de datos

```
Señales crudas UCI PPG/ECG/ABP (.mat)         [NO incluido en el repo, ver nota de datos]
        │
        ▼
Detección de picos (Detector_de_picos.py / Tools.py)
        │
        ▼
Segmentación por latido → ventanas de 500 muestras @ 125 Hz (~4 seg/latido)   (Procesamiento_por_picos.py)
        │
        ▼
Normalización Z-score por canal
        │
        ▼
Archivos .dat memory-mapped + metadatos .pt     [tampoco incluidos, se regeneran localmente]
        │
        ├──► UCIDataset (entrenamiento estándar)
        ├──► MetaDataset / PatientWiseSet (episodios MAML)
        └──► Intrapatientset (evaluación clínica)
```

---

## Convención de nombres de checkpoints

Los checkpoints siguen el patrón:

```
checkpoint_<experiment_name>.pt   ← estado más reciente (resumable)
best_<experiment_name>.pt         ← snapshot de mejor pérdida de validación
```

Cada checkpoint contiene: `epoch`, `model_state_dict`, `optimizer_state_dict`, `best_loss`.

Dentro de `Busqueda_Hiperparametros.py`, `<experiment_name>` se arma automáticamente a partir
de la combinación de hiperparámetros de esa corrida (ej. `hsearch_a0.01_m0.001_k5_G4_p5_q10`),
así ninguna corrida pisa el checkpoint de otra.

---

## Hiperparámetros clave

| Parámetro | Valor por defecto | Descripción |
|---|---|---|
| `shots` | 5 | Tamaño del support set por tarea (muestras de calibración few-shot) |
| `adapt_lr` | 0.005 | Tasa de aprendizaje del inner loop (específica por tarea) |
| `meta_lr` | 0.001 | Tasa de aprendizaje del outer loop (meta) |
| `adapt_steps` | 5 | Pasos de gradiente del inner loop por tarea |
| `tasks_per_batch` | 4 | Cantidad de tareas por actualización de meta-gradiente |
| `signal_length` | 500 | Muestras por ventana de entrada (4 seg @ 125 Hz) |
| `alpha` (Delta) | 0.75 | Peso de la pérdida de Pearson en la función de pérdida híbrida |
