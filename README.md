# BP-ANN: Estimación no invasiva de presión arterial mediante CNNs y Meta-Learning

<p align="center">
  <img src="https://img.shields.io/badge/License-Apache%202.0-blue.svg" alt="License">
  <img src="https://img.shields.io/badge/Python-3.9+-blue" alt="Python">
  <img src="https://img.shields.io/badge/PyTorch-2.0+-red" alt="PyTorch">
  <img src="https://img.shields.io/badge/learn2learn-MAML-green" alt="learn2learn">
  <img src="https://img.shields.io/badge/Dataset-UCI%20PPG--ECG--BP-orange" alt="Dataset">
  <img src="https://img.shields.io/badge/Status-Final%20Project-blueviolet" alt="Status">
</p>

**BP-ANN** es un proyecto de investigación que implementa redes neuronales artificiales para la **estimación no invasiva de la presión arterial** a partir de señales de **Fotopletismografía (PPG)** y **Electrocardiografía (ECG)**. Este trabajo es el Proyecto Final de la carrera de Bioingeniería de la *Universidad Nacional de Entre Ríos (UNER)* y propone un enfoque innovador que combina arquitecturas 1D-CNN tradicionales con técnicas de **meta-aprendizaje (MAML)** para lograr una adaptación rápida y específica por paciente con una cantidad mínima de datos de calibración.

---

## 📋 Índice

- [Descripción general](#descripción-general)
- [Características principales](#características-principales)
- [Arquitectura](#arquitectura)
- [Estructura del proyecto](#estructura-del-proyecto)
- [Instalación](#instalación)
- [Uso](#uso)
- [Paradigmas de evaluación](#paradigmas-de-evaluación)
- [Validación del procesamiento de señales](#validación-del-procesamiento-de-señales)
- [Dataset](#dataset)
- [Búsqueda de hiperparámetros](#búsqueda-de-hiperparámetros)
- [Resultados](#resultados)
- [Licencia](#licencia)

---

## Descripción general

El monitoreo continuo y no invasivo de la presión arterial es un desafío clínico aún sin resolver. Los métodos gold-standard actuales son invasivos (línea arterial) o intermitentes (basados en manguito). Este proyecto propone un pipeline de deep learning que:

1. Extrae representaciones latido a latido a partir de señales sincronizadas de PPG, ECG y ABP.
2. **Entrena un modelo base** con un enfoque supervisado estándar (partición aleatoria o partición por paciente).
3. **Meta-entrena el modelo** utilizando MAML, de modo que aprenda una inicialización capaz de adaptarse rápidamente a la fisiología de un nuevo paciente usando solo unos pocos latidos (Support Set).
4. **Evalúa clínicamente** simulando escenarios reales: generalización zero-shot, calibración única (respuesta al escalón) y recalibración periódica durante monitoreo prolongado.

---

## Características principales

- 🧠 **Tres arquitecturas neuronales**: `ConvolucionalV1` (1D-CNN + ELU + Dropout), `ConvolucionalV2` (1D-CNN liviana) e `InceptionTime` (CNN residual multi-escala).
- 📊 **Procesamiento de señales avanzado**: segmentación basada en picos y preprocesamiento por ventana fija para señales PPG/ECG/ABP.
- 🔄 **Meta-aprendizaje con MAML**: tres paradigmas de construcción de tareas: traditional (support y query set muestreados aleatoriamente del registro de un mismo paciente), patient_wise (tareas formadas mezclando segmentos de señal de múltiples pacientes para forzar la generalización entre sujetos), y desacople temporal (muestras secuenciales para el support set y muestras separadas en el tiempo para el query set, dentro del mismo paciente).
- ⚡ **Delta Learning**: predice *variaciones* de presión arterial respecto de una línea de base de calibración, usando una función de pérdida híbrida MSE + Pearson para capturar tanto la magnitud del error como la correlación de tendencia.
- 🎯 **Estimación de SBP y DBP**: predicción simultánea de presión arterial sistólica y diastólica.
- 📈 **Búsqueda exhaustiva de hiperparámetros**: grid search automatizado sobre las tasas de aprendizaje interna/externa de MAML, pasos de adaptación y tamaños de grupo de tareas.
- 🏥 **Evaluación integral**: evaluación de modelos estáticos (partición aleatoria y por paciente) y del seguimiento dinámico del meta-aprendizaje (zero-shot, calibración inicial única y recalibración periódica), todo comparado contra los estándares de medición continua AAMI/ISO 81060-3.

---

## Arquitectura

### Modelos de red neuronal

| Modelo | Descripción | Propiedades clave |
|---|---|---|
| `ConvolucionalV1` | 1D-CNN de 4 capas + regresor denso de 4 capas | BatchNorm, Dropout (0.5), activaciones ELU |
| `ConvolucionalV2` | 1D-CNN liviana de 4 capas | Regularización reducida, activaciones ReLU |
| `InceptionTime` | Bloques inception multi-escala + conexiones residuales | Global Average Pooling, profundidad configurable |

### Pipeline de meta-aprendizaje

```
Modelo base (CNN)
      │
      ▼
  Meta-entrenamiento MAML
  ┌─────────────────────────────────┐
  │  Para cada episodio (tarea):     │
  │  ┌──────────────────────────┐   │
  │  │  Support Set (few shots)  │   │
  │  │  → Inner Loop (adaptación)│   │
  │  └──────────────────────────┘   │
  │  ┌──────────────────────────┐   │
  │  │  Query Set                │   │
  │  │  → Outer Loop (meta-grad) │   │
  │  └──────────────────────────┘   │
  └─────────────────────────────────┘
      │
      ▼
  Modelo meta-entrenado
      │
      ├──► Evaluación Zero-Shot
      ├──► Ajuste fino Few-Shot (Respuesta al escalón)
      └──► Recalibración periódica (monitoreo prolongado)
```

### Delta Learning

En lugar de predecir valores absolutos de presión arterial, la variante Delta Learning predice **desviaciones respecto de una línea de base de calibración específica del paciente** (calculada como la media del support set). Esto se combina con una **pérdida híbrida**:

```
Loss = (1 - λ) · MSE(pred_delta, true_delta) + λ · (1 - Pearson(pred_delta, true_delta))
```

---

## Estructura del proyecto

Para la documentación detallada de directorios, ver [STRUCTURE.md](./STRUCTURE.md).

```
BP-ANN/
├── src/
│   ├── features/        # Preprocesamiento de señales y detección de picos
│   ├── data/             # Clases de Dataset (UCI, Meta, Patient-Wise, Tuning, Intrapatient)
│   ├── models/           # Arquitecturas CNN (V1, V2, InceptionTime)
│   ├── utils/             # Herramientas auxiliares
│   └── entrenamiento/    # Scripts de entrenamiento y configuración
├── metalearning/         # Entrenamiento MAML, adaptación few-shot, evaluación, búsqueda de hiperparámetros
├── models/               # Checkpoints guardados y mejores modelos
├── notebooks/            # Scripts de visualización y análisis
├── data/                 # Salidas de datos procesados (el dataset de entrenamiento en sí no está versionado, ver nota abajo)
├── Prueba_modelos/       # Scripts de inferencia y prueba
└── validation/           # Validación independiente de los detectores de picos QRS (ECG) y ABP
    ├── ECG/               # Auditoría del detector de picos contra MIT-BIH Arrhythmia Database
    └── ABP/               # Auditoría del detector de picos contra VitalDB (subset AAMI)
```

> ⚠️ **Nota:** el dataset procesado de entrenamiento (`data/processed/data_UCI/*.pt`) no está
> incluido en el repositorio por su tamaño. Ver la sección "Nota importante sobre los datos" en
> [STRUCTURE.md](./STRUCTURE.md) para los pasos de regeneración antes de correr cualquier script
> de `src/entrenamiento/` o `metalearning/`.

---

## Instalación

### Requisitos previos

- Python 3.9+
- GPU compatible con CUDA (recomendado para el entrenamiento)

### Configuración

```bash
# 1. Clonar el repositorio
git clone https://github.com/JuanGrigolatto/BP-ANN.git
cd BP-ANN

# 2. Crear y activar un entorno virtual
python -m venv venv
# En Windows:
venv\Scripts\activate
# En Linux/Mac:
source venv/bin/activate

# 3. Instalar PyTorch con soporte CUDA 12.1 (necesario para aceleración por GPU)
pip install torch==2.5.1+cu121 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# 4. Instalar dependencias
pip install -r requirements.txt

# 5. Instalar el paquete en modo editable
pip install -e .
```

---

## Uso

*Nota: por defecto, todos los scripts de entrenamiento están configurados para usar la arquitectura `ConvolucionalV1` y el dataset preprocesado Cuff-Less Blood Pressure (MIMIC II) (entradas PPG + ECG). Modificar arquitecturas o hiperparámetros se puede hacer directamente en los diccionarios de configuración dentro de cada script.*

### 1. Entrenamiento estándar (partición aleatoria)

```bash
python src/entrenamiento/Entrenamiento.py
```

Entrena la CNN base usando una partición aleatoria tradicional 70/20/10 de ventanas sobre todo el dataset. Usa entrenamiento de precisión mixta (AMP) y early stopping.
Salida: los mejores pesos del modelo se guardan automáticamente en `models/best_models/`.

### 2. Entrenamiento estándar (partición por paciente)

```bash
python src/entrenamiento/Entrenamiento_patient_subject.py
```

Entrena el modelo usando una partición estricta por paciente para evaluar la generalización estática (baseline zero-shot), garantizando que ningún paciente aparezca tanto en el conjunto de entrenamiento como en el de test.

### 3. Meta-entrenamiento con MAML

```bash
python metalearning/Metaentrenamiento.py
```

Meta-entrena el modelo usando MAML. Admite dos modos configurables dentro del script:

-traditional: las tareas se forman con ventanas de señal aleatorias del registro de un único paciente.

-patient_wise: las tareas se forman mezclando datos de distintos pacientes, forzando la generalización entre sujetos.

Salida: los checkpoints meta-entrenados se guardan en `models/best_meta_models/`.

### 4. Meta-entrenamiento con Delta Learning

```bash
python metalearning/Metaentrenamiento_delta.py
```

Meta-entrena usando el paradigma Delta Learning con una pérdida híbrida MSE + Pearson y annealing adaptativo de los pasos del inner loop, para predecir variaciones de presión arterial respecto de una línea de base de calibración específica del paciente.

### 5. Evaluación Few-Shot

```bash
python metalearning/Fewshot.py
```

Evalúa el modelo meta-entrenado sobre pacientes de test no vistos, realizando una adaptación rápida con un support set pequeño (por defecto: 5 ciclos cardíacos), y mide el MAE y RMSE antes y después de la adaptación.

---

## Paradigmas de evaluación

### Zero-Shot

Evalúa el meta-modelo sobre pacientes no vistos **sin ningún ajuste fino**. Mide la capacidad de generalización de la meta-inicialización aprendida.

```bash
python metalearning/zero_shot.py
```

### Respuesta al escalón (calibración única)

Simula una **única calibración inicial** usando los primeros N latidos de un paciente. Las capas convolucionales se congelan y solo se actualiza el regresor denso. Evalúa el desempeño de seguimiento del modelo durante el resto de la sesión.

```bash
python metalearning/Respuesta_escalon.py
```

### Recalibración periódica (monitoreo prolongado)

Simula una **sesión de monitoreo continuo** donde el modelo se recalibra cada X minutos usando un pequeño lote de mediciones nuevas. Analiza la degradación del error a lo largo del tiempo y la efectividad del ajuste fino parcial periódico.

```bash
python metalearning/Intrapatient_eval.py
```

---

## Validación del procesamiento de señales

Más allá del propio pipeline de estimación de presión arterial, `validation/` audita los dos
detectores de picos que alimentan las etiquetas de referencia y la segmentación sincronizada con
ECG (`Tools.py`) contra bases de datos de referencia independientes y anotadas públicamente.

### ECG (detección de complejos QRS) — MIT-BIH Arrhythmia Database

```bash
python validation/ECG/Validar_picos_ECG_MITbih2.py --data_path path/to/mit-bih-arrhythmia-database-1.0.0
```

Corre el detector de QRS propio del proyecto sobre todos los registros de la MIT-BIH Arrhythmia
Database, remuestreados a 125 Hz (remuestreo polifásico FIR) para igualar la frecuencia de
muestreo de trabajo del pipeline. Las detecciones se emparejan contra las anotaciones expertas de
la base de datos (clases de latido AAMI, más latidos ectópicos/raros) mediante un barrido de dos
punteros con ventana de tolerancia configurable (±50 ms por defecto).

Reporta, de forma global y por registro: sensibilidad, precisión (PPV), error temporal medio
(sesgo), desvío estándar, e intervalo de error empírico del 95%. Exporta un CSV por registro
(`resultados_por_registro_mitbih.csv`), un histograma de errores temporales de detección, y un
gráfico hexbin de error de detección vs. el intervalo R-R previo.

*Descargá la base de datos por separado desde [PhysioNet](https://physionet.org/content/mitdb/1.0.0/).
`--data_path` por defecto apunta a `data/raw/mit-bih-arrhythmia-database-1.0.0/` relativo a la
raíz del repo si se omite; en caso contrario, pasá una ruta absoluta. Funciona tanto como script
directo (`python validation/ECG/Validar_picos_ECG_MITbih2.py`) como módulo
(`python -m validation.ECG.Validar_picos_ECG_MITbih2`).*

### ABP (detección de picos sistólico/diastólico) — VitalDB (subset AAMI)

```bash
python validation/ABP/prepare_vital_for_validation.py --data_mat "path/to/VitalDB_AAMI_Test_Subset.mat"
python validation/ABP/run_full_vital_validation.py --data_mat "path/to/VitalDB_AAMI_Test_Subset.mat"
```

Valida el detector de picos ABP del proyecto (`filtrar_abp`, `detectar_picos_abp` en `Tools.py`)
contra el subset de test AAMI de VitalDB. Calcula estimaciones de SBP/DBP a nivel de segmento y
reporta cobertura, métricas de error, y gráficos de Bland-Altman (escala completa y acotada a
±30 mmHg) contra los valores de referencia.

*`--data_mat` por defecto apunta a `data/raw/VitalDB_AAMI_Test_Subset/VitalDB_AAMI_Test_Subset.mat`
relativo a la raíz del repo en ambos scripts si se omite; en caso contrario, pasá una ruta absoluta.*

---

## Dataset

Este proyecto usa el **Cuff-Less Blood Pressure Estimation Dataset** del UCI Machine Learning Repository. Este dataset es un subconjunto preprocesado y validado de la base de datos **MIMIC II** (Multiparameter Intelligent Monitoring in Intensive Care), que contiene señales sincronizadas de Fotopletismografía (PPG), Electrocardiografía (ECG, derivación II) y Presión Arterial invasiva (ABP) muestreadas a **125 Hz**.

**Pipeline de preprocesamiento:**
1. Detección adaptativa de picos en ECG (picos R para la segmentación por ventanas sincronizadas) y en la señal ABP invasiva (para la extracción de las etiquetas de referencia SBP/DBP).
2. Segmentación latido a latido en ventanas de longitud fija (500 muestras ≈ 4 segundos).
3. Normalización Z-score (SBP: μ=134.02, σ=22.75 mmHg | DBP: μ=63.47, σ=23.69 mmHg).
4. Almacenamiento como arrays memory-mapped (`.dat`) con metadatos `.pt` para una carga eficiente.

**Paradigmas de partición de datos:**
Para evaluar tanto el desempeño estático como la adaptación dinámica, el dataset se particiona de dos formas distintas:

1. **Partición aleatoria (baseline ideal):** una partición estándar 70/20/10 de todas las ventanas de señal a lo largo de todo el dataset, sin respetar los límites entre pacientes (usada en `Entrenamiento.py`).
2. **Partición a nivel de paciente (test de generalización y meta-aprendizaje):** una división estricta por ID de paciente para evitar fuga de datos. Esta partición se usa para demostrar la falla de generalización de los modelos estáticos (`Entrenamiento_patient_subject.py`) y para entrenar/evaluar el pipeline MAML:
   - **70%** pacientes de entrenamiento
   - **15%** pacientes de validación
   - **15%** pacientes de test (reservados para adaptación zero-shot y few-shot)

---

## Búsqueda de hiperparámetros

Se proveen dos scripts automatizados de grid search:

**Búsqueda de hiperparámetros de MAML** (fase de meta-entrenamiento):
```bash
python metalearning/Busqueda_Hiperparametros.py
```
Explora `adapt_lr`, `meta_lr`, `k_adapt_steps`, y tamaños de grupo de pacientes. Los resultados se guardan en JSON y CSV con gráficos comparativos.

**Búsqueda de tasa de aprendizaje Few-Shot** (fase de adaptación):
```bash
python metalearning/Hiperparametros_fewshot.py
```
Evalúa distintas tasas de aprendizaje para el ajuste fino intra-paciente, midiendo el MAE antes/después de la adaptación y la tasa de mejora sobre la población.

---
## Resultados

> Las métricas de desempeño se evalúan contra los estándares clínicos AAMI/ISO 81060-3:2022 (Error Medio ≤ ±5 mmHg, SD ≤ 8 mmHg).

| Paradigma de evaluación | SBP (MAE / RMSE) | SBP (ME ± SD) | DBP (MAE / RMSE) | DBP (ME ± SD) | Cumplimiento AAMI/ISO |
|---------------------|------------------|---------------|------------------|---------------|---------------------|
| **Partición aleatoria** (Baseline V1) | 3.84 / 5.74 | 0.65 ± 6.60 | 3.84 / 5.74* | 0.07 ± 4.67 | **Cumple** |
| **Por paciente** (Zero-Shot V1) | 7.31 / 12.54 | -1.92 ± 13.51 | 7.31 / 12.54* | -2.75 ± 10.99 | **No cumple** (deriva severa de datos) |
| **Delta Meta-Learning** (5-Shot)| 7.14 / 10.87 | 0.18 ± 10.87 | 4.73 / 7.51 | -0.71 ± 7.48 | **Parcial** (DBP cumple, SD de SBP > 8) |

*(Nota: las métricas MAE/RMSE del baseline reflejan el desempeño global promedio combinando ambos canales SBP/DBP).*

### Hallazgos principales
* **Falla del modelo estático:** las CNN tradicionales tienen buen desempeño con partición aleatoria pero fallan en generalizar a pacientes no vistos (partición por paciente), mostrando una fuerte regresión a la media y superando los límites de error clínico.
* **Éxito de MAML + Delta Learning:** el ajuste fino con solo 5 muestras de soporte eliminó exitosamente el sesgo sistemático (Error Medio < 1 mmHg tanto para SBP como para DBP) y redujo el error absoluto en el **62.1%** (SBP) y **77.0%** (DBP) de los pacientes no vistos.
* **Compromiso de varianza:** si bien Delta Learning sigue las variaciones relativas, carece de memoria temporal, lo que aumenta la susceptibilidad al ruido de alta frecuencia. Esto mantiene el desvío estándar de SBP (10.87 mmHg) por encima del umbral clínico de 8 mmHg.
---

## Cita

Si usás este trabajo en tu investigación, por favor citá:

```bibtex
@thesis{grigolatto2026bpann,
  author  = {Grigolatto, Juan Marcos},
  title   = {Non-Invasive Blood Pressure Estimation using Convolutional Neural Networks and Meta-Learning},
  school  = {Universidad Nacional de Entre Ríos (UNER)},
  year    = {2026},
  type    = {Final Bioengineering Project}
}
```

---

## Licencia

Este proyecto está licenciado bajo la [Licencia Apache 2.0](./LICENSE).

---

<p align="center">
  Desarrollado como Proyecto Final de Bioingeniería · UNER · 2026<br>
  Autor: <strong>Juan Marcos Grigolatto</strong>
</p>
