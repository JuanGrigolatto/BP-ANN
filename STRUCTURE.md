# Project Structure Documentation

## Directory Hierarchy

```
BP-ANN/
│
├── 📄 README.md                          # Main project documentation
├── 📄 LICENSE                            # Apache 2.0 License
├── 📄 STRUCTURE.md                       # This file
├── 📄 requirements.txt                   # Python dependencies
├── 📄 setup.py                           # Package configuration
│
├── 📁 src/                               # Main source package
│   │
│   ├── 📁 features/                      # Signal processing & feature engineering
│   │   │
│   │   ├── 📁 data_processing/           # Preprocessing strategies
│   │   │   ├── Procesamiento_datos_UCI.py          # UCI standard preprocessing
│   │   │   ├── Procesamiento_por_picos.py          # Beat-by-beat (peak-based) segmentation
│   │   │   └── Procesamiento_ventana_fija.py       # Fixed-window segmentation
│   │   │
│   │   └── Detector_de_picos.py          # Peak detection algorithm for PPG/ECG signals
│   │
│   ├── 📁 data/                          # Data loading & dataset management
│   │   │
│   │   └── 📁 data_chargers/             # PyTorch Dataset classes
│   │       ├── Clase_UCIDataset.py       # Base UCI dataset class (memmap-backed)
│   │       ├── MetaDataset.py            # Task dataset for MAML (few-shot episodes)
│   │       ├── PatientWiseSet.py         # Patient-wise task construction for MAML
│   │       ├── TuningDataset.py          # Dataset for few-shot fine-tuning
│   │       ├── IntrapatientSet.py        # Per-patient sequential dataset for clinical eval
│   │       └── PatientWiseSet.py         # Strict patient-level split dataset
│   │
│   ├── 📁 models/                        # Neural network architectures
│   │   ├── ConvolucionalV1.py            # 1D-CNN with ELU activations, BatchNorm & Dropout
│   │   ├── ConvolucionalV2.py            # Lightweight 1D-CNN with reduced regularization
│   │   └── InceptionTime.py             # InceptionTime with multi-scale blocks & residual connections
│   │
│   ├── 📁 utils/                         # General utilities
│   │   └── 📁 Tools/
│   │       └── Tools.py                  # Shared helper functions
│   │
│   └── 📁 entrenamiento/                 # Standard supervised training
│       ├── Entrenamiento.py              # Training with random 70/20/10 split + AMP + Early Stopping
│       └── Entrenamiento_patient_subject.py  # Training with strict patient-wise split
│
├── 📁 metalearning/                      # Meta-learning pipeline
│   │
│   ├── Metaentrenamiento.py              # MAML meta-training (traditional & patient_wise modes)
│   ├── Metaentrenamiento_delta.py        # Delta Learning meta-training (hybrid MSE+Pearson loss)
│   ├── Fewshot.py                        # Few-shot fine-tuning evaluation on test patients
│   ├── zero_shot.py                      # Zero-shot evaluation (frozen meta-model)
│   ├── Respuesta_escalon.py              # Step-response: single initial calibration experiment
│   ├── Intrapatient_eval.py              # Periodic recalibration during prolonged monitoring
│   ├── Busqueda_Hiperparametros.py       # Grid search over MAML hyperparameters
│   ├── Hiperparametros_fewshot.py        # Grid search over few-shot adaptation learning rates
│   │
│   └── 📁 logs/                          # Meta-training loss logs (CSV per experiment)
│
├── 📁 models/                            # Persisted model weights
│   │
│   ├── 📁 best_models/                   # Best standard (supervised) model checkpoints
│   │   └── best_model.pt
│   │
│   ├── 📁 best_meta_models/              # Best meta-trained model checkpoints
│   │   └── best_meta_model.pt
│   │
│   └── 📁 checkpoints/                   # All training checkpoints (latest + best per experiment)
│       └── (checkpoint_<experiment_name>.pt files)
│
├── 📁 notebooks/                         # Visualization & analysis scripts
│   ├── graficacion_datos.py              # Dataset characterization (random split)
│   ├── Graficacion_datos_meta.py         # Meta-learning dataset characterization
│   ├── Graficacion_perdida.py            # Meta-training loss curves
│   └── Graficacion_combinada.py          # Hyperparameter search visualization
│
├── 📁 Prueba_modelos/                    # Standalone inference & testing
│   └── Prueba_modelo.py                  # Load a checkpoint and run inference on test data
│
├── 📁 resultados_intrapatient/           # Output folder for clinical simulation plots
│   └── (per-experiment subfolders with .png tracking plots per patient)
│
└── 📁 data/                              # Data storage
    ├── 📁 interim/                       # Intermediate processed data
    └── 📁 processed/                     # Final processed datasets
        └── 📁 data_UCI/
            ├── dataset_parte_1_por_picos.pt       # Processed signal chunks (memmap metadata)
            ├── dataset_parte_2_por_picos.pt
            ├── dataset_parte_3_por_picos.pt
            ├── dataset_parte_4_por_picos.pt
            └── few_shot_patient_data.pt           # Held-out test patient IDs for few-shot eval
```

---

## Module Descriptions

### `src/models/`

| File | Architecture | Use Case |
|---|---|---|
| `ConvolucionalV1.py` | 4× Conv1d → 4× Linear, ELU, BN, Dropout(0.5) | Main model for meta-training and few-shot |
| `ConvolucionalV2.py` | 4× Conv1d → 4× Linear, ReLU, reduced BN | Lightweight baseline comparison |
| `InceptionTime.py` | Inception blocks + residual + GAP | Multi-scale temporal feature extraction |

All models accept input tensors of shape `(batch, channels=2, signal_length)` and output `(batch, 2)` for simultaneous SBP and DBP regression.

---

### `src/data/data_chargers/`

| File | Purpose |
|---|---|
| `Clase_UCIDataset.py` | Base dataset backed by memory-mapped `.dat` files for efficient large-scale loading |
| `MetaDataset.py` | Constructs MAML episodes: samples N+K windows per patient as Support + Query sets |
| `PatientWiseSet.py` | Constructs cross-patient MAML episodes: Support and Query from different patients |
| `TuningDataset.py` | Wraps a small per-patient subset for few-shot fine-tuning |
| `IntrapatientSet.py` | Returns all sequential windows from a single patient for clinical evaluation |

---

### `metalearning/`

| File | Purpose |
|---|---|
| `Metaentrenamiento.py` | Main MAML loop. Supports `traditional` (random task) and `patient_wise` (cross-patient) modes |
| `Metaentrenamiento_delta.py` | MAML with Delta Learning: predicts BP deltas from support mean; hybrid loss; step annealing |
| `Fewshot.py` | Applies K-shot fine-tuning on test patients; computes pre/post MAE, RMSE, and improvement rate |
| `zero_shot.py` | Runs the frozen meta-model on unseen patients; no adaptation performed |
| `Respuesta_escalon.py` | Single calibration at t=0 (freeze conv layers, update only dense regressor); tracks performance over full session |
| `Intrapatient_eval.py` | Recalibrates every X minutes; plots SBP/DBP tracking and MAE-over-time per patient |
| `Busqueda_Hiperparametros.py` | Exhaustive grid over `adapt_lr × meta_lr × k_steps × group_size`; saves JSON + CSV results |
| `Hiperparametros_fewshot.py` | Sweeps adaptation learning rates; reports population improvement rate (% patients that improve post-adaptation) |

---

## Data Flow

```
Raw UCI PPG/ECG signals
        │
        ▼
Peak Detection (Detector_de_picos.py)
        │
        ▼
Beat Segmentation → 500-sample windows @ 125 Hz (~4 sec/beat)
        │
        ▼
Z-score Normalization per channel
        │
        ▼
Memory-mapped .dat files + .pt metadata
        │
        ├──► UCIDataset (standard training)
        ├──► MetaDataset / PatientWiseSet (MAML episodes)
        └──► IntrapatientSet (clinical evaluation)
```

---

## Checkpoint Naming Convention

Checkpoints follow the pattern:

```
checkpoint_<experiment_name>.pt   ← latest state (resumable)
best_<experiment_name>.pt         ← best validation loss snapshot
```

Each checkpoint contains: `epoch`, `model_state_dict`, `optimizer_state_dict`, `best_loss`.

---

## Key Hyperparameters

| Parameter | Default | Description |
|---|---|---|
| `shots` | 5 | Support set size per task (few-shot calibration samples) |
| `adapt_lr` | 0.005 | Inner-loop (task-specific) learning rate |
| `meta_lr` | 0.001 | Outer-loop (meta) learning rate |
| `adapt_steps` | 5 | Inner-loop gradient steps per task |
| `tasks_per_batch` | 4 | Number of tasks per meta-gradient update |
| `signal_length` | 500 | Samples per input window (4 sec @ 125 Hz) |
| `alpha` (Delta) | 0.75 | Weight of Pearson loss in hybrid loss function |