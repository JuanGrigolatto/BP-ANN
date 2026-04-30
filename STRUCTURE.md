# Project Structure Documentation

## Directory Hierarchy

```
BP-ANN/
│
├── 📄 README.md                    # Main project documentation
├── 📄 LICENSE                      # Apache 2.0 License
├── 📄 STRUCTURE.md                 # This file
├── 📄 requirements.txt             # Python dependencies
├── 📄 setup.py                     # Package configuration
│
├── 📁 src/                         # Source code (main package)
│   │
│   ├── 📁 features/                # Feature engineering & signal processing
│   │   │
│   │   ├── 📁 data_processing/     # Various preprocessing strategies
│   │   │   ├── Procesamiento_datos_UCI.py      # UCI standard preprocessing
│   │   │   ├── Procesamiento_por_picos.py      # Peak-based feature extraction
│   │   │   └── Procesamiento_ventana_fija.py   # Fixed-window preprocessing
│   │   │
│   │   └── Detector_de_picos.py    # Peak detection algorithm for PPG signals
│   │
│   ├── 📁 data/                    # Data loading and dataset management
│   │   │
│   │   └── 📁 data_chargers/       # Dataset classes and loaders
│   │       ├── Clase_UCIDataset.py # UCI dataset class
│   │       ├── MetaDataset.py      # Meta-learning dataset (few-shot)
│   │       ├── TuningDataset.py    # Fine-tuning dataset class
│   │       ├── IntrapatientSet.py  # Intra-patient dataset for fine-tuning
│   │       └── PatientWiseSet.py   # Patient-wise split dataset
│   │
│   ├── 📁 models/                  # Neural network architectures
│   │   ├── ConvolucionalV1.py      # 1D-CNN with ReLU & Dropout regularization
│   │   ├── ConvolucionalV2.py      # Lightweight 1D-CNN with reduced regularization
│   │   ├── InceptionTime.py        # InceptionTime architecture with residual connections
│   │   └── utils.py                # Model utility functions
│   │
│   ├── 📁 utils/                   # General utilities
│   │   │
│   │   ├── 📁 Tools/               # General helper tools
│   │   │   └── Tools.py            # Utility functions and helpers
│   │   │
│   │   ├── Visualizador_datos.py   # Data visualization utilities
│   │   └── metrics.py              # Evaluation metrics (MAE, RMSE, R²)
│   │
│   └── 📁 entrenamiento/           # Training orchestration
│       ├── Entrenamiento.py        # Main training script with random split
│       ├── Entrenamiento_patient_subject.py  # Training with patient-wise splitting
│       └── config.py               # Configuration management
│
├── 📁 models/                      # Trained model checkpoints
│   │
│   ├── 📁 best_models/             # Best performing models
│   │   └── best_model.pt           # Best standard model weights
│   │
│   ├── 📁 best_meta_models/        # Best meta-learning models
│   │   └── best_meta_model.pt      # Best meta-learning model weights
│   │
│   ├── 📁 checkpoints/             # All saved checkpoints
│   │   └── (checkpoint files)
│   │
│   └── tuning_model.pt             # Pre-trained foundation model for fine-tuning
│
├── 📁 notebooks/                   # Visualization and analysis scripts
│   ├── graficacion_datos.py        # Dataset characterization with random split
│   ├── Graficacion_datos_meta.py   # Meta-learning dataset characterization
│   ├── Graficacion_perdida.py      # Meta-learning loss curves
│   └── Graficacion_combinada.py    # Hyperparameter tuning visualization
│
├── 📁 Prueba_modelos/              # Model testing and evaluation
│   └── Prueba_modelo.py            # Model inference and testing script
│
├── 📁 metalearning/                # Meta-learning outputs and logs
│   └── logs/                       # Meta-learning training logs
│
└── 📁 data/                        # Data storage (raw and processed)
    └── processed/                  # Processed datasets
        └── data_UCI/               # UCI dataset files
```
