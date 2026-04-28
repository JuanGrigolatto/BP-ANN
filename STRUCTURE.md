# Project Structure Documentation

## Overview

BP-ANN is organized following a modular architecture that separates concerns into distinct layers:
- **Data Layer**: Data loading, preprocessing, and management
- **Feature Layer**: Signal processing and feature extraction
- **Model Layer**: Neural network architectures
- **Training Layer**: Training scripts and utilities
- **Utils Layer**: General utilities and visualization

---

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
│   │       └── TuningDataset.py    # Fine-tuning dataset class
│   │
│   ├── 📁 models/                  # Neural network architectures
│   │   ├── standard_nn.py          # Standard fully-connected networks
│   │   ├── meta_learning.py        # Meta-learning models (MAML, etc)
│   │   └── utils.py                # Model utility functions
│   │
│   ├── 📁 utils/                   # General utilities
│   │   │
│   │   ├── 📁 Tools/               # General helper tools
│   │   ├── Visualizador_datos.py   # Data visualization utilities
│   │   └── metrics.py              # Evaluation metrics (MAE, RMSE, R²)
│   │
│   └── 📁 entrenamiento/           # Training orchestration
│       ├── train.py                # Main training script
│       ├── evaluate.py             # Model evaluation script
│       └── config.py               # Configuration management
│
├── 📁 models/                      # Trained model checkpoints
│   │
│   ├── 📁 best_models/             # Best performing models
│   │   ├── best_model.pt           # Best standard model weights
│   │   ├── best_meta_model.pt      # Best meta-learning model weights
│   │   └── metadata.json           # Training metadata & hyperparameters
│   │
│   ├── model_v1.pt                 # Version history of models
│   └── model_v2.pt
│
├── 📁 notebooks/                   # Jupyter notebooks for analysis
│   ├── Analisis_error.ipynb        # Error analysis and visualization
│   ├── Prueba_recorte.ipynb        # Signal windowing experiments
│   └── prueba_seniales_ruido.ipynb # Noise robustness testing
│
└── 📁 tests/                       # Unit and integration tests
    ├── test_data_processing.py     # Data processing tests
    ├── test_models.py              # Model tests
    └── test_training.py            # Training pipeline tests
```

---

## Module Descriptions

### 🎯 src/features/
**Purpose**: Signal processing and feature extraction from PPG signals

#### Components:
- **Procesamiento_datos_UCI.py**
  - Standard preprocessing following UCI dataset conventions
  - Normalization and standardization
  - Use case: Cross-patient generalization studies

- **Procesamiento_por_picos.py**
  - Extracts features based on detected peaks in PPG signal
  - Peak-to-peak analysis
  - Use case: Interpretable feature representation

- **Procesamiento_ventana_fija.py**
  - Fixed-size sliding window extraction
  - Efficient for real-time processing
  - Use case: Embedded system applications

- **Detector_de_picos.py**
  - Peak detection algorithm implementation
  - Handles noisy PPG signals
  - Methods: Peak prominence, gradient-based detection

**Example Usage**:
```python
from src.features.data_processing import Procesamiento_ventana_fija
from src.features.Detector_de_picos import PeakDetector

processor = Procesamiento_ventana_fija(window_size=128, stride=64)
detector = PeakDetector(min_distance=20)

features = processor.process(raw_signal)
peaks = detector.detect(raw_signal)
```

---

### 📊 src/data/
**Purpose**: Data loading, dataset management, and batch creation

#### Components:
- **Clase_UCIDataset.py**
  - Loads UCI PPG dataset
  - Implements PyTorch Dataset interface
  - Handles train/val/test splits
  - Features: Caching, augmentation options

- **MetaDataset.py**
  - Creates episodic tasks for meta-learning
  - N-way k-shot sampling
  - Supports few-shot learning scenarios
  - Hyperparameters: n_way, k_shot, n_query

- **TuningDataset.py**
  - Dataset for patient-specific fine-tuning
  - Stratified sampling
  - Balanced label distribution

**Example Usage**:
```python
from src.data.data_chargers import Clase_UCIDataset, MetaDataset
import torch

# Standard dataset
dataset = Clase_UCIDataset(
    path='data/raw/datos/',
    split='train',
    normalize=True
)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)

# Meta-learning dataset
meta_dataset = MetaDataset(
    n_way=5,
    k_shot=5,
    n_query=15,
    episodes=1000
)
```

---

### 🧠 src/models/
**Purpose**: Neural network architectures for BP estimation

#### Components:
- **standard_nn.py**
  - Fully-connected neural networks
  - Architectures: 1-layer, 2-layer, 3-layer MLPs
  - Activation functions: ReLU, Tanh
  - Regularization: Dropout, BatchNorm

- **meta_learning.py**
  - Meta-learning implementations
  - Algorithms: MAML (Model-Agnostic Meta-Learning)
  - Fast adaptation to new tasks
  - Inner and outer loop optimization

- **utils.py**
  - Model initialization
  - Weight management
  - Parameter counting

**Example Usage**:
```python
from src.models.standard_nn import MLPRegressor
from src.models.meta_learning import MAML

# Standard model
model = MLPRegressor(
    input_size=128,
    hidden_sizes=[256, 128, 64],
    output_size=2  # SBP, DBP
)

# Meta-learning model
meta_model = MAML(
    model=model,
    inner_lr=0.01,
    outer_lr=0.001
)
```

---

### 🛠️ src/utils/
**Purpose**: Utility functions and general helpers

#### Components:
- **Visualizador_datos.py**
  - Signal visualization
  - Plot PPG signals with annotations
  - Error distribution plots
  - Bland-Altman agreement plots

- **metrics.py**
  - MAE (Mean Absolute Error)
  - RMSE (Root Mean Squared Error)
  - R² (Coefficient of Determination)
  - Correlation analysis

- **Tools/**
  - Helper functions
  - Data utilities
  - Configuration management

**Example Usage**:
```python
from src.utils.metrics import calculate_metrics
from src.utils.Visualizador_datos import plot_signal

metrics = calculate_metrics(predictions, targets)
print(f"MAE: {metrics['mae']:.2f} mmHg")

plot_signal(raw_signal, processed_signal, predictions)
```

---

### 🏋️ src/entrenamiento/
**Purpose**: Training orchestration and model optimization

#### Components:
- **train.py**
  - Main training loop
  - Loss computation
  - Gradient updates
  - Model checkpointing

- **evaluate.py**
  - Model evaluation on test set
  - Metric computation
  - Result logging

- **config.py**
  - Hyperparameter management
  - Training configuration
  - Dataset configuration

**Example Usage**:
```bash
# From command line
python -m src.entrenamiento.train --config config.yaml

# From Python
from src.entrenamiento.train import train_model
train_model(config='configs/train_config.yaml')
```

---

### 🤖 models/
**Purpose**: Storage of trained model weights and metadata

#### File Types:
- **.pt files**: PyTorch model state dictionaries
- **metadata.json**: Training hyperparameters, epoch count, loss values
- **training_log.csv**: Detailed training metrics per epoch

**Organization**:
```
models/
├── best_models/
│   ├── best_model.pt              # Best overall model
│   ├── best_meta_model.pt         # Best meta-learning model
│   └── metadata.json              # Training details
└── checkpoints/                   # All saved checkpoints
    ├── epoch_10.pt
    ├── epoch_20.pt
    └── ...
```

---

### 📓 notebooks/
**Purpose**: Interactive analysis and experimentation

#### Notebooks:
1. **Analisis_error.ipynb**
   - Error distribution analysis
   - Statistical tests
   - Outlier detection

2. **Prueba_recorte.ipynb**
   - Window size optimization
   - Signal truncation experiments
   - Performance vs. computation trade-off

3. **prueba_seniales_ruido.ipynb**
   - Noise robustness evaluation
   - Synthetic noise injection
   - Noise type comparison

---

### ✅ tests/
**Purpose**: Automated testing and quality assurance

#### Test Modules:
- **test_data_processing.py**
  - Data loading correctness
  - Preprocessing output shapes
  - Normalization validation

- **test_models.py**
  - Model instantiation
  - Forward pass shapes
  - Gradient computation

- **test_training.py**
  - Training loop execution
  - Loss decrease over epochs
  - Checkpoint saving/loading

**Running Tests**:
```bash
# Run all tests
pytest tests/

# With coverage
pytest --cov=src tests/

# Specific test
pytest tests/test_models.py::test_forward_pass
```

---

## Data Flow Diagram

```
Raw PPG Signal
      ↓
┌─────────────────────────────────────┐
│  src/features/                      │
│  • Detector_de_picos.py            │
│  • data_processing/                 │
└─────────────────────────────────────┘
      ↓
Processed Features
      ↓
┌─────────────────────────────────────┐
│  src/data/                          │
│  • Dataset Classes                  │
│  • DataLoaders                      │
└─────────────────────────────────────┘
      ↓
Batched Data
      ↓
┌─────────────────────────────────────┐
│  src/models/                        │
│  • Neural Network Forward Pass      │
└─────────────────────────────────────┘
      ↓
Predictions (SBP, DBP)
      ↓
┌─────────────────────────────────────┐
│  src/utils/                         │
│  • Metrics Computation              │
│  • Visualization                    │
└─────────────────────────────────────┘
      ↓
Results & Analysis
```

---

## Naming Conventions

| Type | Convention | Example |
|------|-----------|---------|
| Python files | snake_case | `train.py`, `peak_detector.py` |
| Classes | PascalCase | `MLPRegressor`, `MetaDataset` |
| Functions | snake_case | `calculate_metrics()`, `plot_signal()` |
| Constants | UPPER_CASE | `WINDOW_SIZE = 128`, `LEARNING_RATE = 0.001` |
| Directories | snake_case | `data_processing/`, `best_models/` |
| Models (Spanish) | Snake_Case_Spanish | `Procesamiento_ventana_fija.py` |

---

## Best Practices for Adding Code

### 1. Feature Processing
- Always normalize/standardize features
- Document preprocessing steps
- Provide preprocessing metadata

### 2. Dataset Classes
- Inherit from `torch.utils.data.Dataset`
- Implement `__len__()` and `__getitem__()`
- Include data validation

### 3. Models
- Add docstrings with architecture description
- Include input/output shape documentation
- Implement `forward()` method

### 4. Training
- Log metrics at regular intervals
- Save checkpoints periodically
- Validate on separate set

### 5. Utilities
- Add type hints
- Include docstrings with examples
- Unit test utility functions

---

## Common Workflows

### 1. Training a Standard Model
```
src/entrenamiento/train.py
  ├── Load config
  ├── Create dataset → src/data/
  ├── Initialize model → src/models/
  ├── Training loop
  │   ├── Get batch
  │   ├── Forward pass
  │   ├── Compute loss
  │   ├── Backward pass
  │   └── Update weights
  ├── Validate on val_set
  └── Save best model → models/
```

### 2. Meta-Learning Pipeline
```
MetaDataset (src/data/)
  ├── Sample N-way k-shot task
  ├── Split to support/query sets
  ├── Inner loop: Adapt on support set
  ├── Outer loop: Update on query set
  └── Repeat for multiple episodes
```

### 3. Model Evaluation
```
src/entrenamiento/evaluate.py
  ├── Load trained model
  ├── Create test dataset
  ├── Get predictions
  ├── Compute metrics → src/utils/metrics.py
  └── Generate plots → src/utils/Visualizador_datos.py
```

---

## Troubleshooting

### Data Issues
- Ensure data files are in correct format (.mat, .dat, .npy)
- Check file paths in config
- Verify data normalization

### Model Issues
- Check input/output dimensions match
- Verify gradient flow with small dataset
- Inspect model parameters count

### Training Issues
- Adjust learning rate if training diverges
- Check for GPU memory issues
- Verify loss decreases over epochs

---

## Contributing Guidelines

When adding new code:
1. Follow naming conventions
2. Add docstrings and type hints
3. Write tests for new functionality
4. Update this documentation
5. Ensure code is well-commented

---

## References

- **Dataset**: UCI Machine Learning Repository - PPG-Dalia Wearable Dataset
- **Framework**: PyTorch Official Documentation
- **Meta-Learning**: Finn et al., "Model-Agnostic Meta-Learning for Fast Adaptation"
- **Signal Processing**: Numpy/Scipy Documentation

---

<div align="center">

**Last Updated**: April 2025

For questions or clarifications, please refer to README.md or contact the maintainer.

</div>
