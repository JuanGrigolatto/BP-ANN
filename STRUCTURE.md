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
│   │   ├── best_model.pt           # Best standard model weights
│   │   ├── best_meta_model.pt      # Best meta-learning model weights
│   │   └── metadata.json           # Training metadata & hyperparameters
│   │
│   ├── model_v1.pt                 # Version history of models
��   └── model_v2.pt
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
  - Loads UCI PPG dataset from multiple .pt files
  - Memory-mapped arrays for efficient large-scale data loading
  - Implements PyTorch Dataset interface
  - Returns: (signal, labels, patient_id, index)
  - Features: Worker initialization, batch prefetching

- **MetaDataset.py**
  - Creates episodic tasks for meta-learning
  - N-way k-shot sampling
  - Supports few-shot learning scenarios
  - Hyperparameters: n_way, k_shot, n_query

- **TuningDataset.py**
  - Dataset for patient-specific fine-tuning
  - Stratified sampling
  - Balanced label distribution

- **IntrapatientSet.py**
  - Intra-patient dataset for personalized model adaptation
  - Supports leave-one-out cross-validation
  - Use case: Patient-specific model fine-tuning

- **PatientWiseSet.py**
  - Implements patient-wise data splitting
  - Prevents data leakage by ensuring patient signals don't cross train/val/test
  - Supports episodic patient-based sampling
  - Use case: Rigorous evaluation of generalization to new patients

**Example Usage**:
```python
from src.data.data_chargers import Clase_UCIDataset, PatientWiseSet
import torch

# Standard dataset with UCIDataset
dataset = Clase_UCIDataset(
    pt_files=['data/processed/data_UCI/dataset_parte_1.pt',
              'data/processed/data_UCI/dataset_parte_2.pt']
)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=256, shuffle=True)

# Patient-wise dataset splitting
patient_dataset = PatientWiseSet(
    pt_files=['data/processed/data_UCI/dataset_parte_1.pt'],
    train_ratio=0.7,
    val_ratio=0.2,
    seed=42
)
```

---

### 🧠 src/models/
**Purpose**: Neural network architectures for BP estimation from PPG/ECG signals

#### Components:
- **ConvolucionalV1.py** - `Modelo_ConvolucionalV1`
  - Standard 1D-CNN architecture
  - 4 convolutional layers with BatchNorm
  - 4 dense layers with Dropout (0.5) for regularization
  - Activations: ReLU (convolutional), ELU (dense)
  - Output: 2 values (Systolic and Diastolic BP)
  - Configuration: Input window = 250 or 500 samples

- **ConvolucionalV2.py** - `Modelo_ConvolucionalV2`
  - Lightweight 1D-CNN variant
  - Reduced regularization compared to V1
  - Eliminates Dropout in dense block
  - Replaces ELU with ReLU for regression
  - 4 convolutional layers (reduced BatchNorm)
  - Suitable for embedded systems and real-time inference

- **InceptionTime.py** - `InceptionTime`
  - Advanced architecture with Inception modules
  - Multiple parallel convolutional branches (3 conv + 1 maxpool)
  - Residual connections every 3 modules for gradient flow
  - Global Average Pooling (GAP) for feature reduction
  - Depth: 6 inception blocks with n_filters=32
  - State-of-the-art performance on time series tasks

- **utils.py**
  - Model initialization utilities
  - Weight management
  - Parameter counting

**Example Usage**:
```python
from src.models.ConvolucionalV1 import Modelo_ConvolucionalV1
from src.models.ConvolucionalV2 import Modelo_ConvolucionalV2
from src.models.InceptionTime import InceptionTime
import torch

# ConvolucionalV1 - Standard CNN
model_v1 = Modelo_ConvolucionalV1(
    in_channels=2,      # PPG + ECG
    out_channels=2,     # SBP + DBP
    long_signal=250     # window size
)

# ConvolucionalV2 - Lightweight variant
model_v2 = Modelo_ConvolucionalV2(
    in_channels=2,
    out_channels=2,
    long_signal=500
)

# InceptionTime - Advanced architecture
inception_model = InceptionTime(
    c_in=2,           # input channels
    c_out=2,          # output channels
    n_filters=32,     # filters per inception module
    depth=6           # number of inception blocks
)

# Training example
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = inception_model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = torch.nn.MSELoss()
```

---

### 🛠️ src/utils/
**Purpose**: Utility functions and general helpers

#### Components:
- **Tools/Tools.py**
  - Helper functions for data manipulation
  - Configuration utilities
  - Path management

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
- **Entrenamiento.py** - Random Split Training
  - Main training loop with mixed precision (AMP)
  - Random train/val/test split (70/20/10)
  - Early stopping based on validation loss improvement
  - Checkpoint saving for best models
  - Dynamic test dataset saving to memory-mapped files
  - Scheduler: ReduceLROnPlateau with min_lr threshold
  - Features:
    - Gradient accumulation with GradScaler
    - Persistent workers for efficient data loading
    - Prefetch factor for batch optimization
    - Comprehensive logging to CSV

- **Entrenamiento_patient_subject.py** - Patient-Wise Split Training
  - Advanced training with Patient-Subject Splitting (PSS)
  - Prevents data leakage by isolating patient samples
  - Uses scikit-learn's train_test_split for stratified patient division
  - Train/Val/Test split by unique patients (not samples)
  - Same optimization features as Entrenamiento.py
  - Better evaluation of model generalization to new patients
  - Includes patient ID tracking and metadata logging
  - Recommended for clinical validation

- **config.py**
  - Hyperparameter management
  - Training configuration templates
  - Dataset configuration

**Key Features (Both Training Scripts)**:
- Mixed precision training (torch.amp.autocast)
- Gradient scaling for numerical stability
- Early stopping with patience mechanism
- Model checkpointing with best validation loss tracking
- Training logs saved to CSV for analysis
- Reproducible results with seed management
- GPU acceleration with CUDA support
- ReduceLROnPlateau scheduler for adaptive learning rates

**Example Usage**:
```bash
# Random split training
python -m src.entrenamiento.Entrenamiento

# Patient-wise split training (recommended)
python -m src.entrenamiento.Entrenamiento_patient_subject.py

# Training with specific model
# Edit Entrenamiento.py and uncomment desired model:
# model = InceptionTime(c_in=2, c_out=2, seq_len=None, n_filters=32, depth=6)
# model = Modelo_ConvolucionalV1(in_channels=2, out_channels=2, long_signal=250)
# model = Modelo_ConvolucionalV2(in_channels=2, out_channels=2, long_signal=500)
```

**Training Parameters**:
- Batch size: 256
- Optimizer: Adam (lr=1e-3, weight_decay=1e-4)
- Loss: MSELoss
- Epochs: 200 (with early stopping)
- Patience: 8 epochs
- Min improvement delta: 0.0001

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
│   ├── best_model.pt                              # Best overall model
│   ├── best_model_conv_time32_200_epocas_picos_def_early8.pt
│   ├── best_model_conv_time32_200_epocas_picos_def_early8_ps.pt
│   └── metadata.json                              # Training details
└── checkpoints/                                   # All saved checkpoints
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
Raw PPG Signal (2 channels: PPG + ECG)
      ↓
┌─────────────────────────────────────┐
│  src/features/                      │
│  • Detector_de_picos.py            │
│  • data_processing/                 │
│    (Various preprocessing methods)  │
└─────────────────────────────────────┘
      ↓
Processed Features
      ↓
┌─────────────────────────────────────┐
│  src/data/                          │
│  • UCIDataset (memory-mapped)       │
│  • PatientWiseSet (no data leakage) │
│  • IntrapatientSet (personalized)   │
└─────────────────────────────────────┘
      ↓
Batched Data (batch_size=256)
      ↓
┌─────────────────────────────────────┐
│  src/models/                        │
│  • ConvolucionalV1/V2 (1D-CNN)     │
│  • InceptionTime (Advanced)         │
│  • Forward Pass                     │
└─────────────────────────────────────┘
      ↓
Predictions (SBP, DBP)
      ↓
┌─────────────────────────────────────┐
│  src/utils/                         │
│  • Metrics Computation              │
│  • Visualization & Analysis         │
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
| Model Classes | PascalCase_Spanish | `Modelo_ConvolucionalV1`, `Detector_de_picos` |

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
- Handle memory-mapped arrays for large datasets

### 3. Models
- Add docstrings with architecture description
- Include input/output shape documentation
- Implement `forward()` method
- Test with small batches before full training

### 4. Training
- Log metrics at regular intervals (CSV format)
- Save checkpoints periodically
- Validate on separate set
- Use mixed precision (torch.amp) for efficiency
- Implement early stopping for robustness
- Use Patient-Subject Splitting when evaluating generalization

### 5. Utilities
- Add type hints
- Include docstrings with examples
- Unit test utility functions

---

## Common Workflows

### 1. Training with Random Split
```
src/entrenamiento/Entrenamiento.py
  ├── Load dataset (multiple .pt files)
  ├── Random split: 70% train, 20% val, 10% test
  ├── Create UCIDataset from memory-mapped files
  ├── Initialize model (Inception/ConvV1/ConvV2)
  ├── Training loop per epoch:
  │   ├── Forward pass with mixed precision
  │   ├── Compute MSELoss
  │   ├── Backward pass with GradScaler
  │   ├── Update weights
  │   └── Validate on validation set
  ├── Early stopping if no improvement
  ├── Save best model checkpoint
  └── Generate loss curve plot
```

### 2. Training with Patient-Subject Split (Recommended)
```
src/entrenamiento/Entrenamiento_patient_subject.py
  ├── Load dataset (multiple .pt files)
  ├── Extract unique patient IDs
  ├── Stratified patient-level split:
  │   ├── Split patients: 70% train, 20% val, 10% test
  │   └── No overlap of same patient across splits
  ├── Map patient indices to sample indices
  ├── Create subsets using torch.utils.data.Subset
  ├── Training loop with same optimization as random split
  ├── Better evaluation of true generalization
  └── Save test set metadata for analysis
```

### 3. Model Evaluation
```
Evaluation workflow:
  ├── Load trained model checkpoint
  ├── Load test dataset (isolated samples)
  ├── Perform inference on test samples
  ├── Compute metrics (MAE, RMSE, R²)
  ├── Analyze error distribution
  ├── Generate visualization plots
  └── Save results to CSV
```

---

## Training vs Patient-Subject Split

### Random Split (Entrenamiento.py)
- ✅ Fast training
- ✅ Larger batches per split
- ❌ Data leakage: same patient in train/val/test
- ❌ Optimistic performance metrics
- **Use case**: Proof-of-concept, benchmarking

### Patient-Subject Split (Entrenamiento_patient_subject.py)
- ✅ Prevents data leakage
- ✅ True generalization to new patients
- ✅ Clinically relevant evaluation
- ❌ Smaller batches per patient
- ❌ More realistic but lower metrics
- **Use case**: Clinical validation, model selection, final evaluation

---

## Troubleshooting

### Data Issues
- Ensure data files are in correct format (.pt with metadata)
- Check file paths in configuration
- Verify data normalization and shape (N, 2, segment_length)
- Use `UCIDataset.worker_init()` for multi-worker DataLoaders

### Model Issues
- Check input/output dimensions match
- Verify gradient flow with small batch
- Inspect model parameters count with `print(model)`
- Ensure mixed precision compatibility with model architecture

### Training Issues
- Adjust learning rate if training diverges
- Check GPU memory with `torch.cuda.memory_allocated()`
- Verify loss decreases over epochs (check CSV log)
- Use early stopping to prevent overfitting
- For Patient-Subject Split: expect slightly higher loss than random split

### Memory Issues
- Use memory-mapped arrays (np.memmap) for large datasets
- Reduce batch_size if GPU runs out of memory
- Use persistent_workers=True for DataLoader efficiency
- Enable gradient checkpointing for very deep models

---

## Contributing Guidelines

When adding new code:
1. Follow naming conventions consistently
2. Add docstrings with parameter descriptions
3. Include type hints where applicable
4. Write tests for new functionality
5. Update this STRUCTURE.md documentation
6. Ensure code is well-commented in Spanish/English
7. Test with both small and large datasets
8. Use Patient-Subject Split for any generalization studies

---

## References

- **Dataset**: UCI Machine Learning Repository - PPG-Dalia Wearable Dataset
- **Framework**: PyTorch 2.0+ Official Documentation
- **1D-CNN Architecture**: Time Series Classification Papers
- **InceptionTime**: Fawaz et al., "InceptionTime: Finding AlexNet for Time Series Classification"
- **Meta-Learning**: Finn et al., "Model-Agnostic Meta-Learning for Fast Adaptation"
- **Signal Processing**: Numpy/Scipy Official Documentation
- **Mixed Precision Training**: NVIDIA Automatic Mixed Precision (AMP) Guide

---

<div align="center">

**Last Updated**: April 2026

**Maintainer**: Juan Marcos Grigolatto

For questions or clarifications, please refer to README.md or open an issue.

</div>
