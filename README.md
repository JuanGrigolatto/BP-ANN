# BP-ANN: Non-Invasive Blood Pressure Estimation via CNNs and Meta-Learning

<p align="center">
  <img src="https://img.shields.io/badge/License-Apache%202.0-blue.svg" alt="License">
  <img src="https://img.shields.io/badge/Python-3.9+-blue" alt="Python">
  <img src="https://img.shields.io/badge/PyTorch-2.0+-red" alt="PyTorch">
  <img src="https://img.shields.io/badge/learn2learn-MAML-green" alt="learn2learn">
  <img src="https://img.shields.io/badge/Dataset-UCI%20PPG--ECG--BP-orange" alt="Dataset">
  <img src="https://img.shields.io/badge/Status-Final%20Project-blueviolet" alt="Status">
</p>

**BP-ANN** is a research project that implements artificial neural networks for **non-invasive blood pressure estimation** from **Photoplethysmography (PPG)** and **Electrocardiography (ECG)** signals. This work is the Final Project for the Bioengineering degree at the *Universidad Nacional de Entre Ríos (UNER)* and proposes an innovative approach combining traditional 1D-CNN architectures with **meta-learning (MAML)** techniques to achieve fast patient-specific adaptation with minimal calibration data.

---

## 📋 Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [Architecture](#architecture)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Evaluation Paradigms](#evaluation-paradigms)
- [Dataset](#dataset)
- [Results](#results)
- [License](#license)

---

## Overview

Continuous non-invasive blood pressure monitoring is an unsolved clinical challenge. Current gold-standard methods are either invasive (arterial line) or intermittent (cuff-based). This project proposes a deep learning pipeline that:

1. Extracts beat-by-beat raw data representations from synchronized PPG, ECG and ABP signals.
2. **Trains a base model** with a standard supervised approach (random split or patient-wise split).
3. **Meta-trains the model** using MAML so that it learns an initialization that adapts quickly to a new patient's physiology using only a few heartbeats (Support Set).
4. **Evaluates clinically** by simulating real-world scenarios: zero-shot generalization, one-time calibration (step response), and periodic recalibration during prolonged monitoring.

---

## Key Features

- 🧠 **Three neural architectures**: `ConvolucionalV1` (1D-CNN + ELU + Dropout), `ConvolucionalV2` (lightweight 1D-CNN), and `InceptionTime` (multi-scale residual CNN).
- 📊 **Advanced signal processing**: Peak-based segmentation and fixed-window preprocessing for PPG/ECG/ABP signals.
- 🔄 **Meta-learning with MAML**: Three task-construction paradigms: traditional (support and query sets sampled randomly from a single patient's record), patient_wise (tasks formed by mixing signal segments from multiple patients to force cross-subject generalization), and temporal decoupling (sequential samples for the support set and temporally gapped samples for the query set from the same patient).
- ⚡ **Delta Learning**: Predicts blood pressure *variations* relative to a calibration baseline, using a hybrid MSE + Pearson loss to capture both error magnitude and trend correlation.
- 🎯 **SBP and DBP estimation**: Simultaneous prediction of Systolic and Diastolic Blood Pressure.
- 📈 **Exhaustive hyperparameter search**: Automated grid search over MAML inner/outer learning rates, adaptation steps, and task group sizes.
- 🏥 Comprehensive Evaluation: Assessment of static models (random and patient-wise splits) and meta-learning dynamic tracking (zero-shot, one-time initial calibration, and periodic recalibration), all benchmarked against AAMI/ISO 81060-3 continuous measurement standards.

---

## Architecture

### Neural Network Models

| Model | Description | Key Properties |
|---|---|---|
| `ConvolucionalV1` | 4-layer 1D-CNN + 4-layer dense regressor | BatchNorm, Dropout (0.5), ELU activations |
| `ConvolucionalV2` | Lightweight 4-layer 1D-CNN | Reduced regularization, ReLU activations |
| `InceptionTime` | Multi-scale inception blocks + residual connections | Global Average Pooling, configurable depth |

### Meta-Learning Pipeline

```
Base Model (CNN)
      │
      ▼
  MAML Meta-Training
  ┌─────────────────────────────────┐
  │  For each episode (task):        │
  │  ┌──────────────────────────┐   │
  │  │  Support Set (few shots)  │   │
  │  │  → Inner Loop (adapt)     │   │
  │  └──────────────────────────┘   │
  │  ┌──────────────────────────┐   │
  │  │  Query Set                │   │
  │  │  → Outer Loop (meta-grad) │   │
  │  └──────────────────────────┘   │
  └─────────────────────────────────┘
      │
      ▼
  Meta-Trained Model
      │
      ├──► Zero-Shot Evaluation
      ├──► Few-Shot Fine-Tuning (Step Response)
      └──► Periodic Recalibration (Prolonged Monitoring)
```

### Delta Learning

Instead of predicting absolute blood pressure values, the Delta Learning variant predicts **deviations from a patient-specific calibration baseline** (computed as the mean of the support set). This is combined with a **Hybrid Loss**:

```
Loss = (1 - λ) · MSE(pred_delta, true_delta) + λ · (1 - Pearson(pred_delta, true_delta))
```

---

## Project Structure

For detailed directory documentation, see [STRUCTURE.md](./STRUCTURE.md).

```
BP-ANN/
├── src/
│   ├── features/        # Signal preprocessing and peak detection
│   ├── data/            # Dataset classes (UCI, Meta, Few-Shot, Patient-Wise)
│   ├── models/          # CNN architectures (V1, V2, InceptionTime)
│   ├── utils/           # Helper tools
│   └── entrenamiento/   # Training scripts and config
├── metalearning/        # MAML training, few-shot adaptation, evaluation
├── models/              # Saved checkpoints and best models
├── notebooks/           # Visualization and analysis scripts
├── data/                # Raw and processed datasets
├── Prueba_modelos/      # Inference and testing scripts
└── Validation/          # Algorithmic validation for signal processing.
```

---

## Installation

### Prerequisites

- Python 3.9+
- CUDA-compatible GPU (recommended for training)

### Setup

```bash
# 1. Clone the repository
git clone [https://github.com/JuanGrigolatto/BP-ANN.git](https://github.com/JuanGrigolatto/BP-ANN.git)
cd BP-ANN

# 2. Create and activate a virtual environment
python -m venv venv
# On Windows:
venv\Scripts\activate
# On Linux/Mac:
source venv/bin/activate

# 3. Install PyTorch with CUDA 12.1 support (Required for GPU acceleration)
pip install torch==2.5.1+cu121 torchvision torchaudio --index-url [https://download.pytorch.org/whl/cu121](https://download.pytorch.org/whl/cu121)

# 4. Install dependencies 
pip install -r requirements.txt

# 5. Install dependencies and the package in editable mode
pip install -e .
```

---

## Usage

*Note: By default, all training scripts are configured to use the `ConvolucionalV1` architecture and the preprocessed Cuff-Less Blood Pressure (MIMIC II) dataset (PPG + ECG inputs). Modifying architectures or hyperparameters can be done directly within the configuration dictionaries inside each script.*

### 1. Standard Training (Random Split)

```bash
python src/entrenamiento/Entrenamiento.py
```

Trains the baseline CNN using a traditional 70/20/10 random window split across the entire dataset. Uses mixed-precision training (AMP) and early stopping.
Output: The best model weights are saved automatically to models/best_models/.

### 2. Standard Training (Patient-Wise Split)

```bash
python src/entrenamiento/Entrenamiento_patient_subject.py
```

Trains the model using a strict patient-level data split to evaluate static generalization (zero-shot baseline), ensuring no patient appears in both training and test sets.

### 3. Meta-Training with MAML

```bash
python metalearning/Metaentrenamiento.py
```

Meta-trains the model using MAML. Supports two modes configured inside the script:

-traditional: Tasks are formed by random signal windows from a single patient's record.

-patient_wise: Tasks are formed by mixing data from different patients, enforcing cross-subject generalization.

Output: Meta-trained checkpoints are saved to models/best_meta_models/.

### 4. Delta Learning Meta-Training

```bash
python metalearning/Metaentrenamiento_delta.py
```

Meta-trains using the Delta Learning paradigm with a hybrid MSE + Pearson loss and adaptive annealing of inner-loop steps to predict blood pressure variations from a patient-specific calibration baseline.

### 5. Few-Shot Evaluation

```bash
python metalearning/Fewshot.py
```

Evaluates the meta-trained model on held-out test patients by performing fast adaptation using a small support set (default: 5 cardiac cycles), then measuring MAE and RMSE before and after adaptation.

---

## Evaluation Paradigms

### Zero-Shot

Evaluates the meta-model on unseen patients **without any fine-tuning**. Measures the generalization capability of the learned meta-initialization.

```bash
python metalearning/zero_shot.py
```

### Step Response (One-Time Calibration)

Simulates a **single initial calibration** using the first N heartbeats of a patient. The convolutional layers are frozen and only the dense regressor is updated. Evaluates the model's tracking performance for the entire remaining session.

```bash
python metalearning/Respuesta_escalon.py
```

### Periodic Recalibration (Prolonged Monitoring)

Simulates a **continuous monitoring session** where the model is recalibrated every X minutes using a small batch of new measurements. Analyzes error degradation over time and the effectiveness of periodic partial fine-tuning.

```bash
python metalearning/Intrapatient_eval.py
```

---

## Dataset

This project uses the **UCI Blood Pressure dataset** (PPG-BP), which contains simultaneous PPG and invasive arterial blood pressure (ABP) recordings from ICU patients. Signals are sampled at **125 Hz**.

**Preprocessing pipeline:**
1. Adaptive peak detection on ECG (R-peaks for synchronized window segmentation) and invasive ABP signals (for SBP/DBP ground truth extraction).
2. Beat-by-beat segmentation into fixed-length windows (500 samples ≈ 4 seconds)
3. Z-score normalization (SBP: μ=134.02, σ=22.75 mmHg | DBP: μ=63.47, σ=23.69 mmHg)
4. Storage as memory-mapped arrays (`.dat`) with `.pt` metadata for efficient loading

The processed dataset is split at the patient level:
- **70%** training patients
- **15%** validation patients  
- **15%** test patients (held out for few-shot evaluation)

---

## Hyperparameter Search

Two automated grid search scripts are provided:

**MAML hyperparameter search** (meta-training phase):
```bash
python metalearning/Busqueda_Hiperparametros.py
```
Searches over `adapt_lr`, `meta_lr`, `k_adapt_steps`, and patient group sizes. Results saved as JSON and CSV with comparison plots.

**Few-Shot learning rate search** (adaptation phase):
```bash
python metalearning/Hiperparametros_fewshot.py
```
Evaluates different learning rates for intra-patient fine-tuning, measuring pre/post adaptation MAE and the population improvement rate.

---

## Results

> Performance metrics are evaluated against AAMI/ISO 81060-3:2022 clinical standards (Mean Error ≤ ±5 mmHg, SD ≤ 8 mmHg).

| Evaluation Paradigm | SBP (ME ± SD) [mmHg] | DBP (ME ± SD) [mmHg] | AAMI/ISO Compliance |
|---------------------|----------------------|----------------------|---------------------|
| **Random Split** (Baseline V1) | 0.65 ± 6.60 | 0.07 ± 4.67 | **Pass** |
| **Patient-Wise** (Zero-Shot V1) | -1.92 ± 13.51 | -2.75 ± 10.99 | **Fail** (Severe data drift) |
| **Delta Meta-Learning** (5-Shot) | 0.18 ± 10.87 | -0.71 ± 7.48 | **Partial** (DBP Pass, SBP SD > 8) |

### Key Findings
* **Static Model Failure:** Traditional CNNs perform well on random splits but fail to generalize to unseen patients (Patient-Wise), exhibiting a strong regression to the mean.
* **MAML + Delta Learning Success:** Fine-tuning with only 5 support samples successfully eliminated the systematic bias (Mean Error < 1 mmHg for both SBP and DBP).
* **Variance Trade-off:** While Delta Learning tracks relative variations, it increases susceptibility to high-frequency noise, keeping the SBP standard deviation above the 8 mmHg threshold.

---

## Citation

If you use this work in your research, please cite:

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

## License

This project is licensed under the [Apache 2.0 License](./LICENSE).

---

<p align="center">
  Developed as a Final Bioengineering Project · UNER · 2026<br>
  Author: <strong>Juan Marcos Grigolatto</strong>
</p>