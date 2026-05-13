# 🧠 Brain Tumor Binary Segmentation Pipeline

### Binary Brain Tumor Segmentation from T1-Weighted MRI Using EfficientNet-B4 UNet with SCSE Attention and RMIF-Weighted Focal-Dice Loss

[![Python 3.10+](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![SMP](https://img.shields.io/badge/Segmentation_Models-PyTorch-blue)](https://github.com/qubvel/segmentation_models.pytorch)

---

## Overview

Automated **binary** brain tumor segmentation pipeline for the **BRISC 2025** dataset. Takes T1-weighted MRI scans as input and produces pixel-level binary segmentation maps (tumor vs no tumor).

| Class | Label | Description |
|---|---|---|
| Background | 0 | Healthy tissue (no tumor) |
| Tumor | 1 | Any tumor type (glioma, meningioma, pituitary merged) |

### Key Features
- **EfficientNet-B4** encoder with ImageNet transfer learning
- **SCSE Attention** at decoder skip connections
- **RMIF-Weighted Focal-Dice Loss** for class imbalance
- **Dual-Head Architecture** — joint segmentation + classification
- **3-Channel MRI Input** — Grayscale + CLAHE + Sobel edges
- **Research-Grade Metrics** — Dice, IoU, HD95, ASD, Volume Similarity, ROC-AUC

---

## Project Structure

```
├── Code/
│   ├── dataset.py              # Preprocessing, 3-ch input, augmentation, DataLoader
│   ├── model.py                # UNet + EfficientNet-B4 + SCSE attention (2-class)
│   ├── model_multitask.py      # Dual-Head UNet (segmentation + classification)
│   ├── loss.py                 # Focal Loss, Dice Loss, RMIF class weights
│   ├── metrics.py              # Dice, IoU, F1, confusion matrix, ROC-AUC
│   ├── train.py                # Experiment 1: Focal+Dice loss
│   ├── train_ce.py             # Experiment 2: Weighted CrossEntropy
│   ├── train_multitask.py      # Experiment 3: Multi-task dual-head
│   ├── evaluate.py             # Advanced metrics (HD95, ASD, VolSim)
│   ├── compare_results.py      # Side-by-side experiment comparison
│   ├── step1_exploration.py    # Dataset exploration & class weight computation
│   └── debug_train.py          # Overfitting sanity check (40 images)
├── outputs/                    # Auto-generated training outputs
└── README.md
```

---

## Installation

```bash
git clone https://github.com/yourusername/brain-tumor-segmentation.git
cd brain-tumor-segmentation
pip install -r requirements.txt
```

### Dependencies
```
torch>=2.0
torchvision
segmentation-models-pytorch
albumentations
opencv-python
scikit-learn
numpy
matplotlib
tqdm
optuna
scipy
```

---

## Dataset

**BRISC 2025** — 6,000 T1-weighted MRI images with pixel-level segmentation masks.

Download and extract so the structure is:
```
archive/brisc2025/segmentation_task/
├── train/
│   ├── images/    # .jpg MRI scans
│   └── masks/     # .png segmentation masks
└── test/
    ├── images/
    └── masks/
```

> **Note:** The original masks contain multi-class labels (glioma, meningioma, pituitary). This binary pipeline automatically merges all tumor types into a single "tumor" class (1) vs "background" (0).

---

## Usage

### 1. Set Environment Variables

```bash
# Linux / macOS
export DATASET_ROOT="/path/to/archive/brisc2025"
export OUTPUT_DIR="outputs"

# Windows PowerShell
$env:DATASET_ROOT = "C:\path\to\archive\brisc2025"
$env:OUTPUT_DIR = "outputs"
```

### 2. Explore Dataset (Optional)
```bash
python Code/step1_exploration.py
```

### 3. Train Models

```bash
# Experiment 1: Focal+Dice (primary)
python Code/train.py

# Experiment 2: Weighted CE (control)
python Code/train_ce.py

# Experiment 3: Multi-Task Dual-Head
python Code/train_multitask.py
```

Quick test mode (3 epochs, 100 images):
```bash
python Code/train.py --quick
```

### 4. Evaluate & Compare
```bash
python Code/compare_results.py    # Side-by-side metrics
python Code/evaluate.py           # HD95, ASD, Volume Similarity
```

---

## Running on Kaggle

```python
import os, sys
os.environ["DATASET_ROOT"] = "/kaggle/input/datasets/briscdataset/brisc2025/brisc2025"
os.environ["OUTPUT_DIR"]   = "/kaggle/working/outputs"
sys.path.insert(0, "/kaggle/working/project")

from train import train
train(quick=False)
```

---

## Architecture

```
Input (3, 256, 256)                    
  │  [Gray | CLAHE | Sobel]            
  ▼                                    
┌─────────────────────────┐            
│   EfficientNet-B4       │ ← ImageNet 
│   Encoder (17.5M)       │            
└─────────┬───────────────┘            
          │                            
   ┌──────┴──────┐                     
   ▼             ▼                     
┌────────┐  ┌──────────┐              
│  UNet  │  │ GAP → FC │              
│Decoder │  │ 448→128  │              
│+ SCSE  │  │ → 2 cls  │              
└───┬────┘  └────┬─────┘              
    ▼             ▼                    
 Seg Mask    Cls Label                 
(2,256,256)    (2,)                    
```

### Training Strategy
| Phase | Epochs | Description |
|---|---|---|
| Frozen encoder | 1–5 | Decoder learns; encoder weights protected |
| Full fine-tuning | 6–50 | Encoder at 0.01× LR; cosine annealing |
| Early stopping | — | On tumor Dice, patience=15 |

---

## Evaluation Metrics

All metrics from the DAC_202_report are preserved:

| Category | Metrics |
|---|---|
| **Overall** | Pixel Accuracy, Macro F1, Weighted F1, Mean Dice, Mean IoU, ROC-AUC |
| **Per-Class** | Dice, IoU, Precision, Recall, Specificity (for background and tumor) |
| **Boundary** | Hausdorff Distance (HD), HD95, Average Surface Distance (ASD), Volume Similarity |

---

## Binary Segmentation Changes

This pipeline is adapted from the original 4-class segmentation for binary (tumor vs no tumor):

| Component | Original (4-class) | Binary (2-class) |
|---|---|---|
| `NUM_CLASSES` | 4 | 2 |
| Classes | background, glioma, meningioma, pituitary | background, tumor |
| Mask reading | Class from filename (`_gl_`, `_me_`, `_pi_`) | Threshold > 127 → tumor |
| Model output | (B, 4, H, W) | (B, 2, H, W) |
| Dice Loss | Average over 3 tumor classes | Single tumor class |
| Classification | 4-class (tumor type) | 2-class (tumor present/absent) |
| Stratification | By rarest tumor class | By tumor presence |

---

## References

- Tan & Le, *"EfficientNet: Rethinking Model Scaling for CNNs"*, ICML 2019
- Roy et al., *"Concurrent Spatial and Channel SE in FCNs"*, MICCAI 2018
- Lin et al., *"Focal Loss for Dense Object Detection"*, ICCV 2017
- Chauhan et al., *"LandSeg: RMIF for Land Cover Segmentation"*, 2024
- Yakubovskiy, *"Segmentation Models Pytorch"*, GitHub 2019

---

## License

Academic project — BRISC 2025 Brain Tumor Segmentation Challenge.
