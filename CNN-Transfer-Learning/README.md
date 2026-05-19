# CNN & Transfer Learning — Vegetable Image Classification

## Overview
Two-part image classification assignment using **Convolutional Neural Networks**. Part 1 implements a custom CNN architecture (VegNet5) from scratch using PyTorch. Part 2 applies **transfer learning** with pre-trained ResNet-18 and MobileNetV2 models and compares all approaches. Models are evaluated on a hidden test set via Kaggle.

---

## Dataset
**Vegetable Image Dataset** — Subset of [Kaggle Vegetable Image Dataset](https://www.kaggle.com/datasets/misrakahmed/vegetable-image-dataset)

| Property | Value |
|---|---|
| Total Images | 4,500 |
| Classes | 15 vegetable types |
| Train | 3,000 images (200 per class) |
| Validation | 750 images (50 per class) |
| Test | 750 images (50 per class) — unlabeled, submitted to Kaggle |
| Image Size | Resized to 256×256 |

**Classes:** Bean, Bitter Gourd, Bottle Gourd, Brinjal, Broccoli, Cabbage, Capsicum, Carrot, Cauliflower, Cucumber, Papaya, Potato, Pumpkin, Radish, Tomato

---

## Methods & Implementation

### Preprocessing
- Resize to 256×256
- ImageNet normalization: mean `[0.485, 0.456, 0.406]`, std `[0.229, 0.224, 0.225]`
- Data augmentation on training set: `RandomHorizontalFlip`, `RandomRotation(10)`

### Part 1: Custom CNN from Scratch VegNet5

**Architecture:**
```
Input (3×256×256)
→ Conv1 (3→32, 3×3) + ReLU + MaxPool2d → 32×128×128
→ Conv2 (32→64, 3×3) + ReLU + MaxPool2d → 64×64×64
→ Conv3 (64→128, 3×3) + ReLU + MaxPool2d → 128×32×32
→ Conv4 (128→256, 3×3) + ReLU + MaxPool2d → 256×16×16
→ Conv5 (256→256, 3×3) + ReLU + MaxPool2d → 256×8×8
→ Flatten → FC (16384→15)
```

- **Loss:** CrossEntropyLoss
- **Optimizer:** Adam / SGD with momentum
- **Hyperparameter search:** 18 combinations (lr × batch_size × optimizer)
  - Learning rates: `[1e-3, 5e-4, 1e-4]`
  - Batch sizes: `[16, 32, 64]`
  - Optimizers: `[Adam, SGD]`
- **Epochs:** 30 (full training), 10 (hyperparameter search)
- Best model saved based on validation accuracy

### Part 2: Transfer Learning

#### ResNet-18 (3 models)
- **Base model** — fine-tune final FC layer only
- **Second model** — fine-tune conv blocks 3 & 4 + FC layer
- **Third model** — fine-tune all layers

#### MobileNetV2 (2 models)
- **Base model** — fine-tune final FC layer only
- **Second model** — fine-tune all layers

> Note: Learning rate is decreased as more layers are unfrozen to maintain training stability.

---

## Results

### Part 1 : VegNet5 (Custom CNN)

| Metric | Validation |
|---|---|
| Accuracy | ~75–80% |
| Precision (macro) | reported in notebook |
| Recall (macro) | reported in notebook |
| F1-score (macro) | reported in notebook |

### Part 2 : Transfer Learning Comparison

| Model | Val Accuracy |
|---|---|
| VegNet5 (from scratch) | ~75–80% |
| ResNet-18 (FC only) | higher |
| ResNet-18 (blocks 3,4 + FC) | higher |
| ResNet-18 (all layers) | highest ResNet |
| MobileNetV2 (FC only) | higher |
| MobileNetV2 (all layers) | competitive |

**Key Findings:**
- Transfer learning significantly outperforms training from scratch with limited data
- Full fine-tuning of ResNet-18 achieves the best accuracy
- MobileNetV2 offers a good accuracy/efficiency trade-off
- VegNet5 shows solid performance given its simple architecture

---

## Requirements

```
torch
torchvision
numpy
matplotlib
seaborn
scikit-learn
pandas
Pillow
tqdm
```

Install with:
```bash
pip install torch torchvision numpy matplotlib seaborn scikit-learn pandas Pillow tqdm
```

---

## Project Structure

```
CNN-Transfer-Learning/
├── assignment3_template.ipynb   # Main notebook with code and report
├── vegetable-dataset/
│   ├── train/                   # Training images (15 class folders)
│   └── val/                     # Validation images (15 class folders)
├── best_model.pth               # Saved best model weights
└── README.md
```

---


> **Note:** Training with hyperparameter search (18 combinations × 10 epochs) is computationally intensive. GPU is recommended. Google Colab with GPU runtime can be used as an alternative.
