# Perceptron Binary Classification — Raisin Dataset

## Overview
Implementation of the **Perceptron Learning Algorithm from scratch** for binary classification. The task is to classify raisin samples as either **Kecimen** (class 1) or **Besni** (class -1) based on their physical characteristics. All core components  the Perceptron model, evaluation metrics, and Fisher's Linear Discriminant  are implemented from scratch without using premade ML libraries.

---

## Dataset
**Raisin Dataset** — [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/850/raisin)

| Property | Value |
|---|---|
| Samples | 900 |
| Features | 7 (all numerical) |
| Classes | 2 (Kecimen, Besni) |
| Class Distribution | Balanced |

**Features:**
- `Area` — Number of pixels within the boundaries of the raisin
- `MajorAxisLength` : Length of the main axis
- `MinorAxisLength` :Length of the small axis
- `Eccentricity` : Eccentricity of the ellipse
- `ConvexArea` : Number of pixels of the convex hull region
- `Extent` :Ratio of the region to the bounding box pixels
- `Perimeter` :Perimeter of the raisin

---

## Methods & Implementation

### Preprocessing
- **Standardization (Z-score normalization)** applied to all features using `StandardScaler`
- Ensures equal feature influence on the Perceptron's weight updates
- 80/20 stratified train-validation split

### Perceptron (from scratch)
- Weight vector and bias initialized to zero
- Iterative weight update rule: `w ← w + η·y·x`, `b ← b + η·y`
- Random shuffling of samples each epoch via `numpy.random`
- Early stopping when convergence is reached (zero errors)
- Hyperparameters: `learning_rate=0.01`, `epochs=5000`, `random_state=42`

### Evaluation Metrics (from scratch)
All metrics implemented without sklearn:
- **Accuracy** — Overall correct predictions ratio
- **Precision** — TP / (TP + FP)
- **Recall** — TP / (TP + FN)
- **F1 Score** — Harmonic mean of Precision and Recall

### Visualization
- **Correlation Matrix Heatmap** — To identify least-correlated feature pairs
- **Decision Boundary Plots** — 2D hyperplane visualization for different feature pairs
  - Highest separability pair
  - Lowest separability pair
  - Average separability pair
- **Fisher's Linear Discriminant (from scratch)** — Projects data to 1D space for class separation analysis
  - Computes within-class scatter matrix
  - Finds optimal projection direction: `w = S⁻¹_within · (μ₁ - μ₀)`
  - Visualizes class distributions via histogram

---

## Results

| Metric | Training | Validation |
|---|---|---|
| Accuracy | ~86% | ~85.28% |
| Precision | — | ~83.33% |
| Recall | — | ~86.71% |
| F1 Score | — | ~84.99% |

**Key Findings:**
- The Perceptron converged successfully, confirming the dataset is linearly separable
- Feature pair with highest separability: `Area` & `Extent` (correlation: 0.01)
- Feature pair with lowest separability: `MinorAxisLength` & `Eccentricity` (correlation: 0.03)
- Fisher's LDA projection provides better class separation than 2D Perceptron boundaries

---

## Requirements

```
numpy
pandas
matplotlib
scikit-learn
openpyxl
ucimlrepo
```

Install with:
```bash
pip install numpy pandas matplotlib scikit-learn openpyxl ucimlrepo
```

---

## Project Structure

```
Perceptron-Binary-Classification/
├── assignment1.ipynb       # Main notebook with code and report
├── Raisin_Dataset.xlsx     # Dataset
└── README.md
```

---
> **Note:** All cells should be run in order from top to bottom.
