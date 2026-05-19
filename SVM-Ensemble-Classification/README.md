# SVM & Ensemble Classification

## Overview
Two part classification assignment covering **binary classification with Support Vector Machines** and **multi-class classification** using Multinomial Logistic Regression (from scratch), Decision Tree, and XGBoost. Hyperparameter tuning is performed via GridSearchCV with 5-fold cross-validation throughout.

---

## Datasets

### Part 1 — Sonar Dataset (Binary Classification)
[UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/151/connectionist+bench+sonar+mines+vs+rocks)

| Property | Value |
|---|---|
| Samples | 208 |
| Features | 60 (sonar signal energy values) |
| Classes | 2 (Mine vs Rock) |

### Part 2 — Dry Bean Dataset (Multi-Class Classification)
[UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/602/dry+bean+dataset)

| Property | Value |
|---|---|
| Samples | 13,611 |
| Features | 16 (morphological shape features) |
| Classes | 7 (Barbunya, Bombay, Cali, Dermason, Horoz, Seker, Sira) |

**Features:** Area, Perimeter, MajorAxisLength, MinorAxisLength, AspectRatio, Eccentricity, ConvexArea, EquivalentDiameter, Extent, Solidity, Roundness, Compactness, ShapeFactor1–4

---

## Methods & Implementation

### Part 1: Binary Classification with SVM

#### 1.1 Linear SVM (no tuning)
- Pipeline: `StandardScaler` → `LinearSVC(C=5.0)`
- Baseline model for comparison

#### 1.2 Tuned SVM with GridSearchCV
- `StratifiedKFold` with 5 splits
- Kernels tested: **Linear**, **RBF**, **Polynomial**
- Parameter grid:
  - Linear: `C ∈ [0.1, 1, 10, 100]`
  - RBF: `C ∈ [0.1, 1, 10, 100]`, `gamma ∈ [scale, auto, 0.001, 0.01]`
  - Polynomial: `C`, `gamma`, `degree ∈ [2, 3, 4]`

### Part 2: Multi-Class Classification

#### 2.1 Multinomial Logistic Regression (from scratch)
- Softmax activation for multi-class probabilities
- Cross-entropy loss with optional L2 regularization
- Gradient descent weight updates
- GridSearchCV tuning: `learning_rate`, `epochs`, `reg_lambda`
- Fully compatible with sklearn's `GridSearchCV` via `get_params` / `set_params`

#### 2.2 Decision Tree
- sklearn `DecisionTreeClassifier` with GridSearchCV
- Criteria: Gini impurity & Entropy
- Parameter grid: `max_depth`, `min_samples_split`, `min_samples_leaf`

#### 2.3 XGBoost
- `XGBClassifier` with `eval_metric='mlogloss'`
- Parameter grid: `max_depth`, `learning_rate`, `n_estimators`, `subsample`, `colsample_bytree`
- L1/L2 regularization to prevent overfitting

---

## Results

### Part 1 — Sonar Dataset

| Model | Test Accuracy |
|---|---|
| Linear SVM (no tuning) | ~83% |
| Tuned SVM (GridSearchCV) | ~88% |

### Part 2 — Dry Bean Dataset

| Model | Test Accuracy |
|---|---|
| Multinomial Logistic Regression | ~92% |
| Decision Tree | ~91% |
| XGBoost | ~93% |

**Key Findings:**
- Non-linear SVM kernels outperform linear SVM on the Sonar dataset due to complex decision boundaries
- XGBoost achieves the best performance on Dry Beans with strong regularization and boosting
- Logistic Regression provides a strong, interpretable baseline
- Decision Tree is most interpretable but most prone to overfitting

---

## Requirements

```
numpy
pandas
matplotlib
seaborn
scikit-learn
xgboost
ucimlrepo
```

Install with:
```bash
pip install numpy pandas matplotlib seaborn scikit-learn xgboost ucimlrepo
```

---

## Project Structure

```
SVM-Ensemble-Classification/
├── b2230356044.ipynb       # Main notebook with code and report
├── Dry_Bean_Dataset.txt    # Dataset info
└── README.md
```

---


> **Note:** GridSearchCV operations may take several minutes due to large parameter grids. All cells should be run in order from top to bottom.
