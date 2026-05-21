# Machine Learning Projects

A curated portfolio of machine learning projects covering classical algorithms, deep learning, natural language processing, transfer learning, and model evaluation using Python.

## Overview

This repository contains hands-on machine learning projects implemented through Jupyter notebooks. The projects demonstrate end-to-end workflows, including data preprocessing, model development, hyperparameter tuning, evaluation, visualization, and result interpretation.

The repository includes both from-scratch algorithm implementations and applied modeling with modern Python machine learning libraries.

## Projects

| Project | Focus Area | Description |
|---|---|---|
| [Perceptron Binary Classification](Perceptron-Binary-Classification/) | Classical ML | Implements the Perceptron Learning Algorithm from scratch for binary classification on the Raisin dataset, including custom evaluation metrics and Fisher's Linear Discriminant analysis. |
| [SVM & Ensemble Classification](SVM-Ensemble-Classification/) | Supervised Learning | Applies SVM models, multinomial logistic regression, decision trees, and XGBoost with GridSearchCV-based hyperparameter tuning. |
| [CNN Transfer Learning](CNN-Transfer-Learning/) | Computer Vision | Builds a custom CNN for vegetable image classification and compares it with transfer learning models such as ResNet-18 and MobileNetV2. |
| [LSTM Sentiment Analysis](LSTM-Sentiment-Analysis/) | Natural Language Processing | Implements an LSTM network from scratch for IMDB movie review sentiment classification using pretrained FastText word embeddings. |

## Key Skills Demonstrated

- Data preprocessing and feature engineering
- Classification model development
- From-scratch implementation of machine learning algorithms
- Deep learning with PyTorch
- Natural language processing with word embeddings
- Computer vision and transfer learning
- Hyperparameter tuning with cross-validation
- Model evaluation using accuracy, precision, recall, and F1-score
- Data visualization and result analysis

## Technology Stack

- Python
- Jupyter Notebook
- NumPy
- Pandas
- Matplotlib
- Seaborn
- Scikit-learn
- PyTorch
- Torchvision
- XGBoost
- NLTK
- Gensim

## Repository Structure

```text
Machine-Learning-Projects/
├── CNN-Transfer-Learning/
│   ├── cnn.ipynb
│   └── README.md
├── LSTM-Sentiment-Analysis/
│   ├── lstm.ipynb
│   ├── subset10000_IMDB_Dataset.csv
│   └── README.md
├── Perceptron-Binary-Classification/
│   ├── perceptron.ipynb
│   └── README.md
├── SVM-Ensemble-Classification/
│   ├── svm.ipynb
│   └── README.md
└── README.md
```

## Getting Started

Clone the repository:

```bash
git clone https://github.com/your-username/Machine-Learning-Projects.git
cd Machine-Learning-Projects
```

Create and activate a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
```

Install the common dependencies:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn torch torchvision xgboost nltk gensim tqdm beautifulsoup4 pillow openpyxl ucimlrepo
```

Open a project notebook:

```bash
jupyter notebook
```

Each project folder includes its own README with more specific details about the dataset, methodology, results, and required packages.

## Notes

- Some projects use datasets from Kaggle or the UCI Machine Learning Repository.
- Deep learning projects may require a GPU for faster training.
- Notebook results may vary slightly depending on hardware, package versions, random seeds, and training configuration.

## License

This repository is intended for educational and portfolio purposes.
