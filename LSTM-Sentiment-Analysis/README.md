# LSTM Sentiment Analysis — IMDB Movie Reviews

## Overview
Implementation of a **Long Short-Term Memory (LSTM) network from scratch** for binary sentiment classification using PyTorch. Movie reviews from the IMDB dataset are classified as either **positive** or **negative**. All core components : LSTMCell, LSTMLayer, and SentimentLSTM are implemented manually. Pretrained FastText word embeddings are used for word representation.

---

## Dataset
**IMDB Movie Reviews Dataset** — Subset of [Kaggle IMDB Dataset](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews)

| Property | Value |
|---|---|
| Samples | 10,000 reviews |
| Classes | 2 (Positive, Negative) |
| Class Distribution | Balanced (50/50) |
| Split | 70% Train / 20% Val / 10% Test |

---

## Methods & Implementation

### Preprocessing
- HTML tag removal via BeautifulSoup
- Contraction expansion (e.g. `don't` → `do not`)
- Lowercase conversion, special character removal
- Tokenization and lemmatization via NLTK WordNetLemmatizer
- Stop words kept (removing them reduced performance)
- Padding to `MAX_LEN=200` tokens (above 75th percentile of sequence lengths)

### Word Embeddings
- **FastText** (`fasttext-wiki-news-subwords-300`)  300-dimensional pretrained vectors
- Custom vocabulary built from dataset tokens
- OOV tokens mapped to mean vector of all known embeddings
- `<PAD>` token initialized to zero vector
- Embedding weights set to **trainable** (`freeze=False`) for fine tuning

### LSTM Architecture (from scratch)

#### LSTMCell
Single time step computation with all four gates:
```
Forget gate:     f_t = σ(W·[h_{t-1}, x_t] + b)
Input gate:      i_t = σ(W·[h_{t-1}, x_t] + b)
Candidate state: g_t = tanh(W·[h_{t-1}, x_t] + b)
Output gate:     o_t = σ(W·[h_{t-1}, x_t] + b)
Cell update:     c_t = f_t * c_{t-1} + i_t * g_t
Hidden update:   h_t = o_t * tanh(c_t)
```

#### LSTMLayer
- Stacks multiple LSTMCell layers
- **Bidirectional** : processes sequences forward and backward
- Dropout between layers

#### SentimentLSTM
```
Embedding Layer (FastText pretrained, trainable)
→ LSTMLayer (bidirectional, multi-layer)
→ Final hidden state extraction
→ Fully Connected Layer
→ Sigmoid activation → binary classification
```

### Training
- **Loss:** Binary Cross-Entropy
- **Optimizer:** Adam
- **Batch size:** 64
- **Epochs:** 50+
- Best model saved based on validation accuracy

### Word Vector Visualization
- 300 words selected (sentiment-rich + high frequency)
- **PCA** dimensionality reduction to 2D and 3D
- Semantic clusters visible: positive words group together, negative words group separately
- Domain terms (movie, film, scene) form distinct topical clusters

---

## Results

| Metric | Test Set |
|---|---|
| Accuracy | reported in notebook |
| Precision | reported in notebook |
| Recall | reported in notebook |
| F1-score | reported in notebook |

**Key Findings:**
- Bidirectional LSTM captures both forward and backward context, improving sentiment understanding
- Fine tuning FastText embeddings on the dataset improves performance over frozen embeddings
- Keeping stop words improved accuracy compared to removing them
- PCA visualization confirms FastText captures meaningful semantic relationships between words

---

## LSTM Limitations Discussed
- Vanishing gradient problem in very long sequences
- Sequential computation prevents parallelization (unlike Transformers)
- Memory-intensive for large hidden sizes
- Still practical for low latency, resource-constrained, or streaming environments

---

## Requirements

```
torch
torchvision
numpy
pandas
matplotlib
scikit-learn
gensim
nltk
beautifulsoup4
tqdm
```

Install with:
```bash
pip install torch numpy pandas matplotlib scikit-learn gensim nltk beautifulsoup4 tqdm
```

Download NLTK data:
```python
import nltk
nltk.download("stopwords")
nltk.download("wordnet")
nltk.download("omw-1.4")
```

---

## Project Structure

```
LSTM-Sentiment-Analysis/
├── assignment4_template.ipynb       # Main notebook with code and report
├── subset10000_IMDB_Dataset.csv     # Dataset (10,000 reviews)
└── README.md
```

---



> **Note:** Loading FastText vectors (`fasttext-wiki-news-subwords-300`) requires ~1 GB download on first run. Training on CPU is feasible but slow —> Google Colab with GPU is recommended for faster results.
