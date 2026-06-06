# SDG 3 Indicator Text Classification

Multi-label text classification of health-related texts into 27 SDG Goal 3 indicator classes.

## Quick Start (Google Colab)

1. Clone the repository: `git clone <repo-url>`
2. Open `SDG3_Classification_Complete.ipynb` in Google Colab
3. Run Section 0 (installs dependencies, downloads GloVe vectors)
4. Runtime → Run All

> Note: GloVe download (~860 MB) happens automatically in Section 0. Colab's T4 GPU is recommended for Experiments 7–12.

## Project Overview

SDG 3 covers Good Health & Well-Being. The dataset (Devex) contains health-sector project descriptions sourced from international development organizations. Each text may belong to multiple SDG 3 indicator classes, making this a multi-label classification problem with 27 binary output labels.

**Primary metric:** Hamming Loss (lower = better). Baseline: 0.045.

## Repository Structure

```
SDG-Indicator-Text-Classification/
├── SDG3_Classification_Complete.ipynb  # Master notebook — runs all 12 experiments top-to-bottom
├── requirements.txt                    # All Python dependencies
│
├── data/
│   ├── Devex_train.csv                 # Raw training data
│   ├── Devex_test_questions.csv        # Raw test data
│   ├── devex_train_clean.csv           # Preprocessed training data (output of Section 1)
│   └── devex_test_clean.csv            # Preprocessed test data
│
├── src/
│   ├── data_loader.py                  # Dataset loading and splitting utilities
│   ├── evaluation.py                   # Hamming loss, F1, and metrics helpers
│   └── trainer.py                      # Training loop shared across experiments
│
├── models/
│   ├── embedding_models.py             # Word2Vec, GloVe, FastText embedding wrappers
│   ├── neural_networks.py              # DeepNN architectures (Exp 7–12)
│   └── transformer_models.py          # BERT fine-tuning wrapper (Exp 8)
│
├── notebooks/
│   ├── person_a_eda_preprocessing.ipynb  # EDA and preprocessing walkthrough (Person A)
│   └── experiment_5_word2vec.ipynb       # Standalone Word2Vec notebook
│
└── results/
    ├── experiment_5/                   # Threshold optimization results
    ├── experiment_6/                   # Class imbalance results
    ├── experiment_7/                   # Word2Vec + DeepNN results
    ├── experiment_8/                   # BERT results
    ├── experiment_comparison.csv       # Cross-experiment metrics table
    ├── experiment_summary.json         # Aggregated summary
    └── test_predictions.csv            # Final predictions on test set
```

## Experiments Summary

| Exp # | Name | Approach | Key Feature |
|-------|------|----------|-------------|
| 1 | LR + TF-IDF Unigrams | Logistic Regression | Baseline unigram features |
| 2 | LR + TF-IDF Bigrams | Logistic Regression | Bigram features |
| 3 | Vocabulary Size Tuning | Logistic Regression | Grid search over `max_features` |
| 4 | Model Comparison | LR vs LinearSVC vs Random Forest | Best classical model selection |
| 5 | Threshold Optimization | Best model from Exp 4 | Per-label decision threshold tuning |
| 6 | Class Imbalance | Best model from Exp 4 | `class_weight='balanced'` |
| 7 | Word2Vec + DeepNN | Neural Network | Pretrained Word2Vec embeddings |
| 8 | BERT | Transformer | `bert-base-uncased` fine-tuned |
| 9 | Word2Vec + DeepNN (Tuned) | Neural Network | Hyperparameter-tuned Word2Vec model |
| 10 | Word2Vec + Class Weights | Neural Network | Class weights in loss function |
| 11 | GloVe + DeepNN | Neural Network | Pretrained GloVe 6B 300d embeddings |
| 12 | FastText + DeepNN | Neural Network | Subword FastText embeddings (trained) |

> Hamming Loss values are generated at runtime — see `results/` after running the notebook.

## Setup (Local)

```bash
git clone <repo-url>
cd SDG-Indicator-Text-Classification
pip install -r requirements.txt
# Download GloVe vectors (required for Exp 11):
wget https://nlp.stanford.edu/data/glove.6B.zip
unzip glove.6B.zip glove.6B.300d.txt
jupyter notebook SDG3_Classification_Complete.ipynb
```

## Team

- **Person A (Elvis):** EDA & Preprocessing (Section 1, `notebooks/person_a_eda_preprocessing.ipynb`)
- **Person B (Modestine):** Classical ML — Experiments 1–6 (Section 2)
- **Person 3 (Patricia):** Deep Learning — Experiments 7–10 (Section 3)
- **Lorita:** GloVe & FastText — Experiments 11–12 (Section 4), branch consolidation

## Dependencies

See `requirements.txt`. Key: `scikit-learn`, `torch`, `transformers`, `gensim`, `nltk`.
