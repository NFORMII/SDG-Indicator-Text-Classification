"""
Run Experiment 7: BERT/RoBERTa Transformer Models
"""

import sys
sys.path.append('.')

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import json
import os
from tqdm import tqdm

from src.data_loader import SDGDataLoader
from src.evaluation import MultiLabelEvaluator
from src.trainer import TransformerTrainer
from models.transformer_models import BERTClassifier, RoBERTaClassifier
from transformers import BertTokenizer, RobertaTokenizer

# Set random seeds
np.random.seed(42)
torch.manual_seed(42)

print("="*60)
print("EXPERIMENT 7: BERT/RoBERTa Transformer Models")
print("="*60)

# 1. Load data
print("\n[1/5] Loading data...")
loader = SDGDataLoader(
    train_path='data/devex_train_clean.csv',
    test_path='data/devex_test_clean.csv',
    use_cleaned=True
)

train_df, test_df = loader.load_data()

# Find text column
text_column = 'text'
if text_column not in train_df.columns:
    text_candidates = [col for col in train_df.columns if 'text' in col.lower() or 'description' in col.lower()]
    if text_candidates:
        text_column = text_candidates[0]
        print(f"Using column: {text_column}")

# 2. Preprocess
print("\n[2/5] Preprocessing text...")
train_df, test_df = loader.preprocess_dataset(text_column=text_column)
label_columns = loader.get_label_columns()
print(f"Number of labels: {len(label_columns)}")

X_train, X_val, y_train, y_val = loader.train_val_split(test_size=0.2, random_state=42)
print(f"Training samples: {len(X_train)}")
print(f"Validation samples: {len(X_val)}")

# Convert labels to float
y_train = y_train.astype(float)
y_val = y_val.astype(float)

# Convert text to string and handle NaN values
X_train = [str(text) if pd.notna(text) else "" for text in X_train]
X_val = [str(text) if pd.notna(text) else "" for text in X_val]

# 3. Train BERT
print("\n[3/5] Training BERT...")
print("Loading BERT tokenizer...")
bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

print("Creating BERT model...")
bert_model = BERTClassifier(
    model_name='bert-base-uncased',
    num_labels=len(label_columns),
    dropout=0.3
)

# Create datasets
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

evaluator = MultiLabelEvaluator(label_names=label_columns)

bert_trainer = TransformerTrainer(
    model=bert_model,
    train_texts=X_train,
    val_texts=X_val,
    train_labels=y_train,
    val_labels=y_val,
    batch_size=16,
    max_length=256,
    device=device,
    learning_rate=2e-5
)

os.makedirs('results/experiment_7', exist_ok=True)
bert_model = bert_trainer.train(
    num_epochs=5,
    evaluator=evaluator,
    save_path='results/experiment_7/bert'
)

# Evaluate BERT
print("\nEvaluating BERT...")
checkpoint = torch.load('results/experiment_7/bert/best_transformer_model.pt')
bert_model.load_state_dict(checkpoint['model_state_dict'])

bert_pred = bert_trainer.predict(X_val)
bert_metrics = evaluator.compute_metrics(y_val, bert_pred)
print(f"\nBERT Results:")
print(f"  Hamming Loss: {bert_metrics['hamming_loss']:.4f}")
print(f"  F1 Micro: {bert_metrics['f1_micro']:.4f}")
print(f"  F1 Macro: {bert_metrics['f1_macro']:.4f}")

# 4. Train RoBERTa (if time permits)
print("\n[4/5] Training RoBERTa...")
try:
    roberta_model = RoBERTaClassifier(
        model_name='roberta-base',
        num_labels=len(label_columns),
        dropout=0.3
    )
    
    roberta_trainer = TransformerTrainer(
        model=roberta_model,
        train_texts=X_train,
        val_texts=X_val,
        train_labels=y_train,
        val_labels=y_val,
        batch_size=16,
        max_length=256,
        device=device,
        learning_rate=2e-5
    )
    
    roberta_model = roberta_trainer.train(
        num_epochs=5,
        evaluator=evaluator,
        save_path='results/experiment_7/roberta'
    )
    
    # Evaluate RoBERTa
    print("\nEvaluating RoBERTa...")
    checkpoint = torch.load('results/experiment_7/roberta/best_transformer_model.pt')
    roberta_model.load_state_dict(checkpoint['model_state_dict'])
    
    roberta_pred = roberta_trainer.predict(X_val)
    roberta_metrics = evaluator.compute_metrics(y_val, roberta_pred)
    print(f"\nRoBERTa Results:")
    print(f"  Hamming Loss: {roberta_metrics['hamming_loss']:.4f}")
    print(f"  F1 Micro: {roberta_metrics['f1_micro']:.4f}")
    print(f"  F1 Macro: {roberta_metrics['f1_macro']:.4f}")
    
    roberta_results = {
        'model': 'roberta',
        'hamming_loss': float(roberta_metrics['hamming_loss']),
        'f1_micro': float(roberta_metrics['f1_micro']),
        'f1_macro': float(roberta_metrics['f1_macro']),
        'precision_micro': float(roberta_metrics['precision_micro']),
        'recall_micro': float(roberta_metrics['recall_micro'])
    }
except Exception as e:
    print(f"RoBERTa training failed: {e}")
    roberta_results = None

# 5. Save results
results = {
    'experiment_id': 7,
    'experiment_name': 'BERT/RoBERTa Transformer Models',
    'models': {
        'bert': {
            'model': 'bert-base-uncased',
            'hamming_loss': float(bert_metrics['hamming_loss']),
            'f1_micro': float(bert_metrics['f1_micro']),
            'f1_macro': float(bert_metrics['f1_macro']),
            'precision_micro': float(bert_metrics['precision_micro']),
            'recall_micro': float(bert_metrics['recall_micro'])
        }
    }
}

if roberta_results:
    results['models']['roberta'] = roberta_results

with open('results/experiment_7/results.json', 'w') as f:
    json.dump(results, f, indent=2)

print("\n" + "="*60)
print("Experiment 7 completed!")
print("Results saved to results/experiment_7/")
print("="*60)
