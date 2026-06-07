"""
Evaluate BERT model from saved checkpoint
"""

import sys
sys.path.append('.')

import numpy as np
import pandas as pd
import torch
import json
import os

from src.data_loader import SDGDataLoader
from src.evaluation import MultiLabelEvaluator
from models.transformer_models import BERTClassifier

# Set random seeds
np.random.seed(42)
torch.manual_seed(42)

print("="*60)
print("Evaluating BERT Model")
print("="*60)

# 1. Load data
print("\n[1/4] Loading data...")
loader = SDGDataLoader(
    train_path='data/devex_train_clean.csv',
    test_path='data/devex_test_clean.csv',
    use_cleaned=True
)

train_df, test_df = loader.load_data()
text_column = 'text'
train_df, test_df = loader.preprocess_dataset(text_column=text_column)
label_columns = loader.get_label_columns()

X_train, X_val, y_train, y_val = loader.train_val_split(test_size=0.2, random_state=42)
y_train = y_train.astype(float)
y_val = y_val.astype(float)

# Convert text to string
X_train = [str(text) if pd.notna(text) else "" for text in X_train]
X_val = [str(text) if pd.notna(text) else "" for text in X_val]

print(f"Training samples: {len(X_train)}")
print(f"Validation samples: {len(X_val)}")

# 2. Load BERT model
print("\n[2/4] Loading BERT model...")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

bert_model = BERTClassifier(
    model_name='bert-base-uncased',
    num_labels=len(label_columns),
    dropout=0.3
)

checkpoint = torch.load('results/experiment_7/bert/best_transformer_model.pt')
bert_model.load_state_dict(checkpoint['model_state_dict'])
bert_model.to(device)
bert_model.eval()
print("Model loaded successfully!")

# 3. Make predictions
print("\n[3/4] Making predictions...")
from src.trainer import TransformerTrainer

trainer = TransformerTrainer(
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

val_pred = trainer.predict(X_val)
print(f"Predictions shape: {val_pred.shape}")

# 4. Evaluate
print("\n[4/4] Evaluating...")
evaluator = MultiLabelEvaluator(label_names=label_columns)
metrics = evaluator.compute_metrics(y_val, val_pred)

print("\nFinal Metrics:")
print(f"  Hamming Loss: {metrics['hamming_loss']:.4f}")
print(f"  F1 Micro: {metrics['f1_micro']:.4f}")
print(f"  F1 Macro: {metrics['f1_macro']:.4f}")
print(f"  Precision Micro: {metrics['precision_micro']:.4f}")
print(f"  Recall Micro: {metrics['recall_micro']:.4f}")

# Find optimal threshold
optimal_threshold = evaluator.find_optimal_threshold(y_val, val_pred, metric='f1')
metrics_optimal = evaluator.compute_metrics(y_val, val_pred, threshold=optimal_threshold)
print(f"\nWith optimal threshold ({optimal_threshold:.2f}):")
print(f"  Hamming Loss: {metrics_optimal['hamming_loss']:.4f}")
print(f"  F1 Micro: {metrics_optimal['f1_micro']:.4f}")

# Save results
results = {
    'experiment_id': 7,
    'experiment_name': 'BERT (evaluation only)',
    'model': 'bert-base-uncased',
    'results': {
        'hamming_loss': float(metrics['hamming_loss']),
        'hamming_loss_optimal': float(metrics_optimal['hamming_loss']),
        'f1_micro': float(metrics['f1_micro']),
        'f1_micro_optimal': float(metrics_optimal['f1_micro']),
        'f1_macro': float(metrics['f1_macro']),
        'precision_micro': float(metrics['precision_micro']),
        'recall_micro': float(metrics['recall_micro']),
        'optimal_threshold': float(optimal_threshold)
    }
}

with open('results/experiment_7/results.json', 'w') as f:
    json.dump(results, f, indent=2)

print("\n" + "="*60)
print("BERT evaluation complete!")
print("Results saved to results/experiment_7/results.json")
print("="*60)
