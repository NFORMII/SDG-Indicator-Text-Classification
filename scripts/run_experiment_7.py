"""
Run Experiment 7: Word2Vec + Deep Neural Network
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
from src.trainer import Trainer, TextDataset
from models.neural_networks import DeepNN
from models.embedding_models import Word2VecEmbedding

# Set random seeds
np.random.seed(42)
torch.manual_seed(42)

print("="*60)
print("EXPERIMENT 5: Word2Vec + Deep Neural Network")
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
print(f"Training samples: {len(X_train)}")
print(f"Validation samples: {len(X_val)}")

# Convert labels to float
y_train = y_train.astype(float)
y_val = y_val.astype(float)

# 2. Train Word2Vec
print("\n[2/4] Training Word2Vec embeddings...")
X_train_tokens = [text.split() for text in X_train]
X_val_tokens = [text.split() for text in X_val]

word2vec = Word2VecEmbedding(
    vector_size=300,
    window=5,
    min_count=2,
    workers=4,
    sg=1  # Skip-gram
)

word2vec.train(X_train_tokens)
print(f"Vocabulary size: {len(word2vec.model.wv)}")

# 3. Transform text to Word2Vec vectors
print("\n[3/4] Transforming text to Word2Vec vectors...")
X_train_w2v = word2vec.transform(X_train_tokens, method='mean')
X_val_w2v = word2vec.transform(X_val_tokens, method='mean')
print(f"Embedding shape: {X_train_w2v.shape}")

# 4. Build and train neural network
print("\n[4/4] Building and training neural network...")
input_dim = X_train_w2v.shape[1]
hidden_dims = [512, 256, 128]
output_dim = y_train.shape[1]
dropout = 0.3
learning_rate = 0.001
batch_size = 64
num_epochs = 30

model = DeepNN(
    input_dim=input_dim,
    hidden_dims=hidden_dims,
    output_dim=output_dim,
    dropout=dropout,
    use_batch_norm=True
)

train_dataset = TextDataset(X_train_w2v, y_train)
val_dataset = TextDataset(X_val_w2v, y_val)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

evaluator = MultiLabelEvaluator(label_names=label_columns)

trainer = Trainer(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    device=device,
    learning_rate=learning_rate
)

os.makedirs('results/experiment_5', exist_ok=True)
model = trainer.train(
    num_epochs=num_epochs,
    evaluator=evaluator,
    save_path='results/experiment_5'
)

# 5. Evaluate
checkpoint = torch.load('results/experiment_5/best_model.pt')
model.load_state_dict(checkpoint['model_state_dict'])

model.eval()
with torch.no_grad():
    val_predictions = []
    for texts, _ in val_loader:
        texts = texts.to(device)
        outputs = model(texts)
        val_predictions.append(outputs.cpu().numpy())
    val_pred = np.vstack(val_predictions)

metrics = evaluator.compute_metrics(y_val, val_pred)
print("\nFinal Metrics:")
print(f"  Hamming Loss: {metrics['hamming_loss']:.4f}")
print(f"  F1 Micro: {metrics['f1_micro']:.4f}")
print(f"  F1 Macro: {metrics['f1_macro']:.4f}")

# Find optimal threshold
optimal_threshold = evaluator.find_optimal_threshold(y_val, val_pred, metric='f1')
metrics_optimal = evaluator.compute_metrics(y_val, val_pred, threshold=optimal_threshold)
print(f"\nWith optimal threshold ({optimal_threshold:.2f}):")
print(f"  Hamming Loss: {metrics_optimal['hamming_loss']:.4f}")
print(f"  F1 Micro: {metrics_optimal['f1_micro']:.4f}")

# Plot learning curves
evaluator.plot_learning_curves()

# Save results
results = {
    'experiment_id': 5,
    'experiment_name': 'Word2Vec + Deep Neural Network',
    'model_config': {
        'embedding': 'Word2Vec',
        'vector_size': 300,
        'window': 5,
        'sg': 1,
        'hidden_dims': hidden_dims,
        'dropout': dropout,
        'use_batch_norm': True
    },
    'training_config': {
        'learning_rate': learning_rate,
        'batch_size': batch_size,
        'num_epochs': num_epochs,
        'optimizer': 'Adam'
    },
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

with open('results/experiment_5/results.json', 'w') as f:
    json.dump(results, f, indent=2)

print("\n" + "="*60)
print("Experiment 5 completed!")
print("Results saved to results/experiment_5/")
print("="*60)
