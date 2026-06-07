"""
Run Experiment 9: Hyperparameter Tuning on Word2Vec
"""

import sys
sys.path.append('.')

import numpy as np
import pandas as pd
import torch
import json
import os
from itertools import product
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
print("EXPERIMENT 8: Hyperparameter Tuning on Word2Vec")
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

print(f"Training samples: {len(X_train)}")
print(f"Validation samples: {len(X_val)}")

# 2. Define hyperparameter grid
print("\n[2/4] Defining hyperparameter grid...")
param_grid = {
    'vector_size': [100, 200, 300],
    'window': [3, 5, 7],
    'sg': [0, 1],  # 0=CBOW, 1=skip-gram
    'hidden_dims': [[256, 128], [512, 256, 128], [512, 256, 128, 64]],
    'dropout': [0.2, 0.3, 0.5],
    'learning_rate': [0.0005, 0.001, 0.002]
}

# Generate all combinations (limit to 20 to save time)
all_configs = list(product(
    param_grid['vector_size'],
    param_grid['window'],
    param_grid['sg'],
    param_grid['hidden_dims'],
    param_grid['dropout'],
    param_grid['learning_rate']
))

# Randomly sample 5 configurations for faster execution
np.random.shuffle(all_configs)
configs_to_try = all_configs[:5]

print(f"Testing {len(configs_to_try)} configurations...")

# 3. Run experiments
print("\n[3/4] Running hyperparameter search...")
results = []

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
evaluator = MultiLabelEvaluator(label_names=label_columns)

for idx, (vector_size, window, sg, hidden_dims, dropout, lr) in enumerate(tqdm(configs_to_try, desc="Configurations")):
    print(f"\n[{idx+1}/{len(configs_to_try)}] Testing: vector_size={vector_size}, window={window}, sg={sg}, hidden_dims={hidden_dims}, dropout={dropout}, lr={lr}")
    
    try:
        # Tokenize
        X_train_tokens = [text.split() for text in X_train]
        X_val_tokens = [text.split() for text in X_val]
        
        # Train Word2Vec
        word2vec = Word2VecEmbedding(
            vector_size=vector_size,
            window=window,
            min_count=2,
            workers=4,
            sg=sg
        )
        word2vec.train(X_train_tokens)
        
        # Transform
        X_train_w2v = word2vec.transform(X_train_tokens, method='mean')
        X_val_w2v = word2vec.transform(X_val_tokens, method='mean')
        
        # Build model
        model = DeepNN(
            input_dim=X_train_w2v.shape[1],
            hidden_dims=hidden_dims,
            output_dim=y_train.shape[1],
            dropout=dropout,
            use_batch_norm=True
        )
        
        # Train
        train_dataset = TextDataset(X_train_w2v, y_train)
        val_dataset = TextDataset(X_val_w2v, y_val)
        
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=64, shuffle=False)
        
        trainer = Trainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            device=device,
            learning_rate=lr
        )
        
        # Train for 5 epochs (faster for hyperparameter search)
        os.makedirs('results/experiment_8/temp', exist_ok=True)
        model = trainer.train(
            num_epochs=5,
            evaluator=evaluator,
            save_path='results/experiment_8/temp'
        )
        
        # Evaluate
        checkpoint = torch.load('results/experiment_8/temp/best_model.pt')
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
        
        result = {
            'config': {
                'vector_size': vector_size,
                'window': window,
                'sg': sg,
                'hidden_dims': hidden_dims,
                'dropout': dropout,
                'learning_rate': lr
            },
            'hamming_loss': float(metrics['hamming_loss']),
            'f1_micro': float(metrics['f1_micro']),
            'f1_macro': float(metrics['f1_macro']),
            'precision_micro': float(metrics['precision_micro']),
            'recall_micro': float(metrics['recall_micro'])
        }
        
        results.append(result)
        print(f"  Hamming Loss: {metrics['hamming_loss']:.4f}, F1 Micro: {metrics['f1_micro']:.4f}")
        
    except Exception as e:
        print(f"  Error: {e}")
        continue

# 4. Find best configuration
print("\n[4/4] Finding best configuration...")
results_sorted = sorted(results, key=lambda x: x['hamming_loss'])
best_result = results_sorted[0]

print("\nBest Configuration:")
print(f"  vector_size: {best_result['config']['vector_size']}")
print(f"  window: {best_result['config']['window']}")
print(f"  sg: {best_result['config']['sg']}")
print(f"  hidden_dims: {best_result['config']['hidden_dims']}")
print(f"  dropout: {best_result['config']['dropout']}")
print(f"  learning_rate: {best_result['config']['learning_rate']}")
print(f"  Hamming Loss: {best_result['hamming_loss']:.4f}")
print(f"  F1 Micro: {best_result['f1_micro']:.4f}")

# Save all results
with open('results/experiment_8/all_results.json', 'w') as f:
    json.dump(results, f, indent=2)

with open('results/experiment_8/best_config.json', 'w') as f:
    json.dump(best_result, f, indent=2)

print("\n" + "="*60)
print("Experiment 8 completed!")
print("Results saved to results/experiment_8/")
print("="*60)
