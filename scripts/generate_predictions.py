"""
Generate test predictions using the best model (Word2Vec tuned from Experiment 8)
"""

import sys
sys.path.append('.')

import numpy as np
import pandas as pd
import torch
import json
import os

from src.data_loader import SDGDataLoader
from models.neural_networks import DeepNN
from models.embedding_models import Word2VecEmbedding
from src.trainer import TextDataset

# Set random seeds
np.random.seed(42)
torch.manual_seed(42)

print("="*60)
print("Generating Test Predictions")
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

# Use full training data
X_train = train_df['processed_text'].values
y_train = loader.prepare_labels()
y_train = y_train.astype(float)

# Handle test data - it might not have processed_text
if 'processed_text' in test_df.columns:
    X_test = test_df['processed_text'].values
else:
    # Use text column if processed_text doesn't exist
    if 'text' in test_df.columns:
        X_test = test_df['text'].values
    else:
        # Use the first text-like column
        text_cols = [col for col in test_df.columns if 'text' in col.lower()]
        if text_cols:
            X_test = test_df[text_cols[0]].values
        else:
            raise ValueError("No text column found in test data")

print(f"Training samples: {len(X_train)}")
print(f"Test samples: {len(X_test)}")

# 2. Train Word2Vec on full training data
print("\n[2/4] Training Word2Vec on full data...")
X_train_tokens = [text.split() for text in X_train]
X_test_tokens = [text.split() for text in X_test]

word2vec = Word2VecEmbedding(
    vector_size=200,  # Best from Experiment 8
    window=5,
    min_count=2,
    workers=4,
    sg=1
)

word2vec.train(X_train_tokens)
print(f"Vocabulary size: {len(word2vec.model.wv)}")

X_train_w2v = word2vec.transform(X_train_tokens, method='mean')
X_test_w2v = word2vec.transform(X_test_tokens, method='mean')
print(f"Embedding shape: {X_train_w2v.shape}, {X_test_w2v.shape}")

# 3. Load best model configuration and retrain
print("\n[3/4] Training model with best configuration...")
input_dim = X_train_w2v.shape[1]
hidden_dims = [512, 256, 128, 64]  # Best from Experiment 8
dropout = 0.2
learning_rate = 0.001
batch_size = 64
num_epochs = 30

model = DeepNN(
    input_dim=input_dim,
    hidden_dims=hidden_dims,
    output_dim=y_train.shape[1],
    dropout=dropout,
    use_batch_norm=True
)

train_dataset = TextDataset(X_train_w2v, y_train)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

from src.trainer import Trainer
from src.evaluation import MultiLabelEvaluator

evaluator = MultiLabelEvaluator(label_names=label_columns)

# Create a dummy validation loader for training
dummy_val_dataset = TextDataset(X_train_w2v[:100], y_train[:100])
dummy_val_loader = torch.utils.data.DataLoader(dummy_val_dataset, batch_size=batch_size, shuffle=False)

trainer = Trainer(
    model=model,
    train_loader=train_loader,
    val_loader=dummy_val_loader,
    device=device,
    learning_rate=learning_rate
)

os.makedirs('results/final_model', exist_ok=True)
model = trainer.train(
    num_epochs=num_epochs,
    evaluator=evaluator,
    save_path='results/final_model'
)

# 4. Generate predictions
print("\n[4/4] Generating test predictions...")
checkpoint = torch.load('results/final_model/best_model.pt')
model.load_state_dict(checkpoint['model_state_dict'])

model.eval()
with torch.no_grad():
    test_predictions = []
    for texts, _ in train_loader:
        texts = texts.to(device)
        outputs = model(texts)
        test_predictions.append(outputs.cpu().numpy())
    
    # Handle remaining samples
    remaining = len(X_test_w2v) % batch_size
    if remaining > 0:
        start_idx = len(X_test_w2v) - remaining
        remaining_texts = torch.FloatTensor(X_test_w2v[start_idx:]).to(device)
        outputs = model(remaining_texts)
        test_predictions.append(outputs.cpu().numpy())
    
    # For simplicity, predict on test set using batches
    test_predictions = []
    for i in range(0, len(X_test_w2v), batch_size):
        batch_texts = torch.FloatTensor(X_test_w2v[i:i+batch_size]).to(device)
        outputs = model(batch_texts)
        test_predictions.append(outputs.cpu().numpy())
    
    test_pred = np.vstack(test_predictions)

print(f"Predictions shape: {test_pred.shape}")

# Apply threshold (0.5 default)
threshold = 0.5
test_pred_binary = (test_pred >= threshold).astype(int)

# Create submission DataFrame
submission = pd.DataFrame(test_pred_binary, columns=label_columns)
submission.insert(0, 'Unique ID', test_df['Unique ID'].values)

# Save predictions
submission.to_csv('results/test_predictions.csv', index=False)
print("\nTest predictions saved to results/test_predictions.csv")

# Save prediction info
pred_info = {
    'model': 'Word2Vec + DeepNN (Tuned)',
    'threshold': threshold,
    'num_test_samples': len(X_test),
    'num_labels': len(label_columns),
    'label_columns': label_columns
}

with open('results/prediction_info.json', 'w') as f:
    json.dump(pred_info, f, indent=2)

print("\n" + "="*60)
print("Test prediction generation complete!")
print("="*60)
