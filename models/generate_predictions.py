"""
Script to generate predictions on the test set using trained models.
"""

import sys
sys.path.append('..')

import numpy as np
import pandas as pd
import torch
import argparse
import json
from tqdm import tqdm

from src.data_loader import SDGDataLoader
from models.transformer_models import BERTClassifier, RoBERTaClassifier
from models.embedding_models import Word2VecEmbedding, GloVeEmbedding
from models.neural_networks import DeepNN
from src.trainer import TextDataset


def load_trained_model(model_type, checkpoint_path, num_labels, device):
    """
    Load a trained model from checkpoint.
    
    Args:
        model_type: Type of model ('bert', 'roberta', 'word2vec', 'glove')
        checkpoint_path: Path to model checkpoint
        num_labels: Number of output labels
        device: Device to load model on
        
    Returns:
        Loaded model
    """
    print(f"Loading {model_type} model from {checkpoint_path}...")
    
    if model_type == 'bert':
        model = BERTClassifier(
            model_name='bert-base-uncased',
            num_labels=num_labels,
            dropout=0.3,
            freeze_bert=False
        )
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        
    elif model_type == 'roberta':
        model = RoBERTaClassifier(
            model_name='roberta-base',
            num_labels=num_labels,
            dropout=0.3,
            freeze_roberta=False
        )
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        
    elif model_type in ['word2vec', 'glove']:
        # Need embedding dimension and hidden dims from checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=device)
        # These should be saved in the checkpoint or need to be specified
        input_dim = 300  # Default, should be from config
        hidden_dims = [512, 256, 128]  # Default, should be from config
        
        model = DeepNN(
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            output_dim=num_labels,
            dropout=0.3,
            use_batch_norm=True
        )
        model.load_state_dict(checkpoint['model_state_dict'])
    
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    model.to(device)
    model.eval()
    print(f"Model loaded successfully!")
    return model


def generate_predictions_bert(model, texts, batch_size=16, max_length=256, device='cpu'):
    """
    Generate predictions using BERT model.
    
    Args:
        model: BERT model
        texts: List of text strings
        batch_size: Batch size for inference
        max_length: Maximum sequence length
        device: Device to run on
        
    Returns:
        Predictions array
    """
    model.eval()
    predictions = []
    
    num_batches = len(texts) // batch_size + 1
    
    for i in tqdm(range(0, len(texts), batch_size), total=num_batches, desc="Generating predictions"):
        batch_texts = texts[i:i+batch_size]
        
        encoded = model.tokenize(batch_texts, max_length=max_length)
        input_ids = encoded['input_ids'].to(device)
        attention_mask = encoded['attention_mask'].to(device)
        
        with torch.no_grad():
            outputs = model(input_ids, attention_mask)
            predictions.append(outputs.cpu().numpy())
    
    return np.vstack(predictions)


def generate_predictions_nn(model, embeddings, batch_size=64, device='cpu'):
    """
    Generate predictions using neural network with pre-computed embeddings.
    
    Args:
        model: Neural network model
        embeddings: Pre-computed embeddings
        batch_size: Batch size for inference
        device: Device to run on
        
    Returns:
        Predictions array
    """
    model.eval()
    predictions = []
    
    dataset = TextDataset(embeddings, np.zeros((len(embeddings), 1)))  # Dummy labels
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    for texts, _ in tqdm(dataloader, desc="Generating predictions"):
        texts = texts.to(device)
        
        with torch.no_grad():
            outputs = model(texts)
            predictions.append(outputs.cpu().numpy())
    
    return np.vstack(predictions)


def main():
    parser = argparse.ArgumentParser(description='Generate predictions on test set')
    parser.add_argument('--model_type', type=str, required=True,
                       choices=['bert', 'roberta', 'word2vec', 'glove', 'ensemble'],
                       help='Type of model to use')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--test_data', type=str, default='../data/Devex_test_questions.csv',
                       help='Path to test data')
    parser.add_argument('--train_data', type=str, default='../data/Devex_train.csv',
                       help='Path to training data (for label columns)')
    parser.add_argument('--output', type=str, default='../results/test_predictions.csv',
                       help='Path to save predictions')
    parser.add_argument('--batch_size', type=int, default=16,
                       help='Batch size for inference')
    parser.add_argument('--max_length', type=int, default=256,
                       help='Maximum sequence length for transformers')
    parser.add_argument('--device', type=str, default='cpu',
                       help='Device to use (cpu or cuda)')
    
    args = parser.parse_args()
    
    print("="*60)
    print("Test Set Prediction Generation")
    print("="*60)
    print(f"Model type: {args.model_type}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Test data: {args.test_data}")
    print(f"Output: {args.output}")
    print(f"Device: {args.device}")
    print("="*60)
    
    # Load data
    print("\nLoading data...")
    loader = SDGDataLoader(
        train_path=args.train_data,
        test_path=args.test_data
    )
    
    train_df, test_df = loader.load_data()
    label_columns = loader.get_label_columns()
    num_labels = len(label_columns)
    
    print(f"Number of labels: {num_labels}")
    print(f"Test samples: {len(test_df)}")
    
    # Get text column
    text_column = 'text'
    if text_column not in test_df.columns:
        text_candidates = [col for col in test_df.columns if 'text' in col.lower() or 'description' in col.lower()]
        if text_candidates:
            text_column = text_candidates[0]
    
    # Preprocess test data
    if args.model_type in ['word2vec', 'glove']:
        train_df, test_df = loader.preprocess_dataset(text_column=text_column)
        test_texts = test_df['processed_text'].fillna('').tolist()
    else:
        # For transformers, use raw text
        test_texts = test_df[text_column].fillna('').tolist()
    
    device = torch.device(args.device)
    
    # Load model and generate predictions
    if args.model_type == 'ensemble':
        print("\nEnsemble prediction not fully implemented in this script.")
        print("Please use the ensemble notebook to generate ensemble predictions.")
        return
    
    model = load_trained_model(args.model_type, args.checkpoint, num_labels, device)
    
    if args.model_type in ['bert', 'roberta']:
        predictions = generate_predictions_bert(
            model, test_texts, 
            batch_size=args.batch_size,
            max_length=args.max_length,
            device=device
        )
    else:
        # For word2vec/glove, need to compute embeddings first
        print(f"\nComputing {args.model_type} embeddings...")
        
        if args.model_type == 'word2vec':
            # Load trained Word2Vec model
            w2v = Word2VecEmbedding(vector_size=300)
            # This would need to load the trained Word2Vec model
            # For now, placeholder
            print("Note: Word2Vec embedding loading needs to be implemented")
            return
        else:
            # Load GloVe embeddings
            glove = GloVeEmbedding(embedding_dim=300)
            # This would need to load GloVe file
            print("Note: GloVe embedding loading needs to be implemented")
            return
        
        test_tokens = [text.split() for text in test_texts]
        test_embeddings = w2v.transform(test_tokens) if args.model_type == 'word2vec' else glove.transform(test_tokens)
        
        predictions = generate_predictions_nn(
            model, test_embeddings,
            batch_size=args.batch_size,
            device=device
        )
    
    # Save predictions
    print(f"\nSaving predictions to {args.output}...")
    
    # Create output DataFrame
    output_df = pd.DataFrame(predictions, columns=label_columns)
    
    # Add any ID columns from test data if present
    id_columns = [col for col in test_df.columns if 'id' in col.lower()]
    if id_columns:
        for col in id_columns:
            output_df.insert(0, col, test_df[col].values)
    
    output_df.to_csv(args.output, index=False)
    print("Predictions saved successfully!")
    
    # Print summary statistics
    print("\nPrediction Statistics:")
    print(f"  Mean probability: {predictions.mean():.4f}")
    print(f"  Std probability: {predictions.std():.4f}")
    print(f"  Min probability: {predictions.min():.4f}")
    print(f"  Max probability: {predictions.max():.4f}")


if __name__ == "__main__":
    main()
