"""
Training utilities for neural network models.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from tqdm import tqdm
import os


class TextDataset(Dataset):
    """Dataset for text classification."""
    
    def __init__(self, texts, labels):
        """
        Initialize dataset.
        
        Args:
            texts: Text features (can be numpy array or list)
            labels: Labels (numpy array)
        """
        self.texts = torch.FloatTensor(texts)
        self.labels = torch.FloatTensor(labels)
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        return self.texts[idx], self.labels[idx]


class Trainer:
    """
    Trainer class for neural network models.
    """
    
    def __init__(self, model, train_loader, val_loader, device='cpu', 
                 learning_rate=0.001, class_weights=None):
        """
        Initialize trainer.
        
        Args:
            model: PyTorch model
            train_loader: Training data loader
            val_loader: Validation data loader
            device: Device to train on
            learning_rate: Learning rate
            class_weights: Optional class weights for loss function
        """
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        
        # Loss function
        if class_weights is not None:
            class_weights = torch.FloatTensor(class_weights).to(device)
            self.criterion = nn.BCELoss(weight=class_weights)
        else:
            self.criterion = nn.BCELoss()
        
        # Optimizer
        self.optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=3
        )
    
    def train_epoch(self):
        """
        Train for one epoch.
        
        Returns:
            Average training loss
        """
        self.model.train()
        total_loss = 0
        
        for texts, labels in tqdm(self.train_loader, desc="Training"):
            texts, labels = texts.to(self.device), labels.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(texts)
            loss = self.criterion(outputs, labels)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
        
        return total_loss / len(self.train_loader)
    
    def validate(self):
        """
        Validate the model.
        
        Returns:
            Average validation loss, predictions, true labels
        """
        self.model.eval()
        total_loss = 0
        all_predictions = []
        all_labels = []
        
        with torch.no_grad():
            for texts, labels in tqdm(self.val_loader, desc="Validation"):
                texts, labels = texts.to(self.device), labels.to(self.device)
                
                outputs = self.model(texts)
                loss = self.criterion(outputs, labels)
                
                total_loss += loss.item()
                all_predictions.append(outputs.cpu().numpy())
                all_labels.append(labels.cpu().numpy())
        
        avg_loss = total_loss / len(self.val_loader)
        predictions = np.vstack(all_predictions)
        true_labels = np.vstack(all_labels)
        
        return avg_loss, predictions, true_labels
    
    def train(self, num_epochs, evaluator, save_path='results'):
        """
        Train the model for multiple epochs.
        
        Args:
            num_epochs: Number of epochs to train
            evaluator: MultiLabelEvaluator instance
            save_path: Path to save model checkpoints
        """
        os.makedirs(save_path, exist_ok=True)
        
        best_val_loss = float('inf')
        best_model_state = None
        
        print(f"\nTraining for {num_epochs} epochs...")
        print("=" * 50)
        
        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch+1}/{num_epochs}")
            
            # Train
            train_loss = self.train_epoch()
            
            # Validate
            val_loss, val_pred, val_true = self.validate()
            
            # Compute metrics
            val_metrics = evaluator.compute_metrics(val_true, val_pred)
            
            # Update history
            train_hamming = evaluator.compute_metrics(
                np.vstack([labels.cpu().numpy() for _, labels in self.train_loader]),
                np.vstack([self.model(texts.to(self.device)).cpu().detach().numpy() 
                          for texts, _ in self.train_loader])
            )['hamming_loss']
            
            evaluator.update_history(train_loss, val_loss, train_hamming, val_metrics['hamming_loss'])
            
            # Print progress
            print(f"Train Loss: {train_loss:.4f}")
            print(f"Val Loss: {val_loss:.4f}")
            print(f"Val Hamming Loss: {val_metrics['hamming_loss']:.4f}")
            print(f"Val F1 (micro): {val_metrics['f1_micro']:.4f}")
            
            # Learning rate scheduling
            self.scheduler.step(val_loss)
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_state = self.model.state_dict()
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'val_loss': val_loss,
                    'val_metrics': val_metrics
                }, os.path.join(save_path, 'best_model.pt'))
                print("Saved best model!")
        
        # Load best model
        if best_model_state is not None:
            self.model.load_state_dict(best_model_state)
            print(f"\nBest validation loss: {best_val_loss:.4f}")
        
        return self.model


class TransformerTrainer:
    """
    Trainer for transformer-based models (BERT, RoBERTa, etc.).
    """
    
    def __init__(self, model, train_texts, val_texts, train_labels, val_labels,
                 batch_size=16, max_length=512, device='cpu', learning_rate=2e-5):
        """
        Initialize transformer trainer.
        
        Args:
            model: Transformer model
            train_texts: Training texts
            val_texts: Validation texts
            train_labels: Training labels
            val_labels: Validation labels
            batch_size: Batch size
            max_length: Maximum sequence length
            device: Device to train on
            learning_rate: Learning rate
        """
        self.model = model.to(device)
        self.train_texts = train_texts
        self.val_texts = val_texts
        self.train_labels = torch.FloatTensor(train_labels)
        self.val_labels = torch.FloatTensor(val_labels)
        self.batch_size = batch_size
        self.max_length = max_length
        self.device = device
        
        self.criterion = nn.BCELoss()
        self.optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=2
        )
    
    def train_epoch(self):
        """Train for one epoch."""
        self.model.train()
        total_loss = 0
        
        num_batches = len(self.train_texts) // self.batch_size + 1
        
        for i in tqdm(range(0, len(self.train_texts), self.batch_size), 
                     desc="Training", total=num_batches):
            batch_texts = self.train_texts[i:i+self.batch_size]
            batch_labels = self.train_labels[i:i+self.batch_size].to(self.device)
            
            # Tokenize
            encoded = self.model.tokenize(batch_texts, max_length=self.max_length)
            input_ids = encoded['input_ids'].to(self.device)
            attention_mask = encoded['attention_mask'].to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(input_ids, attention_mask)
            loss = self.criterion(outputs, batch_labels)
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            total_loss += loss.item()
        
        return total_loss / num_batches
    
    def validate(self):
        """Validate the model."""
        self.model.eval()
        total_loss = 0
        all_predictions = []
        all_labels = []
        
        num_batches = len(self.val_texts) // self.batch_size + 1
        
        with torch.no_grad():
            for i in tqdm(range(0, len(self.val_texts), self.batch_size),
                         desc="Validation", total=num_batches):
                batch_texts = self.val_texts[i:i+self.batch_size]
                batch_labels = self.val_labels[i:i+self.batch_size]
                
                encoded = self.model.tokenize(batch_texts, max_length=self.max_length)
                input_ids = encoded['input_ids'].to(self.device)
                attention_mask = encoded['attention_mask'].to(self.device)
                
                outputs = self.model(input_ids, attention_mask)
                loss = self.criterion(outputs, batch_labels.to(self.device))
                
                total_loss += loss.item()
                all_predictions.append(outputs.cpu().numpy())
                all_labels.append(batch_labels.numpy())
        
        avg_loss = total_loss / num_batches
        predictions = np.vstack(all_predictions)
        true_labels = np.vstack(all_labels)
        
        return avg_loss, predictions, true_labels
    
    def predict(self, texts):
        """
        Make predictions on texts.
        
        Args:
            texts: List of texts to predict
            
        Returns:
            Predictions array
        """
        self.model.eval()
        all_predictions = []
        
        num_batches = len(texts) // self.batch_size + 1
        
        for i in tqdm(range(0, len(texts), self.batch_size), 
                     desc="Predicting"):
            batch_texts = texts[i:i+self.batch_size]
            batch_texts = [str(text) if pd.notna(text) else "" for text in batch_texts]
            
            encoded = self.model.tokenize(batch_texts, max_length=self.max_length)
            input_ids = encoded['input_ids'].to(self.device)
            attention_mask = encoded['attention_mask'].to(self.device)
            
            with torch.no_grad():
                outputs = self.model(input_ids, attention_mask)
                all_predictions.append(outputs.cpu().numpy())
        
        return np.vstack(all_predictions)
    
    def train(self, num_epochs, evaluator, save_path='results'):
        """
        Train the transformer model.
        
        Args:
            num_epochs: Number of epochs
            evaluator: MultiLabelEvaluator instance
            save_path: Path to save model
        """
        os.makedirs(save_path, exist_ok=True)
        
        best_val_loss = float('inf')
        
        print(f"\nTraining for {num_epochs} epochs...")
        print("=" * 50)
        
        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch+1}/{num_epochs}")
            
            train_loss = self.train_epoch()
            val_loss, val_pred, val_true = self.validate()
            
            val_metrics = evaluator.compute_metrics(val_true, val_pred)
            
            print(f"Train Loss: {train_loss:.4f}")
            print(f"Val Loss: {val_loss:.4f}")
            print(f"Val Hamming Loss: {val_metrics['hamming_loss']:.4f}")
            print(f"Val F1 (micro): {val_metrics['f1_micro']:.4f}")
            
            self.scheduler.step(val_loss)
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'val_loss': val_loss,
                    'val_metrics': val_metrics
                }, os.path.join(save_path, 'best_transformer_model.pt'))
                print("Saved best model!")
        
        print(f"\nBest validation loss: {best_val_loss:.4f}")
        return self.model


if __name__ == "__main__":
    # Example usage
    print("Training utilities loaded successfully!")
