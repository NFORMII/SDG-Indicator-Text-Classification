"""
Transformer-based models (BERT, RoBERTa) for text classification.
"""

import torch
import torch.nn as nn
from transformers import (
    BertModel, BertTokenizer,
    RobertaModel, RobertaTokenizer,
    AutoModel, AutoTokenizer
)
import numpy as np


class BERTClassifier(nn.Module):
    """
    BERT-based classifier for multi-label text classification.
    
    Uses pre-trained BERT model with a classification head.
    """
    
    def __init__(self, model_name='bert-base-uncased', num_labels=10, dropout=0.3, freeze_bert=False):
        """
        Initialize BERT classifier.
        
        Args:
            model_name: Pre-trained BERT model name
            num_labels: Number of output labels
            dropout: Dropout rate
            freeze_bert: Whether to freeze BERT parameters
        """
        super(BERTClassifier, self).__init__()
        
        self.bert = BertModel.from_pretrained(model_name)
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        
        # Freeze BERT parameters if specified
        if freeze_bert:
            for param in self.bert.parameters():
                param.requires_grad = False
        
        # Classification head
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Sequential(
            nn.Linear(self.bert.config.hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_labels)
        )
    
    def forward(self, input_ids, attention_mask=None):
        """
        Forward pass.
        
        Args:
            input_ids: Input token IDs
            attention_mask: Attention mask
            
        Returns:
            Logits for each label
        """
        # Get BERT outputs
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        # Use [CLS] token representation
        pooled_output = outputs.pooler_output
        pooled_output = self.dropout(pooled_output)
        
        # Classification
        logits = self.classifier(pooled_output)
        return torch.sigmoid(logits)
    
    def tokenize(self, texts, max_length=512, padding=True, truncation=True):
        """
        Tokenize texts using BERT tokenizer.
        
        Args:
            texts: List of text strings
            max_length: Maximum sequence length
            padding: Whether to pad sequences
            truncation: Whether to truncate sequences
            
        Returns:
            Dictionary with input_ids and attention_mask
        """
        encoded = self.tokenizer(
            texts,
            max_length=max_length,
            padding=padding,
            truncation=truncation,
            return_tensors='pt'
        )
        return encoded


class RoBERTaClassifier(nn.Module):
    """
    RoBERTa-based classifier for multi-label text classification.
    """
    
    def __init__(self, model_name='roberta-base', num_labels=10, dropout=0.3, freeze_roberta=False):
        """
        Initialize RoBERTa classifier.
        
        Args:
            model_name: Pre-trained RoBERTa model name
            num_labels: Number of output labels
            dropout: Dropout rate
            freeze_roberta: Whether to freeze RoBERTa parameters
        """
        super(RoBERTaClassifier, self).__init__()
        
        self.roberta = RobertaModel.from_pretrained(model_name)
        self.tokenizer = RobertaTokenizer.from_pretrained(model_name)
        
        # Freeze RoBERTa parameters if specified
        if freeze_roberta:
            for param in self.roberta.parameters():
                param.requires_grad = False
        
        # Classification head
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Sequential(
            nn.Linear(self.roberta.config.hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_labels)
        )
    
    def forward(self, input_ids, attention_mask=None):
        """
        Forward pass.
        
        Args:
            input_ids: Input token IDs
            attention_mask: Attention mask
            
        Returns:
            Logits for each label
        """
        # Get RoBERTa outputs
        outputs = self.roberta(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        # Use [CLS] token representation
        pooled_output = outputs.pooler_output
        pooled_output = self.dropout(pooled_output)
        
        # Classification
        logits = self.classifier(pooled_output)
        return torch.sigmoid(logits)
    
    def tokenize(self, texts, max_length=512, padding=True, truncation=True):
        """
        Tokenize texts using RoBERTa tokenizer.
        
        Args:
            texts: List of text strings
            max_length: Maximum sequence length
            padding: Whether to pad sequences
            truncation: Whether to truncate sequences
            
        Returns:
            Dictionary with input_ids and attention_mask
        """
        encoded = self.tokenizer(
            texts,
            max_length=max_length,
            padding=padding,
            truncation=truncation,
            return_tensors='pt'
        )
        return encoded


class DistilBERTClassifier(nn.Module):
    """
    DistilBERT-based classifier for faster training.
    """
    
    def __init__(self, model_name='distilbert-base-uncased', num_labels=10, dropout=0.3):
        """
        Initialize DistilBERT classifier.
        
        Args:
            model_name: Pre-trained DistilBERT model name
            num_labels: Number of output labels
            dropout: Dropout rate
        """
        super(DistilBERTClassifier, self).__init__()
        
        from transformers import DistilBertModel, DistilBertTokenizer
        
        self.distilbert = DistilBertModel.from_pretrained(model_name)
        self.tokenizer = DistilBertTokenizer.from_pretrained(model_name)
        
        # Classification head
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Sequential(
            nn.Linear(self.distilbert.config.hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_labels)
        )
    
    def forward(self, input_ids, attention_mask=None):
        """
        Forward pass.
        """
        outputs = self.distilbert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        # Use [CLS] token representation (first token)
        pooled_output = outputs.last_hidden_state[:, 0, :]
        pooled_output = self.dropout(pooled_output)
        
        logits = self.classifier(pooled_output)
        return torch.sigmoid(logits)
    
    def tokenize(self, texts, max_length=512, padding=True, truncation=True):
        """
        Tokenize texts using DistilBERT tokenizer.
        """
        encoded = self.tokenizer(
            texts,
            max_length=max_length,
            padding=padding,
            truncation=truncation,
            return_tensors='pt'
        )
        return encoded


class BERTEmbedding:
    """
    Extract BERT embeddings for use with other models.
    """
    
    def __init__(self, model_name='bert-base-uncased'):
        """
        Initialize BERT embedding extractor.
        
        Args:
            model_name: Pre-trained BERT model name
        """
        self.model_name = model_name
        self.model = BertModel.from_pretrained(model_name)
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model.eval()
    
    def get_embeddings(self, texts, max_length=512, batch_size=32, device='cpu'):
        """
        Extract BERT embeddings for texts.
        
        Args:
            texts: List of text strings
            max_length: Maximum sequence length
            batch_size: Batch size for processing
            device: Device to run on
            
        Returns:
            Embeddings matrix of shape (len(texts), hidden_size)
        """
        self.model.to(device)
        embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i+batch_size]
            
            encoded = self.tokenizer(
                batch_texts,
                max_length=max_length,
                padding=True,
                truncation=True,
                return_tensors='pt'
            )
            
            with torch.no_grad():
                outputs = self.model(
                    input_ids=encoded['input_ids'].to(device),
                    attention_mask=encoded['attention_mask'].to(device)
                )
            
            # Use [CLS] token representation
            batch_embeddings = outputs.pooler_output.cpu().numpy()
            embeddings.append(batch_embeddings)
        
        return np.vstack(embeddings)


if __name__ == "__main__":
    # Test models
    print("Testing transformer models...")
    
    # Test BERT Classifier
    bert_model = BERTClassifier(num_labels=5)
    texts = ["This is a sample text.", "Another example text."]
    encoded = bert_model.tokenize(texts, max_length=32)
    
    with torch.no_grad():
        outputs = bert_model(encoded['input_ids'], encoded['attention_mask'])
    
    print(f"BERT output shape: {outputs.shape}")
    
    # Test RoBERTa Classifier
    roberta_model = RoBERTaClassifier(num_labels=5)
    encoded = roberta_model.tokenize(texts, max_length=32)
    
    with torch.no_grad():
        outputs = roberta_model(encoded['input_ids'], encoded['attention_mask'])
    
    print(f"RoBERTa output shape: {outputs.shape}")
