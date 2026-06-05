"""
Neural network architectures for multi-label text classification.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FeedForwardNN(nn.Module):
    """
    Simple feedforward neural network for multi-label classification.
    
    Architecture:
    - Input layer (embedding dimension)
    - Hidden layers with ReLU activation
    - Dropout for regularization
    - Output layer with sigmoid activation (multi-label)
    """
    
    def __init__(self, input_dim, hidden_dims, output_dim, dropout=0.3):
        """
        Initialize the feedforward neural network.
        
        Args:
            input_dim: Dimension of input features (embedding dimension)
            hidden_dims: List of hidden layer dimensions
            output_dim: Number of output labels
            dropout: Dropout rate
        """
        super(FeedForwardNN, self).__init__()
        
        layers = []
        prev_dim = input_dim
        
        # Build hidden layers
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim
        
        # Output layer
        layers.append(nn.Linear(prev_dim, output_dim))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (batch_size, input_dim)
            
        Returns:
            Output tensor of shape (batch_size, output_dim)
        """
        return torch.sigmoid(self.network(x))


class DeepNN(nn.Module):
    """
    Deep neural network with batch normalization and residual connections.
    
    Architecture:
    - Input layer
    - Multiple deep blocks with batch normalization and residual connections
    - Dropout for regularization
    - Output layer with sigmoid activation
    """
    
    def __init__(self, input_dim, hidden_dims, output_dim, dropout=0.3, use_batch_norm=True):
        """
        Initialize the deep neural network.
        
        Args:
            input_dim: Dimension of input features
            hidden_dims: List of hidden layer dimensions
            output_dim: Number of output labels
            dropout: Dropout rate
            use_batch_norm: Whether to use batch normalization
        """
        super(DeepNN, self).__init__()
        
        self.layers = nn.ModuleList()
        self.batch_norms = nn.ModuleList() if use_batch_norm else None
        self.use_batch_norm = use_batch_norm
        
        prev_dim = input_dim
        
        # Build deep layers
        for i, hidden_dim in enumerate(hidden_dims):
            self.layers.append(nn.Linear(prev_dim, hidden_dim))
            
            if use_batch_norm:
                self.batch_norms.append(nn.BatchNorm1d(hidden_dim))
            
            prev_dim = hidden_dim
        
        # Output layer
        self.output_layer = nn.Linear(prev_dim, output_dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        """
        Forward pass with residual connections.
        
        Args:
            x: Input tensor
            
        Returns:
            Output tensor
        """
        residual = x
        
        for i, layer in enumerate(self.layers):
            x = layer(x)
            
            if self.use_batch_norm:
                x = self.batch_norms[i](x)
            
            x = F.relu(x)
            x = self.dropout(x)
            
            # Residual connection if dimensions match
            if x.shape == residual.shape:
                x = x + residual
                residual = x
        
        x = self.output_layer(x)
        return torch.sigmoid(x)


class CNNTextClassifier(nn.Module):
    """
    1D CNN for text classification.
    
    Architecture:
    - Embedding layer
    - Multiple convolutional layers with different kernel sizes
    - Max pooling
    - Fully connected layers
    - Output layer with sigmoid activation
    """
    
    def __init__(self, vocab_size, embedding_dim, num_filters, filter_sizes, 
                 output_dim, dropout=0.3, pretrained_embeddings=None):
        """
        Initialize CNN classifier.
        
        Args:
            vocab_size: Size of vocabulary
            embedding_dim: Dimension of word embeddings
            num_filters: Number of filters per kernel size
            filter_sizes: List of kernel sizes (e.g., [3, 4, 5])
            output_dim: Number of output labels
            dropout: Dropout rate
            pretrained_embeddings: Optional pretrained embedding matrix
        """
        super(CNNTextClassifier, self).__init__()
        
        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        
        if pretrained_embeddings is not None:
            self.embedding.weight.data.copy_(pretrained_embeddings)
        
        # Convolutional layers with different kernel sizes
        self.convs = nn.ModuleList([
            nn.Conv1d(embedding_dim, num_filters, kernel_size=fs)
            for fs in filter_sizes
        ])
        
        # Fully connected layers
        self.fc = nn.Linear(len(filter_sizes) * num_filters, output_dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len)
            
        Returns:
            Output tensor of shape (batch_size, output_dim)
        """
        # Embedding
        embedded = self.embedding(x)  # (batch_size, seq_len, embedding_dim)
        embedded = embedded.permute(0, 2, 1)  # (batch_size, embedding_dim, seq_len)
        
        # Convolution and pooling
        conved = [F.relu(conv(embedded)) for conv in self.convs]
        pooled = [F.max_pool1d(conv, conv.shape[2]).squeeze(2) for conv in conved]
        
        # Concatenate
        cat = self.dropout(torch.cat(pooled, dim=1))
        
        # Output
        output = self.fc(cat)
        return torch.sigmoid(output)


class LSTMClassifier(nn.Module):
    """
    LSTM-based classifier for text classification.
    
    Architecture:
    - Embedding layer
    - LSTM layer(s)
    - Fully connected layers
    - Output layer with sigmoid activation
    """
    
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, 
                 num_layers=2, dropout=0.3, bidirectional=True, pretrained_embeddings=None):
        """
        Initialize LSTM classifier.
        
        Args:
            vocab_size: Size of vocabulary
            embedding_dim: Dimension of word embeddings
            hidden_dim: Hidden dimension of LSTM
            output_dim: Number of output labels
            num_layers: Number of LSTM layers
            dropout: Dropout rate
            bidirectional: Whether to use bidirectional LSTM
            pretrained_embeddings: Optional pretrained embedding matrix
        """
        super(LSTMClassifier, self).__init__()
        
        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        
        if pretrained_embeddings is not None:
            self.embedding.weight.data.copy_(pretrained_embeddings)
        
        # LSTM layer
        self.lstm = nn.LSTM(
            embedding_dim, 
            hidden_dim, 
            num_layers=num_layers, 
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional,
            batch_first=True
        )
        
        # Output dimension based on bidirectional
        lstm_output_dim = hidden_dim * 2 if bidirectional else hidden_dim
        
        # Fully connected layers
        self.fc1 = nn.Linear(lstm_output_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len)
            
        Returns:
            Output tensor of shape (batch_size, output_dim)
        """
        # Embedding
        embedded = self.embedding(x)
        
        # LSTM
        lstm_out, (hidden, cell) = self.lstm(embedded)
        
        # Use the final hidden state
        if self.lstm.bidirectional:
            hidden = torch.cat((hidden[-2], hidden[-1]), dim=1)
        else:
            hidden = hidden[-1]
        
        # Fully connected layers
        hidden = self.dropout(F.relu(self.fc1(hidden)))
        output = self.fc2(hidden)
        
        return torch.sigmoid(output)


class FocalLoss(nn.Module):
    """
    Focal Loss for addressing class imbalance.
    
    Focal Loss = -alpha * (1 - pt)^gamma * log(pt)
    where pt is the predicted probability for the true class.
    """
    
    def __init__(self, alpha=1.0, gamma=2.0, reduction='mean'):
        """
        Initialize focal loss.
        
        Args:
            alpha: Weighting factor for rare classes
            gamma: Focusing parameter (higher = more focus on hard examples)
            reduction: Reduction method ('mean', 'sum', or 'none')
        """
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, inputs, targets):
        """
        Compute focal loss.
        
        Args:
            inputs: Predicted probabilities (after sigmoid)
            targets: True labels (0 or 1)
            
        Returns:
            Focal loss value
        """
        bce_loss = F.binary_cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-bce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


if __name__ == "__main__":
    # Test models
    batch_size = 32
    input_dim = 300
    hidden_dims = [256, 128]
    output_dim = 10
    
    # Test FeedForwardNN
    model_ff = FeedForwardNN(input_dim, hidden_dims, output_dim)
    x = torch.randn(batch_size, input_dim)
    output = model_ff(x)
    print(f"FeedForwardNN output shape: {output.shape}")
    
    # Test DeepNN
    model_deep = DeepNN(input_dim, hidden_dims, output_dim)
    output = model_deep(x)
    print(f"DeepNN output shape: {output.shape}")
