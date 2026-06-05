"""
Evaluation metrics for multi-label text classification.
"""

import numpy as np
from sklearn.metrics import hamming_loss, accuracy_score, f1_score, precision_score, recall_score
from sklearn.metrics import multilabel_confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns


class MultiLabelEvaluator:
    """Evaluator for multi-label classification tasks."""
    
    def __init__(self, label_names=None):
        """
        Initialize evaluator.
        
        Args:
            label_names: List of label names for better reporting
        """
        self.label_names = label_names
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_hamming': [],
            'val_hamming': []
        }
    
    def compute_metrics(self, y_true, y_pred, threshold=0.5):
        """
        Compute various evaluation metrics.
        
        Args:
            y_true: True labels (binary matrix)
            y_pred: Predicted probabilities
            threshold: Threshold for binary predictions
            
        Returns:
            Dictionary of metrics
        """
        # Convert probabilities to binary predictions
        y_pred_binary = (y_pred >= threshold).astype(int)
        
        metrics = {
            'hamming_loss': hamming_loss(y_true, y_pred_binary),
            'accuracy': accuracy_score(y_true, y_pred_binary),
            'f1_macro': f1_score(y_true, y_pred_binary, average='macro'),
            'f1_micro': f1_score(y_true, y_pred_binary, average='micro'),
            'precision_macro': precision_score(y_true, y_pred_binary, average='macro'),
            'precision_micro': precision_score(y_true, y_pred_binary, average='micro'),
            'recall_macro': recall_score(y_true, y_pred_binary, average='macro'),
            'recall_micro': recall_score(y_true, y_pred_binary, average='micro'),
            'f1_samples': f1_score(y_true, y_pred_binary, average='samples'),
        }
        
        return metrics
    
    def print_metrics(self, metrics, prefix=""):
        """Print metrics in a formatted way."""
        print(f"\n{prefix}Metrics:")
        print("-" * 40)
        for metric, value in metrics.items():
            print(f"{metric:20s}: {value:.4f}")
    
    def print_classification_report(self, y_true, y_pred, threshold=0.5):
        """
        Print detailed classification report for each label.
        
        Args:
            y_true: True labels
            y_pred: Predicted probabilities
            threshold: Threshold for binary predictions
        """
        y_pred_binary = (y_pred >= threshold).astype(int)
        
        if self.label_names:
            target_names = self.label_names
        else:
            target_names = [f"Label_{i}" for i in range(y_true.shape[1])]
        
        print("\nClassification Report:")
        print(classification_report(y_true, y_pred_binary, target_names=target_names, zero_division=0))
    
    def plot_confusion_matrices(self, y_true, y_pred, threshold=0.5, max_labels=9):
        """
        Plot confusion matrices for each label.
        
        Args:
            y_true: True labels
            y_pred: Predicted probabilities
            threshold: Threshold for binary predictions
            max_labels: Maximum number of labels to plot (to avoid overcrowding)
        """
        y_pred_binary = (y_pred >= threshold).astype(int)
        
        # Compute confusion matrices
        conf_matrices = multilabel_confusion_matrix(y_true, y_pred_binary)
        
        # Limit number of plots
        n_labels = min(len(conf_matrices), max_labels)
        
        fig, axes = plt.subplots(3, 3, figsize=(15, 15))
        axes = axes.flatten()
        
        for i in range(n_labels):
            cm = conf_matrices[i]
            
            # Normalize
            cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            
            sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues', 
                       ax=axes[i], cbar=False)
            
            if self.label_names:
                axes[i].set_title(f'{self.label_names[i]}')
            else:
                axes[i].set_title(f'Label {i}')
            
            axes[i].set_xlabel('Predicted')
            axes[i].set_ylabel('True')
        
        # Hide unused subplots
        for i in range(n_labels, len(axes)):
            axes[i].axis('off')
        
        plt.tight_layout()
        plt.savefig('results/confusion_matrices.png', dpi=150, bbox_inches='tight')
        plt.show()
    
    def plot_learning_curves(self):
        """Plot training and validation learning curves."""
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        
        # Loss curves
        axes[0].plot(self.history['train_loss'], label='Train Loss')
        axes[0].plot(self.history['val_loss'], label='Validation Loss')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].set_title('Training and Validation Loss')
        axes[0].legend()
        axes[0].grid(True)
        
        # Hamming loss curves
        axes[1].plot(self.history['train_hamming'], label='Train Hamming Loss')
        axes[1].plot(self.history['val_hamming'], label='Validation Hamming Loss')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Hamming Loss')
        axes[1].set_title('Training and Validation Hamming Loss')
        axes[1].legend()
        axes[1].grid(True)
        
        plt.tight_layout()
        plt.savefig('results/learning_curves.png', dpi=150, bbox_inches='tight')
        plt.show()
    
    def update_history(self, train_loss, val_loss, train_hamming, val_hamming):
        """
        Update training history.
        
        Args:
            train_loss: Training loss
            val_loss: Validation loss
            train_hamming: Training hamming loss
            val_hamming: Validation hamming loss
        """
        self.history['train_loss'].append(train_loss)
        self.history['val_loss'].append(val_loss)
        self.history['train_hamming'].append(train_hamming)
        self.history['val_hamming'].append(val_hamming)
    
    def find_optimal_threshold(self, y_true, y_pred, metric='f1'):
        """
        Find optimal threshold for binary predictions.
        
        Args:
            y_true: True labels
            y_pred: Predicted probabilities
            metric: Metric to optimize ('f1', 'precision', 'recall')
            
        Returns:
            Optimal threshold value
        """
        thresholds = np.arange(0.1, 0.9, 0.05)
        best_threshold = 0.5
        best_score = 0
        
        for threshold in thresholds:
            y_pred_binary = (y_pred >= threshold).astype(int)
            
            if metric == 'f1':
                score = f1_score(y_true, y_pred_binary, average='micro')
            elif metric == 'precision':
                score = precision_score(y_true, y_pred_binary, average='micro')
            elif metric == 'recall':
                score = recall_score(y_true, y_pred_binary, average='micro')
            else:
                score = f1_score(y_true, y_pred_binary, average='micro')
            
            if score > best_score:
                best_score = score
                best_threshold = threshold
        
        print(f"\nOptimal threshold: {best_threshold:.2f} (Best {metric}: {best_score:.4f})")
        return best_threshold
    
    def save_predictions(self, y_pred, output_path='results/predictions.csv'):
        """
        Save predictions to CSV file.
        
        Args:
            y_pred: Predicted probabilities
            output_path: Path to save predictions
        """
        np.savetxt(output_path, y_pred, delimiter=',')
        print(f"Predictions saved to {output_path}")


if __name__ == "__main__":
    # Example usage
    evaluator = MultiLabelEvaluator()
    
    # Generate dummy data for testing
    y_true = np.random.randint(0, 2, (100, 5))
    y_pred = np.random.random((100, 5))
    
    # Compute metrics
    metrics = evaluator.compute_metrics(y_true, y_pred)
    evaluator.print_metrics(metrics, prefix="Test ")
    
    # Find optimal threshold
    evaluator.find_optimal_threshold(y_true, y_pred)
