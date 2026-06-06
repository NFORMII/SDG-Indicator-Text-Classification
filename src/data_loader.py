"""
Data loading and preprocessing utilities for SDG 3 text classification.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Download required NLTK data
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('punkt_tab', quiet=True)


class SDGDataLoader:
    """Handles data loading and preprocessing for SDG 3 text classification."""
    
    def __init__(self, train_path, test_path=None, use_cleaned=False):
        """
        Initialize the data loader.
        
        Args:
            train_path: Path to training CSV file
            test_path: Path to test CSV file (optional)
            use_cleaned: Whether to use Person 1's cleaned data format
        """
        self.train_path = train_path
        self.test_path = test_path
        self.use_cleaned = use_cleaned
        self.train_df = None
        self.test_df = None
        self.label_columns = None
        self.mlb = MultiLabelBinarizer()
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
        
    def load_data(self):
        """Load training and test datasets."""
        print("Loading training data...")
        self.train_df = pd.read_csv(self.train_path)
        print(f"Training data shape: {self.train_df.shape}")
        
        if self.test_path:
            print("Loading test data...")
            self.test_df = pd.read_csv(self.test_path)
            print(f"Test data shape: {self.test_df.shape}")
        
        # If using cleaned data format, rename columns to standard format
        if self.use_cleaned:
            print("Using Person 1's cleaned data format...")
            # The cleaned data has columns: ID, Type, Text, [12 label columns], cleaned_text
            # Rename to standard format
            if len(self.train_df.columns) >= 16:
                # Rename last column to 'processed_text'
                self.train_df = self.train_df.rename(columns={self.train_df.columns[-1]: 'processed_text'})
                # Rename text column
                if 'Text' in self.train_df.columns:
                    self.train_df = self.train_df.rename(columns={'Text': 'text'})
            
            if self.test_df is not None and len(self.test_df.columns) >= 16:
                self.test_df = self.test_df.rename(columns={self.test_df.columns[-1]: 'processed_text'})
                if 'Text' in self.test_df.columns:
                    self.test_df = self.test_df.rename(columns={'Text': 'text'})
            
        return self.train_df, self.test_df
    
    def explore_data(self):
        """Explore the dataset and print statistics."""
        if self.train_df is None:
            self.load_data()
            
        print("\n" + "="*50)
        print("DATASET EXPLORATION")
        print("="*50)
        print(f"\nTraining set size: {len(self.train_df)}")
        print(f"Columns: {list(self.train_df.columns)}")
        print(f"\nMissing values:\n{self.train_df.isnull().sum()}")
        
        # Identify label columns (assuming they start with specific prefix or are binary)
        # This will need adjustment based on actual dataset structure
        non_text_cols = [col for col in self.train_df.columns 
                        if col.lower() not in ['text', 'title', 'description', 'content']]
        
        print(f"\nPotential label columns: {non_text_cols[:10]}...")
        
        # Check label distribution if available
        if len(non_text_cols) > 0:
            print("\nLabel distribution (first 10 columns):")
            for col in non_text_cols[:10]:
                if col in self.train_df.columns:
                    print(f"{col}: {self.train_df[col].value_counts().to_dict()}")
    
    def preprocess_text(self, text, lowercase=True, remove_stopwords=True, 
                        lemmatize=True, remove_special_chars=True):
        """
        Preprocess text data.
        
        Args:
            text: Input text string
            lowercase: Convert to lowercase
            remove_stopwords: Remove stopwords
            lemmatize: Apply lemmatization
            remove_special_chars: Remove special characters
            
        Returns:
            Preprocessed text string
        """
        if not isinstance(text, str):
            return ""
        
        # Convert to lowercase
        if lowercase:
            text = text.lower()
        
        # Remove special characters and numbers
        if remove_special_chars:
            text = re.sub(r'[^a-zA-Z\s]', '', text)
        
        # Tokenize
        tokens = word_tokenize(text)
        
        # Remove stopwords
        if remove_stopwords:
            tokens = [token for token in tokens if token not in self.stop_words]
        
        # Lemmatize
        if lemmatize:
            tokens = [self.lemmatizer.lemmatize(token) for token in tokens]
        
        # Join back to string
        processed_text = ' '.join(tokens)
        
        return processed_text
    
    def preprocess_dataset(self, text_column='text'):
        """
        Preprocess the entire dataset.
        
        Args:
            text_column: Name of the text column to preprocess
        """
        if self.train_df is None:
            self.load_data()
        
        # If using cleaned data, skip preprocessing
        if self.use_cleaned and 'processed_text' in self.train_df.columns:
            print("\nUsing already cleaned text from Person 1...")
            return self.train_df, self.test_df
        
        print(f"\nPreprocessing text column: {text_column}")
        self.train_df['processed_text'] = self.train_df[text_column].apply(
            lambda x: self.preprocess_text(x)
        )
        
        if self.test_df is not None:
            self.test_df['processed_text'] = self.test_df[text_column].apply(
                lambda x: self.preprocess_text(x)
            )
        
        print("Preprocessing complete!")
        
        return self.train_df, self.test_df
    
    def get_label_columns(self, exclude_cols=['text', 'title', 'description', 
                                              'content', 'processed_text', 'Unique ID', 'Type']):
        """
        Identify label columns in the dataset.
        
        Args:
            exclude_cols: Columns to exclude from labels
            
        Returns:
            List of label column names
        """
        if self.train_df is None:
            self.load_data()
        
        # For cleaned data, labels are columns 4-15 (indices 3-15 in 0-indexed)
        if self.use_cleaned:
            # Get columns starting from index 3 to 15 (12 label columns: Label 1-12)
            if len(self.train_df.columns) >= 16:
                self.label_columns = list(self.train_df.columns[3:15])
            else:
                # Fallback to original logic
                self.label_columns = [col for col in self.train_df.columns 
                                     if col not in exclude_cols]
        else:
            self.label_columns = [col for col in self.train_df.columns 
                                 if col not in exclude_cols]
        return self.label_columns
    
    def prepare_labels(self):
        """
        Prepare labels for multi-label classification.
        
        Returns:
            y_train: Label matrix for training data
            y_val: Label matrix for validation data (if split)
        """
        if self.label_columns is None:
            self.get_label_columns()
        
        y = self.train_df[self.label_columns].values
        
        # For cleaned data, convert text labels to binary
        if self.use_cleaned:
            # Convert to binary: 1 if label present (not NaN/not empty), 0 otherwise
            y = np.where(pd.isna(y), 0, 1).astype(float)
        
        return y
    
    def train_val_split(self, test_size=0.2, random_state=42):
        """
        Split training data into train and validation sets.
        
        Args:
            test_size: Proportion of data for validation
            random_state: Random seed
            
        Returns:
            X_train, X_val, y_train, y_val
        """
        if self.train_df is None:
            self.load_data()
        
        if 'processed_text' not in self.train_df.columns:
            self.preprocess_dataset()
        
        y = self.prepare_labels()
        X = self.train_df['processed_text'].values
        
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        
        print(f"\nTrain size: {len(X_train)}, Validation size: {len(X_val)}")
        print(f"Number of labels: {y_train.shape[1]}")
        
        return X_train, X_val, y_train, y_val
    
    def get_text_statistics(self):
        """Get statistics about text lengths in the dataset."""
        if self.train_df is None:
            self.load_data()
        
        if 'processed_text' not in self.train_df.columns:
            self.preprocess_dataset()
        
        text_lengths = self.train_df['processed_text'].apply(lambda x: len(x.split()))
        
        print("\n" + "="*50)
        print("TEXT STATISTICS")
        print("="*50)
        print(f"Mean text length: {text_lengths.mean():.2f} words")
        print(f"Median text length: {text_lengths.median():.2f} words")
        print(f"Max text length: {text_lengths.max()} words")
        print(f"Min text length: {text_lengths.min()} words")
        print(f"Std text length: {text_lengths.std():.2f} words")
        
        return text_lengths


if __name__ == "__main__":
    # Example usage
    loader = SDGDataLoader(
        train_path="data/Devex_train.csv",
        test_path="data/Devex_test_questions.csv"
    )
    
    # Load and explore data
    loader.load_data()
    loader.explore_data()
    
    # Preprocess
    loader.preprocess_dataset()
    loader.get_text_statistics()
    
    # Split data
    X_train, X_val, y_train, y_val = loader.train_val_split()
