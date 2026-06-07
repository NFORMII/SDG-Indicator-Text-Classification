"""
Embedding models for text representation.
"""

import numpy as np
import torch
from gensim.models import Word2Vec, KeyedVectors
from sklearn.feature_extraction.text import TfidfVectorizer
from tqdm import tqdm
import nltk
from nltk.tokenize import word_tokenize


class Word2VecEmbedding:
    """
    Word2Vec embedding model for text representation.
    """
    
    def __init__(self, vector_size=300, window=5, min_count=1, workers=4, sg=0):
        """
        Initialize Word2Vec embedding.
        
        Args:
            vector_size: Dimensionality of word vectors
            window: Maximum distance between current and predicted word
            min_count: Ignores words with total frequency lower than this
            workers: Number of worker threads
            sg: Training algorithm (0=CBOW, 1=Skip-gram)
        """
        self.vector_size = vector_size
        self.window = window
        self.min_count = min_count
        self.workers = workers
        self.sg = sg
        self.model = None
        self.embedding_matrix = None
        self.word_to_idx = None
        self.idx_to_word = None
    
    def train(self, texts):
        """
        Train Word2Vec model on corpus.
        
        Args:
            texts: List of tokenized texts
        """
        print("Training Word2Vec model...")
        self.model = Word2Vec(
            sentences=texts,
            vector_size=self.vector_size,
            window=self.window,
            min_count=self.min_count,
            workers=self.workers,
            sg=self.sg
        )
        print(f"Vocabulary size: {len(self.model.wv)}")
    
    def load_pretrained(self, path):
        """
        Load pretrained Word2Vec model.
        
        Args:
            path: Path to pretrained model file
        """
        print(f"Loading pretrained Word2Vec from {path}...")
        self.model = KeyedVectors.load_word2vec_format(path, binary=True)
        print(f"Vocabulary size: {len(self.model)}")
    
    def get_document_vector(self, text, method='mean'):
        """
        Get document vector by aggregating word vectors.
        
        Args:
            text: Tokenized text
            method: Aggregation method ('mean', 'sum', 'tfidf')
            
        Returns:
            Document vector
        """
        vectors = []
        for word in text:
            if word in self.model.wv:
                vectors.append(self.model.wv[word])
        
        if len(vectors) == 0:
            return np.zeros(self.vector_size)
        
        vectors = np.array(vectors)
        
        if method == 'mean':
            return np.mean(vectors, axis=0)
        elif method == 'sum':
            return np.sum(vectors, axis=0)
        else:
            return np.mean(vectors, axis=0)
    
    def transform(self, texts, method='mean'):
        """
        Transform list of texts to document vectors.
        
        Args:
            texts: List of tokenized texts
            method: Aggregation method
            
        Returns:
            Document vectors matrix
        """
        print(f"Transforming texts using Word2Vec ({method} method)...")
        vectors = []
        
        for text in tqdm(texts):
            doc_vector = self.get_document_vector(text, method)
            vectors.append(doc_vector)
        
        return np.array(vectors)
    
    def create_embedding_matrix(self, vocab, max_vocab_size=50000):
        """
        Create embedding matrix for neural network.
        
        Args:
            vocab: Dictionary mapping words to indices
            max_vocab_size: Maximum vocabulary size
            
        Returns:
            Embedding matrix of shape (vocab_size, vector_size)
        """
        vocab_size = min(len(vocab), max_vocab_size)
        self.embedding_matrix = np.zeros((vocab_size, self.vector_size))
        
        found = 0
        for word, idx in vocab.items():
            if idx < vocab_size and word in self.model.wv:
                self.embedding_matrix[idx] = self.model.wv[word]
                found += 1
        
        print(f"Found embeddings for {found}/{vocab_size} words ({found/vocab_size*100:.1f}%)")
        return self.embedding_matrix


class GloVeEmbedding:
    """
    GloVe embedding model for text representation.
    """
    
    def __init__(self, embedding_dim=300):
        """
        Initialize GloVe embedding.
        
        Args:
            embedding_dim: Dimensionality of word vectors
        """
        self.embedding_dim = embedding_dim
        self.embeddings = {}
        self.embedding_matrix = None
    
    def load(self, path):
        """
        Load GloVe embeddings from file.
        
        Args:
            path: Path to GloVe file (e.g., 'glove.6B.300d.txt')
        """
        print(f"Loading GloVe embeddings from {path}...")
        
        with open(path, 'r', encoding='utf-8') as f:
            for line in tqdm(f):
                values = line.split()
                word = values[0]
                vector = np.asarray(values[1:], dtype='float32')
                self.embeddings[word] = vector
        
        print(f"Loaded {len(self.embeddings)} word vectors")
        print(f"Embedding dimension: {self.embedding_dim}")
    
    def get_document_vector(self, text, method='mean'):
        """
        Get document vector by aggregating word vectors.
        
        Args:
            text: Tokenized text
            method: Aggregation method ('mean', 'sum')
            
        Returns:
            Document vector
        """
        vectors = []
        for word in text:
            if word in self.embeddings:
                vectors.append(self.embeddings[word])
        
        if len(vectors) == 0:
            return np.zeros(self.embedding_dim)
        
        vectors = np.array(vectors)
        
        if method == 'mean':
            return np.mean(vectors, axis=0)
        elif method == 'sum':
            return np.sum(vectors, axis=0)
        else:
            return np.mean(vectors, axis=0)
    
    def transform(self, texts, method='mean'):
        """
        Transform list of texts to document vectors.
        
        Args:
            texts: List of tokenized texts
            method: Aggregation method
            
        Returns:
            Document vectors matrix
        """
        print(f"Transforming texts using GloVe ({method} method)...")
        vectors = []
        
        for text in tqdm(texts):
            doc_vector = self.get_document_vector(text, method)
            vectors.append(doc_vector)
        
        return np.array(vectors)
    
    def create_embedding_matrix(self, vocab, max_vocab_size=50000):
        """
        Create embedding matrix for neural network.
        
        Args:
            vocab: Dictionary mapping words to indices
            max_vocab_size: Maximum vocabulary size
            
        Returns:
            Embedding matrix of shape (vocab_size, embedding_dim)
        """
        vocab_size = min(len(vocab), max_vocab_size)
        self.embedding_matrix = np.zeros((vocab_size, self.embedding_dim))
        
        found = 0
        for word, idx in vocab.items():
            if idx < vocab_size and word in self.embeddings:
                self.embedding_matrix[idx] = self.embeddings[word]
                found += 1
        
        print(f"Found embeddings for {found}/{vocab_size} words ({found/vocab_size*100:.1f}%)")
        return self.embedding_matrix


class TFIDFEmbedding:
    """
    TF-IDF weighted word embeddings.
    """
    
    def __init__(self, max_features=10000):
        """
        Initialize TF-IDF embedding.
        
        Args:
            max_features: Maximum number of features
        """
        self.max_features = max_features
        self.vectorizer = None
        self.feature_names = None
    
    def fit(self, texts):
        """
        Fit TF-IDF vectorizer on texts.
        
        Args:
            texts: List of strings (not tokenized)
        """
        print("Fitting TF-IDF vectorizer...")
        self.vectorizer = TfidfVectorizer(max_features=self.max_features)
        self.vectorizer.fit(texts)
        self.feature_names = self.vectorizer.get_feature_names_out()
        print(f"Vocabulary size: {len(self.feature_names)}")
    
    def transform(self, texts):
        """
        Transform texts to TF-IDF features.
        
        Args:
            texts: List of strings
            
        Returns:
            TF-IDF feature matrix
        """
        print("Transforming texts to TF-IDF features...")
        return self.vectorizer.transform(texts).toarray()


class FastTextEmbedding:
    """
    FastText embedding model for text representation.
    Handles out-of-vocabulary words using subword information.
    """
    
    def __init__(self, vector_size=300, window=5, min_count=1, workers=4):
        """
        Initialize FastText embedding.
        
        Args:
            vector_size: Dimensionality of word vectors
            window: Maximum distance between current and predicted word
            min_count: Ignores words with total frequency lower than this
            workers: Number of worker threads
        """
        self.vector_size = vector_size
        self.window = window
        self.min_count = min_count
        self.workers = workers
        self.model = None
    
    def train(self, texts):
        """
        Train FastText model on corpus.
        
        Args:
            texts: List of tokenized texts
        """
        print("Training FastText model...")
        from gensim.models import FastText
        self.model = FastText(
            sentences=texts,
            vector_size=self.vector_size,
            window=self.window,
            min_count=self.min_count,
            workers=self.workers
        )
        print(f"Vocabulary size: {len(self.model.wv)}")
    
    def get_document_vector(self, text, method='mean'):
        """
        Get document vector by aggregating word vectors.
        
        Args:
            text: Tokenized text
            method: Aggregation method
            
        Returns:
            Document vector
        """
        vectors = []
        for word in text:
            if word in self.model.wv:
                vectors.append(self.model.wv[word])
        
        if len(vectors) == 0:
            return np.zeros(self.vector_size)
        
        vectors = np.array(vectors)
        
        if method == 'mean':
            return np.mean(vectors, axis=0)
        elif method == 'sum':
            return np.sum(vectors, axis=0)
        else:
            return np.mean(vectors, axis=0)
    
    def transform(self, texts, method='mean'):
        """
        Transform list of texts to document vectors.
        
        Args:
            texts: List of tokenized texts
            method: Aggregation method
            
        Returns:
            Document vectors matrix
        """
        print(f"Transforming texts using FastText ({method} method)...")
        vectors = []
        
        for text in tqdm(texts):
            doc_vector = self.get_document_vector(text, method)
            vectors.append(doc_vector)
        
        return np.array(vectors)


if __name__ == "__main__":
    # Example usage
    texts = [
        ["this", "is", "a", "sample", "text"],
        ["another", "example", "text"],
        ["machine", "learning", "is", "great"]
    ]
    
    # Test Word2Vec
    w2v = Word2VecEmbedding(vector_size=100, window=3)
    w2v.train(texts)
    vectors = w2v.transform(texts)
    print(f"Word2Vec vectors shape: {vectors.shape}")
