import torch
import pandas as pd
import re
from tqdm import tqdm
import random
import numpy as np
from collections import Counter
import nltk
from nltk.tokenize import word_tokenize

# Download NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

class TextPreprocessor:
    def __init__(self, max_vocab_size=10000, sequence_lengths=[25, 50, 100]):
        self.max_vocab_size = max_vocab_size
        self.sequence_lengths = sequence_lengths
        self.vocab = None
        self.word_to_idx = {}
        self.idx_to_word = {}
        
    def clean_text(self, text):
        """Clean and preprocess text"""
        # Lowercase
        text = text.lower()
        # Remove punctuation and special characters (keep only alphanumeric and spaces)
        text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
        return text
    
    def build_vocab(self, texts):
        """Build vocabulary from texts - keep top 10,000 words"""
        print("Building vocabulary")
        all_words = []
        
        for text in tqdm(texts, desc="Processing texts for vocabulary"):
            cleaned_text = self.clean_text(text)
            words = word_tokenize(cleaned_text)
            all_words.extend(words)
        
        # Count word frequencies and keep top 10,000
        word_counts = Counter(all_words)
        most_common = word_counts.most_common(self.max_vocab_size - 2)  # Reserve for <unk> and <pad>
        
        # Create vocabulary
        self.word_to_idx = {'<pad>': 0, '<unk>': 1}
        self.idx_to_word = {0: '<pad>', 1: '<unk>'}
        
        for idx, (word, count) in enumerate(most_common, start=2):
            self.word_to_idx[word] = idx
            self.idx_to_word[idx] = word
        
        self.vocab_size = len(self.word_to_idx)
        print(f"Vocabulary size: {self.vocab_size}")
        print(f"Top 10 words: {most_common[:10]}")
    
    def text_to_sequence(self, text, sequence_length):
        """Convert text to sequence of token IDs with padding/truncation"""
        cleaned_text = self.clean_text(text)
        words = word_tokenize(cleaned_text)
        words = words[:sequence_length]  # Truncate to sequence length
        
        # Convert words to token IDs
        sequence = [self.word_to_idx.get(word, self.word_to_idx['<unk>']) for word in words]
        
        # Pad sequence to fixed length
        if len(sequence) < sequence_length:
            sequence += [self.word_to_idx['<pad>']] * (sequence_length - len(sequence))
        
        return sequence
    
    def load_data(self):
        """Load data from processed CSV files"""
        print("Loading IMDB data")
        
        train_df = pd.read_csv('data/processed/train.csv')
        test_df = pd.read_csv('data/processed/test.csv')
        
        train_texts = train_df['review'].tolist()
        train_labels = train_df['label'].tolist()
        test_texts = test_df['review'].tolist()
        test_labels = test_df['label'].tolist()
        
        print(f"Training data: {len(train_texts)} samples")
        print(f"Test data: {len(test_texts)} samples")
        
        return (train_texts, train_labels), (test_texts, test_labels)
    
    def prepare_datasets(self, batch_size=32):
        """Prepare datasets for all sequence lengths"""
        # Load data
        (train_texts, train_labels), (test_texts, test_labels) = self.load_data()
        
        # Build vocabulary from training data only
        self.build_vocab(train_texts)
        
        data_loaders = {}
        
        for seq_len in self.sequence_lengths:
            print(f"\nPreparing sequences of length {seq_len}")
            
            # Convert texts to sequences
            print("Converting training texts to sequences")
            train_sequences = [self.text_to_sequence(text, seq_len) for text in tqdm(train_texts)]
            
            print("Converting test texts to sequences")
            test_sequences = [self.text_to_sequence(text, seq_len) for text in tqdm(test_texts)]
            
            # Convert to tensors
            train_texts_tensor = torch.tensor(train_sequences, dtype=torch.long)
            train_labels_tensor = torch.tensor(train_labels, dtype=torch.long)
            test_texts_tensor = torch.tensor(test_sequences, dtype=torch.long)
            test_labels_tensor = torch.tensor(test_labels, dtype=torch.long)
            
            # Create datasets
            train_dataset = torch.utils.data.TensorDataset(train_texts_tensor, train_labels_tensor)
            test_dataset = torch.utils.data.TensorDataset(test_texts_tensor, test_labels_tensor)
            
            # Create data loaders
            train_loader = torch.utils.data.DataLoader(
                train_dataset, batch_size=batch_size, shuffle=True
            )
            test_loader = torch.utils.data.DataLoader(
                test_dataset, batch_size=batch_size, shuffle=False
            )
            
            data_loaders[seq_len] = {
                'train': train_loader,
                'test': test_loader
            }
            
            print(f"Sequence length {seq_len}:")
            print(f"  Training batches: {len(train_loader)}")
            print(f"  Test batches: {len(test_loader)}")
        
        return data_loaders, self.vocab_size

if __name__ == "__main__":
    preprocessor = TextPreprocessor()
    data_loaders, vocab_size = preprocessor.prepare_datasets()
    print(f"\nData preparation complete! Vocabulary size: {vocab_size}")