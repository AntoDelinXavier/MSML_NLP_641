import torch
import torch.nn as nn
import torch.nn.functional as F

class RNNModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim=100, hidden_dim=64, 
                 num_layers=2, dropout=0.3, rnn_type='rnn', 
                 activation='relu', bidirectional=False):
        super(RNNModel, self).__init__()
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.dropout1 = nn.Dropout(dropout)
        
        # Choose RNN type
        if rnn_type.lower() == 'lstm':
            self.rnn = nn.LSTM(embedding_dim, hidden_dim, num_layers, 
                              batch_first=True, dropout=dropout, 
                              bidirectional=bidirectional)
        elif rnn_type.lower() == 'gru':
            self.rnn = nn.GRU(embedding_dim, hidden_dim, num_layers,
                             batch_first=True, dropout=dropout,
                             bidirectional=bidirectional)
        else:  # Simple RNN
            self.rnn = nn.RNN(embedding_dim, hidden_dim, num_layers,
                             batch_first=True, dropout=dropout,
                             bidirectional=bidirectional)
        
        # Choose activation function
        if activation.lower() == 'sigmoid':
            self.activation = nn.Sigmoid()
        elif activation.lower() == 'tanh':
            self.activation = nn.Tanh()
        else:  # ReLU
            self.activation = nn.ReLU()
        
        self.bidirectional = bidirectional
        multiplier = 2 if bidirectional else 1
        
        self.fc1 = nn.Linear(hidden_dim * multiplier, hidden_dim)
        self.dropout2 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim, 1)
        
    def forward(self, x):
        # Embedding layer
        embedded = self.embedding(x)
        embedded = self.dropout1(embedded)
        
        # RNN layer
        rnn_out, _ = self.rnn(embedded)
        
        # Use the last hidden state
        if self.bidirectional:
            # For bidirectional, concatenate last forward and last backward
            last_output = rnn_out[:, -1, :]
        else:
            last_output = rnn_out[:, -1, :]
        
        # Fully connected layers
        hidden = self.activation(self.fc1(last_output))
        hidden = self.dropout2(hidden)
        output = torch.sigmoid(self.fc2(hidden))
        
        return output.squeeze()

def create_model(vocab_size, config):
    """Create model based on configuration"""
    return RNNModel(
        vocab_size=vocab_size,
        embedding_dim=config.get('embedding_dim', 100),
        hidden_dim=config.get('hidden_dim', 64),
        num_layers=config.get('num_layers', 2),
        dropout=config.get('dropout', 0.3),
        rnn_type=config.get('rnn_type', 'rnn'),
        activation=config.get('activation', 'relu'),
        bidirectional=config.get('bidirectional', False)
    )