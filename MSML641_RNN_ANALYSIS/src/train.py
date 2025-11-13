import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import time
import numpy as np

class Trainer:
    def __init__(self, model, train_loader, val_loader, config):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        
        # Set device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
        # Optimizer
        optimizer_name = config.get('optimizer', 'adam').lower()
        lr = config.get('learning_rate', 0.001)
        
        if optimizer_name == 'sgd':
            self.optimizer = optim.SGD(model.parameters(), lr=lr)
        elif optimizer_name == 'rmsprop':
            self.optimizer = optim.RMSprop(model.parameters(), lr=lr)
        else:  # Adam
            self.optimizer = optim.Adam(model.parameters(), lr=lr)
        
        # Loss function
        self.criterion = nn.BCELoss()
        
        # Gradient clipping
        self.grad_clip = config.get('grad_clip', None)
        
        self.train_losses = []
        self.val_accuracies = []
        
    def train_epoch(self):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        num_batches = 0
        
        for batch_idx, (data, target) in enumerate(tqdm(self.train_loader, desc="Training", leave=False)):
            data, target = data.to(self.device), target.float().to(self.device)
            
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.criterion(output, target)
            loss.backward()
            
            # Apply gradient clipping if specified
            if self.grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
            
            self.optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
        
        return total_loss / num_batches
    
    def validate(self):
        """Validate the model"""
        self.model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in self.val_loader:
                data, target = data.to(self.device), target.float().to(self.device)
                output = self.model(data)
                pred = (output > 0.5).float()
                correct += (pred == target).sum().item()
                total += target.size(0)
        
        return correct / total
    
    def train(self, epochs=10):
        """Full training loop"""
        print(f"Starting training on {self.device}")
        print(f"Configuration: {self.config}")
        
        start_time = time.time()
        
        for epoch in range(epochs):
            epoch_start = time.time()
            
            # Train
            train_loss = self.train_epoch()
            self.train_losses.append(train_loss)
            
            # Validate
            val_accuracy = self.validate()
            self.val_accuracies.append(val_accuracy)
            
            epoch_time = time.time() - epoch_start
            
            print(f'Epoch {epoch+1}/{epochs}: '
                  f'Train Loss: {train_loss:.4f}, '
                  f'Val Acc: {val_accuracy:.4f}, '
                  f'Time: {epoch_time:.2f}s')
        
        total_time = time.time() - start_time
        print(f'Total training time: {total_time:.2f}s')
        
        return {
            'train_losses': self.train_losses,
            'val_accuracies': self.val_accuracies,
            'total_time': total_time
        }