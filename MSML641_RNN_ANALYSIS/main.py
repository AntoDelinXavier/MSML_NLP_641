import torch
import random
import numpy as np
import os
import pandas as pd
import sys
from datetime import datetime
from data.download_data import process_imdb_data
from src.preprocess import TextPreprocessor
from src.models import create_model, RNNModel
from src.train import Trainer
from src.utils import set_seeds, get_hardware_info, save_results
import time
from sklearn.metrics import accuracy_score, f1_score

class Tee:
    """Class to duplicate output to both terminal and file"""
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, 'w')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()

    def flush(self):
        self.terminal.flush()
        self.log.flush()

def run_complete_project():
    """Run the entire sentiment analysis project"""
    # Create results directory first
    os.makedirs('results', exist_ok=True)
    
    # Redirect output to file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = f'results/experiment_output_{timestamp}.txt'
    tee = Tee(log_file)
    sys.stdout = tee
    
    print("=" * 70)
    print("COMPARATIVE ANALYSIS OF RNN ARCHITECTURES FOR SENTIMENT CLASSIFICATION")
    print("=" * 70)
    print(f"Output logged to: {log_file}")
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Set random seeds for reproducibility
    set_seeds(42)
    
    # Create directories
    os.makedirs('results/plots', exist_ok=True)
    
    # Hardware info
    hardware_info = get_hardware_info()
    print("\nHARDWARE INFORMATION:")
    for key, value in hardware_info.items():
        print(f"  {key}: {value}")
    
    # Step 1: Data Processing
    print("\n" + "="*50)
    print("STEP 1: DATA PROCESSING")
    print("="*50)
    
    if not os.path.exists('data/processed/train.csv'):
        print("Processing IMDB dataset")
        process_imdb_data()
    else:
        print("Processed data already exists. Skipping data processing.")
    
    # Step 2: Data Preprocessing
    print("\n" + "="*50)
    print("STEP 2: DATA PREPROCESSING")
    print("="*50)
    
    preprocessor = TextPreprocessor(max_vocab_size=10000, sequence_lengths=[25, 50, 100])
    data_loaders, vocab_size = preprocessor.prepare_datasets(batch_size=32)
    
    # Step 3: Model Experiments - 14 Systematic Combinations
    print("\n" + "="*50)
    print("STEP 3: RUNNING EXPERIMENTS")
    print("="*50)
    
    # 14 Systematic Combinations: 10 core + 4 critical extras
    experiments = [
        # ========== CORE SYSTEMATIC COMBINATIONS (10) ==========
        # Baseline
        {'name': 'LSTM_ReLU_Adam_seq50', 'rnn_type': 'lstm', 'activation': 'relu', 'optimizer': 'adam', 'seq_len': 50},
        
        # Architecture variations (2)
        {'name': 'RNN_ReLU_Adam_seq50', 'rnn_type': 'rnn', 'activation': 'relu', 'optimizer': 'adam', 'seq_len': 50},
        {'name': 'BiLSTM_ReLU_Adam_seq50', 'rnn_type': 'lstm', 'activation': 'relu', 'optimizer': 'adam', 'seq_len': 50, 'bidirectional': True},
        
        # Activation variations (2)
        {'name': 'LSTM_Sigmoid_Adam_seq50', 'rnn_type': 'lstm', 'activation': 'sigmoid', 'optimizer': 'adam', 'seq_len': 50},
        {'name': 'LSTM_Tanh_Adam_seq50', 'rnn_type': 'lstm', 'activation': 'tanh', 'optimizer': 'adam', 'seq_len': 50},
        
        # Optimizer variations (2)
        {'name': 'LSTM_ReLU_SGD_seq50', 'rnn_type': 'lstm', 'activation': 'relu', 'optimizer': 'sgd', 'seq_len': 50},
        {'name': 'LSTM_ReLU_RMSprop_seq50', 'rnn_type': 'lstm', 'activation': 'relu', 'optimizer': 'rmsprop', 'seq_len': 50},
        
        # Sequence length variations (2)
        {'name': 'LSTM_ReLU_Adam_seq25', 'rnn_type': 'lstm', 'activation': 'relu', 'optimizer': 'adam', 'seq_len': 25},
        {'name': 'LSTM_ReLU_Adam_seq100', 'rnn_type': 'lstm', 'activation': 'relu', 'optimizer': 'adam', 'seq_len': 100},
        
        # Stability strategy (1)
        {'name': 'LSTM_ReLU_Adam_seq50_clip', 'rnn_type': 'lstm', 'activation': 'relu', 'optimizer': 'adam', 'seq_len': 50, 'grad_clip': 1.0},
        
        # ========== CRITICAL EXTRA COMBINATIONS (4) ==========
        {'name': 'RNN_Tanh_Adam_seq50', 'rnn_type': 'rnn', 'activation': 'tanh', 'optimizer': 'adam', 'seq_len': 50},
        {'name': 'BiLSTM_Tanh_Adam_seq50', 'rnn_type': 'lstm', 'activation': 'tanh', 'optimizer': 'adam', 'seq_len': 50, 'bidirectional': True},
        {'name': 'LSTM_Tanh_RMSprop_seq50', 'rnn_type': 'lstm', 'activation': 'tanh', 'optimizer': 'rmsprop', 'seq_len': 50},
        {'name': 'RNN_ReLU_Adam_seq100', 'rnn_type': 'rnn', 'activation': 'relu', 'optimizer': 'adam', 'seq_len': 100},
    ]
    
    print(f"Total experiments to run: {len(experiments)}")
    print("Breakdown:")
    print("  - Core systematic combinations: 10")
    print("  - Critical extra combinations: 4")
    print("\nExperiment List:")
    for i, exp in enumerate(experiments, 1):
        print(f"  {i:2d}. {exp['name']}")
    
    results = {}
    training_curves = {}
    
    for exp_config in experiments:
        print(f"\n{'='*60}")
        print(f"EXPERIMENT: {exp_config['name']}")
        print(f"{'='*60}")
        
        start_time = time.time()
        
        # Get data loader for this sequence length
        seq_len = exp_config['seq_len']
        train_loader = data_loaders[seq_len]['train']
        test_loader = data_loaders[seq_len]['test']
        
        # Create model
        model_config = {
            'vocab_size': vocab_size,
            'embedding_dim': 100,
            'hidden_dim': 64,
            'num_layers': 2,
            'dropout': 0.3,
            'rnn_type': exp_config['rnn_type'],
            'activation': exp_config['activation'],
            'bidirectional': exp_config.get('bidirectional', False)
        }
        
        model = RNNModel(**model_config)
        
        # Create trainer
        trainer_config = {
            'optimizer': exp_config['optimizer'],
            'learning_rate': 0.001,
            'grad_clip': exp_config.get('grad_clip', None)
        }
        
        trainer = Trainer(model, train_loader, test_loader, trainer_config)
        
        # Train model
        training_results = trainer.train(epochs=5)
        
        # Simple evaluation (no Evaluator class needed)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.to(device)
        model.eval()
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.cpu().numpy()
                output = model(data)
                predictions = (output.cpu().numpy() > 0.5).astype(int)
                all_predictions.extend(predictions)
                all_targets.extend(target)
        
        # Calculate metrics
        accuracy = accuracy_score(all_targets, all_predictions)
        f1 = f1_score(all_targets, all_predictions, average='macro')
        
        # Calculate epoch time
        total_time = time.time() - start_time
        epoch_time = total_time / 5  # Average per epoch
        
        # Store results
        results[exp_config['name']] = {
            'accuracy': accuracy,
            'f1_score': f1,
            'epoch_time': epoch_time,
            'total_time': total_time,
            'config': exp_config
        }
        
        training_curves[exp_config['name']] = {
            'train_losses': training_results['train_losses'],
            'val_accuracies': training_results['val_accuracies']
        }
        
        print(f"RESULTS for {exp_config['name']}:")
        print(f"  Accuracy: {accuracy:.4f}")
        print(f"  F1-Score: {f1:.4f}")
        print(f"  Avg Epoch Time: {epoch_time:.2f}s")
        print(f"  Total Time: {total_time:.2f}s")
    
    # Step 4: Results Analysis
    print("\n" + "="*50)
    print("STEP 4: RESULTS ANALYSIS")
    print("="*50)
    
    # Save results
    save_results(results, 'results/experiment_results.json')
    save_results(training_curves, 'results/training_curves.json')
    
    # Create comprehensive summary table in exact required format
    summary_data = []
    for config_name, metrics in results.items():
        config = metrics['config']
        row = {
            'Model': config['rnn_type'].upper() + ('-Bi' if config.get('bidirectional', False) else ''),
            'Activation': config['activation'].capitalize(),
            'Optimizer': config['optimizer'].capitalize(),
            'Seq_Length': config['seq_len'],
            'Grad_Clipping': 'Yes' if config.get('grad_clip') else 'No',
            'Accuracy': metrics['accuracy'],
            'F1': metrics['f1_score'],
            'Epoch_Time_s': metrics['epoch_time']
        }
        summary_data.append(row)
    
    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv('results/metrics.csv', index=False)
    
    print("\nCOMPREHENSIVE SUMMARY TABLE:")
    print("="*100)
    print(summary_df.to_string(index=False, float_format='%.4f'))
    print("="*100)
    
    
    # Import plotting functions
    from src.evaluate import plot_training_curves_best_worst_only, plot_accuracy_f1_all_combinations
    
    # Plot 1: Training Loss vs Epochs (left) and Validation Accuracy vs Epochs (right) for BEST and WORST only
    plot_training_curves_best_worst_only(training_curves, results, 'results/plots/training_curves_best_worst.png')
    
    # Plot 2: Accuracy vs Sequence Length (left) and F1-Score vs Sequence Length (right) for ALL 14 combinations
    plot_accuracy_f1_all_combinations(results, 'results/plots/accuracy_f1_all_combinations.png')
    
    print("Plots saved to results/plots/")
    
    # Find best and worst configurations
    best_config_name = max(results.items(), key=lambda x: x[1]['accuracy'])[0]
    worst_config_name = min(results.items(), key=lambda x: x[1]['accuracy'])[0]
    best_config = results[best_config_name]
    worst_config = results[worst_config_name]
    
    print(f"\n{'='*60}")
    print("BEST AND WORST CONFIGURATIONS")
    print(f"{'='*60}")
    print(f"BEST: {best_config_name}")
    print(f"  Accuracy: {best_config['accuracy']:.4f}")
    print(f"  F1-Score: {best_config['f1_score']:.4f}")
    print(f"  Configuration: {best_config['config']}")
    print(f"\nWORST: {worst_config_name}")
    print(f"  Accuracy: {worst_config['accuracy']:.4f}")
    print(f"  F1-Score: {worst_config['f1_score']:.4f}")
    print(f"  Configuration: {worst_config['config']}")
    
    # Final summary
    total_experiment_time = sum([metrics['total_time'] for metrics in results.values()])
    print(f"\n{'='*60}")
    print("PROJECT SUMMARY")
    print(f"{'='*60}")
    print(f"Results saved in 'results/' directory")
    print(f"Total experiments completed: {len(experiments)}")
    print(f"Total experiment time: {total_experiment_time/60:.1f} minutes")
    print(f"Best model: {best_config_name}")
    print(f"Best accuracy: {best_config['accuracy']:.4f}")
    print(f"Worst model: {worst_config_name}")
    print(f"Worst accuracy: {worst_config['accuracy']:.4f}")
    print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Output log: {log_file}")
    
    # Restore stdout
    sys.stdout = tee.terminal
    tee.log.close()

if __name__ == "__main__":
    run_complete_project()