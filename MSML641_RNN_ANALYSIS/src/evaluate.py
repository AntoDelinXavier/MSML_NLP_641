import torch
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, classification_report
import matplotlib.pyplot as plt  # <-- ADD THIS IMPORT
import seaborn as sns
import pandas as pd
from tqdm import tqdm

class Evaluator:
    def __init__(self, model, test_loader, device):
        self.model = model
        self.test_loader = test_loader
        self.device = device
        self.model.to(self.device)
    
    def evaluate(self):
        """Comprehensive evaluation"""
        self.model.eval()
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for data, target in tqdm(self.test_loader, desc="Evaluating"):
                data, target = data.to(self.device), target.cpu().numpy()
                output = self.model(data)
                predictions = (output.cpu().numpy() > 0.5).astype(int)
                
                all_predictions.extend(predictions)
                all_targets.extend(target)
        
        # Calculate metrics
        accuracy = accuracy_score(all_targets, all_predictions)
        f1 = f1_score(all_targets, all_predictions, average='macro')
        
        return {
            'accuracy': accuracy,
            'f1_score': f1,
            'predictions': all_predictions,
            'targets': all_targets
        }

def plot_training_curves_best_worst_only(experiment_results, results_dict, save_path=None):
    """Plot training loss vs epochs (left) and validation accuracy vs epochs (right) for BEST and WORST models only"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Find best and worst models
    best_model = max(results_dict.items(), key=lambda x: x[1]['accuracy'])[0]
    worst_model = min(results_dict.items(), key=lambda x: x[1]['accuracy'])[0]
    
    print(f"Plotting training curves for BEST: {best_model} and WORST: {worst_model}")
    
    # Colors for best and worst
    colors = {'best': '#2E8B57', 'worst': '#DC143C'}  # SeaGreen and Crimson
    
    models_to_plot = {
        'best': (best_model, colors['best'], 'BEST'),
        'worst': (worst_model, colors['worst'], 'WORST')
    }
    
    for label, (model_name, color, display_name) in models_to_plot.items():
        if model_name in experiment_results:
            results = experiment_results[model_name]
            
            # LEFT: Training Loss vs Epochs
            if 'train_losses' in results:
                epochs = range(1, len(results['train_losses']) + 1)
                ax1.plot(epochs, results['train_losses'], 
                        label=f'{display_name}: {model_name}', 
                        color=color, linewidth=3, alpha=0.8, marker='o')
            
            # RIGHT: Validation Accuracy vs Epochs  
            if 'val_accuracies' in results:
                epochs = range(1, len(results['val_accuracies']) + 1)
                ax2.plot(epochs, results['val_accuracies'], 
                        label=f'{display_name}: {model_name}',
                        color=color, linewidth=3, alpha=0.8, marker='s')
    
    # LEFT PLOT: Training Loss vs Epochs
    ax1.set_title('Training Loss vs Epochs\n(Best and Worst Models)', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Training Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_xticks(range(1, 6))  # Since we have 5 epochs
    
    # RIGHT PLOT: Validation Accuracy vs Epochs
    ax2.set_title('Validation Accuracy vs Epochs\n(Best and Worst Models)', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Validation Accuracy')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_xticks(range(1, 6))  # Since we have 5 epochs
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.show()

def plot_accuracy_f1_all_combinations(metrics_dict, save_path=None):
    """Plot accuracy and F1 for ALL 14 combinations"""
    # Extract data for all combinations
    combinations_data = []
    
    for config_name, metrics in metrics_dict.items():
        config = metrics['config']
        combinations_data.append({
            'Model': config_name,
            'Architecture': config['rnn_type'].upper() + ('-Bi' if config.get('bidirectional', False) else ''),
            'Activation': config['activation'].capitalize(),
            'Optimizer': config['optimizer'].capitalize(),
            'Seq_Length': config['seq_len'],
            'Grad_Clipping': 'Yes' if config.get('grad_clip') else 'No',
            'Accuracy': metrics['accuracy'],
            'F1_Score': metrics['f1_score']
        })
    
    # Create DataFrame
    df = pd.DataFrame(combinations_data)
    
    # Create subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))
    
    # Generate unique colors and markers for each of the 14 combinations
    colors = plt.cm.tab20(np.linspace(0, 1, len(df)))
    markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*', 'h', 'H', '+', 'x', 'd']
    
    # Plot each combination
    for i, (idx, row) in enumerate(df.iterrows()):
        color = colors[i]
        marker = markers[i % len(markers)]
        
        # Accuracy plot (left)
        ax1.scatter(row['Seq_Length'], row['Accuracy'], 
                   color=color, marker=marker, s=150, alpha=0.8,
                   label=row['Model'])
        
        # F1 plot (right)
        ax2.scatter(row['Seq_Length'], row['F1_Score'],
                   color=color, marker=marker, s=150, alpha=0.8,
                   label=row['Model'])
    
    ax1.set_xlabel('Sequence Length', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Accuracy', fontsize=12, fontweight='bold')
    ax1.set_title('Accuracy vs Sequence Length\n(All 14 Combinations)', fontsize=14, fontweight='bold')
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
    ax1.grid(True, alpha=0.3)
    
    ax2.set_xlabel('Sequence Length', fontsize=12, fontweight='bold')
    ax2.set_ylabel('F1-Score', fontsize=12, fontweight='bold')
    ax2.set_title('F1-Score vs Sequence Length\n(All 14 Combinations)', fontsize=14, fontweight='bold')
    ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
    ax2.grid(True, alpha=0.3)
    
    # Set consistent x-axis limits
    ax1.set_xlim(20, 105)
    ax2.set_xlim(20, 105)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.show()
    
    # Print summary
    print(f"\nPlotted all {len(df)} combinations:")
    for i, row in df.iterrows():
        print(f"  {i+1:2d}. {row['Model']}: Acc={row['Accuracy']:.4f}, F1={row['F1_Score']:.4f}")

def create_summary_table(metrics_dict):
    """Create summary table from metrics"""
    summary_data = []
    
    for config_name, metrics in metrics_dict.items():
        row = {
            'Model': config_name,
            'Accuracy': metrics.get('accuracy', 0),
            'F1-Score': metrics.get('f1_score', 0),
            'Epoch Time (s)': metrics.get('epoch_time', 0),
            'Total Time (s)': metrics.get('total_time', 0)
        }
        summary_data.append(row)
    
    return pd.DataFrame(summary_data)