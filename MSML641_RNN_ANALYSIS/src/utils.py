import torch
import random
import numpy as np
import os
import json
from datetime import datetime

def set_seeds(seed=42):
    """Set random seeds for reproducibility"""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_hardware_info():
    """Get hardware information"""
    info = {
        'device': 'GPU' if torch.cuda.is_available() else 'CPU',
        'cpu_cores': os.cpu_count(),
        'cuda_available': torch.cuda.is_available(),
    }
    
    if torch.cuda.is_available():
        info['gpu_name'] = torch.cuda.get_device_name(0)
        info['gpu_memory'] = f"{torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB"
    
    return info

def save_results(results, filename):
    """Save results to JSON file"""
    # Convert numpy values to Python native types
    def convert_values(obj):
        if isinstance(obj, dict):
            return {k: convert_values(v) for k, v in obj.items()}
        elif isinstance(obj, (np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.float64, np.float32)):
            return float(obj)
        else:
            return obj
    
    results = convert_values(results)
    
    with open(filename, 'w') as f:
        json.dump(results, f, indent=2)

def load_results(filename):
    """Load results from JSON file"""
    with open(filename, 'r') as f:
        return json.load(f)