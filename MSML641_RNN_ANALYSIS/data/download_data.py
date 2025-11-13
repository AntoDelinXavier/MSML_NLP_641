import pandas as pd
import os
import numpy as np

def process_imdb_data():
    """Process the IMDB dataset from Kaggle using kagglehub"""
    print("Processing IMDB dataset...")
    
    # Create directories
    os.makedirs('data/raw', exist_ok=True)
    os.makedirs('data/processed', exist_ok=True)
    
    try:
        import kagglehub
        
        # Download latest version
        print("Downloading IMDB dataset from Kaggle...")
        path = kagglehub.dataset_download("lakshmi25npathi/imdb-dataset-of-50k-movie-reviews")
        print(f"Dataset downloaded to: {path}")
        
        # Find the CSV file in the downloaded directory
        csv_file = None
        for file in os.listdir(path):
            if file.endswith('.csv'):
                csv_file = os.path.join(path, file)
                break
        
        if csv_file is None:
            print("Error: Could not find CSV file in downloaded dataset")
            return False
            
        print(f"Found dataset file: {csv_file}")
        
    except ImportError:
        print("kagglehub not available. Looking for existing dataset...")
        csv_file = 'data/raw/IMDB Dataset.csv'
        if not os.path.exists(csv_file):
            print(f"Error: IMDB dataset not found at {csv_file}")
            print("Please install kagglehub: pip install kagglehub")
            print("Or manually download from: https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews")
            return False
    
    # Load the dataset
    print("Loading IMDB dataset...")
    df = pd.read_csv(csv_file)
    
    print(f"Original dataset shape: {df.shape}")
    print(f"Sentiment distribution:\n{df['sentiment'].value_counts()}")
    
    # Convert sentiment to binary (positive=1, negative=0)
    df['label'] = df['sentiment'].map({'positive': 1, 'negative': 0})
    
    # Shuffle the dataset
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    # Split into train and test (25k each)
    train_df = df[:25000]
    test_df = df[25000:50000]
    
    # Save processed data
    train_df[['review', 'label']].to_csv('data/processed/train.csv', index=False)
    test_df[['review', 'label']].to_csv('data/processed/test.csv', index=False)
    
    print(f"Training data: {len(train_df)} samples")
    print(f"Test data: {len(test_df)} samples")
    print("Data processing complete!")
    return True

if __name__ == "__main__":
    process_imdb_data()