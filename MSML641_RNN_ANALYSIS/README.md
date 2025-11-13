# Comparative Analysis of RNN Architectures for Sentiment Classification

A comprehensive comparative analysis of different RNN architectures (RNN, LSTM, Bidirectional LSTM) for sentiment classification on the IMDB movie review dataset.

## Project Overview

This project implements and evaluates various RNN architectures for binary sentiment classification (positive/negative) using the IMDB movie review dataset. The study systematically compares 14 different configurations including architectures, activation functions, optimizers, sequence lengths, and stability strategies.

### Key Findings

- **Best Model**: LSTM + Tanh + Adam + Sequence Length 50 (**77.19% accuracy**)
- **Worst Model**: LSTM + ReLU + SGD + Sequence Length 50 (**49.67% accuracy**)
- **Performance Gap**: LSTM outperforms simple RNN by ~18% absolute improvement
- **Optimal Sequence Length**: 50 tokens (balanced performance vs computational cost)

## Repository Structure

```
├── data/
│   ├── download_data.py          # Dataset downloading and processing
│   ├── raw                       # Raw data
│   └── processed/                # Processed CSV files 
├── src/
│   ├── preprocess.py             # Text preprocessing utilities
│   ├── models.py                 # RNN model implementations
│   ├── train.py                  # Training scripts
│   ├── evaluate.py               # Evaluation and plotting utilities
│   └── utils.py                  # Helper functions
├── results/                      
│   ├── metrics.csv               # Saved metrics CSV files
│   ├── experiment_output.txt     # Complete execution log
│   ├── experiment_results.json.  # Detailed metrics for all models
│   ├── training_curves.json.     # Training history data
│   └── plots/                    # Generated plots and visualizations
├── main.py                       # Main experiment runner
├── requirements.txt              # Python dependencies
├── report.pdf                    # Detailed anlaysis report
└── README.md
```

## Setup Instructions

### Prerequisites

- Python 3.8 or higher
- 4GB+ RAM recommended
- 5GB+ free disk space

### Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/AntoDelinXavier/MSML_NLP_610.git
   cd MSML_NLP_610
   ```

2. **Create virtual environment** (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

### Dependencies

Key packages included in `requirements.txt`:
- `torch>=1.9.0`
- `torchtext>=0.10.0`
- `numpy>=1.21.0`
- `pandas>=1.3.0`
- `scipy>=1.7.0`
- `matplotlib>=3.4.0`
- `seaborn>=0.11.0`
- `scikit-learn>=1.0.0`
- `nltk>=3.6.0`
- `tqdm>=4.62.0`
- `PyYAML>=6.0`
- `requests>=2.25.0`
- `jupyter>=1.0.0`
- `torchsummary>=1.5.1`
- `kagglehub>=0.2.0`



## How to Run

### Run Complete Project

To run the entire comparative analysis with all 14 configurations:

```bash
python main.py
```

This will automatically:
1. Download and preprocess the IMDB dataset
2. Run all 14 experimental configurations
3. Generate comprehensive results and plots
4. Save outputs to the `results/` directory

## Results Summary

| Model | Architecture | Activation | Optimizer | Seq Length | Accuracy | F1-Score | Epoch Time (s) |
|-------|--------------|------------|-----------|------------|----------|----------|----------------|
| Best | LSTM | Tanh | Adam | 50 | **77.19%** | **77.17%** | 71.96 |
| Worst | LSTM | ReLU | SGD | 50 | **49.67%** | **49.65%** | 85.18 |

---

