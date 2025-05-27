# SGD Implementation for Regression Analysis

This project implements Stochastic Gradient Descent (SGD) for polynomial and linear regression with various configurations:
- Different batch sizes (from single sample to full batch)
- Various learning rate schedules
- Different regularization methods (L1, L2, Elastic Net)

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

Run the analysis script to see comparisons of different configurations:
```bash
python analysis.py
```

This will generate three plots:
- `batch_size_comparison.png`: Compares convergence for different batch sizes
- `lr_scheduler_comparison.png`: Compares different learning rate scheduling methods
- `regularization_comparison.png`: Compares different regularization techniques

## Implementation Details

### Batch Sizes
- Single sample (pure SGD)
- Mini-batch (8, 32, 128 samples)
- Full batch (standard Gradient Descent)

### Learning Rate Schedules
- Constant
- Time-based decay
- Step decay
- Exponential decay

### Regularization Methods
- None (no regularization)
- L1 (Lasso)
- L2 (Ridge)
- Elastic Net (combination of L1 and L2)

## Customization

You can modify the `CustomSGDRegressor` parameters in `analysis.py` to experiment with:
- Learning rates
- Number of epochs
- Regularization strength
- Polynomial degree
- Batch sizes
- Early stopping tolerance 