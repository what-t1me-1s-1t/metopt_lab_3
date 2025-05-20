import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import make_regression
from memory_profiler import memory_usage
import time
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Any, Optional
import optuna

class RegularizedLinearRegression(nn.Module):
    def __init__(self, input_dim: int, regularization: Optional[str] = None, lambda_val: float = 0.1):
        super().__init__()
        self.linear = nn.Linear(input_dim, 1)
        self.regularization = regularization
        self.lambda_val = lambda_val
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)
    
    def get_regularization_loss(self) -> torch.Tensor:
        if self.regularization is None:
            return torch.tensor(0.0)
        
        weights = self.linear.weight
        
        if self.regularization == 'L1':
            return self.lambda_val * torch.norm(weights, p=1)
        elif self.regularization == 'L2':
            return self.lambda_val * torch.norm(weights, p=2) ** 2
        elif self.regularization == 'Elastic':
            return self.lambda_val * (0.5 * torch.norm(weights, p=2) ** 2 + 0.5 * torch.norm(weights, p=1))
        else:
            return torch.tensor(0.0)

def generate_data(n_samples: int = 1000, n_features: int = 10, noise: float = 0.1) -> Tuple[np.ndarray, np.ndarray]:
    X, y = make_regression(n_samples=n_samples, n_features=n_features, noise=noise)
    return X, y

def train_model(X: np.ndarray, y: np.ndarray, 
                regularization: Optional[str] = None,
                lambda_val: float = 0.1,
                batch_size: int = 32,
                learning_rate: float = 0.01,
                epochs: int = 1000) -> Dict[str, Any]:
    
    model = RegularizedLinearRegression(X.shape[1], regularization, lambda_val)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    X_tensor = torch.FloatTensor(X)
    y_tensor = torch.FloatTensor(y).reshape(-1, 1)
    
    start_mem = memory_usage(-1, interval=1, timeout=1)
    start_time = time.time()
    
    losses = []
    for epoch in range(epochs):
        epoch_loss = 0
        for i in range(0, len(X), batch_size):
            batch_X = X_tensor[i:i+batch_size]
            batch_y = y_tensor[i:i+batch_size]
            
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y) + model.get_regularization_loss()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
        
        losses.append(epoch_loss / (len(X) // batch_size))
    
    end_time = time.time()
    end_mem = memory_usage(-1, interval=1, timeout=1)
    
    return {
        'memory_used': max(end_mem) - max(start_mem),
        'time_taken': end_time - start_time,
        'weights': model.linear.weight.data.numpy().flatten(),
        'final_loss': losses[-1],
        'loss_history': losses
    }

def objective(trial: optuna.Trial) -> float:
    X, y = generate_data()
    
    regularization = trial.suggest_categorical('regularization', [None, 'L1', 'L2', 'Elastic'])
    lambda_val = trial.suggest_float('lambda_val', 1e-5, 1.0, log=True)
    batch_size = trial.suggest_int('batch_size', 1, len(X))
    learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-1, log=True)
    
    results = train_model(X, y, regularization, lambda_val, batch_size, learning_rate)
    return results['final_loss']

def plot_results(df: pd.DataFrame):
    plt.figure(figsize=(15, 10))
    
    # Plot memory usage
    plt.subplot(2, 2, 1)
    sns.barplot(data=df, x='Regularization', y='Memory Used')
    plt.xticks(rotation=45)
    plt.title('Memory Usage by Regularization')
    
    # Plot time taken
    plt.subplot(2, 2, 2)
    sns.barplot(data=df, x='Regularization', y='Time Taken')
    plt.xticks(rotation=45)
    plt.title('Time Taken by Regularization')
    
    # Plot final loss
    plt.subplot(2, 2, 3)
    sns.barplot(data=df, x='Regularization', y='Final Loss')
    plt.xticks(rotation=45)
    plt.title('Final Loss by Regularization')
    
    plt.tight_layout()
    plt.savefig('regularization_results.png')
    plt.close()

def plot_loss_histories(df: pd.DataFrame):
    plt.figure(figsize=(10, 6))
    
    for _, row in df.iterrows():
        plt.plot(row['Loss History'], label=f"{row['Regularization']} (λ={row['Lambda']})")
    
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss History by Regularization Method')
    plt.legend()
    plt.grid(True)
    plt.savefig('loss_histories.png')
    plt.close()

def main():
    # Generate data
    X, y = generate_data(n_samples=1000, n_features=10)
    
    # Test different regularization methods
    regularizations = [None, 'L1', 'L2', 'Elastic']
    lambda_values = [0.01, 0.1, 1.0]
    
    results = []
    for reg in regularizations:
        for lambda_val in lambda_values:
            result = train_model(X, y, reg, lambda_val)
            results.append({
                'Regularization': reg if reg is not None else 'None',
                'Lambda': lambda_val,
                'Memory Used': result['memory_used'],
                'Time Taken': result['time_taken'],
                'Final Loss': result['final_loss'],
                'Loss History': result['loss_history']
            })
    
    # Convert results to DataFrame
    df = pd.DataFrame(results)
    
    # Plot results
    plot_results(df)
    plot_loss_histories(df)
    
    # Print results
    print("\nResults Summary:")
    print(df[['Regularization', 'Lambda', 'Memory Used', 'Time Taken', 'Final Loss']].to_string())
    
    # Run Optuna optimization
    study = optuna.create_study()
    study.optimize(objective, n_trials=50)
    
    print("\nOptuna Optimization Results:")
    print(f"Best regularization: {study.best_params['regularization']}")
    print(f"Best lambda value: {study.best_params['lambda_val']}")
    print(f"Best batch size: {study.best_params['batch_size']}")
    print(f"Best learning rate: {study.best_params['learning_rate']}")
    print(f"Best loss: {study.best_value}")

if __name__ == '__main__':
    main()