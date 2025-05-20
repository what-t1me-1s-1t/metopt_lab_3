import tensorflow as tf
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import make_regression
import numpy as np
from memory_profiler import memory_usage
import time
import pandas as pd
import optuna
from typing import Callable, List, Tuple, Dict, Any
import matplotlib.pyplot as plt
import seaborn as sns


def loss(X, y, w):
    return np.sum((np.dot(X, w) - y) ** 2) / len(y)


def gradient(X, y, w):
    return 2 * np.dot(X.T, (np.dot(X, w) - y)) / len(y)


def SGD(X, y, h, lambda_val, batch_size=20, learning_rate_schedule=None, max_iter=1000):
    w = np.zeros(X.shape[1])  # инициализировать веса
    Q = loss(X, y, w)  # инициализировать оценку функционала

    for it in range(max_iter):
        if learning_rate_schedule is not None:
            h = learning_rate_schedule(it)

        batch_indices = np.random.choice(X.shape[0], size=batch_size, replace=False)
        X_batch = X[batch_indices]
        y_batch = y[batch_indices]

        eps = loss(X_batch, y_batch, w)  # вычислить потерю
        w = w - h * gradient(X_batch, y_batch, w)  # обновить вектор весов в направлении антиградиента
        Q_new = lambda_val * eps + (1 - lambda_val) * Q  # оценить функционал

        if np.abs(Q_new - Q) < 1e-6:  # проверить сходимость
            break

        Q = Q_new

    return w


def step_decay_schedule(initial_lr=0.1, decay_factor=0.5, step_size=10):
    def schedule(epoch):
        return initial_lr * (decay_factor ** np.floor(epoch / step_size))

    return schedule


def generate_data(n_samples: int = 1000, n_features: int = 10, noise: float = 0.1) -> Tuple[np.ndarray, np.ndarray]:
    X, y = make_regression(n_samples=n_samples, n_features=n_features, noise=noise)
    return X, y


class LinearRegression(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.linear = nn.Linear(input_dim, 1)
        
    def forward(self, x):
        return self.linear(x)


def create_learning_rate_schedules() -> Dict[str, Callable]:
    def constant_lr(epoch: int) -> float:
        return 0.01
    
    def step_decay(epoch: int) -> float:
        initial_lr = 0.01
        decay_factor = 0.5
        step_size = 100
        return initial_lr * (decay_factor ** (epoch // step_size))
    
    def exponential_decay(epoch: int) -> float:
        initial_lr = 0.01
        decay_rate = 0.001
        return initial_lr * np.exp(-decay_rate * epoch)
    
    return {
        'constant': constant_lr,
        'step_decay': step_decay,
        'exponential': exponential_decay
    }


def get_tensorflow_optimizers() -> List[tf.keras.optimizers.Optimizer]:
    return [
        tf.keras.optimizers.SGD(learning_rate=0.01),
        tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.9),
        tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.9, nesterov=True),
        tf.keras.optimizers.Adagrad(learning_rate=0.01),
        tf.keras.optimizers.RMSprop(learning_rate=0.01),
        tf.keras.optimizers.Adam(learning_rate=0.01)
    ]


def get_pytorch_optimizers(model: nn.Module) -> List[optim.Optimizer]:
    return [
        optim.SGD(model.parameters(), lr=0.01),
        optim.SGD(model.parameters(), lr=0.01, momentum=0.9),
        optim.SGD(model.parameters(), lr=0.01, momentum=0.9, nesterov=True),
        optim.Adagrad(model.parameters(), lr=0.01),
        optim.RMSprop(model.parameters(), lr=0.01),
        optim.Adam(model.parameters(), lr=0.01)
    ]


def train_tensorflow_model(X: np.ndarray, y: np.ndarray, optimizer: tf.keras.optimizers.Optimizer, 
                         batch_size: int, epochs: int = 1000) -> Dict[str, Any]:
    model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(1, input_shape=(X.shape[1],))
    ])
    
    model.compile(optimizer=optimizer, loss='mean_squared_error')
    
    start_mem = memory_usage(-1, interval=1, timeout=1)
    start_time = time.time()
    
    history = model.fit(X, y, batch_size=batch_size, epochs=epochs, verbose=0)
    
    end_time = time.time()
    end_mem = memory_usage(-1, interval=1, timeout=1)
    
    return {
        'memory_used': max(end_mem) - max(start_mem),
        'time_taken': end_time - start_time,
        'weights': model.get_weights()[0].flatten(),
        'loss': history.history['loss'][-1]
    }


def train_pytorch_model(X: np.ndarray, y: np.ndarray, optimizer: optim.Optimizer, 
                       batch_size: int, epochs: int = 1000) -> Dict[str, Any]:
    model = LinearRegression(X.shape[1])
    criterion = nn.MSELoss()
    
    X_tensor = torch.FloatTensor(X)
    y_tensor = torch.FloatTensor(y).reshape(-1, 1)
    
    start_mem = memory_usage(-1, interval=1, timeout=1)
    start_time = time.time()
    
    for epoch in range(epochs):
        for i in range(0, len(X), batch_size):
            batch_X = X_tensor[i:i+batch_size]
            batch_y = y_tensor[i:i+batch_size]
            
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
    
    end_time = time.time()
    end_mem = memory_usage(-1, interval=1, timeout=1)
    
    return {
        'memory_used': max(end_mem) - max(start_mem),
        'time_taken': end_time - start_time,
        'weights': model.linear.weight.data.numpy().flatten(),
        'loss': loss.item()
    }


def objective(trial: optuna.Trial) -> float:
    X, y = generate_data()
    batch_size = trial.suggest_int('batch_size', 1, len(X))
    learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-1, log=True)
    
    model = LinearRegression(X.shape[1])
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    results = train_pytorch_model(X, y, optimizer, batch_size)
    return results['loss']


def plot_results(df: pd.DataFrame):
    plt.figure(figsize=(15, 10))
    
    # Plot memory usage
    plt.subplot(2, 2, 1)
    sns.barplot(data=df, x='Method', y='Memory Used')
    plt.xticks(rotation=45)
    plt.title('Memory Usage by Method')
    
    # Plot time taken
    plt.subplot(2, 2, 2)
    sns.barplot(data=df, x='Method', y='Time Taken')
    plt.xticks(rotation=45)
    plt.title('Time Taken by Method')
    
    # Plot loss
    plt.subplot(2, 2, 3)
    sns.barplot(data=df, x='Method', y='Loss')
    plt.xticks(rotation=45)
    plt.title('Final Loss by Method')
    
    plt.tight_layout()
    plt.savefig('optimization_results.png')
    plt.close()


def main():
    # Generate data
    X, y = generate_data(n_samples=1000, n_features=10)
    
    # Initialize results DataFrame
    results = []
    
    # Test different batch sizes
    batch_sizes = [1, 32, 64, 128, 256, 512, 1000]
    for batch_size in batch_sizes:
        # TensorFlow
        for optimizer in get_tensorflow_optimizers():
            results.append({
                'Method': f'TF_{optimizer.__class__.__name__}_batch_{batch_size}',
                **train_tensorflow_model(X, y, optimizer, batch_size)
            })
        
        # PyTorch
        model = LinearRegression(X.shape[1])
        for optimizer in get_pytorch_optimizers(model):
            results.append({
                'Method': f'PT_{optimizer.__class__.__name__}_batch_{batch_size}',
                **train_pytorch_model(X, y, optimizer, batch_size)
            })
    
    # Convert results to DataFrame
    df = pd.DataFrame(results)
    
    # Plot results
    plot_results(df)
    
    # Print results
    print("\nResults Summary:")
    print(df.to_string())
    
    # Run Optuna optimization
    study = optuna.create_study()
    study.optimize(objective, n_trials=50)
    
    print("\nOptuna Optimization Results:")
    print(f"Best batch size: {study.best_params['batch_size']}")
    print(f"Best learning rate: {study.best_params['learning_rate']}")
    print(f"Best loss: {study.best_value}")


if __name__ == '__main__':
    main()