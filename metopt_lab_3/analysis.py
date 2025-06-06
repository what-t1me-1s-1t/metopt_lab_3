import numpy as np
import os
import matplotlib.pyplot as plt
from keras.optimizers import RMSprop, SGD as KerasSGD
from keras.models import Sequential
from keras.layers import Dense
import torch.nn as nn
import torch.optim as optim
import torch
from keras.src.optimizers import SGD
from sklearn.datasets import make_regression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sgd_regression import CustomSGDRegressor
from sklearn.metrics import r2_score, mean_squared_error as mse
import pandas as pd

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Установка seed для воспроизводимости результатов
np.random.seed(42)
torch.manual_seed(42)

# Генерация синтетических данных
print("Генерация данных...")
X, y = make_regression(n_samples=1000, n_features=5, noise=0.1, random_state=42)
if y.ndim == 1:
    y = y.reshape(-1, 1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Стандартизация признаков
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Преобразование данных для PyTorch
X_train_torch = torch.tensor(X_train_scaled, dtype=torch.float32)
y_train_torch = torch.tensor(y_train, dtype=torch.float32)
X_test_torch = torch.tensor(X_test_scaled, dtype=torch.float32)
y_test_torch = torch.tensor(y_test, dtype=torch.float32)

# --- Вспомогательная функция для построения графиков сходимости ---
def plot_convergence(history_data, title, filename):
    plt.figure(figsize=(10, 6))
    for label, losses in history_data.items():
        plt.plot(losses, label=label)
    plt.title(title)
    plt.xlabel('Эпоха')
    plt.ylabel('Функция потерь (MSE)')
    plt.legend()
    plt.grid(True)
    plt.yscale('log')
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()


def experiment_optimizers():
    print("\nСравнение оптимизаторов (SGD vs AdaGrad)")
    optimizers = ['sgd', 'adagrad']
    learning_rates = [0.001, 0.01, 0.1]
    results = []
    convergence_data = {}

    common_max_epochs = 100
    common_batch_size = 32
    common_tol = 1e-3

    for opt_name in optimizers:
        print(f"\n{opt_name.upper()}:\n")
        for lr in learning_rates:
            adjusted_lr = lr * 1000 if opt_name == 'adagrad' else lr
            tol = 0 if opt_name == 'adagrad' else common_tol
            print(f"Learning rate = {lr}:")
            model = CustomSGDRegressor(
                learning_rate=adjusted_lr,
                batch_size=common_batch_size,
                max_epochs=common_max_epochs,
                optimizer=opt_name,
                tol=tol,
                regularization=None,
                random_state=42
            )
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)
            r2 = r2_score(y_test, y_pred)
            mse_val = mse(y_test, y_pred)

            results.append({
                'Optimizer': opt_name,
                'Learning Rate': lr,
                'R2 Score': r2,
                'MSE': mse_val,
                'Training Time (s)': model.training_time,
                'Avg Memory (MB)': np.mean(model.memory_usage) if model.memory_usage else 0,
                'Operations': model.operation_count,
                'Epochs': len(model.loss_history)
            })
            convergence_data[f'{opt_name} (lr={lr})'] = model.loss_history

            # Вывод в требуемом формате
            print(f"  Время обучения: {model.training_time:.3f} сек")
            print(f"  Использование памяти: {np.mean(model.memory_usage) if model.memory_usage else 0:.2f} МБ")
            print(f"  Количество операций: {model.operation_count}")
            print(f"  Точность (R2): {r2:.4f}")
            print(f"  Финальная ошибка: {mse_val:.6f}\n")

    df_results = pd.DataFrame(results)
    print("\nРезультаты эксперимента:")
    print(df_results)
    df_results.to_csv('optimizer_results.csv', index=False)

    plot_convergence(convergence_data, 'Сходимость разных оптимизаторов', 'optimizer_convergence.png')

    return df_results


def experiment_libraries():
    """
    Сравнение эффективности библиотечных реализаций
    """
    learning_rates = [0.001, 0.01, 0.1]
    results = []

    common_max_epochs = 50
    common_batch_size = 32

    # Библиотечные реализации
    for lr in learning_rates:
        print(f"\nLearning rate = {lr}:")

        # MOMENTUM + NESTEROV
        X = np.random.rand(1000, 10)
        y = np.random.rand(1000, 1)

        # Создаём модель
        modelMN = Sequential([
            Dense(64, activation='relu', input_shape=(10,)),
            Dense(1)
        ])

        # Настраиваем оптимизатор SGD с Momentum и Nesterov
        optimizer = SGD(
            learning_rate=lr,
            momentum=0.9,
            nesterov=True,
            clipnorm=1.0
        )

        # Компилируем модель
        modelMN.compile(
            optimizer=optimizer,
            loss='mse',
            metrics=['mae']
        )

        # Обучаем модель
        history = modelMN.fit(
            X, y,
            epochs=common_max_epochs,  # Количество эпох
            batch_size=common_batch_size,  # Размер батча
            validation_split=0.2,
            verbose=0
        )

        print(f"keras (MOMENTUM + NESTEROV) Loss: {history.history['loss'][-1]:.4f}")

        # Создаём модель
        modelMN = Sequential([
            Dense(64, activation='relu', input_shape=(10,)),
            Dense(1)
        ])

        # Наcтраиваем оптимизатор RMSProp
        optimizer = RMSprop(
            learning_rate=lr,
            rho=0.9,
            momentum=0.0,
            epsilon=1e-7
        )

        # Компилируем модель
        modelMN.compile(
            optimizer=optimizer,
            loss='mse',
            metrics=['mae']
        )

        # Обучаем модель
        history = modelMN.fit(
            X, y,
            epochs=common_max_epochs,
            batch_size=common_batch_size,
            verbose=0
        )

        # Проверяем результат
        print(f"keras (RMSPROP) Loss: {history.history['loss'][-1]:.4f}")

        # Создаём синтетические данные
        X = torch.randn(1000, 10)
        y = torch.randn(1000, 1)

        # Инициализируем модель и оптимизатор
        model = nn.Linear(10, 1)
        optimizer = optim.Adagrad(model.parameters(), lr=lr)
        criterion = nn.MSELoss()

        resLoss = 0
        for epoch in range(50):
            optimizer.zero_grad()
            outputs = model(X)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()
            resLoss = loss.item()

        print(f"torch (Adagrad) Loss: {resLoss:.4f}")

        # torch.optim Adam

        # Данные
        X, y = torch.randn(100, 5).float(), torch.randn(100, 1).float()

        # Модель
        model = torch.nn.Sequential(torch.nn.Linear(5, 10), torch.nn.ReLU(), torch.nn.Linear(10, 1))

        # Оптимизаторы
        opt = torch.optim.Adam(model.parameters(), lr=lr)

        resLoss = 0

        # Обучение
        for epoch in range(50):
            opt.zero_grad()
            loss = torch.nn.functional.mse_loss(model(X), y)
            loss.backward()
            if loss is not None:
                resLoss = loss
            opt.step()

        print(f"torch (Adam) Loss: {resLoss:.4f}")

    return results


def experiment_regularization():
    """
    Эксперимент с разными типами регуляризации
    """
    regularizations = [None, 'l1', 'l2', 'elasticnet']
    results = []
    convergence_data = {}

    common_batch_size = 32
    common_max_epochs = 100
    common_lr = 0.01
    common_reg_strength = 0.01

    for reg in regularizations:
        print(f"\nРегуляризация: {reg if reg else 'none'}")
        model = CustomSGDRegressor(
            learning_rate=common_lr,
            batch_size=common_batch_size,
            max_epochs=common_max_epochs,
            regularization=reg,
            reg_strength=common_reg_strength,
            random_state=42
        )
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        r2 = r2_score(y_test, y_pred)
        mse_val = mse(y_test, y_pred)

        results.append({
            'Regularization': reg if reg else 'none',
            'Batch Size': common_batch_size,
            'R2 Score': r2,
            'Training Time (s)': model.training_time,
            'Avg Memory (MB)': np.mean(model.memory_usage) if model.memory_usage else 0,
            'Operations': model.operation_count,
            'Epochs': len(model.loss_history)
        })
        convergence_data[f'Regularization: {reg if reg else "none"}'] = model.loss_history

        print(f"  Размер батча: {common_batch_size}")
        print(f"  Время обучения: {model.training_time:.3f} сек")
        print(f"  Среднее использование памяти: {np.mean(model.memory_usage) if model.memory_usage else 0:.2f} МБ")
        print(f"  Количество операций: {model.operation_count}")
        print(f"  Точность (R2): {r2:.4f}")
        print(f"  Финальная ошибка: {mse_val:.6f}\n")

    df_results = pd.DataFrame(results)
    print("\nРезультаты эксперимента с регуляризацией:")
    print(df_results)
    df_results.to_csv('regularization_results.csv', index=False)

    plot_convergence(convergence_data, 'Сходимость для разных типов регуляризации', 'regularization_comparison.png')

    return df_results

if __name__ == "__main__":
    print("\nЗапуск экспериментов...")
    optimizer_results = experiment_optimizers()
    print("\nТестирование разных типов регуляризации...")
    regularization_results = experiment_regularization()
    library_results_df = experiment_libraries()
