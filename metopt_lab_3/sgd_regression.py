import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from typing import Callable, Tuple, Optional
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
import seaborn as sns
import time
import psutil
import os

class CustomSGDRegressor:
    """
    Реализация стохастического градиентного спуска (SGD) для задач регрессии.
    Поддерживает полиномиальную и линейную регрессию с различными конфигурациями:
    - Разные размеры батча (от одного образца до полного набора данных)
    - Различные схемы изменения скорости обучения
    - Различные методы регуляризации (L1, L2, Elastic Net)
    """
    def __init__(
        self,
        learning_rate: float = 0.01,
        batch_size: int = 1,
        max_epochs: int = 1000,
        tol: float = 1e-4,
        regularization: str = None,
        reg_strength: float = 0.01,
        optimizer: str = 'sgd',
        lr_scheduler: str = 'constant',
        polynomial_degree: int = 1,
        random_state: int = 42,
        epsilon: float = 1e-8
    ):
        """
        Инициализация регрессора SGD.
        
        Параметры:
        - learning_rate: начальная скорость обучения
        - batch_size: размер мини-батча
        - max_epochs: максимальное количество эпох
        - tol: порог сходимости для ранней остановки
        - regularization: тип регуляризации (None, 'l1', 'l2', 'elastic')
        - reg_strength: сила регуляризации
        - optimizer: тип оптимизатора ('sgd' или 'adagrad')
        - lr_scheduler: схема изменения скорости обучения
        - polynomial_degree: степень полинома для полиномиальной регрессии
        - random_state: seed для воспроизводимости результатов
        - epsilon: малое число для численной стабильности AdaGrad
        """
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.tol = tol
        self.regularization = regularization
        self.reg_strength = reg_strength
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.polynomial_degree = polynomial_degree
        self.random_state = random_state
        self.epsilon = epsilon
        self.weights = None
        self.bias = None
        self.accumulated_grads_w = None
        self.accumulated_grads_b = 0
        self.loss_history = []
        self.training_time = 0
        self.memory_usage = []
        self.operation_count = 0
        self.poly = PolynomialFeatures(degree=polynomial_degree, include_bias=False)
        
    def _get_lr(self, epoch: int) -> float:
        """
        Вычисление текущей скорости обучения в зависимости от выбранной схемы.
        """
        initial_lr = self.learning_rate
        if self.lr_scheduler == 'constant':
            return initial_lr
        elif self.lr_scheduler == 'time_decay':
            return initial_lr / (1 + 0.01 * epoch)
        elif self.lr_scheduler == 'step_decay':
            drop_rate = 0.5
            epochs_drop = 10.0
            return initial_lr * np.power(drop_rate, np.floor((1 + epoch) / epochs_drop))
        elif self.lr_scheduler == 'exponential_decay':
            decay_rate = 0.95
            return initial_lr * np.power(decay_rate, epoch)
        return initial_lr

    def _compute_regularization(self) -> float:
        """
        Вычисление значения регуляризации в зависимости от выбранного типа.
        """
        if self.regularization is None:
            return 0
        
        if self.regularization == 'l1':
            return self.reg_strength * np.sum(np.abs(self.weights))
        elif self.regularization == 'l2':
            return self.reg_strength * np.sum(self.weights ** 2)
        elif self.regularization == 'elastic':
            l1_term = self.reg_strength * np.sum(np.abs(self.weights))
            l2_term = self.reg_strength * np.sum(self.weights ** 2)
            return 0.5 * (l1_term + l2_term)
        return 0

    def _compute_regularization_gradient(self) -> np.ndarray:
        """
        Вычисление градиента регуляризации в зависимости от выбранного типа.
        """
        if self.regularization is None:
            return 0
        
        if self.regularization == 'l1':
            return self.reg_strength * np.sign(self.weights)
        elif self.regularization == 'l2':
            return 2 * self.reg_strength * self.weights
        elif self.regularization == 'elastic':
            return self.reg_strength * (np.sign(self.weights) + self.weights)
        return 0

    def _get_memory_usage(self) -> float:
        """
        Получение текущего использования памяти процессом.
        """
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / 1024 / 1024  # В мегабайтах

    def _initialize_adagrad(self, n_features: int):
        """
        Инициализация аккумуляторов градиентов для AdaGrad
        """
        self.accumulated_grads_w = np.zeros(n_features)
        self.accumulated_grads_b = 0

    def _update_parameters(self, grad_w: np.ndarray, grad_b: float, current_lr: float):
        """
        Обновление параметров модели в зависимости от выбранного оптимизатора
        """
        if self.optimizer == 'sgd':
            # Стандартный SGD
            self.weights -= current_lr * grad_w
            self.bias -= current_lr * grad_b
        elif self.optimizer == 'adagrad':
            # AdaGrad
            # Накапливаем квадраты градиентов
            self.accumulated_grads_w += np.square(grad_w)
            self.accumulated_grads_b += np.square(grad_b)
            
            # Вычисляем адаптивные скорости обучения
            adapted_lr_w = current_lr / (np.sqrt(self.accumulated_grads_w + self.epsilon))
            adapted_lr_b = current_lr / (np.sqrt(self.accumulated_grads_b + self.epsilon))
            
            # Обновляем параметры
            self.weights -= adapted_lr_w * grad_w
            self.bias -= adapted_lr_b * grad_b
            
            self.operation_count += 4 * len(grad_w) + 2  # Дополнительные операции AdaGrad

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'CustomSGDRegressor':
        """
        Обучение модели на данных.
        
        Параметры:
        - X: матрица признаков
        - y: целевая переменная
        """
        start_time = time.time()
        np.random.seed(self.random_state)
        
        # Преобразование признаков для полиномиальной регрессии
        X_transformed = self.poly.fit_transform(X)
        n_samples, n_features = X_transformed.shape
        
        # Инициализация весов и смещения
        self.weights = np.zeros(n_features)
        self.bias = 0
        
        # Инициализация AdaGrad если выбран
        if self.optimizer == 'adagrad':
            self._initialize_adagrad(n_features)
        
        # Установка размера батча
        self.batch_size = min(self.batch_size, n_samples)
        
        prev_loss = float('inf')
        for epoch in range(self.max_epochs):
            # Перемешивание данных
            indices = np.random.permutation(n_samples)
            X_shuffled = X_transformed[indices]
            y_shuffled = y[indices]
            
            # Обработка мини-батчей
            for i in range(0, n_samples, self.batch_size):
                X_batch = X_shuffled[i:i + self.batch_size]
                y_batch = y_shuffled[i:i + self.batch_size]
                
                # Вычисление предсказаний
                y_pred = np.dot(X_batch, self.weights) + self.bias
                self.operation_count += X_batch.shape[0] * X_batch.shape[1]
                
                # Вычисление градиентов
                current_lr = self._get_lr(epoch)
                grad_w = -(2/len(X_batch)) * np.dot(X_batch.T, (y_batch - y_pred))
                grad_b = -(2/len(X_batch)) * np.sum(y_batch - y_pred)
                self.operation_count += 2 * X_batch.shape[0] * X_batch.shape[1]
                
                # Добавление градиента регуляризации
                grad_w += self._compute_regularization_gradient()
                
                # Обновление параметров с помощью выбранного оптимизатора
                self._update_parameters(grad_w, grad_b, current_lr)
                
                # Отслеживание использования памяти
                self.memory_usage.append(self._get_memory_usage())
            
            # Вычисление функции потерь на всем наборе данных
            y_pred_all = np.dot(X_transformed, self.weights) + self.bias
            current_loss = mean_squared_error(y, y_pred_all) + self._compute_regularization()
            self.loss_history.append(current_loss)
            
            # Ранняя остановка
            if abs(prev_loss - current_loss) < self.tol:
                break
            prev_loss = current_loss
        
        self.training_time = time.time() - start_time
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Предсказание значений для новых данных.
        """
        X_transformed = self.poly.transform(X)
        return np.dot(X_transformed, self.weights) + self.bias 