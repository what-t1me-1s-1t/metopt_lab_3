import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error
import time
import psutil
import os


class CustomSGDRegressor:
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
            epsilon: float = 1e-1
    ):
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
        self.bias = 0
        self.loss_history = []
        self.training_time = 0
        self.memory_usage = []
        self.operation_count = 0

        self.poly = PolynomialFeatures(degree=self.polynomial_degree, include_bias=False)
        self.accumulated_grads_w = None
        self.accumulated_grads_b = 0

    def _initialize_adagrad(self, n_features: int):
        self.accumulated_grads_w = np.zeros((n_features, 1))
        self.accumulated_grads_b = 0

    def _get_lr(self, epoch: int) -> float:
        if self.lr_scheduler == 'constant':
            return self.learning_rate
        elif self.lr_scheduler == 'time_decay':
            return self.learning_rate / (1 + epoch)
        elif self.lr_scheduler == 'step_decay':
            drop = 0.5
            epochs_drop = 10
            return self.learning_rate * (drop ** np.floor((1 + epoch) / epochs_drop))
        elif self.lr_scheduler == 'exponential_decay':
            k = 0.1
            return self.learning_rate * np.exp(-k * epoch)
        else:
            raise ValueError(f"Неизвестная схема изменения скорости обучения: {self.lr_scheduler}")

    def _compute_regularization_loss(self) -> float:
        if self.regularization == 'l1':
            return self.reg_strength * np.sum(np.abs(self.weights))
        elif self.regularization == 'l2':
            return self.reg_strength * np.sum(self.weights ** 2)
        elif self.regularization == 'elasticnet':
            l1_ratio = 0.5
            l1_loss = l1_ratio * self.reg_strength * np.sum(np.abs(self.weights))
            l2_loss = (1 - l1_ratio) * self.reg_strength * np.sum(self.weights ** 2)
            return l1_loss + l2_loss
        return 0.0

    def _compute_regularization_gradient(self) -> np.ndarray:
        if self.regularization == 'l1':
            return self.reg_strength * np.sign(self.weights)
        elif self.regularization == 'l2':
            return 2 * self.reg_strength * self.weights
        elif self.regularization == 'elasticnet':
            l1_ratio = 0.5
            l1_grad = l1_ratio * self.reg_strength * np.sign(self.weights)
            l2_grad = (1 - l1_ratio) * 2 * self.reg_strength * self.weights
            return l1_grad + l2_grad
        return np.zeros_like(self.weights)

    def _update_parameters(self, grad_w: np.ndarray, grad_b: float, current_lr: float):
        if self.optimizer == 'sgd':
            self.weights -= current_lr * grad_w
            self.bias -= current_lr * grad_b
        elif self.optimizer == 'adagrad':
            # Проверка формы градиентов
            grad_w = grad_w.reshape(-1, 1)
            self.accumulated_grads_w += grad_w ** 2
            self.accumulated_grads_b += grad_b ** 2
            # Адаптивное обновление с учетом epsilon
            adaptive_lr_w = current_lr / (np.sqrt(self.accumulated_grads_w + self.epsilon))
            adaptive_lr_b = current_lr / (np.sqrt(self.accumulated_grads_b + self.epsilon))
            self.weights -= adaptive_lr_w * grad_w
            self.bias -= adaptive_lr_b * grad_b
        else:
            raise ValueError(f"Неизвестный оптимизатор: {self.optimizer}")

    def _get_memory_usage(self) -> float:
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / (1024 * 1024)

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'CustomSGDRegressor':
        start_time = time.time()
        np.random.seed(self.random_state)

        X_transformed = self.poly.fit_transform(X)
        n_samples, n_features = X_transformed.shape

        if y.ndim == 1:
            y = y.reshape(-1, 1)

        self.weights = np.zeros((n_features, 1))
        self.bias = 0

        if self.optimizer == 'adagrad':
            self._initialize_adagrad(n_features)

        self.loss_history = []
        self.memory_usage = []
        self.operation_count = 0

        prev_loss = float('inf')

        for epoch in range(self.max_epochs):
            indices = np.arange(n_samples)
            np.random.shuffle(indices)
            X_shuffled = X_transformed[indices]
            y_shuffled = y[indices]

            for i in range(0, n_samples, self.batch_size):
                X_batch = X_shuffled[i:i + self.batch_size]
                y_batch = y_shuffled[i:i + self.batch_size]

                if len(X_batch) == 0:
                    continue

                y_pred = np.dot(X_batch, self.weights) + self.bias
                self.operation_count += X_batch.shape[0] * X_batch.shape[1]
                self.operation_count += X_batch.shape[0]

                current_lr = self._get_lr(epoch)
                grad_w = -(2 / len(X_batch)) * np.dot(X_batch.T, (y_batch - y_pred))
                grad_b = -(2 / len(X_batch)) * np.sum(y_batch - y_pred)
                self.operation_count += 2 * X_batch.shape[0] * X_batch.shape[1]

                grad_w += self._compute_regularization_gradient()
                self._update_parameters(grad_w, grad_b, current_lr)
                self.memory_usage.append(self._get_memory_usage())

            y_pred_all = np.dot(X_transformed, self.weights) + self.bias
            current_loss = mean_squared_error(y, y_pred_all)
            current_loss += self._compute_regularization_loss()

            self.loss_history.append(current_loss)

            if abs(prev_loss - current_loss) < self.tol:
                break
            prev_loss = current_loss

        self.training_time = time.time() - start_time
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        X_transformed = self.poly.transform(X)
        predictions = np.dot(X_transformed, self.weights) + self.bias
        return predictions.flatten()
