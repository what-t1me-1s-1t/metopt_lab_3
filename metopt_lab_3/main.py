import numpy as np
from sklearn.utils import shuffle  # type: ignore


class LinearRegressionSGD:
    def __init__(self):
        self.weights = None
        self.bias = 0.0
        self.loss_history = []

    @staticmethod
    def compute_gradient(X_batch, y_batch, weights, bias, reg_type, reg_param):
        m = X_batch.shape[0]
        y_pred = np.dot(X_batch, weights) + bias
        error = y_pred - y_batch

        grad_weights = (2.0 / m) * np.dot(X_batch.T, error)
        grad_bias = (2.0 / m) * np.sum(error)

        if reg_type == 'l2':
            grad_weights += reg_param * weights
        elif reg_type == 'l1':
            grad_weights += reg_param * np.sign(weights)
        elif reg_type == 'elastic':
            grad_weights += reg_param[0] * np.sign(weights) + reg_param[1] * weights

        return grad_weights, grad_bias

    @staticmethod
    def compute_loss(X, y, weights, bias, reg_type, reg_param):
        m = X.shape[0]
        y_pred = np.dot(X, weights) + bias
        mse = np.mean((y_pred - y) ** 2)

        if reg_type == 'l2':
            reg = reg_param * np.sum(weights ** 2) / (2 * m)
        elif reg_type == 'l1':
            reg = reg_param * np.sum(np.abs(weights)) / m
        elif reg_type == 'elastic':
            reg = (reg_param[0] * np.sum(np.abs(weights)) + reg_param[1] * np.sum(weights ** 2)) / m
        else:
            reg = 0.0

        return mse + reg

    @staticmethod
    def get_learning_rate(initial_lr, schedule_type, epoch, **kwargs):
        if schedule_type == 'constant':
            return initial_lr
        elif schedule_type == 'time':
            decay_rate = kwargs.get('decay_rate', 0.1)
            return initial_lr / (1 + decay_rate * epoch)
        elif schedule_type == 'step':
            step_size = kwargs.get('step_size', 10)
            decay_rate = kwargs.get('decay_rate', 0.5)
            return initial_lr * (decay_rate ** (epoch // step_size))
        elif schedule_type == 'exponential':
            decay_rate = kwargs.get('decay_rate', 0.01)
            return initial_lr * np.exp(-decay_rate * epoch)
        else:
            return initial_lr

    def fit(self, X, y, batch_size=32, epochs=100, learning_rate=0.01,
            schedule_type='constant', reg_type=None, reg_param=0.0, **kwargs):

        if self.weights is None:
            self.weights = np.zeros(X.shape[1])

        n_samples = X.shape[0]
        self.loss_history = []

        for epoch in range(epochs):
            X_shuffled, y_shuffled = shuffle(X, y)
            current_lr = self.get_learning_rate(learning_rate, schedule_type, epoch, **kwargs)

            for i in range(0, n_samples, batch_size):
                X_batch = X_shuffled[i:i + batch_size]
                y_batch = y_shuffled[i:i + batch_size]

                grad_weights, grad_bias = self.compute_gradient(
                    X_batch, y_batch, self.weights, self.bias, reg_type, reg_param
                )

                self.weights -= current_lr * grad_weights
                self.bias -= current_lr * grad_bias

            epoch_loss = self.compute_loss(X, y, self.weights, self.bias, reg_type, reg_param)
            self.loss_history.append(epoch_loss)

        return self

    def predict(self, X):
        return np.dot(X, self.weights) + self.bias


# Пример использования
if __name__ == "__main__":
    from sklearn.datasets import make_regression
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import mean_squared_error
    import matplotlib.pyplot as plt

    # Генерация данных
    X, y = make_regression(n_samples=1000, n_features=20, noise=0.1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    # Эксперимент с разным размером батча
    batch_sizes = [1, 32, X_train.shape[0]]
    results = {}

    for bs in batch_sizes:
        model = LinearRegressionSGD()
        model.fit(X_train, y_train, batch_size=bs, epochs=100, learning_rate=0.01)
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        results[f'Batch Size {bs}'] = {
            'mse': mse,
            'loss_history': model.loss_history.copy()
        }

    # Визуализация сходимости
    plt.figure(figsize=(10, 6))
    for label, data in results.items():
        plt.plot(data['loss_history'], label=label)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Сравнение сходимости для разных размеров батча')
    plt.legend()
    plt.show()

    # Эксперимент с разными learning rate schedules
    schedules = [
        ('constant', {}),
        ('time', {'decay_rate': 0.1}),
        ('step', {'step_size': 30, 'decay_rate': 0.5}),
        ('exponential', {'decay_rate': 0.01})
    ]

    plt.figure(figsize=(10, 6))
    for schedule_type, params in schedules:
        model = LinearRegressionSGD()
        model.fit(X_train, y_train, batch_size=32, epochs=100,
                  learning_rate=0.1, schedule_type=schedule_type, **params)
        plt.plot(model.loss_history, label=f'{schedule_type} schedule')

    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Сравнение различных стратегий learning rate')
    plt.legend()
    plt.show()

    # Проверка регуляризации
    reg_types = [None, 'l1', 'l2', 'elastic']
    reg_params = {
        'l1': 0.1,
        'l2': 0.1,
        'elastic': (0.05, 0.05)
    }

    plt.figure(figsize=(10, 6))
    for reg_type in reg_types:
        model = LinearRegressionSGD()
        model.fit(X_train, y_train, batch_size=32, epochs=100,
                  learning_rate=0.01, reg_type=reg_type,
                  reg_param=reg_params.get(reg_type, 0.0))
        plt.plot(model.loss_history, label=f'Reg: {reg_type or "None"}')

    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Влияние регуляризации на сходимость')
    plt.legend()
    plt.show()