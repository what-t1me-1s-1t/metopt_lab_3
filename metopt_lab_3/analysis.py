import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_regression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sgd_regression import CustomSGDRegressor
import seaborn as sns
from sklearn.metrics import r2_score
import pandas as pd

# Установка seed для воспроизводимости результатов
np.random.seed(42)

# Генерация синтетических данных
print("Генерация данных...")
X, y = make_regression(n_samples=1000, n_features=5, noise=0.1, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Стандартизация признаков
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

def experiment_batch_sizes():
    """
    Эксперимент с разными размерами батча.
    Сравнивает:
    - Точность (R2 score)
    - Скорость сходимости
    - Использование памяти
    - Количество операций
    """
    batch_sizes = [1, 8, 32, 128, len(X_train)]
    results = []
    
    # Графики для функции потерь
    plt.figure(figsize=(15, 10))
    plt.subplot(2, 2, 1)
    
    for batch_size in batch_sizes:
        model = CustomSGDRegressor(
            learning_rate=0.01,
            batch_size=batch_size,
            max_epochs=100,
            regularization='l2',
            reg_strength=0.01
        )
        
        # Обучение модели
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        r2 = r2_score(y_test, y_pred)
        
        results.append({
            'batch_size': batch_size,
            'r2_score': r2,
            'training_time': model.training_time,
            'avg_memory': np.mean(model.memory_usage),
            'operation_count': model.operation_count,
            'loss_history': model.loss_history
        })
        
        plt.plot(model.loss_history, label=f'Batch size: {batch_size}')
    
    plt.xlabel('Эпоха')
    plt.ylabel('MSE Loss')
    plt.title('Сходимость для разных размеров батча')
    plt.legend()
    plt.yscale('log')
    plt.grid(True)
    
    # График времени обучения
    plt.subplot(2, 2, 2)
    batch_sizes_str = [str(bs) for bs in batch_sizes]
    times = [r['training_time'] for r in results]
    plt.bar(batch_sizes_str, times)
    plt.xlabel('Размер батча')
    plt.ylabel('Время обучения (сек)')
    plt.title('Время обучения для разных размеров батча')
    
    # График использования памяти
    plt.subplot(2, 2, 3)
    memory_usage = [r['avg_memory'] for r in results]
    plt.bar(batch_sizes_str, memory_usage)
    plt.xlabel('Размер батча')
    plt.ylabel('Среднее использование памяти (МБ)')
    plt.title('Использование памяти')
    
    # График точности
    plt.subplot(2, 2, 4)
    r2_scores = [r['r2_score'] for r in results]
    plt.bar(batch_sizes_str, r2_scores)
    plt.xlabel('Размер батча')
    plt.ylabel('R2 Score')
    plt.title('Точность модели')
    
    plt.tight_layout()
    plt.savefig('batch_size_analysis.png')
    plt.close()
    
    # Сохранение результатов в CSV
    df = pd.DataFrame(results)
    df = df.drop('loss_history', axis=1)
    df.to_csv('batch_size_results.csv', index=False)
    
    return results

def experiment_learning_rates():
    """
    Эксперимент с разными схемами изменения скорости обучения
    """
    lr_schedulers = ['constant', 'time_decay', 'step_decay', 'exponential_decay']
    results = []
    
    plt.figure(figsize=(12, 6))
    for scheduler in lr_schedulers:
        model = CustomSGDRegressor(
            learning_rate=0.01,
            batch_size=32,
            max_epochs=100,
            lr_scheduler=scheduler,
            regularization='l2',
            reg_strength=0.01
        )
        model.fit(X_train_scaled, y_train)
        results.append({
            'scheduler': scheduler,
            'loss_history': model.loss_history
        })
        plt.plot(model.loss_history, label=f'Scheduler: {scheduler}')
    
    plt.xlabel('Эпоха')
    plt.ylabel('MSE Loss')
    plt.title('Сходимость для разных схем изменения скорости обучения')
    plt.legend()
    plt.yscale('log')
    plt.grid(True)
    plt.savefig('lr_scheduler_comparison.png')
    plt.close()

def experiment_regularization():
    """
    Эксперимент с разными типами регуляризации
    """
    regularizations = [None, 'l1', 'l2', 'elastic']
    results = []
    
    plt.figure(figsize=(12, 6))
    for reg in regularizations:
        model = CustomSGDRegressor(
            learning_rate=0.01,
            batch_size=32,
            max_epochs=100,
            regularization=reg,
            reg_strength=0.01
        )
        model.fit(X_train_scaled, y_train)
        results.append({
            'regularization': reg if reg else 'none',
            'loss_history': model.loss_history
        })
        plt.plot(model.loss_history, label=f'Regularization: {reg if reg else "none"}')
    
    plt.xlabel('Эпоха')
    plt.ylabel('MSE Loss')
    plt.title('Сходимость для разных типов регуляризации')
    plt.legend()
    plt.yscale('log')
    plt.grid(True)
    plt.savefig('regularization_comparison.png')
    plt.close()

def experiment_optimizers():
    """
    Сравнение эффективности разных оптимизаторов (SGD vs AdaGrad)
    """
    optimizers = ['sgd', 'adagrad']
    learning_rates = [0.001, 0.01, 0.1]  # Уменьшили learning rates для лучшей стабильности
    results = []
    
    # Создаем отдельные графики для лучшей читаемости
    plt.figure(figsize=(12, 6))
    
    # График функции потерь
    for lr in learning_rates:
        for optimizer in optimizers:
            model = CustomSGDRegressor(
                learning_rate=lr,
                batch_size=32,
                max_epochs=100,
                optimizer=optimizer,
                regularization='l2',
                reg_strength=0.01
            )
            
            # Обучение модели
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)
            r2 = r2_score(y_test, y_pred)
            
            results.append({
                'optimizer': optimizer,
                'learning_rate': lr,
                'r2_score': r2,
                'training_time': model.training_time,
                'avg_memory': np.mean(model.memory_usage),
                'operation_count': model.operation_count,
                'loss_history': model.loss_history,
                'final_loss': model.loss_history[-1]
            })
            
            # График функции потерь с улучшенным форматированием
            plt.plot(
                model.loss_history,
                label=f'{optimizer.upper()}, lr={lr}',
                alpha=0.8,
                linewidth=2
            )
    
    plt.xlabel('Эпоха', fontsize=12)
    plt.ylabel('MSE Loss', fontsize=12)
    plt.title('Сходимость разных оптимизаторов', fontsize=14, pad=20)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.yscale('log')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('optimizer_convergence.png', bbox_inches='tight', dpi=300)
    plt.close()
    
    # Преобразование результатов в DataFrame
    df = pd.DataFrame(results)
    
    # Создаем отдельный график для метрик производительности
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Сравнение производительности оптимизаторов', fontsize=16, y=1.02)
    
    # График времени обучения
    sns.barplot(
        data=df,
        x='learning_rate',
        y='training_time',
        hue='optimizer',
        ax=axes[0, 0]
    )
    axes[0, 0].set_title('Время обучения')
    axes[0, 0].set_ylabel('Время (сек)')
    axes[0, 0].set_xlabel('Learning Rate')
    
    # График использования памяти
    sns.barplot(
        data=df,
        x='learning_rate',
        y='avg_memory',
        hue='optimizer',
        ax=axes[0, 1]
    )
    axes[0, 1].set_title('Использование памяти')
    axes[0, 1].set_ylabel('Память (МБ)')
    axes[0, 1].set_xlabel('Learning Rate')
    
    # График точности (R2)
    sns.barplot(
        data=df,
        x='learning_rate',
        y='r2_score',
        hue='optimizer',
        ax=axes[1, 0]
    )
    axes[1, 0].set_title('Точность модели (R2)')
    axes[1, 0].set_ylabel('R2 Score')
    axes[1, 0].set_xlabel('Learning Rate')
    
    # График финальной ошибки
    sns.barplot(
        data=df,
        x='learning_rate',
        y='final_loss',
        hue='optimizer',
        ax=axes[1, 1]
    )
    axes[1, 1].set_title('Финальная ошибка (MSE)')
    axes[1, 1].set_ylabel('MSE Loss')
    axes[1, 1].set_xlabel('Learning Rate')
    axes[1, 1].set_yscale('log')  # Логарифмический масштаб для ошибки
    
    # Улучшаем внешний вид
    for ax in axes.flat:
        ax.grid(True, alpha=0.3)
        ax.set_title(ax.get_title(), pad=20)
    
    plt.tight_layout()
    plt.savefig('optimizer_metrics.png', bbox_inches='tight', dpi=300)
    plt.close()
    
    # Сохранение результатов в CSV
    df_save = df.drop('loss_history', axis=1)
    df_save.to_csv('optimizer_results.csv', index=False)
    
    # Вывод подробной сравнительной статистики
    print("\nСравнительная статистика оптимизаторов:")
    for optimizer in optimizers:
        print(f"\n{optimizer.upper()}:")
        opt_data = df[df['optimizer'] == optimizer]
        
        # Группировка по learning rate
        for lr in learning_rates:
            lr_data = opt_data[opt_data['learning_rate'] == lr]
            print(f"\nLearning rate = {lr}:")
            print(f"  Время обучения: {lr_data['training_time'].mean():.3f} сек")
            print(f"  Использование памяти: {lr_data['avg_memory'].mean():.2f} МБ")
            print(f"  Количество операций: {lr_data['operation_count'].mean():.0f}")
            print(f"  Точность (R2): {lr_data['r2_score'].mean():.4f}")
            print(f"  Финальная ошибка: {lr_data['final_loss'].mean():.6f}")
    
    return results

if __name__ == "__main__":
    print("Запуск экспериментов...")
    
    print("\nТестирование разных размеров батча...")
    batch_results = experiment_batch_sizes()
    
    print("\nТестирование разных схем изменения скорости обучения...")
    experiment_learning_rates()
    
    print("\nТестирование разных типов регуляризации...")
    experiment_regularization()
    
    print("\nСравнение оптимизаторов (SGD vs AdaGrad)...")
    optimizer_results = experiment_optimizers()
    
    print("\nЭксперименты завершены! Проверьте сгенерированные графики и CSV файлы для подробного анализа.") 