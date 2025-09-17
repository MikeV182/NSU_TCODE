import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import seaborn as sns

def generate_synthetic_data(n_samples=1000, n_features=1, n_informative=1, 
                           n_redundant=0, n_clusters_per_class=1, random_state=42):
    """
    Генерирует синтетические данные с помощью make_classification.

    Args:
        n_samples (int): Количество образцов.
        n_features (int): Количество признаков.
        n_informative (int): Количество информативных признаков.
        n_redundant (int): Количество избыточных признаков.
        n_clusters_per_class (int): Количество кластеров на класс.
        random_state (int): Сид для воспроизводимости.

    Returns:
        np.ndarray: Сгенерированные данные.
    """
    try:
        if n_samples <= 0 or n_features < 1 or n_informative < 1 or n_redundant < 0 or n_clusters_per_class < 1:
            raise ValueError("Некорректные параметры: проверьте n_samples, n_features, n_informative, n_redundant, n_clusters_per_class")
        if n_informative > n_features:
            raise ValueError("n_informative не может быть больше n_features")
        if n_clusters_per_class * 2 > 2 ** n_informative:
            raise ValueError(f"n_classes * n_clusters_per_class должно быть <= 2**n_informative")

        X, _ = make_classification(
            n_samples=n_samples,
            n_features=n_features,
            n_informative=n_informative,
            n_redundant=n_redundant,
            n_clusters_per_class=n_clusters_per_class,
            random_state=random_state
        )
        return X
    except ValueError as ve:
        raise ValueError(f"Ошибка генерации данных: {ve}")
    except Exception as e:
        raise Exception(f"Произошла ошибка: {e}")

def scale_data(X):
    """
    Применяет MinMaxScaler и StandardScaler к данным.

    Args:
        X (np.ndarray): Входные данные.

    Returns:
        tuple: (X_minmax, X_standard) - масштабированные данные.
    """
    try:
        if not isinstance(X, np.ndarray):
            raise ValueError("Входные данные должны быть numpy массивом")
        if X.size == 0:
            raise ValueError("Входные данные пусты")

        minmax_scaler = MinMaxScaler()
        standard_scaler = StandardScaler()
        X_minmax = minmax_scaler.fit_transform(X)
        X_standard = standard_scaler.fit_transform(X)
        return X_minmax, X_standard
    except ValueError as ve:
        raise ValueError(f"Ошибка масштабирования: {ve}")
    except Exception as e:
        raise Exception(f"Произошла ошибка: {e}")

def check_standard_metrics(X_standard):
    """
    Проверяет метрики StandardScaler (среднее и стандартное отклонение).

    Args:
        X_standard (np.ndarray): Данные после StandardScaler.

    Returns:
        tuple: (mean, std) - среднее и стандартное отклонение.
    """
    try:
        mean = np.mean(X_standard)
        std = np.std(X_standard)
        return mean, std
    except Exception as e:
        raise Exception(f"Ошибка вычисления метрик: {e}")

def visualize_distributions(X, X_minmax, X_standard, figsize=(15, 5)):
    """
    Визуализирует распределения исходных и масштабированных данных.

    Args:
        X (np.ndarray): Исходные данные.
        X_minmax (np.ndarray): Данные после MinMaxScaler.
        X_standard (np.ndarray): Данные после StandardScaler.
        figsize (tuple): Размер фигуры.

    Returns:
        None: Отображает графики.
    """
    try:
        plt.figure(figsize=figsize)
        
        plt.subplot(1, 3, 1)
        sns.histplot(X[:, 0], kde=True, color='blue')
        plt.title('Исходное распределение')
        plt.xlabel('Значения признака')
        plt.ylabel('Частота')

        plt.subplot(1, 3, 2)
        sns.histplot(X_minmax[:, 0], kde=True, color='green')
        plt.title('MinMaxScaler (0 to 1)')
        plt.xlabel('Значения признака')
        plt.ylabel('Частота')

        plt.subplot(1, 3, 3)
        sns.histplot(X_standard[:, 0], kde=True, color='red')
        plt.title('StandardScaler (mean≈0, std≈1)')
        plt.xlabel('Значения признака')
        plt.ylabel('Частота')

        plt.tight_layout()
        plt.show()
    except Exception as e:
        raise Exception(f"Ошибка визуализации: {e}")

def scale_and_visualize_data(data=None, n_samples=1000, n_features=1, n_informative=1, 
                            n_redundant=0, n_clusters_per_class=1, random_state=42, figsize=(15, 5)):
    """
    Основная функция: генерирует или принимает данные, масштабирует их и визуализирует.

    Args:
        data (np.ndarray, optional): Внешний датасет. Если None, генерируются синтетические данные.
        n_samples (int): Количество образцов (для синтетических данных).
        n_features (int): Количество признаков.
        n_informative (int): Количество информативных признаков.
        n_redundant (int): Количество избыточных признаков.
        n_clusters_per_class (int): Количество кластеров на класс.
        random_state (int): Сид для воспроизводимости.
        figsize (tuple): Размер фигуры для визуализации.

    Returns:
        None: Выводит метрики и отображает графики.
    """
    try:
        # Использовать внешний датасет или сгенерировать новый
        X = data if data is not None else generate_synthetic_data(
            n_samples, n_features, n_informative, n_redundant, n_clusters_per_class, random_state
        )

        # Масштабирование данных
        X_minmax, X_standard = scale_data(X)

        # Проверка метрик StandardScaler
        mean_standard, std_standard = check_standard_metrics(X_standard)
        print(f"StandardScaler: mean = {mean_standard:.4f}, std = {std_standard:.4f}")

        # Визуализация
        visualize_distributions(X, X_minmax, X_standard, figsize)

    except ValueError as ve:
        print(f"Ошибка в параметрах: {ve}")
    except Exception as e:
        print(f"Произошла ошибка: {e}")

if __name__ == "__main__":
    scale_and_visualize_data()
