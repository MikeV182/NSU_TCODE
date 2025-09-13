import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import seaborn as sns

def scale_and_visualize_data(n_samples=1000, n_features=1, n_informative=1, n_redundant=0, 
                            n_clusters_per_class=1, random_state=42, figsize=(15, 5)):
    """
    Генерирует синтетические данные, применяет MinMaxScaler и StandardScaler, 
    выводит метрики и визуализирует распределения.

    Args:
        n_samples (int): Количество образцов в данных.
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
        # Проверка входных параметров
        if n_samples <= 0 or n_features < 1 or n_informative < 1 or n_redundant < 0 or n_clusters_per_class < 1:
            raise ValueError("Некорректные параметры: проверьте n_samples, n_features, n_informative, n_redundant, n_clusters_per_class")
        if n_informative > n_features:
            raise ValueError("n_informative не может быть больше n_features")
        if n_clusters_per_class * 2 > 2 ** n_informative:  # Проверка ограничения make_classification
            raise ValueError(f"n_classes * n_clusters_per_class должно быть <= 2**n_informative")

        # Генерация синтетических данных
        X, _ = make_classification(
            n_samples=n_samples,
            n_features=n_features,
            n_informative=n_informative,
            n_redundant=n_redundant,
            n_clusters_per_class=n_clusters_per_class,
            random_state=random_state
        )

        # Инициализация скейлеров
        minmax_scaler = MinMaxScaler()
        standard_scaler = StandardScaler()

        # Применение скейлеров
        X_minmax = minmax_scaler.fit_transform(X)
        X_standard = standard_scaler.fit_transform(X)

        # Проверка метрик для StandardScaler
        mean_standard = np.mean(X_standard)
        std_standard = np.std(X_standard)
        print(f"StandardScaler: mean = {mean_standard:.4f}, std = {std_standard:.4f}")

        # Визуализация распределений
        plt.figure(figsize=figsize)

        # Исходные данные
        plt.subplot(1, 3, 1)
        sns.histplot(X[:, 0], kde=True, color='blue')
        plt.title('Исходное распределение')
        plt.xlabel('Значения признака')
        plt.ylabel('Частота')

        # MinMaxScaler
        plt.subplot(1, 3, 2)
        sns.histplot(X_minmax[:, 0], kde=True, color='green')
        plt.title('MinMaxScaler (0 to 1)')
        plt.xlabel('Значения признака')
        plt.ylabel('Частота')

        # StandardScaler
        plt.subplot(1, 3, 3)
        sns.histplot(X_standard[:, 0], kde=True, color='red')
        plt.title('StandardScaler (mean≈0, std≈1)')
        plt.xlabel('Значения признака')
        plt.ylabel('Частота')

        plt.tight_layout()
        plt.show()

    except ValueError as ve:
        print(f"Ошибка в параметрах: {ve}")
    except Exception as e:
        print(f"Произошла ошибка: {e}")

if __name__ == "__main__":
    scale_and_visualize_data()
