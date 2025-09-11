import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import seaborn as sns

# Генерация синтетических данных
X, y = make_classification(n_samples=1000, n_features=1, n_informative=1, n_redundant=0, n_clusters_per_class=1, random_state=42)

# Инициализация скейлеров
minmax_scaler = MinMaxScaler()
standard_scaler = StandardScaler()

# Применение MinMaxScaler
X_minmax = minmax_scaler.fit_transform(X)

# Применение StandardScaler
X_standard = standard_scaler.fit_transform(X)

# Проверка метрик для StandardScaler
mean_standard = np.mean(X_standard)
std_standard = np.std(X_standard)
print(f"StandardScaler: mean = {mean_standard:.4f}, std = {std_standard:.4f}")

# Визуализация распределений
plt.figure(figsize=(15, 5))

# Исходные данные
plt.subplot(1, 3, 1)
sns.histplot(X, kde=True, color='blue')
plt.title('Исходное распределение')
plt.xlabel('Значения признака')
plt.ylabel('Частота')

# MinMaxScaler
plt.subplot(1, 3, 2)
sns.histplot(X_minmax, kde=True, color='green')
plt.title('MinMaxScaler (0 to 1)')
plt.xlabel('Значения признака')
plt.ylabel('Частота')

# StandardScaler
plt.subplot(1, 3, 3)
sns.histplot(X_standard, kde=True, color='red')
plt.title('StandardScaler (mean≈0, std≈1)')
plt.xlabel('Значения признака')
plt.ylabel('Частота')

plt.tight_layout()
plt.show()
