# Pairplot Builder

## Описание

Модуль для создания информативных парных графиков (pairplot) с использованием библиотеки seaborn. Предназначен для визуального анализа взаимосвязей между признаками датасета с автоматической раскраской по целевому признаку.

## Особенности

- **Модульная архитектура** - четкое разделение ответственности между компонентами
- **Автоматическое определение целевого признака** - не требует явного указания target_col
- **Полная валидация данных** - проверка на наличие числовых признаков и корректность данных
- **Гибкая кастомизация** - поддержка различных цветовых схем и стилей
- **Универсальность** - работает с любыми датасетами, содержащими числовые признаки
- **Дополнительный анализ** - построение матриц корреляций и сохранение результатов
- **Подробная документация** - полное описание всех методов и параметров

## Установка

### Предварительные требования

- Python 3.8 или выше
- pip

### Установка зависимостей

```bash
git clone https://github.com/MikeV182/NSU_TCODE.git
cd DA-2-05

pip install -r requirements.txt
```

### Минимальная установка

Если у вас уже установлены основные библиотеки, можете установить только недостающие:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

## Использование

### Базовое использование

```python
from pairplot_demo import IrisPairplotBuilder
from sklearn.datasets import load_iris
import pandas as pd

# Загрузка данных
iris = load_iris()
data = pd.DataFrame(iris.data, columns=iris.feature_names)
data['target'] = iris.target

# Создание и отображение pairplot
builder = IrisPairplotBuilder(data, target_col='target')
plot = builder.build_pairplot()
```

### Использование с кастомными параметрами

```python
from pairplot_demo import IrisPairplotBuilder

# Расширенная конфигурация
builder = IrisPairplotBuilder(
    data=data,
    target_col='target',
    palette='husl',           # Цветовая схема
    diag_kind='hist',         # Гистограммы на диагонали
    plot_size=(12, 10)        # Размер фигуры
)

# Построение графика
plot = builder.build_pairplot(show=True)

# Сохранение результата
builder.save_plot('iris_analysis.png', dpi=300)
```

### Автоматическое определение целевого признака

```python
from pairplot_demo import IrisPairplotBuilder

# Без указания target_col - автоматическое определение
builder = IrisPairplotBuilder(data)
print(f"Автоопределенный целевой признак: {builder.target_col}")

plot = builder.build_pairplot()
```

### Анализ корреляций

```python
from pairplot_demo import IrisPairplotBuilder

builder = IrisPairplotBuilder(data, target_col='target')

# Тепловая карта корреляций
corr_plot = builder.plot_correlation_heatmap(method='pearson')

# Сводная информация о данных
summary = builder.get_data_summary()
print("Информация о датасете:")
for key, value in summary.items():
    print(f"  {key}: {value}")
```

## API Документация

### Класс `IrisPairplotBuilder`

#### Конструктор

```python
IrisPairplotBuilder(
    data: pd.DataFrame,
    target_col: str = None,
    palette: str = 'Set1',
    diag_kind: str = 'hist',
    plot_size: tuple = (10, 8),
    **kwargs
)
```

**Параметры:**
- `data` - pandas DataFrame с данными для анализа
- `target_col` - название столбца для раскраски точек (опционально)
- `palette` - цветовая схема seaborn ('Set1', 'husl', 'viridis', и др.)
- `diag_kind` - тип диагональных графиков ('hist', 'kde', None)
- `plot_size` - размер фигуры в дюймах (ширина, высота)
- `**kwargs` - дополнительные параметры для `seaborn.pairplot`

#### Основные методы

| Метод | Описание | Возвращает |
|-------|----------|------------|
| `build_pairplot(show=True)` | Строит pairplot визуализацию | `matplotlib.figure.Figure` |
| `save_plot(filename, dpi=300)` | Сохраняет график в файл | `str` (путь к файлу) |
| `plot_correlation_heatmap()` | Строит тепловую карту корреляций | `matplotlib.figure.Figure` |
| `get_data_summary()` | Возвращает сводку по данным | `dict` |

### Утилитарные функции

```python
def load_sample_iris_data() -> pd.DataFrame
    """Загружает образец данных Iris с расширенными именами признаков"""
```

```python
def create_custom_pairplot(data, target_col, output_path=None, **kwargs)
    """Создает pairplot с автоматическим сохранением"""
```

## Параметры визуализации

### Цветовые схемы

| Схема | Описание | Пример |
|-------|----------|--------|
| `Set1` | Дискретные цвета для категорий | `palette='Set1'` |
| `husl` | Полноцветная схема | `palette='husl'` |
| `viridis` | Перцептуально равномерная | `palette='viridis'` |
| `plasma` | Цветовая схема Matplotlib | `palette='plasma'` |
| `tab10` | 10 дискретных цветов | `palette='tab10'` |

### Типы диагональных графиков

| Тип | Описание | Использование |
|-----|----------|---------------|
| `hist` | Гистограммы частот | `diag_kind='hist'` |
| `kde` | Оценка плотности | `diag_kind='kde'` |
| `None` | Без диагональных графиков | `diag_kind=None` |

## Примеры использования

### 1. Анализ Iris датасета

```python
from sklearn.datasets import load_iris
from pairplot_demo import IrisPairplotBuilder
import pandas as pd

# Подготовка данных
iris = load_iris()
data = pd.DataFrame(iris.data, columns=iris.feature_names)
data['Species'] = pd.Categorical.from_codes(iris.target, iris.target_names)

# Создание pairplot
builder = IrisPairplotBuilder(
    data=data, 
    target_col='Species',
    palette='Set1',
    diag_kind='hist'
)

# Построение и сохранение
plot = builder.build_pairplot(show=True)
builder.save_plot('iris_pairplot.png', dpi=300)
```

### 2. Работа с CSV файлом

```python
import pandas as pd
from pairplot_demo import IrisPairplotBuilder

# Загрузка данных из CSV
df = pd.read_csv('dataset.csv')

# Автоматическое определение целевого признака
builder = IrisPairplotBuilder(df)

# Построение pairplot
plot = builder.build_pairplot()

# Анализ корреляций
corr_plot = builder.plot_correlation_heatmap(method='spearman')
```

### 3. Кастомизация стиля

```python
from pairplot_demo import IrisPairplotBuilder

builder = IrisPairplotBuilder(
    data=data,
    target_col='target',
    palette='husl',
    diag_kind='kde',
    plot_size=(14, 12),
    corner=True,           # Только нижний треугольник
    kind='reg'             # Линии регрессии
)

plot = builder.build_pairplot()
```

## Запуск демонстрации

Демо-скрипт показывает различные сценарии использования:

```bash
python pairplot_builder.py
```

Скрипт выполнит следующие демонстрации:
1. Базовое использование с Iris датасетом
2. Различные цветовые схемы
3. Анализ корреляций
4. Работа с синтетическим датасетом
5. Автоматическое определение целевого признака

## Структура проекта

```
DA-2-05/
├── pairplot_builder.py      # Основной модуль с классом и демонстрациями
├── requirements.txt      # Зависимости проекта
└── README.md            # Документация
```

## Метрики качества

### Визуальные критерии

- **Читаемость корреляций** - линии тренда и плотность точек должны быть различимы
- **Различимость классов** - цветовая схема должна четко разделять категории
- **Информативность диагонали** - гистограммы должны отражать распределения признаков
- **Сбалансированность осей** - пропорции графиков должны соответствовать данным

### Технические критерии

- **Производительность** - построение pairplot для датасетов до 1000 строк < 5 сек
- **Память** - потребление памяти пропорционально размеру датасета
- **Стабильность** - обработка всех типов входных данных без ошибок

## Устранение неполадок

### Ошибка: "Для pairplot требуется минимум 2 числовых признака"

**Причина:** Датасет содержит только категориальные данные или менее 2 числовых столбцов.

**Решение:**
```python
# Проверьте типы данных
print(df.dtypes)
print(df.select_dtypes(include=['number']).columns)

# Убедитесь, что есть числовые столбцы
numeric_df = df.select_dtypes(include=['number'])
```

### Ошибка: "Целевой столбец не найден"

**Причина:** Указанный `target_col` отсутствует в датасете.

**Решение:**
```python
# Проверьте доступные столбцы
print(df.columns.tolist())

# Используйте автоопределение
builder = IrisPairplotBuilder(df)  # без target_col
```

### Медленная работа с большими данными

**Решение:**
```python
# Ограничьте размер выборки
sample_data = df.sample(n=500, random_state=42)

# Используйте только ключевые признаки
key_features = ['feature1', 'feature2', 'feature3', 'target']
subset_data = df[key_features]

builder = IrisPairplotBuilder(subset_data, target_col='target')
```

## Лицензия

Этот проект распространяется под лицензией MIT. См. файл LICENSE для деталей.

## Контакты

Для вопросов и предложений обращайтесь к автору проекта или создавайте issues в репозитории.

---

*Разработано для упрощения анализа данных и визуализации взаимосвязей признаков*
