"""
Комплексное решение для построения pairplot визуализаций.

Модуль содержит класс IrisPairplotBuilder для создания парных графиков с автоматической
раскраской по целевому признаку, валидацией данных и возможностью сохранения результатов.
Также включает демонстрационные функции для различных сценариев использования.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Optional, Tuple, Dict, Any, Union
from sklearn.datasets import load_iris
import warnings
warnings.filterwarnings('ignore')


class IrisPairplotBuilder:
    """
    Class for creating pairplot visualizations with automatic target variable coloring.
    Supports customizable styling and validation for any numerical dataset.
    """
    
    def __init__(
        self,
        data: pd.DataFrame,
        target_col: Optional[str] = None,
        palette: str = 'Set1',
        diag_kind: str = 'hist',
        plot_size: Tuple[float, float] = (10, 8),
        **kwargs
    ):
        """
        Инициализация PairplotBuilder.
        
        Args:
            data (pd.DataFrame): Датасет для визуализации
            target_col (str, optional): Название целевого столбца для раскраски.
                                       Если None, будет попытка автоопределения.
            palette (str): Цветовая схема seaborn
            diag_kind (str): Тип диагональных графиков ('hist', 'kde', None)
            plot_size (tuple): Размер фигуры (ширина, высота)
            **kwargs: Дополнительные параметры для seaborn.pairplot
            
        Raises:
            ValueError: Если данные не содержат числовых признаков или целевой столбец отсутствует
        """
        self.data = self._validate_data(data)
        self.target_col = self._detect_target_column(target_col)
        self.palette = palette
        self.diag_kind = diag_kind
        self.plot_size = plot_size
        self.kwargs = kwargs
        self._last_plot = None
        
        # Настройка стиля по умолчанию
        self._setup_style()
        
        # Проверка совместимости
        self._validate_configuration()
    
    def _validate_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Валидация входных данных.
        
        Args:
            data: Входной датасет
            
        Returns:
            pd.DataFrame: Валидированный датасет
            
        Raises:
            ValueError: Если данные пусты или не содержат числовых столбцов
        """
        if data.empty:
            raise ValueError("Датасет не может быть пустым")
        
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) < 2:
            raise ValueError(
                "Для pairplot требуется минимум 2 числовых признака. "
                f"Найдено: {len(numeric_cols)}"
            )
        
        # Удаляем строки с пропусками в числовых столбцах
        numeric_data = data[numeric_cols].dropna()
        if numeric_data.empty:
            raise ValueError("После удаления пропусков не осталось данных")
        
        return data
    
    def _detect_target_column(self, target_col: Optional[str]) -> str:
        """
        Автоматическое определение целевого столбца.
        
        Args:
            target_col: Явно заданный целевой столбец
            
        Returns:
            str: Название целевого столбца
            
        Raises:
            ValueError: Если целевой столбец не найден и не удалось автоопределить
        """
        if target_col is not None:
            if target_col not in self.data.columns:
                raise ValueError(
                    f"Целевой столбец '{target_col}' не найден в датасете. "
                    f"Доступные столбцы: {list(self.data.columns)}"
                )
            return target_col
        
        # Автоопределение: ищем категориальный столбец с небольшим количеством уникальных значений
        for col in self.data.columns:
            if self.data[col].dtype == 'object' or (
                self.data[col].nunique() <= 10 and 
                self.data[col].dtype in ['int64', 'float64']
            ):
                return col
        
        # Если не нашли подходящий, используем первый категориальный или числовой с малым разбросом
        categorical_cols = self.data.select_dtypes(include=['object']).columns
        if len(categorical_cols) > 0:
            return categorical_cols[0]
        
        # Fallback: первый числовой столбец
        numeric_cols = self.data.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            return numeric_cols[0]
        
        raise ValueError(
            "Не удалось определить целевой столбец. "
            "Укажите target_col явно или добавьте категориальный столбец"
        )
    
    def _setup_style(self):
        """Настройка стиля matplotlib и seaborn."""
        plt.style.use('default')
        sns.set_palette(self.palette)
        sns.set_style("whitegrid")
    
    def _validate_configuration(self):
        """Валидация конфигурации перед построением."""
        valid_diag_kinds = ['hist', 'kde', None]
        if self.diag_kind not in valid_diag_kinds:
            warnings.warn(
                f"Неизвестный тип диагонального графика '{self.diag_kind}'. "
                f"Используется 'hist'. Допустимые значения: {valid_diag_kinds}"
            )
            self.diag_kind = 'hist'
    
    def build_pairplot(
        self, 
        corr_method: str = 'pearson',
        show: bool = True,
        **additional_kwargs
    ) -> plt.Figure:
        """
        Строит pairplot визуализацию.
        
        Args:
            corr_method (str): Метод корреляции ('pearson', 'spearman', 'kendall')
            show (bool): Показывать график сразу
            **additional_kwargs: Дополнительные параметры для seaborn.pairplot
            
        Returns:
            plt.Figure: Объект фигуры matplotlib
            
        Raises:
            ValueError: Если не хватает данных для построения
        """
        # Подготовка данных для pairplot
        plot_data = self.data.select_dtypes(include=[np.number]).dropna()
        
        if plot_data.shape[1] < 2:
            raise ValueError("Для pairplot требуется минимум 2 числовых признака")
        
        # Обновляем параметры
        final_kwargs = {
            'data': self.data,
            'hue': self.target_col,
            'diag_kind': self.diag_kind,
            'palette': self.palette,
            'height': self.plot_size[1] / max(1, len(self.data.select_dtypes(include=[np.number]).columns) - 1),
            'aspect': self.plot_size[0] / self.plot_size[1],
            **self.kwargs,
            **additional_kwargs
        }
        
        try:
            # Строим pairplot
            g = sns.pairplot(**final_kwargs)
            
            # Добавляем заголовок
            target_levels = self.data[self.target_col].unique()
            title = f"Pairplot Analysis: {self.target_col} (n_classes={len(target_levels)})"
            g.fig.suptitle(title, y=1.02, fontsize=16, fontweight='bold')
            
            self._last_plot = g.fig
            
            if show:
                plt.tight_layout()
                plt.show()
            
            return g.fig
            
        except Exception as e:
            raise RuntimeError(
                f"Ошибка при построении pairplot: {str(e)}. "
                f"Проверьте наличие числовых данных и корректность target_col"
            )
    
    def save_plot(
        self, 
        filename: str, 
        dpi: int = 300, 
        format: str = 'png',
        bbox_inches: str = 'tight'
    ) -> str:
        """
        Сохраняет pairplot в файл.
        
        Args:
            filename (str): Имя файла для сохранения
            dpi (int): Разрешение изображения
            format (str): Формат файла ('png', 'pdf', 'svg')
            bbox_inches (str): Обработка границ ('tight', None)
            
        Returns:
            str: Полный путь к сохраненному файлу
        """
        if not hasattr(self, '_last_plot'):
            raise RuntimeError(
                "Сначала необходимо построить pairplot с помощью build_pairplot()"
            )
        
        filepath = f"{filename}.{format}"
        self._last_plot.savefig(
            filepath, 
            dpi=dpi, 
            format=format, 
            bbox_inches=bbox_inches
        )
        print(f"График сохранен: {filepath}")
        return filepath
    
    def get_data_summary(self) -> Dict[str, Any]:
        """
        Возвращает сводную информацию о данных.
        
        Returns:
            Dict[str, Any]: Информация о датасете
        """
        numeric_cols = self.data.select_dtypes(include=[np.number]).columns
        target_info = {
            'target_column': self.target_col,
            'target_unique': self.data[self.target_col].nunique(),
            'target_categories': list(self.data[self.target_col].unique()),
            'numeric_features': len(numeric_cols),
            'total_features': len(self.data.columns),
            'samples': len(self.data),
            'missing_values': self.data.isnull().sum().sum()
        }
        return target_info
    
    def plot_correlation_heatmap(self, method: str = 'pearson', show: bool = True) -> plt.Figure:
        """
        Строит тепловую карту корреляций.
        
        Args:
            method (str): Метод корреляции
            show (bool): Показать график
            
        Returns:
            plt.Figure: Фигура с тепловой картой
        """
        numeric_data = self.data.select_dtypes(include=[np.number])
        corr_matrix = numeric_data.corr(method=method)
        
        plt.figure(figsize=(10, 8))
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        
        sns.heatmap(
            corr_matrix,
            mask=mask,
            annot=True,
            cmap='coolwarm',
            center=0,
            square=True,
            fmt='.2f',
            cbar_kws={'shrink': 0.8}
        )
        
        plt.title(f'Корреляционная матрица (метод: {method})', fontsize=14, fontweight='bold')
        
        if show:
            plt.tight_layout()
            plt.show()
        
        return plt.gcf()


def load_sample_iris_data() -> pd.DataFrame:
    """
    Загружает образец данных Iris для демонстрации.
    
    Returns:
        pd.DataFrame: Iris датасет с расширенными именами признаков
    """
    iris = load_iris()
    feature_names = [
        'Sepal Length (cm)', 'Sepal Width (cm)', 
        'Petal Length (cm)', 'Petal Width (cm)'
    ]
    
    data = pd.DataFrame(iris.data, columns=feature_names)
    data['Species'] = pd.Categorical.from_codes(
        iris.target, iris.target_names
    )
    
    return data


def create_custom_pairplot(
    data: pd.DataFrame,
    target_col: str,
    output_path: Optional[str] = None,
    **plot_kwargs
) -> Tuple[plt.Figure, Optional[str]]:
    """
    Утилита для создания pairplot с автоматическим сохранением.
    
    Args:
        data: Датасет
        target_col: Целевой столбец
        output_path: Путь для сохранения (если None - только отображение)
        **plot_kwargs: Параметры для PairplotBuilder
        
    Returns:
        Tuple[plt.Figure, Optional[str]]: Фигура и путь к файлу
    """
    builder = IrisPairplotBuilder(data, target_col, **plot_kwargs)
    
    # Построение pairplot
    fig = builder.build_pairplot(show=False)
    
    saved_path = None
    if output_path:
        saved_path = builder.save_plot(output_path)
    
    return fig, saved_path


def demo_basic_usage():
    """Демонстрация базового использования."""
    print("Демонстрация базового использования")
    print("=" * 50)
    
    # Загрузка данных
    print("Загружаем Iris датасет...")
    data = load_sample_iris_data()
    print(f"Размер датасета: {data.shape}")
    print(f"Признаки: {list(data.columns[:-1])}")
    print(f"Классы: {data['Species'].unique()}")
    
    # Создание pairplot
    print("\nСтроим базовый pairplot...")
    builder = IrisPairplotBuilder(data, target_col='Species')
    
    # Сводная информация
    summary = builder.get_data_summary()
    print(f"\nСводка данных:")
    for key, value in summary.items():
        print(f"   {key}: {value}")
    
    # Построение и отображение
    fig = builder.build_pairplot(show=True)
    plt.close(fig)


def demo_advanced_configuration():
    """Демонстрация расширенной конфигурации."""
    print("\nДемонстрация расширенной конфигурации")
    print("=" * 50)
    
    data = load_sample_iris_data()
    
    # Разные цветовые схемы
    palettes = ['Set1', 'husl', 'viridis', 'plasma']
    
    for i, palette in enumerate(palettes, 1):
        print(f"\nПалитра {i}/{len(palettes)}: {palette}")
        
        builder = IrisPairplotBuilder(
            data=data,
            target_col='Species',
            palette=palette,
            diag_kind='kde',  # Используем KDE вместо гистограмм
            plot_size=(12, 10)
        )
        
        fig = builder.build_pairplot(show=False)
        
        # Сохранение с разными цветами
        filename = f"iris_pairplot_{palette.lower()}"
        saved_path = builder.save_plot(
            filename=filename,
            dpi=150,
            format='png'
        )
        print(f"Сохранено: {saved_path}")
        
        plt.close(fig)


def demo_correlation_analysis():
    """Демонстрация анализа корреляций."""
    print("\nДемонстрация анализа корреляций")
    print("=" * 50)
    
    data = load_sample_iris_data()
    builder = IrisPairplotBuilder(data, target_col='Species')
    
    # Тепловая карта корреляций
    print("Строим тепловую карту корреляций...")
    fig_corr = builder.plot_correlation_heatmap(method='pearson', show=True)
    plt.close(fig_corr)
    
    # Проверка корреляций
    numeric_data = data.select_dtypes(include='number')
    corr_matrix = numeric_data.corr(method='pearson')
    
    print("\nОсновные корреляции:")
    high_corr = corr_matrix.abs().unstack().sort_values(ascending=False)
    high_corr = high_corr[
        high_corr < 1.0  # Исключаем корреляции с самим собой
    ].drop_duplicates().head(5)
    
    for (feature1, feature2), corr_value in high_corr.items():
        print(f"   {feature1} ↔ {feature2}: {corr_value:.3f}")


def demo_custom_dataset():
    """Демонстрация с кастомным датасетом."""
    print("\nДемонстрация с кастомным датасетом")
    print("=" * 50)
    
    # Создаем синтетический датасет
    np.random.seed(42)
    n_samples = 150
    
    custom_data = pd.DataFrame({
        'feature1': np.random.normal(5, 1.5, n_samples),
        'feature2': np.random.normal(3, 1, n_samples),
        'feature3': np.random.normal(4, 2, n_samples),
        'feature4': np.random.exponential(1, n_samples),
        'target': np.random.choice(['A', 'B', 'C'], n_samples)
    })
    
    # Добавляем корреляцию между feature1 и feature2
    custom_data['feature2'] = (
        custom_data['feature1'] * 0.7 + 
        np.random.normal(0, 0.5, n_samples)
    )
    
    print(f"Создан синтетический датасет: {custom_data.shape}")
    print(f"Классы target: {custom_data['target'].value_counts().to_dict()}")
    
    # Построение pairplot
    builder = IrisPairplotBuilder(
        data=custom_data,
        target_col='target',
        palette='tab10',
        diag_kind='hist'
    )
    
    fig, saved_path = create_custom_pairplot(
        data=custom_data,
        target_col='target',
        output_path='custom_dataset_analysis.png',
        plot_size=(14, 12)
    )
    
    print("Анализ кастомного датасета завершен")
    if saved_path:
        print(f"Сохранено: {saved_path}")
    
    plt.close(fig)


def demo_auto_target_detection():
    """Демонстрация автоопределения целевого столбца."""
    print("\nДемонстрация автоопределения целевого столбца")
    print("=" * 50)
    
    # Датасет без явного target_col
    data = load_sample_iris_data()
    data_no_target = data.drop('Species', axis=1)
    
    print("Датасет без target_col:")
    print(f"Столбцы: {list(data_no_target.columns)}")
    
    try:
        # Попытка автоопределения
        builder = IrisPairplotBuilder(data_no_target)
        detected_target = builder.target_col
        print(f"Автоопределен target_col: '{detected_target}'")
        
        # Построение (будет использовано первый числовой столбец)
        fig = builder.build_pairplot(show=False)
        plt.close(fig)
        
    except Exception as e:
        print(f"Ошибка автоопределения: {e}")


def main():
    """Главная функция демонстрации."""
    print("Iris Pairplot Builder - Демонстрация")
    print("=" * 60)
    print("Этот скрипт демонстрирует возможности модуля PairplotBuilder\n")
    
    try:
        demo_basic_usage()
        demo_advanced_configuration()
        demo_correlation_analysis()
        demo_custom_dataset()
        demo_auto_target_detection()
        
        print("\n" + "=" * 60)
        print("Все демонстрации успешно завершены!")
        print("\nДля использования в своих проектах:")
        print("from pairplot_builder import IrisPairplotBuilder")
        print("builder = IrisPairplotBuilder(data, target_col='your_target')")
        print("builder.build_pairplot()")
        
    except KeyboardInterrupt:
        print("\nДемонстрация прервана пользователем")
    except Exception as e:
        print(f"\nПроизошла ошибка: {e}")
        print("Проверьте установку зависимостей: pip install -r requirements.txt")
    
    finally:
        plt.close('all')


if __name__ == "__main__":
    main()
