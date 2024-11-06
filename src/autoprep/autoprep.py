import numpy as np
import pandas as pd
import logging
import seaborn as sns  # Para gráficos de densidad (kdeplot)
import matplotlib.pyplot as plt  # Para mostrar gráficos
from scipy import stats  # Para posibles pruebas de normalidad (opcional)

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import accuracy_score, mean_squared_error


class AutoPrep:
    """
    Unified class for data processing, including preprocessing, exploratory analysis,
    and model testing, centralized into a single interface.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing the data to be processed.
    """

    def __init__(self, df: pd.DataFrame):
        if not self._validate_dataframe(df):
            raise ValueError("Input must be a DataFrame-like object")
        self.df = df
        self.logger = self._setup_logger()

    @staticmethod
    def _validate_dataframe(df) -> bool:
        """Validate if the input is a DataFrame-like object."""
        required_attrs = ["columns", "index", "values"]
        return all(hasattr(df, attr) for attr in required_attrs)

    @staticmethod
    def _setup_logger() -> logging.Logger:
        """Set up a logger for debugging purposes."""
        logger = logging.getLogger("UnifiedDataProcessor")
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
        return logger

    # ----------- Preprocessing Methods -----------

    def analyze_missing(self, threshold: float = 0.9, impute_strategy: str = "mean") -> dict:
        """
        Analyze and handle missing values in the DataFrame.

        Parameters
        ----------
        threshold : float, optional
            Maximum ratio of missing values allowed, by default 0.9.
        impute_strategy : str, optional
            Strategy for imputing missing values, by default "mean".

        Returns
        -------
        dict
            Dictionary containing missing value analysis results.
        """
        # Calcular estadísticas de valores faltantes
        missing_stats = {
        'total_missing': self.df.isnull().sum().to_dict(),
        'columns_with_missing': {col: val for col, val in self.df.isnull().sum().items() if val > 0},
        'missing_ratio': self.df.isnull().mean().to_dict()
        }

        # Identificar columnas a eliminar
        cols_to_drop = [col for col, ratio in missing_stats['missing_ratio'].items() if ratio > threshold]
        if cols_to_drop:
            self.df.drop(columns=cols_to_drop, inplace=True)
            missing_stats['dropped_columns'] = cols_to_drop

        # Imputación de valores faltantes según la estrategia
        for col in self.df.columns:
            if self.df[col].isnull().any():
                if impute_strategy == "mean" and self.df[col].dtype in ['float64', 'int64']:
                    self.df[col].fillna(self.df[col].mean(), inplace=True)
                elif impute_strategy == "median" and self.df[col].dtype in ['float64', 'int64']:
                    self.df[col].fillna(self.df[col].median(), inplace=True)
                elif impute_strategy == "mode":
                    self.df[col].fillna(self.df[col].mode()[0], inplace=True)

        return missing_stats

    def handle_outliers(self, columns: list = None, method: str = "zscore", threshold: float = 3.0) -> dict:
        """
        Detect and handle outliers in specified columns.

        Parameters
        ----------
        columns : list, optional
            List of columns to check for outliers.
        method : str, optional
            Method to detect outliers ("zscore" or "iqr").
        threshold : float, optional
            Threshold for outlier detection.

        Returns
        -------
        dict
            Dictionary containing outlier analysis results.
        """
        if columns is None:
            columns = self.df.select_dtypes(include=[np.number]).columns

        outliers_stats = {}
        for col in columns:
            if method == "zscore":
                z_scores = np.abs((self.df[col] - self.df[col].mean()) / self.df[col].std())
                outliers_stats[col] = (z_scores > threshold).sum()
            elif method == "iqr":
                Q1, Q3 = self.df[col].quantile(0.25), self.df[col].quantile(0.75)
                IQR = Q3 - Q1
                outliers_stats[col] = ((self.df[col] < (Q1 - 1.5 * IQR)) | (self.df[col] > (Q3 + 1.5 * IQR))).sum()
        return outliers_stats

    # ----------- Exploratory Analysis Methods -----------

    def get_basic_stats(self) -> dict:
        """
        Calculate basic statistics for numerical columns.

        Returns
        -------
        dict
            Dictionary containing basic statistics for each numerical column.
        """
        stats = {}
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            stats[col] = {
                'mean': float(self.df[col].mean()),
                'median': float(self.df[col].median()),
                'std': float(self.df[col].std()),
                'min': float(self.df[col].min()),
                'max': float(self.df[col].max())
            }
        return stats

    def normality_test(self, columns: list = None) -> dict:
        """
        Perform basic normality tests on numerical columns.

        Parameters
        ----------
        columns : list, optional
            List of columns to test.

        Returns
        -------
        dict
            Dictionary containing normality test results.
        """
        if columns is None:
            columns = self.df.select_dtypes(include=[np.number]).columns

        results = {}
        for col in columns:
            skewness = float(self.df[col].skew())
            kurtosis = float(self.df[col].kurtosis())
            results[col] = {
                'skewness': skewness,
                'kurtosis': kurtosis,
                'is_normal': abs(skewness) < 2 and abs(kurtosis) < 7
            }
        return results

    # ----------- Model Testing Methods -----------

    def run_models(self, target: str) -> dict:
        """
        Fit and evaluate multiple models on the data.

        Parameters
        ----------
        target : str
            Name of the target column.

        Returns
        -------
        dict
            Dictionary of model names and their evaluation metrics.
        """
        # Separar los datos en X e y
        X = self.df.drop(columns=[target])
        y = self.df[target]

        # Convertir columnas categóricas a numéricas
        X = pd.get_dummies(X, drop_first=True)

        # Separar los datos en conjuntos de entrenamiento y prueba
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Determinar si el target es numérico o categórico
        if pd.api.types.is_numeric_dtype(y):
            # Si el target es numérico, usar modelos de regresión
            models = {
                "RandomForestRegressor MSE": RandomForestRegressor(),
                "LinearRegression MSE": LinearRegression()
            }
            metric_fn = mean_squared_error  # Usar MSE como métrica para regresión
        else:
            # Si el target es categórico, usar modelos de clasificación
            models = {
                "RandomForestClassifier ACC": RandomForestClassifier(),
                "LogisticRegression ACC": LogisticRegression()
            }
            metric_fn = accuracy_score  # Usar precisión como métrica para clasificación

        # Entrenar y evaluar cada modelo
        scores = {}
        for name, model in models.items():
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            
            # Calcular la métrica adecuada
            if pd.api.types.is_numeric_dtype(y):
                # Para regresión, calcular el error cuadrático medio
                scores[name] = metric_fn(y_test, y_pred)
            else:
                # Para clasificación, calcular la precisión
                scores[name] = metric_fn(y_test, y_pred)

        return scores


    # ----------- Full Analysis Method -----------

    def run_full_analysis(self, target: str = None) -> dict:
        """
        Run a complete analysis pipeline.

        Parameters
        ----------
        target : str, optional
            Target column name for modeling.

        Returns
        -------
        dict
            Dictionary containing all analysis results.
        """
        results = {
            'missing_analysis': self.analyze_missing(),
            'outliers_analysis': self.handle_outliers(),
            'basic_stats': self.get_basic_stats(),
            'normality_tests': self.normality_test()
        }

        if target:
            results['model_accuracy'] = self.run_models(target)
        return results