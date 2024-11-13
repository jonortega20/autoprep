import numpy as np
import pandas as pd
import logging
import seaborn as sns
import matplotlib.pyplot as plt
import numbers
import warnings


from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.feature_selection import RFE, SelectKBest, chi2


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

    def analyze_missing(self, threshold=0.9, impute_strategy="mean") -> dict:
        """
        Analyze and handle missing values in the DataFrame by providing a summary and imputing missing values based on a chosen strategy.

        Parameters
        ----------
        threshold : float, optional
            Maximum ratio of missing values allowed in a column before it is dropped. Default is 0.9 (i.e., columns with more than 90% missing values will be dropped).
        impute_strategy : str, optional
            Strategy for imputing missing values. Can be one of 'mean', 'median', or 'mode'. Default is 'mean'. 

            - **'mean'** replaces missing values with the column's mean (for numeric columns).
            - **'median'** replaces missing values with the column's median (for numeric columns).
            - **'mode'** replaces missing values with the column's mode (for all types of columns, including categorical).

        Returns
        -------
        dict
            A dictionary containing the results of the missing value analysis, which includes the following keys:
            
            - **'total_missing'**: A dictionary with the total number of missing values per column.
            - **'columns_with_missing'**: A dictionary with columns that have missing values and their counts.
            - **'missing_ratio'**: A dictionary with the proportion of missing values per column.
            - **'dropped_columns'**: A list of columns that were dropped due to having missing values above the specified threshold.

        Raises
        ------
        TypeError
            If 'threshold' is not a numeric value or 'impute_strategy' is not a string.
        ValueError
            If 'impute_strategy' is not one of the valid options ('mean', 'median', or 'mode').
        """
        warnings.filterwarnings("ignore")

        if not isinstance(threshold, numbers.Number):
            raise TypeError("Threshold must be a numeric value.")
        
        if not isinstance(impute_strategy, str):
            raise TypeError("Impute strategy must be a string value.")

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
                elif impute_strategy in ["mean", "median"] and self.df[col].dtype == 'object':
                    self.df[col].fillna(self.df[col].mode()[0], inplace=True)
                elif impute_strategy == "mode":
                    self.df[col].fillna(self.df[col].mode()[0], inplace=True)
                else:
                    raise ValueError("Invalid impute strategy. Choose from 'mean', 'median', or 'mode'.")

        return missing_stats

    def handle_outliers(self, columns = None, method = "zscore", threshold = 3.0) -> dict:
        """
        Detect and handle outliers in specified columns of the DataFrame using a chosen method.

        Parameters
        ----------
        columns : list, optional
            List of column names to check for outliers. If None, all numeric columns will be checked. Default is None.
        method : str, optional
            Method to detect outliers. Default is 'zscore'. Options are:

            - **'zscore'**: Detect outliers based on Z-scores, where values beyond the specified threshold are considered outliers.
            - **'iqr'**: Detect outliers using the Interquartile Range (IQR), where values outside the range defined by 1.5 * IQR above Q3 or below Q1 are considered outliers.
    
        threshold : float, optional
            
            The threshold value for outlier detection. For 'zscore' method, it defines the Z-score beyond which values are considered outliers. 
            For 'iqr' method, values outside 1.5 * IQR above Q3 or below Q1 are considered outliers.
            Default is 3.0.

        Returns
        -------
        dict
            A dictionary containing the number of outliers detected in each specified column. 
            The dictionary has column names as keys and the corresponding count of outliers as values.

        Raises
        ------
        TypeError
            If 'columns' is not a list or 'threshold' is not a numeric value.
        ValueError
            If 'method' is not one of the valid options ('zscore' or 'iqr').

        Notes
        -----
        - The 'zscore' method detects outliers based on the standard deviation of each column, where values with Z-scores greater than the threshold are flagged as outliers.
        - The 'iqr' method detects outliers by identifying values outside the range defined by 1.5 * IQR (Interquartile Range), which is the range between the first (Q1) and third (Q3) quartiles.
        """
        warnings.filterwarnings("ignore")
        if columns is None:
            columns = self.df.select_dtypes(include=[np.number]).columns
        
        if not isinstance(columns, list):
            raise TypeError("Columns must be a list of column names.")
        
        if not isinstance(method, str):
            raise TypeError("Method must be a string value.")
        
        if isinstance(threshold, numbers.Number):
            raise TypeError("Threshold must be a numeric value.")

        outliers_stats = {}
        for col in columns:
            if method == "zscore":
                z_scores = np.abs((self.df[col] - self.df[col].mean()) / self.df[col].std())
                outliers_stats[col] = (z_scores > threshold).sum()
            elif method == "iqr":
                Q1, Q3 = self.df[col].quantile(0.25), self.df[col].quantile(0.75)
                IQR = Q3 - Q1
                outliers_stats[col] = ((self.df[col] < (Q1 - 1.5 * IQR)) | (self.df[col] > (Q3 + 1.5 * IQR))).sum()
            else:
                raise ValueError("Invalid method. Choose from 'zscore' or 'iqr'.")
        return outliers_stats

    # ----------- Exploratory Analysis Methods -----------

    def get_basic_stats(self) -> dict:
        """
        Calculate basic statistics for numerical columns in the DataFrame.

        This function computes common statistics for each numerical column, including:
        mean, median, standard deviation, minimum, and maximum values.

        Returns
        -------
        dict
            A dictionary containing the basic statistics for each numerical column.
            The keys of the dictionary are the column names, and the values are dictionaries
            containing the statistics ('mean', 'median', 'std', 'min', 'max') for each column.

        Notes
        -----
        - The function filters the DataFrame to only include numerical columns (i.e., columns with int or float data types).
        - The statistics are calculated as follows:
        - 'mean': The average of the values in the column.
        - 'median': The middle value when the values are sorted.
        - 'std': The standard deviation of the column values.
        - 'min': The smallest value in the column.
        - 'max': The largest value in the column.
        """
        warnings.filterwarnings("ignore")
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

    def normality_test(self, columns = None) -> dict:
        """
        Perform normality tests on numerical columns by calculating skewness and kurtosis.

        This function calculates the skewness and kurtosis for each specified numerical column
        to assess the normality of the data distribution. Additionally, it determines whether 
        the distribution is approximately normal based on skewness and kurtosis thresholds:

        - A skewness value between -2 and 2 is considered acceptable for normality.
        - A kurtosis value between -7 and 7 is considered acceptable for normality.

        Parameters
        ----------
        columns : list, optional
            A list of column names to test for normality. If not provided, all numerical columns
            in the DataFrame will be tested.

        Returns
        -------
        dict
            A dictionary where each key is a column name, and the value is another dictionary
            containing the following keys:

            - **'skewness'**: The skewness of the column.
            - **'kurtosis'**: The kurtosis of the column.
            - **'is_normal'**: A boolean indicating whether the column's distribution is approximately normal based on skewness and kurtosis thresholds.

        Notes
        -----
        - Skewness measures the asymmetry of the distribution: negative skew indicates a left-heavy distribution,
          and positive skew indicates a right-heavy distribution.
        - Kurtosis measures the tailedness of the distribution: a value close to 3 suggests a normal distribution,
          while values significantly higher or lower than 3 suggest deviations from normality.
        - The normality criteria used in this function are based on general empirical thresholds for skewness and kurtosis.
        """
        warnings.filterwarnings("ignore")
        if columns is None:
            columns = self.df.select_dtypes(include=[np.number]).columns
        
        if not isinstance(columns, list):
            raise TypeError("Columns must be a list of column names.")

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

    def run_models(self, target = None) -> dict:
        """
        Fit and evaluate multiple models on the data, using either regression or classification models depending on the target variable.

        This function splits the data into training and test sets, applies the appropriate machine learning models based on the 
        target variable (either regression models for numerical targets or classification models for categorical targets),
        trains the models, and evaluates them using relevant metrics (Mean Squared Error for regression, Accuracy for classification).

        Parameters
        ----------
        target : str
            The name of the target column in the DataFrame. This column is used to predict the target variable using the other columns as features.

        Returns
        -------
        dict
            A dictionary where keys are the names of the models and the values are the corresponding evaluation metrics. 
            The metrics are Mean Squared Error (MSE) for regression models and Accuracy for classification models.

        Raises
        ------
        ValueError
            If the target column is not specified.
        TypeError
            If the target column is not a string.

        Notes
        -----
        - The function automatically detects whether the target variable is numeric (regression) or categorical (classification).
        - The models used for regression are `RandomForestRegressor` and `LinearRegression`, and for classification, 
          the models used are `RandomForestClassifier` and `LogisticRegression`.
        - The evaluation metric for regression is Mean Squared Error (MSE), and for classification, it is Accuracy.
        """
        warnings.filterwarnings("ignore")
        if target is None:
            raise ValueError("Target column must be specified for model testing.")
        
        if not isinstance(target, str):
            raise TypeError("Target column must be a string value.")
        
        X = self.df.drop(columns=[target])
        y = self.df[target]

        X = pd.get_dummies(X, drop_first=True)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        if pd.api.types.is_numeric_dtype(y):
            models = {
                "RandomForestRegressor MSE": RandomForestRegressor(),
                "LinearRegression MSE": LinearRegression()
            }
            metric_fn = mean_squared_error  
        else:          
            models = {
                "RandomForestClassifier ACC": RandomForestClassifier(),
                "LogisticRegression ACC": LogisticRegression()
            }            
            metric_fn = accuracy_score  

        scores = {}
        for name, model in models.items():
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            
            if pd.api.types.is_numeric_dtype(y):
                scores[name] = metric_fn(y_test, y_pred)
            else:
                scores[name] = metric_fn(y_test, y_pred)

        return scores
    
    # ----------- Feature Importance -----------
    
    def simple_feature_importance(self, target, test_size=0.2, top_n=10) -> None:
        """
        Train a simple model to calculate and display feature importances based on the target type.

        This function splits the data into training and test sets, trains a Random Forest model (either regression or classification depending on the target), 
        calculates the feature importances, and visualizes the top features using a bar plot.

        Parameters
        ----------
        target : str
            The name of the target column in the DataFrame. This column is used to predict the target variable using the other columns as features.
        test_size : float, optional
            Proportion of the dataset to include in the test split. The default value is 0.2, meaning 20% of the data will be used for testing.
        top_n : int, optional
            The number of top features to display in the importance plot. The default value is 10, meaning the top 10 features will be shown.

        Returns
        -------
        None
            This function does not return any value, but it displays a bar plot of the top feature importances.

        Raises
        ------
        ValueError
            If the target column is not specified or is not a string, or if the target column does not exist in the DataFrame.
        
        Notes
        -----
        - The function automatically detects whether the target variable is numeric (regression) or categorical (classification) to decide the model type.
        - It uses a Random Forest model (RandomForestRegressor for regression or RandomForestClassifier for classification).
        - The function visualizes the importance of the top features using a bar plot, with the feature names on the y-axis and their importance scores on the x-axis.
        """
        try:

            if not isinstance(target, str):
                raise ValueError("The target parameter must be a string representing the column name.")


            if target not in self.df.columns:
                raise ValueError(f"The target column '{target}' does not exist in the DataFrame.")

            X = self.df.drop(columns=[target])
            y = self.df[target]
            
        except KeyError:
            raise ValueError(f"The target column '{target}' does not exist in the DataFrame.")

        X = pd.get_dummies(X, drop_first=True)


        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

        if pd.api.types.is_numeric_dtype(y):
            model = RandomForestRegressor(random_state=42)
        else:
            model = RandomForestClassifier(random_state=42)
        
        model.fit(X_train, y_train)

        if hasattr(model, "feature_importances_"):
            feature_importances = pd.DataFrame({
                'Feature': X.columns,
                'Importance': model.feature_importances_
            }).sort_values(by="Importance", ascending=False)

            plt.figure(figsize=(10, 6))
            sns.barplot(x="Importance", y="Feature", data=feature_importances.head(top_n), palette="viridis")
            plt.title("Top Feature Importances")
            plt.xlabel("Importance")
            plt.ylabel("Features")
            plt.show()
        else:
            print("The model does not support feature importances.")

    # ----------- Feature Selection -----------

    def select_features(self, method = 'rfe', target = None, k= 10, threshold = 0.9) -> pd.DataFrame:
        """
        Perform automatic feature selection based on the specified method.

        Parameters
        ----------
        method : str, optional
            Feature selection method. Options are 'rfe', 'importance', 'correlation', 'selectkbest', by default 'rfe'.

            - **'rfe'**: Recursive Feature Elimination
            - **'importance'**: Feature importance based on a Random Forest model
            - **'correlation'**: Select features with the highest correlation to the target variable
            - **'selectkbest'**: Select K best features using a statistical test (Chi-squared)
        target : str, optional
            The name of the target column for feature selection. This is used to compute correlations and fit models.
        k : int, optional
            The number of top features to select. Default is 10.
        threshold : float, optional
            Correlation threshold for the 'correlation' method. Features with correlation greater than this value will be selected.
            Default is 0.9.

        Returns
        -------
        pd.DataFrame
            DataFrame containing the selected features based on the chosen method.

        Raises
        ------
        ValueError
            If the target column is not specified, or if the target column contains null values, or if `k` is greater than the number of columns.
        TypeError
            If the method is invalid or if the target column is not a string, or if the DataFrame contains non-numeric columns when performing feature selection.
        
        Notes
        -----
        - The `method` parameter determines the feature selection technique:
        - 'rfe': Uses Recursive Feature Elimination with a model (either linear regression or logistic regression).
        - 'importance': Uses feature importances from a RandomForest model.
        - 'correlation': Selects features based on their correlation with the target variable.
        - 'selectkbest': Selects the top K features based on the Chi-squared test.
        - The `k` parameter must be less than or equal to the number of available features in the DataFrame.
        - The `threshold` parameter is only used for the 'correlation' method to filter features by their correlation with the target.
        """
        warnings.filterwarnings("ignore")
        # Validación de formatos correctos
        if not isinstance(k, int):
            raise TypeError("k must be an integer value.")
        
        if target is None:
                raise ValueError("Target column must be specified for RFE method.")
        
        if not isinstance(target, str):
                raise ValueError("Target column must be specified for RFE method.")
        
        if self.df.isnull().values.any():
            raise ValueError("The DataFrame contains null or NA values.")
        
        df = self.df.copy()
        df.drop(columns = target, inplace = True)
        all_numeric = df.apply(lambda col: pd.api.types.is_numeric_dtype(col)).all()

        if not all_numeric:
            raise ValueError("All columns must be numeric for feature selection methods.")
        
        if k > len(df.columns):
            raise ValueError("k must be less than the number of columns in the DataFrame.")

        if method == 'rfe':
            X = self.df.drop(columns=[target])
            y = self.df[target]
            # Usar un modelo de regresión o clasificación dependiendo del target
            model = LinearRegression() if pd.api.types.is_numeric_dtype(y) else LogisticRegression()
            selector = RFE(model, n_features_to_select=k)
            selector = selector.fit(X, y)
            selected_columns = X.columns[selector.support_]
            return selected_columns

        elif method == 'importance':
            X = self.df.drop(columns=[target])
            y = self.df[target]
            # Usar RandomForest para obtener la importancia de las características
            model = RandomForestClassifier() if not pd.api.types.is_numeric_dtype(y) else RandomForestRegressor()
            model.fit(X, y)
            importances = model.feature_importances_
            indices = np.argsort(importances)[::-1]
            selected_columns = X.columns[indices[:k]]
            return selected_columns

        elif method == 'correlation':
            if self.df[target].dtype not in [np.float64, np.int64, np.float32, np.int32]:
                raise TypeError("Target must be a numeric value to use correlation method.")
            corr_with_target = self.df.corr()[target].abs()
            sorted_corr = corr_with_target.sort_values(ascending=False)
            selected_columns = sorted_corr.index[1:k+1]
            return selected_columns

        elif method == 'selectkbest':
            X = self.df.drop(columns=[target])
            y = self.df[target]
            selector = SelectKBest(chi2, k=k)
            X_new = selector.fit_transform(X, y)
            selected_columns = X.columns[selector.get_support()]
            return selected_columns

        else:
            raise ValueError(f"Invalid method '{method}'. Choose from 'rfe', 'importance', 'correlation', 'selectkbest'.")


    # ----------- Full Analysis Method -----------

    def run_full_analysis(self, target = None) -> dict:
        """
        Run a complete analysis pipeline, including missing value analysis, outlier detection, 
        basic statistics calculation, normality tests, and model fitting and evaluation (if a target is provided).

        This function performs the following analyses:
        - Missing value analysis
        - Outlier detection and handling
        - Calculation of basic statistics for numerical columns
        - Normality tests (skewness and kurtosis)
        If a target column is provided, the function will also fit and evaluate machine learning models to predict the target variable.

        Parameters
        ----------
        target : str, optional
            The name of the target column for modeling. If provided, the function will also fit and evaluate models on the data.

        Returns
        -------
        dict
            A dictionary containing the results of the different analyses:

            - **'missing_analysis'**: Results of the missing value analysis
            - **'outliers_analysis'**: Results of the outlier detection and handling
            - **'basic_stats'**: Basic statistics for numerical columns
            - **'normality_tests'**: Results of normality tests (skewness and kurtosis)
            - **'model_accuracy'** (optional): Model evaluation results (if target is provided), containing accuracy for classification or MSE for regression models.
            
        Raises
        ------
        ValueError
            If the target column is not specified when running the analysis.
        TypeError
            If the target column is not a string.

        Notes
        -----
        - If no target column is specified, the function performs only the exploratory data analysis steps: missing value analysis, outlier detection, basic stats, and normality tests.
        - If a target column is provided, the function will perform model training and evaluation (regression or classification) based on the target type.
        """
        warnings.filterwarnings("ignore")

        if target is None:
            raise ValueError("Target column must be specified for full analysis.")

        if not isinstance(target, str):
            raise TypeError("Target column must be a string value.")
        

        results = {
            'missing_analysis': self.analyze_missing(),
            'outliers_analysis': self.handle_outliers(),
            'basic_stats': self.get_basic_stats(),
            'normality_tests': self.normality_test(),
            'select_features': self.select_features(target),
            'feature_importance': self.simple_feature_importance(target)
        }

        if target:
            results['model_accuracy'] = self.run_models(target)
        return results