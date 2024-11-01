"""Module for basic model testing operations."""

from typing import Optional, List, Dict, Any, Tuple
import numpy as np
from .utils import setup_logger, validate_dataframe
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


logger = setup_logger(__name__)

class ModelTesting:
    """
    Class for basic model testing operations.

    Parameters
    ----------
    df : Any
        DataFrame-like object containing the data to model.
    """

    def __init__(self, df: Any):
        """Initialize ModelTesting with input DataFrame."""
        if not validate_dataframe(df):
            raise ValueError("Input must be a DataFrame-like object")
        self.df = df

    def train_test_split(self, target: str,
                        test_size: float = 0.2,
                        random_state: Optional[int] = None
                        ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Split data into training and testing sets.

        Parameters
        ----------
        target : str
            Name of the target column
        test_size : float, optional
            Proportion of the dataset to include in the test split
        random_state : int, optional
            Random state for reproducibility

        Returns
        -------
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
            X_train, X_test, y_train, y_test arrays
        """
        try:
            if not 0 < test_size < 1:
                raise ValueError("test_size must be between 0 and 1")

            if target not in self.df.columns:
                raise ValueError(f"Target column '{target}' not found in DataFrame")

            # Simple random split
            if random_state is not None:
                np.random.seed(random_state)

            mask = np.random.rand(len(self.df)) >= test_size
            
            X = self.df.drop(columns=[target])
            y = self.df[target]

            X_train = X[mask]
            X_test = X[~mask]
            y_train = y[mask]
            y_test = y[~mask]

            return X_train.values, X_test.values, y_train.values, y_test.values

        except Exception as e:
            logger.error(f"Error in train_test_split: {str(e)}")
            raise

    def run_models(self, target: str) -> Dict[str, float]:
        """
        Fit and evaluate multiple models on the data.

        Parameters
        ----------
        target : str
            Name of the target column

        Returns
        -------
        Dict[str, float]
            Dictionary of model names and accuracy scores
        """
        try:
            X_train, X_test, y_train, y_test = self.train_test_split(target)
            models = {
                "RandomForest": RandomForestClassifier(),
                "LogisticRegression": LogisticRegression()
            }
            scores = {}

            for name, model in models.items():
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                scores[name] = accuracy_score(y_test, y_pred)

            return scores

        except Exception as e:
            logger.error(f"Error in run_models: {str(e)}")
            raise