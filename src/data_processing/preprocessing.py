"""Module for data preprocessing operations."""

import numpy as np
from typing import Optional, Union, List, Dict, Any
from .utils import setup_logger, validate_dataframe

logger = setup_logger(__name__)

class Preprocessing:
    """
    Class for handling data preprocessing operations.

    Parameters
    ----------
    df : Any
        DataFrame-like object containing the data to process.
    """

    def __init__(self, df: Any):
        """Initialize Preprocessing with input DataFrame."""
        if not validate_dataframe(df):
            raise ValueError("Input must be a DataFrame-like object")
        self.df = df
        self.original_df = df.copy()
        
    def analyze_missing(self, threshold: float = 0.9,
                       impute_strategy: str = "mean") -> Dict[str, Any]:
        """
        Analyze and handle missing values in the DataFrame.

        Parameters
        ----------
        threshold : float, optional
            Maximum ratio of missing values allowed, by default 0.9
        impute_strategy : str, optional
            Strategy for imputing missing values, by default "mean"

        Returns
        -------
        Dict[str, Any]
            Dictionary containing missing value analysis results.

        Raises
        ------
        ValueError
            If threshold is not between 0 and 1.
        """
        if not 0 <= threshold <= 1:
            raise ValueError("Threshold must be between 0 and 1")

        try:
            missing_stats = {
                'total_missing': self.df.isnull().sum().sum(),
                'columns_with_missing': self.df.isnull().sum().to_dict(),
                'missing_ratio': self.df.isnull().mean().to_dict()
            }
            
            # Remove columns with too many missing values
            cols_to_drop = [
                col for col, ratio in missing_stats['missing_ratio'].items()
                if ratio > threshold
            ]
            
            if cols_to_drop:
                self.df = self.df.drop(columns=cols_to_drop)
                missing_stats['dropped_columns'] = cols_to_drop
            
            return missing_stats
            
        except Exception as e:
            logger.error(f"Error in analyze_missing: {str(e)}")
            raise

    def handle_outliers(self, columns: Optional[List[str]] = None,
                       method: str = "zscore",
                       threshold: float = 3.0) -> Dict[str, Any]:
        """
        Detect and handle outliers in specified columns.

        Parameters
        ----------
        columns : List[str], optional
            List of columns to check for outliers
        method : str, optional
            Method to detect outliers ("zscore" or "iqr")
        threshold : float, optional
            Threshold for outlier detection

        Returns
        -------
        Dict[str, Any]
            Dictionary containing outlier analysis results.
        """
        try:
            if columns is None:
                columns = self.df.select_dtypes(include=[np.number]).columns
            
            outliers_stats = {}
            
            for col in columns:
                if method == "zscore":
                    z_scores = np.abs((self.df[col] - self.df[col].mean()) 
                                    / self.df[col].std())
                    outliers = (z_scores > threshold).sum()
                elif method == "iqr":
                    Q1 = self.df[col].quantile(0.25)
                    Q3 = self.df[col].quantile(0.75)
                    IQR = Q3 - Q1
                    outliers = ((self.df[col] < (Q1 - 1.5 * IQR)) | 
                              (self.df[col] > (Q3 + 1.5 * IQR))).sum()
                else:
                    raise ValueError(f"Unknown method: {method}")
                
                outliers_stats[col] = outliers
                
            return outliers_stats
            
        except Exception as e:
            logger.error(f"Error in handle_outliers: {str(e)}")
            raise