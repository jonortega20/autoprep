"""Module for exploratory data analysis operations."""

from typing import Optional, List, Dict, Any
import numpy as np
from .utils import setup_logger, validate_dataframe

logger = setup_logger(__name__)

class ExploratoryAnalysis:
    """
    Class for performing exploratory data analysis.

    Parameters
    ----------
    df : Any
        DataFrame-like object to analyze.
    """

    def __init__(self, df: Any):
        """Initialize ExploratoryAnalysis with input DataFrame."""
        if not validate_dataframe(df):
            raise ValueError("Input must be a DataFrame-like object")
        self.df = df

    def get_basic_stats(self) -> Dict[str, Any]:
        """
        Calculate basic statistics for numerical columns.

        Returns
        -------
        Dict[str, Any]
            Dictionary containing basic statistics for each numerical column.
        """
        try:
            stats = {}
            numeric_cols = self.df.select_dtypes(include=[np.number]).columns
            
            for col in numeric_cols:
                col_stats = {
                    'mean': float(self.df[col].mean()),
                    'median': float(self.df[col].median()),
                    'std': float(self.df[col].std()),
                    'min': float(self.df[col].min()),
                    'max': float(self.df[col].max())
                }
                stats[col] = col_stats
                
            return stats
            
        except Exception as e:
            logger.error(f"Error in get_basic_stats: {str(e)}")
            raise

    def normality_test(self, columns: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Perform basic normality tests on numerical columns.

        Parameters
        ----------
        columns : List[str], optional
            List of columns to test. If None, tests all numerical columns.

        Returns
        -------
        Dict[str, Any]
            Dictionary containing normality test results.
        """
        try:
            if columns is None:
                columns = self.df.select_dtypes(include=[np.number]).columns
                
            results = {}
            
            for col in columns:
                data = self.df[col].dropna()
                
                # Simple normality check using skewness and kurtosis
                skew = float(data.skew())
                kurt = float(data.kurtosis())
                
                results[col] = {
                    'skewness': skew,
                    'kurtosis': kurt,
                    'is_normal': abs(skew) < 2 and abs(kurt) < 7
                }
                
            return results
            
        except Exception as e:
            logger.error(f"Error in normality_test: {str(e)}")
            raise