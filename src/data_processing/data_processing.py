"""Main module for data processing operations."""

from typing import Optional, Any
from .preprocessing import Preprocessing
from .exploratory_analysis import ExploratoryAnalysis
from .model_testing import ModelTesting
from .utils import validate_dataframe, setup_logger

logger = setup_logger(__name__)

class DataProcessing:
    """
    Main class for data processing operations.

    This class provides a unified interface for preprocessing,
    exploratory analysis, and model testing operations.

    Parameters
    ----------
    df : Any
        DataFrame-like object to process
    """

    def __init__(self, df: Any):
        """Initialize DataProcessing with input DataFrame."""
        if not validate_dataframe(df):
            raise ValueError("Input must be a DataFrame-like object")
            
        self.df = df
        self.preprocessing = Preprocessing(df)
        self.exploratory_analysis = ExploratoryAnalysis(df)
        self.model_testing = ModelTesting(df)
        
    def run_full_analysis(self, target: Optional[str] = None) -> dict:
        """
        Run a complete analysis pipeline.

        Parameters
        ----------
        target : str, optional
            Target column name for modeling

        Returns
        -------
        dict
            Dictionary containing all analysis results
        """
        try:
            results = {}
            
            # Preprocessing
            results['missing_analysis'] = self.preprocessing.analyze_missing()
            results['outliers_analysis'] = self.preprocessing.handle_outliers()
            
            # Exploratory Analysis
            results['basic_stats'] = self.exploratory_analysis.get_basic_stats()
            results['normality_tests'] = self.exploratory_analysis.normality_test()
            
            # Model Testing (if target is provided)
            if target is not None:
                X_train, X_test, y_train, y_test = self.model_testing.train_test_split(target)
                results['data_split'] = {
                    'train_size': len(X_train),
                    'test_size': len(X_test)
                }
            
            return results
            
        except Exception as e:
            logger.error(f"Error in run_full_analysis: {str(e)}")
            raise