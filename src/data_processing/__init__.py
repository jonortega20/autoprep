"""
Data Processing Library for automated preprocessing and analysis.

This library provides tools for data preprocessing, exploratory analysis,
and basic model testing with a focus on automation and ease of use.
"""

from .preprocessing import Preprocessing
from .exploratory_analysis import ExploratoryAnalysis
from .model_testing import ModelTesting
from .data_processing import DataProcessing

__version__ = '0.1.0'
__all__ = ['DataProcessing', 'Preprocessing', 'ExploratoryAnalysis', 'ModelTesting']