"""Utility functions for data processing operations."""

import logging
from typing import Any, List, Union, Dict
import numpy as np

def setup_logger(name: str) -> logging.Logger:
    """
    Set up a logger with a specific name and configuration.

    Parameters
    ----------
    name : str
        Name for the logger instance.

    Returns
    -------
    logging.Logger
        Configured logger instance.
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger

def validate_dataframe(df: Any) -> bool:
    """
    Validate if the input is a proper DataFrame-like object.

    Parameters
    ----------
    df : Any
        Object to validate.

    Returns
    -------
    bool
        True if object has required DataFrame attributes.

    Raises
    ------
    ValueError
        If input doesn't meet DataFrame requirements.
    """
    required_attrs = ['columns', 'index', 'values']
    
    try:
        return all(hasattr(df, attr) for attr in required_attrs)
    except Exception:
        raise ValueError("Input must be a DataFrame-like object")