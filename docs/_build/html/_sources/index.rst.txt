Data Processing Library Documentation
=====================================

A comprehensive Python library for automated data preprocessing, exploratory analysis, and basic model testing.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   introduction
   installation
   modules/preprocessing
   modules/exploratory_analysis
   modules/model_testing
   modules/utils

Introduction
-----------

The Data Processing Library provides tools for automating common data analysis tasks, including:

* Data preprocessing and cleaning
* Missing value analysis
* Outlier detection
* Basic statistical analysis
* Model testing capabilities

Quick Start
----------

.. code-block:: python

   from data_processing import DataProcessing
   
   # Initialize with your DataFrame
   dp = DataProcessing(df)
   
   # Run complete analysis
   results = dp.run_full_analysis(target='target_column')

Core Components
-------------

DataProcessing
~~~~~~~~~~~~~

.. automodule:: data_processing.data_processing
   :members:
   :undoc-members:
   :show-inheritance:

Preprocessing
~~~~~~~~~~~~

.. automodule:: data_processing.preprocessing
   :members:
   :undoc-members:
   :show-inheritance:

Exploratory Analysis
~~~~~~~~~~~~~~~~~~

.. automodule:: data_processing.exploratory_analysis
   :members:
   :undoc-members:
   :show-inheritance:

Model Testing
~~~~~~~~~~~~

.. automodule:: data_processing.model_testing
   :members:
   :undoc-members:
   :show-inheritance:

Utilities
~~~~~~~~

.. automodule:: data_processing.utils
   :members:
   :undoc-members:
   :show-inheritance:

API Reference
------------

.. toctree::
   :maxdepth: 2

   api/data_processing
   api/preprocessing
   api/exploratory_analysis
   api/model_testing
   api/utils

Examples
--------

Missing Value Analysis
~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Analyze missing values
   missing_stats = dp.preprocessing.analyze_missing(threshold=0.9)
   print(missing_stats['total_missing'])
   print(missing_stats['columns_with_missing'])

Outlier Detection
~~~~~~~~~~~~~~~

.. code-block:: python

   # Detect outliers using z-score method
   outliers = dp.preprocessing.handle_outliers(method='zscore', threshold=3.0)
   print(outliers)

Basic Statistics
~~~~~~~~~~~~~~

.. code-block:: python

   # Get basic statistics for numerical columns
   stats = dp.exploratory_analysis.get_basic_stats()
   print(stats)

Model Testing
~~~~~~~~~~~

.. code-block:: python

   # Split data for modeling
   X_train, X_test, y_train, y_test = dp.model_testing.train_test_split(
       target='target_column',
       test_size=0.2
   )

Indices and Tables
----------------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`