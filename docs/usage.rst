Usage Guide
===========

This guide provides usage examples for common tasks with the library, one per function available in the `AutoPrep` class.


Missing Value Analysis
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Analyze missing values
   missing_stats = dp.analyze_missing(threshold=0.9)
   print(missing_stats['total_missing'])
   print(missing_stats['columns_with_missing'])

Outlier Detection
~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Detect outliers using z-score method
   outliers = dp.handle_outliers(method='zscore', threshold=3.0)
   print(outliers)

Basic Statistics
~~~~~~~~~~~~~~~~

.. code-block:: python

   # Get basic statistics for numerical columns
   stats = dp.get_basic_stats()
   print(stats)

Model Testing
~~~~~~~~~~~~~

.. code-block:: python

   # Split data for modeling
   X_train, X_test, y_train, y_test = dp.train_test_split(
       target='target_column',
       test_size=0.2
   )