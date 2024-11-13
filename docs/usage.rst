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

   # Run models and evaluate performance
   model_scores = dp.run_models(target='target_column')
   print(model_scores)

Feature Importance
~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Get and display feature importance
   feature_importances = dp.simple_feature_importance(target='target_column', top_n=10)
   print(feature_importances)

Feature Selection
~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Perform feature selection using the RFE method
   selected_features = dp.select_features(method='rfe', target='target_column', k=5)
   print(selected_features)

   # Perform feature selection based on correlation
   selected_features_corr = dp.select_features(method='correlation', target='target_column', threshold=0.9)
   print(selected_features_corr)

Normality Test
~~~~~~~~~~~~~~

.. code-block:: python

   # Perform normality test for numerical columns
   normality_results = dp.normality_test(columns=['col1', 'col2'])
   print(normality_results)

Full Analysis Pipeline
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Run the full analysis pipeline, including missing value analysis,
   # outlier detection, basic stats, normality tests, and model testing
   full_analysis = dp.run_full_analysis(target='target_column')
   print(full_analysis)