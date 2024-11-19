Usage Guide
===========

This guide provides usage examples for common tasks with the library, one per function available in the `AutoPrep` class.


Missing Value Analysis
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Analyze missing values
   missing_stats = dp.analyze_missing(threshold=0.9)
   print(missing_stats)

Outlier Detection
~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Detect outliers using z-score method
   outliers = dp.handle_outliers()
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
   model_results = dp.run_models(target='target_column')
   print(model_results)

Feature Importance
~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Get and display feature importance
   feature_importances = dp.simple_feature_importance(target='target_column', top_n=5)
   print(feature_importances)

Feature Selection
~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Perform feature selection using the RFE method (the default method)
   selected_features = dp.select_features(target='target_column', k=2)
   print(selected_features)

   # Perform feature selection based on SelectKBest
   selected_features_kbest = dp.select_features(method='selectkbest', target='species', k=3)
   print(selected_features_kbest)

Normality Test
~~~~~~~~~~~~~~

.. code-block:: python

   # Perform normality test for numerical columns
   normality_results = dp.normality_test()
   print(normality_results)

Full Analysis Pipeline
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Run the full analysis pipeline, including missing value analysis,
   # outlier detection, basic stats, normality tests, and model testing
   full_analysis = dp.run_full_analysis(target='target_column')
   print(full_analysis)