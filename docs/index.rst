Autoprep documentation: the automated data preprocessing library
================================================================

A comprehensive Python library for automated data preprocessing, exploratory analysis, and basic model testing.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   introduction
   usage
   api_reference
   requirements



Quick Start
===========

.. code-block:: python

   from autoprep.autoprep import AutoPrep
   
   # Initialize with your DataFrame
   dp = AutoPrep(df)
   
   # Run complete analysis
   results = dp.run_full_analysis(target='target_column')

More info in the Usage section.



Indices and Tables:
===================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`