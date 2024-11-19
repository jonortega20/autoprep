# Data Processing Library

A Python library for automated data preprocessing, exploratory analysis, and basic model testing.

## Features

- Automated data preprocessing
- Missing value analysis and handling
- Outlier detection
- Basic statistical analysis
- Simple model testing capabilities

## Installation

```bash
pip install .
```

## Quick Start

```python
from data_processing import DataProcessing
import pandas as pd

# Load your data
df = pd.DataFrame(...)  # Your data here

# Initialize the AutoPrep object
dp = AutoPrep(df)

# Run full analysis
results = dp.run_full_analysis(target='target_column')
```

## Documentation

For detailed documentation, please refer to the docstrings in the code or build the Sphinx documentation:

```bash
cd docs
make html
```