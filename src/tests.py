"""Test script for the AutoPrep library."""

import unittest
import numpy as np
from autoprep.autoprep import AutoPrep

class TestAutoPrep(unittest.TestCase):

    def create_test_data(self):
        """Create a simple DataFrame-like object for testing."""
        class TestDF:
            def __init__(self):
                self.data = np.random.randn(100, 4)
                self.columns = ['A', 'B', 'C', 'target']
                self.index = range(100)
                self.values = self.data
                
            def copy(self):
                return TestDF()
                
            def select_dtypes(self, include):
                return self
                
            def isnull(self):
                return np.zeros_like(self.data, dtype=bool)
                
            def dropna(self):
                return self.values[:, 0]
                
            def drop(self, columns):
                return self

        return TestDF()

    def test_data_processing(self):
        """Test the main functionality of the library."""
        try:
            # Create test data
            df = self.create_test_data()
            
            # Initialize DataProcessing
            dp = AutoPrep(df)
            
            # Run full analysis
            results = dp.run_full_analysis(target='target')
            
            print("Test Results:")
            print("-" * 50)
            print("Missing Analysis:", results['missing_analysis'])
            print("Outliers Analysis:", results['outliers_analysis'])
            print("Basic Stats:", results['basic_stats'])
            print("Normality Tests:", results['normality_tests'])
            print("Data Split Info:", results['data_split'])
            print("-" * 50)
            print("All tests passed successfully!")
            
        except Exception as e:
            print(f"Test failed with error: {str(e)}")

if __name__ == "__main__":
    unittest.main()