"""Test script for the AutoPrep library."""

import unittest
import numpy as np
from autoprep.autoprep import AutoPrep

# class TestAutoPrep(unittest.TestCase):

#     def create_test_data(self):
#         """Create a simple DataFrame-like object for testing."""
#         class TestDF:
#             def __init__(self):
#                 self.data = np.random.randn(100, 4)
#                 self.columns = ['A', 'B', 'C', 'target']
#                 self.index = range(100)
#                 self.values = self.data
                
#             def copy(self):
#                 return TestDF()
                
#             def select_dtypes(self, include):
#                 return self
                
#             def isnull(self):
#                 return np.zeros_like(self.data, dtype=bool)
                
#             def dropna(self):
#                 return self.values[:, 0]
                
#             def drop(self, columns):
#                 return self

#         return TestDF()

#     def test_data_processing(self):
#         """Test the main functionality of the library."""
#         try:
#             # Create test data
#             df = self.create_test_data()
            
#             # Initialize DataProcessing
#             dp = AutoPrep(df)
            
#             # Run full analysis
#             results = dp.run_full_analysis(target='target')
            
#             print("Test Results:")
#             print("-" * 50)
#             print("Missing Analysis:", results['missing_analysis'])
#             print("Outliers Analysis:", results['outliers_analysis'])
#             print("Basic Stats:", results['basic_stats'])
#             print("Normality Tests:", results['normality_tests'])
#             print("Data Split Info:", results['data_split'])
#             print("-" * 50)
#             print("All tests passed successfully!")
            
#         except Exception as e:
#             print(f"Test failed with error: {str(e)}")
import pandas as pd
import numpy as np

# Código de tu clase AutoPrep (asegúrate de que ya esté definida aquí o importada)

if __name__ == "__main__":
    #Crear un DataFrame de prueba
    data = {
        'A': [1, 2, np.nan, 4, 5],
        'B': [np.nan, 2, 3, np.nan, 5],
        'C': [1, 2, 3, 4, 5],
        'D': [np.nan, np.nan, np.nan, 4, 5],
        'E': ['AITOR', np.nan, 'MIKEL', np.nan, 'JON'],
        'TARGET': [1, 0, 1, 0, 1]
    }

    #data = pd.read_csv('C:\Users\usuario\OneDrive - Mondragon Unibertsitatea\Escritorio\BDATA\BDATA4\Programacion\autoprep\src\medical_sutents_dataset.csv')
    df = pd.DataFrame(data)

    # Crear una instancia de AutoPrep con el DataFrame
    processor = AutoPrep(df)
    print(' ')

    # Llamar al método analyze_missing y mostrar el resultado
    missing_analysis = processor.analyze_missing(threshold=0.9, impute_strategy="mean")
    print("ANALISIS VALORES FALTANTES:", missing_analysis)
    print(' ')

    # Llamar al método handle_outliers y mostrar el resultado
    outliers_analysis = processor.handle_outliers()
    print("ANALISIS OUTLIERS:", outliers_analysis)
    print('')

    # Llamar al método get_basic_stats y mostrar el resultado
    basic_stats = processor.get_basic_stats()
    print("ESTADISTICAS BASICAS:", basic_stats)
    print(' ')

    # Llamar al método normality_test y mostrar el resultado
    normality_tests = processor.normality_test()
    print("PRUEBAS DE NORMALIDAD:", normality_tests)
    print(' ')

    # Para run_models y run_full_analysis, debes definir el nombre de la columna target si deseas correr modelos de prueba
    trained_model = processor.run_models(target="TARGET")
    print("ENTRENAMIENTO DE MODELO:", trained_model)
    print(' ')

    # Si deseas ejecutar run_full_analysis sin target, puedes hacerlo así:
    full_analysis = processor.run_full_analysis()
    print("ANALISIS COMPLETO:", full_analysis)
    print(' ')
