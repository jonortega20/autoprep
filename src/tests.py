"""Test script for the AutoPrep library."""

from autoprep_jonortega20.autoprep import AutoPrep
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

    # Descubrir la importancia de las caracteristicas en el modelo
    feature_importance = processor.simple_feature_importance(target="TARGET")
    print("IMPORTANCIA DE LAS CARACTERÍSTICAS:", feature_importance)
    print(' ')

    # Seleccionar las características más relevantes
    selected_features = processor.select_features(target="TARGET")
    print("SELECCIÓN DE CARACTERÍSTICAS MÁS RELEVANTES:", selected_features)
    print(' ')

    # Si deseas ejecutar run_full_analysis sin target, puedes hacerlo así:
    full_analysis = processor.run_full_analysis()
    print("ANALISIS COMPLETO:", full_analysis)
    print(' ')