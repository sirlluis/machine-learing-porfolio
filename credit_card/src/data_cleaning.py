"""
    Funiones para limpiar los datos crudos
"""

import pandas as pd

# Elimina duplicados
def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    return df.drop_duplicates().reset_index(drop=True)

# Elimina las filas donde faltan datos
def remove_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    return df.dropna().reset_index(drop=True)

# orquestador de las funciones
def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    df=remove_duplicates(df)
    df=remove_missing_values(df)
    return df


