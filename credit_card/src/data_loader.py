# Carga de datos
import pandas as pd

def load_data(path):
    """
    Función que carga la base de datos
    desde un archivo CSV
    ------
    Parámetros:

    path = Path
        ruta del conjunto de datos
    ------
    Return:

    pd.DataFrame 
    """

    df=pd.read_csv(path)

    return df