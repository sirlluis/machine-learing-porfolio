# Carga de datos
import pandas as pd

def load_data(path):
    """
        Load dataframe from the path

        Parameters
        ----------
        str : path
            Recieve a string with the path to the dataframe 
        Returns
        -------
        df: pandas.DataFrame
    """

    df=pd.read_csv(path)

    return df