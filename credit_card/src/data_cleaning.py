import pandas as pd

def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove duplicates from dataframe
    Parameters
    ----------
    df : pd.DataFrame
        Recive a pandas dataframe

    Returns
    -------
    pd.DataFrame
        Dataframe without duplicates
    """
    return df.drop_duplicates().reset_index(drop=True)

def remove_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove missing values from dataframe
    
    Parameters
    ----------
    df : pd.Dataframe
        Recieve a pandas dataframe

    Returns
    -------
    pd.Dataframe
        Dataframe without missing values
    """
    return df.dropna().reset_index(drop=True)

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
        This function calls the functions that remove
        duplicates and missing values from data frame

        Paramteres
        ----------
        df: pd.DataFrame
            Recieve pandas dataframe
        Returns
        -------
        pd.Dataframe
            Cleaned dataframe
    """
    df=remove_duplicates(df)
    df=remove_missing_values(df)
    return df


