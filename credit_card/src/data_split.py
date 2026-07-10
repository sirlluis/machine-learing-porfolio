from sklearn.model_selection import train_test_split

from config import (
    TARGET,
    TEST_SIZE,
    RANDOM_STATE
    )

def split_data(df):
    """
        Function that splits data into train and test

        Parameters
        ----------
        df : pandas.DataFrame
        
        Returns
        -------
        splittinglist, length=2 * len(arrays)
            List containing train-test split of inputs.
    """
    X=df.drop(columns=TARGET)
    y=df[TARGET]
    
    X_train, X_test, y_train, y_test=train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y
    )
    return X_train, X_test, y_train, y_test