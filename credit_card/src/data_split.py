"""
    Funciones para seprar y preparar los datos
    para el entrenamiento
"""

from sklearn.model_selection import train_test_split

from config import (
    TARGET,
    TEST_SIZE,
    RANDOM_STATE
    )

def split_data(df):
    """
        Función que separará los datos en entrenamiento y prueba
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