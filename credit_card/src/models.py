from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from config import RANDOM_STATE


def build_logistic_regression():
    """
        Set logisitic regression base model

        Returns
        -------
        Logisitic regresion base model
    """
    return LogisticRegression(
        random_state=RANDOM_STATE,
        max_iter=1000
    )
    
def build_random_forest():
    """
        Set random forest base model

        Returns
        -------
        Random forest base model
    """
    return RandomForestClassifier(
        random_state=RANDOM_STATE
    )
    
    