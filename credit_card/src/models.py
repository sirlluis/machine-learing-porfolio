from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from config import RANDOM_STATE


def build_logistic_regression():
    return LogisticRegression(
        random_state=RANDOM_STATE,
        max_iter=1000
    )
    
def build_random_forest():
    return RandomForestClassifier(
        random_state=RANDOM_STATE
    )
    
    