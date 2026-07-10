from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, TargetEncoder
from  config import (
    TARGET_ENCODER_COLUMNS,
    SCALE_COLUMNS,
    RANDOM_STATE
)

def build_preprocessor():
    """
        This function builds a preprocessing pipeline
        usign sklearn.ColumnTransformer to transform 
        specific columns

        Parameters
        ----------
        Set of transformers like: sklearn.TargetEncoder and sklearn.StandardScaler
        applied to specific columns

        Returns
        -------
        ColumnTransformer pipeline
    """
    preprocessor=ColumnTransformer(
        transformers=[
            (
                "target_encoder",
                TargetEncoder(random_state=RANDOM_STATE),
                TARGET_ENCODER_COLUMNS
            ),
            (
                "scaler",
                StandardScaler(),
                SCALE_COLUMNS
            )
        ],
        remainder="passthrough"
    )
    return preprocessor