"""
    Para construir el pipeline del preprocesamiento
"""
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, TargetEncoder
from  config import (
    TARGET_ENCODER_COLUMNS,
    SCALE_COLUMNS
)

def build_preprocessor():
    preprocessor=ColumnTransformer(
        transformers=[
            (
                "target_encoder",
                TargetEncoder(),
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