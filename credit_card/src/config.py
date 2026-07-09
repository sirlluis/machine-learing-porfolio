# librerías
from pathlib import Path

# rutas del proyecto

# raíz
ROOT_DIR=Path(__file__).resolve().parent.parent
DATA_DIR=ROOT_DIR/"data"
RAW_DATA_DIR=DATA_DIR/"raw"
PROCESSED_DATA_DIR=DATA_DIR/"processed"
MODELS_DIR=ROOT_DIR/"models"
REPORTS_DIR=ROOT_DIR/"reports"
FIGURES_PATH=REPORTS_DIR/"figures"

# ruta a los datos RAW
DATA_PATH=RAW_DATA_DIR/"Credit_Card_Applications.csv"

# configuración de modelos
TARGET="Class"
TEST_SIZE=0.2
RANDOM_STATE=14

# columnas útiles
TARGET_ENCODER_COLUMNS=["A5", "A6", "A10", "A13", "A14"]
SCALE_COLUMNS=["A2"]