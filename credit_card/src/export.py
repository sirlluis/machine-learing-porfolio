# Exportar datos
import os
from config import data_processed_path

def export_data(df, path=data_processed_path/"Credit_Card_App_PROCESSED.csv"):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)
    print(f"Datos exportados a {path}")