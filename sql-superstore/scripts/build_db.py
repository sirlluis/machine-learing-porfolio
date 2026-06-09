import sqlite3
import pandas as pd

# carga de datos
def load_data():
    print("Pandas version", pd.__version__)
    # importamos los datos
    print("Importando datos...")
    df=pd.read_csv(r"sql-superstore\data\Sample-Superstore.csv", encoding="latin1")
    print("Datos importados con exito")
    return df

def general_info(data_frame):
    data_frame.info()

def sql_conn(df):
    with sqlite3.connect("../superstore.db") as conn:
        df.to_sql("ventas", conn, if_exists="replace", index=False)
    print(f"Base de datos creada!, {len(df):,} registros listos!")

if __name__=="__main__":
    df=load_data()
    general_info(df)
    sql_conn(df)
