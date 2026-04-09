import pandas as pd
from sqlalchemy import create_engine
from dotenv import load_dotenv
import os


load_dotenv() # carga las variables del archivo .env

def get_engine():
    return create_engine(os.getenv("DATABASE_URL")) # crea la conexión a la base de datos usando el string de neon

def extract_from_csv(path: str) -> pd.DataFrame: # lee el dataset desde un archivo csv local
    print(f"[extract] leyendo archivo: {path}")
    df = pd.read_csv(path)
    print(f"[extract] filas cargadas: {len(df)}")
    return df

def load_raw_to_db(df: pd.DataFrame): # sube el dataframe crudo a la tabla raw_data en la base de datos
    engine = get_engine()
    df.to_sql("raw_data", engine, if_exists="replace", index=False) # if_exists="replace" borra y recarga la tabla si ya existe
    print(f"[extract] datos cargados en raw_data: {len(df)} filas")

def extract_from_db() -> pd.DataFrame:
    # extrae los datos crudos desde la base para pasarlos al transform
    engine = get_engine()
    df = pd.read_sql("SELECT * FROM raw_data", engine)
    print(f"[extract] extraído de db: {len(df)} filas")
    return df