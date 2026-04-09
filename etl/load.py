import pandas as pd
from sqlalchemy import create_engine
from dotenv import load_dotenv
import os

load_dotenv() # carga las variables del archivo .env

def get_engine():
    return create_engine(os.getenv("DATABASE_URL")) # crea la conexión a neon usando el string del .env

def load_to_curated(df: pd.DataFrame):
    engine = get_engine()
    df.to_sql("curated_data", engine, if_exists="replace", index=False) # if_exists="replace" borra y recarga la tabla si ya existe
    print(f"[load] datos guardados en curated_data: {len(df)} filas")