import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) # agrega la raíz del proyecto al path

from etl.extract import extract_from_csv, load_raw_to_db, extract_from_db
from etl.transform import transform
from etl.load import load_to_curated

def run(csv_path: str):
    print("=" * 50)
    print("iniciando pipeline")
    print("=" * 50)

    print("\n[paso 1] extracción")
    df_raw = extract_from_csv(csv_path) # lee el csv
    load_raw_to_db(df_raw) # sube a raw_data en neon

    print("\n[paso 2] transformación")
    df_from_db = extract_from_db() # extrae desde raw_data
    df_clean = transform(df_from_db) # limpia y hace feature engineering

    print("\n[paso 3] carga")
    load_to_curated(df_clean) # guarda en curated_data

    print("\n" + "=" * 50)
    print("pipeline completado")
    print("=" * 50)

if __name__ == "__main__":
    run(sys.argv[1]) # recibe la ruta del csv como argumento