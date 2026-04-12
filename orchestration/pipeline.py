import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) # agrega la raíz del proyecto al path

from etl.extract import extract_from_csv, load_raw_to_db, extract_from_db
from etl.transform import transform
from etl.load import load_to_curated
from model.train import train_model


def print_header(title: str):
    print("\n" + "=" * 50)
    print(title)
    print("=" * 50)

def run(csv_path: str):
    print_header("iniciando pipeline")
    print(f"[pipeline] archivo fuente: {csv_path}")

    print("\n[paso 1] extracción")
    df_raw = extract_from_csv(csv_path) # lee el csv
    load_raw_to_db(df_raw) # sube a raw_data en neon

    print("\n[paso 2] transformación")
    df_from_db = extract_from_db() # extrae desde raw_data
    df_clean = transform(df_from_db) # limpia y hace feature engineering

    print("\n[paso 3] carga")
    load_to_curated(df_clean) # guarda en curated_data

    print("\n[paso 4] entrenamiento")
    metrics = train_model() # entrena el modelo con los datos curados y guarda los artefactos
    print(
        "[pipeline] modelo serializado y listo para prediccion | "
        f"accuracy={metrics['accuracy']} | recall={metrics['recall']} | roc_auc={metrics['roc_auc']}"
    )

    print_header("pipeline completado")

if __name__ == "__main__":
    run(sys.argv[1]) # recibe la ruta del csv como argumento
