import sys
import os
import logging
from pathlib import Path

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) # agrega la raíz del proyecto al path

from etl.extract import extract_from_csv, load_raw_to_db, extract_from_db
from etl.transform import transform
from etl.load import load_to_curated
from model.train import train_model


LOGS_DIR = Path(__file__).resolve().parent.parent / "logs"
LOG_FILE = LOGS_DIR / "pipeline.log"


def setup_logger():
    LOGS_DIR.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger("pipeline")
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

        file_handler = logging.FileHandler(LOG_FILE, encoding="utf-8")
        file_handler.setFormatter(formatter)

        stream_handler = logging.StreamHandler(sys.stdout)
        stream_handler.setFormatter(formatter)

        logger.addHandler(file_handler)
        logger.addHandler(stream_handler)

    return logger


def print_header(title: str):
    print("\n" + "=" * 50)
    print(title)
    print("=" * 50)

def run(csv_path: str):
    logger = setup_logger()
    print_header("iniciando pipeline")
    print(f"[pipeline] archivo fuente: {csv_path}")
    logger.info("inicio de pipeline | archivo=%s", csv_path)

    print("\n[paso 1] extracción")
    logger.info("paso 1 | extraccion")
    df_raw = extract_from_csv(csv_path) # lee el csv
    load_raw_to_db(df_raw) # sube a raw_data en neon

    print("\n[paso 2] transformación")
    logger.info("paso 2 | transformacion")
    df_from_db = extract_from_db() # extrae desde raw_data
    df_clean = transform(df_from_db) # limpia y hace feature engineering

    print("\n[paso 3] carga")
    logger.info("paso 3 | carga")
    load_to_curated(df_clean) # guarda en curated_data

    print("\n[paso 4] entrenamiento")
    logger.info("paso 4 | entrenamiento")
    metrics = train_model() # entrena el modelo con los datos curados, serializa el modelo en /model/artifacts
    print(
        "[pipeline] modelo serializado y listo para prediccion | "
        f"accuracy={metrics['accuracy']} | recall={metrics['recall']} | roc_auc={metrics['roc_auc']}"
    )
    logger.info(
        "modelo entrenado | accuracy=%s | recall=%s | roc_auc=%s",
        metrics["accuracy"],
        metrics["recall"],
        metrics["roc_auc"],
    )

    print_header("pipeline completado")
    logger.info("pipeline completado")
    logger.info("log guardado en %s", LOG_FILE)

if __name__ == "__main__":
    run(sys.argv[1])
