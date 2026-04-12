import argparse
import json
import os
from pathlib import Path
from typing import Optional, List

import joblib
import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


load_dotenv()

ARTIFACTS_DIR = Path(__file__).resolve().parent / "artifacts"
MODEL_PATH = ARTIFACTS_DIR / "mlp_classifier.joblib"
METRICS_PATH = ARTIFACTS_DIR / "metrics.json"


def get_engine():
    database_url = os.getenv("DATABASE_URL")
    if not database_url:
        raise ValueError("DATABASE_URL no está configurada en el entorno.")
    return create_engine(database_url)


def load_curated_data(table_name: str = "curated_data") -> pd.DataFrame:
    print(f"[train] leyendo datos desde la tabla {table_name}...")
    engine = get_engine()
    df = pd.read_sql(f"SELECT * FROM {table_name}", engine)
    print(f"[train] datos cargados desde {table_name}: {len(df)} filas")
    return df


def prepare_features(df: pd.DataFrame):
    if "target" not in df.columns:
        raise ValueError("La tabla curated_data debe incluir la columna 'target'.")

    drop_columns = ["target", "id", "processed_at", "loaded_at"]
    feature_columns = [column for column in df.columns if column not in drop_columns]

    if not feature_columns:
        raise ValueError("No se encontraron columnas predictoras para entrenar el modelo.")

    X = df[feature_columns].copy()
    y = df["target"].astype(int)
    return X, y, feature_columns


def build_pipeline() -> Pipeline:
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            (
                "model",
                MLPClassifier(
                    hidden_layer_sizes=(32, 16),
                    activation="relu",
                    solver="adam",
                    alpha=0.0001,
                    batch_size=32,
                    learning_rate_init=0.001,
                    max_iter=500,
                    random_state=42,
                    early_stopping=True,
                    validation_fraction=0.1,
                    n_iter_no_change=20,
                ),
            ),
        ]
    )


def get_train_test_split(X: pd.DataFrame, y: pd.Series):
    print("[train] separando datos en entrenamiento y prueba...")
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )
    print(f"[train] train shape: {X_train.shape}")
    print(f"[train] test shape: {X_test.shape}")
    return X_train, X_test, y_train, y_test


def evaluate_model(model: Pipeline, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
    print("[train] evaluando modelo en el conjunto de prueba...")
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": round(accuracy_score(y_test, y_pred), 4),
        "recall": round(recall_score(y_test, y_pred), 4),
        "roc_auc": round(roc_auc_score(y_test, y_prob), 4),
        "classification_report": classification_report(y_test, y_pred, output_dict=True),
    }

    print(f"[train] accuracy: {metrics['accuracy']}")
    print(f"[train] recall: {metrics['recall']}")
    print(f"[train] roc_auc: {metrics['roc_auc']}")
    print("[train] classification_report:")
    print(classification_report(y_test, y_pred))
    return metrics


def evaluate_thresholds(
    model: Pipeline,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    thresholds: Optional[List[float]] = None,
) -> dict:
    if thresholds is None:
        thresholds = [0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6]

    print("[threshold] evaluando distintos thresholds de probabilidad...")
    y_prob = model.predict_proba(X_test)[:, 1]
    roc_auc = round(roc_auc_score(y_test, y_prob), 4)
    results = []

    for threshold in thresholds:
        y_pred = (y_prob >= threshold).astype(int)
        result = {
            "threshold": threshold,
            "accuracy": round(accuracy_score(y_test, y_pred), 4),
            "precision": round(precision_score(y_test, y_pred, zero_division=0), 4),
            "recall": round(recall_score(y_test, y_pred, zero_division=0), 4),
            "roc_auc": roc_auc,
        }
        results.append(result)
        print(
            f"[threshold] t={threshold:.2f} | "
            f"accuracy={result['accuracy']:.4f} | "
            f"precision={result['precision']:.4f} | "
            f"recall={result['recall']:.4f} | "
            f"roc_auc={result['roc_auc']:.4f}"
        )

    best_threshold = max(results, key=lambda item: (item["recall"], item["accuracy"]))
    print(
        f"[threshold] mejor threshold por recall: {best_threshold['threshold']:.2f} | "
        f"recall={best_threshold['recall']:.4f} | "
        f"accuracy={best_threshold['accuracy']:.4f} | "
        f"precision={best_threshold['precision']:.4f}"
    )
    return {
        "best_threshold": best_threshold,
        "all_thresholds": results,
    }


def save_artifacts(
    model: Pipeline,
    feature_columns: list[str],
    metrics: dict,
    model_path: Path = MODEL_PATH,
    metrics_path: Path = METRICS_PATH,
):
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    print(f"[train] guardando artefactos en {ARTIFACTS_DIR}...")
    joblib.dump({"model": model, "features": feature_columns}, model_path)

    with open(metrics_path, "w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2)

    print(f"[train] modelo guardado en: {model_path}")
    print(f"[train] metricas guardadas en: {metrics_path}")


def train_model(table_name: str = "curated_data") -> dict:
    print("[train] iniciando entrenamiento base con hiperparametros estandar...")
    df = load_curated_data(table_name)
    X, y, feature_columns = prepare_features(df)

    print(f"[train] columnas usadas: {feature_columns}")
    X_train, X_test, y_train, y_test = get_train_test_split(X, y)

    model = build_pipeline()
    print("[train] entrenando MLPClassifier base...")
    model.fit(X_train, y_train)

    metrics = evaluate_model(model, X_test, y_test)
    save_artifacts(model, feature_columns, metrics)
    return metrics


def train_and_evaluate_thresholds(table_name: str = "curated_data") -> dict:
    print("[threshold] entrenando modelo base y comparando thresholds...")
    df = load_curated_data(table_name)
    X, y, feature_columns = prepare_features(df)

    print(f"[threshold] columnas usadas: {feature_columns}")
    X_train, X_test, y_train, y_test = get_train_test_split(X, y)

    model = build_pipeline()
    print("[threshold] entrenando MLPClassifier base...")
    model.fit(X_train, y_train)

    base_metrics = evaluate_model(model, X_test, y_test)
    threshold_metrics = evaluate_thresholds(model, X_test, y_test)

    metrics = {
        "base_metrics": base_metrics,
        "threshold_analysis": threshold_metrics,
    }

    save_artifacts(model, feature_columns, metrics)
    return metrics


def parse_args():
    parser = argparse.ArgumentParser(description="Entrena el MLPClassifier final del proyecto.")
    parser.add_argument(
        "--table",
        default="curated_data",
        help="Nombre de la tabla desde la que se leeran los datos.",
    )
    parser.add_argument(
        "--thresholds",
        action="store_true",
        help="Entrena el modelo base y compara varios thresholds de probabilidad.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.thresholds:
        train_and_evaluate_thresholds(table_name=args.table)
    else:
        train_model(table_name=args.table)
