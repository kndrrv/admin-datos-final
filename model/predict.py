from pathlib import Path

import joblib
import pandas as pd


MODEL_PATH = Path(__file__).resolve().parent / "artifacts" / "mlp_classifier.joblib"
DEFAULT_THRESHOLD = 0.35


def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"No se encontró el modelo serializado en {MODEL_PATH}. Ejecuta train.py primero."
        )

    artifact = joblib.load(MODEL_PATH)
    return artifact["model"], artifact["features"]


def build_model_input(input_data: dict) -> dict:
    model_input = dict(input_data)
    model_input["high_bp"] = int(model_input["trestbps"] > 130)
    model_input["high_chol"] = int(model_input["chol"] > 240)
    model_input["low_hr"] = int(model_input["thalach"] < 120)
    return model_input


def predict(input_data: dict, threshold: float = DEFAULT_THRESHOLD) -> dict:
    model, feature_columns = load_model()
    model_input = build_model_input(input_data)
    df = pd.DataFrame([model_input]).reindex(columns=feature_columns)

    probability = model.predict_proba(df)[0][1]
    prediction = int(probability >= threshold)

    return {
        "prediction": prediction,
        "probability": round(float(probability), 4),
        "threshold": threshold,
    }


def predict_patient(
    age: int,
    sex: int,
    cp: int,
    trestbps: int,
    chol: int,
    fbs: int,
    restecg: int,
    thalach: int,
    exang: int,
    oldpeak: float,
    slope: int,
    ca: int,
    thal: int,
    threshold: float = DEFAULT_THRESHOLD,
) -> dict:
    input_data = {
        "age": age,
        "sex": sex,
        "cp": cp,
        "trestbps": trestbps,
        "chol": chol,
        "fbs": fbs,
        "restecg": restecg,
        "thalach": thalach,
        "exang": exang,
        "oldpeak": oldpeak,
        "slope": slope,
        "ca": ca,
        "thal": thal,
    }
    return predict(input_data, threshold=threshold)


if __name__ == "__main__":
    sample = {
        "age": 57,
        "sex": 1,
        "cp": 2,
        "trestbps": 130,
        "chol": 236,
        "fbs": 0,
        "restecg": 1,
        "thalach": 174,
        "exang": 0,
        "oldpeak": 0.0,
        "slope": 2,
        "ca": 1,
        "thal": 2,
    }

    print("[predict] resultado:", predict(sample))
