import pandas as pd

def clean(df: pd.DataFrame) -> pd.DataFrame:
    print(f"[transform] iniciando limpieza - shape: {df.shape}")
    df = df.drop_duplicates() # elimina filas duplicadas
    print(f"[transform] después de duplicados: {df.shape}")
    df = df.dropna(subset=["target", "chol", "trestbps", "thalach"]) # elimina filas con nulos en columnas críticas
    df["ca"] = df["ca"].fillna(df["ca"].mode()[0]) # rellena nulos en ca con la moda
    df["thal"] = df["thal"].fillna(df["thal"].mode()[0]) # rellena nulos en thal con la moda
    df["ca"] = df["ca"].astype(int) # convierte ca a entero
    df["thal"] = df["thal"].astype(int) # convierte thal a entero
    df["target"] = df["target"].astype(int) # convierte target a entero
    print(f"[transform] después de limpieza: {df.shape}")
    return df

def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    print("[transform] aplicando feature engineering...")
    df["high_bp"] = (df["trestbps"] > 130).astype(int) # 1 si presión arterial alta
    df["high_chol"] = (df["chol"] > 240).astype(int) # 1 si colesterol alto
    df["low_hr"] = (df["thalach"] < 120).astype(int) # 1 si frecuencia cardíaca baja
    return df

def transform(df: pd.DataFrame) -> pd.DataFrame:
    df = clean(df) # ejecuta limpieza
    df = feature_engineering(df) # ejecuta feature engineering
    print(f"[transform] shape final: {df.shape}")
    return df