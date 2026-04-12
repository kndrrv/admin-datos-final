import streamlit as st
import pandas as pd
import os
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(
    page_title="Predicción de Enfermedad Cardíaca",
    layout="centered"
)

st.title("Predicción de Enfermedad Cardíaca")
st.markdown("Ingrese los datos del paciente para obtener una predicción del modelo de IA.")

# ── sidebar: estado del sistema ──────────────────────────────────────────────
with st.sidebar:
    st.header("Estado del sistema")

    db_url = os.getenv("DATABASE_URL")
    if db_url and "usuario" not in db_url:
        st.success("Base de datos conectada")
    else:
        st.warning("Sin conexión a base de datos")

    model_path = os.path.join(os.path.dirname(__file__), "..", "model", "model.pkl")
    if os.path.exists(model_path):
        st.success("Modelo cargado")
    else:
        st.warning("Modelo no disponible aún")

    st.divider()
    st.caption("Administración de datos · LEAD University")

# ── formulario de entrada ────────────────────────────────────────────────────
st.subheader("Datos del paciente")

col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Edad", min_value=1, max_value=120, value=50, step=1)
    sex = st.selectbox("Sexo", options=[0, 1], format_func=lambda x: "Femenino" if x == 0 else "Masculino")
    cp = st.selectbox(
        "Tipo de dolor de pecho (cp)",
        options=[0, 1, 2, 3],
        format_func=lambda x: {0: "0 – Asintomático", 1: "1 – Atípico", 2: "2 – No anginoso", 3: "3 – Angina típica"}[x]
    )
    trestbps = st.number_input("Presión arterial en reposo (mm Hg)", min_value=80, max_value=250, value=120, step=1)
    chol = st.number_input("Colesterol sérico (mg/dl)", min_value=100, max_value=600, value=200, step=1)
    fbs = st.selectbox(
        "Glucosa en ayunas > 120 mg/dl (fbs)",
        options=[0, 1],
        format_func=lambda x: "No" if x == 0 else "Sí"
    )
    restecg = st.selectbox(
        "Resultado ECG en reposo",
        options=[0, 1, 2],
        format_func=lambda x: {0: "0 – Normal", 1: "1 – Anormalidad ST-T", 2: "2 – Hipertrofia"}[x]
    )

with col2:
    thalach = st.number_input("Frecuencia cardíaca máxima (thalach)", min_value=60, max_value=250, value=150, step=1)
    exang = st.selectbox(
        "Angina inducida por ejercicio (exang)",
        options=[0, 1],
        format_func=lambda x: "No" if x == 0 else "Sí"
    )
    oldpeak = st.number_input("Depresión ST (oldpeak)", min_value=0.0, max_value=10.0, value=1.0, step=0.1, format="%.1f")
    slope = st.selectbox(
        "Pendiente del segmento ST",
        options=[0, 1, 2],
        format_func=lambda x: {0: "0 – Descendente", 1: "1 – Plano", 2: "2 – Ascendente"}[x]
    )
    ca = st.selectbox("Vasos principales coloreados por fluoroscopía (ca)", options=[0, 1, 2, 3])
    thal = st.selectbox(
        "Thal",
        options=[1, 2, 3],
        format_func=lambda x: {1: "1 – Normal", 2: "2 – Defecto fijo", 3: "3 – Defecto reversible"}[x]
    )

# ── boton de prediccion ──────────────────────────────────────────────────────
st.divider()

input_data = pd.DataFrame([{
    "age": age, "sex": sex, "cp": cp, "trestbps": trestbps,
    "chol": chol, "fbs": fbs, "restecg": restecg, "thalach": thalach,
    "exang": exang, "oldpeak": oldpeak, "slope": slope, "ca": ca, "thal": thal
}])

if st.button("Predecir", use_container_width=True, type="primary"):
    model_path = os.path.join(os.path.dirname(__file__), "..", "model", "model.pkl")

    if not os.path.exists(model_path):
        st.warning("El modelo aún no está disponible. Pendiente de entrenamiento.")
    else:
        try:
            import joblib
            model = joblib.load(model_path)
            prediction = model.predict(input_data)[0]
            proba = model.predict_proba(input_data)[0]

            st.subheader("Resultado")
            if prediction == 1:
                st.error(f"Alta probabilidad de enfermedad cardíaca ({proba[1]*100:.1f}%)")
            else:
                st.success(f"Baja probabilidad de enfermedad cardíaca ({proba[0]*100:.1f}%)")

            with st.expander("Ver probabilidades detalladas"):
                st.write(f"- Sin enfermedad: {proba[0]*100:.1f}%")
                st.write(f"- Con enfermedad: {proba[1]*100:.1f}%")
        except Exception as e:
            st.error(f"Error al cargar el modelo: {e}")

# ── vista previa de datos ingresados ─────────────────────────────────────────
with st.expander("Ver datos ingresados"):
    st.dataframe(input_data, use_container_width=True)
