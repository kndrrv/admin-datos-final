import os
import sys

import streamlit as st


sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.predict import predict_patient


st.set_page_config(page_title="Prediccion Cardiaca", page_icon=":bar_chart:", layout="centered")

st.title("Prediccion Cardiaca")
st.write("Ingresa los datos del paciente para obtener una prediccion con el modelo serializado.")

with st.form("prediction_form"):
    age = st.number_input("Edad", min_value=1, max_value=120, value=57, step=1)
    sex = st.selectbox("Sexo", options=[0, 1], format_func=lambda x: "Femenino" if x == 0 else "Masculino")
    cp = st.selectbox("Chest Pain Type (cp)", options=[0, 1, 2, 3], index=2)
    trestbps = st.number_input("Presion arterial en reposo (trestbps)", min_value=50, max_value=250, value=130, step=1)
    chol = st.number_input("Colesterol (chol)", min_value=50, max_value=700, value=236, step=1)
    fbs = st.selectbox("Azucar en sangre en ayunas > 120 mg/dl (fbs)", options=[0, 1])
    restecg = st.selectbox("Resultado ECG en reposo (restecg)", options=[0, 1, 2], index=1)
    thalach = st.number_input("Frecuencia cardiaca maxima (thalach)", min_value=50, max_value=250, value=174, step=1)
    exang = st.selectbox("Angina inducida por ejercicio (exang)", options=[0, 1])
    oldpeak = st.number_input("Oldpeak", min_value=0.0, max_value=10.0, value=0.0, step=0.1)
    slope = st.selectbox("Slope", options=[0, 1, 2], index=2)
    ca = st.selectbox("Numero de vasos principales (ca)", options=[0, 1, 2, 3, 4], index=1)
    thal = st.selectbox("Thal", options=[0, 1, 2, 3], index=2)

    submitted = st.form_submit_button("Predecir")


if submitted:
    try:
        result = predict_patient(
            age=int(age),
            sex=int(sex),
            cp=int(cp),
            trestbps=int(trestbps),
            chol=int(chol),
            fbs=int(fbs),
            restecg=int(restecg),
            thalach=int(thalach),
            exang=int(exang),
            oldpeak=float(oldpeak),
            slope=int(slope),
            ca=int(ca),
            thal=int(thal),
        )

        prediction_label = "Positivo" if result["prediction"] == 1 else "Negativo"

        st.subheader("Resultado")
        st.write(f"Prediccion: `{prediction_label}`")
        st.write(f"Probabilidad estimada: `{result['probability']}`")
        st.write(f"Threshold aplicado: `{result['threshold']}`")
    except Exception as error:
        st.error(f"No fue posible generar la prediccion: {error}")
