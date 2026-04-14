import sys
import os
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

load_dotenv()

sys.path.append(str(Path(__file__).resolve().parent.parent))

from model.predict import predict_patient, MODEL_PATH

st.set_page_config(
    page_title="Predicción de Enfermedad Cardíaca",
    layout="centered"
)

# ── encabezado ───────────────────────────────────────────────────────────────
st.title("Predicción de Enfermedad Cardíaca")
st.markdown(
    "Esta herramienta utiliza un modelo de inteligencia artificial entrenado con datos clínicos "
    "para estimar la probabilidad de que un paciente tenga enfermedad cardíaca. "
    "Complete todos los campos con los datos del paciente y presione **Predecir**."
)

st.info(
    "Los campos marcados con un asterisco (*) son indicadores clínicos estándar del dataset "
    "UCI Heart Disease. Si tiene dudas sobre algún valor, coloque el cursor sobre el campo "
    "para ver una explicación."
)

# ── sección 1: datos demográficos ────────────────────────────────────────────
st.subheader("Datos demográficos")
st.caption("Información general del paciente.")

col1, col2 = st.columns(2)

with col1:
    age = st.number_input(
        "Edad *",
        min_value=1, max_value=120, value=50, step=1,
        help="Edad del paciente en años. El riesgo de enfermedad cardíaca aumenta con la edad."
    )

with col2:
    sex = st.selectbox(
        "Sexo *",
        options=[0, 1],
        format_func=lambda x: "Femenino" if x == 0 else "Masculino",
        help="Sexo biológico del paciente. Los hombres tienen mayor riesgo estadístico de enfermedad cardíaca a edades tempranas."
    )

# ── sección 2: síntomas cardíacos ────────────────────────────────────────────
st.divider()
st.subheader("Síntomas cardíacos")
st.caption("Indicadores relacionados con el funcionamiento del corazón y síntomas reportados.")

col3, col4 = st.columns(2)

with col3:
    cp = st.selectbox(
        "Tipo de dolor de pecho (cp) *",
        options=[0, 1, 2, 3],
        format_func=lambda x: {
            0: "0 — Asintomático",
            1: "1 — Angina atípica",
            2: "2 — Dolor no anginoso",
            3: "3 — Angina típica"
        }[x],
        help=(
            "Describe el tipo de dolor de pecho que reporta el paciente:\n"
            "- Angina típica: dolor de pecho clásico asociado a enfermedad cardíaca.\n"
            "- Angina atípica: dolor que no sigue el patrón clásico.\n"
            "- Dolor no anginoso: dolor de pecho sin relación cardíaca.\n"
            "- Asintomático: sin dolor de pecho. Paradójicamente, este grupo tiene mayor riesgo en este dataset."
        )
    )

with col4:
    exang = st.selectbox(
        "Angina inducida por ejercicio (exang) *",
        options=[0, 1],
        format_func=lambda x: "No" if x == 0 else "Sí",
        help="Indica si el paciente experimenta dolor de pecho al hacer ejercicio. La angina de esfuerzo es un signo importante de enfermedad cardíaca."
    )

# ── sección 3: signos vitales ─────────────────────────────────────────────────
st.divider()
st.subheader("Signos vitales")
st.caption("Mediciones clínicas objetivas del estado cardiovascular del paciente.")

col5, col6 = st.columns(2)

with col5:
    trestbps = st.number_input(
        "Presión arterial en reposo (mm Hg) *",
        min_value=80, max_value=250, value=120, step=1,
        help="Presión arterial medida en reposo en milímetros de mercurio (mm Hg). Valores normales: 90–120 mm Hg. Por encima de 130 se considera presión alta."
    )

    thalach = st.number_input(
        "Frecuencia cardíaca máxima (lpm) *",
        min_value=60, max_value=250, value=150, step=1,
        help="Frecuencia cardíaca máxima alcanzada durante una prueba de esfuerzo, en latidos por minuto. Una frecuencia máxima baja (menor a 120 lpm) puede ser indicador de riesgo."
    )

with col6:
    chol = st.number_input(
        "Colesterol sérico (mg/dl) *",
        min_value=100, max_value=600, value=200, step=1,
        help="Nivel de colesterol en sangre en miligramos por decilitro. Valores normales: por debajo de 200 mg/dl. Por encima de 240 se considera alto."
    )

    oldpeak = st.number_input(
        "Depresión del segmento ST (oldpeak) *",
        min_value=0.0, max_value=10.0, value=1.0, step=0.1, format="%.1f",
        help="Depresión del segmento ST en el electrocardiograma, inducida por el ejercicio respecto al reposo. Valores altos indican mayor riesgo de isquemia cardíaca."
    )

# ── sección 4: resultados de exámenes ────────────────────────────────────────
st.divider()
st.subheader("Resultados de exámenes")
st.caption("Resultados de pruebas clínicas especializadas.")

col7, col8 = st.columns(2)

with col7:
    fbs = st.selectbox(
        "Glucosa en ayunas > 120 mg/dl (fbs) *",
        options=[0, 1],
        format_func=lambda x: "No" if x == 0 else "Sí",
        help="Indica si la glucosa en sangre en ayunas supera los 120 mg/dl. Niveles elevados pueden indicar diabetes, un factor de riesgo cardíaco importante."
    )

    restecg = st.selectbox(
        "Resultado ECG en reposo *",
        options=[0, 1, 2],
        format_func=lambda x: {
            0: "0 — Normal",
            1: "1 — Anormalidad en onda ST-T",
            2: "2 — Hipertrofia ventricular"
        }[x],
        help=(
            "Resultado del electrocardiograma (ECG) tomado en reposo:\n"
            "- Normal: sin alteraciones.\n"
            "- Anormalidad ST-T: cambios en la onda T o depresión/elevación del segmento ST, posible isquemia.\n"
            "- Hipertrofia ventricular: engrosamiento del músculo cardíaco, puede indicar sobrecarga."
        )
    )

    slope = st.selectbox(
        "Pendiente del segmento ST *",
        options=[0, 1, 2],
        format_func=lambda x: {
            0: "0 — Descendente",
            1: "1 — Plano",
            2: "2 — Ascendente"
        }[x],
        help=(
            "Describe la forma de la curva del segmento ST durante el pico de ejercicio en el ECG:\n"
            "- Ascendente: generalmente normal, buen pronóstico.\n"
            "- Plano: moderadamente anormal.\n"
            "- Descendente: mayor asociación con enfermedad cardíaca."
        )
    )

with col8:
    ca = st.selectbox(
        "Vasos coloreados por fluoroscopía (ca) *",
        options=[0, 1, 2, 3],
        help="Número de vasos sanguíneos principales (0 a 3) que se observaron con coloración en una fluoroscopía. A mayor número de vasos afectados, mayor severidad de la enfermedad."
    )

    thal = st.selectbox(
        "Resultado de Talio (thal) *",
        options=[1, 2, 3],
        format_func=lambda x: {
            1: "1 — Normal",
            2: "2 — Defecto fijo",
            3: "3 — Defecto reversible"
        }[x],
        help=(
            "Resultado de una prueba de imagen nuclear con Talio para evaluar el flujo sanguíneo al corazón:\n"
            "- Normal: flujo normal en todas las áreas.\n"
            "- Defecto fijo: área del corazón que no recibe flujo en reposo ni en esfuerzo (posible cicatriz).\n"
            "- Defecto reversible: área con flujo reducido solo durante el esfuerzo (posible isquemia activa)."
        )
    )

# ── botón de predicción ──────────────────────────────────────────────────────
st.divider()

if st.button("Predecir", use_container_width=True, type="primary"):
    if not MODEL_PATH.exists():
        st.warning("El modelo aún no está disponible. Ejecute train.py primero.")
    else:
        try:
            result = predict_patient(
                age=age, sex=sex, cp=cp, trestbps=trestbps,
                chol=chol, fbs=fbs, restecg=restecg, thalach=thalach,
                exang=exang, oldpeak=oldpeak, slope=slope, ca=ca, thal=thal
            )

            prediction = result["prediction"]
            probability = result["probability"]
            threshold = result["threshold"]

            st.subheader("Resultado")

            if prediction == 1:
                st.error(
                    f"Alta probabilidad de enfermedad cardíaca ({probability * 100:.1f}%)\n\n"
                    "El modelo estima que este paciente tiene riesgo elevado. Se recomienda evaluación médica."
                )
            else:
                st.success(
                    f"Baja probabilidad de enfermedad cardíaca ({probability * 100:.1f}%)\n\n"
                    "El modelo no detecta riesgo elevado con los datos ingresados."
                )

            with st.expander("Ver detalles de la predicción"):
                st.write(f"**Probabilidad estimada:** {probability * 100:.1f}%")
                st.write(f"**Umbral de decisión:** {threshold} (valores >= {threshold} se clasifican como positivos)")
                st.write(f"**Resultado numérico:** {prediction} (0 = sin enfermedad, 1 = con enfermedad)")
                st.caption(
                    "Este modelo fue ajustado con un umbral de 0.35 en lugar del estándar 0.5 "
                    "para priorizar la detección de casos positivos (mayor recall), "
                    "reduciendo la probabilidad de pasar por alto un caso real de enfermedad."
                )

        except Exception as e:
            st.error(f"Error al ejecutar la predicción: {e}")

st.caption(
    "Aviso: esta herramienta es un prototipo académico y no reemplaza el criterio médico profesional."
)
