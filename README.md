# Pipeline de Predicción de Enfermedad Cardíaca

Proyecto final de Administración de Datos — LEAD University.  
Esteban Ramírez · Kendra Vega · Ignacio Castillo.

---

## Requisitos previos

- Python 3.9 o superior
- Archivo `.env` en la raíz del proyecto con el connection string de la base de datos:

```
DATABASE_URL=postgresql://usuario:password@host/dbname?sslmode=require
```

---

## Instalación

```bash
git clone https://github.com/kndrrv/admin-datos-final.git
cd admin-datos-final
pip install -r requirements.txt
```

---

## Ejecución

**1. Correr el pipeline** — extrae, transforma, carga y entrena el modelo:

```bash
python orchestration/pipeline.py data/heart.csv
```

**2. Levantar la aplicación** — se recomienda Streamlit por su facilidad de uso:

```bash
streamlit run app/main.py
```

Abre en `http://localhost:8501`. Ingrese los datos clínicos del paciente y presione **Predecir**.

---

## Estructura

```
├── etl/            # Extracción, transformación y carga
├── model/          # Entrenamiento e inferencia
├── orchestration/  # Pipeline maestro
├── app/            # Aplicación Streamlit
└── db/             # Esquema de la base de datos
```

---

## Modelo

Red neuronal MLPClassifier entrenada sobre el UCI Heart Disease Dataset.  
Umbral de decisión: `0.35` — Accuracy: `0.7705` — Recall: `0.9091` — ROC-AUC: `0.8506`.
