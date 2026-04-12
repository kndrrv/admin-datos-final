# admin-datos-final

Proyecto de ETL, entrenamiento y prediccion para un problema de clasificacion binaria con datos tabulares.

## Modelo

Se entreno un `MLPClassifier` como modelo principal. El modelo final del proyecto se basa en el entrenamiento estandar y un ajuste del threshold de decision.

## Criterio final

- Modelo final: `MLPClassifier` con hiperparametros base.
- Umbral de decision usado en prediccion: `0.35`.
- Motivo: mantiene `roc_auc = 0.8506` y mejora `recall` de `0.7879` a `0.9091` sin empeorar la `accuracy` observada (`0.7705`).

## Ejecucion

Entrenamiento base:

```powershell
.\venv\Scripts\python.exe model\train.py
```

Analisis de thresholds:

```powershell
.\venv\Scripts\python.exe model\train.py --thresholds
```

Prediccion:

```powershell
.\venv\Scripts\python.exe model\predict.py
```

## Integracion de frontend

Funciones para la integracion:

- `model.predict.load_model()`: carga el modelo serializado y las columnas esperadas.
- `model.predict.build_model_input(input_data)`: recibe los datos crudos y agrega las features derivadas.
- `model.predict.predict(input_data)`: recibe un `dict` y devuelve `prediction`, `probability` y `threshold`.
- `model.predict.predict_patient(...)`: recibe los campos del paciente como argumentos y devuelve el resultado final listo para mostrar.
