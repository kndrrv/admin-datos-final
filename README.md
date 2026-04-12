# admin-datos-final

Proyecto de ETL, entrenamiento y prediccion para un problema de clasificacion binaria con datos tabulares.

## Modelo

Se entreno un `MLPClassifier` como modelo principal. Aunque se probaron busquedas con `GridSearchCV`, el mejor comportamiento practico se obtuvo con el modelo base y un ajuste del threshold de decision.

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
