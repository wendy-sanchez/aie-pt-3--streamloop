# StreamLoop – Predicción de abandono de clientes

Proyecto de Machine Learning centrado en identificar a los clientes pertenecientes a la clase de abandono (`Yes`) utilizando el conjunto de datos IBM Telco Customer Churn. El objetivo principal es mejorar la detección de casos positivos, priorizando el **Recall** durante la optimización de hiperparámetros.

## Objetivo del proyecto

El proyecto desarrolla y evalúa un clasificador Random Forest para identificar a los clientes que han abandonado el servicio. Se utiliza el Recall de la clase `Yes` como métrica principal porque se prioriza reducir los falsos negativos: los casos en los que un cliente que pertenece a la clase de abandono se clasifica incorrectamente como `No`.

Esta prioridad implica un compromiso: aumentar el Recall también puede incrementar los falsos positivos y reducir la Precision.

## Dataset

El proyecto utiliza el [dataset IBM Telco Customer Churn](https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv).

- Tamaño inicial: 7.043 registros y 21 columnas.
- Variable objetivo: `Churn` (`Yes` / `No`).
- Se elimina `customerID` porque es un identificador único y no una característica predictiva útil.
- `TotalCharges` se convierte a formato numérico; los 11 valores vacíos pasan a ser valores ausentes y se tratan durante el preprocesamiento.

## Metodología

### Preparación de los datos

Los datos se dividen en conjuntos de entrenamiento y prueba antes del preprocesamiento: 80 % para entrenamiento y 20 % para prueba, utilizando `random_state=42` y estratificación mediante la variable objetivo.

El preprocesamiento se implementa dentro de un `Pipeline` y un `ColumnTransformer` de Scikit-learn para evitar la filtración de información (*data leakage*) durante la validación cruzada:

- Variables numéricas: imputación por mediana.
- Variables categóricas: imputación por el valor más frecuente y posterior codificación One-Hot.
- `OneHotEncoder(handle_unknown="ignore")` permite gestionar categorías que no aparezcan durante el ajuste.

### Selección del algoritmo

Se seleccionó Random Forest porque permite modelar relaciones no lineales e interacciones entre características. Al combinar múltiples árboles de decisión, puede reducir la variabilidad asociada a depender de un único árbol. Además, no requiere escalar las variables numéricas.

Se consideraron conceptualmente la Regresión Logística y un Árbol de Decisión individual como alternativas. Este proyecto no incluye una comparación experimental en la que se entrenen y evalúen los tres algoritmos; la comparación se basa en sus características metodológicas.

| Algoritmo | Ventajas | Limitaciones para este problema |
|---|---|---|
| Regresión Logística | Sencilla, rápida e interpretable; modela relaciones entre las características y la probabilidad de pertenecer a una clase. | Su planteamiento básico es lineal y puede no capturar relaciones complejas sin transformaciones adicionales. |
| Árbol de Decisión | Es interpretable y puede capturar relaciones no lineales. | Un árbol individual puede ser inestable y sobreajustarse si no se controla su complejidad. |
| Random Forest | Captura relaciones no lineales e interacciones entre características; combinar árboles ayuda a reducir la variabilidad frente a un árbol individual. | Es menos interpretable que un único árbol y requiere ajustar varios hiperparámetros. |

### Modelo base

Se entrenó un `RandomForestClassifier` de referencia utilizando los hiperparámetros predeterminados del clasificador y `random_state=42`.

### Optimización de hiperparámetros

El proceso de optimización utiliza:

- Métrica principal: Recall para `Churn="Yes"`.
- Validación cruzada: `StratifiedKFold` de 5 particiones, con mezcla de datos (`shuffle=True`) y `random_state=42`.
- `RandomizedSearchCV`: 20 configuraciones seleccionadas aleatoriamente, con un total de 100 ajustes mediante validación cruzada.
- `GridSearchCV`: 64 configuraciones en un espacio de búsqueda refinado, con un total de 320 ajustes.
- `refit=True`: el mejor estimador se vuelve a entrenar con todo el conjunto de entrenamiento.

Los resultados de validación cruzada se analizan teniendo en cuenta tanto el Recall medio como la desviación estándar entre particiones. La configuración final se selecciona según el mayor Recall medio. El conjunto de prueba se mantiene separado durante la optimización y se utiliza para evaluar el modelo base y el modelo final.

## Resultados

### Comparación entre el modelo base y el optimizado

| Métrica | Modelo base | Modelo optimizado |
|---|---:|---:|
| Accuracy | 0.7779 | 0.7367 |
| Precision | 0.6034 | 0.5026 |
| Recall | 0.4759 | 0.7861 |
| F1-score | 0.5321 | 0.6131 |

El modelo optimizado mejora el Recall en el conjunto de prueba de **0.4759 a 0.7861**, lo que supone un incremento de aproximadamente **31,02 puntos porcentuales**. El F1-score también aumenta, mientras que Accuracy y Precision disminuyen. Esto refleja el compromiso elegido para el proyecto: el modelo optimizado identifica una proporción considerablemente mayor de los clientes de la clase positiva, a cambio de generar más falsos positivos.

### Mejores hiperparámetros

La mejor configuración encontrada mediante `GridSearchCV` es:

```python
{
    "classifier__class_weight": "balanced",
    "classifier__max_depth": 5,
    "classifier__max_features": "sqrt",
    "classifier__min_samples_leaf": 2,
    "classifier__min_samples_split": 2,
    "classifier__n_estimators": 300
}
```

Mejor Recall medio en validación cruzada: **0.8080**.

## Cómo ejecutar el proyecto

1. Clona o descarga este repositorio.
2. Abre `streamloop_churn_tuning.ipynb` en Jupyter Notebook, JupyterLab o un entorno compatible, como GitHub Codespaces.
3. Instala los paquetes necesarios si todavía no están disponibles:

   ```bash
   pip install pandas scikit-learn jupyter
   ```

4. Ejecuta las celdas del notebook de principio a fin.

El notebook utiliza un estado aleatorio fijo para facilitar la reproducibilidad. Los resultados indicados se reprodujeron después de reiniciar el kernel y ejecutar todas las celdas en orden.

### Entorno utilizado para la ejecución registrada

- Python 3.14.2
- pandas 3.1.1
- scikit-learn 1.9.1

También pueden funcionar otras versiones compatibles, aunque podrían producir pequeñas diferencias en los resultados.

## Limitaciones e interpretación

- La variable objetivo representa la etiqueta de abandono presente en el dataset. El proyecto no define un horizonte temporal de predicción futura, por lo que el modelo no debe interpretarse como una herramienta que predice exactamente cuándo se irá un cliente.
- El modelo es una herramienta de apoyo a la clasificación, no una garantía sobre el comportamiento de un cliente concreto.
- El mayor Recall se obtiene con una Precision menor. El impacto real en el negocio depende del coste relativo de los falsos positivos y los falsos negativos, que no se ha cuantificado en este proyecto.
- La comparación entre Random Forest, Regresión Logística y Árbol de Decisión es conceptual; en este notebook solo se optimizó y evaluó Random Forest.
- Los resultados se basan en el dataset utilizado y deberían validarse con datos actuales del negocio antes de una aplicación operativa.

## Tecnologías

- Python
- pandas
- Scikit-learn
- Jupyter Notebook

## Archivo del proyecto

- `streamloop_churn_tuning.ipynb`: preparación de datos, entrenamiento del modelo base, optimización de hiperparámetros, análisis de validación cruzada y evaluación final.
- `tuning_report.md` y `tuning_report.es.md` - archivo de documentación del proyecto

---
**Developed by Wendy Sanchez**