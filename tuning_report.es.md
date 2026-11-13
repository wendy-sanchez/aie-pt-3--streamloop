# StreamLoop – Predicción de abandono de clientes (Customer Churn)

## 1. Objetivo del proyecto

El objetivo de este proyecto es desarrollar un modelo de Machine Learning capaz de identificar a los clientes que han abandonado el servicio de StreamLoop, utilizando los datos disponibles sobre sus características y comportamiento como clientes.

El objetivo principal durante el entrenamiento y la optimización del modelo es maximizar el **Recall de la clase positiva (`Yes`)**, ya que se considera especialmente importante identificar correctamente a los clientes que pertenecen a la clase de abandono.

Un falso negativo se produce cuando el modelo clasifica como `No` a un cliente que realmente pertenece a la clase `Yes`. Este tipo de error implica no identificar un caso de abandono. Por otro lado, un falso positivo supone clasificar como `Yes` a un cliente que realmente pertenece a la clase `No`.

Para este proyecto se prioriza reducir los falsos negativos, aunque esto pueda implicar un aumento de los falsos positivos. Esta decisión responde al objetivo de detectar tantos casos de abandono como sea posible.

## 2. Dataset y preprocesamiento

Para desarrollar el modelo se utilizó el conjunto de datos Telco Customer Churn de IBM, compuesto inicialmente por 7.043 registros y 21 columnas.

Antes del entrenamiento se realizaron las siguientes tareas de preparación:

- Se eliminaron los espacios de los nombres de las columnas para facilitar su utilización en el código.
- Se eliminó la columna `customerID`, ya que contiene identificadores únicos que no aportan información predictiva relevante.
- La columna `TotalCharges` se convirtió a formato numérico. Los 11 valores vacíos detectados se transformaron en valores nulos (`NaN`) para poder tratarlos mediante imputación.
- Se identificó `Churn` como variable objetivo, con las categorías `Yes` y `No`.
- Se separaron las variables predictoras de la variable objetivo.
- Se dividieron los datos en conjuntos de entrenamiento y prueba mediante `train_test_split`, utilizando un **80 % para entrenamiento y un 20 % para prueba**. Se estableció `random_state=42` y `stratify=y` para mantener la proporción de las clases en ambos conjuntos.

La separación se realizó antes del preprocesamiento para evitar la filtración de información del conjunto de prueba durante el entrenamiento.

### Estrategia de preprocesamiento

Se utilizó `ColumnTransformer` dentro de un `Pipeline` de Scikit-learn para aplicar las transformaciones correspondientes a cada tipo de variable.

- **Variables numéricas:** se utilizó la imputación por mediana (`SimpleImputer(strategy="median")`). La mediana es menos sensible a valores extremos que la media y permite completar los valores ausentes sin eliminar registros.
- **Variables categóricas:** se utilizó la imputación por la categoría más frecuente (`SimpleImputer(strategy="most_frequent")`) para completar los valores ausentes.
- **Codificación categórica:** se aplicó `OneHotEncoder(handle_unknown="ignore")` para transformar las variables categóricas en columnas numéricas que puedan ser procesadas por el algoritmo. La opción `handle_unknown="ignore"` permite manejar categorías que no hayan aparecido durante el entrenamiento.

Se decidió mantener todo el preprocesamiento dentro del `Pipeline` para garantizar que las transformaciones se ajustaran únicamente con los datos de entrenamiento de cada partición de validación cruzada. De esta manera, se evita la filtración de información (*data leakage*) y se obtiene una evaluación más fiable del rendimiento del modelo.

## 3. Selección del algoritmo: ¿Por qué Random Forest?

Se seleccionó **Random Forest Classifier** como algoritmo principal porque es adecuado para trabajar con conjuntos de datos que contienen diferentes tipos de variables y relaciones que pueden no ser lineales.

El abandono de clientes puede estar relacionado con la combinación de diferentes factores, como la antigüedad del cliente, los cargos mensuales, el tipo de contrato y los servicios contratados. Estas relaciones pueden ser complejas y no necesariamente seguir un patrón lineal.

Random Forest combina múltiples árboles de decisión entrenados sobre distintas muestras y subconjuntos de características. Al combinar sus predicciones, el modelo puede capturar relaciones e interacciones entre variables y reducir la dependencia de un único árbol, lo que ayuda a controlar la variabilidad del modelo.

Además, este algoritmo no requiere normalizar ni estandarizar las variables numéricas, lo que simplifica el proceso de preprocesamiento.

Antes de elegirlo, se consideraron las características de otros algoritmos de clasificación habituales:

| Algoritmo | Ventajas | Limitaciones | Idoneidad para este proyecto |
|---|---|---|---|
| **Regresión logística** | Es simple, rápida e interpretable. | Su formulación básica es lineal y puede no capturar relaciones no lineales sin transformaciones adicionales. | Proporciona una referencia inicial útil, pero podría no representar adecuadamente las relaciones complejas entre las características de los clientes y su comportamiento de abandono. |
| **Árbol de decisión** | Es fácil de interpretar y capaz de capturar relaciones no lineales. | Un árbol individual puede ser inestable y propenso al sobreajuste si no se controla adecuadamente. | Permite identificar patrones asociados al abandono de clientes, aunque puede tener una capacidad de generalización inferior a la de los métodos de conjunto. |
| **Random Forest** | Combina árboles independientes, reduce la variabilidad y captura interacciones entre características. | Es menos interpretable y puede requerir más recursos computacionales. | Es adecuado para datos tabulares y permite priorizar el Recall mediante el ajuste de hiperparámetros y la ponderación de clases. |
| **Gradient Boosting** | Aprende secuencialmente de los errores anteriores y puede capturar patrones complejos con un alto rendimiento predictivo. | Es más sensible al ajuste de hiperparámetros y puede sobreajustarse si no se controla la complejidad del modelo. | Es una alternativa relevante para problemas de clasificación con datos tabulares que podría haberse evaluado experimentalmente, aunque no fue entrenada ni comparada en este proyecto. |

**→ Random Forest** se consideró la opción más equilibrada para este problema, teniendo en cuenta la naturaleza de los datos, la posibilidad de detectar relaciones complejas y la necesidad de obtener un modelo con capacidad de generalización.

```
Se seleccionó Random Forest como algoritmo principal porque ofrece un equilibrio práctico entre capacidad predictiva, eficiencia de entrenamiento y facilidad de optimización. Su estructura de árboles independientes permite el entrenamiento en paralelo, mientras que su enfoque de conjunto ayuda a reducir la variabilidad y proporciona un rendimiento robusto en conjuntos de datos tabulares. Además, ofrece una configuración de hiperparámetros relativamente sencilla y opciones integradas de ponderación de clases, lo que lo convierte en una alternativa adecuada para este proyecto, cuyo objetivo principal era priorizar la detección del abandono de clientes mediante la optimización del Recall.

Aunque Gradient Boosting también es un algoritmo sólido para problemas de clasificación con datos tabulares, su proceso de entrenamiento secuencial y su mayor sensibilidad a la configuración de hiperparámetros pueden hacer que su optimización sea más exigente. Por ello, Random Forest se consideró un punto de partida más práctico para este proyecto, al ofrecer un enfoque robusto y manejable sin requerir una estrategia de optimización más compleja.
```

## 4. Modelo base (Baseline)

Como punto de partida se entrenó un modelo `RandomForestClassifier` con sus hiperparámetros predeterminados, estableciendo únicamente `random_state=42` para garantizar la reproducibilidad.

El modelo base se evaluó sobre el conjunto de prueba utilizando las siguientes métricas:

| Métrica | Resultado |
|---|---:|
| Accuracy | 0.7779 |
| Precision | 0.6034 |
| Recall | 0.4759 |
| F1-score | 0.5321 |

La matriz de confusión obtenida fue:

| | Predicción: No | Predicción: Yes |
|---|---:|---:|
| **Real: No** | 918 | 117 |
| **Real: Yes** | 196 | 178 |

El modelo base obtuvo un Recall de 0.4759 para la clase `Yes`, lo que significa que identificó correctamente aproximadamente el 47,59 % de los clientes pertenecientes a dicha clase.

Este resultado mostró que el modelo inicial no detectaba una proporción suficientemente alta de los casos positivos. Por ello, se decidió realizar una optimización de hiperparámetros priorizando el Recall.

## 5. Estrategia de optimización de hiperparámetros

Para mejorar el rendimiento del modelo se utilizaron dos técnicas de búsqueda de hiperparámetros: `RandomizedSearchCV` y `GridSearchCV`.

Se eligió el Recall como métrica principal de evaluación mediante `make_scorer(recall_score, pos_label="Yes")`, ya que el objetivo del proyecto es identificar la mayor cantidad posible de casos de abandono.

También se utilizó validación cruzada estratificada (`StratifiedKFold`) con 5 particiones, `shuffle=True` y `random_state=42`.

La estratificación permite mantener aproximadamente la misma proporción de las clases `Yes` y `No` en cada partición. Esto es especialmente importante cuando existe un desequilibrio entre las clases, ya que evita que algunas particiones tengan una representación poco adecuada de la clase minoritaria.

Se utilizaron cinco particiones para obtener varias estimaciones del rendimiento del modelo y evaluar su comportamiento en diferentes subconjuntos de los datos, en lugar de depender de una única división de entrenamiento y validación.

### 5.1. RandomizedSearchCV

En la primera etapa se utilizó `RandomizedSearchCV` para explorar un espacio amplio de hiperparámetros mediante una selección aleatoria de 20 combinaciones.

Con cinco particiones de validación cruzada, se realizaron un total de 100 ajustes del modelo.

Esta técnica permite explorar diferentes configuraciones sin tener que evaluar todas las combinaciones posibles de un espacio de búsqueda amplio, lo que resulta útil para identificar regiones prometedoras y orientar una búsqueda posterior más específica.

El mejor resultado obtenido fue:

- **Recall medio en validación cruzada:** 0.7980

Los mejores hiperparámetros encontrados fueron:

```python
{
    'classifier__n_estimators': 500,
    'classifier__min_samples_split': 10,
    'classifier__min_samples_leaf': 2,
    'classifier__max_features': 'log2',
    'classifier__max_depth': 5,
    'classifier__class_weight': 'balanced_subsample'
}
```

Estos resultados se utilizaron como referencia para definir un espacio de búsqueda más concreto en la siguiente etapa.

### 5.2. GridSearchCV

Después de analizar los resultados de `RandomizedSearchCV`, se utilizó `GridSearchCV` para realizar una búsqueda sistemática dentro de un espacio de hiperparámetros más específico.

En esta segunda etapa se evaluaron 64 combinaciones, con cinco particiones de validación cruzada, lo que supuso un total de 320 ajustes del modelo.

A diferencia de la búsqueda aleatoria, `GridSearchCV` evalúa todas las combinaciones definidas en el espacio de búsqueda. En este caso, se utilizó para comparar de manera sistemática las configuraciones seleccionadas a partir de los resultados de la primera búsqueda.

El mejor resultado obtenido fue:

- **Recall medio en validación cruzada:** 0.8080

Los mejores hiperparámetros fueron:

```python
{
    'classifier__class_weight': 'balanced',
    'classifier__max_depth': 5,
    'classifier__max_features': 'sqrt',
    'classifier__min_samples_leaf': 2,
    'classifier__min_samples_split': 2,
    'classifier__n_estimators': 300
}
```

El parámetro `class_weight='balanced'` asigna pesos inversamente proporcionales a la frecuencia de las clases. Se utilizó para compensar el desequilibrio entre las categorías y ayudar al modelo a prestar mayor atención a la clase minoritaria.

Otros parámetros relevantes fueron:

- `n_estimators`: controla el número de árboles del bosque. Se ajustó para encontrar un equilibrio entre capacidad de aprendizaje y estabilidad.
- `max_depth`: limita la profundidad de los árboles y ayuda a controlar su complejidad, reduciendo el riesgo de sobreajuste.
- `min_samples_split`: establece el número mínimo de muestras necesario para dividir un nodo.
- `min_samples_leaf`: establece el número mínimo de muestras que debe contener una hoja, ayudando a evitar divisiones demasiado específicas.
- `max_features`: controla el número de características consideradas en cada división, favoreciendo la diversidad entre los árboles.

## 6. Resultados de validación cruzada y estabilidad del modelo

Además de comparar el Recall medio, se analizaron los resultados individuales de las cinco particiones de validación cruzada y su desviación estándar.

Este análisis permite comprobar si el rendimiento del modelo se mantiene relativamente estable entre las diferentes particiones o si presenta variaciones importantes.

Entre las configuraciones con mejores resultados se compararon especialmente las siguientes:

| Configuración | Recall medio | Desviación estándar |
|---|---:|---:|
| Candidato 1 | 0.808027 | 0.031841 |
| Candidato 5 | 0.807358 | 0.030550 |

Los resultados individuales de las cinco particiones fueron:

**Candidato 1:**
- Fold 1: 0.806020
- Fold 2: 0.769231
- Fold 3: 0.816054
- Fold 4: 0.862876
- Fold 5: 0.785953

**Candidato 5:**
- Fold 1: 0.809365
- Fold 2: 0.769231
- Fold 3: 0.812709
- Fold 4: 0.859532
- Fold 5: 0.785953

El candidato 1 obtuvo el Recall medio más alto, mientras que el candidato 5 presentó una desviación estándar ligeramente inferior, lo que indica una variabilidad algo menor entre las particiones.

La diferencia entre ambos resultados es pequeña. Sin embargo, como el objetivo principal del proyecto es maximizar el Recall, se seleccionó el candidato 1, ya que obtuvo el mejor promedio de esta métrica.

Esta decisión prioriza el rendimiento medio en la métrica definida como principal, teniendo también en cuenta que la variabilidad observada entre las particiones no presenta diferencias importantes.

## 7. Selección y entrenamiento del modelo final

Una vez finalizada la búsqueda de hiperparámetros, se seleccionó la mejor configuración obtenida mediante `GridSearchCV`.

El modelo final utiliza los siguientes parámetros:

| Hiperparámetro | Valor seleccionado |
|---|---|
| `n_estimators` | 300 |
| `max_depth` | 5 |
| `max_features` | sqrt |
| `min_samples_split` | 2 |
| `min_samples_leaf` | 2 |
| `class_weight` | balanced |

Se utilizó `best_estimator_` para recuperar el mejor modelo encontrado por la búsqueda.

La configuración de búsqueda se estableció con `refit=True`, por lo que Scikit-learn volvió a entrenar automáticamente el mejor estimador utilizando todo el conjunto de entrenamiento.

El conjunto de prueba se mantuvo separado durante las etapas de optimización y validación cruzada. Únicamente se utilizó para evaluar el rendimiento final del modelo y compararlo con el modelo base.

## 8. Evaluación final del modelo

El modelo optimizado se evaluó sobre el mismo conjunto de prueba utilizado para medir el rendimiento del modelo base.

Los resultados obtenidos fueron:

| Métrica | Modelo base | Modelo optimizado |
|---|---:|---:|
| Accuracy | 0.7779 | 0.7367 |
| Precision | 0.6034 | 0.5026 |
| Recall | 0.4759 | 0.7861 |
| F1-score | 0.5321 | 0.6131 |

El modelo optimizado obtuvo un Recall de 0.7861, frente al 0.4759 del modelo base.

Esto representa un incremento de aproximadamente **31,02 puntos porcentuales** en la capacidad de identificar correctamente los casos pertenecientes a la clase `Yes`.

Por otro lado, el Accuracy disminuyó de 0.7779 a 0.7367 y la Precision pasó de 0.6034 a 0.5026. Sin embargo, el F1-score aumentó de 0.5321 a 0.6131.

Esta variación refleja el equilibrio buscado durante la optimización: el modelo consigue identificar una proporción mucho mayor de los casos positivos, aunque a cambio genera más falsos positivos.

Dado que el objetivo principal es maximizar la detección de clientes pertenecientes a la clase de abandono, se considera que el incremento del Recall representa una mejora relevante respecto al modelo base.

## 9. Interpretación de los resultados desde una perspectiva de negocio

Desde el punto de vista del negocio, el modelo optimizado permite identificar aproximadamente 79 de cada 100 clientes que pertenecen a la clase `Yes` en el conjunto de prueba.

Esto supone una mejora considerable respecto al modelo base, que identificaba aproximadamente 48 de cada 100.

Una mayor capacidad de detección puede ayudar a una empresa a identificar grupos de clientes que requieren un análisis más detallado y que podrían ser considerados para futuras estrategias de retención.

No obstante, la reducción de Precision indica que el modelo también clasifica como posibles casos de abandono a algunos clientes que realmente pertenecen a la clase `No`.

Por tanto, los resultados deben interpretarse teniendo en cuenta el coste de ambos tipos de error. Si una empresa considera que contactar a un cliente que finalmente no abandona el servicio tiene un coste asumible, puede resultar conveniente priorizar el Recall. Sin embargo, para determinar el impacto económico real sería necesario analizar los costes asociados a los falsos positivos y falsos negativos, así como definir una estrategia de intervención.

El modelo constituye una herramienta de apoyo para la identificación de patrones de abandono, pero sus predicciones no deben interpretarse como una certeza sobre el comportamiento futuro de cada cliente.

## 10. Conclusión

El proyecto permitió desarrollar y optimizar un modelo de clasificación basado en Random Forest para identificar clientes pertenecientes a la clase de abandono (`Yes`).

La estrategia de optimización combinó `RandomizedSearchCV` y `GridSearchCV`, utilizando validación cruzada estratificada y priorizando el Recall como métrica principal.

El modelo final alcanzó un Recall de 0.7861 en el conjunto de prueba, frente al 0.4759 obtenido por el modelo base. Aunque se produjo una reducción en Accuracy y Precision, el F1-score mejoró y se consiguió el objetivo principal de aumentar la detección de casos positivos.

La selección de Random Forest se fundamentó en su capacidad para capturar relaciones no lineales e interacciones entre variables, así como en la reducción de variabilidad que ofrece la combinación de múltiples árboles de decisión.

En conclusión, el modelo optimizado representa una mejora respecto al modelo base para el objetivo específico de identificar clientes pertenecientes a la clase de abandono. Su aplicación práctica podría ayudar a orientar estrategias de retención, aunque sería necesario evaluar los costes de los errores de clasificación y validar su utilidad con datos y condiciones reales de negocio.