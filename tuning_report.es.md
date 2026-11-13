# StreamLoop – Informe de Optimización del Modelo de Cancelación (Churn)

## 1. Objetivo

El objetivo fue mejorar la detección de clientes con probabilidades de cancelar sus suscripciones.

Se seleccionó el Recall de la clase positiva (`Yes`) como métrica principal de evaluación, porque no detectar a un cliente que está a punto de abandonar el servicio tiene un coste mayor que identificar incorrectamente a un cliente como posible cancelación.

## 2. Modelo Baseline

Se implementó un clasificador Random Forest utilizando un pipeline de preprocesamiento y clasificación.

El modelo baseline utilizó los hiperparámetros predeterminados del clasificador y fue evaluado con el conjunto de test.

| Métrica | Baseline |
|---|---:|
| Accuracy | 0.7779 |
| Precision | 0.6034 |
| Recall | 0.4759 |
| F1-score | 0.5321 |

El modelo baseline detectó únicamente el 47.59% de los clientes que realmente cancelaron sus suscripciones.

## 3. Optimización de Hiperparámetros

Se aplicaron dos estrategias de búsqueda utilizando validación cruzada estratificada de 5 folds y Recall como métrica de evaluación.

### RandomizedSearchCV

- Número de combinaciones de hiperparámetros probadas: 20.
- Total de entrenamientos en validación cruzada: 100.
- Mejor Recall medio: 0.7980.

### GridSearchCV

El espacio de búsqueda se refinó utilizando los resultados de RandomizedSearchCV, con el objetivo de reducir el coste computacional y centrarse en los valores de hiperparámetros más prometedores.

- Número de combinaciones de hiperparámetros probadas: 64.
- Total de entrenamientos en validación cruzada: 320.
- Mejor Recall medio: 0.8080.

## 4. Hiperparámetros Finales

Se seleccionó el candidato con mejor rendimiento de GridSearchCV.

| Hiperparámetro | Valor seleccionado |
|---|---|
| n_estimators | 300 |
| max_depth | 5 |
| max_features | sqrt |
| min_samples_split | 2 |
| min_samples_leaf | 2 |
| class_weight | balanced |

## 5. Estabilidad y Selección del Modelo

El candidato seleccionado obtuvo:

- Recall medio en validación cruzada: 0.808027.
- Desviación estándar: 0.031841.

Un segundo candidato obtuvo un Recall medio ligeramente inferior, de 0.807358, y una desviación estándar de 0.030550.

Aunque el segundo candidato mostró una variabilidad ligeramente menor, la diferencia en estabilidad fue pequeña. Se seleccionó el primer candidato porque obtuvo el mayor Recall medio, lo que responde mejor al objetivo principal del negocio.

## 6. Evaluación Final en el Conjunto de Test

El modelo optimizado fue evaluado con el conjunto de test y comparado con el modelo baseline.

| Métrica | Baseline | Modelo optimizado |
|---|---:|---:|
| Accuracy | 0.7779 | 0.7367 |
| Precision | 0.6034 | 0.5026 |
| Recall | 0.4759 | 0.7861 |
| F1-score | 0.5321 | 0.6131 |

## 7. Interpretación Empresarial y Conclusión

El modelo optimizado aumentó el Recall del 47.59% al 78.61%, lo que representa una mejora de 31.02 puntos porcentuales.

Esto significa que el modelo optimizado detecta una proporción considerablemente mayor de clientes que probablemente cancelarán sus suscripciones.

Aunque Precision y Accuracy disminuyeron, este intercambio es aceptable en el contexto empresarial de StreamLoop, ya que no detectar a un cliente que abandona el servicio tiene un coste mayor que contactar a un cliente que finalmente no iba a cancelar.

El F1-score también mejoró, pasando de 0.5321 a 0.6131.

**Conclusión:** Se prefiere el modelo Random Forest optimizado frente al baseline porque responde mejor al objetivo empresarial principal de StreamLoop: identificar a la mayor cantidad posible de clientes con riesgo de cancelar sus suscripciones.