# Proyecto Estadística: Clasificador Bayesiano Naive para Diabetes con Optimización de Semilla y Paralelización CPU

Este proyecto implementa un **clasificador bayesiano naive** para el dataset **Pima Indians Diabetes**, con un enfoque en la optimización del rendimiento mediante la búsqueda paralelizada de la mejor semilla aleatoria. Además, incluye análisis estadísticos, evaluación exhaustiva y visualizaciones para interpretar los resultados.

---

## Contenido

- Código principal que realiza la carga de datos, búsqueda de semilla óptima, entrenamiento, evaluación y visualización.
- Dependencias: Paquetes R necesarios para la ejecución.

---

## Fundamentos Teóricos

### Clasificador Bayesiano Naive

El clasificador bayesiano naive se basa en el **Teorema de Bayes** y la suposición de independencia condicional entre las variables predictoras.

\[
P(C*k | \mathbf{x}) = \frac{P(C_k) \prod*{j=1}^n P(x_j | C_k)}{P(\mathbf{x})}
\]

Donde:

- \(P(C_k | \mathbf{x})\) es la probabilidad posterior de la clase \(C_k\) dado el vector de características \(\mathbf{x}\).
- \(P(C_k)\) es la probabilidad previa de la clase \(C_k\).
- \(P(x_j | C_k)\) es la probabilidad condicional de la característica \(x_j\) dado \(C_k\).
- \(P(\mathbf{x})\) es la probabilidad marginal de los datos (constante para todas las clases).

### Suposición Naive

Se asume que las características son independientes entre sí dado la clase, lo que simplifica el cálculo de la probabilidad conjunta:

\[
P(\mathbf{x} | C*k) = \prod*{j=1}^n P(x_j | C_k)
\]

### Distribución Normal para Variables Continuas

Para variables continuas, se modela \(P(x_j | C_k)\) como una distribución normal:

\[
P(x*j | C_k) = \frac{1}{\sqrt{2\pi \sigma*{jk}^2}} \exp\left(-\frac{(x*j - \mu*{jk})^2}{2\sigma\_{jk}^2}\right)
\]

Donde \(\mu*{jk}\) y \(\sigma*{jk}\) son la media y desviación estándar de la característica \(j\) en la clase \(k\).

### Log-Verosimilitud

Para evitar problemas numéricos con productos de probabilidades pequeñas, se trabaja con logaritmos:

\[
\log P(C*k | \mathbf{x}) \propto \log P(C_k) + \sum*{j=1}^n \log P(x_j | C_k)
\]

El clasificador asigna la clase con la mayor log-verosimilitud posterior.

---

## Descripción del Código

El script realiza los siguientes pasos:

1. **Preparación del entorno**  
   Limpia el entorno de trabajo y carga/instala los paquetes necesarios (`mlbench`, `ggplot2`, `dplyr`, `tidyr`, `pROC`, `patchwork`, `parallel`).

2. **Carga del dataset**  
   Utiliza el dataset `PimaIndiansDiabetes` del paquete `mlbench`, que contiene variables médicas y la variable objetivo `diabetes` (positivo/negativo).

3. **Búsqueda de la mejor semilla**  
   Ejecuta un pipeline completo para 1000 semillas diferentes en paralelo usando múltiples núcleos del CPU para encontrar la semilla que maximiza el **F1-Score** y el **AUC** (Área Bajo la Curva ROC).

4. **Entrenamiento y evaluación con la semilla óptima**  
   Divide los datos de forma estratificada para mantener la proporción de clases, calcula estadísticas por clase (media y desviación estándar), entrena el clasificador bayesiano naive y evalúa su rendimiento.

5. **Evaluación con múltiples umbrales**  
   Realiza un barrido de umbrales de decisión \(\epsilon\) para encontrar el umbral óptimo que maximiza el F1-Score, ajustando la sensibilidad y especificidad del modelo.

6. **Visualizaciones**  
   Genera gráficos que incluyen:

   - Métricas (Accuracy, Sensitivity, Specificity, F1-Score) vs. umbral de decisión
   - Matriz de confusión (heatmap) con conteos y porcentajes
   - Distribución de scores (log-ratio de probabilidades) por clase
   - Curva ROC con AUC para evaluar la capacidad discriminativa del modelo

7. **Resumen final**  
   Muestra las métricas finales, la semilla óptima utilizada, el tiempo de ejecución y los núcleos CPU usados.

---

## Métricas de Evaluación

- **Accuracy (Exactitud):** Proporción de predicciones correctas.
- **Sensitivity (Recall o Sensibilidad):** Proporción de positivos correctamente identificados.
- **Specificity (Especificidad):** Proporción de negativos correctamente identificados.
- **Precision (Precisión):** Proporción de predicciones positivas correctas.
- **F1-Score:** Media armónica entre precisión y recall, balancea falsos positivos y falsos negativos.
- **AUC (Área Bajo la Curva ROC):** Mide la capacidad del modelo para distinguir entre clases.

---

## Paralelización

Se utiliza el paquete `parallel` para distribuir la evaluación de semillas en múltiples núcleos del CPU, acelerando significativamente la búsqueda de la semilla óptima.

---

## Requisitos

- R versión 4.0 o superior recomendada.
- Paquetes R:  
  `mlbench`, `ggplot2`, `dplyr`, `tidyr`, `pROC`, `patchwork`, `parallel`

Puedes instalar los paquetes con:

```r
install.packages(c("mlbench", "ggplot2", "dplyr", "tidyr", "pROC", "patchwork", "parallel"))
```
