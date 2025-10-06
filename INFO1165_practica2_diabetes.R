# Limpiar entorno
rm(list = ls())
gc()

# Instalar y cargar paquetes necesarios
packages <- c("mlbench", "ggplot2", "dplyr", "tidyr")
for (pkg in packages) {
  if (!require(pkg, character.only = TRUE, quietly = TRUE)) {
    install.packages(pkg, dependencies = TRUE)
    library(pkg, character.only = TRUE)
  }
}

# Configuración de reproducibilidad
set.seed(123)
options(scipen = 999, digits = 6)

# Función auxiliar para mensajes formateados
print_section <- function(title, symbol = "=", width = 80) {
  cat("\n")
  cat(paste(rep(symbol, width), collapse = ""), "\n")
  cat(paste0(" ", title, "\n"))
  cat(paste(rep(symbol, width), collapse = ""), "\n\n")
}

print_subsection <- function(title) {
  cat("\n--- ", title, " ---\n")
}

print_section("CLASIFICADOR BAYESIANO NAIVE - DIABETES", "=")
cat("Dataset: Pima Indians Diabetes\n")
cat("Fecha de ejecución:", format(Sys.time(), "%Y-%m-%d %H:%M:%S"), "\n")


#===============================================================================
# 1. CARGA Y EXPLORACIÓN INICIAL DE DATOS
#===============================================================================

print_section("1. CARGA Y ANÁLISIS EXPLORATORIO DE DATOS (EDA)")

# Cargar dataset
data("PimaIndiansDiabetes")
df <- PimaIndiansDiabetes

print_subsection("Estructura del Dataset")
str(df)

print_subsection("Dimensiones")
cat(sprintf("Número de observaciones: %d\n", nrow(df)))
cat(sprintf("Número de variables: %d\n", ncol(df)))
cat(sprintf("Variables predictoras: %d\n", ncol(df) - 1))

print_subsection("Primeras Observaciones")
print(head(df))

# Análisis de valores faltantes y anómalos
print_subsection("Análisis de Calidad de Datos")

# Valores faltantes
na_count <- colSums(is.na(df))
cat("\nValores faltantes (NA) por variable:\n")
print(na_count)
cat(sprintf("Total de NAs en el dataset: %d\n", sum(na_count)))

# Identificar ceros anómalos (en variables médicas donde cero es imposible)
# Variables donde cero es biológicamente imposible: glucose, pressure, triceps, insulin, mass
zero_vars <- c("glucose", "pressure", "triceps", "insulin", "mass")
cat("\nValores cero anómalos (mediciones imposibles):\n")
for (var in zero_vars) {
  if (var %in% names(df)) {
    zero_count <- sum(df[[var]] == 0, na.rm = TRUE)
    zero_pct <- 100 * zero_count / nrow(df)
    cat(sprintf("  %s: %d ceros (%.2f%%)\n", var, zero_count, zero_pct))
  }
}

cat("\nNOTA: Los ceros en variables médicas se mantendrán como están para este análisis.\n")
cat("En un escenario real, deberían ser tratados como valores faltantes o imputados.\n")

# Estadísticas descriptivas por clase
print_subsection("Estadísticas Descriptivas por Clase")
stats_summary <- df %>%
  group_by(diabetes) %>%
  summarise(
    n = n(),
    across(where(is.numeric), list(
      media = ~mean(.x, na.rm = TRUE),
      mediana = ~median(.x, na.rm = TRUE),
      sd = ~sd(.x, na.rm = TRUE)
    ), .names = "{.col}_{.fn}")
  )

print(stats_summary)

# Análisis de desbalance de clases
print_subsection("Distribución de Clases (Desbalance)")
class_dist <- table(df$diabetes)
class_prop <- prop.table(class_dist)

cat("\nConteo absoluto:\n")
print(class_dist)

cat("\nProporción relativa:\n")
print(class_prop)

cat(sprintf("\nDesbalance: %.2f%% negativos vs %.2f%% positivos\n", 
            class_prop["neg"] * 100, class_prop["pos"] * 100))
cat(sprintf("Ratio neg:pos = %.2f:1\n", class_prop["neg"] / class_prop["pos"]))

cat("\n⚠️  IMPORTANTE: El dataset está desbalanceado hacia la clase negativa.\n")
cat("   Esto afectará las métricas de rendimiento y justifica el uso del umbral ε.\n")


#===============================================================================
# 2. DIVISIÓN ESTRATIFICADA DE DATOS (80% - 20%)
#===============================================================================

print_section("2. DIVISIÓN ESTRATIFICADA DE DATOS")

# Función para muestreo estratificado
stratified_split <- function(data, target_col, train_prop = 0.8, seed = 123) {
  set.seed(seed)
  
  # Obtener índices por clase
  classes <- unique(data[[target_col]])
  train_indices <- c()
  
  for (cls in classes) {
    cls_indices <- which(data[[target_col]] == cls)
    n_train <- floor(length(cls_indices) * train_prop)
    train_cls_indices <- sample(cls_indices, n_train)
    train_indices <- c(train_indices, train_cls_indices)
  }
  
  # Crear conjuntos
  train_data <- data[train_indices, ]
  test_data <- data[-train_indices, ]
  
  return(list(train = train_data, test = test_data, train_indices = train_indices))
}

# Aplicar división estratificada
split_result <- stratified_split(df, "diabetes", train_prop = 0.8, seed = 123)
train <- split_result$train
test <- split_result$test

cat(sprintf("Total de observaciones: %d\n", nrow(df)))
cat(sprintf("Conjunto de entrenamiento: %d (%.1f%%)\n", nrow(train), 100*nrow(train)/nrow(df)))
cat(sprintf("Conjunto de prueba: %d (%.1f%%)\n\n", nrow(test), 100*nrow(test)/nrow(df)))

# Verificar proporciones de clases
print_subsection("Verificación de Proporciones por Clase")

prop_original <- prop.table(table(df$diabetes))
prop_train <- prop.table(table(train$diabetes))
prop_test <- prop.table(table(test$diabetes))

comparison_df <- data.frame(
  Conjunto = c("Original", "Entrenamiento", "Prueba"),
  Negativos = c(prop_original["neg"], prop_train["neg"], prop_test["neg"]) * 100,
  Positivos = c(prop_original["pos"], prop_train["pos"], prop_test["pos"]) * 100
)

print(comparison_df)

cat(sprintf("\nDiferencia absoluta entre train y test (clase pos): %.4f%%\n",
            abs(prop_train["pos"] - prop_test["pos"]) * 100))

cat("\n✓ División estratificada completada con éxito.\n")
cat("  Las proporciones de clases son prácticamente idénticas en ambos conjuntos.\n")


#===============================================================================
# 3. CÁLCULO DE ESTADÍSTICAS POR CLASE (μ y σ)
#===============================================================================

print_section("3. CÁLCULO DE PARÁMETROS DEL MODELO")

print_subsection("Medias y Desviaciones Estándar por Clase")

# Identificar variables predictoras numéricas (todas excepto 'diabetes')
predictor_vars <- setdiff(names(train), "diabetes")
cat(sprintf("Variables predictoras: %d\n", length(predictor_vars)))
cat("Variables:", paste(predictor_vars, collapse = ", "), "\n\n")

# Calcular estadísticas por clase usando dplyr
class_stats <- train %>%
  group_by(diabetes) %>%
  summarise(across(all_of(predictor_vars), 
                   list(mean = mean, sd = sd), 
                   .names = "{.col}_{.fn}"))

print(class_stats)

# Extraer estadísticas en formato de vectores para cálculos
mean_pos <- class_stats %>% 
  filter(diabetes == "pos") %>% 
  select(ends_with("_mean")) %>% 
  as.numeric()

sd_pos <- class_stats %>% 
  filter(diabetes == "pos") %>% 
  select(ends_with("_sd")) %>% 
  as.numeric()

mean_neg <- class_stats %>% 
  filter(diabetes == "neg") %>% 
  select(ends_with("_mean")) %>% 
  as.numeric()

sd_neg <- class_stats %>% 
  filter(diabetes == "neg") %>% 
  select(ends_with("_sd")) %>% 
  as.numeric()

# Manejo de desviaciones estándar cero o muy pequeñas
print_subsection("Manejo de Varianzas Pequeñas o Cero")

sd_threshold <- 1e-6
sd_pos_zero <- sum(sd_pos < sd_threshold | is.na(sd_pos))
sd_neg_zero <- sum(sd_neg < sd_threshold | is.na(sd_neg))

if (sd_pos_zero > 0 | sd_neg_zero > 0) {
  cat(sprintf("⚠️  Variables con desviación estándar < %.1e:\n", sd_threshold))
  cat(sprintf("   Clase positiva: %d variables\n", sd_pos_zero))
  cat(sprintf("   Clase negativa: %d variables\n", sd_neg_zero))
  
  # Identificar cuáles son
  if (sd_pos_zero > 0) {
    problematic_vars_pos <- predictor_vars[sd_pos < sd_threshold | is.na(sd_pos)]
    cat("   Variables (pos):", paste(problematic_vars_pos, collapse = ", "), "\n")
  }
  if (sd_neg_zero > 0) {
    problematic_vars_neg <- predictor_vars[sd_neg < sd_threshold | is.na(sd_neg)]
    cat("   Variables (neg):", paste(problematic_vars_neg, collapse = ", "), "\n")
  }
  
  # Aplicar ajuste
  sd_pos[sd_pos < sd_threshold | is.na(sd_pos)] <- sd_threshold
  sd_neg[sd_neg < sd_threshold | is.na(sd_neg)] <- sd_threshold
  
  cat(sprintf("\n   Ajuste aplicado: σ_min = %.1e para prevenir inestabilidad numérica.\n", sd_threshold))
} else {
  cat("✓ Todas las desviaciones estándar son apropiadas (σ >= %.1e).\n", sd_threshold)
}

# Probabilidades previas (prior probabilities)
print_subsection("Probabilidades Previas")

p_pos <- mean(train$diabetes == "pos")
p_neg <- 1 - p_pos

cat(sprintf("P(Positivo) = %d/%d = %.6f\n", sum(train$diabetes == "pos"), nrow(train), p_pos))
cat(sprintf("P(Negativo) = %d/%d = %.6f\n", sum(train$diabetes == "neg"), nrow(train), p_neg))

cat("\nEstas probabilidades previas reflejan el desbalance de clases en los datos.\n")


#===============================================================================
# 4. IMPLEMENTACIÓN DEL CLASIFICADOR BAYESIANO NAIVE
#===============================================================================

print_section("4. CLASIFICACIÓN BAYESIANA CON LOG-VEROSIMILITUDES")

print_subsection("Implementación Matemática")

cat("Fórmula de clasificación (del PDF):\n")
cat("  P(x_ij|Clase) = (1/√(2πσ²)) * exp(-(x_ij - μ)²/(2σ²))\n")
cat("  Implementada con: dnorm(x, mean = μ, sd = σ)\n\n")

cat("Log-verosimilitud posterior:\n")
cat("  log(P(Clase|X)) = log(P(Clase)) + Σ log(P(x_ij|Clase))\n\n")

# Función para calcular log-verosimilitud (vectorizada)
calculate_log_likelihood <- function(X_matrix, class_means, class_sds) {
  # X_matrix: matriz de predictores (n × p)
  # class_means: vector de medias por variable (p)
  # class_sds: vector de desviaciones estándar por variable (p)
  
  # Calcular log-densidades para cada observación y variable
  log_densities <- matrix(0, nrow = nrow(X_matrix), ncol = ncol(X_matrix))
  
  for (j in 1:ncol(X_matrix)) {
    log_densities[, j] <- dnorm(X_matrix[, j], 
                                mean = class_means[j], 
                                sd = class_sds[j], 
                                log = TRUE)
  }
  
  # Sumar log-densidades por fila (por observación)
  log_likelihoods <- rowSums(log_densities)
  
  return(log_likelihoods)
}

# Preparar datos de prueba
X_test <- as.matrix(test[, predictor_vars])
y_test <- test$diabetes

cat("Calculando log-verosimilitudes para", nrow(X_test), "observaciones de prueba...\n")

# Calcular log-verosimilitudes para ambas clases
log_likelihood_pos <- log(p_pos) + calculate_log_likelihood(X_test, mean_pos, sd_pos)
log_likelihood_neg <- log(p_neg) + calculate_log_likelihood(X_test, mean_neg, sd_neg)

# Diferencia de log-verosimilitudes (score de clasificación)
log_diff <- log_likelihood_pos - log_likelihood_neg

cat("✓ Cálculo completado.\n")
cat(sprintf("  Rango de log(P(pos|X)) - log(P(neg|X)): [%.3f, %.3f]\n", 
            min(log_diff), max(log_diff)))


#===============================================================================
# 5. EVALUACIÓN CON MÚLTIPLES UMBRALES (ε)
#===============================================================================

print_section("5. EVALUACIÓN DE MÉTRICAS CON DIFERENTES UMBRALES")

print_subsection("Barrido de Umbrales ε ∈ [-3, 3]")

# Definir rango de umbrales (exactamente 100 valores equiespaciados)
epsilon_values <- seq(-3, 3, length.out = 100)
cat(sprintf("Número de umbrales evaluados: %d\n", length(epsilon_values)))
cat(sprintf("Rango: [%.3f, %.3f]\n", min(epsilon_values), max(epsilon_values)))

# Función para calcular métricas desde matriz de confusión
calculate_metrics <- function(cm) {
  # Extraer valores de la matriz de confusión con manejo de NA
  TP <- ifelse(!is.na(cm["pos", "pos"]), cm["pos", "pos"], 0)
  TN <- ifelse(!is.na(cm["neg", "neg"]), cm["neg", "neg"], 0)
  FP <- ifelse(!is.na(cm["pos", "neg"]), cm["pos", "neg"], 0)
  FN <- ifelse(!is.na(cm["neg", "pos"]), cm["neg", "pos"], 0)
  
  # Calcular métricas
  Accuracy <- (TP + TN) / sum(cm)
  Sensitivity <- ifelse((TP + FN) > 0, TP / (TP + FN), NA)  # Recall, TPR
  Specificity <- ifelse((TN + FP) > 0, TN / (TN + FP), NA)  # TNR
  Precision <- ifelse((TP + FP) > 0, TP / (TP + FP), NA)
  F1_Score <- ifelse((2 * TP + FP + FN) > 0, (2 * TP) / (2 * TP + FP + FN), NA)
  
  return(c(Accuracy = Accuracy, 
           Sensitivity = Sensitivity, 
           Specificity = Specificity,
           Precision = Precision,
           F1_Score = F1_Score,
           TP = TP, TN = TN, FP = FP, FN = FN))
}

# Evaluar para cada umbral
cat("Evaluando métricas para cada umbral...\n")

results <- data.frame(
  epsilon = epsilon_values,
  Accuracy = NA,
  Sensitivity = NA,
  Specificity = NA,
  Precision = NA,
  F1_Score = NA,
  TP = NA, TN = NA, FP = NA, FN = NA
)

for (k in 1:length(epsilon_values)) {
  eps <- epsilon_values[k]
  
  # Clasificar usando el umbral actual
  predictions <- ifelse(log_diff > eps, "pos", "neg")
  predictions <- factor(predictions, levels = c("neg", "pos"))
  
  # Matriz de confusión
  cm <- table(Predicted = predictions, Actual = y_test)
  
  # Calcular métricas
  metrics <- calculate_metrics(cm)
  results[k, 2:10] <- metrics
}

cat("✓ Evaluación completada.\n")


#===============================================================================
# 6. IDENTIFICACIÓN DEL UMBRAL ÓPTIMO
#===============================================================================

print_section("6. SELECCIÓN DEL UMBRAL ÓPTIMO")

print_subsection("Criterio: Maximización del F1-Score")

# Encontrar umbral óptimo
best_idx <- which.max(results$F1_Score)
epsilon_optimal <- results$epsilon[best_idx]
best_F1 <- results$F1_Score[best_idx]
best_Accuracy <- results$Accuracy[best_idx]
best_Sensitivity <- results$Sensitivity[best_idx]
best_Specificity <- results$Specificity[best_idx]
best_Precision <- results$Precision[best_idx]

cat(sprintf("Umbral óptimo (ε*): %.6f\n", epsilon_optimal))
cat(sprintf("F1-Score máximo: %.6f\n\n", best_F1))

print_subsection("Métricas en el Umbral Óptimo (ε*)")
cat(sprintf("  Accuracy:    %.4f (%.2f%%)\n", best_Accuracy, best_Accuracy * 100))
cat(sprintf("  Sensitivity: %.4f (%.2f%%) - Recall, capacidad de detectar positivos\n", 
            best_Sensitivity, best_Sensitivity * 100))
cat(sprintf("  Specificity: %.4f (%.2f%%) - Capacidad de detectar negativos\n", 
            best_Specificity, best_Specificity * 100))
cat(sprintf("  Precision:   %.4f (%.2f%%) - De los predichos positivos, cuántos son reales\n", 
            best_Precision, best_Precision * 100))
cat(sprintf("  F1-Score:    %.4f - Media armónica de Precision y Recall\n", best_F1))

# Matriz de confusión en ε*
print_subsection("Matriz de Confusión en ε*")
pred_optimal <- ifelse(log_diff > epsilon_optimal, "pos", "neg")
pred_optimal <- factor(pred_optimal, levels = c("neg", "pos"))
cm_optimal <- table(Predicted = pred_optimal, Actual = y_test)
print(cm_optimal)

cat("\nInterpretación:\n")
cat(sprintf("  Verdaderos Positivos (TP): %d - Casos positivos correctamente identificados\n", 
            results$TP[best_idx]))
cat(sprintf("  Verdaderos Negativos (TN): %d - Casos negativos correctamente identificados\n", 
            results$TN[best_idx]))
cat(sprintf("  Falsos Positivos (FP): %d - Casos negativos incorrectamente clasificados como positivos\n", 
            results$FP[best_idx]))
cat(sprintf("  Falsos Negativos (FN): %d - Casos positivos incorrectamente clasificados como negativos\n", 
            results$FN[best_idx]))

# Comparar con umbral neutro ε = 0
print_subsection("Comparación con Umbral Neutro (ε = 0)")
idx_zero <- which.min(abs(results$epsilon))
eps_zero <- results$epsilon[idx_zero]

cat(sprintf("Umbral neutro ε ≈ %.6f:\n", eps_zero))
cat(sprintf("  Accuracy:    %.4f (%.2f%%)\n", results$Accuracy[idx_zero], results$Accuracy[idx_zero] * 100))
cat(sprintf("  Sensitivity: %.4f (%.2f%%)\n", results$Sensitivity[idx_zero], results$Sensitivity[idx_zero] * 100))
cat(sprintf("  Specificity: %.4f (%.2f%%)\n", results$Specificity[idx_zero], results$Specificity[idx_zero] * 100))
cat(sprintf("  F1-Score:    %.4f\n\n", results$F1_Score[idx_zero]))

cat("Mejora al usar ε* vs ε=0:\n")
cat(sprintf("  ΔF1-Score:    %+.4f (%.2f%%)\n", 
            best_F1 - results$F1_Score[idx_zero],
            100 * (best_F1 - results$F1_Score[idx_zero]) / results$F1_Score[idx_zero]))
cat(sprintf("  ΔSensitivity: %+.4f\n", best_Sensitivity - results$Sensitivity[idx_zero]))
cat(sprintf("  ΔSpecificity: %+.4f\n", best_Specificity - results$Specificity[idx_zero]))

# Tabla comparativa de umbrales clave
print_subsection("Tabla Comparativa: Umbrales Clave")

key_epsilons <- c(-1, 0, epsilon_optimal, 1)
comparison_table <- data.frame(
  Umbral = character(),
  Epsilon = numeric(),
  Accuracy = numeric(),
  Sensitivity = numeric(),
  Specificity = numeric(),
  F1_Score = numeric()
)

for (eps_val in key_epsilons) {
  idx <- which.min(abs(results$epsilon - eps_val))
  label <- if (abs(results$epsilon[idx] - epsilon_optimal) < 0.001) "ε* (óptimo)" else sprintf("ε = %.1f", eps_val)
  
  comparison_table <- rbind(comparison_table, data.frame(
    Umbral = label,
    Epsilon = results$epsilon[idx],
    Accuracy = results$Accuracy[idx],
    Sensitivity = results$Sensitivity[idx],
    Specificity = results$Specificity[idx],
    F1_Score = results$F1_Score[idx]
  ))
}

print(comparison_table)


#===============================================================================
# 7. VISUALIZACIONES
#===============================================================================

print_section("7. VISUALIZACIONES")

print_subsection("Gráfico 1: Métricas vs. Umbral de Decisión (ε)")

# Preparar datos en formato largo para ggplot2
results_long <- results %>%
  select(epsilon, Accuracy, Sensitivity, Specificity, F1_Score) %>%
  pivot_longer(cols = c("Accuracy", "Sensitivity", "Specificity", "F1_Score"),
               names_to = "Metric", 
               values_to = "Value")

# (opción manual sin tidyr):
#results_long <- rbind(
#  data.frame(epsilon = results$epsilon, Metric = "Accuracy", Value = results$Accuracy),
#  data.frame(epsilon = results$epsilon, Metric = "Sensitivity", Value = results$Sensitivity),
#  data.frame(epsilon = results$epsilon, Metric = "Specificity", Value = results$Specificity),
#  data.frame(epsilon = results$epsilon, Metric = "F1_Score", Value = results$F1_Score)
#)

# Crear gráfico principal
p1 <- ggplot(results_long, aes(x = epsilon, y = Value, color = Metric)) +
  geom_line(size = 1.2, alpha = 0.9) +
  geom_vline(xintercept = epsilon_optimal, 
             linetype = "dashed", 
             color = "black", 
             size = 0.8) +
  geom_point(data = results_long[results_long$Metric == "F1_Score", ],
             aes(x = epsilon_optimal, y = best_F1),
             color = "red", size = 3, shape = 19) +
  annotate("text", 
           x = epsilon_optimal, 
           y = 0.95, 
           label = sprintf("ε* = %.3f\nF1 = %.3f", epsilon_optimal, best_F1),
           hjust = -0.1, 
           size = 4, 
           fontface = "bold",
           color = "red") +
  scale_color_manual(values = c("Accuracy" = "#1f77b4", 
                                "Sensitivity" = "#ff7f0e",
                                "Specificity" = "#2ca02c",
                                "F1_Score" = "#d62728"),
                     labels = c("Accuracy", "Sensitivity (Recall)", "Specificity", "F1-Score")) +
  theme_minimal(base_size = 12) +
  labs(title = "Efecto del Umbral (ε) en las Métricas del Clasificador Bayesiano",
       subtitle = sprintf("Umbral óptimo: ε* = %.4f (maximiza F1-Score = %.4f)", 
                          epsilon_optimal, best_F1),
       x = "Umbral de decisión (ε)",
       y = "Valor de la métrica",
       color = "Métrica") +
  theme(
    plot.title = element_text(size = 14, face = "bold", hjust = 0.5),
    plot.subtitle = element_text(size = 11, hjust = 0.5),
    legend.position = "bottom",
    legend.title = element_text(face = "bold"),
    panel.grid.major = element_line(color = "gray90"),
    panel.grid.minor = element_line(color = "gray95")
  ) +
  ylim(0, 1)

print(p1)



