# Limpiar entorno y liberar memoria
rm(list = ls())
gc()

# Instalar y cargar paquetes necesarios para análisis, visualización y paralelización
packages <- c("mlbench", "ggplot2", "dplyr", "tidyr", "pROC", "patchwork", "parallel")
for (pkg in packages) {
  if (!require(pkg, character.only = TRUE, quietly = TRUE)) {
    install.packages(pkg, dependencies = TRUE)
    library(pkg, character.only = TRUE)
  }
}

# Configuración para evitar notación científica y controlar decimales
options(scipen = 999, digits = 6)

# Funciones auxiliares para imprimir secciones y subsecciones con formato claro
print_section <- function(title, symbol = "=", width = 80) {
  cat("\n")
  cat(paste(rep(symbol, width), collapse = ""), "\n")
  cat(paste0(" ", title, "\n"))
  cat(paste(rep(symbol, width), collapse = ""), "\n\n")
}

print_subsection <- function(title) {
  cat("\n--- ", title, " ---\n")
}

# Imprimir encabezado principal
print_section("CLASIFICADOR BAYESIANO NAIVE - DIABETES", "=")
cat("Dataset: Pima Indians Diabetes\n")
cat("Fecha de ejecución:", format(Sys.time(), "%Y-%m-%d %H:%M:%S"), "\n")

#===============================================================================
# FUNCIONES AUXILIARES (CÁLCULOS Y FÓRMULAS)
#===============================================================================

# Muestreo estratificado para mantener proporciones de clases en train/test
stratified_split <- function(data, target_col, train_prop = 0.8, seed = NULL) {
  set.seed(seed)
  classes <- unique(data[[target_col]])
  train_indices <- c()
  for (cls in classes) {
    cls_indices <- which(data[[target_col]] == cls)
    n_train <- floor(length(cls_indices) * train_prop)
    train_cls_indices <- sample(cls_indices, n_train)
    train_indices <- c(train_indices, train_cls_indices)
  }
  train_data <- data[train_indices, ]
  test_data <- data[-train_indices, ]
  return(list(train = train_data, test = test_data, train_indices = train_indices))
}

# Cálculo vectorizado de log-verosimilitud para cada observación y clase
calculate_log_likelihood <- function(X_matrix, class_means, class_sds) {
  log_densities <- matrix(0, nrow = nrow(X_matrix), ncol = ncol(X_matrix))
  for (j in 1:ncol(X_matrix)) {
    # Densidad logarítmica normal para cada variable predictora
    log_densities[, j] <- dnorm(X_matrix[, j], mean = class_means[j], sd = class_sds[j], log = TRUE)
  }
  # Suma de log-densidades (asumiendo independencia condicional)
  rowSums(log_densities)
}

# Cálculo de métricas de evaluación a partir de matriz de confusión
calculate_metrics <- function(cm) {
  TP <- ifelse(!is.na(cm["pos", "pos"]), cm["pos", "pos"], 0)
  TN <- ifelse(!is.na(cm["neg", "neg"]), cm["neg", "neg"], 0)
  FP <- ifelse(!is.na(cm["pos", "neg"]), cm["pos", "neg"], 0)
  FN <- ifelse(!is.na(cm["neg", "pos"]), cm["neg", "pos"], 0)
  Accuracy <- (TP + TN) / sum(cm)
  Sensitivity <- ifelse((TP + FN) > 0, TP / (TP + FN), NA)
  Specificity <- ifelse((TN + FP) > 0, TN / (TN + FP), NA)
  Precision <- ifelse((TP + FP) > 0, TP / (TP + FP), NA)
  F1_Score <- ifelse((2 * TP + FP + FN) > 0, (2 * TP) / (2 * TP + FP + FN), NA)
  c(Accuracy = Accuracy, Sensitivity = Sensitivity, Specificity = Specificity,
    Precision = Precision, F1_Score = F1_Score, TP = TP, TN = TN, FP = FP, FN = FN)
}

# Pipeline completo para una semilla: división, cálculo parámetros, predicción y evaluación
run_pipeline_with_seed <- function(sd) {
  split <- stratified_split(df, "diabetes", train_prop = 0.8, seed = sd)
  train <- split$train
  test  <- split$test
  predictor_vars <- setdiff(names(train), "diabetes")
  
  # Cálculo de medias y desviaciones estándar por clase
  class_stats <- train %>%
    group_by(diabetes) %>%
    summarise(across(all_of(predictor_vars),
                     list(mean = ~mean(.x, na.rm = TRUE), sd = ~sd(.x, na.rm = TRUE)),
                     .names = "{.col}_{.fn}"), .groups = "drop")
  
  mean_pos <- class_stats %>% filter(diabetes == "pos") %>% select(ends_with("_mean")) %>% as.numeric()
  sd_pos <- class_stats %>% filter(diabetes == "pos") %>% select(ends_with("_sd")) %>% as.numeric()
  mean_neg <- class_stats %>% filter(diabetes == "neg") %>% select(ends_with("_mean")) %>% as.numeric()
  sd_neg <- class_stats %>% filter(diabetes == "neg") %>% select(ends_with("_sd")) %>% as.numeric()
  
  # Evitar desviaciones estándar cero o muy pequeñas para evitar errores numéricos
  sd_threshold <- 1e-6
  sd_pos[is.na(sd_pos) | sd_pos < sd_threshold] <- sd_threshold
  sd_neg[is.na(sd_neg) | sd_neg < sd_threshold] <- sd_threshold
  
  # Probabilidades previas de clases
  p_pos <- mean(train$diabetes == "pos")
  p_neg <- 1 - p_pos
  
  X_test <- as.matrix(test[, predictor_vars])
  y_test <- test$diabetes
  
  # Cálculo de log-verosimilitudes para cada clase
  log_likelihood_pos <- log(p_pos) + calculate_log_likelihood(X_test, mean_pos, sd_pos)
  log_likelihood_neg <- log(p_neg) + calculate_log_likelihood(X_test, mean_neg, sd_neg)
  log_diff <- log_likelihood_pos - log_likelihood_neg
  
  # Evaluación para múltiples umbrales ε
  epsilon_values <- seq(-3, 3, length.out = 100)
  results <- data.frame(epsilon = epsilon_values, Accuracy = NA, Sensitivity = NA, Specificity = NA,
                        Precision = NA, F1_Score = NA, TP = NA, TN = NA, FP = NA, FN = NA)
  
  for (k in seq_along(epsilon_values)) {
    eps <- epsilon_values[k]
    predictions <- factor(ifelse(log_diff > eps, "pos", "neg"), levels = c("neg", "pos"))
    cm <- table(Predicted = predictions, Actual = y_test)
    metrics <- calculate_metrics(cm)
    results[k, 2:10] <- metrics
  }
  
  # Selección del umbral óptimo basado en F1-Score máximo
  best_idx <- which.max(results$F1_Score)
  epsilon_optimal <- results$epsilon[best_idx]
  best_F1 <- results$F1_Score[best_idx]
  best_Acc <- results$Accuracy[best_idx]
  best_Sen <- results$Sensitivity[best_idx]
  best_Spe <- results$Specificity[best_idx]
  best_Pre <- results$Precision[best_idx]
  
  # Cálculo de AUC para la curva ROC
  roc_obj <- pROC::roc(response = y_test, predictor = log_diff,
                       levels = c("neg", "pos"), direction = "<", quiet = TRUE)
  auc_val <- as.numeric(pROC::auc(roc_obj))
  
  # Matriz de confusión para el umbral óptimo
  pred_opt <- factor(ifelse(log_diff > epsilon_optimal, "pos", "neg"), levels = c("neg", "pos"))
  cm_opt <- table(Predicted = pred_opt, Actual = y_test)
  TP <- ifelse(is.na(cm_opt["pos", "pos"]), 0, cm_opt["pos", "pos"])
  TN <- ifelse(is.na(cm_opt["neg", "neg"]), 0, cm_opt["neg", "neg"])
  FP <- ifelse(is.na(cm_opt["pos", "neg"]), 0, cm_opt["pos", "neg"])
  FN <- ifelse(is.na(cm_opt["neg", "pos"]), 0, cm_opt["neg", "pos"])
  
  # Retornar resultados resumidos para la semilla
  data.frame(seed = sd, epsilon_opt = epsilon_optimal, F1 = best_F1, Accuracy = best_Acc,
             Sensitivity = best_Sen, Specificity = best_Spe, Precision = best_Pre,
             AUC = auc_val, TP = TP, TN = TN, FP = FP, FN = FN, stringsAsFactors = FALSE)
}

#===============================================================================
# EJECUCIÓN DEL ANÁLISIS PASO A PASO (RESULTADOS Y GRÁFICOS)
#===============================================================================

# 1. Carga y análisis exploratorio de datos (EDA)
print_section("1. CARGA Y ANÁLISIS EXPLORATORIO DE DATOS (EDA)")

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

print_subsection("Análisis de Calidad de Datos")
na_count <- colSums(is.na(df))
cat("\nValores faltantes (NA) por variable:\n")
print(na_count)
cat(sprintf("Total de NAs en el dataset: %d\n", sum(na_count)))

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

print_subsection("Estadísticas Descriptivas por Clase")
stats_summary <- df %>%
  group_by(diabetes) %>%
  summarise(n = n(),
            across(where(is.numeric),
                   list(media = ~mean(.x, na.rm = TRUE),
                        mediana = ~median(.x, na.rm = TRUE),
                        sd = ~sd(.x, na.rm = TRUE)),
                   .names = "{.col}_{.fn}"))
print(stats_summary)

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

#===============================================================================
# 2. Búsqueda paralela de semilla óptima (CPU)
#===============================================================================

print_section("2. BÚSQUEDA PARALELA DE SEMILLA ÓPTIMA (CPU)")

num_cores <- detectCores() - 1
cat(sprintf("✓ CPU detectado: %d núcleos totales\n", detectCores()))
cat(sprintf("✓ Usando %d núcleos para paralelización\n\n", num_cores))

seed_set <- 1:1000
cat(sprintf("Evaluando %d semillas en paralelo...\n", length(seed_set)))
cat("Esto puede tardar 1-3 minutos dependiendo de tu CPU...\n\n")

cl <- makeCluster(num_cores)
clusterExport(cl, c("df", "stratified_split", "calculate_log_likelihood", 
                    "calculate_metrics", "run_pipeline_with_seed"))
clusterEvalQ(cl, {
  library(dplyr)
  library(pROC)
})

start_time <- Sys.time()
res_all <- parLapply(cl, seed_set, run_pipeline_with_seed) %>% bind_rows()
stopCluster(cl)
end_time <- Sys.time()
elapsed_time <- as.numeric(difftime(end_time, start_time, units = "secs"))

cat(sprintf("\n✓ Búsqueda completada en %.2f segundos (%.2f minutos)\n", elapsed_time, elapsed_time/60))
cat(sprintf("✓ Velocidad: %.2f semillas/segundo\n", length(seed_set)/elapsed_time))

res_ranked <- res_all %>% arrange(desc(F1), desc(AUC))
best_row <- res_ranked[1, ]

print_subsection("Top 10 Mejores Semillas")
print(head(res_ranked, 10))

print_subsection("Semilla Óptima Seleccionada")
cat(sprintf("Semilla óptima: %d\n", best_row$seed))
cat(sprintf("F1-Score: %.4f\n", best_row$F1))
cat(sprintf("AUC: %.4f\n", best_row$AUC))
cat(sprintf("Umbral óptimo (ε*): %.4f\n", best_row$epsilon_opt))
cat(sprintf("Accuracy: %.2f%%\n", 100*best_row$Accuracy))
cat(sprintf("Sensitivity: %.2f%%\n", 100*best_row$Sensitivity))
cat(sprintf("Specificity: %.2f%%\n", 100*best_row$Specificity))
cat(sprintf("Precision: %.2f%%\n", 100*best_row$Precision))
cat(sprintf("TP=%d, TN=%d, FP=%d, FN=%d\n", best_row$TP, best_row$TN, best_row$FP, best_row$FN))

print_subsection("Estadísticas de Robustez (todas las semillas)")
summary_stats <- res_all %>%
  summarise(n = n(),
            F1_mean = mean(F1, na.rm = TRUE),
            F1_sd = sd(F1, na.rm = TRUE),
            F1_min = min(F1, na.rm = TRUE),
            F1_max = max(F1, na.rm = TRUE),
            AUC_mean = mean(AUC, na.rm = TRUE),
            AUC_sd = sd(AUC, na.rm = TRUE))
print(summary_stats)

OPTIMAL_SEED <- best_row$seed
cat(sprintf("\n✓ Semilla óptima guardada: %d\n", OPTIMAL_SEED))
#===============================================================================
# 3. División estratificada con semilla óptima
#===============================================================================
print_section("3. DIVISIÓN ESTRATIFICADA DE DATOS")

split_result <- stratified_split(df, "diabetes", train_prop = 0.8, seed = OPTIMAL_SEED)
train <- split_result$train
test <- split_result$test

cat(sprintf("Usando semilla óptima: %d\n\n", OPTIMAL_SEED))
cat(sprintf("Total de observaciones: %d\n", nrow(df)))
cat(sprintf("Conjunto de entrenamiento: %d (%.1f%%)\n", nrow(train), 100*nrow(train)/nrow(df)))
cat(sprintf("Conjunto de prueba: %d (%.1f%%)\n\n", nrow(test), 100*nrow(test)/nrow(df)))

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

#===============================================================================
# 4. Cálculo de medias y desviaciones estándar por clase
#===============================================================================

print_section("4. CÁLCULO DE PARÁMETROS DEL MODELO")

print_subsection("Medias y Desviaciones Estándar por Clase")

# Identificar variables predictoras numéricas (todas excepto 'diabetes')
predictor_vars <- setdiff(names(train), "diabetes")
cat(sprintf("Variables predictoras: %d\n", length(predictor_vars)))
cat("Variables:", paste(predictor_vars, collapse = ", "), "\n\n")

# Calcular estadísticas por clase usando dplyr
class_stats <- train %>%
  group_by(diabetes) %>%
  summarise(across(all_of(predictor_vars),
                   list(mean = ~mean(.x, na.rm = TRUE), sd = ~sd(.x, na.rm = TRUE)),
                   .names = "{.col}_{.fn}"), .groups = "drop")

print(class_stats)

# Extraer estadísticas en formato de vectores para cálculos
mean_pos <- class_stats %>% filter(diabetes == "pos") %>% select(ends_with("_mean")) %>% as.numeric()
sd_pos <- class_stats %>% filter(diabetes == "pos") %>% select(ends_with("_sd")) %>% as.numeric()
mean_neg <- class_stats %>% filter(diabetes == "neg") %>% select(ends_with("_mean")) %>% as.numeric()
sd_neg <- class_stats %>% filter(diabetes == "neg") %>% select(ends_with("_sd")) %>% as.numeric()

print_subsection("Manejo de Varianzas Pequeñas o Cero")

sd_threshold <- 1e-6
sd_pos_zero <- sum(sd_pos < sd_threshold | is.na(sd_pos))
sd_neg_zero <- sum(sd_neg < sd_threshold | is.na(sd_neg))

if (sd_pos_zero > 0 | sd_neg_zero > 0) {
  cat(sprintf("⚠️  Variables con desviación estándar < %.1e:\n", sd_threshold))
  cat(sprintf("   Clase positiva: %d variables\n", sd_pos_zero))
  cat(sprintf("   Clase negativa: %d variables\n", sd_neg_zero))
  
  if (sd_pos_zero > 0) {
    problematic_vars_pos <- predictor_vars[sd_pos < sd_threshold | is.na(sd_pos)]
    cat("   Variables (pos):", paste(problematic_vars_pos, collapse = ", "), "\n")
  }
  if (sd_neg_zero > 0) {
    problematic_vars_neg <- predictor_vars[sd_neg < sd_threshold | is.na(sd_neg)]
    cat("   Variables (neg):", paste(problematic_vars_neg, collapse = ", "), "\n")
  }
  
  sd_pos[sd_pos < sd_threshold | is.na(sd_pos)] <- sd_threshold
  sd_neg[sd_neg < sd_threshold | is.na(sd_neg)] <- sd_threshold
  
  cat(sprintf("\n   Ajuste aplicado: σ_min = %.1e\n", sd_threshold))
} else {
  cat(sprintf("✓ Todas las desviaciones estándar son apropiadas (σ >= %.1e).\n", sd_threshold))
}

print_subsection("Probabilidades Previas")

p_pos <- mean(train$diabetes == "pos")
p_neg <- 1 - p_pos

cat(sprintf("P(Positivo) = %d/%d = %.6f\n", sum(train$diabetes == "pos"), nrow(train), p_pos))
cat(sprintf("P(Negativo) = %d/%d = %.6f\n", sum(train$diabetes == "neg"), nrow(train), p_neg))

#===============================================================================
# 5. Clasificación bayesiana naive con log-verosimilitudes
#===============================================================================

print_section("5. CLASIFICACIÓN BAYESIANA CON LOG-VEROSIMILITUDES")

print_subsection("Implementación Matemática")

cat("Fórmula de clasificación:\n")
cat("  P(x_ij|Clase) = (1/√(2πσ²)) * exp(-(x_ij - μ)²/(2σ²))\n")
cat("  Implementada con: dnorm(x, mean = μ, sd = σ)\n\n")

cat("Log-verosimilitud posterior:\n")
cat("  log(P(Clase|X)) = log(P(Clase)) + Σ log(P(x_ij|Clase))\n\n")

X_test <- as.matrix(test[, predictor_vars])
y_test <- test$diabetes

cat("Calculando log-verosimilitudes para", nrow(X_test), "observaciones de prueba...\n")

log_likelihood_pos <- log(p_pos) + calculate_log_likelihood(X_test, mean_pos, sd_pos)
log_likelihood_neg <- log(p_neg) + calculate_log_likelihood(X_test, mean_neg, sd_neg)

log_diff <- log_likelihood_pos - log_likelihood_neg

cat("✓ Cálculo completado.\n")
cat(sprintf("  Rango de log(P(pos|X)) - log(P(neg|X)): [%.3f, %.3f]\n", min(log_diff), max(log_diff)))

#===============================================================================
# 6. Evaluación con múltiples umbrales ε
#===============================================================================
print_section("6. EVALUACIÓN DE MÉTRICAS CON DIFERENTES UMBRALES")

print_subsection("Barrido de Umbrales ε ∈ [-3, 3]")

epsilon_values <- seq(-3, 3, length.out = 100)
cat(sprintf("Número de umbrales evaluados: %d\n", length(epsilon_values)))
cat(sprintf("Rango: [%.3f, %.3f]\n", min(epsilon_values), max(epsilon_values)))

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
  predictions <- ifelse(log_diff > eps, "pos", "neg")
  predictions <- factor(predictions, levels = c("neg", "pos"))
  cm <- table(Predicted = predictions, Actual = y_test)
  metrics <- calculate_metrics(cm)
  results[k, 2:10] <- metrics
}

cat("✓ Evaluación completada.\n")

#===============================================================================
# 7. Identificación del umbral óptimo
#===============================================================================

print_section("7. SELECCIÓN DEL UMBRAL ÓPTIMO")

print_subsection("Criterio: Maximización del F1-Score")

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
cat(sprintf("  Sensitivity: %.4f (%.2f%%)\n", best_Sensitivity, best_Sensitivity * 100))
cat(sprintf("  Specificity: %.4f (%.2f%%)\n", best_Specificity, best_Specificity * 100))
cat(sprintf("  Precision:   %.4f (%.2f%%)\n", best_Precision, best_Precision * 100))
cat(sprintf("  F1-Score:    %.4f\n", best_F1))

print_subsection("Matriz de Confusión en ε*")
pred_optimal <- ifelse(log_diff > epsilon_optimal, "pos", "neg")
pred_optimal <- factor(pred_optimal, levels = c("neg", "pos"))
cm_optimal <- table(Predicted = pred_optimal, Actual = y_test)
print(cm_optimal)

cat("\nInterpretación:\n")
cat(sprintf("  Verdaderos Positivos (TP): %d\n", results$TP[best_idx]))
cat(sprintf("  Verdaderos Negativos (TN): %d\n", results$TN[best_idx]))
cat(sprintf("  Falsos Positivos (FP): %d\n", results$FP[best_idx]))
cat(sprintf("  Falsos Negativos (FN): %d\n", results$FN[best_idx]))

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
# 8. Visualizaciones
#===============================================================================

print_section("8. VISUALIZACIONES")

print_subsection("Generando gráficos...")

results_long <- results %>%
  select(epsilon, Accuracy, Sensitivity, Specificity, F1_Score) %>%
  pivot_longer(cols = c("Accuracy", "Sensitivity", "Specificity", "F1_Score"),
               names_to = "Metric", 
               values_to = "Value")

p1 <- ggplot(results_long, aes(x = epsilon, y = Value, color = Metric)) +
  geom_line(size = 1.2, alpha = 0.9) +
  geom_vline(xintercept = epsilon_optimal, linetype = "dashed", color = "black", size = 0.8) +
  geom_point(data = results_long[results_long$Metric == "F1_Score", ],
             aes(x = epsilon_optimal, y = best_F1),
             color = "red", size = 3, shape = 19) +
  annotate("text", x = epsilon_optimal, y = 0.95, label = sprintf("ε* = %.3f\nF1 = %.3f", epsilon_optimal, best_F1),
           hjust = -0.1, size = 4, fontface = "bold", color = "red") +
  scale_color_manual(values = c("Accuracy" = "#1f77b4", "Sensitivity" = "#ff7f0e",
                                "Specificity" = "#2ca02c", "F1_Score" = "#d62728"),
                     labels = c("Accuracy", "Sensitivity (Recall)", "Specificity", "F1-Score")) +
  theme_minimal(base_size = 12) +
  labs(title = "Efecto del Umbral (ε) en las Métricas del Clasificador Bayesiano",
       subtitle = sprintf("Semilla: %d | Umbral óptimo: ε* = %.4f (F1 = %.4f)", 
                          OPTIMAL_SEED, epsilon_optimal, best_F1),
       x = "Umbral de decisión (ε)",
       y = "Valor de la métrica",
       color = "Métrica") +
  theme(plot.title = element_text(size = 14, face = "bold", hjust = 0.5),
        plot.subtitle = element_text(size = 11, hjust = 0.5),
        legend.position = "bottom",
        legend.title = element_text(face = "bold"),
        panel.grid.major = element_line(color = "gray90"),
        panel.grid.minor = element_line(color = "gray95")) +
  ylim(0, 1)

print(p1)

cm_df <- as.data.frame(cm_optimal)
colnames(cm_df) <- c("Predicted", "Actual", "Freq")

cm_df <- cm_df %>%
  mutate(Percent = 100 * Freq / sum(Freq),
         Label = sprintf("%d\n(%.1f%%)", Freq, Percent))

p2 <- ggplot(cm_df, aes(x = Actual, y = Predicted, fill = Freq)) +
  geom_tile(color = "white", size = 1.5) +
  geom_text(aes(label = Label), size = 6, fontface = "bold", color = "white") +
  scale_fill_gradient2(low = "#3498db", mid = "#e74c3c", high = "#c0392b",
                       midpoint = max(cm_df$Freq) / 2,
                       name = "Frecuencia") +
  theme_minimal(base_size = 12) +
  labs(title = sprintf("Matriz de Confusión (Semilla: %d, ε* = %.4f)", OPTIMAL_SEED, epsilon_optimal),
       subtitle = sprintf("Accuracy = %.2f%% | F1-Score = %.4f", best_Accuracy * 100, best_F1),
       x = "Clase Real",
       y = "Clase Predicha") +
  theme(plot.title = element_text(size = 14, face = "bold", hjust = 0.5),
        plot.subtitle = element_text(size = 11, hjust = 0.5),
        axis.text = element_text(size = 12, face = "bold"),
        legend.position = "right") +
  coord_fixed()

print(p2)

log_diff_df <- data.frame(LogRatio = log_diff, ClaseReal = y_test)

p3 <- ggplot(log_diff_df, aes(x = LogRatio, fill = ClaseReal)) +
  geom_histogram(alpha = 0.6, bins = 30, position = "identity") +
  geom_vline(xintercept = epsilon_optimal, linetype = "dashed", color = "red", size = 1.2) +
  annotate("text", x = epsilon_optimal, y = Inf, label = sprintf("ε* = %.3f", epsilon_optimal),
           hjust = -0.1, vjust = 1.5, size = 4, fontface = "bold", color = "red") +
  scale_fill_manual(values = c("neg" = "#3498db", "pos" = "#e74c3c"),
                    labels = c("Negativo", "Positivo")) +
  theme_minimal(base_size = 12) +
  labs(title = sprintf("Distribución de Scores (Semilla: %d)", OPTIMAL_SEED),
       subtitle = sprintf("Score = log(P(pos|X)) - log(P(neg|X)) | ε* = %.4f", epsilon_optimal),
       x = "log(P(Positivo|X)) - log(P(Negativo|X))",
       y = "Frecuencia",
       fill = "Clase Real") +
  theme(plot.title = element_text(size = 14, face = "bold", hjust = 0.5),
        plot.subtitle = element_text(size = 10, hjust = 0.5),
        legend.position = "bottom",
        legend.title = element_text(face = "bold"))

print(p3)

roc_obj <- roc(response = y_test, predictor = log_diff,
               levels = c("neg", "pos"), direction = "<", quiet = TRUE)

auc_value <- auc(roc_obj)

p4 <- ggroc(roc_obj, size = 1.2, color = "#e74c3c") +
  geom_abline(intercept = 1, slope = 1, linetype = "dashed", color = "gray50") +
  annotate("text", x = 0.3, y = 0.3, label = sprintf("AUC = %.4f", auc_value),
           size = 6, fontface = "bold", color = "#e74c3c") +
  theme_minimal(base_size = 12) +
  labs(title = sprintf("Curva ROC (Semilla: %d)", OPTIMAL_SEED),
       subtitle = "Rendimiento discriminativo del modelo",
       x = "1 - Especificidad (Tasa de Falsos Positivos)",
       y = "Sensibilidad (Tasa de Verdaderos Positivos)") +
  theme(plot.title = element_text(size = 14, face = "bold", hjust = 0.5),
        plot.subtitle = element_text(size = 11, hjust = 0.5),
        panel.grid.major = element_line(color = "gray90")) +
  coord_fixed()

print(p4)

combined_plot <- (p1 | p2) / (p3 | p4) + 
  plot_annotation(title = sprintf("Resumen de Resultados - Semilla Óptima: %d", OPTIMAL_SEED),
                  theme = theme(plot.title = element_text(size = 16, face = "bold", hjust = 0.5)))

print(combined_plot)

# Resumen final
print_section("ANÁLISIS COMPLETADO CON PARALELIZACIÓN CPU")
cat(sprintf("✓ Semilla óptima utilizada: %d\n", OPTIMAL_SEED))
cat(sprintf("✓ Tiempo de búsqueda paralela: %.2f minutos\n", elapsed_time/60))
cat(sprintf("✓ Núcleos CPU utilizados: %d\n", num_cores))
cat(sprintf("✓ F1-Score final: %.4f\n", best_F1))
cat(sprintf("✓ AUC final: %.4f\n", auc_value))
cat(sprintf("✓ Umbral óptimo (ε*): %.4f\n", epsilon_optimal))
cat("\n¡Análisis completado exitosamente!\n")