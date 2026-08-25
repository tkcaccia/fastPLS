#!/usr/bin/env Rscript

args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 4L) {
  stop("Usage: worker.R CONFIG_RDS RESULT_RDS PID_FILE MEASUREMENT_DONE", call. = FALSE)
}

cfg <- readRDS(args[[1L]])
result_path <- args[[2L]]
pid_file <- args[[3L]]
measurement_done <- args[[4L]]
dir.create(dirname(result_path), recursive = TRUE, showWarnings = FALSE)
writeLines(as.character(Sys.getpid()), pid_file)

suppressPackageStartupMessages(library(fastPLS))

rss_mb <- function() {
  if (!requireNamespace("ps", quietly = TRUE)) return(NA_real_)
  info <- tryCatch(ps::ps_memory_info(ps::ps_handle()), error = function(e) NULL)
  if (is.null(info)) return(NA_real_)
  as.numeric(info[["rss"]]) / 1024^2
}

make_task <- function(cfg) {
  set.seed(cfg$data_seed)
  n <- cfg$n_train + cfg$n_test
  X <- matrix(rnorm(n * cfg$p), nrow = n, ncol = cfg$p)
  rank <- min(cfg$latent_rank, cfg$p, max(1L, cfg$q))
  U <- matrix(rnorm(cfg$p * rank), cfg$p, rank) / sqrt(cfg$p)
  Z <- X %*% U
  if (identical(cfg$task_type, "classification")) {
    W <- matrix(rnorm(rank * cfg$class_count), rank, cfg$class_count) / sqrt(rank)
    score <- Z %*% W + matrix(rnorm(n * cfg$class_count, sd = cfg$noise), n)
    y <- max.col(score, ties.method = "first")
    y[seq_len(cfg$class_count)] <- seq_len(cfg$class_count)
    Y <- factor(y, levels = seq_len(cfg$class_count))
  } else {
    V <- matrix(rnorm(rank * cfg$q), rank, cfg$q) / sqrt(rank)
    Y <- Z %*% V + matrix(rnorm(n * cfg$q, sd = cfg$noise), nrow = n)
  }
  train <- seq_len(cfg$n_train)
  test <- cfg$n_train + seq_len(cfg$n_test)
  list(
    Xtrain = X[train, , drop = FALSE],
    Ytrain = if (is.factor(Y)) Y[train] else Y[train, , drop = FALSE],
    Xtest = X[test, , drop = FALSE],
    Ytest = if (is.factor(Y)) Y[test] else Y[test, , drop = FALSE]
  )
}

last_component <- function(x, ncomp) {
  if (is.list(x) && !is.data.frame(x)) return(x[[length(x)]])
  if (is.data.frame(x)) return(x[[ncol(x)]])
  if (is.array(x) && length(dim(x)) == 3L) return(x[, , dim(x)[3L], drop = TRUE])
  x
}

prediction_artifact <- function(pred, task_type, ncomp) {
  if (identical(task_type, "classification")) {
    labels <- as.character(last_component(pred$Ypred, ncomp))
    scores <- if (!is.null(pred$Yscore)) last_component(pred$Yscore, ncomp) else NULL
    return(list(labels = labels, scores = scores))
  }
  list(prediction = as.matrix(last_component(pred$Ypred, ncomp)))
}

regression_metrics <- function(observed, predicted, ytrain) {
  observed <- as.matrix(observed)
  predicted <- as.matrix(predicted)
  baseline <- matrix(colMeans(ytrain), nrow(observed), ncol(observed), byrow = TRUE)
  press <- sum((observed - predicted)^2)
  list(rmsd = sqrt(mean((observed - predicted)^2)), q2 = 1 - press / sum((observed - baseline)^2))
}

numeric_agreement <- function(candidate, reference) {
  a <- as.numeric(candidate)
  b <- as.numeric(reference)
  denom <- sqrt(sum(b * b))
  list(
    relative_error = sqrt(sum((a - b)^2)) / max(denom, .Machine$double.eps),
    correlation = if (length(a) > 1L && stats::sd(a) > 0 && stats::sd(b) > 0) stats::cor(a, b) else NA_real_
  )
}

row <- data.frame(
  run_id = cfg$run_id,
  scenario_id = cfg$scenario_id,
  factor_name = cfg$factor_name,
  factor_value = cfg$factor_value,
  factor_label = cfg$factor_label,
  task_type = cfg$task_type,
  method = "simpls",
  route = cfg$route,
  backend = cfg$backend,
  svd_method = cfg$svd_method,
  xprod_requested = cfg$xprod,
  xprod_used = NA_character_,
  precision = "float64",
  n_train = cfg$n_train,
  n_test = cfg$n_test,
  p = cfg$p,
  q = cfg$q,
  class_count = cfg$class_count,
  latent_rank = cfg$latent_rank,
  max_ncomp = max(cfg$ncomp),
  requested_prefixes = cfg$requested_prefixes,
  crosscov_mb = cfg$p * cfg$q * 8 / 1024^2,
  replicate = cfg$replicate,
  data_seed = cfg$data_seed,
  fit_seed = cfg$fit_seed,
  oversample = cfg$oversample,
  power = cfg$power,
  baseline_rss_mb = NA_real_,
  rss_after_fit_mb = NA_real_,
  rss_after_prediction_mb = NA_real_,
  process_peak_rss_mb = NA_real_,
  incremental_peak_rss_mb = NA_real_,
  gpu_process_peak_mb = NA_real_,
  gpu_total_baseline_mb = NA_real_,
  gpu_total_peak_mb = NA_real_,
  gpu_total_incremental_mb = NA_real_,
  fit_sec = NA_real_,
  prediction_sec = NA_real_,
  total_sec = NA_real_,
  model_size_mb = NA_real_,
  input_size_mb = NA_real_,
  rmsd = NA_real_,
  q2 = NA_real_,
  accuracy = NA_real_,
  prediction_relative_error = NA_real_,
  prediction_correlation = NA_real_,
  label_agreement = NA_real_,
  score_relative_error = NA_real_,
  score_correlation = NA_real_,
  metric_absolute_difference = NA_real_,
  numerical_status = NA_character_,
  status = "failed",
  warnings = "",
  error = "",
  stringsAsFactors = FALSE
)

warnings_seen <- character()
tryCatch({
  if (cfg$backend == "cuda" && !isTRUE(fastPLS::has_cuda())) stop("CUDA backend unavailable")
  if (cfg$backend == "metal" && !isTRUE(fastPLS::has_metal())) stop("Metal backend unavailable")

  task <- make_task(cfg)
  row$input_size_mb <- as.numeric(object.size(task)) / 1024^2
  gc()
  row$baseline_rss_mb <- rss_mb()

  if (cfg$xprod == "auto") {
    Sys.unsetenv(c("FASTPLS_ABLATION_MODE", "FASTPLS_ABLATION_XPROD"))
  } else {
    Sys.setenv(
      FASTPLS_ABLATION_MODE = "1",
      FASTPLS_ABLATION_XPROD = if (cfg$xprod == "implicit") "1" else "0"
    )
  }

  row$fit_sec <- system.time({
    fit <- withCallingHandlers(
      fastPLS::pls(
        task$Xtrain, task$Ytrain,
        ncomp = cfg$ncomp,
        method = "simpls",
        backend = cfg$backend,
        svd.method = cfg$svd_method,
        classifier = "argmax",
        scaling = "centering",
        fit = FALSE,
        return_variance = FALSE,
        return_loadings = FALSE,
        oversample = cfg$oversample,
        power = cfg$power,
        seed = cfg$fit_seed
      ),
      warning = function(w) {
        warnings_seen <<- c(warnings_seen, conditionMessage(w))
        invokeRestart("muffleWarning")
      }
    )
  })[["elapsed"]]
  row$rss_after_fit_mb <- rss_mb()
  row$model_size_mb <- as.numeric(object.size(fit)) / 1024^2

  row$prediction_sec <- system.time({
    pred <- predict(
      fit, task$Xtest,
      raw_scores = identical(cfg$task_type, "classification")
    )
  })[["elapsed"]]
  row$rss_after_prediction_mb <- rss_mb()
  row$total_sec <- row$fit_sec + row$prediction_sec
  artifact <- prediction_artifact(pred, cfg$task_type, cfg$ncomp)

  internal <- attr(fit, "fastPLS_internal", exact = TRUE)
  if (is.null(internal)) internal <- list()
  xprod_used <- internal$xprod_mode
  if (is.null(xprod_used)) xprod_used <- internal$xprod_default
  if (!is.null(xprod_used) && length(xprod_used)) {
    row$xprod_used <- as.character(xprod_used[[1L]])
  }

  if (identical(cfg$task_type, "classification")) {
    row$accuracy <- mean(artifact$labels == as.character(task$Ytest))
  } else {
    metrics <- regression_metrics(task$Ytest, artifact$prediction, task$Ytrain)
    row$rmsd <- metrics$rmsd
    row$q2 <- metrics$q2
  }

  file.create(measurement_done)

  if (isTRUE(cfg$reference)) {
    saveRDS(list(artifact = artifact, rmsd = row$rmsd, q2 = row$q2, accuracy = row$accuracy), cfg$reference_file)
    row$prediction_relative_error <- 0
    row$prediction_correlation <- 1
    row$label_agreement <- if (identical(cfg$task_type, "classification")) 1 else NA_real_
    row$score_relative_error <- if (identical(cfg$task_type, "classification")) 0 else NA_real_
    row$score_correlation <- if (identical(cfg$task_type, "classification")) 1 else NA_real_
    row$metric_absolute_difference <- 0
    row$numerical_status <- "deterministic_reference"
  } else if (file.exists(cfg$reference_file)) {
    ref <- readRDS(cfg$reference_file)
    if (identical(cfg$task_type, "classification")) {
      row$label_agreement <- mean(artifact$labels == ref$artifact$labels)
      if (!is.null(artifact$scores) && !is.null(ref$artifact$scores)) {
        agree <- numeric_agreement(artifact$scores, ref$artifact$scores)
        row$score_relative_error <- agree$relative_error
        row$score_correlation <- agree$correlation
      }
      row$metric_absolute_difference <- abs(row$accuracy - ref$accuracy)
      row$numerical_status <- if (is.finite(row$label_agreement) && row$label_agreement >= 0.99 && row$metric_absolute_difference <= 0.01) "within_tolerance" else "outside_tolerance"
    } else {
      agree <- numeric_agreement(artifact$prediction, ref$artifact$prediction)
      row$prediction_relative_error <- agree$relative_error
      row$prediction_correlation <- agree$correlation
      row$metric_absolute_difference <- abs(row$rmsd - ref$rmsd)
      row$numerical_status <- if (agree$relative_error <= 0.05 && is.finite(agree$correlation) && agree$correlation >= 0.99) "within_tolerance" else "outside_tolerance"
    }
  } else {
    row$numerical_status <- "reference_missing"
  }

  row$status <- "success"
  row$warnings <- paste(unique(warnings_seen), collapse = " | ")
}, error = function(e) {
  file.create(measurement_done)
  row$error <<- conditionMessage(e)
  row$warnings <<- paste(unique(warnings_seen), collapse = " | ")
})

saveRDS(row, result_path)
write.csv(row, sub("[.]rds$", ".csv", result_path), row.names = FALSE)
