#!/usr/bin/env Rscript

# Reproduce the current fastPLS CUDA SIMPLS-LDA route on the stored
# ImageNet/DINOv2 development split. Each invocation fits one component count
# so top-5 scores never require a component-path score cube.

options(stringsAsFactors = FALSE, fastPLS.fused_cuda_lda = TRUE)

env <- function(name, default = "") {
  value <- Sys.getenv(name, unset = default)
  if (nzchar(value)) value else default
}
stamp <- function(...) {
  message(
    "[", format(Sys.time(), "%Y-%m-%d %H:%M:%S"), "] ",
    paste0(..., collapse = "")
  )
}
rss_mb <- function() {
  if (!file.exists("/proc/self/status")) return(NA_real_)
  line <- grep("^VmRSS:", readLines("/proc/self/status", warn = FALSE),
               value = TRUE)
  if (!length(line)) return(NA_real_)
  as.numeric(sub("^VmRSS:\\s*([0-9.]+).*", "\\1", line[[1L]])) / 1024
}
gpu_used_mb <- function() {
  value <- tryCatch(
    suppressWarnings(system2(
      "nvidia-smi",
      c("--query-gpu=memory.used", "--format=csv,noheader,nounits"),
      stdout = TRUE, stderr = FALSE
    )),
    error = function(e) character()
  )
  if (!length(value)) return(NA_real_)
  as.numeric(trimws(value[[1L]]))
}
as_double_matrix <- function(x) {
  if (inherits(x, "float32")) {
    return(as.matrix(float::dbl(x)))
  }
  as.matrix(x)
}
component_value <- function(x) {
  if (is.list(x) && !is.data.frame(x)) return(x[[1L]])
  if (is.data.frame(x)) return(x[[1L]])
  dims <- dim(x)
  if (length(dims) == 3L) return(x[, , 1L, drop = TRUE])
  if (length(dims) == 2L && ncol(x) == 1L) return(x[, 1L])
  x
}
classification_metrics <- function(truth, predicted, top_labels) {
  truth <- as.character(truth)
  predicted <- as.character(predicted)
  top_labels <- as.matrix(top_labels)
  top5 <- mean(vapply(seq_along(truth), function(i) {
    truth[[i]] %in% top_labels[i, seq_len(min(5L, ncol(top_labels)))]
  }, logical(1L)))
  lev <- sort(unique(truth))
  tab <- table(
    factor(truth, levels = lev),
    factor(predicted, levels = lev)
  )
  support <- rowSums(tab)
  predicted_n <- colSums(tab)
  recall <- ifelse(support > 0, diag(tab) / support, NA_real_)
  precision <- ifelse(predicted_n > 0, diag(tab) / predicted_n, 0)
  f1 <- ifelse(
    is.finite(precision + recall) & precision + recall > 0,
    2 * precision * recall / (precision + recall),
    0
  )
  c(
    top1_accuracy = mean(predicted == truth),
    top5_accuracy = top5,
    balanced_accuracy = mean(recall, na.rm = TRUE),
    macro_f1 = mean(f1, na.rm = TRUE)
  )
}

lib <- env("FASTPLS_LIB")
if (nzchar(lib)) .libPaths(unique(c(lib, .libPaths())))
suppressPackageStartupMessages(library(fastPLS))
loaded_package_path <- normalizePath(
  system.file(package = "fastPLS"),
  winslash = "/",
  mustWork = TRUE
)
loaded_package_version <- as.character(utils::packageVersion("fastPLS"))
source_archive_sha256 <- env("SOURCE_ARCHIVE_SHA256", "")

task_file <- path.expand(env(
  "TASK_RDS",
  "~/Documents/fastpls/data/imagenet_float32_seed123_train1000000_task.rds"
))
output_csv <- path.expand(env(
  "OUTPUT_CSV",
  "imagenet_current_fused_lda.csv"
))
ncomp <- as.integer(env("NCOMP", "100"))
oversample <- as.integer(env("OVERSAMPLE", "20"))
power <- as.integer(env("POWER", "2"))
seed <- as.integer(env("SEED", "123"))
replicate_id <- as.integer(env("REPLICATE", "1"))
precision <- match.arg(env("PRECISION", "float32"), c("float32", "float64"))

dir.create(dirname(output_csv), recursive = TRUE, showWarnings = FALSE)
row <- data.frame(
  dataset = "imagenet",
  train_n = NA_integer_,
  test_n = NA_integer_,
  p = NA_integer_,
  q = NA_integer_,
  method = "simpls",
  solver = "rsvd",
  backend = "cuda",
  classifier = "lda",
  precision = precision,
  ncomp_requested = ncomp,
  ncomp_effective = NA_integer_,
  oversample = oversample,
  power = power,
  seed = seed,
  replicate = replicate_id,
  fit_predict_time_sec = NA_real_,
  top5_prediction_time_sec = NA_real_,
  total_time_sec = NA_real_,
  top1_accuracy = NA_real_,
  top5_accuracy = NA_real_,
  balanced_accuracy = NA_real_,
  macro_f1 = NA_real_,
  rss_before_data_mb = rss_mb(),
  rss_before_fit_mb = NA_real_,
  rss_after_fit_mb = NA_real_,
  rss_after_top5_mb = NA_real_,
  gpu_before_fit_mb = NA_real_,
  gpu_after_fit_mb = NA_real_,
  gpu_after_top5_mb = NA_real_,
  gpu_peak_mb = NA_real_,
  gpu_incremental_peak_mb = NA_real_,
  process_peak_rss_mb = NA_real_,
  incremental_peak_rss_mb = NA_real_,
  executed_estimator = NA_character_,
  prediction_backend = NA_character_,
  classifier_train_backend = NA_character_,
  model_gpu_resident = NA,
  fit_residency = if (precision == "float32") {
    "hybrid_host_simpls_cuda_rsvd"
  } else {
    "cuda_resident_fit"
  },
  prediction_residency = if (precision == "float32") {
    "host_score_projection_cuda_lda"
  } else {
    "cuda_fused_lda"
  },
  audit_status = if (oversample >= 20L) {
    "qualified_approximate_controls"
  } else {
    "workflow_only_controls"
  },
  loaded_package_path = loaded_package_path,
  loaded_package_version = loaded_package_version,
  source_archive_sha256 = source_archive_sha256,
  status = "started",
  error = "",
  stringsAsFactors = FALSE
)

tryCatch({
  if (!file.exists(task_file)) stop("Task metadata not found: ", task_file)
  if (!isTRUE(has_cuda())) stop("CUDA backend is unavailable")
  task <- readRDS(task_file)
  required <- c("Xtrain_rds", "Ytrain", "Xtest_rds", "Ytest")
  if (!all(required %in% names(task))) {
    stop("Task metadata must contain: ", paste(required, collapse = ", "))
  }
  row$train_n <- task$n_train
  row$test_n <- task$n_test
  row$p <- task$p
  row$q <- task$n_classes

  stamp("Loading training matrix in ", precision)
  Xtrain_stored <- readRDS(task$Xtrain_rds)
  Xtrain <- if (precision == "float32") {
    if (!inherits(Xtrain_stored, "float32")) {
      float::fl(as.matrix(Xtrain_stored))
    } else {
      Xtrain_stored
    }
  } else {
    as_double_matrix(Xtrain_stored)
  }
  rm(Xtrain_stored)
  gc(FALSE)

  stamp("Loading held-out matrix in ", precision)
  Xtest_stored <- readRDS(task$Xtest_rds)
  Xtest <- if (precision == "float32") {
    if (!inherits(Xtest_stored, "float32")) {
      float::fl(as.matrix(Xtest_stored))
    } else {
      Xtest_stored
    }
  } else {
    as_double_matrix(Xtest_stored)
  }
  rm(Xtest_stored)
  gc(FALSE)

  row$rss_before_fit_mb <- rss_mb()
  row$gpu_before_fit_mb <- gpu_used_mb()
  set.seed(seed)
  stamp(
    "Fitting current CUDA SIMPLS-LDA: ncomp=", ncomp,
    " oversample=", oversample, " power=", power
  )
  fit_time <- unname(system.time({
    fit <- pls(
      Xtrain, task$Ytrain, Xtest, task$Ytest,
      ncomp = ncomp,
      method = "simpls",
      svd.method = "rsvd",
      backend = "cuda",
      classifier = "lda",
      scaling = "centering",
      fit = FALSE,
      return_variance = FALSE,
      oversample = oversample,
      power = power,
      seed = seed
    )
  })[["elapsed"]])
  row$rss_after_fit_mb <- rss_mb()
  row$gpu_after_fit_mb <- gpu_used_mb()

  internal <- attr(fit, "fastPLS_internal", exact = TRUE)
  row$executed_estimator <- as.character(internal$pls_method)[1L]
  row$prediction_backend <- as.character(internal$predict_backend)[1L]
  row$classifier_train_backend <- as.character(fit$lda$train_backend)[1L]
  row$model_gpu_resident <- isTRUE(internal$gpu_resident)
  if (!identical(row$executed_estimator, "simpls")) {
    stop(
      "Requested SIMPLS but executed estimator was ",
      row$executed_estimator
    )
  }
  expected_prediction_backend <- if (precision == "float32") {
    "float32_cuda"
  } else {
    "cuda_fused_lda"
  }
  if (!identical(row$prediction_backend, expected_prediction_backend)) {
    stop(
      "Expected ", expected_prediction_backend,
      " prediction backend but observed ",
      row$prediction_backend
    )
  }

  stamp("Computing top-5 LDA predictions")
  top5_time <- unname(system.time({
    pred <- predict(fit, Xtest, top = 5L, top5 = TRUE)
  })[["elapsed"]])
  row$rss_after_top5_mb <- rss_mb()
  row$gpu_after_top5_mb <- gpu_used_mb()

  predicted <- component_value(pred$Ypred)
  top_labels <- component_value(pred$Ypred_top)
  metrics <- classification_metrics(task$Ytest, predicted, top_labels)
  row$ncomp_effective <- as.integer(internal$ncomp)[1L]
  row$fit_predict_time_sec <- fit_time
  row$top5_prediction_time_sec <- top5_time
  row$total_time_sec <- fit_time + top5_time
  row$top1_accuracy <- metrics[["top1_accuracy"]]
  row$top5_accuracy <- metrics[["top5_accuracy"]]
  row$balanced_accuracy <- metrics[["balanced_accuracy"]]
  row$macro_f1 <- metrics[["macro_f1"]]
  row$status <- "success"
}, error = function(e) {
  row$status <<- "failed"
  row$error <<- conditionMessage(e)
})

write.csv(row, output_csv, row.names = FALSE, na = "")
print(row)
if (!identical(row$status, "success")) quit(save = "no", status = 1L)
