#!/usr/bin/env Rscript

# Evaluate one maximal ImageNet SIMPLS path at several requested prefixes.
# This script is intentionally separate from the package API and records the
# exact rSVD controls used by the manuscript analysis.

options(stringsAsFactors = FALSE)

env <- function(name, default = "") {
  value <- Sys.getenv(name, unset = default)
  if (nzchar(value)) value else default
}
`%||%` <- function(x, y) if (is.null(x) || !length(x)) y else x
stamp <- function(...) {
  message("[", format(Sys.time(), "%Y-%m-%d %H:%M:%S"), "] ",
          paste0(..., collapse = ""))
}

fastpls_lib <- env("FASTPLS_LIB", "")
if (nzchar(fastpls_lib)) {
  .libPaths(unique(c(fastpls_lib, .libPaths())))
}
suppressPackageStartupMessages(library(fastPLS))

task_file <- path.expand(env(
  "TASK_RDS",
  "~/Documents/fastpls/data/imagenet_float32_seed123_train1000000_task.rds"
))
output_csv <- path.expand(env(
  "OUTPUT_CSV",
  "imagenet_qualified_top5_path.csv"
))
classifier <- match.arg(env("CLASSIFIER", "argmax"), c("argmax", "lda"))
backend <- match.arg(env("BACKEND", "cuda"), c("cpu", "cuda"))
ncomp <- as.integer(strsplit(
  env("NCOMP", "100,200,300,400,500,600,700,800,900,1000"),
  ",", fixed = TRUE
)[[1L]])
ncomp <- sort(unique(ncomp[is.finite(ncomp) & ncomp > 0L]))
oversample <- as.integer(env("OVERSAMPLE", "20"))
power <- as.integer(env("POWER", "1"))
seed <- as.integer(env("SEED", "123"))
replicate_id <- as.integer(env("REPLICATE", "1"))

rss_mb <- function() {
  if (file.exists("/proc/self/status")) {
    line <- grep("^VmRSS:", readLines("/proc/self/status", warn = FALSE),
                 value = TRUE)
    if (length(line)) {
      return(as.numeric(sub("^VmRSS:\\s*([0-9.]+).*", "\\1", line[[1L]])) /
               1024)
    }
  }
  NA_real_
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

component_value <- function(x, i, n_expected) {
  if (is.list(x) && !is.data.frame(x)) return(x[[i]])
  if (is.data.frame(x)) return(x[[i]])
  dims <- dim(x)
  if (length(dims) == 3L) return(x[, , i, drop = TRUE])
  if (length(dims) == 2L && ncol(x) == n_expected) return(x[, i])
  x
}

metric_row <- function(truth, predicted, top_labels) {
  truth_chr <- as.character(truth)
  predicted_chr <- as.character(predicted)
  top_labels <- as.matrix(top_labels)
  top5 <- mean(vapply(seq_along(truth_chr), function(i) {
    truth_chr[[i]] %in% top_labels[i, seq_len(min(5L, ncol(top_labels)))]
  }, logical(1L)))
  lev <- levels(factor(truth))
  tab <- table(
    factor(truth_chr, levels = lev),
    factor(predicted_chr, levels = lev)
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
    top1_accuracy = mean(predicted_chr == truth_chr),
    top5_accuracy = top5,
    balanced_accuracy = mean(recall, na.rm = TRUE),
    macro_f1 = mean(f1, na.rm = TRUE)
  )
}

dir.create(dirname(output_csv), recursive = TRUE, showWarnings = FALSE)
base_row <- data.frame(
  dataset = "imagenet",
  train_n = NA_integer_,
  test_n = NA_integer_,
  p = NA_integer_,
  q = NA_integer_,
  method = "simpls",
  svd_method = "rsvd",
  backend = backend,
  classifier = classifier,
  precision = NA_character_,
  oversample = oversample,
  power = power,
  seed = seed,
  replicate = replicate_id,
  ncomp = ncomp,
  fit_time_sec = NA_real_,
  predict_time_sec = NA_real_,
  total_time_sec = NA_real_,
  top1_accuracy = NA_real_,
  top5_accuracy = NA_real_,
  balanced_accuracy = NA_real_,
  macro_f1 = NA_real_,
  rss_before_fit_mb = NA_real_,
  rss_after_fit_mb = NA_real_,
  rss_after_predict_mb = NA_real_,
  gpu_before_fit_mb = NA_real_,
  gpu_after_fit_mb = NA_real_,
  gpu_after_predict_mb = NA_real_,
  audit_status = if (backend == "cuda" && oversample >= 20L) {
    "qualified_approximate_controls"
  } else if (backend == "cpu" && power >= 2L) {
    "qualified_approximate_controls"
  } else {
    "workflow_only_controls"
  },
  status = "started",
  error = "",
  stringsAsFactors = FALSE
)

result <- base_row
tryCatch({
  if (!file.exists(task_file)) stop("Task metadata not found: ", task_file)
  task <- readRDS(task_file)
  required <- c("Xtrain_rds", "Ytrain", "Xtest_rds", "Ytest")
  if (!all(required %in% names(task))) {
    stop("Task metadata must contain: ", paste(required, collapse = ", "))
  }
  if (backend == "cuda" && !isTRUE(has_cuda())) {
    stop("CUDA backend is unavailable")
  }

  result$train_n <- task$n_train %||% length(task$Ytrain)
  result$test_n <- task$n_test %||% length(task$Ytest)
  result$p <- task$p
  result$q <- task$n_classes %||% nlevels(factor(task$Ytrain))
  result$precision <- task$precision %||% "float32"

  stamp("Reading ImageNet training features")
  Xtrain <- readRDS(task$Xtrain_rds)
  result$rss_before_fit_mb <- rss_mb()
  result$gpu_before_fit_mb <- gpu_used_mb()

  set.seed(seed)
  stamp(
    "Fitting one SIMPLS path: backend=", backend,
    " classifier=", classifier,
    " max_ncomp=", max(ncomp),
    " oversample=", oversample,
    " power=", power
  )
  fit_time <- unname(system.time({
    model <- pls(
      Xtrain, task$Ytrain,
      ncomp = ncomp,
      method = "simpls",
      svd.method = "rsvd",
      backend = backend,
      classifier = classifier,
      scaling = "centering",
      fit = FALSE,
      return_variance = FALSE,
      oversample = oversample,
      power = power,
      seed = seed
    )
  })[["elapsed"]])
  result$rss_after_fit_mb <- rss_mb()
  result$gpu_after_fit_mb <- gpu_used_mb()

  rm(Xtrain)
  gc(FALSE)
  stamp("Reading ImageNet held-out features")
  Xtest <- readRDS(task$Xtest_rds)
  stamp("Predicting top-5 labels for all requested prefixes")
  predict_time <- unname(system.time({
    pred <- predict(model, Xtest, backend = backend, top = 5L, top5 = TRUE)
  })[["elapsed"]])
  result$rss_after_predict_mb <- rss_mb()
  result$gpu_after_predict_mb <- gpu_used_mb()

  for (i in seq_along(ncomp)) {
    predicted <- component_value(pred$Ypred, i, length(ncomp))
    if (is.matrix(predicted) || is.data.frame(predicted)) {
      predicted <- predicted[, 1L]
    }
    top_labels <- component_value(pred$Ypred_top, i, length(ncomp))
    metrics <- metric_row(task$Ytest, predicted, top_labels)
    result$top1_accuracy[[i]] <- metrics[["top1_accuracy"]]
    result$top5_accuracy[[i]] <- metrics[["top5_accuracy"]]
    result$balanced_accuracy[[i]] <- metrics[["balanced_accuracy"]]
    result$macro_f1[[i]] <- metrics[["macro_f1"]]
  }
  result$fit_time_sec <- fit_time
  result$predict_time_sec <- predict_time
  result$total_time_sec <- fit_time + predict_time
  result$status <- "success"
}, error = function(e) {
  result$status <<- "failed"
  result$error <<- conditionMessage(e)
})

write.csv(result, output_csv, row.names = FALSE, na = "")
print(result)
if (!all(result$status == "success")) quit(save = "no", status = 1L)
