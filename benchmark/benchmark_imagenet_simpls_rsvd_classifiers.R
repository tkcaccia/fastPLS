#!/usr/bin/env Rscript

fastpls_lib <- Sys.getenv("FASTPLS_LIB", unset = "")
if (nzchar(fastpls_lib)) .libPaths(c(fastpls_lib, .libPaths()))
suppressPackageStartupMessages(library(fastPLS))

env <- function(name, default = "") {
  value <- Sys.getenv(name, unset = default)
  if (nzchar(value)) value else default
}
`%||%` <- function(x, y) if (is.null(x) || !length(x)) y else x
stamp <- function(...) {
  message("[", format(Sys.time(), "%Y-%m-%d %H:%M:%S"), "] ", paste0(..., collapse = ""))
}
is_float32 <- function(x) inherits(x, "float32") || methods::is(x, "float32")

rss_mb <- function() {
  status <- "/proc/self/status"
  if (file.exists(status)) {
    line <- grep("^VmRSS:", readLines(status, warn = FALSE), value = TRUE)
    if (length(line)) return(as.numeric(gsub("[^0-9.]", "", line[[1L]])) / 1024)
  }
  value <- system2("ps", c("-o", "rss=", "-p", as.character(Sys.getpid())), stdout = TRUE)
  as.numeric(value[[1L]]) / 1024
}

last_result <- function(x) {
  if (is.list(x) && !is.data.frame(x)) return(x[[length(x)]])
  if (length(dim(x)) == 3L) return(x[, , dim(x)[3L], drop = TRUE])
  x
}

prediction_labels <- function(pred) {
  value <- last_result(pred$Ypred)
  if (is.data.frame(value)) value <- value[[1L]]
  if (is.matrix(value) || is.array(value)) value <- value[, 1L]
  as.character(value)
}

top_labels <- function(pred) {
  value <- pred$Ypred_top
  if (is.null(value)) return(matrix(prediction_labels(pred), ncol = 1L))
  value <- last_result(value)
  if (is.data.frame(value)) value <- as.matrix(value)
  if (is.null(dim(value))) value <- matrix(value, ncol = 1L)
  matrix(as.character(value), nrow = nrow(value), ncol = ncol(value))
}

classification_metrics <- function(truth, predicted, top = NULL) {
  lev <- levels(factor(truth))
  truth <- factor(truth, levels = lev)
  predicted <- factor(predicted, levels = lev)
  tab <- table(truth, predicted)
  support <- rowSums(tab)
  predicted_n <- colSums(tab)
  recall <- ifelse(support > 0, diag(tab) / support, NA_real_)
  precision <- ifelse(predicted_n > 0, diag(tab) / predicted_n, 0)
  f1 <- ifelse(
    is.finite(precision + recall) & (precision + recall) > 0,
    2 * precision * recall / (precision + recall),
    0
  )
  top5 <- NA_real_
  if (!is.null(top)) {
    top <- as.matrix(top)
    use_k <- min(5L, ncol(top))
    top5 <- mean(vapply(seq_along(truth), function(i) {
      as.character(truth[[i]]) %in% top[i, seq_len(use_k), drop = TRUE]
    }, logical(1L)))
  }
  list(
    accuracy = mean(predicted == truth),
    top5_accuracy = top5,
    balanced_accuracy = mean(recall, na.rm = TRUE),
    macro_f1 = mean(f1, na.rm = TRUE)
  )
}

task_file <- path.expand(env(
  "TASK_RDS",
  "~/Documents/fastpls/data/imagenet_float32_seed123_train1000000_task.rds"
))
row_csv <- path.expand(env("ROW_CSV", tempfile(fileext = ".csv")))
backend <- env("BACKEND", "cpu")
classifier <- env("CLASSIFIER", "argmax")
ncomp <- as.integer(env("NCOMP", "300"))
replicate <- as.integer(env("REPLICATE", "1"))
scaling <- env("SCALING", "centering")
seed <- as.integer(env("SEED", "123"))
cknn_memory <- env("CKNN_MEMORY", "streaming")
pid_file <- env("PID_FILE", "")

if (nzchar(pid_file)) writeLines(as.character(Sys.getpid()), pid_file)
dir.create(dirname(row_csv), recursive = TRUE, showWarnings = FALSE)

row <- data.frame(
  dataset = "imagenet",
  replicate = replicate,
  train_n = NA_integer_, test_n = NA_integer_, p = NA_integer_, q = NA_integer_,
  input_precision = NA_character_, input_storage = NA_character_,
  requested_method = "simpls", executed_method = NA_character_, method_note = "",
  svd_method = "rsvd", backend = backend, classifier = classifier,
  classifier_backend = NA_character_, classifier_numeric_path = NA_character_,
  requested_cknn_memory = if (identical(classifier, "cknn")) cknn_memory else NA_character_,
  cknn_memory = NA_character_,
  ncomp = ncomp, effective_ncomp = NA_integer_, scaling = scaling,
  rss_start_mb = rss_mb(), rss_after_task_mb = NA_real_, rss_after_data_mb = NA_real_,
  rss_after_fit_mb = NA_real_, rss_after_predict_mb = NA_real_,
  fit_time_sec = NA_real_, predict_time_sec = NA_real_, total_time_sec = NA_real_,
  top1_accuracy = NA_real_, top5_accuracy = NA_real_, balanced_accuracy = NA_real_,
  macro_f1 = NA_real_, status = "started", error_message = "",
  stringsAsFactors = FALSE
)

started <- proc.time()[["elapsed"]]
tryCatch({
  if (!file.exists(task_file)) stop("Task metadata not found: ", task_file)
  task <- readRDS(task_file)
  row$rss_after_task_mb <- rss_mb()
  required <- c("Xtrain_rds", "Ytrain", "Xtest_rds", "Ytest")
  if (!all(required %in% names(task))) {
    stop("Task metadata must contain: ", paste(required, collapse = ", "))
  }
  if (!file.exists(task$Xtrain_rds) || !file.exists(task$Xtest_rds)) {
    stop("A task matrix RDS file is missing")
  }

  row$train_n <- task$n_train %||% length(task$Ytrain)
  row$test_n <- task$n_test %||% length(task$Ytest)
  row$p <- task$p
  row$q <- task$n_classes %||% nlevels(factor(task$Ytrain))
  row$input_precision <- task$precision %||% "unknown"
  row$input_storage <- "float::float32"

  if (identical(backend, "cuda") && !isTRUE(tryCatch(has_cuda(), error = function(e) FALSE))) {
    stop("CUDA backend is unavailable")
  }

  stamp("Reading float32 training matrix")
  Xtrain <- readRDS(task$Xtrain_rds)
  if (!is_float32(Xtrain)) stop("Training matrix is not float32")
  row$rss_after_data_mb <- rss_mb()
  set.seed(seed)
  stamp("Fit backend=", backend, " classifier=", classifier, " ncomp=", ncomp)
  fit_time <- system.time({
    model <- pls(
      Xtrain, task$Ytrain,
      ncomp = ncomp,
      method = "simpls",
      svd.method = "rsvd",
      backend = backend,
      classifier = classifier,
      scaling = scaling,
      cknn_memory = cknn_memory,
      fit = FALSE,
      return_variance = FALSE,
      seed = seed
    )
  })[["elapsed"]]
  row$rss_after_fit_mb <- rss_mb()

  internal <- attr(model, "fastPLS_internal", exact = TRUE)
  row$executed_method <- as.character(internal$pls_method %||% "simpls")[[1L]]
  if (!identical(row$executed_method, "simpls")) {
    stop(
      sprintf(
        "Estimator mismatch: requested simpls but executed %s",
        row$executed_method
      ),
      call. = FALSE
    )
  }
  row$method_note <- ""
  row$effective_ncomp <- max(as.integer(model$ncomp %||% ncomp))
  if (identical(classifier, "argmax")) {
    row$classifier_backend <- paste0("float32_", backend, "_argmax")
    row$classifier_numeric_path <- "float32"
  } else if (identical(classifier, "lda")) {
    row$classifier_backend <- model$lda$train_backend %||% paste0("float32_", backend, "_lda")
    row$classifier_numeric_path <- "float32"
  } else {
    row$classifier_backend <- internal$candidate_knn$backend %||% "unknown"
    row$cknn_memory <- internal$candidate_knn$memory %||% "unknown"
    row$classifier_numeric_path <- "float32_pls_with_double_cknn_scores"
  }

  rm(Xtrain)
  gc(FALSE)
  stamp("Reading float32 test matrix")
  Xtest <- readRDS(task$Xtest_rds)
  if (!is_float32(Xtest)) stop("Test matrix is not float32")
  stamp("Predict backend=", backend, " classifier=", classifier, " ncomp=", ncomp)
  predict_time <- system.time({
    pred <- predict(model, Xtest, backend = backend, top = 5L, top5 = TRUE)
  })[["elapsed"]]
  row$rss_after_predict_mb <- rss_mb()

  predicted <- prediction_labels(pred)
  top <- top_labels(pred)
  if (length(predicted) != length(task$Ytest)) {
    stop("Prediction length does not match held-out labels")
  }
  metrics <- classification_metrics(task$Ytest, predicted, top)
  row$fit_time_sec <- unname(fit_time)
  row$predict_time_sec <- unname(predict_time)
  row$total_time_sec <- row$fit_time_sec + row$predict_time_sec
  row$top1_accuracy <- metrics$accuracy
  row$top5_accuracy <- metrics$top5_accuracy
  row$balanced_accuracy <- metrics$balanced_accuracy
  row$macro_f1 <- metrics$macro_f1
  row$status <- "ok"
}, error = function(e) {
  row$status <<- "error"
  row$error_message <<- conditionMessage(e)
  row$total_time_sec <<- proc.time()[["elapsed"]] - started
})

write.csv(row, row_csv, row.names = FALSE, na = "")
stamp(
  "Finished status=", row$status, " top1=", row$top1_accuracy,
  " top5=", row$top5_accuracy, " total_sec=", round(row$total_time_sec, 3)
)
if (!identical(row$status, "ok")) quit(save = "no", status = 1L)
