#!/usr/bin/env Rscript

fastpls_lib <- Sys.getenv("FASTPLS_LIB", unset = "")
if (nzchar(fastpls_lib)) {
  .libPaths(c(fastpls_lib, .libPaths()))
}

suppressPackageStartupMessages({
  library(fastPLS)
})

timestamp <- function() format(Sys.time(), "%Y-%m-%d %H:%M:%S")
log_msg <- function(...) message("[", timestamp(), "] ", paste0(..., collapse = ""))

env <- function(name, default = "") {
  value <- Sys.getenv(name, unset = default)
  if (!nzchar(value)) default else value
}

append_csv <- function(path, row) {
  dir.create(dirname(path), recursive = TRUE, showWarnings = FALSE)
  write.table(
    row,
    file = path,
    sep = ",",
    row.names = FALSE,
    col.names = !file.exists(path),
    append = file.exists(path),
    qmethod = "double"
  )
}

cache_complete <- function(path, n, p) {
  file.exists(path) && isTRUE(file.info(path)$size == as.double(n) * as.double(p) * 8)
}

read_matrix_binary_cache <- function(path, n, p, cols = NULL) {
  if (!cache_complete(path, n, p)) {
    stop("Matrix cache is missing or incomplete: ", path)
  }
  log_msg("Reading binary matrix cache: ", path, " rows=", n, ", cols=", p)
  con <- file(path, open = "rb")
  on.exit(close(con), add = TRUE)
  out <- matrix(NA_real_, nrow = n, ncol = p)
  if (!is.null(cols) && length(cols) == p) colnames(out) <- cols
  for (j in seq_len(p)) {
    col <- readBin(con, what = "double", n = n, size = 8, endian = "little")
    if (length(col) != n) {
      stop("Unexpected cache column length at column ", j, ": got ", length(col), ", expected ", n)
    }
    out[, j] <- col
    if ((j %% 128L) == 0L) gc(FALSE)
  }
  out
}

get_pred_labels <- function(pred) {
  ypred <- pred$Ypred
  if (is.null(ypred)) stop("Prediction object does not contain Ypred")
  if (is.data.frame(ypred)) return(as.character(ypred[[1L]]))
  if (is.matrix(ypred) || is.array(ypred)) return(as.character(ypred[, 1L]))
  as.character(ypred)
}

get_top_labels <- function(pred) {
  top <- pred$Ypred_top
  if (is.null(top)) {
    return(matrix(get_pred_labels(pred), ncol = 1L))
  }
  if (is.list(top) && !is.data.frame(top)) {
    top <- top[[1L]]
  }
  if (is.data.frame(top)) {
    top <- as.matrix(top)
  }
  if (length(dim(top)) == 3L) {
    top <- top[, , 1L, drop = TRUE]
  }
  if (is.null(dim(top))) {
    top <- matrix(top, ncol = 1L)
  }
  matrix(as.character(top), nrow = nrow(top), ncol = ncol(top))
}

topk_accuracy <- function(top_labels, truth, k = 5L) {
  top_labels <- as.matrix(top_labels)
  k <- min(as.integer(k)[1L], ncol(top_labels))
  truth <- as.character(truth)
  mean(vapply(seq_along(truth), function(i) {
    truth[[i]] %in% top_labels[i, seq_len(k), drop = TRUE]
  }, logical(1L)), na.rm = TRUE)
}

out_dir <- env("OUT_DIR", file.path(getwd(), "benchmark_results", "imagenet_simpls_rsvd_classifiers"))
out_csv <- file.path(out_dir, "imagenet_simpls_rsvd_classifiers_raw.csv")
task_rds <- env(
  "TASK_RDS",
  "/home/chiamaka/fastPLS_classbias_top1_pipeline/results/imagenet_full_binarycache_1m_probe_20260512_201204/matrix_cache/imagenet_seed123_train1000000_testrest_task.rds"
)

backend <- env("BACKEND", "cpu")
classifier <- env("CLASSIFIER", "argmax")
ncomp <- as.integer(env("NCOMP", "300"))
scaling <- env("SCALING", "centering")
seed <- as.integer(env("SEED", "123"))
gaussian_y <- tolower(env("GAUSSIAN_Y", "false")) %in% c("1", "true", "yes", "y")
gaussian_y_dim <- as.integer(env("GAUSSIAN_Y_DIM", "100"))
predict_backend <- env("PREDICT_BACKEND", if (identical(backend, "cuda")) "cuda_flash" else "cpu")

set.seed(seed)

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
  ncomp = ncomp,
  scaling = scaling,
  gaussian_y = gaussian_y,
  gaussian_y_dim = if (isTRUE(gaussian_y)) gaussian_y_dim else NA_integer_,
  fit_time_sec = NA_real_,
  predict_time_sec = NA_real_,
  total_fit_predict_sec = NA_real_,
  accuracy = NA_real_,
  top5_accuracy = NA_real_,
  status = "started",
  error_message = "",
  stringsAsFactors = FALSE
)

start_total <- proc.time()[["elapsed"]]
status <- "ok"
err <- ""

tryCatch({
  if (!file.exists(task_rds)) stop("Task metadata not found: ", task_rds)
  task <- readRDS(task_rds)
  base_row$train_n <- task$n_train
  base_row$test_n <- task$n_test
  base_row$p <- task$p
  base_row$q <- task$n_classes

  if (identical(backend, "cuda") && !isTRUE(tryCatch(has_cuda(), error = function(e) FALSE))) {
    stop("CUDA backend is not available according to fastPLS::has_cuda()")
  }

  Xtrain <- read_matrix_binary_cache(task$train_bin, task$n_train, task$p, task$feat_cols)
  Ytrain <- task$Ytrain
  log_msg("Fitting simpls/rsvd backend=", backend, " classifier=", classifier, " ncomp=", ncomp)
  t_fit <- system.time({
    model <- pls(
      Xtrain,
      Ytrain,
      ncomp = ncomp,
      method = "simpls",
      svd.method = "rsvd",
      backend = backend,
      classifier = classifier,
      scaling = scaling,
      gaussian_y = gaussian_y,
      gaussian_y_dim = if (isTRUE(gaussian_y)) gaussian_y_dim else NULL,
      fit = FALSE,
      return_variance = FALSE
    )
  })[["elapsed"]]
  rm(Xtrain)
  gc()

  Xtest <- read_matrix_binary_cache(task$test_bin, task$n_test, task$p, task$feat_cols)
  Ytest <- task$Ytest
  log_msg("Predicting backend=", predict_backend)
  t_pred <- system.time({
    pred <- predict(model, Xtest, backend = predict_backend, top = 5L, top5 = TRUE)
  })[["elapsed"]]
  pred_labels <- get_pred_labels(pred)
  top_labels <- get_top_labels(pred)
  acc <- mean(pred_labels == as.character(Ytest), na.rm = TRUE)
  top5 <- topk_accuracy(top_labels, Ytest, k = 5L)

  base_row$fit_time_sec <- as.numeric(t_fit)
  base_row$predict_time_sec <- as.numeric(t_pred)
  base_row$total_fit_predict_sec <- as.numeric(t_fit + t_pred)
  base_row$accuracy <- as.numeric(acc)
  base_row$top5_accuracy <- as.numeric(top5)
}, error = function(e) {
  status <<- "error"
  err <<- conditionMessage(e)
})

base_row$status <- status
base_row$error_message <- err
base_row$total_fit_predict_sec <- if (is.na(base_row$total_fit_predict_sec)) {
  proc.time()[["elapsed"]] - start_total
} else {
  base_row$total_fit_predict_sec
}

append_csv(out_csv, base_row)
log_msg("Finished status=", status, " accuracy=", base_row$accuracy, " top5_accuracy=", base_row$top5_accuracy, " total_sec=", base_row$total_fit_predict_sec)
if (!identical(status, "ok")) quit(save = "no", status = 1L)
