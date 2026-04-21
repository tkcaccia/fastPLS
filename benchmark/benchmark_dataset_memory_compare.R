#!/usr/bin/env Rscript

script_arg <- grep("^--file=", commandArgs(trailingOnly = FALSE), value = TRUE)
script_file <- if (length(script_arg)) sub("^--file=", "", script_arg[[1L]]) else file.path(getwd(), "benchmark_dataset_memory_compare.R")
script_dir <- dirname(normalizePath(script_file, winslash = "/", mustWork = FALSE))
source(file.path(script_dir, "helpers_dataset_memory_compare.R"))

args <- parse_kv_args()
mode <- arg_value(args, "mode", required = TRUE)

pls_pkg_fit <- function(task, effective_ncomp) {
  if (!requireNamespace("pls", quietly = TRUE)) {
    stop("pls package is not available")
  }
  Ymm <- model.matrix(~ task$Ytrain - 1)
  colnames(Ymm) <- levels(task$Ytrain)
  t0 <- proc.time()[3]
  mdl <- pls::simpls.fit(task$Xtrain, Ymm, ncomp = as.integer(effective_ncomp), center = TRUE, stripped = TRUE)
  fit_ms <- (proc.time()[3] - t0) * 1000
  list(model = mdl, fit_ms = as.numeric(fit_ms))
}

pls_pkg_predict <- function(model, Xtest, levels_y, ncomp) {
  coef_arr <- model$coefficients
  coef_mat <- coef_arr[, , as.integer(ncomp), drop = TRUE]
  Xc <- sweep(as.matrix(Xtest), 2, model$Xmeans, "-", check.margin = FALSE)
  pred <- Xc %*% coef_mat + matrix(model$Ymeans, nrow = nrow(Xc), ncol = length(model$Ymeans), byrow = TRUE)
  factor(colnames(pred)[max.col(pred, ties.method = "first")], levels = levels_y)
}

if (identical(mode, "prepare_task")) {
  dataset_id <- tolower(arg_value(args, "dataset_id", required = TRUE))
  task_rds <- arg_value(args, "task_rds", required = TRUE)
  meta_rds <- arg_value(args, "meta_rds", required = TRUE)
  split_seed <- suppressWarnings(as.integer(arg_value(args, "split_seed", default = "123")))
  if (!is.finite(split_seed) || is.na(split_seed)) split_seed <- 123L

  data_path <- find_dataset_rdata(dataset_id)
  task <- as_task(data_path, dataset_id = dataset_id, split_seed = split_seed)
  saveRDS(task, task_rds)
  saveRDS(task[c("dataset", "dataset_path", "split_seed", "n_train", "n_test", "p", "n_classes")], meta_rds)
  cat(normalizePath(task_rds, winslash = "/", mustWork = FALSE), "\n")
  quit(save = "no", status = 0)
}

if (!identical(mode, "run_one")) {
  stop("Unsupported mode: ", mode)
}

task_rds <- arg_value(args, "task_rds", required = TRUE)
row_out <- arg_value(args, "row_out", required = TRUE)
pid_file <- arg_value(args, "pid_file", default = "")
pred_out <- arg_value(args, "pred_out", default = "")
variant_name <- arg_value(args, "variant_name", required = TRUE)
lib_loc <- normalizePath(arg_value(args, "lib_loc", required = TRUE), winslash = "/", mustWork = TRUE)
requested_ncomp <- suppressWarnings(as.integer(arg_value(args, "requested_ncomp", required = TRUE)))
replicate_id <- suppressWarnings(as.integer(arg_value(args, "replicate", required = TRUE)))

task <- readRDS(task_rds)
spec <- variant_spec(variant_name)

if (nzchar(pid_file)) {
  dir.create(dirname(pid_file), recursive = TRUE, showWarnings = FALSE)
  writeLines(as.character(Sys.getpid()), pid_file)
}

row_template <- data.frame(
  dataset = task$dataset,
  variant_name = variant_name,
  method_family = spec$method_family,
  method_panel = method_panel_label(spec$method_family),
  engine = spec$engine,
  backend = spec$backend,
  implementation_label = spec$implementation_label,
  replicate = as.integer(replicate_id),
  requested_ncomp = as.integer(requested_ncomp),
  effective_ncomp = NA_integer_,
  n_train = as.integer(task$n_train),
  n_test = as.integer(task$n_test),
  p = as.integer(task$p),
  n_classes = as.integer(task$n_classes),
  fit_time_ms = NA_real_,
  predict_time_ms = NA_real_,
  total_time_ms = NA_real_,
  accuracy = NA_real_,
  prediction_file = if (nzchar(pred_out)) pred_out else NA_character_,
  peak_host_rss_mb = NA_real_,
  peak_gpu_mem_mb = NA_real_,
  status = "error",
  msg = "",
  dataset_path = task$dataset_path,
  split_seed = as.integer(task$split_seed),
  stringsAsFactors = FALSE
)

result_row <- tryCatch({
  suppressPackageStartupMessages(library("fastPLS", lib.loc = lib_loc, character.only = TRUE))

  if (identical(spec$engine, "GPU") && !isTRUE(fastPLS::has_cuda())) {
    row_template$status <- "skipped_no_cuda"
    row_template$msg <- sprintf("GPU backend not available for library at %s", lib_loc)
    return(row_template)
  }

  effective_cap <- safe_effective_ncomp(task, requested_ncomp)
  status <- if (effective_cap < requested_ncomp) "capped" else "ok"

  fit_fun <- switch(
    variant_name,
    cpp_plssvd_cpu_rsvd = function() fastPLS::pls(
      Xtrain = task$Xtrain, Ytrain = task$Ytrain, ncomp = as.integer(effective_cap),
      method = "plssvd", svd.method = "cpu_rsvd", fit = FALSE, seed = 123L + as.integer(replicate_id)
    ),
    r_plssvd_cpu_rsvd = function() fastPLS::pls_r(
      Xtrain = task$Xtrain, Ytrain = task$Ytrain, ncomp = as.integer(effective_cap),
      method = "plssvd", svd.method = "cpu_rsvd", fit = FALSE, seed = 123L + as.integer(replicate_id)
    ),
    gpu_plssvd_fp64 = function() fastPLS::plssvd_gpu(
      Xtrain = task$Xtrain, Ytrain = task$Ytrain, ncomp = as.integer(effective_cap),
      fit = FALSE, seed = 123L + as.integer(replicate_id), gpu_train_fp32 = FALSE
    ),
    gpu_plssvd_fp32 = function() fastPLS::plssvd_gpu(
      Xtrain = task$Xtrain, Ytrain = task$Ytrain, ncomp = as.integer(effective_cap),
      fit = FALSE, seed = 123L + as.integer(replicate_id), gpu_train_fp32 = TRUE
    ),
    cpp_simpls_cpu_rsvd = function() fastPLS::pls(
      Xtrain = task$Xtrain, Ytrain = task$Ytrain, ncomp = as.integer(effective_cap),
      method = "simpls", svd.method = "cpu_rsvd", fit = FALSE, seed = 123L + as.integer(replicate_id)
    ),
    r_simpls_cpu_rsvd = function() fastPLS::pls_r(
      Xtrain = task$Xtrain, Ytrain = task$Ytrain, ncomp = as.integer(effective_cap),
      method = "simpls", svd.method = "cpu_rsvd", fit = FALSE, seed = 123L + as.integer(replicate_id)
    ),
    pls_pkg_simpls = function() pls_pkg_fit(task, effective_ncomp = effective_cap),
    cpp_simpls_fast_cpu_rsvd = function() fastPLS::pls(
      Xtrain = task$Xtrain, Ytrain = task$Ytrain, ncomp = as.integer(effective_cap),
      method = "simpls_fast", svd.method = "cpu_rsvd", fit = FALSE, seed = 123L + as.integer(replicate_id)
    ),
    r_simpls_fast_cpu_rsvd = function() fastPLS::pls_r(
      Xtrain = task$Xtrain, Ytrain = task$Ytrain, ncomp = as.integer(effective_cap),
      method = "simpls_fast", svd.method = "cpu_rsvd", fit = FALSE, seed = 123L + as.integer(replicate_id)
    ),
    gpu_simpls_fast_fp64 = function() fastPLS::simpls_gpu(
      Xtrain = task$Xtrain, Ytrain = task$Ytrain, ncomp = as.integer(effective_cap),
      fit = FALSE, seed = 123L + as.integer(replicate_id), gpu_train_fp32 = FALSE
    ),
    gpu_simpls_fast_fp32 = function() fastPLS::simpls_gpu(
      Xtrain = task$Xtrain, Ytrain = task$Ytrain, ncomp = as.integer(effective_cap),
      fit = FALSE, seed = 123L + as.integer(replicate_id), gpu_train_fp32 = TRUE
    ),
    stop("Unsupported variant_name: ", variant_name)
  )

  if (identical(variant_name, "pls_pkg_simpls")) {
    fit_obj <- fit_fun()
    fit_ms <- fit_obj$fit_ms
    pred_ms <- system.time({
      pred_labels <- pls_pkg_predict(fit_obj$model, task$Xtest, levels(task$Ytrain), effective_cap)
    })[["elapsed"]] * 1000
  } else {
    fit_ms <- system.time({
      fit_obj <- fit_fun()
    })[["elapsed"]] * 1000
    pred_ms <- system.time({
      pred_res <- predict(fit_obj, task$Xtest, Ytest = NULL, proj = FALSE)
      pred_labels <- extract_pred_labels(pred_res)
    })[["elapsed"]] * 1000
  }

  acc <- safe_accuracy(task$Ytest, pred_labels)

  if (nzchar(pred_out)) {
    dir.create(dirname(pred_out), recursive = TRUE, showWarnings = FALSE)
    saveRDS(
      list(
        variant_name = variant_name,
        replicate = as.integer(replicate_id),
        requested_ncomp = as.integer(requested_ncomp),
        effective_ncomp = as.integer(effective_cap),
        pred = as.character(pred_labels),
        truth = as.character(task$Ytest)
      ),
      pred_out
    )
  }

  row_ok <- row_template
  row_ok$effective_ncomp <- as.integer(effective_cap)
  row_ok$fit_time_ms <- as.numeric(fit_ms)
  row_ok$predict_time_ms <- as.numeric(pred_ms)
  row_ok$total_time_ms <- as.numeric(fit_ms + pred_ms)
  row_ok$accuracy <- as.numeric(acc)
  row_ok$status <- status
  row_ok$msg <- ""
  row_ok
}, error = function(e) {
  row_err <- row_template
  row_err$status <- "error"
  row_err$msg <- conditionMessage(e)
  row_err
})

write_one_row_csv(result_row, row_out)
quit(save = "no", status = 0)
