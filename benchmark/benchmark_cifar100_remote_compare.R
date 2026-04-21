#!/usr/bin/env Rscript

script_arg <- grep("^--file=", commandArgs(trailingOnly = FALSE), value = TRUE)
script_file <- if (length(script_arg)) sub("^--file=", "", script_arg[[1L]]) else file.path(getwd(), "benchmark_cifar100_remote_compare.R")
script_dir <- dirname(normalizePath(script_file, winslash = "/", mustWork = FALSE))
source(file.path(script_dir, "helpers_cifar100_remote_compare.R"))

args <- parse_kv_args()
mode <- arg_value(args, "mode", required = TRUE)

if (identical(mode, "prepare_task")) {
  task_rds <- arg_value(args, "task_rds", required = TRUE)
  meta_rds <- arg_value(args, "meta_rds", required = TRUE)
  split_seed <- suppressWarnings(as.integer(arg_value(args, "split_seed", default = "123")))
  if (!is.finite(split_seed) || is.na(split_seed)) split_seed <- 123L

  cifar_path <- find_cifar100_rdata()
  task <- load_cifar100_task(cifar_path, split_seed = split_seed)
  saveRDS(task, task_rds)
  saveRDS(task[c("dataset", "dataset_path", "split_seed", "train_idx", "test_idx", "n_train", "n_test", "p", "n_classes")], meta_rds)
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
include_optional_cpu <- arg_flag(args, "include_optional_cpu", default = TRUE)
include_test_cpu <- arg_flag(args, "include_test_cpu", default = TRUE)

task <- readRDS(task_rds)
spec <- variant_spec(
  variant_name,
  include_optional_cpu = include_optional_cpu,
  include_test_cpu = include_test_cpu
)

if (nzchar(pid_file)) {
  dir.create(dirname(pid_file), recursive = TRUE, showWarnings = FALSE)
  writeLines(as.character(Sys.getpid()), pid_file)
}

row_template <- data.frame(
  variant_name = variant_name,
  code_tree = spec$code_tree,
  method_family = spec$method_family,
  engine = spec$engine,
  backend = spec$backend,
  precision_mode = spec$precision_mode,
  label_mode = spec$label_mode,
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
  reference_variant_name = reference_variant_name(spec$method_family, spec$engine, spec$backend),
  prediction_agreement = NA_real_,
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

  requested_ncomp <- as.integer(requested_ncomp)
  effective_cap <- requested_ncomp
  status <- "ok"
  if (identical(spec$method_family, "plssvd")) {
    effective_cap <- min(requested_ncomp, nrow(task$Xtrain), ncol(task$Xtrain), task$n_classes)
    if (effective_cap < requested_ncomp) status <- "capped"
  }

  switch_key <- paste(spec$engine, spec$method_family, spec$backend, spec$precision_mode, sep = "::")

  fit_fun <- switch(
    switch_key,
    "GPU::plssvd::gpu_native::default" = function() fastPLS::plssvd_gpu(
      Xtrain = task$Xtrain,
      Ytrain = task$Ytrain,
      ncomp = as.integer(effective_cap),
      fit = FALSE,
      seed = 123L + as.integer(replicate_id)
    ),
    "GPU::simpls_fast::gpu_native::default" = function() fastPLS::simpls_gpu(
      Xtrain = task$Xtrain,
      Ytrain = task$Ytrain,
      ncomp = as.integer(effective_cap),
      fit = FALSE,
      seed = 123L + as.integer(replicate_id)
    ),
    "GPU::plssvd::gpu_native::host_qr_eig" = function() fastPLS::plssvd_gpu(
      Xtrain = task$Xtrain,
      Ytrain = task$Ytrain,
      ncomp = as.integer(effective_cap),
      fit = FALSE,
      seed = 123L + as.integer(replicate_id),
      gpu_qr = FALSE,
      gpu_eig = FALSE,
      gpu_qless_qr = FALSE
    ),
    "GPU::simpls_fast::gpu_native::host_qr_eig" = function() fastPLS::simpls_gpu(
      Xtrain = task$Xtrain,
      Ytrain = task$Ytrain,
      ncomp = as.integer(effective_cap),
      fit = FALSE,
      seed = 123L + as.integer(replicate_id),
      gpu_qr = FALSE,
      gpu_eig = FALSE,
      gpu_qless_qr = FALSE
    ),
    "GPU::plssvd::gpu_native::qless_host" = function() fastPLS::plssvd_gpu(
      Xtrain = task$Xtrain,
      Ytrain = task$Ytrain,
      ncomp = as.integer(effective_cap),
      fit = FALSE,
      seed = 123L + as.integer(replicate_id),
      gpu_qr = FALSE,
      gpu_eig = FALSE,
      gpu_qless_qr = TRUE
    ),
    "GPU::simpls_fast::gpu_native::qless_host" = function() fastPLS::simpls_gpu(
      Xtrain = task$Xtrain,
      Ytrain = task$Ytrain,
      ncomp = as.integer(effective_cap),
      fit = FALSE,
      seed = 123L + as.integer(replicate_id),
      gpu_qr = FALSE,
      gpu_eig = FALSE,
      gpu_qless_qr = TRUE
    ),
    "GPU::plssvd::gpu_native::mixed_fp32" = function() fastPLS::plssvd_gpu(
      Xtrain = task$Xtrain,
      Ytrain = task$Ytrain,
      ncomp = as.integer(effective_cap),
      fit = FALSE,
      seed = 123L + as.integer(replicate_id),
      gpu_train_fp32 = TRUE
    ),
    "GPU::simpls_fast::gpu_native::mixed_fp32" = function() fastPLS::simpls_gpu(
      Xtrain = task$Xtrain,
      Ytrain = task$Ytrain,
      ncomp = as.integer(effective_cap),
      fit = FALSE,
      seed = 123L + as.integer(replicate_id),
      gpu_train_fp32 = TRUE
    ),
    "CPU::plssvd::cpu_rsvd::default" = function() fastPLS::pls(
      Xtrain = task$Xtrain,
      Ytrain = task$Ytrain,
      ncomp = as.integer(effective_cap),
      method = "plssvd",
      svd.method = "cpu_rsvd",
      fit = FALSE,
      seed = 123L + as.integer(replicate_id)
    ),
    "CPU::simpls_fast::cpu_rsvd::default" = function() fastPLS::pls(
      Xtrain = task$Xtrain,
      Ytrain = task$Ytrain,
      ncomp = as.integer(effective_cap),
      method = "simpls_fast",
      svd.method = "cpu_rsvd",
      fit = FALSE,
      seed = 123L + as.integer(replicate_id)
    ),
    "CPU::plssvd::irlba::default" = function() fastPLS::pls(
      Xtrain = task$Xtrain,
      Ytrain = task$Ytrain,
      ncomp = as.integer(effective_cap),
      method = "plssvd",
      svd.method = "irlba",
      fit = FALSE,
      seed = 123L + as.integer(replicate_id)
    ),
    "CPU::simpls_fast::irlba::default" = function() fastPLS::pls(
      Xtrain = task$Xtrain,
      Ytrain = task$Ytrain,
      ncomp = as.integer(effective_cap),
      method = "simpls_fast",
      svd.method = "irlba",
      fit = FALSE,
      seed = 123L + as.integer(replicate_id)
    ),
    stop("Unsupported variant configuration for ", variant_name, " [", switch_key, "]")
  )

  fit_ms <- system.time({
    model <- fit_fun()
  })[["elapsed"]] * 1000

  pred_ms <- system.time({
    pred_res <- predict(model, task$Xtest, Ytest = NULL, proj = FALSE)
  })[["elapsed"]] * 1000

  pred_labels <- extract_pred_labels(pred_res)
  acc <- safe_accuracy(task$Ytest, pred_labels)
  eff_ncomp <- safe_effective_ncomp(model, requested_ncomp, fallback_cap = effective_cap)
  if (isTRUE(eff_ncomp < requested_ncomp) && identical(status, "ok")) status <- "capped"

  if (nzchar(pred_out)) {
    dir.create(dirname(pred_out), recursive = TRUE, showWarnings = FALSE)
    saveRDS(
      list(
        variant_name = variant_name,
        replicate = as.integer(replicate_id),
        requested_ncomp = as.integer(requested_ncomp),
        effective_ncomp = as.integer(eff_ncomp),
        pred = as.character(pred_labels),
        truth = as.character(task$Ytest)
      ),
      pred_out
    )
  }

  row_ok <- row_template
  row_ok$effective_ncomp <- as.integer(eff_ncomp)
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
