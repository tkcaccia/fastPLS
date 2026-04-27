#!/usr/bin/env Rscript

suppressPackageStartupMessages({
  library(data.table)
})

`%||%` <- function(x, y) if (is.null(x) || length(x) == 0L || is.na(x)) y else x

cmd_all <- commandArgs(FALSE)
file_arg <- sub("^--file=", "", cmd_all[grep("^--file=", cmd_all)])
script_dir <- if (length(file_arg)) dirname(normalizePath(file_arg[[1L]], mustWork = TRUE)) else getwd()
source(file.path(script_dir, "helpers_dataset_memory_compare.R"))

parse_bool <- function(x, default = FALSE) {
  if (is.null(x) || !nzchar(x)) return(default)
  tolower(x) %in% c("1", "true", "yes", "y")
}

split_arg <- function(x, default) {
  if (is.null(x) || !nzchar(x)) return(default)
  trimws(strsplit(x, ",", fixed = TRUE)[[1L]])
}

now_iso <- function() format(Sys.time(), "%Y-%m-%d %H:%M:%S")

time_expr <- function(expr) {
  gc()
  elapsed <- system.time(value <- force(expr))[["elapsed"]] * 1000
  list(value = value, ms = as.numeric(elapsed))
}

prediction_metric <- function(task, pred_obj) {
  metric_from_pred(task$Ytest, pred_obj, y_train = task$Ytrain)
}

fit_variant <- function(task, spec, ncomp, seed) {
  fastpls_fit <- function(method, svd_method) {
    fastPLS::pls(
      Xtrain = task$Xtrain,
      Ytrain = task$Ytrain,
      ncomp = as.integer(ncomp),
      method = method,
      svd.method = svd_method,
      fit = FALSE,
      seed = seed
    )
  }

  switch(
    spec$variant_name,
    cpp_plssvd_cpu_rsvd = fastpls_fit("plssvd", "cpu_rsvd"),
    cpp_plssvd_irlba = fastpls_fit("plssvd", "irlba"),
    cpp_simpls_cpu_rsvd = fastpls_fit("simpls", "cpu_rsvd"),
    cpp_simpls_irlba = fastpls_fit("simpls", "irlba"),
    gpu_plssvd_fp64 = fastPLS::plssvd_gpu(
      Xtrain = task$Xtrain, Ytrain = task$Ytrain, ncomp = as.integer(ncomp),
      fit = FALSE, seed = seed
    ),
    gpu_simpls_fp64 = fastPLS::simpls_gpu(
      Xtrain = task$Xtrain, Ytrain = task$Ytrain, ncomp = as.integer(ncomp),
      fit = FALSE, seed = seed
    ),
    stop("Unsupported Cpp/CUDA variant: ", spec$variant_name)
  )
}

fit_variant_with_xtest <- function(task, spec, ncomp, seed) {
  fastpls_fit <- function(method, svd_method) {
    fastPLS::pls(
      Xtrain = task$Xtrain,
      Ytrain = task$Ytrain,
      Xtest = task$Xtest,
      Ytest = NULL,
      ncomp = as.integer(ncomp),
      method = method,
      svd.method = svd_method,
      fit = FALSE,
      seed = seed
    )
  }

  switch(
    spec$variant_name,
    cpp_plssvd_cpu_rsvd = fastpls_fit("plssvd", "cpu_rsvd"),
    cpp_plssvd_irlba = fastpls_fit("plssvd", "irlba"),
    cpp_simpls_cpu_rsvd = fastpls_fit("simpls", "cpu_rsvd"),
    cpp_simpls_irlba = fastpls_fit("simpls", "irlba"),
    gpu_plssvd_fp64 = fastPLS::plssvd_gpu(
      Xtrain = task$Xtrain, Ytrain = task$Ytrain, Xtest = task$Xtest, Ytest = NULL,
      ncomp = as.integer(ncomp), fit = FALSE, seed = seed
    ),
    gpu_simpls_fp64 = fastPLS::simpls_gpu(
      Xtrain = task$Xtrain, Ytrain = task$Ytrain, Xtest = task$Xtest, Ytest = NULL,
      ncomp = as.integer(ncomp), fit = FALSE, seed = seed
    ),
    stop("Unsupported Cpp/CUDA variant: ", spec$variant_name)
  )
}

args <- parse_kv_args()
datasets <- split_arg(arg_value(args, "datasets", "metref,singlecell,nmr"), c("metref", "singlecell", "nmr"))
ncomp <- as.integer(arg_value(args, "ncomp", "50"))
reps <- as.integer(arg_value(args, "reps", "3"))
split_seed <- as.integer(arg_value(args, "split_seed", "123"))
tag <- arg_value(args, "tag", "run")
lib_loc <- arg_value(args, "lib", Sys.getenv("FASTPLS_LIB", unset = ""))
out_dir <- path.expand(arg_value(args, "out_dir", file.path(getwd(), "prediction_50_cpp_cuda_results")))
include_combined <- parse_bool(arg_value(args, "include_combined", "false"), default = FALSE)
combined_reps <- as.integer(arg_value(args, "combined_reps", "1"))
variant_names <- split_arg(
  arg_value(args, "variants", ""),
  c(
    "cpp_plssvd_cpu_rsvd",
    "cpp_plssvd_irlba",
    "cpp_simpls_cpu_rsvd",
    "cpp_simpls_irlba",
    "gpu_plssvd_fp64",
    "gpu_simpls_fp64"
  )
)

dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)
raw_path <- file.path(out_dir, sprintf("prediction_50_cpp_cuda_%s_raw.csv", tag))
summary_path <- file.path(out_dir, sprintf("prediction_50_cpp_cuda_%s_summary.csv", tag))

if (nzchar(lib_loc)) {
  suppressPackageStartupMessages(library("fastPLS", lib.loc = lib_loc, character.only = TRUE))
} else {
  suppressPackageStartupMessages(library("fastPLS"))
}

rows <- list()
row_i <- 0L

append_row <- function(row) {
  row_i <<- row_i + 1L
  rows[[row_i]] <<- as.data.frame(row, stringsAsFactors = FALSE)
  data.table::fwrite(data.table::rbindlist(rows, fill = TRUE), raw_path)
}

cat(sprintf("[%s] fastPLS prediction benchmark tag=%s ncomp=%d reps=%d\n", now_iso(), tag, ncomp, reps))
cat(sprintf("[%s] output=%s\n", now_iso(), out_dir))

specs <- variant_specs()
specs <- specs[specs$variant_name %in% variant_names, , drop = FALSE]
specs <- specs[specs$implementation_label %in% c("Cpp", "CUDA 64-bit"), , drop = FALSE]

for (dataset_id in datasets) {
  cat(sprintf("[%s] loading dataset=%s\n", now_iso(), dataset_id))
  task <- as_task(find_dataset_rdata(dataset_id), dataset_id = dataset_id, split_seed = split_seed)
  cat(sprintf(
    "[%s] dataset=%s type=%s n_train=%d n_test=%d p=%d y_cols_or_classes=%d\n",
    now_iso(), dataset_id, task$task_type, task$n_train, task$n_test, task$p, task$n_classes
  ))

  for (jj in seq_len(nrow(specs))) {
    spec <- specs[jj, , drop = FALSE]
    variant <- spec$variant_name
    seed <- 123L

    base_row <- list(
      tag = tag,
      dataset = dataset_id,
      task_type = task$task_type,
      variant_name = variant,
      method_family = spec$method_family,
      engine = spec$engine,
      backend = spec$backend,
      implementation_label = spec$implementation_label,
      requested_ncomp = ncomp,
      effective_ncomp = NA_integer_,
      mode = "separate_predict",
      replicate = NA_integer_,
      n_train = task$n_train,
      n_test = task$n_test,
      p = task$p,
      n_classes = task$n_classes,
      fit_time_ms = NA_real_,
      predict_time_ms = NA_real_,
      total_time_ms = NA_real_,
      metric_name = NA_character_,
      metric_value = NA_real_,
      status = "error",
      msg = ""
    )

    if (identical(spec$engine, "GPU") && !isTRUE(fastPLS::has_cuda())) {
      base_row$status <- "skipped_no_cuda"
      base_row$msg <- "fastPLS was not built with CUDA"
      append_row(base_row)
      next
    }

    if (identical(spec$method_family, "plssvd") && ncomp > task$n_classes) {
      base_row$effective_ncomp <- as.integer(task$n_classes)
      base_row$status <- if (identical(task$task_type, "classification")) {
        "skipped_ncomp_above_class_cap"
      } else {
        "skipped_ncomp_above_y_cap"
      }
      base_row$msg <- sprintf("requested_ncomp=%d exceeds PLSSVD cap=%d", ncomp, task$n_classes)
      append_row(base_row)
      next
    }

    effective_ncomp <- safe_effective_ncomp(task, ncomp, method_family = spec$method_family)
    cat(sprintf("[%s] fit dataset=%s variant=%s effective_ncomp=%d\n", now_iso(), dataset_id, variant, effective_ncomp))

    fit_res <- tryCatch(
      time_expr(fit_variant(task, spec, effective_ncomp, seed = seed)),
      error = function(e) e
    )
    if (inherits(fit_res, "error")) {
      base_row$effective_ncomp <- effective_ncomp
      base_row$status <- "fit_error"
      base_row$msg <- conditionMessage(fit_res)
      append_row(base_row)
      next
    }

    model <- fit_res$value
    for (rep_id in seq_len(reps)) {
      pred_row <- base_row
      pred_row$effective_ncomp <- effective_ncomp
      pred_row$replicate <- rep_id
      pred_row$fit_time_ms <- fit_res$ms

      pred_res <- tryCatch(
        time_expr(predict(model, task$Xtest, Ytest = NULL, proj = FALSE)),
        error = function(e) e
      )
      if (inherits(pred_res, "error")) {
        pred_row$status <- "predict_error"
        pred_row$msg <- conditionMessage(pred_res)
        append_row(pred_row)
        next
      }

      metric <- tryCatch(prediction_metric(task, pred_res$value), error = function(e) e)
      if (inherits(metric, "error")) {
        pred_row$status <- "metric_error"
        pred_row$msg <- conditionMessage(metric)
        append_row(pred_row)
        next
      }

      pred_row$predict_time_ms <- pred_res$ms
      pred_row$total_time_ms <- fit_res$ms + pred_res$ms
      pred_row$metric_name <- metric$metric_name
      pred_row$metric_value <- metric$metric_value
      pred_row$status <- if (effective_ncomp < ncomp) "capped" else "ok"
      pred_row$msg <- ""
      append_row(pred_row)
    }

    rm(model)
    gc()

    if (isTRUE(include_combined)) {
      for (rep_id in seq_len(combined_reps)) {
        combined_row <- base_row
        combined_row$mode <- "fit_with_xtest"
        combined_row$effective_ncomp <- effective_ncomp
        combined_row$replicate <- rep_id
        combined_row <- tryCatch({
          combined_res <- time_expr(fit_variant_with_xtest(task, spec, effective_ncomp, seed = seed + rep_id))
          metric <- prediction_metric(task, list(Ypred = combined_res$value$Ypred, lev = combined_res$value$lev))
          combined_row$fit_time_ms <- combined_res$ms
          combined_row$total_time_ms <- combined_res$ms
          combined_row$metric_name <- metric$metric_name
          combined_row$metric_value <- metric$metric_value
          combined_row$status <- if (effective_ncomp < ncomp) "capped" else "ok"
          combined_row$msg <- ""
          combined_row
        }, error = function(e) {
          combined_row$status <- "fit_with_xtest_error"
          combined_row$msg <- conditionMessage(e)
          combined_row
        })
        append_row(combined_row)
      }
    }
  }

  rm(task)
  gc()
}

raw <- data.table::fread(raw_path)
ok <- raw[status %in% c("ok", "capped")]
if (nrow(ok)) {
  summary <- ok[, .(
    n = .N,
    median_fit_time_ms = median(fit_time_ms, na.rm = TRUE),
    median_predict_time_ms = median(predict_time_ms, na.rm = TRUE),
    median_total_time_ms = median(total_time_ms, na.rm = TRUE),
    metric_name = metric_name[which.max(!is.na(metric_name))],
    median_metric_value = median(metric_value, na.rm = TRUE),
    min_metric_value = min(metric_value, na.rm = TRUE),
    max_metric_value = max(metric_value, na.rm = TRUE)
  ), by = .(tag, dataset, task_type, variant_name, method_family, engine, backend, implementation_label, requested_ncomp, effective_ncomp, mode, status)]
  data.table::fwrite(summary, summary_path)
} else {
  data.table::fwrite(data.table::data.table(), summary_path)
}

cat(sprintf("[%s] wrote raw=%s\n", now_iso(), raw_path))
cat(sprintf("[%s] wrote summary=%s\n", now_iso(), summary_path))
