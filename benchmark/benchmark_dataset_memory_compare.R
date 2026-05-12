#!/usr/bin/env Rscript

script_arg <- grep("^--file=", commandArgs(trailingOnly = FALSE), value = TRUE)
script_file <- if (length(script_arg)) sub("^--file=", "", script_arg[[1L]]) else file.path(getwd(), "benchmark_dataset_memory_compare.R")
script_dir <- dirname(normalizePath(script_file, winslash = "/", mustWork = FALSE))
source(file.path(script_dir, "helpers_dataset_memory_compare.R"))

args <- parse_kv_args()
mode <- arg_value(args, "mode", required = TRUE)

pls_pkg_fit <- function(task, effective_ncomp, fit_method = c("simpls", "kernelpls", "opls")) {
  fit_method <- match.arg(fit_method)
  if (!requireNamespace("pls", quietly = TRUE)) {
    stop("pls package is not available")
  }
  if (identical(task$task_type, "classification")) {
    Ymm <- model.matrix(~ task$Ytrain - 1)
    colnames(Ymm) <- levels(task$Ytrain)
  } else {
    Ymm <- as.matrix(task$Ytrain)
  }
  fit_fun <- switch(
    fit_method,
    simpls = pls::simpls.fit,
    kernelpls = pls::kernelpls.fit,
    opls = pls::oscorespls.fit
  )
  ncomp_used <- as.integer(effective_ncomp)
  t0 <- proc.time()[3]
  mdl <- fit_fun(task$Xtrain, Ymm, ncomp = ncomp_used, center = TRUE, stripped = TRUE)
  fit_ms <- (proc.time()[3] - t0) * 1000
  list(model = mdl, fit_ms = as.numeric(fit_ms), ncomp_used = ncomp_used)
}

pls_pkg_predict <- function(model, Xtest, levels_y, ncomp, task_type = "classification") {
  coef_arr <- model$coefficients
  coef_mat <- coef_arr[, , as.integer(ncomp), drop = TRUE]
  if (is.null(dim(coef_mat))) {
    ncol_out <- if (identical(task_type, "classification")) length(levels_y) else length(model$Ymeans)
    coef_mat <- matrix(coef_mat, ncol = ncol_out)
  }
  if (identical(task_type, "classification") && is.null(colnames(coef_mat)) && ncol(coef_mat) == length(levels_y)) {
    colnames(coef_mat) <- levels_y
  }
  Xc <- sweep(as.matrix(Xtest), 2, model$Xmeans, "-", check.margin = FALSE)
  pred <- Xc %*% coef_mat + matrix(model$Ymeans, nrow = nrow(Xc), ncol = length(model$Ymeans), byrow = TRUE)
  if (identical(task_type, "classification")) {
    pred_names <- colnames(pred)
    if (is.null(pred_names) && ncol(pred) == length(levels_y)) {
      pred_names <- levels_y
    }
    return(list(
      Ypred = pred,
      lev = pred_names
    ))
  }
  list(Ypred = pred)
}

benchmark_opls_layout <- function(total_ncomp) {
  total_ncomp <- max(1L, as.integer(total_ncomp))
  # Keep benchmark semantics as total components = predictive + orthogonal.
  # When only one component is requested, fall back to predictive-only because
  # OPLS still needs at least one predictive component to fit.
  north <- min(1L, max(0L, total_ncomp - 1L))
  predictive_ncomp <- max(1L, total_ncomp - north)
  list(
    total_ncomp = total_ncomp,
    predictive_ncomp = predictive_ncomp,
    north = north
  )
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
  saveRDS(task[c("dataset", "task_type", "dataset_path", "split_seed", "n_train", "n_test", "p", "n_classes")], meta_rds)
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
  task_type = task$task_type,
  variant_name = variant_name,
  method_family = spec$method_family,
  method_panel = method_panel_label(spec$method_family),
  engine = spec$engine,
  backend = spec$backend,
  implementation_label = spec$implementation_label,
  classifier = spec$classifier,
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
  metric_name = if (identical(task$task_type, "classification")) {
    "accuracy"
  } else if (isTRUE(task$n_classes == 1L)) {
    "q2"
  } else {
    "rmsd"
  },
  metric_value = NA_real_,
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

  skip_row <- NULL
  if (identical(spec$engine, "GPU") && !isTRUE(fastPLS::has_cuda())) {
    row_template$status <- "skipped_no_cuda"
    row_template$msg <- sprintf("GPU backend not available for library at %s", lib_loc)
    skip_row <- row_template
  } else if (identical(spec$method_family, "plssvd") &&
             is.finite(requested_ncomp) &&
             requested_ncomp > task$n_classes) {
    row_template$effective_ncomp <- as.integer(task$n_classes)
    if (identical(task$task_type, "classification")) {
      row_template$status <- "skipped_ncomp_above_class_cap"
      row_template$msg <- sprintf(
        "%s skipped: requested_ncomp=%s exceeds n_classes=%s for classification task",
        spec$method_family,
        as.integer(requested_ncomp),
        as.integer(task$n_classes)
      )
    } else {
      row_template$status <- "skipped_ncomp_above_y_cap"
      row_template$msg <- sprintf(
        "%s skipped: requested_ncomp=%s exceeds ncol(Y)=%s for regression task",
        spec$method_family,
        as.integer(requested_ncomp),
        as.integer(task$n_classes)
      )
    }
    skip_row <- row_template
  } else if (!identical(spec$classifier, "argmax") &&
             !identical(task$task_type, "classification")) {
    row_template$status <- "skipped_classifier_nonclassification"
    row_template$msg <- sprintf(
      "%s skipped: classifier=%s is only meaningful for classification tasks",
      variant_name,
      spec$classifier
    )
    skip_row <- row_template
  }

  if (!is.null(skip_row)) {
    skip_row
  } else {
    effective_cap <- safe_effective_ncomp(task, requested_ncomp, method_family = spec$method_family)
    opls_layout <- benchmark_opls_layout(effective_cap)
    status <- if (effective_cap < requested_ncomp) "capped" else "ok"

  fastpls_fit <- function(method, svd_method) {
    fastPLS::pls(
      Xtrain = task$Xtrain, Ytrain = task$Ytrain, ncomp = as.integer(effective_cap),
      method = method, backend = "cpp", svd.method = svd_method, fit = FALSE,
      classifier = spec$classifier,
      return_variance = FALSE,
      seed = 123L + as.integer(replicate_id)
    )
  }

  kernel_fit <- function(backend, method, svd_method = "cpu_rsvd") {
    fastPLS::pls(
      Xtrain = task$Xtrain, Ytrain = task$Ytrain, ncomp = as.integer(effective_cap),
      method = "kernelpls", backend = backend, inner.method = method,
      kernel = "linear", svd.method = svd_method, fit = FALSE,
      classifier = spec$classifier,
      return_variance = FALSE,
      seed = 123L + as.integer(replicate_id)
    )
  }

  opls_fit <- function(backend, method, svd_method = "cpu_rsvd") {
    fastPLS::pls(
      Xtrain = task$Xtrain, Ytrain = task$Ytrain, ncomp = as.integer(opls_layout$predictive_ncomp),
      method = "opls", backend = backend, inner.method = method,
      svd.method = svd_method, north = as.integer(opls_layout$north),
      fit = FALSE, classifier = spec$classifier, return_variance = FALSE,
      seed = 123L + as.integer(replicate_id)
    )
  }

    base_variant_name <- sub("_(lda|class_bias)$", "", variant_name)
  fit_fun <- switch(
    base_variant_name,
    cpp_plssvd_cpu_rsvd = function() fastpls_fit("plssvd", "cpu_rsvd"),
    cpp_plssvd_irlba = function() fastpls_fit("plssvd", "irlba"),
    gpu_plssvd_fp64 = function() fastPLS::pls(
      Xtrain = task$Xtrain, Ytrain = task$Ytrain, ncomp = as.integer(effective_cap),
      method = "plssvd", backend = "cuda", fit = FALSE,
      classifier = spec$classifier,
      return_variance = FALSE,
      seed = 123L + as.integer(replicate_id)
    ),
    cpp_simpls_cpu_rsvd = function() fastpls_fit("simpls", "cpu_rsvd"),
    cpp_simpls_irlba = function() fastpls_fit("simpls", "irlba"),
    gpu_simpls_fp64 = function() fastPLS::pls(
      Xtrain = task$Xtrain, Ytrain = task$Ytrain, ncomp = as.integer(effective_cap),
      method = "simpls", backend = "cuda", fit = FALSE,
      classifier = spec$classifier,
      return_variance = FALSE,
      seed = 123L + as.integer(replicate_id)
    ),
    pls_pkg_simpls = function() pls_pkg_fit(task, effective_ncomp = effective_cap, fit_method = "simpls"),
    cpp_kernelpls_cpu_rsvd = function() kernel_fit("cpp", "simpls", "cpu_rsvd"),
    cpp_kernelpls_irlba = function() kernel_fit("cpp", "simpls", "irlba"),
    gpu_kernelpls_fp64 = function() kernel_fit("cuda", "simpls"),
    pls_pkg_kernelpls = function() pls_pkg_fit(task, effective_ncomp = effective_cap, fit_method = "kernelpls"),
    cpp_opls_cpu_rsvd = function() opls_fit("cpp", "simpls", "cpu_rsvd"),
    cpp_opls_irlba = function() opls_fit("cpp", "simpls", "irlba"),
    gpu_opls_fp64 = function() opls_fit("cuda", "simpls"),
    # `pls::oscorespls.fit()` does not expose `north`, so keep benchmark
    # parity by reserving one total component slot for the orthogonal part.
    pls_pkg_opls = function() pls_pkg_fit(task, effective_ncomp = opls_layout$predictive_ncomp, fit_method = "opls"),
    stop("Unsupported variant_name: ", variant_name)
  )

  if (identical(spec$backend, "pls_pkg")) {
    fit_obj <- fit_fun()
    fit_ms <- fit_obj$fit_ms
    pred_ms <- system.time({
      pred_obj <- pls_pkg_predict(
        fit_obj$model,
        task$Xtest,
        if (is.factor(task$Ytrain)) levels(task$Ytrain) else NULL,
        fit_obj$ncomp_used,
        task_type = task$task_type
      )
    })[["elapsed"]] * 1000
  } else {
    fit_ms <- system.time({
      fit_obj <- fit_fun()
    })[["elapsed"]] * 1000
    pred_ms <- system.time({
      pred_obj <- predict(fit_obj, task$Xtest, Ytest = NULL, proj = FALSE, top5 = FALSE)
    })[["elapsed"]] * 1000
  }

  metric <- metric_from_pred(task$Ytest, pred_obj, y_train = task$Ytrain)

  if (nzchar(pred_out)) {
    dir.create(dirname(pred_out), recursive = TRUE, showWarnings = FALSE)
    saveRDS(
      list(
        variant_name = variant_name,
        replicate = as.integer(replicate_id),
        requested_ncomp = as.integer(requested_ncomp),
        effective_ncomp = as.integer(effective_cap),
        classifier = spec$classifier,
        metric_name = metric$metric_name,
        metric_value = metric$metric_value,
        pred = metric$pred,
        truth = task$Ytest
      ),
      pred_out
    )
  }

  row_ok <- row_template
  row_ok$effective_ncomp <- as.integer(effective_cap)
  row_ok$fit_time_ms <- as.numeric(fit_ms)
  row_ok$predict_time_ms <- as.numeric(pred_ms)
  row_ok$total_time_ms <- as.numeric(fit_ms + pred_ms)
  row_ok$metric_name <- metric$metric_name
  row_ok$metric_value <- as.numeric(metric$metric_value)
  row_ok$accuracy <- if (identical(metric$metric_name, "accuracy")) as.numeric(metric$metric_value) else NA_real_
    row_ok$status <- status
    row_ok$msg <- ""
    row_ok
  }
}, error = function(e) {
  row_err <- row_template
  row_err$status <- "error"
  row_err$msg <- conditionMessage(e)
  row_err
})

write_one_row_csv(result_row, row_out)
quit(save = "no", status = 0)
