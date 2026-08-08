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
    Ymm <- as_double_matrix(task$Ytrain)
  }
  fit_fun <- switch(
    fit_method,
    simpls = pls::simpls.fit,
    kernelpls = pls::kernelpls.fit,
    opls = pls::oscorespls.fit
  )
  ncomp_used <- as.integer(effective_ncomp)
  t0 <- proc.time()[3]
  mdl <- fit_fun(
    as_double_matrix(task$Xtrain), Ymm,
    ncomp = ncomp_used, center = TRUE, stripped = TRUE
  )
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
  Xc <- sweep(as_double_matrix(Xtest), 2, model$Xmeans, "-", check.margin = FALSE)
  pred <- Xc %*% coef_mat + matrix(model$Ymeans, nrow = nrow(Xc), ncol = length(model$Ymeans), byrow = TRUE)
  if (identical(task_type, "classification")) {
    pred_names <- colnames(pred)
    if (is.null(pred_names) && ncol(pred) == length(levels_y)) {
      pred_names <- levels_y
    }
    top_k <- min(5L, ncol(pred))
    top_idx <- t(apply(pred, 1L, function(score) {
      head(order(score, decreasing = TRUE), top_k)
    }))
    top_labels <- matrix(
      pred_names[top_idx], nrow = nrow(pred), ncol = top_k,
      dimnames = list(NULL, paste0("rank", seq_len(top_k)))
    )
    return(list(
      Ypred = pred,
      Ypred_top = top_labels,
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
  precision <- tolower(arg_value(
    args,
    "precision",
    default = Sys.getenv("FASTPLS_BENCH_PRECISION", "native")
  ))
  if (!precision %in% c("native", "float32", "float64")) {
    stop("Unsupported benchmark precision: ", precision)
  }

  data_path <- find_dataset_rdata(dataset_id)
  task <- as_task(data_path, dataset_id = dataset_id, split_seed = split_seed)
  task <- coerce_task_precision(task, precision = precision)
  saveRDS(task, task_rds)
  saveRDS(task[c(
    "dataset", "task_type", "dataset_path", "split_seed", "n_train",
    "n_test", "p", "n_classes", "precision", "input_storage_mb"
  )], meta_rds)
  cat(normalizePath(task_rds, winslash = "/", mustWork = FALSE), "\n")
  quit(save = "no", status = 0)
}

if (!identical(mode, "run_one")) {
  stop("Unsupported mode: ", mode)
}

task_rds <- arg_value(args, "task_rds", required = TRUE)
row_out <- arg_value(args, "row_out", required = TRUE)
pid_file <- arg_value(args, "pid_file", default = "")
fit_ready_file <- arg_value(args, "fit_ready_file", default = "")
sampler_ready_file <- arg_value(args, "sampler_ready_file", default = "")
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
  requested_method = spec$method_family,
  executed_method = spec$method_family,
  method_family = spec$method_family,
  method_panel = method_panel_label(spec$method_family),
  engine = spec$engine,
  backend = spec$backend,
  implementation_label = spec$implementation_label,
  classifier = spec$classifier,
  replicate = as.integer(replicate_id),
  requested_ncomp = as.integer(requested_ncomp),
  effective_ncomp = NA_integer_,
  opls_total_ncomp = if (identical(spec$method_family, "opls")) {
    as.integer(requested_ncomp)
  } else {
    NA_integer_
  },
  opls_predictive_ncomp = if (identical(spec$method_family, "opls")) {
    max(1L, as.integer(requested_ncomp) - min(1L, max(0L, as.integer(requested_ncomp) - 1L)))
  } else {
    NA_integer_
  },
  opls_north = if (identical(spec$method_family, "opls")) {
    min(1L, max(0L, as.integer(requested_ncomp) - 1L))
  } else {
    NA_integer_
  },
  kernel_type = if (identical(spec$method_family, "kernelpls")) "linear" else NA_character_,
  kernel_gamma = NA_real_,
  kernel_degree = NA_integer_,
  kernel_coef0 = NA_real_,
  n_train = as.integer(task$n_train),
  n_test = as.integer(task$n_test),
  p = as.integer(task$p),
  n_classes = as.integer(task$n_classes),
  precision = if (!is.null(task$precision)) task$precision else benchmark_matrix_precision(task$Xtrain),
  execution_precision = NA_character_,
  classifier_backend = NA_character_,
  classifier_numeric_path = NA_character_,
  input_storage_mb = if (!is.null(task$input_storage_mb)) as.numeric(task$input_storage_mb) else NA_real_,
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
  top5_accuracy = NA_real_,
  balanced_accuracy = NA_real_,
  macro_f1 = NA_real_,
  prediction_file = if (nzchar(pred_out)) pred_out else NA_character_,
  peak_host_rss_mb = NA_real_,
  peak_gpu_mem_mb = NA_real_,
  fit_window_peak_host_rss_mb = NA_real_,
  incremental_host_rss_mb = NA_real_,
  rss_before_fit_mb = NA_real_,
  rss_after_fit_mb = NA_real_,
  rss_after_predict_mb = NA_real_,
  gpu_before_fit_mb = NA_real_,
  gpu_after_fit_mb = NA_real_,
  gpu_after_predict_mb = NA_real_,
  incremental_gpu_mem_mb = NA_real_,
  status = "error",
  msg = "",
  dataset_path = task$dataset_path,
  split_seed = as.integer(task$split_seed),
  stringsAsFactors = FALSE
)

result_row <- tryCatch({
  suppressPackageStartupMessages(library("fastPLS", lib.loc = lib_loc, character.only = TRUE))

  skip_row <- NULL
  gpu_backend <- if ("native_backend" %in% names(spec) && nzchar(spec$native_backend[[1L]])) {
    as.character(spec$native_backend[[1L]])
  } else {
    benchmark_gpu_backend()
  }
  gpu_available <- if (identical(gpu_backend, "metal")) {
    isTRUE(fastPLS::has_metal())
  } else {
    isTRUE(fastPLS::has_cuda())
  }
  response_cap <- if (identical(task$task_type, "classification")) {
    as.integer(task$n_classes) - 1L
  } else {
    as.integer(task$n_classes)
  }
  if (identical(spec$engine, "GPU") && !isTRUE(gpu_available)) {
    row_template$status <- paste0("skipped_no_", gpu_backend)
    row_template$msg <- sprintf("%s backend not available for library at %s", gpu_backend, lib_loc)
    skip_row <- row_template
  } else if (identical(spec$method_family, "plssvd") &&
             is.finite(requested_ncomp) &&
             requested_ncomp > response_cap) {
    row_template$effective_ncomp <- response_cap
    if (identical(task$task_type, "classification")) {
      row_template$status <- "skipped_ncomp_above_class_cap"
      row_template$msg <- sprintf(
        "%s skipped: requested_ncomp=%s exceeds centered class-response rank cap=%s",
        spec$method_family,
        as.integer(requested_ncomp),
        response_cap
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
      method = method, backend = "cpu", svd.method = svd_method, fit = FALSE,
      classifier = spec$classifier,
      return_variance = FALSE,
      seed = 123L + as.integer(replicate_id)
    )
  }

  kernel_fit <- function(backend, method, svd_method = "rsvd") {
    fastPLS::pls(
      Xtrain = task$Xtrain, Ytrain = task$Ytrain, ncomp = as.integer(effective_cap),
      method = "kernelpls", backend = backend,
      kernel = "linear", svd.method = svd_method, fit = FALSE,
      classifier = spec$classifier,
      return_variance = FALSE,
      seed = 123L + as.integer(replicate_id)
    )
  }

  opls_fit <- function(backend, method, svd_method = "rsvd") {
    fastPLS::pls(
      Xtrain = task$Xtrain, Ytrain = task$Ytrain, ncomp = as.integer(opls_layout$predictive_ncomp),
      method = "opls", backend = backend,
      svd.method = svd_method, north = as.integer(opls_layout$north),
      fit = FALSE, classifier = spec$classifier, return_variance = FALSE,
      seed = 123L + as.integer(replicate_id)
    )
  }

    base_variant_name <- sub("_lda$", "", variant_name)
  fit_fun <- switch(
    base_variant_name,
    cpp_plssvd_cpu_rsvd = function() fastpls_fit("plssvd", "rsvd"),
    cpp_plssvd_irlba = function() fastpls_fit("plssvd", "irlba"),
    gpu_plssvd_rsvd = function() fastPLS::pls(
      Xtrain = task$Xtrain, Ytrain = task$Ytrain, ncomp = as.integer(effective_cap),
      method = "plssvd", backend = gpu_backend, svd.method = "rsvd", fit = FALSE,
      classifier = spec$classifier,
      return_variance = FALSE,
      seed = 123L + as.integer(replicate_id)
    ),
    cpp_simpls_cpu_rsvd = function() fastpls_fit("simpls", "rsvd"),
    cpp_simpls_irlba = function() fastpls_fit("simpls", "irlba"),
    gpu_simpls_rsvd = function() fastPLS::pls(
      Xtrain = task$Xtrain, Ytrain = task$Ytrain, ncomp = as.integer(effective_cap),
      method = "simpls", backend = gpu_backend, svd.method = "rsvd", fit = FALSE,
      classifier = spec$classifier,
      return_variance = FALSE,
      seed = 123L + as.integer(replicate_id)
    ),
    pls_pkg_simpls = function() pls_pkg_fit(task, effective_ncomp = effective_cap, fit_method = "simpls"),
    cpp_kernelpls_cpu_rsvd = function() kernel_fit("cpu", "simpls", "rsvd"),
    cpp_kernelpls_irlba = function() kernel_fit("cpu", "simpls", "irlba"),
    gpu_kernelpls_rsvd = function() kernel_fit(gpu_backend, "simpls"),
    pls_pkg_kernelpls = function() pls_pkg_fit(task, effective_ncomp = effective_cap, fit_method = "kernelpls"),
    cpp_opls_cpu_rsvd = function() opls_fit("cpu", "simpls", "rsvd"),
    cpp_opls_irlba = function() opls_fit("cpu", "simpls", "irlba"),
    gpu_opls_rsvd = function() opls_fit(gpu_backend, "simpls"),
    # `pls::oscorespls.fit()` does not expose `north`, so keep benchmark
    # parity by reserving one total component slot for the orthogonal part.
    pls_pkg_opls = function() pls_pkg_fit(task, effective_ncomp = opls_layout$predictive_ncomp, fit_method = "opls"),
    stop("Unsupported variant_name: ", variant_name)
  )

  if (identical(spec$backend, "pls_pkg")) {
    invisible(gc())
    rss_before_fit <- current_process_rss_mb()
    gpu_before_fit <- current_process_gpu_memory_mb()
    signal_memory_sampler(fit_ready_file, sampler_ready_file)
    fit_obj <- fit_fun()
    rss_after_fit <- current_process_rss_mb()
    gpu_after_fit <- current_process_gpu_memory_mb()
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
    rss_after_predict <- current_process_rss_mb()
    gpu_after_predict <- current_process_gpu_memory_mb()
  } else {
    invisible(gc())
    rss_before_fit <- current_process_rss_mb()
    gpu_before_fit <- current_process_gpu_memory_mb()
    signal_memory_sampler(fit_ready_file, sampler_ready_file)
    fit_ms <- system.time({
      fit_obj <- fit_fun()
    })[["elapsed"]] * 1000
    rss_after_fit <- current_process_rss_mb()
    gpu_after_fit <- current_process_gpu_memory_mb()
    pred_ms <- system.time({
      pred_obj <- predict(
        fit_obj, task$Xtest, Ytest = NULL, proj = FALSE,
        top5 = identical(task$task_type, "classification")
      )
    })[["elapsed"]] * 1000
    rss_after_predict <- current_process_rss_mb()
    gpu_after_predict <- current_process_gpu_memory_mb()
  }

  metric <- metric_from_pred(task$Ytest, pred_obj, y_train = task$Ytrain)
  internal <- if (!identical(spec$backend, "pls_pkg")) {
    attr(fit_obj, "fastPLS_internal", exact = TRUE)
  } else {
    NULL
  }
  executed <- if (!identical(spec$backend, "pls_pkg")) {
    benchmark_executed_method(fit_obj, spec$method_family)
  } else {
    as.character(spec$method_family)[1L]
  }

  if (nzchar(pred_out)) {
    dir.create(dirname(pred_out), recursive = TRUE, showWarnings = FALSE)
    saveRDS(
      list(
        variant_name = variant_name,
        requested_method = spec$method_family,
        executed_method = executed,
        replicate = as.integer(replicate_id),
        requested_ncomp = as.integer(requested_ncomp),
        effective_ncomp = as.integer(effective_cap),
        opls_total_ncomp = if (identical(spec$method_family, "opls")) {
          as.integer(opls_layout$total_ncomp)
        } else {
          NA_integer_
        },
        opls_predictive_ncomp = if (identical(spec$method_family, "opls")) {
          as.integer(opls_layout$predictive_ncomp)
        } else {
          NA_integer_
        },
        opls_north = if (identical(spec$method_family, "opls")) {
          as.integer(opls_layout$north)
        } else {
          NA_integer_
        },
        kernel_type = if (identical(spec$method_family, "kernelpls")) "linear" else NA_character_,
        kernel_gamma = NA_real_,
        kernel_degree = NA_integer_,
        kernel_coef0 = NA_real_,
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
  if (!identical(spec$backend, "pls_pkg")) {
    row_ok$executed_method <- executed
    if (!identical(executed, as.character(spec$method_family)[1L])) {
      row_ok$status <- "error_estimator_mismatch"
      row_ok$msg <- sprintf(
        "requested_method=%s; executed_method=%s; benchmark row rejected",
        spec$method_family,
        executed
      )
    }
    row_ok$execution_precision <- benchmark_execution_precision(fit_obj, task$precision %||% "float64")
    row_ok$classifier_backend <- benchmark_classifier_backend(fit_obj, spec$classifier)
    row_ok$classifier_numeric_path <- benchmark_classifier_numeric_path(
      fit_obj, spec$classifier, task$precision %||% "float64"
    )
  } else {
    row_ok$execution_precision <- "float64"
    row_ok$classifier_backend <- "pls_pkg"
    row_ok$classifier_numeric_path <- "float64"
  }
  row_ok$effective_ncomp <- as.integer(effective_cap)
  if (identical(spec$method_family, "opls")) {
    row_ok$opls_total_ncomp <- as.integer(opls_layout$total_ncomp)
    row_ok$opls_predictive_ncomp <- as.integer(opls_layout$predictive_ncomp)
    row_ok$opls_north <- as.integer(opls_layout$north)
  }
  if (identical(spec$method_family, "kernelpls")) {
    row_ok$kernel_type <- "linear"
    row_ok$kernel_gamma <- NA_real_
    row_ok$kernel_degree <- NA_integer_
    row_ok$kernel_coef0 <- NA_real_
  }
  row_ok$fit_time_ms <- as.numeric(fit_ms)
  row_ok$predict_time_ms <- as.numeric(pred_ms)
  row_ok$total_time_ms <- as.numeric(fit_ms + pred_ms)
  row_ok$metric_name <- metric$metric_name
  row_ok$metric_value <- as.numeric(metric$metric_value)
  row_ok$accuracy <- if (identical(metric$metric_name, "accuracy")) as.numeric(metric$metric_value) else NA_real_
  row_ok$top5_accuracy <- if (!is.null(metric$top5_accuracy)) as.numeric(metric$top5_accuracy) else NA_real_
  row_ok$balanced_accuracy <- if (!is.null(metric$balanced_accuracy)) as.numeric(metric$balanced_accuracy) else NA_real_
  row_ok$macro_f1 <- if (!is.null(metric$macro_f1)) as.numeric(metric$macro_f1) else NA_real_
  row_ok$rss_before_fit_mb <- as.numeric(rss_before_fit)
  row_ok$rss_after_fit_mb <- as.numeric(rss_after_fit)
  row_ok$rss_after_predict_mb <- as.numeric(rss_after_predict)
  row_ok$gpu_before_fit_mb <- as.numeric(gpu_before_fit)
  row_ok$gpu_after_fit_mb <- as.numeric(gpu_after_fit)
  row_ok$gpu_after_predict_mb <- as.numeric(gpu_after_predict)
    row_ok$status <- status
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
