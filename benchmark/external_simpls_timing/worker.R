#!/usr/bin/env Rscript

options(stringsAsFactors = FALSE)

bench_lib <- Sys.getenv("FASTPLS_BENCH_LIB", "")
if (nzchar(bench_lib) && dir.exists(bench_lib)) {
  .libPaths(unique(c(normalizePath(bench_lib, winslash = "/", mustWork = TRUE), .libPaths())))
}

parse_args <- function(x = commandArgs(trailingOnly = TRUE)) {
  out <- list()
  for (item in x) {
    if (!startsWith(item, "--")) next
    bits <- strsplit(substring(item, 3L), "=", fixed = TRUE)[[1L]]
    out[[gsub("-", "_", bits[[1L]], fixed = TRUE)]] <-
      if (length(bits) > 1L) paste(bits[-1L], collapse = "=") else "TRUE"
  }
  out
}

args <- parse_args()
arg <- function(name, default = NULL) {
  value <- args[[name]]
  if (is.null(value) || !nzchar(value)) default else value
}

script_file <- grep("^--file=", commandArgs(FALSE), value = TRUE)
script_file <- if (length(script_file)) sub("^--file=", "", script_file[[1L]]) else "worker.R"
repo_root <- normalizePath(file.path(dirname(script_file), "..", ".."), winslash = "/", mustWork = TRUE)
source(file.path(repo_root, "benchmark", "helpers_dataset_memory_compare.R"))

dataset_id <- tolower(arg("dataset"))
implementation <- match.arg(arg("implementation"), c("fastpls", "pls"))
profile <- match.arg(arg("profile"), c("estimator_kernel", "complete_workflow"))
ncomp <- as.integer(arg("ncomp"))
replicate_id <- as.integer(arg("replicate", "1"))
split_seed <- as.integer(arg("seed", "123"))
row_out <- arg("row_out")
timeout_sec <- as.numeric(arg("timeout_sec", "10000"))

if (!nzchar(dataset_id) || is.na(ncomp) || ncomp < 1L || !nzchar(row_out)) {
  stop("dataset, ncomp, and row-out are required.")
}

# Namespace and shared-library loading are deliberately outside the timed fit.
if (!requireNamespace("fastPLS", quietly = TRUE)) stop("fastPLS is not installed.")
if (!requireNamespace("pls", quietly = TRUE)) stop("pls is not installed.")

task <- as_task(find_dataset_rdata(dataset_id), dataset_id = dataset_id, split_seed = split_seed)
Xtrain <- if (is_float32_matrix(task$Xtrain)) float::dbl(task$Xtrain) else as.matrix(task$Xtrain)
Xtest <- if (is_float32_matrix(task$Xtest)) float::dbl(task$Xtest) else as.matrix(task$Xtest)
Ytrain <- task$Ytrain
Ytest <- task$Ytest

if (!identical(task$task_type, "classification")) {
  stop("The publication external SIMPLS timing comparison is classification-only.")
}
Ytrain <- droplevels(as.factor(Ytrain))
Ytest <- factor(Ytest, levels = levels(Ytrain))
keep <- !is.na(Ytest)
Xtest <- Xtest[keep, , drop = FALSE]
Ytest <- droplevels(Ytest[keep])
class_levels <- levels(Ytrain)
Ydummy <- stats::model.matrix(~ Ytrain - 1)
colnames(Ydummy) <- class_levels

elapsed <- function(expr) {
  start <- proc.time()[[3L]]
  value <- force(expr)
  list(value = value, seconds = unname(proc.time()[[3L]] - start))
}

size_mb <- function(x) as.numeric(utils::object.size(x)) / 1024^2

current_rss_mb <- function() {
  status_file <- "/proc/self/status"
  if (file.exists(status_file)) {
    line <- grep("^VmRSS:", readLines(status_file, warn = FALSE), value = TRUE)
    if (length(line)) {
      value_kb <- suppressWarnings(as.numeric(sub("^VmRSS:[[:space:]]*([0-9]+).*", "\\1", line[[1L]])))
      if (is.finite(value_kb)) return(value_kb / 1024)
    }
  }
  value_kb <- suppressWarnings(as.numeric(system2(
    "ps", c("-o", "rss=", "-p", as.character(Sys.getpid())), stdout = TRUE, stderr = FALSE
  )))
  if (length(value_kb) && is.finite(value_kb[[1L]])) value_kb[[1L]] / 1024 else NA_real_
}

dense_mb <- function(...) prod(as.double(c(...))) * 8 / 1024^2

named_size_mb <- function(x, candidates) {
  if (is.null(x)) return(0)
  total <- 0
  for (name in intersect(candidates, names(x))) {
    value <- x[[name]]
    if (length(value)) total <- total + size_mb(value)
  }
  total
}

decode <- function(scores) {
  scores <- as.matrix(scores)
  factor(class_levels[max.col(scores, ties.method = "first")], levels = class_levels)
}

fit_fastpls <- function() {
  old_store <- Sys.getenv("FASTPLS_STORE_B", unset = NA_character_)
  on.exit({
    if (is.na(old_store)) Sys.unsetenv("FASTPLS_STORE_B") else Sys.setenv(FASTPLS_STORE_B = old_store)
  }, add = TRUE)
  if (identical(profile, "estimator_kernel")) Sys.setenv(FASTPLS_STORE_B = "always")

  call <- list(
    Xtrain = Xtrain,
    Ytrain = Ytrain,
    ncomp = if (identical(profile, "estimator_kernel")) seq_len(ncomp) else ncomp,
    method = "simpls",
    backend = "cpu",
    svd.method = "irlba",
    scaling = "centering",
    seed = split_seed + replicate_id
  )
  if (identical(profile, "estimator_kernel")) {
    call$fit <- FALSE
    call$return_variance <- FALSE
    call$return_loadings <- FALSE
    call$proj <- FALSE
  }
  do.call(fastPLS::pls, call)
}

predict_fastpls <- function(fit) {
  ans <- stats::predict(fit, Xtest)
  pred <- ans$Ypred
  if (is.data.frame(pred)) pred <- pred[[ncol(pred)]]
  factor(pred, levels = class_levels)
}

fit_pls <- function() {
  pls::simpls.fit(
    Xtrain,
    Ydummy,
    ncomp = ncomp,
    center = TRUE,
    stripped = identical(profile, "estimator_kernel")
  )
}

predict_pls <- function(fit) {
  coefficient <- fit$coefficients[, , ncomp, drop = TRUE]
  if (is.null(dim(coefficient))) coefficient <- matrix(coefficient, ncol = ncol(Ydummy))
  scores <- sweep(Xtest, 2L, fit$Xmeans, "-") %*% coefficient
  scores <- sweep(scores, 2L, fit$Ymeans, "+")
  decode(scores)
}

status <- "success"
error_message <- ""
warning_message <- character()
fit <- prediction <- NULL
fit_sec <- prediction_sec <- NA_real_

gc(FALSE)
prefit_rss_mb <- current_rss_mb()
tryCatch(
  withCallingHandlers({
    fitted <- elapsed(if (identical(implementation, "fastpls")) fit_fastpls() else fit_pls())
    fit <- fitted$value
    fit_sec <- fitted$seconds
    predicted <- elapsed(if (identical(implementation, "fastpls")) predict_fastpls(fit) else predict_pls(fit))
    prediction <- predicted$value
    prediction_sec <- predicted$seconds
  }, warning = function(w) {
    warning_message <<- c(warning_message, conditionMessage(w))
    invokeRestart("muffleWarning")
  }),
  error = function(e) {
    status <<- "failed"
    error_message <<- conditionMessage(e)
  }
)
final_rss_mb <- current_rss_mb()

internal <- if (identical(implementation, "fastpls") && !is.null(fit)) {
  attr(fit, "fastPLS_internal", exact = TRUE)
} else NULL
materialized <- if (identical(implementation, "fastpls")) c(fit, internal) else fit

coefficient_path_mb <- named_size_mb(materialized, c("B", "coefficients"))
score_mb <- named_size_mb(materialized, c("Ttrain", "scores", "TT", "U"))
loading_mb <- named_size_mb(materialized, c("P", "loadings", "loadingsX", "loading.weights"))
fitted_mb <- named_size_mb(materialized, c("Yfit", "fitted.values", "fitted"))
variance_mb <- named_size_mb(materialized, c("variance_explained", "Xvar", "Xtotvar", "explvar"))

accuracy <- if (identical(status, "success")) {
  mean(as.character(prediction) == as.character(Ytest))
} else NA_real_

theoretical_cross_covariance_mb <- dense_mb(ncol(Xtrain), ncol(Ydummy))
theoretical_final_coefficient_mb <- dense_mb(ncol(Xtrain), ncol(Ydummy))
theoretical_coefficient_path_mb <- dense_mb(ncol(Xtrain), ncol(Ydummy), ncomp)
theoretical_fitted_path_mb <- dense_mb(nrow(Xtrain), ncol(Ydummy), ncomp)
theoretical_residual_path_mb <- theoretical_fitted_path_mb
theoretical_train_scores_mb <- dense_mb(nrow(Xtrain), ncomp)
theoretical_test_scores_mb <- dense_mb(nrow(Xtest), ncol(Ydummy))
if (identical(profile, "estimator_kernel")) {
  largest_retained_name <- "coefficient path (p x q x A)"
  largest_retained_mb <- theoretical_coefficient_path_mb
} else if (identical(implementation, "pls")) {
  largest_retained_name <- "fitted/residual response path (n_train x q x A each)"
  largest_retained_mb <- theoretical_fitted_path_mb
} else {
  largest_retained_name <- "final coefficient matrix (p x q)"
  largest_retained_mb <- theoretical_final_coefficient_mb
}

row <- data.frame(
  dataset = dataset_id,
  comparison_profile = profile,
  implementation = implementation,
  function_name = if (identical(implementation, "fastpls")) "fastPLS::pls" else "pls::simpls.fit",
  package_version = as.character(utils::packageVersion(if (identical(implementation, "fastpls")) "fastPLS" else "pls")),
  estimator = "deterministic SIMPLS",
  solver = if (identical(implementation, "fastpls")) "IRLBA" else "eigen",
  precision = "float64",
  output_contract = if (identical(profile, "estimator_kernel")) {
    "coefficient path plus final test predictions; scores/loadings/fitted arrays suppressed"
  } else {
    "ordinary public fit object plus final test predictions"
  },
  warmup_policy = "none; every repetition starts in a fresh R process",
  timeout_sec = timeout_sec,
  split_seed = split_seed,
  replicate = replicate_id,
  n_train = nrow(Xtrain),
  n_test = nrow(Xtest),
  p = ncol(Xtrain),
  q = ncol(Ydummy),
  ncomp = ncomp,
  fit_sec = fit_sec,
  prediction_sec = prediction_sec,
  total_sec = fit_sec + prediction_sec,
  accuracy = accuracy,
  fit_object_mb = if (is.null(fit)) NA_real_ else size_mb(fit),
  prediction_object_mb = if (is.null(prediction)) NA_real_ else size_mb(prediction),
  coefficient_path_mb = coefficient_path_mb,
  score_outputs_mb = score_mb,
  loading_outputs_mb = loading_mb,
  fitted_outputs_mb = fitted_mb,
  variance_outputs_mb = variance_mb,
  prefit_process_rss_mb = prefit_rss_mb,
  final_process_rss_mb = final_rss_mb,
  theoretical_cross_covariance_mb = theoretical_cross_covariance_mb,
  theoretical_final_coefficient_mb = theoretical_final_coefficient_mb,
  theoretical_coefficient_path_mb = theoretical_coefficient_path_mb,
  theoretical_fitted_path_mb = theoretical_fitted_path_mb,
  theoretical_residual_path_mb = theoretical_residual_path_mb,
  theoretical_train_scores_mb = theoretical_train_scores_mb,
  theoretical_test_scores_mb = theoretical_test_scores_mb,
  theoretical_largest_retained_name = largest_retained_name,
  theoretical_largest_retained_mb = largest_retained_mb,
  status = status,
  warning_message = paste(unique(warning_message), collapse = " | "),
  error_message = error_message,
  stringsAsFactors = FALSE
)

dir.create(dirname(row_out), recursive = TRUE, showWarnings = FALSE)
utils::write.csv(row, row_out, row.names = FALSE, quote = TRUE, na = "")
print(row[, c("dataset", "comparison_profile", "implementation", "replicate", "total_sec", "accuracy", "status")])
