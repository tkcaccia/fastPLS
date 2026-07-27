#!/usr/bin/env Rscript

# Matched NMR comparison against the deposited Nature Communications
# fastsimpls implementation. Each invocation runs one isolated variant.

args <- commandArgs(trailingOnly = TRUE)
get_arg <- function(name, default = NULL) {
  key <- paste0("--", name, "=")
  value <- args[startsWith(args, key)]
  if (!length(value)) return(default)
  sub(key, "", value[[1L]], fixed = TRUE)
}

input <- get_arg("input")
output <- get_arg("output")
prediction_output <- get_arg("prediction_output")
variant <- get_arg("variant")
reference_source <- get_arg("reference_source")
ncomp <- as.integer(get_arg("ncomp", "100"))
seed <- as.integer(get_arg("seed", "123"))

if (any(!nzchar(c(input, output, prediction_output, variant)))) {
  stop("Provide --input, --output, --prediction_output, and --variant.", call. = FALSE)
}

variants <- list(
  deposited_fastsimpls_irlba = list(reference = TRUE, method = "plssvd",
                                    backend = "cpu", svd_method = "irlba"),
  fastpls_plssvd_cpu_irlba = list(reference = FALSE, method = "plssvd",
                                  backend = "cpu", svd_method = "irlba"),
  fastpls_plssvd_cpu_rsvd = list(reference = FALSE, method = "plssvd",
                                 backend = "cpu", svd_method = "rsvd"),
  fastpls_plssvd_cuda_rsvd = list(reference = FALSE, method = "plssvd",
                                  backend = "cuda", svd_method = "rsvd"),
  fastpls_simpls_cpu_rsvd = list(reference = FALSE, method = "simpls",
                                 backend = "cpu", svd_method = "rsvd"),
  fastpls_simpls_cuda_rsvd = list(reference = FALSE, method = "simpls",
                                  backend = "cuda", svd_method = "rsvd")
)
spec <- variants[[variant]]
if (is.null(spec)) stop("Unknown variant: ", variant, call. = FALSE)

script_arg <- grep("^--file=", commandArgs(trailingOnly = FALSE), value = TRUE)
script_path <- if (length(script_arg)) {
  normalizePath(sub("^--file=", "", script_arg[[1L]]), mustWork = TRUE)
} else {
  normalizePath("benchmark/review_nmr_reference_comparison.R", mustWork = TRUE)
}
source(file.path(dirname(script_path), "nmr_protocol_helpers.R"))
protocol <- fastpls_nmr_protocol(input)
Xtrain <- protocol$Xtrain
Ytrain <- protocol$Ytrain
Xtest <- protocol$Xtest
Ytest <- protocol$Ytest
water_columns <- protocol$water_columns

extract_prediction <- function(x, k) {
  key <- paste0("ncomp=", k)
  if (is.list(x) && !is.data.frame(x)) {
    value <- x[[key]]
    if (is.null(value)) value <- x[[length(x)]]
    return(as.matrix(value))
  }
  if (length(dim(x)) == 3L) {
    names_k <- dimnames(x)[[3L]]
    index <- match(key, names_k)
    if (is.na(index)) index <- dim(x)[[3L]]
    return(x[, , index, drop = FALSE][, , 1L])
  }
  as.matrix(x)
}

set.seed(seed)
gc(full = TRUE)
if (isTRUE(spec$reference)) {
  if (is.null(reference_source) || !file.exists(reference_source)) {
    stop("The deposited fastsimpls source file is required.", call. = FALSE)
  }
  source(reference_source, local = .GlobalEnv)
  fit_time <- system.time({
    model <- fastsimpls(
      Xtrain, Ytrain, ncomp = ncomp, cent = TRUE, scal = FALSE,
      fit = FALSE, fast = TRUE, iter = FALSE
    )
  })[["elapsed"]]
  predict_time <- system.time({
    predicted <- as.matrix(predict.simpls(model, Xtest, Ypred = TRUE)$Ypred)
  })[["elapsed"]]
  package_version <- NA_character_
} else {
  suppressPackageStartupMessages(library(fastPLS))
  if (identical(spec$backend, "cuda") && !isTRUE(has_cuda())) {
    stop("CUDA backend is unavailable.", call. = FALSE)
  }
  fit_time <- system.time({
    model <- pls(
      Xtrain, Ytrain, ncomp = ncomp, method = spec$method,
      backend = spec$backend, svd.method = spec$svd_method,
      scaling = "centering", fit = FALSE, return_variance = FALSE,
      seed = seed
    )
  })[["elapsed"]]
  predict_time <- system.time({
    predicted_object <- predict(model, Xtest, Ytest)
    predicted <- extract_prediction(predicted_object$Ypred, ncomp)
  })[["elapsed"]]
  package_version <- as.character(utils::packageVersion("fastPLS"))
}

if (!identical(dim(predicted), dim(Ytest))) {
  stop("Prediction dimensions do not match Ytest.", call. = FALSE)
}

evaluated <- fastPLS::evaluate(
  observed = Ytest, predicted = predicted, ytrain = Ytrain
)$metrics
r2 <- unname(evaluated[["R2"]])
q2 <- unname(evaluated[["Q2"]])
rmsd <- unname(evaluated[["RMSD"]])
mae <- unname(evaluated[["MAE"]])

per_sample_rmsd <- sqrt(rowMeans((Ytest - predicted)^2))
row <- data.frame(
  variant = variant,
  algorithm = spec$method,
  backend = spec$backend,
  svd_method = spec$svd_method,
  precision = "float64",
  ncomp = ncomp,
  n_train = nrow(Xtrain),
  n_test = nrow(Xtest),
  p = ncol(Xtrain),
  q = ncol(Ytrain),
  fit_time_sec = unname(fit_time),
  predict_time_sec = unname(predict_time),
  total_time_sec = unname(fit_time + predict_time),
  R2 = r2,
  Q2 = q2,
  RMSD = rmsd,
  MAE = mae,
  median_sample_RMSD = stats::median(per_sample_rmsd),
  host_rss_mb = NA_real_,
  gpu_peak_mb = NA_real_,
  package_version = package_version,
  status = "ok",
  stringsAsFactors = FALSE
)

dir.create(dirname(output), recursive = TRUE, showWarnings = FALSE)
utils::write.csv(row, output, row.names = FALSE)
saveRDS(
  list(
    row = row,
    observed = Ytest,
    predicted = predicted,
    per_sample_RMSD = per_sample_rmsd,
    protocol = protocol$metadata,
    water_columns_removed = length(water_columns),
    session = utils::sessionInfo()
  ),
  prediction_output
)
print(row)
