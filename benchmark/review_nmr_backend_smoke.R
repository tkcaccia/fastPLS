#!/usr/bin/env Rscript

# Smoke test for the fixed NMR split used in the manuscript review cycle.
# It deliberately separates fitting from prediction and writes one portable RDS
# record for a requested CPU/CUDA and float64/float32 configuration.

args <- commandArgs(trailingOnly = TRUE)
get_arg <- function(name, default = NULL) {
  key <- paste0("--", name, "=")
  hit <- args[startsWith(args, key)]
  if (!length(hit)) return(default)
  sub(key, "", hit[[1L]], fixed = TRUE)
}

input <- get_arg("input")
output <- get_arg("output")
backend <- get_arg("backend", "cpu")
method <- get_arg("method", "simpls")
precision <- get_arg("precision", "float64")
ncomp <- as.integer(get_arg("ncomp", "10"))
ntrain_limit <- as.integer(get_arg("ntrain", "0"))
ntest_limit <- as.integer(get_arg("ntest", "0"))
p_limit <- as.integer(get_arg("p", "0"))
q_limit <- as.integer(get_arg("q", "0"))

if (is.null(input) || is.null(output)) {
  stop("Usage: review_nmr_backend_smoke.R --input=FILE --output=FILE [--backend=cpu|cuda] [--method=simpls|plssvd] [--precision=float64|float32] [--ncomp=10]", call. = FALSE)
}
if (!precision %in% c("float64", "float32")) stop("precision must be float64 or float32", call. = FALSE)

# R installations on shared machines can prepend a persistent user library.
# FASTPLS_LIB makes the package build under evaluation unambiguous.
fastpls_lib <- Sys.getenv("FASTPLS_LIB", unset = "")
if (nzchar(fastpls_lib)) .libPaths(c(fastpls_lib, .libPaths()))

library(fastPLS)

or_else <- function(x, fallback) if (is.null(x)) fallback else x
as_numeric_matrix <- function(x) {
  if (inherits(x, "float32")) return(as.matrix(float::dbl(x)))
  as.matrix(x)
}

data_env <- new.env(parent = emptyenv())
load(input, envir = data_env)
required <- c("Xtrain", "Ytrain", "Xtest", "Ytest")
if (!all(required %in% ls(data_env))) {
  stop("NMR RData must contain Xtrain, Ytrain, Xtest, and Ytest", call. = FALSE)
}

Xtrain <- get("Xtrain", envir = data_env)
Ytrain <- get("Ytrain", envir = data_env)
Xtest <- get("Xtest", envir = data_env)
Ytest <- get("Ytest", envir = data_env)

if (is.finite(ntrain_limit) && ntrain_limit > 0L) {
  rows <- seq_len(min(nrow(Xtrain), ntrain_limit))
  Xtrain <- Xtrain[rows, , drop = FALSE]
  Ytrain <- Ytrain[rows, , drop = FALSE]
}
if (is.finite(ntest_limit) && ntest_limit > 0L) {
  rows <- seq_len(min(nrow(Xtest), ntest_limit))
  Xtest <- Xtest[rows, , drop = FALSE]
  Ytest <- Ytest[rows, , drop = FALSE]
}
if (is.finite(p_limit) && p_limit > 0L) {
  cols <- seq_len(min(ncol(Xtrain), p_limit))
  Xtrain <- Xtrain[, cols, drop = FALSE]
  Xtest <- Xtest[, cols, drop = FALSE]
}
if (is.finite(q_limit) && q_limit > 0L) {
  cols <- seq_len(min(ncol(Ytrain), q_limit))
  Ytrain <- Ytrain[, cols, drop = FALSE]
  Ytest <- Ytest[, cols, drop = FALSE]
}

if (identical(precision, "float32")) {
  Xtrain <- float::fl(Xtrain)
  Ytrain <- float::fl(Ytrain)
  Xtest <- float::fl(Xtest)
  Ytest <- float::fl(Ytest)
}

gc(full = TRUE)
fit_time <- system.time({
  fit <- pls(
    Xtrain = Xtrain,
    Ytrain = Ytrain,
    ncomp = ncomp,
    scaling = "centering",
    method = method,
    backend = backend,
    svd.method = "rsvd",
    fit = TRUE,
    bycol = FALSE,
    return_variance = FALSE,
    seed = 123
  )
})
predict_time <- system.time({
  pred <- predict(fit, Xtest, Ytest)
})

metric_key <- paste0("ncomp=", ncomp)
prediction_matrix <- pred$Ypred
if (is.list(prediction_matrix) && !is.data.frame(prediction_matrix)) {
  prediction_matrix <- or_else(prediction_matrix[[metric_key]], prediction_matrix[[length(prediction_matrix)]])
}
if (length(dim(prediction_matrix)) == 3L) {
  component_names <- dimnames(prediction_matrix)[[3L]]
  component_index <- match(metric_key, component_names)
  if (is.na(component_index)) component_index <- dim(prediction_matrix)[[3L]]
  prediction_matrix <- prediction_matrix[, , component_index, drop = FALSE]
  dim(prediction_matrix) <- dim(prediction_matrix)[1:2]
}
eval_observed <- as_numeric_matrix(Ytest)
eval_predicted <- as_numeric_matrix(prediction_matrix)
eval_ytrain <- as_numeric_matrix(Ytrain)
if (!is.numeric(eval_observed) || !is.numeric(eval_predicted) || !is.numeric(eval_ytrain)) {
  stop(sprintf(
    "Evaluation conversion failed: observed=[%s;%s] predicted=[%s;%s] ytrain=[%s;%s]",
    paste(class(eval_observed), collapse = ","), typeof(eval_observed),
    paste(class(eval_predicted), collapse = ","), typeof(eval_predicted),
    paste(class(eval_ytrain), collapse = ","), typeof(eval_ytrain)
  ), call. = FALSE)
}
metric <- or_else(pred$metrics[[metric_key]], evaluate(
  observed = eval_observed,
  predicted = eval_predicted,
  ytrain = eval_ytrain,
  bycol = FALSE
))

out <- list(
  package_version = as.character(utils::packageVersion("fastPLS")),
  backend_requested = backend,
  method_requested = method,
  method_executed = or_else(fit$executed_method, or_else(fit$pls_method, method)),
  precision = precision,
  ncomp = ncomp,
  dimensions = c(n_train = nrow(Xtrain), n_test = nrow(Xtest), p = ncol(Xtrain), q = ncol(Ytrain)),
  fit_time_sec = unname(fit_time[["elapsed"]]),
  predict_time_sec = unname(predict_time[["elapsed"]]),
  total_time_sec = unname(fit_time[["elapsed"]] + predict_time[["elapsed"]]),
  metrics = metric,
  session = utils::sessionInfo()
)

dir.create(dirname(output), recursive = TRUE, showWarnings = FALSE)
saveRDS(out, output)
cat(sprintf(
  "method=%s backend=%s precision=%s ncomp=%d total=%.3fs RMSD=%.8f\n",
  out$method_executed,
  backend,
  precision,
  ncomp,
  out$total_time_sec,
  out$metrics$metrics[["RMSD"]]
))
