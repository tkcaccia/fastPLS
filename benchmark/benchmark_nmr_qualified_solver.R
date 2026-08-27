#!/usr/bin/env Rscript

# Matched NMR solver/backend run using the publication preprocessing protocol.
# Each invocation holds the PLS family, precision, split, component count, and
# rSVD controls fixed and repeats fitting/prediction for timing dispersion.

options(stringsAsFactors = FALSE)

parse_args <- function(x = commandArgs(trailingOnly = TRUE)) {
  out <- list()
  for (item in x) {
    if (!startsWith(item, "--")) next
    fields <- strsplit(substring(item, 3L), "=", fixed = TRUE)[[1L]]
    out[[gsub("-", "_", fields[[1L]])]] <-
      if (length(fields) > 1L) paste(fields[-1L], collapse = "=") else "TRUE"
  }
  out
}
args <- parse_args()
arg <- function(name, default = NULL) {
  value <- args[[name]]
  if (is.null(value) || !nzchar(value)) default else value
}

input <- normalizePath(arg(
  "input",
  "/home/chiamaka/Documents/fastpls/data/nmr.RData"
), mustWork = TRUE)
output <- arg("output", "nmr_qualified_solver.csv")
prediction_output <- arg("prediction_output", "")
family <- match.arg(arg("family", "simpls"), c("simpls", "plssvd"))
backend <- match.arg(arg("backend", "cpu"), c("cpu", "cuda"))
solver <- match.arg(arg("solver", "rsvd"), c("rsvd", "irlba"))
ncomp <- as.integer(arg("ncomp", if (family == "simpls") "50" else "5"))
oversample <- as.integer(arg("oversample", "20"))
power <- as.integer(arg("power", "2"))
seed <- as.integer(arg("seed", "123"))
replicates <- as.integer(arg("replicates", "3"))
source_archive_sha256 <- Sys.getenv(
  "FASTPLS_SOURCE_ARCHIVE_SHA256", unset = NA_character_
)
input_sha256 <- Sys.getenv("FASTPLS_INPUT_SHA256", unset = NA_character_)

script_arg <- grep("^--file=", commandArgs(trailingOnly = FALSE), value = TRUE)
script_path <- normalizePath(
  sub("^--file=", "", script_arg[[1L]]),
  mustWork = TRUE
)
source(file.path(dirname(script_path), "nmr_protocol_helpers.R"))

fastpls_lib <- Sys.getenv("FASTPLS_LIB", unset = "")
if (nzchar(fastpls_lib)) {
  .libPaths(unique(c(fastpls_lib, .libPaths())))
}
suppressPackageStartupMessages(library(fastPLS))
if (backend == "cuda" && !isTRUE(has_cuda())) {
  stop("CUDA backend is unavailable.", call. = FALSE)
}
if (backend != "cpu" && solver == "irlba") {
  stop("IRLBA is available only for the CPU reference route.", call. = FALSE)
}

protocol <- fastpls_nmr_protocol(input)
Xtrain <- protocol$Xtrain
Ytrain <- protocol$Ytrain
Xtest <- protocol$Xtest
Ytest <- protocol$Ytest

extract_prediction <- function(x, k) {
  key <- paste0("ncomp=", k)
  if (is.list(x) && !is.data.frame(x)) {
    return(as.matrix(x[[key]] %||% x[[length(x)]]))
  }
  if (length(dim(x)) == 3L) {
    index <- match(key, dimnames(x)[[3L]])
    if (is.na(index)) index <- dim(x)[[3L]]
    return(matrix(x[, , index], nrow = dim(x)[1L], ncol = dim(x)[2L]))
  }
  as.matrix(x)
}

rss_mb <- function() {
  if (!file.exists("/proc/self/status")) return(NA_real_)
  line <- grep("^VmRSS:", readLines("/proc/self/status", warn = FALSE),
               value = TRUE)
  if (!length(line)) return(NA_real_)
  as.numeric(sub("^VmRSS:\\s*([0-9.]+).*", "\\1", line[[1L]])) / 1024
}

regression_metrics <- function(observed, predicted, training) {
  observed <- as.numeric(observed)
  predicted <- as.numeric(predicted)
  training <- as.numeric(training)
  error <- predicted - observed
  sse <- sum(error^2)
  tss <- sum((observed - mean(training))^2)
  c(
    RMSD = sqrt(mean(error^2)),
    Q2 = if (is.finite(tss) && tss > 0) 1 - sse / tss else NA_real_,
    MAE = mean(abs(error))
  )
}

rows <- vector("list", replicates)
reference_prediction <- NULL
for (replicate_id in seq_len(replicates)) {
  gc(full = TRUE)
  baseline_rss <- rss_mb()
  set.seed(seed)
  fit_time <- unname(system.time({
    model <- pls(
      Xtrain, Ytrain,
      ncomp = ncomp,
      method = family,
      backend = backend,
      svd.method = solver,
      scaling = "centering",
      fit = FALSE,
      return_variance = FALSE,
      oversample = oversample,
      power = power,
      seed = seed
    )
  })[["elapsed"]])
  after_fit_rss <- rss_mb()
  predict_time <- unname(system.time({
    prediction_object <- predict(model, Xtest)
  })[["elapsed"]])
  prediction <- extract_prediction(prediction_object$Ypred, ncomp)
  if (!identical(dim(prediction), dim(Ytest))) {
    stop("Prediction dimensions do not match the held-out response.")
  }
  metrics <- regression_metrics(Ytest, prediction, Ytrain)
  per_sample_rmsd <- sqrt(rowMeans((Ytest - prediction)^2))
  diagnostics <- model$diagnostics
  rows[[replicate_id]] <- data.frame(
    dataset = "nmr",
    package_version = as.character(utils::packageVersion("fastPLS")),
    source_archive_sha256 = source_archive_sha256,
    input_sha256 = input_sha256,
    family = family,
    backend = backend,
    solver = solver,
    precision = "float64",
    ncomp = ncomp,
    oversample = if (solver == "rsvd") oversample else NA_integer_,
    power = if (solver == "rsvd") power else NA_integer_,
    seed = seed,
    replicate = replicate_id,
    fit_time_sec = fit_time,
    predict_time_sec = predict_time,
    total_time_sec = fit_time + predict_time,
    RMSD = unname(metrics[["RMSD"]]),
    Q2 = unname(metrics[["Q2"]]),
    MAE = unname(metrics[["MAE"]]),
    median_sample_RMSD = median(per_sample_rmsd),
    p95_sample_RMSD = unname(quantile(per_sample_rmsd, 0.95)),
    baseline_rss_mb = baseline_rss,
    after_fit_rss_mb = after_fit_rss,
    diagnostics_status = diagnostics$status %||% NA_character_,
    diagnostics_approximation_audited =
      diagnostics$approximation_audited %||% NA,
    status = "success",
    stringsAsFactors = FALSE
  )
  if (replicate_id == 1L) {
    reference_prediction <- list(
      observed = Ytest,
      predicted = prediction,
      per_sample_rmsd = per_sample_rmsd,
      per_response_rmsd = sqrt(colMeans((Ytest - prediction)^2)),
      per_response_mae = colMeans(abs(Ytest - prediction)),
      protocol = protocol$metadata,
      model_diagnostics = diagnostics
    )
  }
  rm(model, prediction_object, prediction)
}

dir.create(dirname(output), recursive = TRUE, showWarnings = FALSE)
write.csv(do.call(rbind, rows), output, row.names = FALSE, na = "")
if (nzchar(prediction_output)) {
  dir.create(dirname(prediction_output), recursive = TRUE, showWarnings = FALSE)
  saveRDS(reference_prediction, prediction_output, compress = FALSE)
}
print(do.call(rbind, rows))
