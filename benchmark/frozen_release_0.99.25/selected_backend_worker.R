#!/usr/bin/env Rscript

args <- commandArgs(trailingOnly = TRUE)
arg <- function(name, default = "") {
  hit <- grep(paste0("^--", name, "="), args, value = TRUE)
  if (!length(hit)) return(default)
  sub(paste0("^--", name, "="), "", hit[[1L]])
}
lib <- arg("lib")
if (nzchar(lib)) .libPaths(c(lib, .libPaths()))
helper <- normalizePath(arg("helper"), mustWork = TRUE)
source(helper)
suppressPackageStartupMessages(library(fastPLS))
stopifnot(as.character(packageVersion("fastPLS")) == "0.99.25")

dataset <- tolower(arg("dataset"))
data_path <- normalizePath(arg("data"), mustWork = TRUE)
backend <- match.arg(arg("backend"), c("cpu", "cuda"))
ncomp <- as.integer(arg("ncomp"))
replicate <- as.integer(arg("replicate", "1"))
out <- arg("out")
dir.create(dirname(out), recursive = TRUE, showWarnings = FALSE)

task <- as_task(data_path, dataset_id = dataset, split_seed = 123L)
task <- coerce_task_precision(task, "float64")
if (backend == "cuda" && !isTRUE(has_cuda())) stop("CUDA is unavailable")

rss_mb <- function() {
  if (!file.exists("/proc/self/status")) return(NA_real_)
  line <- grep("^VmRSS:", readLines("/proc/self/status", warn = FALSE), value = TRUE)
  if (!length(line)) return(NA_real_)
  as.numeric(sub("^VmRSS:\\s*([0-9.]+).*", "\\1", line[[1L]])) / 1024
}
gpu_mb <- function() {
  value <- tryCatch(system2(
    "nvidia-smi",
    c("--query-gpu=memory.used", "--format=csv,noheader,nounits"),
    stdout = TRUE, stderr = FALSE
  ), error = function(e) character())
  if (!length(value)) return(NA_real_)
  as.numeric(trimws(value[[1L]]))
}

gc(full = TRUE)
rss_before <- rss_mb(); gpu_before <- gpu_mb()
set.seed(123L)
fit_sec <- unname(system.time({
  fit <- pls(
    task$Xtrain, task$Ytrain,
    ncomp = ncomp, method = "simpls", backend = backend,
    svd.method = "rsvd", classifier = "argmax", fit = FALSE,
    return_variance = FALSE,
    oversample = 20L, power = 2L, seed = 123L
  )
})[["elapsed"]])
rss_fit <- rss_mb(); gpu_fit <- gpu_mb()
pred_sec <- unname(system.time({
  pred <- predict(fit, task$Xtest, top = 5L, top5 = TRUE)
})[["elapsed"]])
rss_pred <- rss_mb(); gpu_pred <- gpu_mb()
metric <- metric_from_pred(task$Ytest, pred, y_train = task$Ytrain)
diagnostics <- fit$diagnostics

row <- data.frame(
  dataset = dataset,
  package_version = as.character(packageVersion("fastPLS")),
  source_archive_sha256 = "604f74d72f5e9540f7378efd37f2ca6ed87ccb9e73b912211331a8cd97233481",
  backend = backend,
  method = "simpls",
  solver = "rsvd",
  precision = "float64",
  classifier = "argmax",
  oversample = 20L,
  power = 2L,
  seed = 123L,
  replicate = replicate,
  n_train = task$n_train,
  n_test = task$n_test,
  p = task$p,
  q = task$n_classes,
  ncomp = ncomp,
  fit_time_sec = fit_sec,
  prediction_time_sec = pred_sec,
  total_time_sec = fit_sec + pred_sec,
  accuracy = metric$metric_value,
  top5_accuracy = metric$top5_accuracy,
  balanced_accuracy = metric$balanced_accuracy,
  macro_f1 = metric$macro_f1,
  rss_before_fit_mb = rss_before,
  rss_after_fit_mb = rss_fit,
  rss_after_prediction_mb = rss_pred,
  gpu_before_fit_mb = gpu_before,
  gpu_after_fit_mb = gpu_fit,
  gpu_after_prediction_mb = gpu_pred,
  diagnostics_status = diagnostics$status,
  diagnostics_qualified_panel = diagnostics$rsvd$qualified_on_prespecified_panel,
  status = "success",
  stringsAsFactors = FALSE
)
write.csv(row, out, row.names = FALSE, na = "")
print(row)
