#!/usr/bin/env Rscript

suppressPackageStartupMessages(library(data.table))

cmd_all <- commandArgs(FALSE)
file_arg <- sub("^--file=", "", cmd_all[grep("^--file=", cmd_all)])
script_dir <- if (length(file_arg)) dirname(normalizePath(file_arg[[1L]], mustWork = TRUE)) else getwd()
source(file.path(script_dir, "helpers_dataset_memory_compare.R"))

now_iso <- function() format(Sys.time(), "%Y-%m-%d %H:%M:%S")

time_expr <- function(expr) {
  gc()
  elapsed <- system.time(value <- force(expr))[["elapsed"]] * 1000
  list(value = value, ms = as.numeric(elapsed))
}

as_full_cv_data <- function(task) {
  X <- rbind(task$Xtrain, task$Xtest)
  if (is.factor(task$Ytrain)) {
    lev <- levels(task$Ytrain)
    Y <- factor(c(as.character(task$Ytrain), as.character(task$Ytest)), levels = lev)
  } else {
    Y <- rbind(as.matrix(task$Ytrain), as.matrix(task$Ytest))
  }
  list(X = X, Y = Y, constrain = seq_len(nrow(X)))
}

args <- parse_kv_args()
dataset <- arg_value(args, "dataset", required = TRUE)
variant <- arg_value(args, "variant", required = TRUE)
ncomp <- as.integer(arg_value(args, "ncomp", "50"))
kfold <- as.integer(arg_value(args, "kfold", "10"))
seed <- as.integer(arg_value(args, "seed", "123"))
lib_loc <- arg_value(args, "lib", Sys.getenv("FASTPLS_LIB", unset = ""))
out_dir <- path.expand(arg_value(args, "out_dir", file.path(getwd(), "cv_50_cpp_cuda_results")))
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)
out_csv <- file.path(out_dir, "cv_50_cpp_cuda_raw.csv")

if (nzchar(lib_loc)) {
  suppressPackageStartupMessages(library("fastPLS", lib.loc = lib_loc, character.only = TRUE))
} else {
  suppressPackageStartupMessages(library("fastPLS"))
}

cat(sprintf("[%s] loading dataset=%s variant=%s\n", now_iso(), dataset, variant))
task <- as_task(find_dataset_rdata(dataset), dataset_id = dataset, split_seed = seed)
cv <- as_full_cv_data(task)

fn <- switch(
  variant,
  cpp_plssvd_xprod = function() fastPLS::plssvd_cv_cpp(cv$X, cv$Y, cv$constrain, ncomp = ncomp, kfold = kfold, seed = seed, xprod = TRUE),
  cpp_plssvd = function() fastPLS::plssvd_cv_cpp(cv$X, cv$Y, cv$constrain, ncomp = ncomp, kfold = kfold, seed = seed, xprod = FALSE),
  cpp_simpls = function() fastPLS::simpls_cv_cpp(cv$X, cv$Y, cv$constrain, ncomp = ncomp, kfold = kfold, seed = seed),
  cpp_simpls_fast_xprod = function() fastPLS::simpls_fast_cv_cpp(cv$X, cv$Y, cv$constrain, ncomp = ncomp, kfold = kfold, seed = seed, xprod = TRUE),
  cpp_simpls_fast = function() fastPLS::simpls_fast_cv_cpp(cv$X, cv$Y, cv$constrain, ncomp = ncomp, kfold = kfold, seed = seed, xprod = FALSE),
  cuda_plssvd_xprod = function() fastPLS::plssvd_cv_cuda(cv$X, cv$Y, cv$constrain, ncomp = ncomp, kfold = kfold, seed = seed, xprod = TRUE),
  cuda_plssvd = function() fastPLS::plssvd_cv_cuda(cv$X, cv$Y, cv$constrain, ncomp = ncomp, kfold = kfold, seed = seed, xprod = FALSE),
  cuda_simpls_fast_xprod = function() fastPLS::simpls_fast_cv_cuda(cv$X, cv$Y, cv$constrain, ncomp = ncomp, kfold = kfold, seed = seed, xprod = TRUE),
  cuda_simpls_fast = function() fastPLS::simpls_fast_cv_cuda(cv$X, cv$Y, cv$constrain, ncomp = ncomp, kfold = kfold, seed = seed, xprod = FALSE),
  stop("Unknown variant: ", variant)
)

row <- data.frame(
  dataset = dataset,
  variant = variant,
  task_type = task$task_type,
  n = nrow(cv$X),
  p = ncol(cv$X),
  y_cols_or_classes = if (is.factor(cv$Y)) nlevels(cv$Y) else ncol(as.matrix(cv$Y)),
  ncomp = ncomp,
  kfold = kfold,
  time_ms = NA_real_,
  metric_name = NA_character_,
  metric_value = NA_real_,
  effective_ncomp = NA_integer_,
  status = "error",
  msg = "",
  stringsAsFactors = FALSE
)

res <- tryCatch(time_expr(fn()), error = function(e) e)
if (inherits(res, "error")) {
  row$status <- "error"
  row$msg <- conditionMessage(res)
} else {
  metrics <- res$value$metrics
  row$time_ms <- res$ms
  row$metric_name <- metrics$metric_name[[1L]]
  row$metric_value <- metrics$metric_value[[1L]]
  row$effective_ncomp <- as.integer(res$value$ncomp[[1L]])
  row$status <- "ok"
}

if (file.exists(out_csv)) {
  old <- data.table::fread(out_csv)
  data.table::fwrite(data.table::rbindlist(list(old, as.data.table(row)), fill = TRUE), out_csv)
} else {
  data.table::fwrite(row, out_csv)
}
print(row)
