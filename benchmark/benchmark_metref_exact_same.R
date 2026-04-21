#!/usr/bin/env Rscript

# Exact thin wrapper around the main multianalysis benchmark,
# restricted to the MetRef dataset only.
# This keeps the same benchmark engine, method grid, and defaults
# as the broader benchmark workflow.

set_env_default <- function(name, value) {
  cur <- Sys.getenv(name, unset = NA_character_)
  if (length(cur) != 1L || is.na(cur) || !nzchar(cur)) {
    do.call(Sys.setenv, stats::setNames(list(as.character(value)), name))
  }
}

args <- commandArgs(trailingOnly = FALSE)
file_arg <- grep("^--file=", args, value = TRUE)
script_dir <- if (length(file_arg)) {
  dirname(normalizePath(sub("^--file=", "", file_arg[1L])))
} else {
  getwd()
}

set_env_default("FASTPLS_MULTI_OUTDIR", "metref_exact_same_benchmark")
set_env_default("FASTPLS_MULTI_APPEND", "false")
set_env_default("FASTPLS_ONLY_NCOMP", "true")
set_env_default("FASTPLS_DATASETS", "metref")

# Keep the same benchmark knobs as the remote/HPC-style run unless overridden.
set_env_default("FASTPLS_INCLUDE_CUDA", "true")
set_env_default("FASTPLS_INCLUDE_R_IMPL", "true")
set_env_default("FASTPLS_INCLUDE_PLS_PKG", "true")
set_env_default("FASTPLS_INCLUDE_SIMPLS_FAST_INCREMENTAL", "true")

set_env_default("FASTPLS_NCOMP_LIST", "2,5,10,20,50,100")
set_env_default("FASTPLS_METREF_DEFAULT_NCOMP", "20")
set_env_default("FASTPLS_METREF_REPS", "10")
set_env_default("FASTPLS_SEED", "123")

set_env_default("FASTPLS_THREADS", "1")
set_env_default("OMP_NUM_THREADS", Sys.getenv("FASTPLS_THREADS", "1"))
set_env_default("OPENBLAS_NUM_THREADS", Sys.getenv("FASTPLS_THREADS", "1"))
set_env_default("MKL_NUM_THREADS", Sys.getenv("FASTPLS_THREADS", "1"))
set_env_default("VECLIB_MAXIMUM_THREADS", Sys.getenv("FASTPLS_THREADS", "1"))
set_env_default("NUMEXPR_NUM_THREADS", Sys.getenv("FASTPLS_THREADS", "1"))

source(file.path(script_dir, "hpc_full_multianalysis_benchmark.R"), chdir = TRUE)
