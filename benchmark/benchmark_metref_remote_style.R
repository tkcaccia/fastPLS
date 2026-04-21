#!/usr/bin/env Rscript

# MetRef-only wrapper around the main multianalysis benchmark.
# This keeps the benchmark configuration aligned with the remote/HPC workflow:
# - same method grid generation
# - same R / Rcpp / pls_pkg / GPU inclusion logic
# - ncomp-only analysis

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

set_env_default("FASTPLS_MULTI_OUTDIR", "metref_remote_style_benchmark")
set_env_default("FASTPLS_MULTI_APPEND", "false")
set_env_default("FASTPLS_ONLY_NCOMP", "true")
set_env_default("FASTPLS_DATASETS", "metref")
set_env_default("FASTPLS_INCLUDE_CUDA", "true")
set_env_default("FASTPLS_INCLUDE_R_IMPL", "true")
set_env_default("FASTPLS_INCLUDE_PLS_PKG", "true")
set_env_default("FASTPLS_INCLUDE_SIMPLS_FAST_INCREMENTAL", "true")
set_env_default("FASTPLS_METREF_REPS", "10")
set_env_default("FASTPLS_NCOMP_LIST", "2,5,10,20,50,100")
set_env_default("FASTPLS_METREF_DEFAULT_NCOMP", "20")

source(file.path(script_dir, "hpc_full_multianalysis_benchmark.R"), chdir = TRUE)
