#!/usr/bin/env Rscript

args <- commandArgs(trailingOnly = TRUE)

parse_args <- function(args) {
  opts <- list(
    n = 4000L,
    p = 1000L,
    k = 10L,
    signal_rank = 10L,
    noise_sd = 0.02,
    reps = 3L,
    seed = 42L,
    rsvd_oversample = 10L,
    rsvd_power = 1L,
    use_openblas = 0L,
    openblas_threads = 0L,
    out_dir = file.path("benchmark", "results")
  )
  if (!length(args)) return(opts)

  for (arg in args) {
    if (!grepl("^--", arg)) next
    parts <- strsplit(sub("^--", "", arg), "=", fixed = TRUE)[[1]]
    key <- gsub("-", "_", parts[[1]])
    value <- if (length(parts) > 1L) parts[[2]] else "TRUE"
    if (!key %in% names(opts)) stop(sprintf("Unknown option: %s", key))
    cur <- opts[[key]]
    if (is.integer(cur)) {
      opts[[key]] <- as.integer(value)
    } else if (is.numeric(cur)) {
      opts[[key]] <- as.numeric(value)
    } else {
      opts[[key]] <- value
    }
  }
  opts
}

opts <- parse_args(args)

script_arg <- commandArgs()[grep("^--file=", commandArgs())][1]
script_path <- if (length(script_arg) == 0L || is.na(script_arg)) {
  file.path("benchmark", "benchmark_large_rsvd.R")
} else {
  sub("^--file=", "", script_arg)
}
repo_root <- normalizePath(file.path(dirname(script_path), ".."), winslash = "/", mustWork = FALSE)
setwd(repo_root)

bench_lib <- if (isTRUE(as.integer(opts$use_openblas) == 1L)) {
  file.path(repo_root, ".bench-lib-openblas-bench")
} else {
  file.path(repo_root, ".bench-lib-default-bench")
}
dir.create(bench_lib, recursive = TRUE, showWarnings = FALSE)
install_env <- character()
if (isTRUE(as.integer(opts$use_openblas) == 1L)) {
  install_env <- c(
    "FASTPLS_USE_OPENBLAS=1",
    sprintf("OPENBLAS_ROOT=%s", Sys.getenv("OPENBLAS_ROOT", "/opt/homebrew/opt/openblas"))
  )
}
status <- system2("R", c("CMD", "INSTALL", "--preclean", ".", "-l", bench_lib), env = install_env)
if (!identical(status, 0L)) stop("Failed to install fastPLS into benchmark library")

if (isTRUE(as.integer(opts$openblas_threads) > 0L)) {
  Sys.setenv(OPENBLAS_NUM_THREADS = as.character(as.integer(opts$openblas_threads)))
}

suppressPackageStartupMessages(library("fastPLS", lib.loc = bench_lib, character.only = TRUE))

make_matrix <- function(n, p, signal_rank, noise_sd, seed) {
  set.seed(seed)
  u <- qr.Q(qr(matrix(rnorm(n * signal_rank), nrow = n, ncol = signal_rank)))
  v <- qr.Q(qr(matrix(rnorm(p * signal_rank), nrow = p, ncol = signal_rank)))
  d <- diag(seq(signal_rank, 1, length.out = signal_rank), nrow = signal_rank)
  u %*% d %*% t(v) + matrix(rnorm(n * p, sd = noise_sd), nrow = n, ncol = p)
}

run_timed <- function(expr, reps) {
  elapsed <- numeric(reps)
  last <- NULL
  for (i in seq_len(reps)) {
    gc(FALSE)
    t0 <- proc.time()[["elapsed"]]
    last <- force(expr)
    elapsed[i] <- proc.time()[["elapsed"]] - t0
  }
  list(times = elapsed, value = last)
}

subspace_alignment <- function(u_ref, u_est) {
  mean(diag(abs(crossprod(u_ref, u_est))))
}

A <- make_matrix(opts$n, opts$p, opts$signal_rank, opts$noise_sd, opts$seed)

cpu_rsvd <- run_timed(
  fastPLS:::truncated_svd_debug(
    A = A,
    k = opts$k,
    svd_method = 4L,
    rsvd_oversample = opts$rsvd_oversample,
    rsvd_power = opts$rsvd_power,
    svds_tol = 0,
    seed = opts$seed,
    left_only = FALSE
  ),
  reps = opts$reps
)

cpu_exact <- run_timed(
  fastPLS:::truncated_svd_debug(
    A = A,
    k = opts$k,
    svd_method = 3L,
    rsvd_oversample = opts$rsvd_oversample,
    rsvd_power = opts$rsvd_power,
    svds_tol = 0,
    seed = opts$seed,
    left_only = FALSE
  ),
  reps = 1L
)

rsvd_out <- cpu_rsvd$value
exact_out <- cpu_exact$value

results <- data.frame(
  method = c("cpu_rsvd", "cpu_exact"),
  median_seconds = c(median(cpu_rsvd$times), median(cpu_exact$times)),
  min_seconds = c(min(cpu_rsvd$times), min(cpu_exact$times)),
  max_seconds = c(max(cpu_rsvd$times), max(cpu_exact$times)),
  max_rel_sigma_error = c(
    max(abs(rsvd_out$d - exact_out$d) / pmax(abs(exact_out$d), .Machine$double.eps)),
    0
  ),
  left_subspace_alignment = c(
    subspace_alignment(exact_out$u[, seq_len(opts$k), drop = FALSE], rsvd_out$u[, seq_len(opts$k), drop = FALSE]),
    1
  ),
  use_openblas = c(as.integer(opts$use_openblas), as.integer(opts$use_openblas)),
  openblas_threads = c(as.integer(opts$openblas_threads), as.integer(opts$openblas_threads)),
  stringsAsFactors = FALSE
)

dir.create(opts$out_dir, recursive = TRUE, showWarnings = FALSE)
stamp <- format(Sys.time(), "%Y%m%d-%H%M%S")
csv_path <- file.path(opts$out_dir, sprintf("large-rsvd-%s.csv", stamp))
write.csv(results, csv_path, row.names = FALSE)

cat("Saved benchmark to ", csv_path, "\n", sep = "")
print(results, row.names = FALSE)
