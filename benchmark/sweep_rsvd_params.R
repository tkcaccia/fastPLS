#!/usr/bin/env Rscript

args <- commandArgs(trailingOnly = TRUE)

parse_args <- function(args) {
  opts <- list(
    n = 3000L,
    p = 800L,
    k = 10L,
    signal_rank = 10L,
    noise_sd = 0.02,
    reps = 10L,
    seed = 42L,
    oversamples = "4,6,8,10,12,16",
    powers = "0,1,2",
    use_openblas = 1L,
    openblas_threads = 1L,
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
  file.path("benchmark", "sweep_rsvd_params.R")
} else {
  sub("^--file=", "", script_arg)
}
repo_root <- normalizePath(file.path(dirname(script_path), ".."), winslash = "/", mustWork = FALSE)
setwd(repo_root)

bench_lib <- if (isTRUE(as.integer(opts$use_openblas) == 1L)) {
  file.path(repo_root, ".bench-lib-openblas-sweep")
} else {
  file.path(repo_root, ".bench-lib-default-sweep")
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
if (!identical(status, 0L)) stop("Failed to install fastPLS into sweep library")

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

subspace_alignment <- function(u_ref, u_est) {
  mean(diag(abs(crossprod(u_ref, u_est))))
}

run_avg <- function(A, k, oversample, power, reps, seed) {
  t0 <- proc.time()[["elapsed"]]
  last <- NULL
  for (i in seq_len(reps)) {
    last <- fastPLS:::truncated_svd_debug(
      A = A,
      k = k,
      svd_method = 4L,
      rsvd_oversample = oversample,
      rsvd_power = power,
      svds_tol = 0,
      seed = seed + i - 1L,
      left_only = FALSE
    )
  }
  list(avg = (proc.time()[["elapsed"]] - t0) / reps, res = last)
}

to_ints <- function(x) as.integer(strsplit(x, ",", fixed = TRUE)[[1]])

A <- make_matrix(opts$n, opts$p, opts$signal_rank, opts$noise_sd, opts$seed)
exact <- fastPLS:::truncated_svd_debug(
  A = A,
  k = opts$k,
  svd_method = 3L,
  rsvd_oversample = 10L,
  rsvd_power = 1L,
  svds_tol = 0,
  seed = opts$seed,
  left_only = FALSE
)

oversamples <- to_ints(opts$oversamples)
powers <- to_ints(opts$powers)
rows <- vector("list", length(oversamples) * length(powers))
idx <- 1L

for (oversample in oversamples) {
  for (power in powers) {
    cur <- run_avg(A, opts$k, oversample, power, opts$reps, opts$seed)
    rows[[idx]] <- data.frame(
      oversample = oversample,
      power = power,
      avg_seconds = cur$avg,
      max_rel_sigma_error = max(abs(cur$res$d - exact$d) / pmax(abs(exact$d), .Machine$double.eps)),
      left_subspace_alignment = subspace_alignment(
        exact$u[, seq_len(opts$k), drop = FALSE],
        cur$res$u[, seq_len(opts$k), drop = FALSE]
      ),
      use_openblas = as.integer(opts$use_openblas),
      openblas_threads = as.integer(opts$openblas_threads),
      stringsAsFactors = FALSE
    )
    idx <- idx + 1L
  }
}

res <- do.call(rbind, rows)
res <- res[order(res$avg_seconds, -res$left_subspace_alignment), ]

dir.create(opts$out_dir, recursive = TRUE, showWarnings = FALSE)
stamp <- format(Sys.time(), "%Y%m%d-%H%M%S")
csv_path <- file.path(opts$out_dir, sprintf("rsvd-param-sweep-%s.csv", stamp))
write.csv(res, csv_path, row.names = FALSE)

cat("Saved sweep to ", csv_path, "\n", sep = "")
print(res, row.names = FALSE)
