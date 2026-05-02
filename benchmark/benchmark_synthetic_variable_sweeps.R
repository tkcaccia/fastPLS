#!/usr/bin/env Rscript

suppressPackageStartupMessages({
  library(data.table)
})

bench_lib <- Sys.getenv("FASTPLS_BENCH_LIB", "")
if (nzchar(bench_lib)) {
  bench_lib <- path.expand(bench_lib)
  if (dir.exists(bench_lib)) {
    .libPaths(unique(c(normalizePath(bench_lib, winslash = "/", mustWork = FALSE), .libPaths())))
  }
}

suppressPackageStartupMessages({
  library(fastPLS)
})

`%||%` <- function(x, y) {
  if (is.null(x) || length(x) == 0L) return(y)
  if (length(x) == 1L && is.na(x)) return(y)
  x_chr <- as.character(x)
  if (all(is.na(x_chr) | !nzchar(x_chr))) return(y)
  x
}

script_arg <- grep("^--file=", commandArgs(trailingOnly = FALSE), value = TRUE)
script_file <- if (length(script_arg)) sub("^--file=", "", script_arg[[1L]]) else file.path(getwd(), "benchmark", "benchmark_synthetic_variable_sweeps.R")
script_dir <- dirname(normalizePath(script_file, winslash = "/", mustWork = FALSE))
repo_root <- normalizePath(file.path(script_dir, ".."), winslash = "/", mustWork = FALSE)

bool_env <- function(name, default = FALSE) {
  val <- tolower(Sys.getenv(name, if (isTRUE(default)) "true" else "false"))
  val %in% c("1", "true", "yes", "y")
}

num_list_env <- function(name, default) {
  raw <- Sys.getenv(name, "")
  if (!nzchar(raw)) return(default)
  vals <- suppressWarnings(as.numeric(trimws(strsplit(raw, ",", fixed = TRUE)[[1L]])))
  vals[is.finite(vals)]
}

chr_list_env <- function(name, default) {
  raw <- Sys.getenv(name, "")
  if (!nzchar(raw)) return(default)
  vals <- trimws(strsplit(raw, ",", fixed = TRUE)[[1L]])
  vals[nzchar(vals)]
}

threads <- suppressWarnings(as.integer(Sys.getenv("FASTPLS_THREADS", "1")))
if (!is.finite(threads) || is.na(threads) || threads < 1L) threads <- 1L
Sys.setenv(
  OMP_NUM_THREADS = as.character(threads),
  OPENBLAS_NUM_THREADS = as.character(threads),
  MKL_NUM_THREADS = as.character(threads),
  VECLIB_MAXIMUM_THREADS = as.character(threads),
  NUMEXPR_NUM_THREADS = as.character(threads)
)
if (requireNamespace("RhpcBLASctl", quietly = TRUE)) {
  RhpcBLASctl::blas_set_num_threads(threads)
  RhpcBLASctl::omp_set_num_threads(threads)
}

out_dir <- path.expand(Sys.getenv("FASTPLS_SYNTH_VAR_OUTDIR", file.path(repo_root, "benchmark_results_synthetic_variable_sweeps")))
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)
raw_file <- file.path(out_dir, "synthetic_variable_sweeps_raw.csv")
summary_file <- file.path(out_dir, "synthetic_variable_sweeps_summary.csv")
manifest_file <- file.path(out_dir, "synthetic_variable_sweeps_manifest.txt")
progress_file <- file.path(out_dir, "synthetic_variable_sweeps_progress.log")

log_msg <- function(...) {
  txt <- paste0("[", format(Sys.time(), "%Y-%m-%d %H:%M:%S"), "] ", paste(..., collapse = ""))
  message(txt)
  cat(txt, "\n", file = progress_file, append = TRUE)
}

reps <- suppressWarnings(as.integer(Sys.getenv("FASTPLS_SYNTH_VAR_REPS", "3")))
if (!is.finite(reps) || is.na(reps) || reps < 1L) reps <- 3L
ncomp <- suppressWarnings(as.integer(Sys.getenv("FASTPLS_SYNTH_VAR_NCOMP", "20")))
if (!is.finite(ncomp) || is.na(ncomp) || ncomp < 1L) ncomp <- 20L
timeout_sec <- suppressWarnings(as.numeric(Sys.getenv("FASTPLS_SYNTH_VAR_TIMEOUT_SEC", "1200")))
if (!is.finite(timeout_sec) || is.na(timeout_sec) || timeout_sec <= 0) timeout_sec <- 1200
max_host_rss_mb <- suppressWarnings(as.numeric(Sys.getenv("FASTPLS_SYNTH_VAR_MAX_HOST_RSS_MB", "Inf")))
if (!is.finite(max_host_rss_mb) || is.na(max_host_rss_mb) || max_host_rss_mb <= 0) max_host_rss_mb <- Inf
base_seed <- suppressWarnings(as.integer(Sys.getenv("FASTPLS_SYNTH_VAR_SEED", "123")))
if (!is.finite(base_seed) || is.na(base_seed)) base_seed <- 123L

include_gpu <- bool_env("FASTPLS_SYNTH_VAR_INCLUDE_GPU", TRUE)
include_r <- bool_env("FASTPLS_SYNTH_VAR_INCLUDE_R", FALSE)
include_pls_pkg <- bool_env("FASTPLS_SYNTH_VAR_INCLUDE_PLS_PKG", TRUE)
include_classification <- bool_env("FASTPLS_SYNTH_VAR_INCLUDE_CLASSIFICATION", TRUE)
measure_memory <- bool_env("FASTPLS_MEASURE_MEMORY", TRUE)
cuda_ok <- tryCatch(isTRUE(fastPLS::has_cuda()), error = function(e) FALSE)

proc_rss_mb <- function(pid = Sys.getpid()) {
  status_file <- file.path("/proc", as.character(pid), "status")
  if (!file.exists(status_file)) return(NA_real_)
  line <- grep("^VmRSS:", readLines(status_file, warn = FALSE), value = TRUE)
  if (!length(line)) return(NA_real_)
  kb <- suppressWarnings(as.numeric(strsplit(line[[1L]], "[[:space:]]+")[[1L]][2L]))
  if (is.finite(kb)) kb / 1024 else NA_real_
}

child_pids <- function(pid) {
  kids <- suppressWarnings(system2("pgrep", c("-P", as.character(pid)), stdout = TRUE, stderr = FALSE))
  kids <- suppressWarnings(as.integer(kids))
  kids <- kids[is.finite(kids)]
  unique(kids)
}

proc_tree_pids <- function(pid) {
  out <- as.integer(pid)
  frontier <- child_pids(pid)
  while (length(frontier)) {
    out <- c(out, frontier)
    frontier <- unique(unlist(lapply(frontier, child_pids), use.names = FALSE))
    frontier <- setdiff(frontier, out)
  }
  unique(out)
}

proc_tree_rss_mb <- function(pid) {
  vals <- vapply(proc_tree_pids(pid), proc_rss_mb, numeric(1L))
  vals <- vals[is.finite(vals)]
  if (length(vals)) sum(vals) else NA_real_
}

families <- chr_list_env(
  "FASTPLS_SYNTH_VAR_FAMILIES",
  c("reg_n", "reg_p", "reg_q", "class_n", "class_p")
)
if (!isTRUE(include_classification)) {
  families <- families[!grepl("^class_", families)]
}

grids <- list(
  reg_n = num_list_env("FASTPLS_SYNTH_VAR_GRID_REG_N", c(200, 500, 1000, 2000, 5000, 10000, 20000, 30000, 50000)),
  reg_p = num_list_env("FASTPLS_SYNTH_VAR_GRID_REG_P", c(100, 500, 1000, 2000, 5000, 10000, 20000, 30000, 50000)),
  reg_q = num_list_env("FASTPLS_SYNTH_VAR_GRID_REG_Q", c(10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000)),
  reg_noise = num_list_env("FASTPLS_SYNTH_VAR_GRID_REG_NOISE", c(0.10, 0.25, 0.50, 1.00, 2.00, 4.00, 8.00, 16.00, 32.00)),
  class_n = num_list_env("FASTPLS_SYNTH_VAR_GRID_CLASS_N", c(200, 500, 1000, 2000, 5000, 10000, 20000, 30000, 50000)),
  class_p = num_list_env("FASTPLS_SYNTH_VAR_GRID_CLASS_P", c(100, 500, 1000, 2000, 5000, 10000, 20000, 30000, 50000)),
  class_noise = num_list_env("FASTPLS_SYNTH_VAR_GRID_CLASS_NOISE", c(0.10, 0.25, 0.50, 1.00, 2.00, 4.00, 8.00, 16.00, 32.00))
)

family_meta <- list(
  reg_n = list(task_type = "regression", variable = "n_train", x_label = "Training samples", base_n = 800L, base_p = 1000L, base_q = 100L, base_noise = 0.50),
  reg_p = list(task_type = "regression", variable = "p", x_label = "X variables (p)", base_n = 5000L, base_p = 1000L, base_q = 100L, base_noise = 0.50),
  reg_q = list(task_type = "regression", variable = "q", x_label = "Y variables (q)", base_n = 5000L, base_p = 1000L, base_q = 100L, base_noise = 0.50),
  reg_noise = list(task_type = "regression", variable = "noise", x_label = "Noise SD", base_n = 5000L, base_p = 1000L, base_q = 100L, base_noise = 0.50),
  class_n = list(task_type = "classification", variable = "n_train", x_label = "Training samples", base_n = 800L, base_p = 1000L, base_k = 5L, base_noise = 0.50),
  class_p = list(task_type = "classification", variable = "p", x_label = "X variables (p)", base_n = 5000L, base_p = 1000L, base_k = 5L, base_noise = 0.50),
  class_noise = list(task_type = "classification", variable = "noise", x_label = "Noise SD", base_n = 5000L, base_p = 1000L, base_k = 5L, base_noise = 0.50)
)

unknown_families <- setdiff(families, names(family_meta))
if (length(unknown_families)) {
  stop("Unknown FASTPLS_SYNTH_VAR_FAMILIES entries: ", paste(unknown_families, collapse = ", "))
}

normalize_columns <- function(x) {
  x <- qr.Q(qr(x))
  x[, seq_len(min(ncol(x), nrow(x))), drop = FALSE]
}

center_train_test <- function(train, test) {
  mu <- colMeans(train)
  list(
    train = sweep(train, 2L, mu, "-", check.margin = FALSE),
    test = sweep(test, 2L, mu, "-", check.margin = FALSE)
  )
}

make_regression_task <- function(family, x_value, seed) {
  cfg <- family_meta[[family]]
  ntrain <- as.integer(cfg$base_n)
  ntest <- max(100L, as.integer(round(0.25 * ntrain)))
  p <- as.integer(cfg$base_p)
  q <- as.integer(cfg$base_q)
  noise <- as.numeric(cfg$base_noise)

  if (identical(cfg$variable, "n_train")) {
    ntrain <- as.integer(x_value)
    ntest <- max(100L, as.integer(round(0.25 * ntrain)))
  } else if (identical(cfg$variable, "p")) {
    p <- as.integer(x_value)
  } else if (identical(cfg$variable, "q")) {
    q <- as.integer(x_value)
  } else if (identical(cfg$variable, "noise")) {
    noise <- as.numeric(x_value)
  }

  set.seed(as.integer(seed))
  rank_true <- max(1L, min(10L, ntrain - 1L, p, q))
  decay <- seq(1, 0.2, length.out = rank_true)
  P <- normalize_columns(matrix(rnorm(p * rank_true), nrow = p, ncol = rank_true))
  C <- normalize_columns(matrix(rnorm(q * rank_true), nrow = q, ncol = rank_true))
  Ttrain <- matrix(rnorm(ntrain * rank_true), nrow = ntrain, ncol = rank_true)
  Ttest <- matrix(rnorm(ntest * rank_true), nrow = ntest, ncol = rank_true)

  Xtrain <- sweep(Ttrain, 2L, decay, "*", check.margin = FALSE) %*% t(P) +
    noise * matrix(rnorm(ntrain * p), nrow = ntrain, ncol = p)
  Xtest <- sweep(Ttest, 2L, decay, "*", check.margin = FALSE) %*% t(P) +
    noise * matrix(rnorm(ntest * p), nrow = ntest, ncol = p)
  Ytrain <- sweep(Ttrain, 2L, decay, "*", check.margin = FALSE) %*% t(C) +
    noise * matrix(rnorm(ntrain * q), nrow = ntrain, ncol = q)
  Ytest <- sweep(Ttest, 2L, decay, "*", check.margin = FALSE) %*% t(C) +
    noise * matrix(rnorm(ntest * q), nrow = ntest, ncol = q)

  Xc <- center_train_test(Xtrain, Xtest)
  Yc <- center_train_test(Ytrain, Ytest)
  list(
    task_type = "regression",
    Xtrain = Xc$train,
    Xtest = Xc$test,
    Ytrain = Yc$train,
    Ytest = Yc$test,
    n_train = ntrain,
    n_test = ntest,
    p = p,
    q = q,
    n_classes = q,
    noise = noise,
    rank_true = rank_true
  )
}

make_classification_task <- function(family, x_value, seed) {
  cfg <- family_meta[[family]]
  ntrain <- as.integer(cfg$base_n)
  ntest <- max(100L, as.integer(round(0.25 * ntrain)))
  p <- as.integer(cfg$base_p)
  k <- as.integer(cfg$base_k)
  noise <- as.numeric(cfg$base_noise)

  if (identical(cfg$variable, "n_train")) {
    ntrain <- as.integer(x_value)
    ntest <- max(100L, as.integer(round(0.25 * ntrain)))
  } else if (identical(cfg$variable, "p")) {
    p <- as.integer(x_value)
  } else if (identical(cfg$variable, "noise")) {
    noise <- as.numeric(x_value)
  }

  set.seed(as.integer(seed))
  rank_true <- max(2L, min(10L, p, k))
  P <- normalize_columns(matrix(rnorm(p * rank_true), nrow = p, ncol = rank_true))
  prototypes <- matrix(rnorm(k * rank_true), nrow = k, ncol = rank_true)
  prototypes <- prototypes / (sqrt(rowSums(prototypes^2)) + 1e-8)
  prototypes <- 3 * prototypes

  make_block <- function(n) {
    cls <- sample(rep(seq_len(k), length.out = n))
    T <- prototypes[cls, , drop = FALSE] + 0.35 * matrix(rnorm(n * rank_true), nrow = n, ncol = rank_true)
    X <- T %*% t(P) + noise * matrix(rnorm(n * p), nrow = n, ncol = p)
    y <- factor(paste0("class", cls), levels = paste0("class", seq_len(k)))
    list(X = X, y = y)
  }

  tr <- make_block(ntrain)
  te <- make_block(ntest)
  Xc <- center_train_test(tr$X, te$X)
  list(
    task_type = "classification",
    Xtrain = Xc$train,
    Xtest = Xc$test,
    Ytrain = tr$y,
    Ytest = te$y,
    n_train = ntrain,
    n_test = ntest,
    p = p,
    q = 1L,
    n_classes = k,
    noise = noise,
    rank_true = rank_true
  )
}

make_task <- function(family, x_value, seed) {
  if (identical(family_meta[[family]]$task_type, "classification")) {
    make_classification_task(family, x_value, seed)
  } else {
    make_regression_task(family, x_value, seed)
  }
}

read_proc_rss_mb <- function(pid = Sys.getpid()) {
  path <- file.path("/proc", as.character(pid), "status")
  if (!file.exists(path)) return(NA_real_)
  line <- grep("^VmRSS:", readLines(path, warn = FALSE), value = TRUE)
  if (!length(line)) return(NA_real_)
  kb <- suppressWarnings(as.numeric(sub("^VmRSS:\\s*([0-9.]+).*", "\\1", line[[1L]])))
  if (!is.finite(kb)) NA_real_ else kb / 1024
}

read_gpu_process_mem_mb <- function(pid = Sys.getpid()) {
  if (!nzchar(Sys.which("nvidia-smi"))) return(NA_real_)
  x <- tryCatch(
    system2(
      "nvidia-smi",
      c("--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"),
      stdout = TRUE,
      stderr = FALSE
    ),
    error = function(e) character()
  )
  if (!length(x)) return(NA_real_)
  vals <- vapply(strsplit(x, ",", fixed = TRUE), function(parts) {
    parts <- trimws(parts)
    if (length(parts) < 2L || !identical(parts[[1L]], as.character(pid))) return(NA_real_)
    suppressWarnings(as.numeric(parts[[2L]]))
  }, numeric(1L))
  vals <- vals[is.finite(vals)]
  if (!length(vals)) NA_real_ else sum(vals)
}

start_memory_monitor <- function(track_gpu = FALSE) {
  pid <- Sys.getpid()
  rss_before <- read_proc_rss_mb(pid)
  gpu_before <- if (isTRUE(track_gpu)) read_gpu_process_mem_mb(pid) else NA_real_
  if (!isTRUE(measure_memory) || !file.exists(file.path("/proc", as.character(pid), "status"))) {
    return(list(pid = pid, sampler_pid = NA_integer_, path = NA_character_, rss_before_mb = rss_before, gpu_before_mb = gpu_before, track_gpu = track_gpu))
  }
  interval <- suppressWarnings(as.numeric(Sys.getenv("FASTPLS_MEMORY_SAMPLE_SEC", "0.2")))
  if (!is.finite(interval) || interval <= 0) interval <- 0.2
  tmp <- tempfile("fastpls_synth_mem_", fileext = ".csv")
  nvidia_cmd <- if (isTRUE(track_gpu) && nzchar(Sys.which("nvidia-smi"))) {
    sprintf(
      "gpu=$(nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader,nounits 2>/dev/null | awk -F, -v pid=%d '{gsub(/ /,\"\",$1); gsub(/ /,\"\",$2); if ($1==pid) s+=$2} END{if (s==\"\") print \"NA\"; else print s}')",
      pid
    )
  } else {
    "gpu=NA"
  }
  loop <- sprintf(
    "printf 'rss_mb,gpu_mem_mb\\n' > %s; (while :; do rss=$(awk '/VmRSS:/ {print $2/1024}' /proc/%d/status 2>/dev/null); if [ -z \"$rss\" ]; then rss=NA; fi; %s; printf '%%s,%%s\\n' \"$rss\" \"$gpu\" >> %s; sleep %.3f; done) >/dev/null 2>&1 & echo $!",
    shQuote(tmp), pid, nvidia_cmd, shQuote(tmp), interval
  )
  sampler_pid <- suppressWarnings(as.integer(system(paste("sh -c", shQuote(loop)), intern = TRUE)))
  if (!length(sampler_pid) || !is.finite(sampler_pid[[1L]])) sampler_pid <- NA_integer_
  list(pid = pid, sampler_pid = sampler_pid[[1L]], path = tmp, rss_before_mb = rss_before, gpu_before_mb = gpu_before, track_gpu = track_gpu)
}

stop_memory_monitor <- function(mon) {
  if (is.null(mon)) return(list(peak_host_rss_mb = NA_real_, host_rss_delta_mb = NA_real_, peak_gpu_mem_mb = NA_real_, gpu_mem_delta_mb = NA_real_))
  if (is.finite(mon$sampler_pid)) {
    try(system2("kill", as.character(mon$sampler_pid), stdout = FALSE, stderr = FALSE), silent = TRUE)
  }
  Sys.sleep(0.05)
  gpu_after <- if (isTRUE(mon$track_gpu)) read_gpu_process_mem_mb(mon$pid) else NA_real_
  rss_vals <- c(mon$rss_before_mb, read_proc_rss_mb(mon$pid))
  gpu_vals <- c(mon$gpu_before_mb, gpu_after)
  if (is.character(mon$path) && file.exists(mon$path)) {
    smp <- tryCatch(fread(mon$path), error = function(e) NULL)
    if (!is.null(smp)) {
      if ("rss_mb" %in% names(smp)) rss_vals <- c(rss_vals, suppressWarnings(as.numeric(smp$rss_mb)))
      if ("gpu_mem_mb" %in% names(smp)) gpu_vals <- c(gpu_vals, suppressWarnings(as.numeric(smp$gpu_mem_mb)))
    }
    unlink(mon$path)
  }
  rss_vals <- rss_vals[is.finite(rss_vals)]
  gpu_vals <- gpu_vals[is.finite(gpu_vals)]
  peak_rss <- if (length(rss_vals)) max(rss_vals, na.rm = TRUE) else NA_real_
  peak_gpu <- if (length(gpu_vals)) max(gpu_vals, na.rm = TRUE) else NA_real_
  list(
    peak_host_rss_mb = peak_rss,
    host_rss_delta_mb = if (is.finite(peak_rss) && is.finite(mon$rss_before_mb)) max(0, peak_rss - mon$rss_before_mb) else NA_real_,
    peak_gpu_mem_mb = peak_gpu,
    gpu_mem_delta_mb = if (is.finite(peak_gpu) && is.finite(mon$gpu_before_mb)) max(0, peak_gpu - mon$gpu_before_mb) else NA_real_
  )
}

metric_from_prediction <- function(task, pred) {
  if (identical(task$task_type, "classification")) {
    yhat <- pred$Ypred
    if (is.data.frame(yhat)) yhat <- yhat[[ncol(yhat)]]
    acc <- mean(as.character(yhat) == as.character(task$Ytest), na.rm = TRUE)
    return(list(metric_name = "accuracy", metric_value = as.numeric(acc), accuracy = as.numeric(acc), q2 = NA_real_, rmsd = NA_real_))
  }
  ypred <- pred$Ypred
  if (is.null(ypred)) stop("Prediction object does not contain Ypred")
  if (length(dim(ypred)) == 3L) ypred <- ypred[, , dim(ypred)[3L], drop = TRUE]
  ypred <- as.matrix(ypred)
  ytrue <- as.matrix(task$Ytest)
  if (!all(dim(ypred) == dim(ytrue))) {
    ypred <- matrix(as.numeric(ypred), nrow = nrow(ytrue), ncol = ncol(ytrue))
  }
  rmsd <- sqrt(mean((ypred - ytrue)^2))
  sst <- sum((ytrue - matrix(colMeans(ytrue), nrow = nrow(ytrue), ncol = ncol(ytrue), byrow = TRUE))^2)
  q2 <- if (is.finite(sst) && sst > 0) 1 - sum((ypred - ytrue)^2) / sst else NA_real_
  metric_name <- if (ncol(ytrue) == 1L) "Q2" else "RMSD"
  metric_value <- if (identical(metric_name, "Q2")) q2 else rmsd
  list(metric_name = metric_name, metric_value = as.numeric(metric_value), accuracy = NA_real_, q2 = as.numeric(q2), rmsd = as.numeric(rmsd))
}

pls_pkg_fit <- function(task, method_panel, ncomp_run) {
  if (!requireNamespace("pls", quietly = TRUE)) stop("pls package not available")
  if (identical(task$task_type, "classification")) {
    Ymm <- model.matrix(~ task$Ytrain - 1)
    colnames(Ymm) <- levels(task$Ytrain)
  } else {
    Ymm <- as.matrix(task$Ytrain)
  }
  fit_fun <- switch(
    method_panel,
    simpls = pls::simpls.fit,
    opls = pls::oscorespls.fit,
    kernelpls = pls::kernelpls.fit,
    stop("pls_pkg is not available for ", method_panel)
  )
  t0 <- proc.time()[3]
  mdl <- fit_fun(task$Xtrain, Ymm, ncomp = as.integer(ncomp_run), center = TRUE, stripped = TRUE)
  fit_ms <- (proc.time()[3] - t0) * 1000
  list(model = mdl, fit_ms = fit_ms, levels_y = if (is.factor(task$Ytrain)) levels(task$Ytrain) else NULL)
}

pls_pkg_predict <- function(fit, task, ncomp_run) {
  mdl <- fit$model
  coef_arr <- mdl$coefficients
  coef_mat <- coef_arr[, , as.integer(ncomp_run), drop = TRUE]
  if (is.null(dim(coef_mat))) {
    nout <- if (identical(task$task_type, "classification")) length(fit$levels_y) else length(mdl$Ymeans)
    coef_mat <- matrix(coef_mat, ncol = nout)
  }
  Xc <- sweep(as.matrix(task$Xtest), 2L, mdl$Xmeans, "-", check.margin = FALSE)
  pred <- Xc %*% coef_mat + matrix(mdl$Ymeans, nrow = nrow(Xc), ncol = length(mdl$Ymeans), byrow = TRUE)
  if (identical(task$task_type, "classification")) {
    lev <- colnames(pred) %||% fit$levels_y
    yhat <- factor(lev[max.col(pred, ties.method = "first")], levels = fit$levels_y)
    return(list(Ypred = yhat))
  }
  list(Ypred = pred)
}

variant_specs <- function(task) {
  specs <- data.table(
    variant_name = c(
      "cpp_plssvd_cpu_rsvd", "cpp_plssvd_irlba",
      "cpp_simpls_cpu_rsvd", "cpp_simpls_irlba",
      "cpp_opls_cpu_rsvd", "cpp_opls_irlba",
      "cpp_kernelpls_cpu_rsvd", "cpp_kernelpls_irlba"
    ),
    method_panel = c("plssvd", "plssvd", "simpls", "simpls", "opls", "opls", "kernelpls", "kernelpls"),
    engine = "CPU",
    implementation = "cpp",
    backend_algorithm = c("rsvd", "irlba", "rsvd", "irlba", "rsvd", "irlba", "rsvd", "irlba")
  )
  if (isTRUE(include_r)) {
    specs <- rbind(
      specs,
      data.table(
        variant_name = c("r_plssvd_cpu_rsvd", "r_plssvd_irlba", "r_simpls_cpu_rsvd", "r_simpls_irlba", "r_opls_cpu_rsvd", "r_opls_irlba", "r_kernelpls_cpu_rsvd", "r_kernelpls_irlba"),
        method_panel = c("plssvd", "plssvd", "simpls", "simpls", "opls", "opls", "kernelpls", "kernelpls"),
        engine = "CPU",
        implementation = "R",
        backend_algorithm = c("rsvd", "irlba", "rsvd", "irlba", "rsvd", "irlba", "rsvd", "irlba")
      ),
      fill = TRUE
    )
  }
  if (isTRUE(include_gpu) && isTRUE(cuda_ok)) {
    specs <- rbind(
      specs,
      data.table(
        variant_name = c(
          "gpu_plssvd_fp64", "gpu_simpls_fp64", "gpu_opls_fp64", "gpu_kernelpls_fp64"
        ),
        method_panel = c(
          "plssvd", "simpls", "opls", "kernelpls"
        ),
        engine = "GPU",
        implementation = "cuda",
        backend_algorithm = "rsvd"
      ),
      fill = TRUE
    )
  }
  if (isTRUE(include_pls_pkg) && requireNamespace("pls", quietly = TRUE)) {
    specs <- rbind(
      specs,
      data.table(
        variant_name = c("pls_pkg_simpls", "pls_pkg_opls", "pls_pkg_kernelpls"),
        method_panel = c("simpls", "opls", "kernelpls"),
        engine = "CPU",
        implementation = "pls_pkg",
        backend_algorithm = "pls_pkg"
      ),
      fill = TRUE
    )
  }
  specs[]
}

fit_predict_variant <- function(task, spec, ncomp_run, seed) {
  # PLSSVD is capped by Y dimensionality/classes.
  if (identical(spec$method_panel, "plssvd") && ncomp_run > task$n_classes) {
    return(list(status = "skipped_plssvd_cap", msg = sprintf("requested ncomp=%d exceeds Y/classes=%d", ncomp_run, task$n_classes)))
  }

  opls_layout <- list(ncomp = as.integer(ncomp_run), north = 0L)
  if (identical(spec$method_panel, "opls")) {
    opls_layout$north <- min(1L, max(0L, as.integer(ncomp_run) - 1L))
    opls_layout$ncomp <- max(1L, as.integer(ncomp_run) - opls_layout$north)
  }

  mon <- start_memory_monitor(track_gpu = identical(spec$engine, "GPU"))
  on.exit(try(stop_memory_monitor(mon), silent = TRUE), add = TRUE)

  if (identical(spec$implementation, "pls_pkg")) {
    fit <- pls_pkg_fit(task, spec$method_panel, opls_layout$ncomp)
    fit_ms <- fit$fit_ms
    t1 <- proc.time()[3]
    pred <- pls_pkg_predict(fit, task, opls_layout$ncomp)
    predict_ms <- (proc.time()[3] - t1) * 1000
    metric <- metric_from_prediction(task, pred)
    mem <- stop_memory_monitor(mon)
    mon <- NULL
    return(c(
      list(status = "ok", msg = "", fit_time_ms = as.numeric(fit_ms), predict_time_ms = as.numeric(predict_ms), total_time_ms = as.numeric(fit_ms + predict_ms), model_size_mb = as.numeric(object.size(fit$model)) / (1024^2)),
      metric,
      mem
    ))
  }

  fit_call <- function() {
    if (identical(spec$engine, "GPU")) {
      if (identical(spec$method_panel, "plssvd")) {
        return(fastPLS::plssvd_gpu(task$Xtrain, task$Ytrain, ncomp = as.integer(ncomp_run), fit = FALSE, seed = as.integer(seed)))
      }
      if (identical(spec$method_panel, "simpls")) {
        return(fastPLS::simpls_gpu(task$Xtrain, task$Ytrain, ncomp = as.integer(ncomp_run), fit = FALSE, seed = as.integer(seed)))
      }
      if (identical(spec$method_panel, "opls")) {
        return(fastPLS::opls_cuda(task$Xtrain, task$Ytrain, ncomp = opls_layout$ncomp, north = opls_layout$north, method = "simpls", fit = FALSE, seed = as.integer(seed)))
      }
      if (identical(spec$method_panel, "kernelpls")) {
        return(fastPLS::kernel_pls_cuda(task$Xtrain, task$Ytrain, ncomp = as.integer(ncomp_run), kernel = "linear", method = "simpls", fit = FALSE, seed = as.integer(seed)))
      }
    }

    fn <- if (identical(spec$implementation, "R")) fastPLS::pls_r else fastPLS::pls
    svd_method <- if (identical(spec$backend_algorithm, "rsvd")) "cpu_rsvd" else "irlba"
    if (identical(spec$method_panel, "plssvd") || identical(spec$method_panel, "simpls")) {
      return(fn(task$Xtrain, task$Ytrain, ncomp = as.integer(ncomp_run), method = spec$method_panel, svd.method = svd_method, fit = FALSE, seed = as.integer(seed)))
    }
    if (identical(spec$method_panel, "opls")) {
      ofn <- if (identical(spec$implementation, "R")) fastPLS::opls_r else fastPLS::opls_cpp
      return(ofn(task$Xtrain, task$Ytrain, ncomp = opls_layout$ncomp, north = opls_layout$north, method = "simpls", svd.method = svd_method, fit = FALSE, seed = as.integer(seed)))
    }
    if (identical(spec$method_panel, "kernelpls")) {
      kfn <- if (identical(spec$implementation, "R")) fastPLS::kernel_pls_r else fastPLS::kernel_pls_cpp
      return(kfn(task$Xtrain, task$Ytrain, ncomp = as.integer(ncomp_run), kernel = "linear", method = "simpls", svd.method = svd_method, fit = FALSE, seed = as.integer(seed)))
    }
    stop("Unsupported method panel: ", spec$method_panel)
  }

  t0 <- proc.time()[3]
  model <- fit_call()
  fit_ms <- (proc.time()[3] - t0) * 1000
  t1 <- proc.time()[3]
  pred <- predict(model, newdata = task$Xtest, Ytest = NULL, proj = FALSE)
  predict_ms <- (proc.time()[3] - t1) * 1000
  metric <- metric_from_prediction(task, pred)
  mem <- stop_memory_monitor(mon)
  mon <- NULL

  c(
    list(status = "ok", msg = "", fit_time_ms = as.numeric(fit_ms), predict_time_ms = as.numeric(predict_ms), total_time_ms = as.numeric(fit_ms + predict_ms), model_size_mb = as.numeric(object.size(model)) / (1024^2)),
    metric,
    mem
  )
}

run_with_timeout <- function(task, spec, ncomp_run, seed, timeout_sec) {
  if (is.finite(max_host_rss_mb)) {
    gc(verbose = FALSE)
    parent_rss <- proc_rss_mb()
    if (is.finite(parent_rss) && parent_rss > max_host_rss_mb) {
      return(list(
        status = "skipped_host_rss_cap",
        msg = sprintf("Parent RSS %.0f MB exceeds FASTPLS_SYNTH_VAR_MAX_HOST_RSS_MB=%.0f MB", parent_rss, max_host_rss_mb),
        peak_host_rss_mb = parent_rss
      ))
    }
  }

  if (.Platform$OS.type == "windows" || identical(spec$engine, "GPU")) {
    return(tryCatch(
      fit_predict_variant(task, spec, ncomp_run, seed),
      error = function(e) list(status = "error", msg = conditionMessage(e))
    ))
  }

  job <- parallel::mcparallel(
    tryCatch(
      fit_predict_variant(task, spec, ncomp_run, seed),
      error = function(e) list(status = "error", msg = conditionMessage(e))
    ),
    silent = TRUE
  )
  deadline <- Sys.time() + timeout_sec
  last_rss_check <- Sys.time()
  repeat {
    ans <- parallel::mccollect(job, wait = FALSE)
    if (!is.null(ans) && length(ans)) {
      return(ans[[1L]])
    }
    if (is.finite(max_host_rss_mb) && as.numeric(difftime(Sys.time(), last_rss_check, units = "secs")) >= 0.5) {
      last_rss_check <- Sys.time()
      rss_mb <- proc_tree_rss_mb(job$pid)
      if (is.finite(rss_mb) && rss_mb > max_host_rss_mb) {
        if (!is.null(job$pid) && is.finite(job$pid)) {
          try(system2("pkill", c("-TERM", "-P", as.character(job$pid)), stdout = FALSE, stderr = FALSE), silent = TRUE)
        }
        try(parallel::mckill(job, signal = 15L), silent = TRUE)
        Sys.sleep(1)
        if (!is.null(job$pid) && is.finite(job$pid)) {
          try(system2("pkill", c("-KILL", "-P", as.character(job$pid)), stdout = FALSE, stderr = FALSE), silent = TRUE)
        }
        try(parallel::mckill(job, signal = 9L), silent = TRUE)
        try(parallel::mccollect(job, wait = FALSE), silent = TRUE)
        return(list(
          status = "killed_host_rss_cap",
          msg = sprintf("Peak process-tree RSS %.0f MB exceeded FASTPLS_SYNTH_VAR_MAX_HOST_RSS_MB=%.0f MB", rss_mb, max_host_rss_mb),
          peak_host_rss_mb = rss_mb
        ))
      }
    }
    if (Sys.time() >= deadline) {
      if (!is.null(job$pid) && is.finite(job$pid)) {
        try(system2("pkill", c("-TERM", "-P", as.character(job$pid)), stdout = FALSE, stderr = FALSE), silent = TRUE)
      }
      try(parallel::mckill(job, signal = 15L), silent = TRUE)
      Sys.sleep(1)
      if (!is.null(job$pid) && is.finite(job$pid)) {
        try(system2("pkill", c("-KILL", "-P", as.character(job$pid)), stdout = FALSE, stderr = FALSE), silent = TRUE)
      }
      try(parallel::mckill(job, signal = 9L), silent = TRUE)
      try(parallel::mccollect(job, wait = FALSE), silent = TRUE)
      return(list(
        status = "killed_timeout",
        msg = sprintf("Exceeded FASTPLS_SYNTH_VAR_TIMEOUT_SEC=%.0f seconds", timeout_sec)
      ))
    }
    Sys.sleep(0.2)
  }
}

if (file.exists(raw_file)) unlink(raw_file)
if (file.exists(progress_file)) unlink(progress_file)

log_msg("Synthetic variable-sweep benchmark started")
log_msg("out_dir=", out_dir)
log_msg("families=", paste(families, collapse = ","))
log_msg("reps=", reps, " ncomp=", ncomp, " timeout_sec=", timeout_sec, " max_host_rss_mb=", if (is.finite(max_host_rss_mb)) max_host_rss_mb else "Inf", " include_gpu=", include_gpu, " cuda_ok=", cuda_ok, " include_r=", include_r, " include_pls_pkg=", include_pls_pkg)

all_rows <- list()
row_idx <- 0L
timed_out_keys <- new.env(parent = emptyenv())
append_result <- function(row) {
  row_idx <<- row_idx + 1L
  all_rows[[row_idx]] <<- row
  fwrite(row, raw_file, append = file.exists(raw_file), col.names = !file.exists(raw_file))
}

for (family in families) {
  meta <- family_meta[[family]]
  values <- grids[[family]]
  for (x_value in values) {
    for (rep_id in seq_len(reps)) {
      seed <- as.integer(base_seed + match(family, names(family_meta)) * 100000L + rep_id * 1000L + round(as.numeric(x_value) * 10))
      task <- make_task(family, x_value, seed)
      ncomp_eff <- min(as.integer(ncomp), task$n_train - 1L, task$p, task$n_classes)
      specs <- variant_specs(task)
      log_msg("[RUN] family=", family, " ", meta$variable, "=", x_value, " rep=", rep_id, " methods=", nrow(specs), " ncomp_eff=", ncomp_eff)
      for (i in seq_len(nrow(specs))) {
        spec <- specs[i]
        timeout_key <- paste(family, spec$variant_name, sep = "::")
        if (isTRUE(timed_out_keys[[timeout_key]])) {
          result <- list(
            status = "skipped_after_timeout",
            msg = sprintf("Skipped after earlier timeout for %s in %s", spec$variant_name, family)
          )
        } else {
          result <- run_with_timeout(task, spec, ncomp_eff, seed + i, timeout_sec)
          if (result$status %in% c("killed_timeout", "killed_host_rss_cap", "skipped_host_rss_cap")) {
            timed_out_keys[[timeout_key]] <- TRUE
          }
        }
        status <- result$status %||% "error"
        row <- data.table(
          family = family,
          task_type = task$task_type,
          swept_variable = meta$variable,
          x_value = as.numeric(x_value),
          x_label = meta$x_label,
          replicate = as.integer(rep_id),
          variant_name = spec$variant_name,
          method_panel = spec$method_panel,
          engine = spec$engine,
          implementation = spec$implementation,
          backend_algorithm = spec$backend_algorithm,
          requested_ncomp = as.integer(ncomp),
          effective_ncomp = as.integer(ncomp_eff),
          n_train = as.integer(task$n_train),
          n_test = as.integer(task$n_test),
          p = as.integer(task$p),
          q = as.integer(task$q),
          n_classes = as.integer(task$n_classes),
          noise = as.numeric(task$noise),
          rank_true = as.integer(task$rank_true),
          fit_time_ms = as.numeric(result$fit_time_ms %||% NA_real_),
          predict_time_ms = as.numeric(result$predict_time_ms %||% NA_real_),
          total_time_ms = as.numeric(result$total_time_ms %||% NA_real_),
          metric_name = as.character(result$metric_name %||% if (identical(task$task_type, "classification")) "accuracy" else if (task$n_classes == 1L) "Q2" else "RMSD"),
          metric_value = as.numeric(result$metric_value %||% NA_real_),
          accuracy = as.numeric(result$accuracy %||% NA_real_),
          q2 = as.numeric(result$q2 %||% NA_real_),
          rmsd = as.numeric(result$rmsd %||% NA_real_),
          model_size_mb = as.numeric(result$model_size_mb %||% NA_real_),
          peak_host_rss_mb = as.numeric(result$peak_host_rss_mb %||% NA_real_),
          host_rss_delta_mb = as.numeric(result$host_rss_delta_mb %||% NA_real_),
          peak_gpu_mem_mb = as.numeric(result$peak_gpu_mem_mb %||% NA_real_),
          gpu_mem_delta_mb = as.numeric(result$gpu_mem_delta_mb %||% NA_real_),
          status = status,
          msg = as.character(result$msg %||% "")
        )
        append_result(row)
      }
    }
      rm(task)
      gc(verbose = FALSE)
  }
}

raw <- rbindlist(all_rows, fill = TRUE)
summary_dt <- raw[status == "ok", .(
  fit_time_ms_median = median(fit_time_ms, na.rm = TRUE),
  predict_time_ms_median = median(predict_time_ms, na.rm = TRUE),
  total_time_ms_median = median(total_time_ms, na.rm = TRUE),
  metric_value_median = median(metric_value, na.rm = TRUE),
  accuracy_median = median(accuracy, na.rm = TRUE),
  q2_median = median(q2, na.rm = TRUE),
  rmsd_median = median(rmsd, na.rm = TRUE),
  peak_host_rss_mb_median = median(peak_host_rss_mb, na.rm = TRUE),
  peak_gpu_mem_mb_median = median(peak_gpu_mem_mb, na.rm = TRUE),
  n_ok = .N
), by = .(
  family, task_type, swept_variable, x_value, x_label,
  variant_name, method_panel, engine, implementation, backend_algorithm,
  requested_ncomp, effective_ncomp, n_train, n_test, p, q, n_classes, noise, rank_true, metric_name
)]
fwrite(summary_dt, summary_file)

writeLines(c(
  "Synthetic variable-sweep benchmark",
  sprintf("out_dir = %s", normalizePath(out_dir, winslash = "/", mustWork = FALSE)),
  sprintf("families = %s", paste(families, collapse = ",")),
  sprintf("reps = %d", reps),
  sprintf("requested_ncomp = %d", ncomp),
  sprintf("timeout_sec = %.0f", timeout_sec),
  sprintf("max_host_rss_mb = %s", if (is.finite(max_host_rss_mb)) sprintf("%.0f", max_host_rss_mb) else "Inf"),
  sprintf("cuda_available = %s", cuda_ok),
  sprintf("include_gpu = %s", include_gpu),
  sprintf("include_r = %s", include_r),
  sprintf("include_pls_pkg = %s", include_pls_pkg),
  sprintf("rows = %d", nrow(raw)),
  sprintf("ok_rows = %d", nrow(raw[status == "ok"]))
), manifest_file)

log_msg("Synthetic variable-sweep benchmark completed")
log_msg("raw=", raw_file)
log_msg("summary=", summary_file)
