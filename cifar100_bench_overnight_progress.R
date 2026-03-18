#!/usr/bin/env Rscript

suppressPackageStartupMessages({
  library(fastPLS)
  library(bench)
})

set.seed(123)

# User-requested source dataset file containing object `r`
data_file <- Sys.getenv("FASTPLS_CIFAR100_RDATA", "~/Documents/Rdatasets/CIFAR100.RData")
out_csv <- Sys.getenv("FASTPLS_CIFAR100_OUT", "~/gpupls_test/GPUPLS/cifar100_benchmark_overnight_progress.csv")
out_log <- Sys.getenv("FASTPLS_CIFAR100_LOG", "~/gpupls_test/GPUPLS/cifar100_benchmark_overnight_progress.log")
reps <- as.integer(Sys.getenv("FASTPLS_BENCH_REPS", "3"))
if (!is.finite(reps) || is.na(reps) || reps < 1L) reps <- 3L

# Requested component grid (user-specified)
ncomp_env <- Sys.getenv("FASTPLS_NCOMP_LIST", "2,5,10,20,50,100,200,500")
ncomp_grid_common <- as.integer(strsplit(ncomp_env, ",", fixed = TRUE)[[1]])
ncomp_grid_common <- sort(unique(ncomp_grid_common[is.finite(ncomp_grid_common) & ncomp_grid_common >= 1L]))
if (!length(ncomp_grid_common)) ncomp_grid_common <- c(2L, 5L, 10L, 20L, 50L, 100L, 200L, 500L)

stamp <- function() format(Sys.time(), "%Y-%m-%d %H:%M:%S")
log_msg <- function(...) {
  msg <- paste0("[", stamp(), "] ", paste(..., collapse = ""))
  cat(msg, "\n")
  cat(msg, "\n", file = path.expand(out_log), append = TRUE)
  flush.console()
}

gpu_mem_used_mb <- function() {
  if (!nzchar(Sys.which("nvidia-smi"))) return(NA_real_)
  out <- tryCatch(
    system2("nvidia-smi", c("--query-gpu=memory.used", "--format=csv,noheader,nounits"), stdout = TRUE, stderr = FALSE),
    error = function(e) character(0)
  )
  if (!length(out)) return(NA_real_)
  vals <- suppressWarnings(as.numeric(trimws(out)))
  vals <- vals[is.finite(vals)]
  if (!length(vals)) return(NA_real_)
  max(vals)
}

gpu_compute_mem_mb <- function() {
  if (!nzchar(Sys.which("nvidia-smi"))) return(NA_real_)
  out <- tryCatch(
    system2("nvidia-smi", c("--query-compute-apps=used_gpu_memory", "--format=csv,noheader,nounits"), stdout = TRUE, stderr = FALSE),
    error = function(e) character(0)
  )
  if (!length(out)) return(NA_real_)
  vals <- suppressWarnings(as.numeric(trimws(out)))
  vals <- vals[is.finite(vals)]
  if (!length(vals)) return(0)
  sum(vals)
}

extract_pred <- function(model_out) {
  yp <- model_out$Ypred
  if (is.data.frame(yp)) return(as.factor(yp[[1]]))
  if (is.matrix(yp)) return(as.factor(yp[, 1]))
  if (is.vector(yp)) return(as.factor(yp))
  if (length(dim(yp)) == 3L) {
    mat <- yp[, , 1, drop = FALSE]
    cls <- apply(mat, 1, which.max)
    lev <- model_out$lev
    return(factor(lev[cls], levels = lev))
  }
  stop("Unsupported Ypred format")
}

fit_once <- function(engine, algorithm, svd_method, ncomp, Xtrain, Ytrain, Xtest) {
  if (engine == "Rcpp") {
    pls(Xtrain, Ytrain, Xtest, ncomp = ncomp, method = algorithm, svd.method = svd_method, scaling = "centering")
  } else {
    pls_r(Xtrain, Ytrain, Xtest, ncomp = ncomp, method = algorithm, svd.method = svd_method, scaling = "centering")
  }
}

# Load CIFAR100 object `r`
load(path.expand(data_file))
if (!exists("r")) stop("Object `r` not found in CIFAR100 RData")
if (!("label_idx" %in% colnames(r))) stop("Column `label_idx` not found in object `r`")

# User-provided split logic
# data=r[,-c(1:3)]
# labels=as.factor(r[,"label_idx"])
# ss=sample(nrow(data),round(nrow(data)/2))
# Xtrain=data[ss,]; Ytrain=labels[ss]
# Xtest=data[-ss,]; Ytest=labels[-ss]
data <- r[, -c(1:3)]
labels <- as.factor(r[, "label_idx"])
# CIFAR table may carry non-numeric columns even after dropping first metadata fields.
# Force numeric predictors and drop columns that remain entirely non-numeric.
data_num <- as.data.frame(lapply(data, function(x) suppressWarnings(as.numeric(as.character(x)))))
keep <- vapply(data_num, function(x) any(is.finite(x)), logical(1))
data_num <- data_num[, keep, drop = FALSE]
data <- as.matrix(data_num)
ss <- sample(nrow(data), round(nrow(data) / 2))
Xtrain <- data[ss, , drop = FALSE]
Ytrain <- labels[ss]
Xtest <- data[-ss, , drop = FALSE]
Ytest <- labels[-ss]

plssvd_cap <- min(nrow(Xtrain), ncol(Xtrain), nlevels(Ytrain))
cuda_ok <- tryCatch(isTRUE(has_cuda()), error = function(e) FALSE)

methods <- rbind(
  data.frame(
    engine = "Rcpp",
    algorithm = rep(c("simpls", "plssvd"), each = 4),
    svd_method = rep(c("irlba", "dc", "cpu_rsvd", "cuda_rsvd"), 2),
    stringsAsFactors = FALSE
  ),
  data.frame(
    engine = "R",
    algorithm = rep(c("simpls", "plssvd"), each = 3),
    svd_method = rep(c("irlba", "dc", "cpu_rsvd"), 2),
    stringsAsFactors = FALSE
  )
)
methods$config <- paste(methods$engine, methods$algorithm, methods$svd_method, sep = "_")

# Build row plan on the requested ncomp grid
rows_plan <- list()
for (i in seq_len(nrow(methods))) {
  cfg <- methods[i, ]
  grid <- ncomp_grid_common
  for (nc in grid) {
    rows_plan[[length(rows_plan) + 1L]] <- data.frame(
      config = cfg$config,
      engine = cfg$engine,
      algorithm = cfg$algorithm,
      svd_method = cfg$svd_method,
      ncomp = as.integer(nc),
      stringsAsFactors = FALSE
    )
  }
}
plan <- do.call(rbind, rows_plan)

out_csv <- path.expand(out_csv)
out_log <- path.expand(out_log)
dir.create(dirname(out_csv), recursive = TRUE, showWarnings = FALSE)
cat("", file = out_log, append = FALSE)

header <- data.frame(
  config = character(),
  engine = character(),
  algorithm = character(),
  svd_method = character(),
  ncomp = integer(),
  plssvd_cap = integer(),
  reps = integer(),
  status = character(),
  time_ms_median_reps = numeric(),
  time_ms_mean_reps = numeric(),
  bench_median_ms = numeric(),
  bench_itr_sec = numeric(),
  ram_alloc_mb = numeric(),
  gpu_mem_before_mb_median = numeric(),
  gpu_mem_after_mb_median = numeric(),
  gpu_mem_delta_mb_median = numeric(),
  gpu_compute_mem_mb_max = numeric(),
  acc_median_reps = numeric(),
  acc_mean_reps = numeric(),
  msg = character(),
  rep_times_ms = character(),
  rep_acc = character(),
  started_at = character(),
  finished_at = character(),
  stringsAsFactors = FALSE
)
write.csv(header, out_csv, row.names = FALSE, quote = TRUE)

log_msg("CIFAR100 overnight benchmark started")
log_msg("Data file: ", path.expand(data_file))
log_msg("Rows train/test: ", nrow(Xtrain), "/", nrow(Xtest), "; features: ", ncol(Xtrain), "; classes: ", nlevels(Ytrain))
log_msg("Repetitions: ", reps, "; plssvd cap: ", plssvd_cap, "; CUDA: ", cuda_ok)
log_msg("ncomp grid: ", paste(ncomp_grid_common, collapse = ","))
log_msg("Total runs planned: ", nrow(plan))

for (i in seq_len(nrow(plan))) {
  cfg <- plan[i, ]
  started_at <- stamp()
  log_msg(sprintf("[%d/%d] START %s ncomp=%d", i, nrow(plan), cfg$config, cfg$ncomp))

  if (cfg$algorithm == "plssvd" && cfg$ncomp > plssvd_cap) {
    row <- data.frame(
      config = cfg$config, engine = cfg$engine, algorithm = cfg$algorithm, svd_method = cfg$svd_method,
      ncomp = cfg$ncomp, plssvd_cap = plssvd_cap, reps = reps, status = "skipped_ncomp_above_plssvd_cap",
      time_ms_median_reps = NA_real_, time_ms_mean_reps = NA_real_, bench_median_ms = NA_real_, bench_itr_sec = NA_real_,
      ram_alloc_mb = NA_real_, gpu_mem_before_mb_median = NA_real_, gpu_mem_after_mb_median = NA_real_,
      gpu_mem_delta_mb_median = NA_real_, gpu_compute_mem_mb_max = NA_real_,
      acc_median_reps = NA_real_, acc_mean_reps = NA_real_, msg = "", rep_times_ms = "", rep_acc = "",
      started_at = started_at, finished_at = stamp(), stringsAsFactors = FALSE
    )
    write.table(row, file = out_csv, sep = ",", row.names = FALSE, col.names = FALSE, append = TRUE, qmethod = "double")
    log_msg(sprintf("[%d/%d] SKIP %s ncomp=%d (above plssvd cap)", i, nrow(plan), cfg$config, cfg$ncomp))
    next
  }

  if (cfg$svd_method == "cuda_rsvd" && !cuda_ok) {
    row <- data.frame(
      config = cfg$config, engine = cfg$engine, algorithm = cfg$algorithm, svd_method = cfg$svd_method,
      ncomp = cfg$ncomp, plssvd_cap = plssvd_cap, reps = reps, status = "skipped_cuda_unavailable",
      time_ms_median_reps = NA_real_, time_ms_mean_reps = NA_real_, bench_median_ms = NA_real_, bench_itr_sec = NA_real_,
      ram_alloc_mb = NA_real_, gpu_mem_before_mb_median = NA_real_, gpu_mem_after_mb_median = NA_real_,
      gpu_mem_delta_mb_median = NA_real_, gpu_compute_mem_mb_max = NA_real_,
      acc_median_reps = NA_real_, acc_mean_reps = NA_real_, msg = "", rep_times_ms = "", rep_acc = "",
      started_at = started_at, finished_at = stamp(), stringsAsFactors = FALSE
    )
    write.table(row, file = out_csv, sep = ",", row.names = FALSE, col.names = FALSE, append = TRUE, qmethod = "double")
    log_msg(sprintf("[%d/%d] SKIP %s ncomp=%d (cuda unavailable)", i, nrow(plan), cfg$config, cfg$ncomp))
    next
  }

  elapsed_ms <- rep(NA_real_, reps)
  rep_acc <- rep(NA_real_, reps)
  gpu_before <- rep(NA_real_, reps)
  gpu_after <- rep(NA_real_, reps)
  gpu_delta <- rep(NA_real_, reps)
  gpu_compute <- rep(NA_real_, reps)
  err <- NULL

  for (ridx in seq_len(reps)) {
    gc(FALSE)
    if (cfg$svd_method == "cuda_rsvd") gpu_before[ridx] <- gpu_mem_used_mb()
    t0 <- proc.time()[3]
    fit <- tryCatch(
      fit_once(cfg$engine, cfg$algorithm, cfg$svd_method, cfg$ncomp, Xtrain, Ytrain, Xtest),
      error = function(e) e
    )
    elapsed_ms[ridx] <- (proc.time()[3] - t0) * 1000
    if (cfg$svd_method == "cuda_rsvd") gpu_after[ridx] <- gpu_mem_used_mb()
    gpu_delta[ridx] <- gpu_after[ridx] - gpu_before[ridx]
    if (cfg$svd_method == "cuda_rsvd") gpu_compute[ridx] <- gpu_compute_mem_mb()

    if (inherits(fit, "error")) {
      err <- conditionMessage(fit)
      log_msg(sprintf("[%d/%d] %s ncomp=%d rep %d/%d ERROR: %s", i, nrow(plan), cfg$config, cfg$ncomp, ridx, reps, err))
      break
    }

    pred <- extract_pred(fit)
    rep_acc[ridx] <- mean(pred == Ytest)
    log_msg(sprintf("[%d/%d] %s ncomp=%d rep %d/%d done in %.3fms acc=%.4f", i, nrow(plan), cfg$config, cfg$ncomp, ridx, reps, elapsed_ms[ridx], rep_acc[ridx]))
  }

  if (!is.null(err)) {
    row <- data.frame(
      config = cfg$config, engine = cfg$engine, algorithm = cfg$algorithm, svd_method = cfg$svd_method,
      ncomp = cfg$ncomp, plssvd_cap = plssvd_cap, reps = reps, status = "error",
      time_ms_median_reps = NA_real_, time_ms_mean_reps = NA_real_, bench_median_ms = NA_real_, bench_itr_sec = NA_real_,
      ram_alloc_mb = NA_real_,
      gpu_mem_before_mb_median = if (cfg$svd_method == "cuda_rsvd") median(gpu_before, na.rm = TRUE) else NA_real_,
      gpu_mem_after_mb_median = if (cfg$svd_method == "cuda_rsvd") median(gpu_after, na.rm = TRUE) else NA_real_,
      gpu_mem_delta_mb_median = if (cfg$svd_method == "cuda_rsvd") median(gpu_delta, na.rm = TRUE) else NA_real_,
      gpu_compute_mem_mb_max = if (cfg$svd_method == "cuda_rsvd") max(gpu_compute, na.rm = TRUE) else NA_real_,
      acc_median_reps = NA_real_, acc_mean_reps = NA_real_, msg = gsub('"', "'", err, fixed = TRUE),
      rep_times_ms = paste(sprintf("%.6f", elapsed_ms[is.finite(elapsed_ms)]), collapse = ";"),
      rep_acc = paste(sprintf("%.6f", rep_acc[is.finite(rep_acc)]), collapse = ";"),
      started_at = started_at, finished_at = stamp(), stringsAsFactors = FALSE
    )
    write.table(row, file = out_csv, sep = ",", row.names = FALSE, col.names = FALSE, append = TRUE, qmethod = "double")
    log_msg(sprintf("[%d/%d] END %s ncomp=%d with error", i, nrow(plan), cfg$config, cfg$ncomp))
    next
  }

  b <- bench::mark(
    {
      o <- fit_once(cfg$engine, cfg$algorithm, cfg$svd_method, cfg$ncomp, Xtrain, Ytrain, Xtest)
      invisible(o$Ypred)
    },
    iterations = reps,
    check = FALSE,
    memory = TRUE,
    time_unit = "ms"
  )

  row <- data.frame(
    config = cfg$config, engine = cfg$engine, algorithm = cfg$algorithm, svd_method = cfg$svd_method,
    ncomp = cfg$ncomp, plssvd_cap = plssvd_cap, reps = reps, status = "ok",
    time_ms_median_reps = median(elapsed_ms, na.rm = TRUE),
    time_ms_mean_reps = mean(elapsed_ms, na.rm = TRUE),
    bench_median_ms = as.numeric(b$median),
    bench_itr_sec = as.numeric(b$`itr/sec`),
    ram_alloc_mb = as.numeric(b$mem_alloc) / (1024^2),
    gpu_mem_before_mb_median = if (cfg$svd_method == "cuda_rsvd") median(gpu_before, na.rm = TRUE) else NA_real_,
    gpu_mem_after_mb_median = if (cfg$svd_method == "cuda_rsvd") median(gpu_after, na.rm = TRUE) else NA_real_,
    gpu_mem_delta_mb_median = if (cfg$svd_method == "cuda_rsvd") median(gpu_delta, na.rm = TRUE) else NA_real_,
    gpu_compute_mem_mb_max = if (cfg$svd_method == "cuda_rsvd") max(gpu_compute, na.rm = TRUE) else NA_real_,
    acc_median_reps = median(rep_acc, na.rm = TRUE),
    acc_mean_reps = mean(rep_acc, na.rm = TRUE),
    msg = "",
    rep_times_ms = paste(sprintf("%.6f", elapsed_ms), collapse = ";"),
    rep_acc = paste(sprintf("%.6f", rep_acc), collapse = ";"),
    started_at = started_at,
    finished_at = stamp(),
    stringsAsFactors = FALSE
  )

  write.table(row, file = out_csv, sep = ",", row.names = FALSE, col.names = FALSE, append = TRUE, qmethod = "double")
  log_msg(sprintf("[%d/%d] END %s ncomp=%d median=%.3fms ram=%.2fMB gpu_compute_max=%.2fMB acc=%.4f",
                  i, nrow(plan), cfg$config, cfg$ncomp,
                  row$time_ms_median_reps, row$ram_alloc_mb,
                  ifelse(is.na(row$gpu_compute_mem_mb_max), NA_real_, row$gpu_compute_mem_mb_max), row$acc_median_reps))
}

log_msg("CIFAR100 overnight benchmark completed")
cat("Done\n")
cat("Output:", out_csv, "\n")
cat("Log:", out_log, "\n")
