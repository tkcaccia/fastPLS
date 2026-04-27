#!/usr/bin/env Rscript

suppressPackageStartupMessages({
  library(fastPLS)
  library(KODAMA)
  library(bench)
  library(data.table)
})

set.seed(123)

ncomp_grid <- c(2:30, 35, 40, 45, 50, 60, 70, 80, 90, 100)
reps <- as.integer(Sys.getenv("FASTPLS_BENCH_REPS", "3"))
if (!is.finite(reps) || is.na(reps) || reps < 1L) reps <- 3L
fast_threshold_ms <- as.numeric(Sys.getenv("FASTPLS_FAST_THRESHOLD_MS", "25"))
fast_reps_min <- as.integer(Sys.getenv("FASTPLS_FAST_REPS_MIN", "10"))
if (!is.finite(fast_threshold_ms) || is.na(fast_threshold_ms) || fast_threshold_ms <= 0) fast_threshold_ms <- 25
if (!is.finite(fast_reps_min) || is.na(fast_reps_min) || fast_reps_min < reps) fast_reps_min <- max(10L, reps)

out_csv <- Sys.getenv("FASTPLS_METREF_OUT", "metref_benchmark_results_ncomp_custom.csv")

# MetRef data prep
# Same pipeline used in previous scripts.
data(MetRef, package = "KODAMA")
u <- MetRef$data
u <- u[, -which(colSums(u) == 0), drop = FALSE]
u <- normalization(u)$newXtrain
class <- as.numeric(as.factor(MetRef$donor))

ss <- sample(nrow(u), 100)
Xtrain <- u[-ss, , drop = FALSE]
Ytrain <- as.factor(class)[-ss]
Xtest <- u[ss, , drop = FALSE]
Ytest <- as.factor(class)[ss]

plssvd_cap <- min(nrow(Xtrain), ncol(Xtrain), nlevels(Ytrain))

methods <- rbindlist(list(
  data.table(
    engine = "Rcpp",
    algorithm = rep(c("simpls", "plssvd"), each = 5),
    svd_method = rep(c("irlba", "cpu_exact", "cpu_rsvd", "cuda_rsvd"), 2)
  ),
  data.table(
    engine = "R",
    algorithm = rep(c("simpls", "plssvd"), each = 4),
    svd_method = rep(c("irlba", "cpu_exact", "cpu_rsvd"), 2)
  )
))
methods[, config := paste(engine, algorithm, svd_method, sep = "_")]

cuda_ok <- tryCatch(isTRUE(has_cuda()), error = function(e) FALSE)
has_nvidia_smi <- nzchar(Sys.which("nvidia-smi"))

gpu_mem_used_mb <- function() {
  if (!has_nvidia_smi) return(NA_real_)
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
  if (!has_nvidia_smi) return(NA_real_)
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
  if (length(dim(yp)) == 3) {
    mat <- yp[, , 1, drop = FALSE]
    cls <- apply(mat, 1, which.max)
    lev <- model_out$lev
    return(factor(lev[cls], levels = lev))
  }
  stop("Unsupported Ypred format")
}

fit_once <- function(engine, algorithm, svd_method, ncomp) {
  if (engine == "Rcpp") {
    pls(Xtrain, Ytrain, Xtest, ncomp = ncomp, method = algorithm, svd.method = svd_method, scaling = "centering")
  } else {
    pls_r(Xtrain, Ytrain, Xtest, ncomp = ncomp, method = algorithm, svd.method = svd_method, scaling = "centering")
  }
}

rows <- vector("list", nrow(methods) * length(ncomp_grid))
k <- 1L

for (i in seq_len(nrow(methods))) {
  cfg <- methods[i]
  message(sprintf("Running %s (%d/%d)", cfg$config, i, nrow(methods)))

  for (nc in ncomp_grid) {
    if (cfg$algorithm == "plssvd" && nc > plssvd_cap) {
      rows[[k]] <- data.table(
        config = cfg$config,
        engine = cfg$engine,
        algorithm = cfg$algorithm,
        svd_method = cfg$svd_method,
        ncomp = nc,
        plssvd_cap = plssvd_cap,
        reps = reps,
        reps_used = reps,
        status = "skipped_ncomp_above_plssvd_cap",
        time_ms_median_reps = NA_real_,
        time_ms_mean_reps = NA_real_,
        bench_median_ms = NA_real_,
        bench_itr_sec = NA_real_,
        ram_alloc_mb = NA_real_,
        gpu_mem_before_mb_median = NA_real_,
        gpu_mem_after_mb_median = NA_real_,
        gpu_mem_delta_mb_median = NA_real_,
        gpu_compute_mem_mb_max = NA_real_,
        acc_median_reps = NA_real_,
        acc_mean_reps = NA_real_,
        rep_times_ms = "",
        rep_acc = ""
      )
      k <- k + 1L
      next
    }

    if (cfg$svd_method == "cuda_rsvd" && !cuda_ok) {
      rows[[k]] <- data.table(
        config = cfg$config,
        engine = cfg$engine,
        algorithm = cfg$algorithm,
        svd_method = cfg$svd_method,
        ncomp = nc,
        plssvd_cap = plssvd_cap,
        reps = reps,
        reps_used = reps,
        status = "skipped_cuda_unavailable",
        time_ms_median_reps = NA_real_,
        time_ms_mean_reps = NA_real_,
        bench_median_ms = NA_real_,
        bench_itr_sec = NA_real_,
        ram_alloc_mb = NA_real_,
        gpu_mem_before_mb_median = NA_real_,
        gpu_mem_after_mb_median = NA_real_,
        gpu_mem_delta_mb_median = NA_real_,
        gpu_compute_mem_mb_max = NA_real_,
        acc_median_reps = NA_real_,
        acc_mean_reps = NA_real_,
        rep_times_ms = "",
        rep_acc = ""
      )
      k <- k + 1L
      next
    }

    # Pilot run: increase repetitions for very fast configurations.
    gc(FALSE)
    pilot_before <- if (cfg$svd_method == "cuda_rsvd") gpu_mem_used_mb() else NA_real_
    pilot_t0 <- proc.time()[3]
    pilot_fit <- tryCatch(fit_once(cfg$engine, cfg$algorithm, cfg$svd_method, nc), error = function(e) e)
    pilot_elapsed_ms <- (proc.time()[3] - pilot_t0) * 1000
    pilot_after <- if (cfg$svd_method == "cuda_rsvd") gpu_mem_used_mb() else NA_real_
    pilot_delta <- pilot_after - pilot_before
    pilot_compute_mb <- if (cfg$svd_method == "cuda_rsvd") gpu_compute_mem_mb() else NA_real_

    reps_used <- if (is.finite(pilot_elapsed_ms) && pilot_elapsed_ms < fast_threshold_ms) max(reps, fast_reps_min) else reps

    elapsed_ms <- rep(NA_real_, reps_used)
    rep_acc <- rep(NA_real_, reps_used)
    gpu_before <- rep(NA_real_, reps_used)
    gpu_after <- rep(NA_real_, reps_used)
    gpu_delta <- rep(NA_real_, reps_used)
    gpu_compute <- rep(NA_real_, reps_used)
    err <- NULL

    if (inherits(pilot_fit, "error")) {
      err <- conditionMessage(pilot_fit)
    } else {
      pilot_pred <- extract_pred(pilot_fit)
      elapsed_ms[1] <- pilot_elapsed_ms
      rep_acc[1] <- mean(pilot_pred == Ytest)
      gpu_before[1] <- pilot_before
      gpu_after[1] <- pilot_after
      gpu_delta[1] <- pilot_delta
      gpu_compute[1] <- pilot_compute_mb
    }

    for (r in seq(from = 2L, to = reps_used)) {
      gc(FALSE)
      if (cfg$svd_method == "cuda_rsvd") gpu_before[r] <- gpu_mem_used_mb()
      t0 <- proc.time()[3]
      fit <- tryCatch(fit_once(cfg$engine, cfg$algorithm, cfg$svd_method, nc), error = function(e) e)
      elapsed_ms[r] <- (proc.time()[3] - t0) * 1000
      if (cfg$svd_method == "cuda_rsvd") gpu_after[r] <- gpu_mem_used_mb()
      gpu_delta[r] <- gpu_after[r] - gpu_before[r]
      if (cfg$svd_method == "cuda_rsvd") gpu_compute[r] <- gpu_compute_mem_mb()

      if (inherits(fit, "error")) {
        err <- conditionMessage(fit)
        break
      }

      pred <- extract_pred(fit)
      rep_acc[r] <- mean(pred == Ytest)
    }

    if (!is.null(err)) {
      rows[[k]] <- data.table(
        config = cfg$config,
        engine = cfg$engine,
        algorithm = cfg$algorithm,
        svd_method = cfg$svd_method,
        ncomp = nc,
        plssvd_cap = plssvd_cap,
        reps = reps,
        reps_used = reps_used,
        status = paste0("error: ", err),
        time_ms_median_reps = NA_real_,
        time_ms_mean_reps = NA_real_,
        bench_median_ms = NA_real_,
        bench_itr_sec = NA_real_,
        ram_alloc_mb = NA_real_,
        gpu_mem_before_mb_median = if (cfg$svd_method == "cuda_rsvd") median(gpu_before, na.rm = TRUE) else NA_real_,
        gpu_mem_after_mb_median = if (cfg$svd_method == "cuda_rsvd") median(gpu_after, na.rm = TRUE) else NA_real_,
        gpu_mem_delta_mb_median = if (cfg$svd_method == "cuda_rsvd") median(gpu_delta, na.rm = TRUE) else NA_real_,
        gpu_compute_mem_mb_max = if (cfg$svd_method == "cuda_rsvd") max(gpu_compute, na.rm = TRUE) else NA_real_,
        acc_median_reps = NA_real_,
        acc_mean_reps = NA_real_,
        rep_times_ms = paste(sprintf("%.6f", elapsed_ms[is.finite(elapsed_ms)]), collapse = ";"),
        rep_acc = paste(sprintf("%.6f", rep_acc[is.finite(rep_acc)]), collapse = ";")
      )
      k <- k + 1L
      next
    }

    b <- bench::mark(
      {
        o <- fit_once(cfg$engine, cfg$algorithm, cfg$svd_method, nc)
        invisible(o$Ypred)
      },
      iterations = reps_used,
      check = FALSE,
      memory = TRUE,
      time_unit = "ms"
    )

    rows[[k]] <- data.table(
      config = cfg$config,
      engine = cfg$engine,
      algorithm = cfg$algorithm,
      svd_method = cfg$svd_method,
      ncomp = nc,
      plssvd_cap = plssvd_cap,
      reps = reps,
      reps_used = reps_used,
      status = "ok",
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
      rep_times_ms = paste(sprintf("%.6f", elapsed_ms), collapse = ";"),
      rep_acc = paste(sprintf("%.6f", rep_acc), collapse = ";")
    )
    k <- k + 1L
  }
}

res <- rbindlist(rows, fill = TRUE)
setorder(res, config, ncomp)
fwrite(res, out_csv)

cat("Done\n")
cat("Output:", normalizePath(out_csv), "\n")
cat("Rows:", nrow(res), "\n")
cat("plssvd cap:", plssvd_cap, "\n")
cat("CUDA available:", cuda_ok, "\n")
