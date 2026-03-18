#!/usr/bin/env Rscript

suppressPackageStartupMessages({
  library(fastPLS)
})

set.seed(123)

data_path <- Sys.getenv("FASTPLS_NMR_DATA", "~/gpupls_test/GPUPLS/NMR.RData")
out_csv <- Sys.getenv("FASTPLS_NMR_OUT", "~/gpupls_test/GPUPLS/nmr_bench_all_methods_reps3_cap.csv")
ncomp_requested <- as.integer(Sys.getenv("FASTPLS_NCOMP", "50"))
if (!is.finite(ncomp_requested) || is.na(ncomp_requested) || ncomp_requested < 1L) {
  ncomp_requested <- 50L
}
reps <- 3L

load(path.expand(data_path))
Xtrain <- as.matrix(Xtrain)
Ytrain <- as.matrix(Ytrain)
Xtest <- as.matrix(Xtest)
Ytest <- as.matrix(Ytest)

rmsd <- function(pred, truth) sqrt(mean((pred - truth)^2))

get_pred <- function(fit) {
  yp <- fit$Ypred
  if (length(dim(yp)) == 3L) return(yp[, , 1, drop = FALSE][, , 1])
  if (is.matrix(yp)) return(yp[, 1])
  as.numeric(yp)
}

y_cap_dim <- if (is.matrix(Ytrain) || is.data.frame(Ytrain)) ncol(Ytrain) else 1L
plssvd_cap <- min(nrow(Xtrain), ncol(Xtrain), y_cap_dim)

methods <- rbind(
  data.frame(
    engine = "Rcpp",
    algorithm = rep(c("simpls", "plssvd"), each = 5),
    svd_method = rep(c("irlba", "arpack", "cpu_exact", "cpu_rsvd", "cuda_rsvd"), 2),
    stringsAsFactors = FALSE
  ),
  data.frame(
    engine = "R",
    algorithm = rep(c("simpls", "plssvd"), each = 4),
    svd_method = rep(c("irlba", "arpack", "cpu_exact", "cpu_rsvd"), 2),
    stringsAsFactors = FALSE
  )
)
methods$config <- paste(methods$engine, methods$algorithm, methods$svd_method, sep = "_")

cuda_ok <- FALSE
cuda_ok <- tryCatch(isTRUE(has_cuda()), error = function(e) FALSE)

fit_once <- function(engine, algorithm, svd_method, ncomp) {
  if (engine == "Rcpp") {
    pls(
      Xtrain, Ytrain, Xtest,
      ncomp = ncomp,
      method = algorithm,
      svd.method = svd_method
    )
  } else {
    pls_r(
      Xtrain, Ytrain, Xtest,
      ncomp = ncomp,
      method = algorithm,
      svd.method = svd_method
    )
  }
}

rows <- vector("list", nrow(methods))

for (i in seq_len(nrow(methods))) {
  cfg <- methods[i, ]

  if (cfg$algorithm == "plssvd" && ncomp_requested > plssvd_cap) {
    rows[[i]] <- data.frame(
      config = cfg$config,
      engine = cfg$engine,
      algorithm = cfg$algorithm,
      svd_method = cfg$svd_method,
      ncomp_requested = ncomp_requested,
      ncomp_run = NA_integer_,
      plssvd_cap = plssvd_cap,
      reps = reps,
      status = "skipped_ncomp_above_plssvd_cap",
      time_s_median_reps = NA_real_,
      time_s_mean_reps = NA_real_,
      rmsd_median_reps = NA_real_,
      msg = "",
      rep_times_s = "",
      rep_rmsd = "",
      stringsAsFactors = FALSE
    )
    next
  }

  if (cfg$svd_method == "cuda_rsvd" && !cuda_ok) {
    rows[[i]] <- data.frame(
      config = cfg$config,
      engine = cfg$engine,
      algorithm = cfg$algorithm,
      svd_method = cfg$svd_method,
      ncomp_requested = ncomp_requested,
      ncomp_run = NA_integer_,
      plssvd_cap = plssvd_cap,
      reps = reps,
      status = "skipped_cuda_unavailable",
      time_s_median_reps = NA_real_,
      time_s_mean_reps = NA_real_,
      rmsd_median_reps = NA_real_,
      msg = "",
      rep_times_s = "",
      rep_rmsd = "",
      stringsAsFactors = FALSE
    )
    next
  }

  elapsed <- rep(NA_real_, reps)
  rep_rmsd <- rep(NA_real_, reps)
  err <- NULL

  for (r in seq_len(reps)) {
    gc(FALSE)
    t0 <- proc.time()[3]
    fit <- tryCatch(
      fit_once(cfg$engine, cfg$algorithm, cfg$svd_method, ncomp_requested),
      error = function(e) e
    )
    elapsed[r] <- as.numeric(proc.time()[3] - t0)

    if (inherits(fit, "error")) {
      err <- conditionMessage(fit)
      break
    }
    rep_rmsd[r] <- rmsd(get_pred(fit), Ytest)
  }

  if (!is.null(err)) {
    rows[[i]] <- data.frame(
      config = cfg$config,
      engine = cfg$engine,
      algorithm = cfg$algorithm,
      svd_method = cfg$svd_method,
      ncomp_requested = ncomp_requested,
      ncomp_run = NA_integer_,
      plssvd_cap = plssvd_cap,
      reps = reps,
      status = "error",
      time_s_median_reps = NA_real_,
      time_s_mean_reps = NA_real_,
      rmsd_median_reps = NA_real_,
      msg = gsub("\"", "'", err, fixed = TRUE),
      rep_times_s = paste(sprintf("%.6f", elapsed[is.finite(elapsed)]), collapse = ";"),
      rep_rmsd = paste(sprintf("%.10f", rep_rmsd[is.finite(rep_rmsd)]), collapse = ";"),
      stringsAsFactors = FALSE
    )
  } else {
    rows[[i]] <- data.frame(
      config = cfg$config,
      engine = cfg$engine,
      algorithm = cfg$algorithm,
      svd_method = cfg$svd_method,
      ncomp_requested = ncomp_requested,
      ncomp_run = ncomp_requested,
      plssvd_cap = plssvd_cap,
      reps = reps,
      status = "ok",
      time_s_median_reps = median(elapsed, na.rm = TRUE),
      time_s_mean_reps = mean(elapsed, na.rm = TRUE),
      rmsd_median_reps = median(rep_rmsd, na.rm = TRUE),
      msg = "",
      rep_times_s = paste(sprintf("%.6f", elapsed), collapse = ";"),
      rep_rmsd = paste(sprintf("%.10f", rep_rmsd), collapse = ";"),
      stringsAsFactors = FALSE
    )
  }
}

res <- do.call(rbind, rows)
dir.create(dirname(path.expand(out_csv)), recursive = TRUE, showWarnings = FALSE)
write.csv(res, path.expand(out_csv), row.names = FALSE, quote = TRUE)

cat("Done\n")
cat("Output:", path.expand(out_csv), "\n")
cat("Repetitions:", reps, "\n")
cat("Requested ncomp:", ncomp_requested, "\n")
cat("PLSSVD cap:", plssvd_cap, "\n")
cat("CUDA available:", cuda_ok, "\n")
