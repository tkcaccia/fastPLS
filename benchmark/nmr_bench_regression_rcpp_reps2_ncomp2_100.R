#!/usr/bin/env Rscript

suppressPackageStartupMessages({
  library(fastPLS)
})

set.seed(as.integer(Sys.getenv("FASTPLS_SEED", "123")))

data_path <- Sys.getenv("FASTPLS_NMR_DATA", "~/gpupls_test/GPUPLS/NMR.RData")
out_csv <- Sys.getenv("FASTPLS_NMR_OUT", "~/gpupls_test/GPUPLS/nmr_bench_regression_rcpp_reps2_ncomp2_100.csv")
out_log <- Sys.getenv("FASTPLS_NMR_PROGRESS_LOG", "~/gpupls_test/GPUPLS/nmr_bench_regression_rcpp_reps2_ncomp2_100.log")
reps <- 2L
ncomp_grid <- c(2L, 100L)

stamp <- function() format(Sys.time(), "%Y-%m-%d %H:%M:%S")
log_msg <- function(...) {
  msg <- paste0("[", stamp(), "] ", paste(..., collapse = ""))
  cat(msg, "\n")
  cat(msg, "\n", file = path.expand(out_log), append = TRUE)
  flush.console()
}

rmsd <- function(pred, truth) sqrt(mean((pred - truth)^2, na.rm = TRUE))

get_pred <- function(fit) {
  yp <- fit$Ypred
  if (length(dim(yp)) == 3L) return(yp[, , 1, drop = FALSE][, , 1])
  if (is.matrix(yp)) return(yp[, 1])
  as.numeric(yp)
}

load(path.expand(data_path))
Xtrain <- as.matrix(Xtrain)
Ytrain <- as.matrix(Ytrain)
Xtest <- as.matrix(Xtest)
Ytest <- as.matrix(Ytest)

y_cap_dim <- if (is.matrix(Ytrain) || is.data.frame(Ytrain)) ncol(Ytrain) else 1L
plssvd_cap <- min(nrow(Xtrain), ncol(Xtrain), y_cap_dim)

cuda_ok <- tryCatch(isTRUE(has_cuda()), error = function(e) FALSE)

methods <- rbind(
  data.frame(engine = "Rcpp", algorithm = rep(c("simpls", "plssvd"), each = 4),
             svd_method = rep(c("irlba", "cpu_rsvd", "cuda_rsvd"), 2), stringsAsFactors = FALSE),
  data.frame(engine = "Rcpp", algorithm = "simpls_fast",
             svd_method = c("irlba", "cpu_rsvd", "cuda_rsvd"), stringsAsFactors = FALSE)
)
methods$config <- paste(methods$engine, methods$algorithm, methods$svd_method, sep = "_")

fit_once <- function(algorithm, svd_method, ncomp) {
  pls(Xtrain, Ytrain, Xtest, ncomp = ncomp, method = algorithm, svd.method = svd_method, scaling = "centering")
}

out_csv <- path.expand(out_csv)
out_log <- path.expand(out_log)
dir.create(dirname(out_csv), recursive = TRUE, showWarnings = FALSE)
cat("", file = out_log, append = FALSE)

header <- data.frame(
  config = character(),
  algorithm = character(),
  svd_method = character(),
  ncomp_requested = integer(),
  ncomp_run = integer(),
  plssvd_cap = integer(),
  reps = integer(),
  status = character(),
  time_s_median_reps = numeric(),
  time_s_mean_reps = numeric(),
  rmsd_median_reps = numeric(),
  msg = character(),
  rep_times_s = character(),
  rep_rmsd = character(),
  started_at = character(),
  finished_at = character(),
  stringsAsFactors = FALSE
)
write.csv(header, out_csv, row.names = FALSE, quote = TRUE)

plan <- do.call(rbind, lapply(seq_len(nrow(methods)), function(i) {
  cfg <- methods[i, ]
  do.call(rbind, lapply(ncomp_grid, function(nc) {
    data.frame(config = cfg$config, algorithm = cfg$algorithm, svd_method = cfg$svd_method,
               ncomp = as.integer(nc), stringsAsFactors = FALSE)
  }))
}))

log_msg("NMR benchmark started")
log_msg("Data: ", path.expand(data_path))
log_msg("Rows train/test: ", nrow(Xtrain), "/", nrow(Xtest), "; features: ", ncol(Xtrain))
log_msg("Reps: ", reps, "; ncomp: ", paste(ncomp_grid, collapse = ","), "; CUDA: ", cuda_ok, "; plssvd cap: ", plssvd_cap)
log_msg("Total runs planned: ", nrow(plan), " (Rcpp only)")

for (i in seq_len(nrow(plan))) {
  cfg <- plan[i, ]
  started_at <- stamp()
  log_msg(sprintf("[%d/%d] START %s ncomp=%d", i, nrow(plan), cfg$config, cfg$ncomp))

  if (cfg$algorithm == "plssvd" && cfg$ncomp > plssvd_cap) {
    row <- data.frame(config = cfg$config, algorithm = cfg$algorithm, svd_method = cfg$svd_method,
      ncomp_requested = cfg$ncomp, ncomp_run = NA_integer_, plssvd_cap = plssvd_cap, reps = reps,
      status = "skipped_ncomp_above_plssvd_cap", time_s_median_reps = NA_real_, time_s_mean_reps = NA_real_,
      rmsd_median_reps = NA_real_, msg = "", rep_times_s = "", rep_rmsd = "",
      started_at = started_at, finished_at = stamp(), stringsAsFactors = FALSE)
    write.table(row, file = out_csv, sep = ",", row.names = FALSE, col.names = FALSE, append = TRUE, qmethod = "double")
    log_msg(sprintf("[%d/%d] SKIP %s ncomp=%d (above plssvd cap)", i, nrow(plan), cfg$config, cfg$ncomp))
    next
  }

  if (cfg$svd_method == "cuda_rsvd" && !cuda_ok) {
    row <- data.frame(config = cfg$config, algorithm = cfg$algorithm, svd_method = cfg$svd_method,
      ncomp_requested = cfg$ncomp, ncomp_run = NA_integer_, plssvd_cap = plssvd_cap, reps = reps,
      status = "skipped_cuda_unavailable", time_s_median_reps = NA_real_, time_s_mean_reps = NA_real_,
      rmsd_median_reps = NA_real_, msg = "", rep_times_s = "", rep_rmsd = "",
      started_at = started_at, finished_at = stamp(), stringsAsFactors = FALSE)
    write.table(row, file = out_csv, sep = ",", row.names = FALSE, col.names = FALSE, append = TRUE, qmethod = "double")
    log_msg(sprintf("[%d/%d] SKIP %s ncomp=%d (cuda unavailable)", i, nrow(plan), cfg$config, cfg$ncomp))
    next
  }

  elapsed <- rep(NA_real_, reps)
  rep_rmsd <- rep(NA_real_, reps)
  err <- NULL

  for (r in seq_len(reps)) {
    gc(FALSE)
    t0 <- proc.time()[3]
    fit <- tryCatch(fit_once(cfg$algorithm, cfg$svd_method, cfg$ncomp), error = function(e) e)
    elapsed[r] <- as.numeric(proc.time()[3] - t0)

    if (inherits(fit, "error")) {
      err <- conditionMessage(fit)
      log_msg(sprintf("[%d/%d] %s ncomp=%d rep %d/%d ERROR: %s", i, nrow(plan), cfg$config, cfg$ncomp, r, reps, err))
      break
    }

    rep_rmsd[r] <- rmsd(get_pred(fit), Ytest)
    log_msg(sprintf("[%d/%d] %s ncomp=%d rep %d/%d done in %.3fs RMSD=%.6f",
                    i, nrow(plan), cfg$config, cfg$ncomp, r, reps, elapsed[r], rep_rmsd[r]))
  }

  if (!is.null(err)) {
    row <- data.frame(config = cfg$config, algorithm = cfg$algorithm, svd_method = cfg$svd_method,
      ncomp_requested = cfg$ncomp, ncomp_run = NA_integer_, plssvd_cap = plssvd_cap, reps = reps,
      status = "error", time_s_median_reps = NA_real_, time_s_mean_reps = NA_real_, rmsd_median_reps = NA_real_,
      msg = gsub('"', "'", err, fixed = TRUE),
      rep_times_s = paste(sprintf("%.6f", elapsed[is.finite(elapsed)]), collapse = ";"),
      rep_rmsd = paste(sprintf("%.10f", rep_rmsd[is.finite(rep_rmsd)]), collapse = ";"),
      started_at = started_at, finished_at = stamp(), stringsAsFactors = FALSE)
  } else {
    row <- data.frame(config = cfg$config, algorithm = cfg$algorithm, svd_method = cfg$svd_method,
      ncomp_requested = cfg$ncomp, ncomp_run = cfg$ncomp, plssvd_cap = plssvd_cap, reps = reps,
      status = "ok", time_s_median_reps = median(elapsed, na.rm = TRUE), time_s_mean_reps = mean(elapsed, na.rm = TRUE),
      rmsd_median_reps = median(rep_rmsd, na.rm = TRUE), msg = "",
      rep_times_s = paste(sprintf("%.6f", elapsed), collapse = ";"),
      rep_rmsd = paste(sprintf("%.10f", rep_rmsd), collapse = ";"),
      started_at = started_at, finished_at = stamp(), stringsAsFactors = FALSE)
    log_msg(sprintf("[%d/%d] END %s ncomp=%d median=%.3fs RMSD=%.6f",
                    i, nrow(plan), cfg$config, cfg$ncomp, row$time_s_median_reps, row$rmsd_median_reps))
  }

  write.table(row, file = out_csv, sep = ",", row.names = FALSE, col.names = FALSE, append = TRUE, qmethod = "double")
}

log_msg("NMR benchmark completed")
cat("Done\n")
cat("Output:", out_csv, "\n")
cat("Progress log:", out_log, "\n")
