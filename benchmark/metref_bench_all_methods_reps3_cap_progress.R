#!/usr/bin/env Rscript

suppressPackageStartupMessages({
  library(fastPLS)
  library(KODAMA)
})

set.seed(123)

out_csv <- Sys.getenv("FASTPLS_METREF_OUT", "~/gpupls_test/GPUPLS/metref_bench_all_methods_reps3_cap_progress.csv")
out_log <- Sys.getenv("FASTPLS_METREF_PROGRESS_LOG", "~/gpupls_test/GPUPLS/metref_bench_all_methods_reps3_cap_progress.log")
ncomp_requested <- as.integer(Sys.getenv("FASTPLS_NCOMP", "50"))
if (!is.finite(ncomp_requested) || is.na(ncomp_requested) || ncomp_requested < 1L) ncomp_requested <- 50L
reps <- 3L

stamp <- function() format(Sys.time(), "%Y-%m-%d %H:%M:%S")
log_msg <- function(...) {
  msg <- paste0("[", stamp(), "] ", paste(..., collapse = ""))
  cat(msg, "\n")
  cat(msg, "\n", file = path.expand(out_log), append = TRUE)
  flush.console()
}

# MetRef prep (same split style used in existing benchmark scripts)
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

accuracy <- function(pred, truth) mean(pred == truth)

plssvd_cap <- min(nrow(Xtrain), ncol(Xtrain), nlevels(Ytrain))

methods <- rbind(
  data.frame(
    engine = "Rcpp",
    algorithm = rep(c("simpls", "plssvd"), each = 5),
    svd_method = rep(c("irlba", "cpu_exact", "cpu_rsvd", "cuda_rsvd"), 2),
    stringsAsFactors = FALSE
  ),
  data.frame(
    engine = "R",
    algorithm = rep(c("simpls", "plssvd"), each = 4),
    svd_method = rep(c("irlba", "cpu_exact", "cpu_rsvd"), 2),
    stringsAsFactors = FALSE
  )
)
methods$config <- paste(methods$engine, methods$algorithm, methods$svd_method, sep = "_")

cuda_ok <- tryCatch(isTRUE(has_cuda()), error = function(e) FALSE)

fit_once <- function(engine, algorithm, svd_method, ncomp) {
  if (engine == "Rcpp") {
    pls(Xtrain, Ytrain, Xtest, ncomp = ncomp, method = algorithm, svd.method = svd_method, scaling = "centering")
  } else {
    pls_r(Xtrain, Ytrain, Xtest, ncomp = ncomp, method = algorithm, svd.method = svd_method, scaling = "centering")
  }
}

out_csv <- path.expand(out_csv)
out_log <- path.expand(out_log)
dir.create(dirname(out_csv), recursive = TRUE, showWarnings = FALSE)

header <- data.frame(
  config = character(),
  engine = character(),
  algorithm = character(),
  svd_method = character(),
  ncomp_requested = integer(),
  ncomp_run = integer(),
  plssvd_cap = integer(),
  reps = integer(),
  status = character(),
  time_s_median_reps = numeric(),
  time_s_mean_reps = numeric(),
  acc_median_reps = numeric(),
  msg = character(),
  rep_times_s = character(),
  rep_acc = character(),
  started_at = character(),
  finished_at = character(),
  stringsAsFactors = FALSE
)
write.csv(header, out_csv, row.names = FALSE, quote = TRUE)
cat("", file = out_log, append = FALSE)

log_msg("MetRef benchmark started")
log_msg("Output CSV: ", out_csv)
log_msg("Requested ncomp: ", ncomp_requested, "; reps: ", reps, "; plssvd cap: ", plssvd_cap, "; CUDA: ", cuda_ok)

n_total <- nrow(methods)
for (i in seq_len(n_total)) {
  cfg <- methods[i, ]
  started_at <- stamp()
  log_msg(sprintf("[%d/%d] START %s", i, n_total, cfg$config))

  if (cfg$algorithm == "plssvd" && ncomp_requested > plssvd_cap) {
    row <- data.frame(
      config = cfg$config, engine = cfg$engine, algorithm = cfg$algorithm, svd_method = cfg$svd_method,
      ncomp_requested = ncomp_requested, ncomp_run = NA_integer_, plssvd_cap = plssvd_cap, reps = reps,
      status = "skipped_ncomp_above_plssvd_cap", time_s_median_reps = NA_real_, time_s_mean_reps = NA_real_,
      acc_median_reps = NA_real_, msg = "", rep_times_s = "", rep_acc = "",
      started_at = started_at, finished_at = stamp(), stringsAsFactors = FALSE
    )
    write.table(row, file = out_csv, sep = ",", row.names = FALSE, col.names = FALSE, append = TRUE, qmethod = "double")
    log_msg(sprintf("[%d/%d] SKIP %s (ncomp above plssvd cap)", i, n_total, cfg$config))
    next
  }

  if (cfg$svd_method == "cuda_rsvd" && !cuda_ok) {
    row <- data.frame(
      config = cfg$config, engine = cfg$engine, algorithm = cfg$algorithm, svd_method = cfg$svd_method,
      ncomp_requested = ncomp_requested, ncomp_run = NA_integer_, plssvd_cap = plssvd_cap, reps = reps,
      status = "skipped_cuda_unavailable", time_s_median_reps = NA_real_, time_s_mean_reps = NA_real_,
      acc_median_reps = NA_real_, msg = "", rep_times_s = "", rep_acc = "",
      started_at = started_at, finished_at = stamp(), stringsAsFactors = FALSE
    )
    write.table(row, file = out_csv, sep = ",", row.names = FALSE, col.names = FALSE, append = TRUE, qmethod = "double")
    log_msg(sprintf("[%d/%d] SKIP %s (cuda unavailable)", i, n_total, cfg$config))
    next
  }

  elapsed <- rep(NA_real_, reps)
  rep_acc <- rep(NA_real_, reps)
  err <- NULL

  for (r in seq_len(reps)) {
    gc(FALSE)
    t0 <- proc.time()[3]
    fit <- tryCatch(fit_once(cfg$engine, cfg$algorithm, cfg$svd_method, ncomp_requested), error = function(e) e)
    elapsed[r] <- as.numeric(proc.time()[3] - t0)

    if (inherits(fit, "error")) {
      err <- conditionMessage(fit)
      log_msg(sprintf("[%d/%d] %s rep %d/%d ERROR: %s", i, n_total, cfg$config, r, reps, err))
      break
    }

    pred <- extract_pred(fit)
    rep_acc[r] <- accuracy(pred, Ytest)
    log_msg(sprintf("[%d/%d] %s rep %d/%d done in %.3fs", i, n_total, cfg$config, r, reps, elapsed[r]))
  }

  if (!is.null(err)) {
    row <- data.frame(
      config = cfg$config, engine = cfg$engine, algorithm = cfg$algorithm, svd_method = cfg$svd_method,
      ncomp_requested = ncomp_requested, ncomp_run = NA_integer_, plssvd_cap = plssvd_cap, reps = reps,
      status = "error", time_s_median_reps = NA_real_, time_s_mean_reps = NA_real_, acc_median_reps = NA_real_,
      msg = gsub('"', "'", err, fixed = TRUE),
      rep_times_s = paste(sprintf("%.6f", elapsed[is.finite(elapsed)]), collapse = ";"),
      rep_acc = paste(sprintf("%.6f", rep_acc[is.finite(rep_acc)]), collapse = ";"),
      started_at = started_at, finished_at = stamp(), stringsAsFactors = FALSE
    )
    write.table(row, file = out_csv, sep = ",", row.names = FALSE, col.names = FALSE, append = TRUE, qmethod = "double")
    log_msg(sprintf("[%d/%d] END %s with error", i, n_total, cfg$config))
  } else {
    row <- data.frame(
      config = cfg$config, engine = cfg$engine, algorithm = cfg$algorithm, svd_method = cfg$svd_method,
      ncomp_requested = ncomp_requested, ncomp_run = ncomp_requested, plssvd_cap = plssvd_cap, reps = reps,
      status = "ok", time_s_median_reps = median(elapsed, na.rm = TRUE), time_s_mean_reps = mean(elapsed, na.rm = TRUE),
      acc_median_reps = median(rep_acc, na.rm = TRUE), msg = "",
      rep_times_s = paste(sprintf("%.6f", elapsed), collapse = ";"),
      rep_acc = paste(sprintf("%.6f", rep_acc), collapse = ";"),
      started_at = started_at, finished_at = stamp(), stringsAsFactors = FALSE
    )
    write.table(row, file = out_csv, sep = ",", row.names = FALSE, col.names = FALSE, append = TRUE, qmethod = "double")
    log_msg(sprintf("[%d/%d] END %s median=%.3fs acc=%.4f", i, n_total, cfg$config, row$time_s_median_reps, row$acc_median_reps))
  }
}

log_msg("MetRef benchmark completed")
cat("Done\n")
cat("Output:", out_csv, "\n")
cat("Progress log:", out_log, "\n")
