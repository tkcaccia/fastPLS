suppressPackageStartupMessages(library(fastPLS))
set.seed(123)

out_csv <- Sys.getenv('FASTPLS_OUT_CSV', '/Users/stefano/Documents/GPUPLS/bench_cpp_fast_param_cycles.csv')
out_log <- Sys.getenv('FASTPLS_OUT_LOG', '/Users/stefano/Documents/GPUPLS/bench_cpp_fast_param_cycles.log')

# Restart from cycle 1 as requested
if (file.exists(out_csv)) file.remove(out_csv)
if (file.exists(out_log)) file.remove(out_log)

load('/Users/stefano/Documents/GPUPLS/CIFAR100.RData')
stopifnot(exists('r'))

data <- r[, -c(1:3)]
labels_all <- as.factor(r[, 'label_idx'])
data_num <- as.data.frame(lapply(data, function(x) suppressWarnings(as.numeric(as.character(x)))))
keep <- vapply(data_num, function(x) any(is.finite(x)), logical(1))
X_all <- as.matrix(data_num[, keep, drop = FALSE])

stamp <- function() format(Sys.time(), '%Y-%m-%d %H:%M:%S')
log_msg <- function(...) {
  msg <- paste0('[', stamp(), '] ', paste(..., collapse=''))
  cat(msg, '\n')
  cat(msg, '\n', file = out_log, append = TRUE)
}

# At least 10 replicates with different train/test splits
nsplits <- as.integer(Sys.getenv('FASTPLS_SPLITS', '10'))
if (!is.finite(nsplits) || is.na(nsplits) || nsplits < 10L) nsplits <- 10L
ncomp <- as.integer(Sys.getenv('FASTPLS_NCOMP', '100'))

# Literature-inspired candidate space (randomized SVD tuning + deflation controls)
cand_space <- expand.grid(
  block = c(2L, 3L, 4L, 5L, 6L, 8L, 10L),
  center_t = c(0L, 1L),
  reorth_v = c(0L, 1L),
  rsvd_oversample = c(6L, 10L, 14L, 20L),
  rsvd_power = c(0L, 1L, 2L),
  stringsAsFactors = FALSE
)

write.csv(data.frame(
  row_type=character(), cycle=integer(), splits=integer(),
  block=integer(), center_t=integer(), reorth_v=integer(), rsvd_oversample=integer(), rsvd_power=integer(),
  train_ms=numeric(), acc=numeric(), train_all=character(), acc_all=character(),
  plssvd_train_ms=numeric(), plssvd_train_all=character(), speedup_vs_plssvd=numeric(),
  baseline_train_ms=numeric(), baseline_acc=numeric(),
  incumbent_train_ms=numeric(), incumbent_acc=numeric(),
  speedup_vs_baseline=numeric(), acc_delta_vs_baseline=numeric(),
  speedup_vs_incumbent=numeric(), acc_delta_vs_incumbent=numeric(),
  consecutive_non_improve=integer(), status=character(), note=character(),
  started_at=character(), finished_at=character(),
  stringsAsFactors = FALSE
), out_csv, row.names = FALSE)

fit_eval <- function(kind, cfg, split_seed_start, nsplits) {
  train_ms <- numeric(nsplits)
  acc <- numeric(nsplits)

  old_block <- Sys.getenv('FASTPLS_FAST_BLOCK', '')
  old_center <- Sys.getenv('FASTPLS_FAST_CENTER_T', '')
  old_reorth <- Sys.getenv('FASTPLS_FAST_REORTH_V', '')
  on.exit({
    if (nzchar(old_block)) Sys.setenv(FASTPLS_FAST_BLOCK = old_block) else Sys.unsetenv('FASTPLS_FAST_BLOCK')
    if (nzchar(old_center)) Sys.setenv(FASTPLS_FAST_CENTER_T = old_center) else Sys.unsetenv('FASTPLS_FAST_CENTER_T')
    if (nzchar(old_reorth)) Sys.setenv(FASTPLS_FAST_REORTH_V = old_reorth) else Sys.unsetenv('FASTPLS_FAST_REORTH_V')
  }, add = TRUE)

  if (kind == 'fast') {
    Sys.setenv(
      FASTPLS_FAST_BLOCK = as.character(cfg$block),
      FASTPLS_FAST_CENTER_T = as.character(cfg$center_t),
      FASTPLS_FAST_REORTH_V = as.character(cfg$reorth_v)
    )
  }

  for (i in seq_len(nsplits)) {
    set.seed(split_seed_start + i)
    ss <- sample.int(nrow(X_all), round(nrow(X_all)/2))

    Xtrain <- X_all[ss, , drop = FALSE]
    Xtest <- X_all[-ss, , drop = FALSE]
    Ytrain <- labels_all[ss]
    Ytest <- labels_all[-ss]
    Ytrain_mat <- transformy(Ytrain)

    gc(FALSE)
    t0 <- proc.time()[3]
    model <- if (kind == 'simpls') {
      fastPLS:::pls_model2(Xtrain, Ytrain_mat, as.integer(ncomp), 1L, FALSE, 4L, 10L, 1L, as.integer(7000 + split_seed_start + i))
    } else if (kind == 'plssvd') {
      # cap for plssvd
      nc_eff <- as.integer(min(ncomp, nrow(Xtrain), ncol(Xtrain), ncol(Ytrain_mat)))
      fastPLS:::pls_model1(Xtrain, Ytrain_mat, nc_eff, 1L, FALSE, 4L, 10L, 1L, as.integer(7000 + split_seed_start + i))
    } else {
      fastPLS:::pls_model2_fast(
        Xtrain, Ytrain_mat, as.integer(ncomp), 1L, FALSE, 4L,
        as.integer(cfg$rsvd_oversample), as.integer(cfg$rsvd_power),
        as.integer(7000 + split_seed_start + i)
      )
    }
    train_ms[i] <- (proc.time()[3] - t0) * 1000

    # Accuracy still computed but excluded from timing
    pred <- fastPLS:::pls_predict(model, Xtest, FALSE)
    yp <- pred$Ypred
    cls <- apply(yp[, , 1, drop = FALSE], 1, which.max)
    lev <- levels(Ytrain)
    yhat <- factor(lev[cls], levels = lev)
    acc[i] <- mean(yhat == Ytest)
  }

  list(
    train_median_ms = median(train_ms),
    acc_median = median(acc),
    train_all = paste(sprintf('%.3f', train_ms), collapse=';'),
    acc_all = paste(sprintf('%.6f', acc), collapse=';')
  )
}

# Baselines
base_cfg <- data.frame(block=NA_integer_, center_t=NA_integer_, reorth_v=NA_integer_, rsvd_oversample=10L, rsvd_power=1L)
base_simpls <- fit_eval('simpls', base_cfg, split_seed_start = 10000L, nsplits = nsplits)
base_plssvd <- fit_eval('plssvd', base_cfg, split_seed_start = 10000L, nsplits = nsplits)

baseline_row <- data.frame(
  row_type='baseline', cycle=1L, splits=nsplits,
  block=NA_integer_, center_t=NA_integer_, reorth_v=NA_integer_, rsvd_oversample=10L, rsvd_power=1L,
  train_ms=base_simpls$train_median_ms, acc=base_simpls$acc_median,
  train_all=base_simpls$train_all, acc_all=base_simpls$acc_all,
  plssvd_train_ms=base_plssvd$train_median_ms, plssvd_train_all=base_plssvd$train_all,
  speedup_vs_plssvd=base_plssvd$train_median_ms / base_simpls$train_median_ms,
  baseline_train_ms=base_simpls$train_median_ms, baseline_acc=base_simpls$acc_median,
  incumbent_train_ms=base_simpls$train_median_ms, incumbent_acc=base_simpls$acc_median,
  speedup_vs_baseline=1, acc_delta_vs_baseline=0,
  speedup_vs_incumbent=1, acc_delta_vs_incumbent=0,
  consecutive_non_improve=0L, status='baseline', note='cycle restart + 10-split baseline',
  started_at=stamp(), finished_at=stamp(),
  stringsAsFactors = FALSE
)
write.table(baseline_row, file=out_csv, sep=',', row.names=FALSE, col.names=FALSE, append=TRUE, qmethod='double')
log_msg(sprintf('Baseline cycle 1: simpls %.1fms acc %.6f | plssvd %.1fms | splits=%d',
                base_simpls$train_median_ms, base_simpls$acc_median, base_plssvd$train_median_ms, nsplits))

incumbent_train <- base_simpls$train_median_ms
incumbent_acc <- base_simpls$acc_median
incumbent_tag <- 'cycle_1_baseline'
baseline_train <- base_simpls$train_median_ms
baseline_acc <- base_simpls$acc_median

consecutive_non_improve <- 0L
max_stale <- 10L
cycle_id <- 2L

while (consecutive_non_improve < max_stale) {
  started_at <- stamp()
  pick <- cand_space[sample.int(nrow(cand_space), 1L), , drop = FALSE]
  split_seed_start <- 50000L + cycle_id * 100L

  log_msg(sprintf('Cycle %d start: block=%d center_t=%d reorth_v=%d overs=%d power=%d',
                  cycle_id, pick$block, pick$center_t, pick$reorth_v, pick$rsvd_oversample, pick$rsvd_power))

  fast <- fit_eval('fast', pick, split_seed_start = split_seed_start, nsplits = nsplits)
  plssvd_cmp <- fit_eval('plssvd', base_cfg, split_seed_start = split_seed_start, nsplits = nsplits)

  inc_train_prev <- incumbent_train
  inc_acc_prev <- incumbent_acc

  speedup_base <- baseline_train / fast$train_median_ms
  acc_delta_base <- fast$acc_median - baseline_acc
  speedup_inc <- inc_train_prev / fast$train_median_ms
  acc_delta_inc <- fast$acc_median - inc_acc_prev

  improved <- (fast$train_median_ms < inc_train_prev) && (fast$acc_median >= inc_acc_prev)
  status <- if (improved) 'accepted' else 'rejected'

  if (improved) {
    incumbent_train <- fast$train_median_ms
    incumbent_acc <- fast$acc_median
    incumbent_tag <- sprintf('cycle_%d', cycle_id)
    consecutive_non_improve <- 0L
  } else {
    consecutive_non_improve <- consecutive_non_improve + 1L
  }

  row <- data.frame(
    row_type='cycle', cycle=cycle_id, splits=nsplits,
    block=pick$block, center_t=pick$center_t, reorth_v=pick$reorth_v,
    rsvd_oversample=pick$rsvd_oversample, rsvd_power=pick$rsvd_power,
    train_ms=fast$train_median_ms, acc=fast$acc_median,
    train_all=fast$train_all, acc_all=fast$acc_all,
    plssvd_train_ms=plssvd_cmp$train_median_ms, plssvd_train_all=plssvd_cmp$train_all,
    speedup_vs_plssvd=plssvd_cmp$train_median_ms / fast$train_median_ms,
    baseline_train_ms=baseline_train, baseline_acc=baseline_acc,
    incumbent_train_ms=inc_train_prev, incumbent_acc=inc_acc_prev,
    speedup_vs_baseline=speedup_base, acc_delta_vs_baseline=acc_delta_base,
    speedup_vs_incumbent=speedup_inc, acc_delta_vs_incumbent=acc_delta_inc,
    consecutive_non_improve=consecutive_non_improve,
    status=status,
    note=sprintf('incumbent=%s stale=%d/%d', incumbent_tag, consecutive_non_improve, max_stale),
    started_at=started_at, finished_at=stamp(),
    stringsAsFactors = FALSE
  )
  write.table(row, file=out_csv, sep=',', row.names=FALSE, col.names=FALSE, append=TRUE, qmethod='double')

  log_msg(sprintf('Cycle %d done: %s | inc %.1f -> fast %.1f ms | acc %.6f -> %.6f | plssvd %.1f ms | stale=%d/%d',
                  cycle_id, status, inc_train_prev, fast$train_median_ms, inc_acc_prev, fast$acc_median,
                  plssvd_cmp$train_median_ms, consecutive_non_improve, max_stale))

  cycle_id <- cycle_id + 1L
}

log_msg(sprintf('Stop condition reached: no improvement in %d consecutive cycles', max_stale))
