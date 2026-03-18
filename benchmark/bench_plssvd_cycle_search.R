suppressPackageStartupMessages(library(fastPLS))
set.seed(123)

out_csv <- Sys.getenv('FASTPLS_PLSSVD_OUT_CSV', '/Users/stefano/Documents/GPUPLS/bench_plssvd_param_cycles.csv')
out_log <- Sys.getenv('FASTPLS_PLSSVD_OUT_LOG', '/Users/stefano/Documents/GPUPLS/bench_plssvd_param_cycles.log')

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

nsplits <- as.integer(Sys.getenv('FASTPLS_SPLITS', '10'))
if (!is.finite(nsplits) || is.na(nsplits) || nsplits < 10L) nsplits <- 10L
ncomp <- as.integer(Sys.getenv('FASTPLS_NCOMP', '100'))

# PLSSVD tuning space (CPU only)
cands <- expand.grid(
  svd_method = c(1L, 2L, 3L, 4L),   # irlba, dc, cpu_exact, cpu_rsvd
  rsvd_oversample = c(4L, 6L, 10L, 14L, 20L),
  rsvd_power = c(0L, 1L, 2L),
  stringsAsFactors = FALSE
)
# For non-rsvd methods, oversample/power have no effect; keep one canonical row each
cands <- unique(do.call(rbind, lapply(seq_len(nrow(cands)), function(i) {
  x <- cands[i, , drop = FALSE]
  if (x$svd_method != 4L) {
    x$rsvd_oversample <- 10L
    x$rsvd_power <- 1L
  }
  x
})))

write.csv(data.frame(
  row_type=character(), cycle=integer(), splits=integer(),
  svd_method=integer(), rsvd_oversample=integer(), rsvd_power=integer(),
  train_ms=numeric(), acc=numeric(), train_all=character(), acc_all=character(),
  baseline_train_ms=numeric(), baseline_acc=numeric(),
  incumbent_train_ms=numeric(), incumbent_acc=numeric(),
  speedup_vs_baseline=numeric(), acc_delta_vs_baseline=numeric(),
  speedup_vs_incumbent=numeric(), acc_delta_vs_incumbent=numeric(),
  consecutive_non_improve=integer(), status=character(), note=character(),
  started_at=character(), finished_at=character(), stringsAsFactors = FALSE
), out_csv, row.names = FALSE)

fit_plssvd <- function(svd_method, overs, power, split_seed_start, nsplits) {
  train_ms <- numeric(nsplits)
  acc <- numeric(nsplits)
  for (i in seq_len(nsplits)) {
    set.seed(split_seed_start + i)
    ss <- sample.int(nrow(X_all), round(nrow(X_all)/2))
    Xtrain <- X_all[ss, , drop = FALSE]
    Xtest <- X_all[-ss, , drop = FALSE]
    Ytrain <- labels_all[ss]
    Ytest <- labels_all[-ss]
    Ymat <- transformy(Ytrain)

    nc_eff <- as.integer(min(ncomp, nrow(Xtrain), ncol(Xtrain), ncol(Ymat)))
    gc(FALSE)
    t0 <- proc.time()[3]
    model <- fastPLS:::pls_model1(
      Xtrain, Ymat, nc_eff, 1L, FALSE,
      as.integer(svd_method), as.integer(overs), as.integer(power),
      as.integer(11000 + split_seed_start + i)
    )
    train_ms[i] <- (proc.time()[3] - t0) * 1000

    pred <- fastPLS:::pls_predict(model, Xtest, FALSE)
    cls <- apply(pred$Ypred[, , 1, drop = FALSE], 1, which.max)
    yhat <- factor(levels(Ytrain)[cls], levels = levels(Ytrain))
    acc[i] <- mean(yhat == Ytest)
  }
  list(
    train_median_ms = median(train_ms),
    acc_median = median(acc),
    train_all = paste(sprintf('%.3f', train_ms), collapse=';'),
    acc_all = paste(sprintf('%.6f', acc), collapse=';')
  )
}

# Baseline PLSSVD (cpu_rsvd overs=10 power=1)
base <- fit_plssvd(4L, 10L, 1L, split_seed_start = 10000L, nsplits = nsplits)
base_row <- data.frame(
  row_type='baseline', cycle=1L, splits=nsplits,
  svd_method=4L, rsvd_oversample=10L, rsvd_power=1L,
  train_ms=base$train_median_ms, acc=base$acc_median,
  train_all=base$train_all, acc_all=base$acc_all,
  baseline_train_ms=base$train_median_ms, baseline_acc=base$acc_median,
  incumbent_train_ms=base$train_median_ms, incumbent_acc=base$acc_median,
  speedup_vs_baseline=1, acc_delta_vs_baseline=0,
  speedup_vs_incumbent=1, acc_delta_vs_incumbent=0,
  consecutive_non_improve=0L, status='baseline', note='PLSSVD baseline',
  started_at=stamp(), finished_at=stamp(), stringsAsFactors = FALSE
)
write.table(base_row, file=out_csv, sep=',', row.names=FALSE, col.names=FALSE, append=TRUE, qmethod='double')
log_msg(sprintf('Baseline PLSSVD cycle 1: train %.1f ms acc %.6f (svd=cpu_rsvd overs=10 power=1, splits=%d)', base$train_median_ms, base$acc_median, nsplits))

inc_train <- base$train_median_ms
inc_acc <- base$acc_median
inc_tag <- 'cycle_1_baseline'
stale <- 0L
max_stale <- as.integer(Sys.getenv('FASTPLS_MAX_STALE', '10'))
if (!is.finite(max_stale) || is.na(max_stale) || max_stale < 1L) max_stale <- 10L
cycle_id <- 2L

while (stale < max_stale) {
  started_at <- stamp()
  pick <- cands[sample.int(nrow(cands), 1L), , drop = FALSE]
  log_msg(sprintf('Cycle %d start: svd=%d overs=%d power=%d', cycle_id, pick$svd_method, pick$rsvd_oversample, pick$rsvd_power))

  cur <- fit_plssvd(pick$svd_method, pick$rsvd_oversample, pick$rsvd_power, split_seed_start = 50000L + cycle_id*100L, nsplits = nsplits)

  speedup_base <- base$train_median_ms / cur$train_median_ms
  acc_delta_base <- cur$acc_median - base$acc_median
  speedup_inc <- inc_train / cur$train_median_ms
  acc_delta_inc <- cur$acc_median - inc_acc

  improved <- (cur$train_median_ms < inc_train) && (cur$acc_median >= inc_acc)
  status <- if (improved) 'accepted' else 'rejected'
  if (improved) {
    inc_train <- cur$train_median_ms
    inc_acc <- cur$acc_median
    inc_tag <- sprintf('cycle_%d', cycle_id)
    stale <- 0L
  } else {
    stale <- stale + 1L
  }

  row <- data.frame(
    row_type='cycle', cycle=cycle_id, splits=nsplits,
    svd_method=pick$svd_method, rsvd_oversample=pick$rsvd_oversample, rsvd_power=pick$rsvd_power,
    train_ms=cur$train_median_ms, acc=cur$acc_median,
    train_all=cur$train_all, acc_all=cur$acc_all,
    baseline_train_ms=base$train_median_ms, baseline_acc=base$acc_median,
    incumbent_train_ms=if (status=='accepted') inc_train else inc_train,
    incumbent_acc=if (status=='accepted') inc_acc else inc_acc,
    speedup_vs_baseline=speedup_base, acc_delta_vs_baseline=acc_delta_base,
    speedup_vs_incumbent=speedup_inc, acc_delta_vs_incumbent=acc_delta_inc,
    consecutive_non_improve=stale, status=status,
    note=sprintf('incumbent=%s stale=%d/%d', inc_tag, stale, max_stale),
    started_at=started_at, finished_at=stamp(), stringsAsFactors = FALSE
  )
  write.table(row, file=out_csv, sep=',', row.names=FALSE, col.names=FALSE, append=TRUE, qmethod='double')

  log_msg(sprintf('Cycle %d done: %s | inc %.1f -> cur %.1f ms | acc %.6f -> %.6f | stale=%d/%d',
                  cycle_id, status, if (status=='accepted') inc_train else inc_train, cur$train_median_ms,
                  if (status=='accepted') inc_acc else inc_acc, cur$acc_median, stale, max_stale))

  cycle_id <- cycle_id + 1L
}

log_msg(sprintf('Stop condition reached: no PLSSVD improvement in %d consecutive cycles', max_stale))
