suppressPackageStartupMessages(library(fastPLS))
set.seed(123)
load('/Users/stefano/Documents/GPUPLS/CIFAR100.RData')
data <- r[, -c(1:3)]
labels <- as.factor(r[, 'label_idx'])
data_num <- as.data.frame(lapply(data, function(x) suppressWarnings(as.numeric(as.character(x)))))
keep <- vapply(data_num, function(x) any(is.finite(x)), logical(1))
X <- as.matrix(data_num[, keep, drop = FALSE])

run_cfg <- function(incr, iters, reorth, reps=5L) {
  Sys.setenv(FASTPLS_FAST_INCREMENTAL=as.character(incr), FASTPLS_FAST_INC_ITERS=as.character(iters), FASTPLS_FAST_DEFLCACHE='1', FASTPLS_FAST_BLOCK='6', FASTPLS_FAST_CENTER_T='1', FASTPLS_FAST_REORTH_V=as.character(reorth))
  train_ms <- numeric(reps); acc <- numeric(reps)
  for (i in seq_len(reps)) {
    set.seed(2000+i)
    ss <- sample.int(nrow(X), round(nrow(X)/2))
    Xtrain <- X[ss,,drop=FALSE]; Xtest <- X[-ss,,drop=FALSE]
    Ytrain <- labels[ss]; Ytest <- labels[-ss]
    Ymat <- transformy(Ytrain)
    t0 <- proc.time()[3]
    model <- fastPLS:::pls_model2_fast(Xtrain, Ymat, as.integer(100), 1L, FALSE, 4L, 14L, 2L, as.integer(10000+i))
    train_ms[i] <- (proc.time()[3] - t0)*1000
    p <- fastPLS:::pls_predict(model, Xtest, FALSE)
    cls <- apply(p$Ypred[, , 1, drop = FALSE], 1, which.max)
    yhat <- factor(levels(Ytrain)[cls], levels=levels(Ytrain))
    acc[i] <- mean(yhat == Ytest)
  }
  data.frame(incr=incr, iters=iters, reorth=reorth, train_median_ms=median(train_ms), acc_median=median(acc), stringsAsFactors=FALSE)
}

res <- do.call(rbind, list(
  run_cfg(0,2,0),
  run_cfg(1,2,0),
  run_cfg(1,3,0),
  run_cfg(1,4,0),
  run_cfg(1,2,1),
  run_cfg(1,3,1)
))
write.csv(res, '/Users/stefano/Documents/GPUPLS/bench_incremental_tune.csv', row.names=FALSE)
print(res)
