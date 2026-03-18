suppressPackageStartupMessages(library(fastPLS))

set.seed(123)
load('/Users/stefano/Documents/GPUPLS/CIFAR100.RData')
stopifnot(exists('r'))

data <- r[, -c(1:3)]
labels <- as.factor(r[, 'label_idx'])
data_num <- as.data.frame(lapply(data, function(x) suppressWarnings(as.numeric(as.character(x)))))
keep <- vapply(data_num, function(x) any(is.finite(x)), logical(1))
X <- as.matrix(data_num[, keep, drop = FALSE])

ss <- sample(nrow(X), round(nrow(X)/2))
Xtrain <- X[ss, , drop = FALSE]
Ytrain <- labels[ss]
Xtest <- X[-ss, , drop = FALSE]
Ytest <- labels[-ss]

Ytrain_mat <- transformy(Ytrain)

run_one <- function(kind, reps = 3L, ncomp = 100L, svd_method = 4L) {
  train_ms <- numeric(reps)
  acc <- numeric(reps)
  for (i in seq_len(reps)) {
    gc(FALSE)
    t0 <- proc.time()[3]
    model <- if (kind == 'base') {
      fastPLS:::pls_model2(Xtrain, Ytrain_mat, as.integer(ncomp), 1L, FALSE, svd_method, 10L, 1L, as.integer(100+i))
    } else {
      fastPLS:::pls_model2_fast(Xtrain, Ytrain_mat, as.integer(ncomp), 1L, FALSE, svd_method, 10L, 1L, as.integer(100+i))
    }
    train_ms[i] <- (proc.time()[3] - t0) * 1000

    pred <- fastPLS:::pls_predict(model, Xtest, FALSE)

    yp <- pred$Ypred
    if (length(dim(yp)) == 3L) {
      cls <- apply(yp[, , 1, drop = FALSE], 1, which.max)
      lev <- levels(Ytrain)
      yhat <- factor(lev[cls], levels = lev)
    } else if (is.data.frame(yp)) {
      yhat <- as.factor(yp[[1]])
    } else if (is.matrix(yp)) {
      yhat <- factor(colnames(yp)[max.col(yp)], levels = levels(Ytrain))
    } else {
      stop('Unsupported prediction shape')
    }
    acc[i] <- mean(yhat == Ytest)
  }
  data.frame(
    kind = kind,
    reps = reps,
    train_median_ms = median(train_ms),
    acc_median = median(acc),
    train_all = paste(sprintf('%.3f', train_ms), collapse=';'),
    acc_all = paste(sprintf('%.6f', acc), collapse=';')
  )
}

out <- rbind(
  run_one('base', reps = 3L, ncomp = 100L, svd_method = 4L),
  run_one('fast', reps = 3L, ncomp = 100L, svd_method = 4L)
)
write.csv(out, '/Users/stefano/Documents/GPUPLS/bench_cpp_fast_cycle_eval.csv', row.names = FALSE)
print(out)
