suppressPackageStartupMessages({
  library(fastPLS)
  library(KODAMA)
  library(bench)
})

set.seed(123)

# MetRef preparation
metref_data <- local({
  data(MetRef, package = "KODAMA")
  u <- MetRef$data
  u <- u[, -which(colSums(u) == 0), drop = FALSE]
  u <- normalization(u)$newXtrain
  class <- as.numeric(as.factor(MetRef$donor))
  ss <- sample(nrow(u), 100)
  list(
    Xtrain = u[-ss, , drop = FALSE],
    Ytrain = as.factor(class)[-ss],
    Xtest  = u[ss, , drop = FALSE],
    Ytest  = as.factor(class)[ss]
  )
})

Xtrain <- metref_data$Xtrain
Ytrain <- metref_data$Ytrain
Xtest  <- metref_data$Xtest
Ytest  <- metref_data$Ytest

ncomp <- 20
reps <- 3L

cfg <- expand.grid(
  algorithm = c("simpls", "plssvd"),
  svd_method = c("irlba", "dc", "cpu_exact", "cpu_rsvd", "cuda_rsvd"),
  stringsAsFactors = FALSE
)
cfg$method_id <- paste0("Rcpp_", cfg$algorithm, "_", cfg$svd_method)

safe_mb <- function(x) as.numeric(object.size(x)) / (1024^2)

balanced_accuracy <- function(truth, pred) {
  lv <- union(levels(truth), levels(pred))
  truth <- factor(truth, levels = lv)
  pred <- factor(pred, levels = lv)
  tab <- table(pred, truth)
  recalls <- numeric(length(lv))
  for (i in seq_along(lv)) {
    d <- sum(tab[, i])
    recalls[i] <- if (d == 0) NA_real_ else tab[i, i] / d
  }
  mean(recalls, na.rm = TRUE)
}

macro_f1 <- function(truth, pred) {
  lv <- union(levels(truth), levels(pred))
  truth <- factor(truth, levels = lv)
  pred <- factor(pred, levels = lv)
  tab <- table(pred, truth)
  f1 <- numeric(length(lv))
  for (i in seq_along(lv)) {
    tp <- tab[i, i]
    fp <- sum(tab[i, ]) - tp
    fn <- sum(tab[, i]) - tp
    denom <- 2 * tp + fp + fn
    f1[i] <- if (denom == 0) NA_real_ else (2 * tp / denom)
  }
  mean(f1, na.rm = TRUE)
}

fit_once <- function(algorithm, svd_method) {
  pls(
    Xtrain, Ytrain, Xtest,
    ncomp = ncomp,
    method = algorithm,
    svd.method = svd_method,
    scaling = "centering"
  )
}

rows <- vector("list", nrow(cfg))
for (i in seq_len(nrow(cfg))) {
  alg <- cfg$algorithm[i]
  svd <- cfg$svd_method[i]
  id <- cfg$method_id[i]
  message("Running ", id)

  warn_txt <- character(0)
  err <- NULL
  out <- NULL

  t0 <- proc.time()[3]
  tryCatch({
    out <- withCallingHandlers(
      fit_once(alg, svd),
      warning = function(w) {
        warn_txt <<- c(warn_txt, conditionMessage(w))
        invokeRestart("muffleWarning")
      }
    )
  }, error = function(e) {
    err <<- conditionMessage(e)
  })
  elapsed <- proc.time()[3] - t0

  if (!is.null(err)) {
    rows[[i]] <- data.frame(
      method_id = id, algorithm = alg, svd_method = svd,
      status = paste0("error: ", err),
      elapsed_s = NA_real_, bench_median_ms = NA_real_, bench_itr_sec = NA_real_,
      mem_alloc_mb = NA_real_, gc_sec = NA_real_,
      model_size_mb = NA_real_,
      accuracy = NA_real_, balanced_acc = NA_real_, macro_f1 = NA_real_,
      stringsAsFactors = FALSE
    )
    next
  }

  pred <- as.factor(out$Ypred[, 1])
  acc <- mean(pred == Ytest)
  bacc <- balanced_accuracy(Ytest, pred)
  mf1 <- macro_f1(Ytest, pred)

  b <- bench::mark(
    {
      o <- fit_once(alg, svd)
      invisible(o$Ypred)
    },
    iterations = reps,
    check = FALSE,
    memory = TRUE,
    time_unit = "ms"
  )

  rows[[i]] <- data.frame(
    method_id = id,
    algorithm = alg,
    svd_method = svd,
    status = if (length(warn_txt)) paste(unique(warn_txt), collapse = " | ") else "ok",
    elapsed_s = as.numeric(elapsed),
    bench_median_ms = as.numeric(b$median),
    bench_itr_sec = as.numeric(b$`itr/sec`),
    mem_alloc_mb = as.numeric(b$mem_alloc) / (1024^2),
    gc_sec = as.numeric(b$`gc/sec`),
    model_size_mb = safe_mb(out),
    accuracy = as.numeric(acc),
    balanced_acc = as.numeric(bacc),
    macro_f1 = as.numeric(mf1),
    stringsAsFactors = FALSE
  )
}

res <- do.call(rbind, rows)
res <- res[order(res$algorithm, res$svd_method), ]

write.csv(res, "metref_all_svd_impl_results.csv", row.names = FALSE)
cat("\nResults:\n")
print(res, row.names = FALSE)

ok <- res[grepl("^ok", res$status), ]
if (nrow(ok) > 0) {
  cat("\nSummary (sorted by median time):\n")
  print(ok[order(ok$bench_median_ms), c("method_id", "bench_median_ms", "mem_alloc_mb", "accuracy", "macro_f1")], row.names = FALSE)
}
