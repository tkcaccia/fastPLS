suppressPackageStartupMessages({
  library(fastPLS)
  library(KODAMA)
  library(bench)
})

set.seed(123)

# -------------------------
# Data preparation (MetRef)
# -------------------------
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

ncomp <- 20
reps <- 3L

# -------------------------
# Method configurations
# -------------------------
configs <- data.frame(
  id = sprintf("%02d", 1:10),
  name = c(
    "R_simpls_svd",
    "R_simpls_irlba",
    "R_plssvd_svd",
    "R_plssvd_irlba",
    "Rcpp_simpls_svd",
    "Rcpp_simpls_irlba",
    "Rcpp_plssvd_svd",
    "Rcpp_plssvd_irlba",
    "CUDA_simpls",
    "CUDA_plssvd"
  ),
  engine = c(rep("R", 4), rep("Rcpp", 4), rep("Rcpp", 2)),
  algorithm = c("simpls", "simpls", "plssvd", "plssvd", "simpls", "simpls", "plssvd", "plssvd", "simpls", "plssvd"),
  svd_method = c("cpu_exact", "irlba", "cpu_exact", "irlba", "cpu_exact", "irlba", "cpu_exact", "irlba", "cuda_rsvd", "cuda_rsvd"),
  stringsAsFactors = FALSE
)

cuda_ok <- has_cuda()

safe_mb <- function(x) as.numeric(object.size(x)) / (1024^2)

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
    denom <- (2 * tp + fp + fn)
    f1[i] <- if (denom == 0) NA_real_ else (2 * tp / denom)
  }
  mean(f1, na.rm = TRUE)
}

balanced_accuracy <- function(truth, pred) {
  lv <- union(levels(truth), levels(pred))
  truth <- factor(truth, levels = lv)
  pred <- factor(pred, levels = lv)
  tab <- table(pred, truth)
  recalls <- numeric(length(lv))
  for (i in seq_along(lv)) {
    denom <- sum(tab[, i])
    recalls[i] <- if (denom == 0) NA_real_ else tab[i, i] / denom
  }
  mean(recalls, na.rm = TRUE)
}

extract_pred <- function(model_out) {
  yp <- model_out$Ypred
  if (is.data.frame(yp)) return(as.factor(yp[[1]]))
  if (is.matrix(yp)) return(as.factor(yp[, 1]))
  if (is.vector(yp)) return(as.factor(yp))
  if (length(dim(yp)) == 3) {
    # numeric output fallback
    mat <- yp[, , 1, drop = FALSE]
    cls <- apply(mat, 1, which.max)
    lev <- model_out$lev
    return(factor(lev[cls], levels = lev))
  }
  stop("Unsupported Ypred format")
}

fit_once <- function(cfg) {
  if (cfg$engine == "R") {
    out <- pls_r(
      Xtrain, Ytrain, Xtest,
      ncomp = ncomp,
      method = cfg$algorithm,
      svd.method = cfg$svd_method,
      scaling = "centering"
    )
  } else {
    out <- pls(
      Xtrain, Ytrain, Xtest,
      ncomp = ncomp,
      method = cfg$algorithm,
      svd.method = cfg$svd_method,
      scaling = "centering"
    )
  }
  out
}

run_cfg <- function(cfg) {
  if (cfg$svd_method == "cuda_rsvd" && !cuda_ok) {
    return(data.frame(
      id = cfg$id, name = cfg$name, engine = cfg$engine,
      algorithm = cfg$algorithm, svd_method = cfg$svd_method,
      status = "skipped_cuda_unavailable",
      elapsed_s = NA_real_, bench_median_ms = NA_real_, bench_itr_sec = NA_real_,
      mem_alloc_mb = NA_real_, gc_sec = NA_real_,
      model_size_mb = NA_real_, ypred_size_mb = NA_real_,
      accuracy = NA_real_, balanced_acc = NA_real_, macro_f1 = NA_real_,
      stringsAsFactors = FALSE
    ))
  }

  warn_txt <- character(0)
  err_txt <- NULL

  fit <- NULL
  elapsed <- NA_real_

  res <- tryCatch({
    gc()
    t0 <- proc.time()[3]
    fit <- withCallingHandlers(
      fit_once(cfg),
      warning = function(w) {
        warn_txt <<- c(warn_txt, conditionMessage(w))
        invokeRestart("muffleWarning")
      }
    )
    elapsed <- proc.time()[3] - t0
    NULL
  }, error = function(e) {
    err_txt <<- conditionMessage(e)
    NULL
  })

  if (!is.null(err_txt)) {
    return(data.frame(
      id = cfg$id, name = cfg$name, engine = cfg$engine,
      algorithm = cfg$algorithm, svd_method = cfg$svd_method,
      status = paste0("error: ", err_txt),
      elapsed_s = NA_real_, bench_median_ms = NA_real_, bench_itr_sec = NA_real_,
      mem_alloc_mb = NA_real_, gc_sec = NA_real_,
      model_size_mb = NA_real_, ypred_size_mb = NA_real_,
      accuracy = NA_real_, balanced_acc = NA_real_, macro_f1 = NA_real_,
      stringsAsFactors = FALSE
    ))
  }

  pred <- extract_pred(fit)
  acc <- mean(pred == Ytest)
  bacc <- balanced_accuracy(Ytest, pred)
  mf1 <- macro_f1(Ytest, pred)

  bench_res <- bench::mark(
    {
      o <- fit_once(cfg)
      invisible(o$Ypred)
    },
    iterations = reps,
    check = FALSE,
    memory = TRUE,
    time_unit = "ms"
  )

  status <- if (length(warn_txt)) paste(unique(warn_txt), collapse = " | ") else "ok"

  data.frame(
    id = cfg$id,
    name = cfg$name,
    engine = cfg$engine,
    algorithm = cfg$algorithm,
    svd_method = cfg$svd_method,
    status = status,
    elapsed_s = as.numeric(elapsed),
    bench_median_ms = as.numeric(bench_res$median),
    bench_itr_sec = as.numeric(bench_res$`itr/sec`),
    mem_alloc_mb = as.numeric(bench_res$mem_alloc) / (1024^2),
    gc_sec = as.numeric(bench_res$`gc/sec`),
    model_size_mb = safe_mb(fit),
    ypred_size_mb = safe_mb(fit$Ypred),
    accuracy = as.numeric(acc),
    balanced_acc = as.numeric(bacc),
    macro_f1 = as.numeric(mf1),
    stringsAsFactors = FALSE
  )
}

results <- do.call(rbind, lapply(seq_len(nrow(configs)), function(i) run_cfg(configs[i, ])))

# agreement versus reference (Rcpp_simpls_irlba)
ref_name <- "Rcpp_simpls_irlba"
if (ref_name %in% results$name && !is.na(results$accuracy[results$name == ref_name])) {
  ref_cfg <- configs[configs$name == ref_name, ]
  ref_fit <- fit_once(ref_cfg)
  ref_pred <- extract_pred(ref_fit)

  agreements <- rep(NA_real_, nrow(configs))
  for (i in seq_len(nrow(configs))) {
    cfg <- configs[i, ]
    if (cfg$svd_method == "cuda_rsvd" && !cuda_ok) next
    p <- tryCatch(extract_pred(fit_once(cfg)), error = function(e) NULL)
    if (!is.null(p)) agreements[i] <- mean(p == ref_pred)
  }
  results$agree_with_rcpp_simpls_irlba <- agreements
} else {
  results$agree_with_rcpp_simpls_irlba <- NA_real_
}

results <- results[order(results$id), ]

out_csv <- "/Users/stefano/Documents/GPUPLS/metref_benchmark_10options.csv"
write.csv(results, out_csv, row.names = FALSE)

cat("\nBenchmark complete. Results:\n")
print(results, row.names = FALSE)
cat("\nSaved CSV:", out_csv, "\n")
