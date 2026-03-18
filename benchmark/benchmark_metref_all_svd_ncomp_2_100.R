suppressPackageStartupMessages({
  library(fastPLS)
  library(KODAMA)
  library(bench)
  library(data.table)
  library(ggplot2)
})

set.seed(123)

# -------------------------
# Config
# -------------------------
ncomp_grid <- 2:100
reps <- 1L
out_dir <- "metref_all_svd_ncomp_2_100"
dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)

# -------------------------
# Data prep
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

methods <- data.table(
  method_id = c(
    "Rcpp_simpls_irlba",
    "Rcpp_simpls_dc",
    "Rcpp_simpls_cpu_exact",
    "Rcpp_simpls_cpu_rsvd",
    "Rcpp_simpls_cuda_rsvd",
    "Rcpp_plssvd_irlba",
    "Rcpp_plssvd_dc",
    "Rcpp_plssvd_cpu_exact",
    "Rcpp_plssvd_cpu_rsvd",
    "Rcpp_plssvd_cuda_rsvd"
  ),
  algorithm = c(rep("simpls", 5), rep("plssvd", 5)),
  svd_method = rep(c("irlba", "dc", "cpu_exact", "cpu_rsvd", "cuda_rsvd"), 2)
)

cuda_ok <- has_cuda()

safe_mb <- function(x) as.numeric(object.size(x)) / (1024^2)

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

fit_once <- function(algorithm, svd_method, ncomp) {
  pls(
    Xtrain, Ytrain, Xtest,
    ncomp = ncomp,
    method = algorithm,
    svd.method = svd_method,
    scaling = "centering"
  )
}

rows <- vector("list", nrow(methods) * length(ncomp_grid))
k <- 1L

for (mi in seq_len(nrow(methods))) {
  alg <- methods$algorithm[mi]
  svd <- methods$svd_method[mi]
  mid <- methods$method_id[mi]
  cat(sprintf("Running %s (%d/%d)\n", mid, mi, nrow(methods)))

  for (nc in ncomp_grid) {
    if (svd == "cuda_rsvd" && !cuda_ok) {
      rows[[k]] <- data.table(
        method_id = mid, algorithm = alg, svd_method = svd, ncomp = nc,
        status = "skipped_cuda_unavailable",
        elapsed_s = NA_real_, bench_median_ms = NA_real_, bench_itr_sec = NA_real_,
        mem_alloc_mb = NA_real_, gc_sec = NA_real_,
        model_size_mb = NA_real_, ypred_size_mb = NA_real_,
        accuracy = NA_real_, balanced_acc = NA_real_, macro_f1 = NA_real_
      )
      k <- k + 1L
      next
    }

    warn_txt <- character(0)
    err <- NULL
    out <- NULL

    t0 <- proc.time()[3]
    tryCatch({
      out <- withCallingHandlers(
        fit_once(alg, svd, nc),
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
      rows[[k]] <- data.table(
        method_id = mid, algorithm = alg, svd_method = svd, ncomp = nc,
        status = paste0("error: ", err),
        elapsed_s = NA_real_, bench_median_ms = NA_real_, bench_itr_sec = NA_real_,
        mem_alloc_mb = NA_real_, gc_sec = NA_real_,
        model_size_mb = NA_real_, ypred_size_mb = NA_real_,
        accuracy = NA_real_, balanced_acc = NA_real_, macro_f1 = NA_real_
      )
      k <- k + 1L
      next
    }

    pred <- extract_pred(out)
    acc <- mean(pred == Ytest)
    bacc <- balanced_accuracy(Ytest, pred)
    mf1 <- macro_f1(Ytest, pred)

    b <- bench::mark(
      {
        o <- fit_once(alg, svd, nc)
        invisible(o$Ypred)
      },
      iterations = reps,
      check = FALSE,
      memory = TRUE,
      time_unit = "ms"
    )

    rows[[k]] <- data.table(
      method_id = mid,
      algorithm = alg,
      svd_method = svd,
      ncomp = nc,
      status = if (length(warn_txt)) paste(unique(warn_txt), collapse = " | ") else "ok",
      elapsed_s = as.numeric(elapsed),
      bench_median_ms = as.numeric(b$median),
      bench_itr_sec = as.numeric(b$`itr/sec`),
      mem_alloc_mb = as.numeric(b$mem_alloc) / (1024^2),
      gc_sec = as.numeric(b$`gc/sec`),
      model_size_mb = safe_mb(out),
      ypred_size_mb = safe_mb(out$Ypred),
      accuracy = as.numeric(acc),
      balanced_acc = as.numeric(bacc),
      macro_f1 = as.numeric(mf1)
    )

    k <- k + 1L
  }
}

res <- rbindlist(rows, fill = TRUE)
setorder(res, method_id, ncomp)

# agreement vs reference
res[, agree_with_ref := NA_real_]
ref_method <- "Rcpp_simpls_irlba"
for (nc in ncomp_grid) {
  ref_cfg <- methods[method_id == ref_method]
  ref_pred <- tryCatch(extract_pred(fit_once(ref_cfg$algorithm, ref_cfg$svd_method, nc)), error = function(e) NULL)
  if (is.null(ref_pred)) next
  for (mi in seq_len(nrow(methods))) {
    cfg <- methods[mi]
    if (cfg$svd_method == "cuda_rsvd" && !cuda_ok) next
    p <- tryCatch(extract_pred(fit_once(cfg$algorithm, cfg$svd_method, nc)), error = function(e) NULL)
    if (!is.null(p)) {
      res[method_id == cfg$method_id & ncomp == nc, agree_with_ref := mean(p == ref_pred)]
    }
  }
}

fwrite(res, file.path(out_dir, "benchmark_results.csv"))

plot_metric <- function(df, ycol, ylab, filename) {
  p <- ggplot(df[!grepl('^error', status) & status != 'skipped_cuda_unavailable'],
              aes(x = ncomp, y = get(ycol), color = method_id)) +
    geom_line(linewidth = 0.6) +
    theme_minimal(base_size = 11) +
    labs(x = "ncomp", y = ylab, color = "method")
  ggsave(file.path(out_dir, filename), p, width = 12, height = 6, dpi = 150)
}

plot_metric(res, "bench_median_ms", "Time (ms)", "time_vs_ncomp.png")
plot_metric(res, "mem_alloc_mb", "Memory alloc (MB)", "memory_alloc_vs_ncomp.png")
plot_metric(res, "model_size_mb", "Model size (MB)", "model_size_vs_ncomp.png")
plot_metric(res, "accuracy", "Accuracy", "accuracy_vs_ncomp.png")
plot_metric(res, "balanced_acc", "Balanced accuracy", "balanced_accuracy_vs_ncomp.png")
plot_metric(res, "macro_f1", "Macro F1", "macro_f1_vs_ncomp.png")
plot_metric(res, "agree_with_ref", "Agreement vs Rcpp_simpls_irlba", "agreement_vs_ref_vs_ncomp.png")

summary_tab <- res[
  !grepl('^error', status) & status != 'skipped_cuda_unavailable',
  .(
    time_ms_median = median(bench_median_ms, na.rm = TRUE),
    time_ms_min = min(bench_median_ms, na.rm = TRUE),
    mem_alloc_mb_median = median(mem_alloc_mb, na.rm = TRUE),
    best_accuracy = max(accuracy, na.rm = TRUE),
    best_macro_f1 = max(macro_f1, na.rm = TRUE)
  ),
  by = .(method_id)
][order(time_ms_median)]

fwrite(summary_tab, file.path(out_dir, "summary_by_method.csv"))

cat("\nDone. Output directory:\n", normalizePath(out_dir), "\n", sep = "")
cat("CUDA available:", cuda_ok, "\n")
