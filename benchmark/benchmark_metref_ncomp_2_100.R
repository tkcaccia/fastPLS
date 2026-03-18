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
out_dir <- "metref_benchmark_ncomp_2_100"
dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)

# -------------------------
# Data prep (same as requested)
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

cuda_ok <- has_cuda()

methods <- data.table(
  method_id = c(
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
  engine = c(rep("R", 4), rep("Rcpp", 6)),
  algorithm = c("simpls", "simpls", "plssvd", "plssvd", "simpls", "simpls", "plssvd", "plssvd", "simpls", "plssvd"),
  svd_method = c("cpu_exact", "irlba", "cpu_exact", "irlba", "cpu_exact", "irlba", "cpu_exact", "irlba", "cuda_rsvd", "cuda_rsvd")
)

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
    denom <- sum(tab[, i])
    recalls[i] <- if (denom == 0) NA_real_ else tab[i, i] / denom
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
    denom <- (2 * tp + fp + fn)
    f1[i] <- if (denom == 0) NA_real_ else (2 * tp / denom)
  }
  mean(f1, na.rm = TRUE)
}

fit_once <- function(cfg, ncomp) {
  if (cfg$engine == "R") {
    pls_r(
      Xtrain, Ytrain, Xtest,
      ncomp = ncomp,
      method = cfg$algorithm,
      svd.method = cfg$svd_method,
      scaling = "centering"
    )
  } else {
    pls(
      Xtrain, Ytrain, Xtest,
      ncomp = ncomp,
      method = cfg$algorithm,
      svd.method = cfg$svd_method,
      scaling = "centering"
    )
  }
}

rows <- vector("list", length = nrow(methods) * length(ncomp_grid))
k <- 1L

for (mi in seq_len(nrow(methods))) {
  cfg <- methods[mi]
  cat(sprintf("Running %s (%d/%d)\n", cfg$method_id, mi, nrow(methods)))

  for (nc in ncomp_grid) {
    if (cfg$svd_method == "cuda_rsvd" && !cuda_ok) {
      rows[[k]] <- data.table(
        method_id = cfg$method_id,
        engine = cfg$engine,
        algorithm = cfg$algorithm,
        svd_method = cfg$svd_method,
        ncomp = nc,
        status = "skipped_cuda_unavailable",
        elapsed_s = NA_real_,
        bench_median_ms = NA_real_,
        bench_itr_sec = NA_real_,
        mem_alloc_mb = NA_real_,
        gc_sec = NA_real_,
        model_size_mb = NA_real_,
        ypred_size_mb = NA_real_,
        accuracy = NA_real_,
        balanced_acc = NA_real_,
        macro_f1 = NA_real_
      )
      k <- k + 1L
      next
    }

    warn_txt <- character(0)
    err_txt <- NULL
    fit <- NULL
    elapsed <- NA_real_

    tryCatch({
      gc()
      t0 <- proc.time()[3]
      fit <- withCallingHandlers(
        fit_once(cfg, nc),
        warning = function(w) {
          warn_txt <<- c(warn_txt, conditionMessage(w))
          invokeRestart("muffleWarning")
        }
      )
      elapsed <- proc.time()[3] - t0
    }, error = function(e) {
      err_txt <<- conditionMessage(e)
    })

    if (!is.null(err_txt)) {
      rows[[k]] <- data.table(
        method_id = cfg$method_id,
        engine = cfg$engine,
        algorithm = cfg$algorithm,
        svd_method = cfg$svd_method,
        ncomp = nc,
        status = paste0("error: ", err_txt),
        elapsed_s = NA_real_,
        bench_median_ms = NA_real_,
        bench_itr_sec = NA_real_,
        mem_alloc_mb = NA_real_,
        gc_sec = NA_real_,
        model_size_mb = NA_real_,
        ypred_size_mb = NA_real_,
        accuracy = NA_real_,
        balanced_acc = NA_real_,
        macro_f1 = NA_real_
      )
      k <- k + 1L
      next
    }

    pred <- extract_pred(fit)
    acc <- mean(pred == Ytest)
    bacc <- balanced_accuracy(Ytest, pred)
    mf1 <- macro_f1(Ytest, pred)

    b <- bench::mark(
      {
        o <- fit_once(cfg, nc)
        invisible(o$Ypred)
      },
      iterations = reps,
      check = FALSE,
      memory = TRUE,
      time_unit = "ms"
    )

    status <- if (length(warn_txt)) paste(unique(warn_txt), collapse = " | ") else "ok"

    rows[[k]] <- data.table(
      method_id = cfg$method_id,
      engine = cfg$engine,
      algorithm = cfg$algorithm,
      svd_method = cfg$svd_method,
      ncomp = nc,
      status = status,
      elapsed_s = as.numeric(elapsed),
      bench_median_ms = as.numeric(b$median),
      bench_itr_sec = as.numeric(b$`itr/sec`),
      mem_alloc_mb = as.numeric(b$mem_alloc) / (1024^2),
      gc_sec = as.numeric(b$`gc/sec`),
      model_size_mb = safe_mb(fit),
      ypred_size_mb = safe_mb(fit$Ypred),
      accuracy = as.numeric(acc),
      balanced_acc = as.numeric(bacc),
      macro_f1 = as.numeric(mf1)
    )

    k <- k + 1L
  }
}

res <- rbindlist(rows, fill = TRUE)
setorder(res, method_id, ncomp)

# Agreement vs reference method (if present and runnable)
res[, agree_with_ref := NA_real_]
ref_method <- "Rcpp_simpls_irlba"
if (ref_method %in% res$method_id) {
  for (nc in ncomp_grid) {
    ref_cfg <- methods[method_id == ref_method]
    if (nrow(ref_cfg) == 1) {
      ref_pred <- tryCatch(extract_pred(fit_once(ref_cfg, nc)), error = function(e) NULL)
      if (!is.null(ref_pred)) {
        for (mi in unique(res$method_id)) {
          cfg <- methods[method_id == mi]
          if (nrow(cfg) != 1) next
          if (cfg$svd_method == "cuda_rsvd" && !cuda_ok) next
          p <- tryCatch(extract_pred(fit_once(cfg, nc)), error = function(e) NULL)
          if (!is.null(p)) {
            res[method_id == mi & ncomp == nc, agree_with_ref := mean(p == ref_pred)]
          }
        }
      }
    }
  }
}

fwrite(res, file.path(out_dir, "benchmark_results.csv"))

# -------------------------
# Plots
# -------------------------
plot_metric <- function(df, ycol, ylab, filename) {
  p <- ggplot(df[status == "ok"], aes(x = ncomp, y = get(ycol), color = method_id)) +
    geom_line(linewidth = 0.6) +
    geom_point(size = 0.7) +
    theme_minimal(base_size = 11) +
    labs(x = "ncomp", y = ylab, color = "method")
  ggsave(file.path(out_dir, filename), p, width = 12, height = 6, dpi = 150)
}

plot_metric(res, "bench_median_ms", "Time (ms, median benchmark)", "time_vs_ncomp.png")
plot_metric(res, "mem_alloc_mb", "Memory allocated (MB)", "memory_alloc_vs_ncomp.png")
plot_metric(res, "model_size_mb", "Model size (MB)", "model_size_vs_ncomp.png")
plot_metric(res, "accuracy", "Accuracy", "accuracy_vs_ncomp.png")
plot_metric(res, "balanced_acc", "Balanced Accuracy", "balanced_accuracy_vs_ncomp.png")
plot_metric(res, "macro_f1", "Macro F1", "macro_f1_vs_ncomp.png")
plot_metric(res, "agree_with_ref", "Agreement vs Rcpp_simpls_irlba", "agreement_vs_ref_vs_ncomp.png")

# Combined facet plot
long <- melt(
  res[status == "ok", .(method_id, ncomp, bench_median_ms, mem_alloc_mb, accuracy, balanced_acc, macro_f1)],
  id.vars = c("method_id", "ncomp"),
  variable.name = "metric",
  value.name = "value"
)

p_all <- ggplot(long, aes(x = ncomp, y = value, color = method_id)) +
  geom_line(linewidth = 0.5) +
  facet_wrap(~ metric, scales = "free_y", ncol = 2) +
  theme_minimal(base_size = 10) +
  labs(x = "ncomp", y = "value", color = "method")

ggsave(file.path(out_dir, "all_metrics_faceted.png"), p_all, width = 14, height = 10, dpi = 150)

summary_tab <- res[
  status == "ok",
  .(
    time_ms_median_over_grid = median(bench_median_ms, na.rm = TRUE),
    time_ms_min_over_grid = min(bench_median_ms, na.rm = TRUE),
    mem_alloc_mb_median_over_grid = median(mem_alloc_mb, na.rm = TRUE),
    best_accuracy = max(accuracy, na.rm = TRUE),
    best_macro_f1 = max(macro_f1, na.rm = TRUE)
  ),
  by = .(method_id)
][order(time_ms_median_over_grid)]

fwrite(summary_tab, file.path(out_dir, "summary_by_method.csv"))

cat("\nDone. Output directory:\n", normalizePath(out_dir), "\n", sep = "")
cat("CUDA available:", cuda_ok, "\n")
