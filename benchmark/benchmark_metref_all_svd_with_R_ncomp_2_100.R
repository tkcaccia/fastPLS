suppressPackageStartupMessages({
  library(fastPLS)
  library(KODAMA)
  library(bench)
  library(data.table)
  library(ggplot2)
})

set.seed(123)

ncomp_grid <- 2:100
reps <- as.integer(Sys.getenv("FASTPLS_BENCH_REPS", "5"))
if (is.na(reps) || reps < 1L) reps <- 5L

out_dir <- "metref_all_svd_with_R_ncomp_2_100"
dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)
dir.create(file.path(out_dir, "subanalysis"), showWarnings = FALSE, recursive = TRUE)

# MetRef prep
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
plssvd_cap <- min(nrow(Xtrain), ncol(Xtrain), nlevels(Ytrain))

cuda_ok <- has_cuda()
has_nvidia_smi <- nzchar(Sys.which("nvidia-smi"))

# Rcpp engine supports all 5; pure R supports 4 (no cuda_rsvd)
methods <- rbindlist(list(
  data.table(
    engine = "Rcpp",
    algorithm = rep(c("simpls", "plssvd"), each = 5),
    svd_method = rep(c("irlba", "arpack", "cpu_exact", "cpu_rsvd", "cuda_rsvd"), 2)
  ),
  data.table(
    engine = "R",
    algorithm = rep(c("simpls", "plssvd"), each = 4),
    svd_method = rep(c("irlba", "arpack", "cpu_exact", "cpu_rsvd"), 2)
  )
))
methods[, method_id := paste(engine, algorithm, svd_method, sep = "_")]

safe_mb <- function(x) as.numeric(object.size(x)) / (1024^2)

gpu_mem_used_mb <- function() {
  if (!has_nvidia_smi) return(NA_real_)
  out <- tryCatch(
    system2(
      "nvidia-smi",
      c("--query-gpu=memory.used", "--format=csv,noheader,nounits"),
      stdout = TRUE,
      stderr = FALSE
    ),
    error = function(e) character(0)
  )
  if (!length(out)) return(NA_real_)
  vals <- suppressWarnings(as.numeric(trimws(out)))
  vals <- vals[is.finite(vals)]
  if (!length(vals)) return(NA_real_)
  max(vals)
}

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

fit_once <- function(engine, algorithm, svd_method, ncomp) {
  if (engine == "Rcpp") {
    pls(
      Xtrain, Ytrain, Xtest,
      ncomp = ncomp,
      method = algorithm,
      svd.method = svd_method,
      scaling = "centering"
    )
  } else {
    pls_r(
      Xtrain, Ytrain, Xtest,
      ncomp = ncomp,
      method = algorithm,
      svd.method = svd_method,
      scaling = "centering"
    )
  }
}

timed_fit_reps <- function(engine, algorithm, svd_method, ncomp, reps) {
  elapsed_ms <- rep(NA_real_, reps)
  gpu_before <- rep(NA_real_, reps)
  gpu_after <- rep(NA_real_, reps)
  gpu_delta <- rep(NA_real_, reps)
  warn_txt <- character(0)
  err <- NULL
  model_last <- NULL

  for (r in seq_len(reps)) {
    gc(FALSE)
    gpu_before[r] <- gpu_mem_used_mb()
    t0 <- proc.time()[3]

    out <- NULL
    tryCatch({
      out <- withCallingHandlers(
        fit_once(engine, algorithm, svd_method, ncomp),
        warning = function(w) {
          warn_txt <<- c(warn_txt, conditionMessage(w))
          invokeRestart("muffleWarning")
        }
      )
    }, error = function(e) {
      err <<- conditionMessage(e)
    })

    elapsed_ms[r] <- (proc.time()[3] - t0) * 1000
    gpu_after[r] <- gpu_mem_used_mb()
    gpu_delta[r] <- gpu_after[r] - gpu_before[r]

    if (!is.null(err)) {
      return(list(err = err, warn_txt = warn_txt, model = NULL,
                  elapsed_ms = elapsed_ms, gpu_before = gpu_before,
                  gpu_after = gpu_after, gpu_delta = gpu_delta))
    }
    model_last <- out
  }

  list(err = NULL, warn_txt = warn_txt, model = model_last,
       elapsed_ms = elapsed_ms, gpu_before = gpu_before,
       gpu_after = gpu_after, gpu_delta = gpu_delta)
}

rows <- vector("list", nrow(methods) * length(ncomp_grid))
k <- 1L

for (mi in seq_len(nrow(methods))) {
  eng <- methods$engine[mi]
  alg <- methods$algorithm[mi]
  svd <- methods$svd_method[mi]
  mid <- methods$method_id[mi]

  cat(sprintf("Running %s (%d/%d)\n", mid, mi, nrow(methods)))

  for (nc in ncomp_grid) {
    if (alg == "plssvd" && nc > plssvd_cap) {
      rows[[k]] <- data.table(
        method_id = mid, engine = eng, algorithm = alg, svd_method = svd, ncomp = nc,
        status = "skipped_ncomp_above_plssvd_cap",
        time_ms_median_reps = NA_real_, time_ms_mean_reps = NA_real_,
        bench_median_ms = NA_real_, bench_itr_sec = NA_real_,
        mem_alloc_mb = NA_real_, gc_sec = NA_real_,
        gpu_mem_before_mb_median = NA_real_, gpu_mem_after_mb_median = NA_real_,
        gpu_mem_delta_mb_median = NA_real_,
        model_size_mb = NA_real_, ypred_size_mb = NA_real_,
        accuracy = NA_real_, balanced_acc = NA_real_, macro_f1 = NA_real_
      )
      k <- k + 1L
      next
    }

    if (svd == "cuda_rsvd" && !cuda_ok) {
      rows[[k]] <- data.table(
        method_id = mid, engine = eng, algorithm = alg, svd_method = svd, ncomp = nc,
        status = "skipped_cuda_unavailable",
        time_ms_median_reps = NA_real_, time_ms_mean_reps = NA_real_,
        bench_median_ms = NA_real_, bench_itr_sec = NA_real_,
        mem_alloc_mb = NA_real_, gc_sec = NA_real_,
        gpu_mem_before_mb_median = NA_real_, gpu_mem_after_mb_median = NA_real_,
        gpu_mem_delta_mb_median = NA_real_,
        model_size_mb = NA_real_, ypred_size_mb = NA_real_,
        accuracy = NA_real_, balanced_acc = NA_real_, macro_f1 = NA_real_
      )
      k <- k + 1L
      next
    }

    run <- timed_fit_reps(eng, alg, svd, nc, reps)

    if (!is.null(run$err)) {
      rows[[k]] <- data.table(
        method_id = mid, engine = eng, algorithm = alg, svd_method = svd, ncomp = nc,
        status = paste0("error: ", run$err),
        time_ms_median_reps = NA_real_, time_ms_mean_reps = NA_real_,
        bench_median_ms = NA_real_, bench_itr_sec = NA_real_,
        mem_alloc_mb = NA_real_, gc_sec = NA_real_,
        gpu_mem_before_mb_median = median(run$gpu_before, na.rm = TRUE),
        gpu_mem_after_mb_median = median(run$gpu_after, na.rm = TRUE),
        gpu_mem_delta_mb_median = median(run$gpu_delta, na.rm = TRUE),
        model_size_mb = NA_real_, ypred_size_mb = NA_real_,
        accuracy = NA_real_, balanced_acc = NA_real_, macro_f1 = NA_real_
      )
      k <- k + 1L
      next
    }

    out <- run$model
    pred <- extract_pred(out)
    acc <- mean(pred == Ytest)
    bacc <- balanced_accuracy(Ytest, pred)
    mf1 <- macro_f1(Ytest, pred)

    b <- bench::mark(
      {
        o <- fit_once(eng, alg, svd, nc)
        invisible(o$Ypred)
      },
      iterations = reps,
      check = FALSE,
      memory = TRUE,
      time_unit = "ms"
    )

    rows[[k]] <- data.table(
      method_id = mid,
      engine = eng,
      algorithm = alg,
      svd_method = svd,
      ncomp = nc,
      status = if (length(run$warn_txt)) paste(unique(run$warn_txt), collapse = " | ") else "ok",
      time_ms_median_reps = median(run$elapsed_ms, na.rm = TRUE),
      time_ms_mean_reps = mean(run$elapsed_ms, na.rm = TRUE),
      bench_median_ms = as.numeric(b$median),
      bench_itr_sec = as.numeric(b$`itr/sec`),
      mem_alloc_mb = as.numeric(b$mem_alloc) / (1024^2),
      gc_sec = as.numeric(b$`gc/sec`),
      gpu_mem_before_mb_median = median(run$gpu_before, na.rm = TRUE),
      gpu_mem_after_mb_median = median(run$gpu_after, na.rm = TRUE),
      gpu_mem_delta_mb_median = median(run$gpu_delta, na.rm = TRUE),
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

# agreement vs reference (Rcpp_simpls_irlba)
res[, agree_with_ref := NA_real_]
ref_method <- "Rcpp_simpls_irlba"
for (nc in ncomp_grid) {
  ref_cfg <- methods[method_id == ref_method]
  if (ref_cfg$algorithm == "plssvd" && nc > plssvd_cap) next
  ref_pred <- tryCatch(extract_pred(fit_once(ref_cfg$engine, ref_cfg$algorithm, ref_cfg$svd_method, nc)), error = function(e) NULL)
  if (is.null(ref_pred)) next
  for (mi in seq_len(nrow(methods))) {
    cfg <- methods[mi]
    if (cfg$svd_method == "cuda_rsvd" && !cuda_ok) next
    if (cfg$algorithm == "plssvd" && nc > plssvd_cap) next
    p <- tryCatch(extract_pred(fit_once(cfg$engine, cfg$algorithm, cfg$svd_method, nc)), error = function(e) NULL)
    if (!is.null(p)) {
      res[method_id == cfg$method_id & ncomp == nc, agree_with_ref := mean(p == ref_pred)]
    }
  }
}

fwrite(res, file.path(out_dir, "benchmark_results.csv"))

plot_metric <- function(df, ycol, ylab, filename) {
  d <- df[
    !grepl('^error', status) &
      status != 'skipped_cuda_unavailable' &
      status != 'skipped_ncomp_above_plssvd_cap'
  ]
  p <- ggplot(d, aes(x = ncomp, y = get(ycol), color = method_id)) +
    geom_line(linewidth = 0.6) +
    theme_minimal(base_size = 11) +
    labs(x = "ncomp", y = ylab, color = "method")
  ggsave(file.path(out_dir, filename), p, width = 13, height = 7, dpi = 150)
}

plot_subset <- function(df, subset_name, subset_expr) {
  d <- df[eval(subset_expr)]
  d <- d[
    !grepl('^error', status) &
      status != 'skipped_cuda_unavailable' &
      status != 'skipped_ncomp_above_plssvd_cap'
  ]
  if (!nrow(d)) return(invisible(NULL))

  mk <- function(ycol, ylab, suffix) {
    p <- ggplot(d, aes(x = ncomp, y = get(ycol), color = method_id)) +
      geom_line(linewidth = 0.7) +
      theme_minimal(base_size = 11) +
      labs(title = subset_name, x = "ncomp", y = ylab, color = "method")
    ggsave(
      file.path(out_dir, "subanalysis", paste0(gsub("[^A-Za-z0-9]+", "_", tolower(subset_name)), "_", suffix, ".png")),
      p, width = 12, height = 6, dpi = 150
    )
  }

  mk("time_ms_median_reps", "Time median over reps (ms)", "time")
  mk("mem_alloc_mb", "R memory alloc (MB)", "mem")
  mk("gpu_mem_delta_mb_median", "GPU memory delta median (MB)", "gpu_mem_delta")
  mk("accuracy", "Accuracy", "accuracy")
  mk("balanced_acc", "Balanced accuracy", "balanced_accuracy")
  mk("macro_f1", "Macro F1", "macro_f1")
}

plot_metric(res, "time_ms_median_reps", "Time median over reps (ms)", "time_vs_ncomp.png")
plot_metric(res, "mem_alloc_mb", "R memory alloc (MB)", "memory_alloc_vs_ncomp.png")
plot_metric(res, "gpu_mem_delta_mb_median", "GPU memory delta median (MB)", "gpu_memory_delta_vs_ncomp.png")
plot_metric(res, "model_size_mb", "Model size (MB)", "model_size_vs_ncomp.png")
plot_metric(res, "accuracy", "Accuracy", "accuracy_vs_ncomp.png")
plot_metric(res, "balanced_acc", "Balanced accuracy", "balanced_accuracy_vs_ncomp.png")
plot_metric(res, "macro_f1", "Macro F1", "macro_f1_vs_ncomp.png")
plot_metric(res, "agree_with_ref", "Agreement vs Rcpp_simpls_irlba", "agreement_vs_ref_vs_ncomp.png")

plot_subset(res, "R only", quote(engine == "R"))
plot_subset(res, "Rcpp only", quote(engine == "Rcpp"))
plot_subset(res, "CUDA only", quote(svd_method == "cuda_rsvd"))
plot_subset(res, "plssvd only", quote(algorithm == "plssvd"))
plot_subset(res, "simpls only", quote(algorithm == "simpls"))

summary_tab <- res[
  !grepl('^error', status) &
    status != 'skipped_cuda_unavailable' &
    status != 'skipped_ncomp_above_plssvd_cap',
  .(
    time_ms_median = median(time_ms_median_reps, na.rm = TRUE),
    time_ms_min = min(time_ms_median_reps, na.rm = TRUE),
    mem_alloc_mb_median = median(mem_alloc_mb, na.rm = TRUE),
    gpu_mem_delta_mb_median = median(gpu_mem_delta_mb_median, na.rm = TRUE),
    best_accuracy = max(accuracy, na.rm = TRUE),
    best_macro_f1 = max(macro_f1, na.rm = TRUE)
  ),
  by = .(method_id)
][order(time_ms_median)]

fwrite(summary_tab, file.path(out_dir, "summary_by_method.csv"))

cat("\nDone. Output directory:\n", normalizePath(out_dir), "\n", sep = "")
cat("CUDA available:", cuda_ok, "\n")
cat("nvidia-smi available:", has_nvidia_smi, "\n")
cat("Repetitions:", reps, "\n")
cat("PLSSVD cap:", plssvd_cap, "\n")
