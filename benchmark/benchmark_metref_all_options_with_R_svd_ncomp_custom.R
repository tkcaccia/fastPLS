suppressPackageStartupMessages({
  library(fastPLS)
  library(KODAMA)
  library(bench)
  library(data.table)
  library(ggplot2)
  library(pls)
})

set.seed(as.integer(Sys.getenv("FASTPLS_SEED", "123")))

ncomp_env <- Sys.getenv("FASTPLS_NCOMP_LIST", "2,5,10,15,20,22,30,50,100")
ncomp_grid <- as.integer(strsplit(ncomp_env, ",", fixed = TRUE)[[1]])
ncomp_grid <- sort(unique(ncomp_grid[is.finite(ncomp_grid) & ncomp_grid >= 1L]))
if (!length(ncomp_grid)) ncomp_grid <- c(2L, 5L, 10L, 15L, 20L, 22L, 30L, 50L, 100L)

reps <- as.integer(Sys.getenv("FASTPLS_BENCH_REPS", "10"))
if (is.na(reps) || reps < 1L) reps <- 10L

out_dir <- Sys.getenv("FASTPLS_OUTDIR", "metref_all_options_with_R_svd_ncomp_custom")
dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)
dir.create(file.path(out_dir, "subanalysis"), showWarnings = FALSE, recursive = TRUE)

# MetRef prep
data(MetRef, package = "KODAMA")
u <- MetRef$data
u <- u[, colSums(u) != 0, drop = FALSE]
u <- normalization(u)$newXtrain
class <- as.numeric(as.factor(MetRef$donor))

ss <- sample(nrow(u), 100)
Xtrain <- as.matrix(u[-ss, , drop = FALSE])
Ytrain <- as.factor(class)[-ss]
Xtest <- as.matrix(u[ss, , drop = FALSE])
Ytest <- as.factor(class)[ss]
plssvd_cap <- min(nrow(Xtrain), ncol(Xtrain), nlevels(Ytrain))

cuda_ok <- has_cuda()
has_nvidia_smi <- nzchar(Sys.which("nvidia-smi"))

methods <- rbindlist(list(
  data.table(
    engine = "Rcpp",
    algorithm = rep(c("simpls", "plssvd"), each = 4),
    svd_method = rep(c("irlba", "cpu_rsvd", "cuda_rsvd"), 2),
    fast_profile = "none"
  ),
  data.table(
    engine = "R",
    algorithm = rep(c("simpls", "plssvd"), each = 3),
    svd_method = rep(c("irlba", "cpu_rsvd"), 2),
    fast_profile = "none"
  ),
  data.table(
    engine = "Rcpp",
    algorithm = c("simpls_fast", "simpls_fast"),
    svd_method = c("fast_internal", "fast_internal"),
    fast_profile = c("default", "incdefl")
  ),
  data.table(
    engine = "pls_pkg",
    algorithm = "pls_pkg_simpls",
    svd_method = "N/A",
    fast_profile = "none"
  )
), fill = TRUE)

methods[, method_id := ifelse(
  algorithm == "simpls_fast",
  paste(engine, algorithm, fast_profile, sep = "_"),
  paste(engine, algorithm, svd_method, sep = "_")
)]

safe_mb <- function(x) as.numeric(object.size(x)) / (1024^2)

gpu_mem_used_mb <- function() {
  if (!has_nvidia_smi) return(NA_real_)
  out <- tryCatch(
    system2("nvidia-smi", c("--query-gpu=memory.used", "--format=csv,noheader,nounits"), stdout = TRUE, stderr = FALSE),
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

fit_pls_pkg <- function(ncomp) {
  class_lev <- levels(Ytrain)
  Ymm <- model.matrix(~ Ytrain - 1)
  colnames(Ymm) <- paste0("cls_", seq_len(ncol(Ymm)))
  x_names <- paste0("x_", seq_len(ncol(Xtrain)))
  Xtr <- Xtrain
  colnames(Xtr) <- x_names
  df_train <- data.frame(Ymm, Xtr, check.names = FALSE)
  form <- as.formula(paste0("cbind(", paste(colnames(Ymm), collapse = ","), ") ~ ."))
  mdl <- pls::plsr(form, data = df_train, ncomp = ncomp, method = "simpls", scale = FALSE, validation = "none")
  list(model = mdl, class_lev = class_lev, y_cols = colnames(Ymm), x_names = x_names)
}

predict_pls_pkg <- function(obj, ncomp) {
  Xte <- as.data.frame(Xtest)
  colnames(Xte) <- obj$x_names
  pa <- predict(obj$model, newdata = Xte, ncomp = ncomp)
  pm <- pa[, , 1, drop = FALSE]
  idx <- apply(pm, 1, which.max)
  raw <- obj$y_cols[idx]
  pred_num <- as.integer(sub("^cls_", "", raw))
  factor(obj$class_lev[pred_num], levels = obj$class_lev)
}

fit_once <- function(cfg, ncomp) {
  if (cfg$engine == "pls_pkg") return(fit_pls_pkg(ncomp))

  if (cfg$engine == "Rcpp") {
    if (cfg$algorithm == "simpls_fast") {
      if (cfg$fast_profile == "incdefl") {
        return(pls(Xtrain, Ytrain,
                   ncomp = ncomp,
                   method = "simpls_fast",
                   scaling = "centering",
                   fast_incremental = TRUE,
                   fast_inc_iters = 2L,
                   fast_defl_cache = TRUE,
                   fast_center_t = FALSE,
                   fast_reorth_v = FALSE,
                   fast_block = 8L))
      }
      return(pls(Xtrain, Ytrain,
                 ncomp = ncomp,
                 method = "simpls_fast",
                 scaling = "centering"))
    }
    return(pls(Xtrain, Ytrain,
               ncomp = ncomp,
               method = cfg$algorithm,
               svd.method = cfg$svd_method,
               scaling = "centering"))
  }

  pls_r(Xtrain, Ytrain,
        ncomp = ncomp,
        method = cfg$algorithm,
        svd.method = cfg$svd_method,
        scaling = "centering")
}

predict_once <- function(cfg, fit_obj, ncomp) {
  if (cfg$engine == "pls_pkg") return(predict_pls_pkg(fit_obj, ncomp))
  extract_pred(predict(fit_obj, Xtest, Ytest = Ytest, proj = FALSE))
}

timed_fit_reps <- function(cfg, ncomp, reps) {
  elapsed_ms <- rep(NA_real_, reps)
  gpu_before <- rep(NA_real_, reps)
  gpu_after <- rep(NA_real_, reps)
  gpu_delta <- rep(NA_real_, reps)
  warn_txt <- character(0)
  err <- NULL
  fit_last <- NULL

  for (r in seq_len(reps)) {
    gc(FALSE)
    gpu_before[r] <- gpu_mem_used_mb()
    t0 <- proc.time()[3]

    out <- NULL
    tryCatch({
      out <- withCallingHandlers(
        fit_once(cfg, ncomp),
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
      return(list(err = err, warn_txt = warn_txt, fit_obj = NULL,
                  elapsed_ms = elapsed_ms, gpu_before = gpu_before,
                  gpu_after = gpu_after, gpu_delta = gpu_delta))
    }
    fit_last <- out
  }

  list(err = NULL, warn_txt = warn_txt, fit_obj = fit_last,
       elapsed_ms = elapsed_ms, gpu_before = gpu_before,
       gpu_after = gpu_after, gpu_delta = gpu_delta)
}

rows <- vector("list", nrow(methods) * length(ncomp_grid))
k <- 1L

for (mi in seq_len(nrow(methods))) {
  cfg <- methods[mi]
  cat(sprintf("Running %s (%d/%d)\n", cfg$method_id, mi, nrow(methods)))

  for (nc in ncomp_grid) {
    if (cfg$algorithm == "plssvd" && nc > plssvd_cap) {
      rows[[k]] <- data.table(
        method_id = cfg$method_id, engine = cfg$engine, algorithm = cfg$algorithm,
        svd_method = cfg$svd_method, fast_profile = cfg$fast_profile, ncomp = nc,
        status = "skipped_ncomp_above_plssvd_cap",
        time_ms_median_reps = NA_real_, time_ms_mean_reps = NA_real_,
        bench_median_ms = NA_real_, bench_itr_sec = NA_real_,
        mem_alloc_mb = NA_real_, gc_sec = NA_real_,
        gpu_mem_before_mb_median = NA_real_, gpu_mem_after_mb_median = NA_real_,
        gpu_mem_delta_mb_median = NA_real_,
        model_size_mb = NA_real_, accuracy = NA_real_, balanced_acc = NA_real_, macro_f1 = NA_real_
      )
      k <- k + 1L
      next
    }

    if (cfg$svd_method == "cuda_rsvd" && !cuda_ok) {
      rows[[k]] <- data.table(
        method_id = cfg$method_id, engine = cfg$engine, algorithm = cfg$algorithm,
        svd_method = cfg$svd_method, fast_profile = cfg$fast_profile, ncomp = nc,
        status = "skipped_cuda_unavailable",
        time_ms_median_reps = NA_real_, time_ms_mean_reps = NA_real_,
        bench_median_ms = NA_real_, bench_itr_sec = NA_real_,
        mem_alloc_mb = NA_real_, gc_sec = NA_real_,
        gpu_mem_before_mb_median = NA_real_, gpu_mem_after_mb_median = NA_real_,
        gpu_mem_delta_mb_median = NA_real_,
        model_size_mb = NA_real_, accuracy = NA_real_, balanced_acc = NA_real_, macro_f1 = NA_real_
      )
      k <- k + 1L
      next
    }

    run <- timed_fit_reps(cfg, nc, reps)

    if (!is.null(run$err)) {
      rows[[k]] <- data.table(
        method_id = cfg$method_id, engine = cfg$engine, algorithm = cfg$algorithm,
        svd_method = cfg$svd_method, fast_profile = cfg$fast_profile, ncomp = nc,
        status = paste0("error: ", run$err),
        time_ms_median_reps = NA_real_, time_ms_mean_reps = NA_real_,
        bench_median_ms = NA_real_, bench_itr_sec = NA_real_,
        mem_alloc_mb = NA_real_, gc_sec = NA_real_,
        gpu_mem_before_mb_median = median(run$gpu_before, na.rm = TRUE),
        gpu_mem_after_mb_median = median(run$gpu_after, na.rm = TRUE),
        gpu_mem_delta_mb_median = median(run$gpu_delta, na.rm = TRUE),
        model_size_mb = NA_real_, accuracy = NA_real_, balanced_acc = NA_real_, macro_f1 = NA_real_
      )
      k <- k + 1L
      next
    }

    fit_obj <- run$fit_obj
    pred <- tryCatch(predict_once(cfg, fit_obj, nc), error = function(e) NULL)
    acc <- if (is.null(pred)) NA_real_ else mean(pred == Ytest)
    bacc <- if (is.null(pred)) NA_real_ else balanced_accuracy(Ytest, pred)
    mf1 <- if (is.null(pred)) NA_real_ else macro_f1(Ytest, pred)

    b <- tryCatch(
      bench::mark({
        o <- fit_once(cfg, nc)
        invisible(o)
      }, iterations = reps, check = FALSE, memory = TRUE, time_unit = "ms"),
      error = function(e) NULL
    )

    rows[[k]] <- data.table(
      method_id = cfg$method_id,
      engine = cfg$engine,
      algorithm = cfg$algorithm,
      svd_method = cfg$svd_method,
      fast_profile = cfg$fast_profile,
      ncomp = nc,
      status = if (length(run$warn_txt)) paste(unique(run$warn_txt), collapse = " | ") else "ok",
      time_ms_median_reps = median(run$elapsed_ms, na.rm = TRUE),
      time_ms_mean_reps = mean(run$elapsed_ms, na.rm = TRUE),
      bench_median_ms = if (is.null(b)) NA_real_ else as.numeric(b$median),
      bench_itr_sec = if (is.null(b)) NA_real_ else as.numeric(b$`itr/sec`),
      mem_alloc_mb = if (is.null(b)) NA_real_ else as.numeric(b$mem_alloc) / (1024^2),
      gc_sec = if (is.null(b)) NA_real_ else as.numeric(b$`gc/sec`),
      gpu_mem_before_mb_median = median(run$gpu_before, na.rm = TRUE),
      gpu_mem_after_mb_median = median(run$gpu_after, na.rm = TRUE),
      gpu_mem_delta_mb_median = median(run$gpu_delta, na.rm = TRUE),
      model_size_mb = safe_mb(fit_obj),
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
  if (!nrow(ref_cfg)) next
  if (ref_cfg$algorithm == "plssvd" && nc > plssvd_cap) next
  if (ref_cfg$svd_method == "cuda_rsvd" && !cuda_ok) next
  ref_fit <- tryCatch(fit_once(ref_cfg, nc), error = function(e) NULL)
  if (is.null(ref_fit)) next
  ref_pred <- tryCatch(predict_once(ref_cfg, ref_fit, nc), error = function(e) NULL)
  if (is.null(ref_pred)) next

  for (mi in seq_len(nrow(methods))) {
    cfg <- methods[mi]
    if (cfg$svd_method == "cuda_rsvd" && !cuda_ok) next
    if (cfg$algorithm == "plssvd" && nc > plssvd_cap) next
    pfit <- tryCatch(fit_once(cfg, nc), error = function(e) NULL)
    if (is.null(pfit)) next
    p <- tryCatch(predict_once(cfg, pfit, nc), error = function(e) NULL)
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
    geom_point(size = 1.4) +
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
      geom_point(size = 1.5) +
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

plot_metric(res, "time_ms_median_reps", "Model-building time median over reps (ms)", "time_vs_ncomp.png")
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
plot_subset(res, "simpls only", quote(algorithm %in% c("simpls", "simpls_fast")))
plot_subset(res, "pls package only", quote(engine == "pls_pkg"))
plot_subset(res, "fast simpls profiles", quote(algorithm == "simpls_fast"))

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
cat("ncomp grid:", paste(ncomp_grid, collapse = ","), "\n")
