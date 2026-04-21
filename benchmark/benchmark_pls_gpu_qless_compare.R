#!/usr/bin/env Rscript

root_dir <- Sys.getenv("FASTPLS_DATA_ROOT", "/home/chiamaka/Documents/fastpls/data")
out_dir <- Sys.getenv("FASTPLS_BENCH_OUT", file.path(getwd(), "benchmark_results_pls_gpu_qless"))
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

if (!requireNamespace("fastPLS", quietly = TRUE)) stop("fastPLS must be installed")
if (!requireNamespace("data.table", quietly = TRUE)) stop("data.table must be installed")

library(data.table)
set.seed(12345)

parse_int_list <- function(x, default) {
  x <- trimws(x)
  if (!nzchar(x)) return(default)
  vals <- suppressWarnings(as.integer(strsplit(x, ",", fixed = TRUE)[[1L]]))
  vals <- vals[is.finite(vals) & !is.na(vals) & vals > 0L]
  if (!length(vals)) return(default)
  unique(vals)
}

bench_ncomp <- parse_int_list(Sys.getenv("FASTPLS_BENCH_NCOMP_LIST", ""), c(50L))
bench_reps <- {
  v <- suppressWarnings(as.integer(Sys.getenv("FASTPLS_BENCH_REPS", "3")))
  if (!is.finite(v) || is.na(v) || v < 1L) 3L else v
}
bench_datasets <- {
  x <- trimws(Sys.getenv("FASTPLS_BENCH_DATASETS", ""))
  if (!nzchar(x)) character() else unique(trimws(strsplit(x, ",", fixed = TRUE)[[1L]]))
}

metric_accuracy <- function(truth, pred) {
  mean(as.character(truth) == as.character(pred), na.rm = TRUE)
}

make_stratified_split <- function(y, train_frac = 0.9) {
  y <- droplevels(as.factor(y))
  idx <- seq_along(y)
  by_class <- split(idx, y)
  train_idx <- unlist(lapply(by_class, function(ii) {
    n_train <- max(1L, floor(length(ii) * train_frac))
    sample(ii, n_train)
  }), use.names = FALSE)
  test_idx <- setdiff(idx, train_idx)
  list(train = sort(train_idx), test = sort(test_idx))
}

as_task <- function(path, dataset_id) {
  e <- new.env(parent = emptyenv())
  load(path, envir = e)
  objs <- ls(e)

  if (all(c("Xtrain", "Ytrain", "Xtest", "Ytest") %in% objs)) {
    return(list(
      dataset = dataset_id,
      Xtrain = as.matrix(get("Xtrain", envir = e)),
      Ytrain = {
        y <- get("Ytrain", envir = e)
        if (is.factor(y)) droplevels(y) else y
      },
      Xtest = as.matrix(get("Xtest", envir = e)),
      Ytest = {
        ytr <- get("Ytrain", envir = e)
        yte <- get("Ytest", envir = e)
        if (is.factor(yte)) factor(yte, levels = levels(ytr)) else yte
      }
    ))
  }

  if ("out" %in% objs && is.list(get("out", envir = e)) &&
      all(c("Xtrain", "Ytrain", "Xtest", "Ytest") %in% names(get("out", envir = e)))) {
    obj <- get("out", envir = e)
    return(list(
      dataset = dataset_id,
      Xtrain = as.matrix(obj$Xtrain),
      Ytrain = if (is.factor(obj$Ytrain)) droplevels(obj$Ytrain) else obj$Ytrain,
      Xtest = as.matrix(obj$Xtest),
      Ytest = if (is.factor(obj$Ytest)) factor(obj$Ytest, levels = levels(obj$Ytrain)) else obj$Ytest
    ))
  }

  if ("r" %in% objs && is.data.frame(e$r) && "label_idx" %in% colnames(e$r)) {
    dt <- as.data.table(get("r", envir = e))
    feat_cols <- grep("^feat_", names(dt), value = TRUE)
    if ("split" %in% names(dt)) {
      train_idx <- which(dt$split == "train")
      test_idx <- which(dt$split == "test")
    } else {
      sp <- make_stratified_split(dt$label_idx)
      train_idx <- sp$train
      test_idx <- sp$test
    }
    y_all <- factor(dt$label_idx)
    return(list(
      dataset = dataset_id,
      Xtrain = as.matrix(dt[train_idx, ..feat_cols]),
      Ytrain = droplevels(y_all[train_idx]),
      Xtest = as.matrix(dt[test_idx, ..feat_cols]),
      Ytest = factor(y_all[test_idx], levels = levels(y_all[train_idx]))
    ))
  }

  if (all(c("data", "labels") %in% objs)) {
    X <- as.matrix(get("data", envir = e))
    y <- droplevels(as.factor(get("labels", envir = e)))
    sp <- make_stratified_split(y)
    return(list(
      dataset = dataset_id,
      Xtrain = X[sp$train, , drop = FALSE],
      Ytrain = droplevels(y[sp$train]),
      Xtest = X[sp$test, , drop = FALSE],
      Ytest = factor(y[sp$test], levels = levels(y[sp$train]))
    ))
  }

  stop(sprintf("Unsupported dataset format for %s", dataset_id))
}

with_env <- function(vars, expr) {
  old <- Sys.getenv(names(vars), unset = NA_character_)
  names(old) <- names(vars)
  on.exit({
    for (nm in names(vars)) {
      if (is.na(old[[nm]])) {
        Sys.unsetenv(nm)
      } else {
        Sys.setenv(structure(old[[nm]], names = nm))
      }
    }
  }, add = TRUE)
  do.call(Sys.setenv, as.list(vars))
  force(expr)
}

run_one <- function(task, engine, rep_id, ncomp = 50L) {
  fit_obj <- NULL
  elapsed <- NA_real_
  pred <- NULL
  err_msg <- NA_character_

  train_call <- switch(
    engine,
    hybrid_cpu = function() fastPLS::pls(
      Xtrain = task$Xtrain, Ytrain = task$Ytrain,
      Xtest = task$Xtest, Ytest = task$Ytest,
      ncomp = ncomp, method = "simpls_fast", svd.method = "cpu_rsvd",
      fit = FALSE, seed = 12345L + rep_id
    ),
    gpu_explicit = function() with_env(
      c(FASTPLS_GPU_DEVICE_STATE = "0", FASTPLS_GPU_QLESS_QR = "0"),
      fastPLS::simpls_gpu(
        Xtrain = task$Xtrain, Ytrain = task$Ytrain,
        Xtest = task$Xtest, Ytest = task$Ytest,
        ncomp = ncomp, fit = FALSE, seed = 12345L + rep_id
      )
    ),
    gpu_qless = function() with_env(
      c(FASTPLS_GPU_DEVICE_STATE = "0", FASTPLS_GPU_QLESS_QR = "1"),
      fastPLS::simpls_gpu(
        Xtrain = task$Xtrain, Ytrain = task$Ytrain,
        Xtest = task$Xtest, Ytest = task$Ytest,
        ncomp = ncomp, fit = FALSE, seed = 12345L + rep_id
      )
    )
  )

  tryCatch({
    if (engine != "hybrid_cpu") invisible(train_call())
    elapsed <- system.time({ fit_obj <- train_call() })[["elapsed"]]
    pred <- fit_obj$Ypred[[1L]]
  }, error = function(err) {
    err_msg <<- conditionMessage(err)
  })

  data.frame(
    dataset = task$dataset,
    engine = engine,
    rep = rep_id,
    train_time_seconds = elapsed,
    accuracy = if (is.null(pred)) NA_real_ else metric_accuracy(task$Ytest, pred),
    ncomp = ncomp,
    p = ncol(task$Xtrain),
    train_n = nrow(task$Xtrain),
    test_n = nrow(task$Xtest),
    classes = if (is.factor(task$Ytrain)) nlevels(task$Ytrain) else ncol(as.matrix(task$Ytrain)),
    ok = is.na(err_msg),
    error = if (is.na(err_msg)) "" else err_msg,
    stringsAsFactors = FALSE
  )
}

dataset_files <- c(
  metref = file.path(root_dir, "metref.RData"),
  singlecell = file.path(root_dir, "singlecell.RData"),
  cifar100 = file.path(root_dir, "CIFAR100.RData"),
  gtex_v8 = file.path(root_dir, "gtex.RData"),
  tcga_pan_cancer = file.path(root_dir, "tcga_pan_cancer.RData"),
  ccle = file.path(root_dir, "ccle.RData")
)
if (length(bench_datasets)) {
  dataset_files <- dataset_files[names(dataset_files) %in% bench_datasets]
}
tasks <- lapply(names(dataset_files), function(nm) as_task(dataset_files[[nm]], nm))
names(tasks) <- names(dataset_files)

engines <- c("hybrid_cpu")
if (fastPLS::has_cuda()) {
  engines <- c(engines, "gpu_explicit", "gpu_qless")
}

results <- rbindlist(lapply(tasks, function(task) {
  rbindlist(lapply(bench_ncomp, function(ncomp_i) {
    rbindlist(lapply(engines, function(engine) {
      rbindlist(lapply(seq_len(bench_reps), function(rep_id) {
        as.data.table(run_one(task, engine, rep_id, ncomp = ncomp_i))
      }))
    }))
  }))
}), use.names = TRUE)

summary_dt <- results[, .(
  train_time_seconds_median = median(train_time_seconds, na.rm = TRUE),
  train_time_seconds_mean = mean(train_time_seconds, na.rm = TRUE),
  accuracy_median = median(accuracy, na.rm = TRUE),
  accuracy_mean = mean(accuracy, na.rm = TRUE),
  p = unique(p)[1],
  train_n = unique(train_n)[1],
  test_n = unique(test_n)[1],
  classes = unique(classes)[1],
  ok_runs = sum(ok, na.rm = TRUE),
  total_runs = .N,
  any_error = any(!ok),
  example_error = {
    errs <- unique(error[!ok & nzchar(error)])
    if (length(errs)) errs[1] else ""
  }
), by = .(dataset, engine, ncomp)]

compare_dt <- merge(
  summary_dt[engine == "gpu_explicit"],
  summary_dt[engine == "gpu_qless"],
  by = c("dataset", "ncomp"),
  suffixes = c("_explicit", "_qless")
)
compare_dt[, `:=`(
  speedup_qless_vs_explicit = train_time_seconds_median_explicit / train_time_seconds_median_qless,
  accuracy_diff_qless_minus_explicit = accuracy_median_qless - accuracy_median_explicit
)]

fwrite(results, file.path(out_dir, "pls_gpu_qless_raw.csv"))
fwrite(summary_dt, file.path(out_dir, "pls_gpu_qless_summary.csv"))
fwrite(compare_dt, file.path(out_dir, "pls_gpu_qless_compare.csv"))

sink(file.path(out_dir, "sessionInfo.txt"))
print(sessionInfo())
sink()

cat("Results written to:", out_dir, "\n")
