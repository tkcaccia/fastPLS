#!/usr/bin/env Rscript

suppressPackageStartupMessages({
  library(fastPLS)
  library(data.table)
})

script_arg <- grep("^--file=", commandArgs(trailingOnly = FALSE), value = TRUE)
script_file <- if (length(script_arg)) sub("^--file=", "", script_arg[[1]]) else file.path(getwd(), "benchmark", "hpc_full_multianalysis_benchmark.R")
script_dir <- dirname(normalizePath(script_file, winslash = "/", mustWork = FALSE))
repo_root <- normalizePath(file.path(script_dir, ".."), winslash = "/", mustWork = FALSE)
default_data_root <- if (dir.exists("/Users/stefano/HPC-firenze/image_analysis/dinoV2/Rdatasets")) {
  "/Users/stefano/HPC-firenze/image_analysis/dinoV2/Rdatasets"
} else if (dir.exists(file.path(repo_root, "Rdataset"))) {
  file.path(repo_root, "Rdataset")
} else if (dir.exists(file.path(repo_root, "Data"))) {
  file.path(repo_root, "Data")
} else {
  "/scratch/firenze/image_analysis/dinoV2/Rdatasets"
}
default_out_root <- if (dir.exists(repo_root)) {
  file.path(repo_root, "benchmark_results_local")
} else {
  file.path(default_data_root, "benchmark_results")
}

set.seed(as.integer(Sys.getenv("FASTPLS_SEED", "123")))
threads <- as.integer(Sys.getenv("FASTPLS_THREADS", "1"))
if (!is.finite(threads) || is.na(threads) || threads < 1L) threads <- 1L
Sys.setenv(
  OMP_NUM_THREADS = as.character(threads),
  OPENBLAS_NUM_THREADS = as.character(threads),
  MKL_NUM_THREADS = as.character(threads),
  VECLIB_MAXIMUM_THREADS = as.character(threads),
  NUMEXPR_NUM_THREADS = as.character(threads)
)
if (requireNamespace("RhpcBLASctl", quietly = TRUE)) {
  RhpcBLASctl::blas_set_num_threads(threads)
  RhpcBLASctl::omp_set_num_threads(threads)
}

base_dir <- path.expand(Sys.getenv("FASTPLS_DATA_ROOT", default_data_root))
out_dir <- path.expand(Sys.getenv("FASTPLS_MULTI_OUTDIR", file.path(default_out_root, "multianalysis")))
out_csv <- file.path(out_dir, "multianalysis_raw.csv")
out_log <- file.path(out_dir, "multianalysis_progress.log")
append_mode <- tolower(Sys.getenv("FASTPLS_MULTI_APPEND", "false")) %in% c("1", "true", "yes", "y")

reps <- as.integer(Sys.getenv("FASTPLS_REPS", "3"))
if (!is.finite(reps) || is.na(reps) || reps < 1L) reps <- 3L
nmr_reps <- as.integer(Sys.getenv("FASTPLS_NMR_REPS", "1"))
if (!is.finite(nmr_reps) || is.na(nmr_reps) || nmr_reps < 1L) nmr_reps <- 1L
metref_reps <- as.integer(Sys.getenv("FASTPLS_METREF_REPS", "10"))
if (!is.finite(metref_reps) || is.na(metref_reps) || metref_reps < 1L) metref_reps <- 10L
singlecell_reps <- as.integer(Sys.getenv("FASTPLS_SINGLECELL_REPS", "5"))
if (!is.finite(singlecell_reps) || is.na(singlecell_reps) || singlecell_reps < 1L) singlecell_reps <- 5L
cifar100_reps <- as.integer(Sys.getenv("FASTPLS_CIFAR100_REPS", "5"))
if (!is.finite(cifar100_reps) || is.na(cifar100_reps) || cifar100_reps < 1L) cifar100_reps <- 5L

ncomp_vec <- as.integer(strsplit(Sys.getenv("FASTPLS_NCOMP_LIST", "2,5,10,20,50,100"), ",", fixed = TRUE)[[1]])
ncomp_vec <- sort(unique(ncomp_vec[is.finite(ncomp_vec) & ncomp_vec >= 1L]))
if (!length(ncomp_vec)) ncomp_vec <- c(2L, 5L, 10L, 20L, 50L, 100L)
ncomp_nmr_vec <- as.integer(strsplit(Sys.getenv("FASTPLS_NMR_NCOMP_LIST", "2,3,5"), ",", fixed = TRUE)[[1]])
ncomp_nmr_vec <- sort(unique(ncomp_nmr_vec[is.finite(ncomp_nmr_vec) & ncomp_nmr_vec >= 1L]))
if (!length(ncomp_nmr_vec)) ncomp_nmr_vec <- c(2L, 3L, 5L)
ncomp_singlecell_vec <- as.integer(strsplit(Sys.getenv("FASTPLS_SINGLECELL_NCOMP_LIST", "2,5,10,20,50"), ",", fixed = TRUE)[[1]])
ncomp_singlecell_vec <- sort(unique(ncomp_singlecell_vec[is.finite(ncomp_singlecell_vec) & ncomp_singlecell_vec >= 1L]))
if (!length(ncomp_singlecell_vec)) ncomp_singlecell_vec <- c(2L, 5L, 10L, 20L, 50L)
imagenet_train_n <- as.integer(Sys.getenv("FASTPLS_IMAGENET_TRAIN_N", "1000000"))
if (!is.finite(imagenet_train_n) || is.na(imagenet_train_n) || imagenet_train_n < 1L) imagenet_train_n <- 1000000L
default_ncomp <- as.integer(Sys.getenv("FASTPLS_DEFAULT_NCOMP", "2"))
if (!is.finite(default_ncomp) || is.na(default_ncomp) || default_ncomp < 1L) default_ncomp <- 2L
metref_default_ncomp <- as.integer(Sys.getenv("FASTPLS_METREF_DEFAULT_NCOMP", "20"))
if (!is.finite(metref_default_ncomp) || is.na(metref_default_ncomp) || metref_default_ncomp < 1L) metref_default_ncomp <- 20L
singlecell_default_ncomp <- as.integer(Sys.getenv("FASTPLS_SINGLECELL_DEFAULT_NCOMP", "50"))
if (!is.finite(singlecell_default_ncomp) || is.na(singlecell_default_ncomp) || singlecell_default_ncomp < 1L) singlecell_default_ncomp <- 50L
cifar100_default_ncomp <- as.integer(Sys.getenv("FASTPLS_CIFAR100_DEFAULT_NCOMP", "100"))
if (!is.finite(cifar100_default_ncomp) || is.na(cifar100_default_ncomp) || cifar100_default_ncomp < 1L) cifar100_default_ncomp <- 100L

sample_fracs <- as.numeric(strsplit(Sys.getenv("FASTPLS_SAMPLE_FRACS", "0.33,0.66,1.0"), ",", fixed = TRUE)[[1]])
sample_fracs <- sort(unique(sample_fracs[is.finite(sample_fracs) & sample_fracs > 0 & sample_fracs <= 1]))
if (!length(sample_fracs)) sample_fracs <- c(0.33, 0.66, 1.0)

xvar_fracs <- as.numeric(strsplit(Sys.getenv("FASTPLS_XVAR_FRACS", "0.10,0.20,0.50,1.0"), ",", fixed = TRUE)[[1]])
xvar_fracs <- sort(unique(xvar_fracs[is.finite(xvar_fracs) & xvar_fracs > 0 & xvar_fracs <= 1]))
if (!length(xvar_fracs)) xvar_fracs <- c(0.10, 0.20, 0.50, 1.0)

yvar_fracs <- as.numeric(strsplit(Sys.getenv("FASTPLS_YVAR_FRACS", "0.10,0.20,0.50,1.0"), ",", fixed = TRUE)[[1]])
yvar_fracs <- sort(unique(yvar_fracs[is.finite(yvar_fracs) & yvar_fracs > 0 & yvar_fracs <= 1]))
if (!length(yvar_fracs)) yvar_fracs <- c(0.10, 0.20, 0.50, 1.0)

irlba_svtol <- as.numeric(Sys.getenv("FASTPLS_IRLBA_SVTOL", "1e-6"))
if (!is.finite(irlba_svtol) || is.na(irlba_svtol) || irlba_svtol < 0) irlba_svtol <- 1e-6
rsvd_tol <- as.numeric(Sys.getenv("FASTPLS_RSVD_TOL", "0"))
if (!is.finite(rsvd_tol) || is.na(rsvd_tol) || rsvd_tol < 0) rsvd_tol <- 0

include_r_impl <- tolower(Sys.getenv("FASTPLS_INCLUDE_R_IMPL", "false")) %in% c("1", "true", "yes", "y")
include_cuda <- tolower(Sys.getenv("FASTPLS_INCLUDE_CUDA", "true")) %in% c("1", "true", "yes", "y")
include_simpls_fast_incremental <- tolower(Sys.getenv("FASTPLS_INCLUDE_SIMPLS_FAST_INCREMENTAL", "true")) %in% c("1", "true", "yes", "y")
metref_include_r <- tolower(Sys.getenv("FASTPLS_METREF_INCLUDE_R", "true")) %in% c("1", "true", "yes", "y")
metref_include_pls_pkg <- tolower(Sys.getenv("FASTPLS_METREF_INCLUDE_PLS_PKG", "true")) %in% c("1", "true", "yes", "y")
skip_arpack_on_nmr <- tolower(Sys.getenv("FASTPLS_SKIP_ARPACK_ON_NMR", "true")) %in% c("1", "true", "yes", "y")
skip_plssvd_on_nmr <- tolower(Sys.getenv("FASTPLS_SKIP_PLSSVD_ON_NMR", "true")) %in% c("1", "true", "yes", "y")
dataset_filter <- tolower(trimws(Sys.getenv("FASTPLS_DATASETS", "metref,cifar100,nmr,singlecell,imagenet")))
dataset_filter <- unlist(strsplit(dataset_filter, ",", fixed = TRUE), use.names = FALSE)
dataset_filter <- dataset_filter[nzchar(dataset_filter)]
if (!length(dataset_filter)) dataset_filter <- c("metref", "cifar100", "nmr", "singlecell", "imagenet")

stamp <- function() format(Sys.time(), "%Y-%m-%d %H:%M:%S")
log_msg <- function(...) {
  msg <- paste0("[", stamp(), "] ", paste(..., collapse = ""))
  cat(msg, "\n")
  cat(msg, "\n", file = out_log, append = TRUE)
  flush.console()
}

filter_call_args <- function(fun, args) {
  nms <- names(args)
  if (is.null(nms)) return(list())
  keep <- (!is.na(nms)) & nzchar(nms) & (nms %in% names(formals(fun)))
  args[keep]
}

safe_factor <- function(y) {
  if (is.factor(y)) return(y)
  if (is.data.frame(y)) {
    if (ncol(y) < 1L) stop("Cannot convert empty data.frame to factor")
    y <- y[[1]]
  }
  as.factor(y)
}

metric_from_pred <- function(y_true, pred_obj) {
  yp <- pred_obj$Ypred
  if (is.factor(y_true)) {
    pred <- NULL
    if (is.data.frame(yp)) pred <- as.factor(yp[[1]])
    if (is.null(pred) && is.matrix(yp)) pred <- as.factor(yp[, 1])
    if (is.null(pred) && is.vector(yp)) pred <- as.factor(yp)
    if (is.null(pred) && length(dim(yp)) == 3L) {
      mat <- yp[, , 1, drop = FALSE]
      lev <- pred_obj$lev
      cls <- apply(mat, 1, which.max)
      pred <- factor(lev[cls], levels = lev)
    }
    if (is.null(pred)) stop("Cannot decode classification predictions")
    val <- mean(as.character(pred) == as.character(y_true), na.rm = TRUE)
    return(list(metric_name = "accuracy", metric_value = as.numeric(val)))
  }

  # regression
  y_num <- as.matrix(y_true)
  pred_num <- NULL
  if (length(dim(yp)) == 3L) {
    pred_num <- yp[, , 1, drop = FALSE]
  } else if (is.matrix(yp)) {
    pred_num <- yp
  } else {
    pred_num <- matrix(as.numeric(yp), ncol = 1)
  }
  if (!all(dim(pred_num) == dim(y_num))) {
    pred_num <- matrix(as.numeric(pred_num), nrow = nrow(y_num), ncol = ncol(y_num))
  }
  rmsd <- sqrt(mean((pred_num - y_num)^2))
  list(metric_name = "rmsd", metric_value = as.numeric(rmsd))
}

half_split_idx <- function(n) {
  ss <- sample.int(n, size = max(1L, round(n / 2)))
  list(train = ss, test = setdiff(seq_len(n), ss))
}

stratified_half_split <- function(y) {
  idx_by <- split(seq_along(y), y)
  test_idx <- unlist(lapply(idx_by, function(ix) {
    n <- length(ix)
    if (n <= 1L) return(integer(0))
    sample(ix, size = max(1L, floor(n / 2)))
  }), use.names = FALSE)
  test_idx <- sort(unique(test_idx))
  train_idx <- setdiff(seq_along(y), test_idx)
  list(train = train_idx, test = test_idx)
}

fixed_train_split <- function(n, train_n) {
  if (n < 2L) stop("Need at least 2 rows to split train/test")
  train_n_eff <- min(max(1L, as.integer(train_n)), n - 1L)
  train_idx <- sample.int(n, size = train_n_eff)
  test_idx <- setdiff(seq_len(n), train_idx)
  list(train = train_idx, test = test_idx)
}

load_dataset <- function(name) {
  name <- tolower(name)
  if (name == "cifar100") {
    p <- file.path(base_dir, "CIFAR100.RData")
    e <- new.env(parent = emptyenv())
    load(p, envir = e)
    stopifnot(exists("r", envir = e), "label_idx" %in% colnames(e$r))
    feat_cols <- grep("^feat_", colnames(e$r), value = TRUE)
    if (!length(feat_cols)) {
      feat_cols <- setdiff(colnames(e$r), c("image_path", "split", "label_idx", "label_name"))
    }
    X <- as.matrix(e$r[, ..feat_cols])
    storage.mode(X) <- "double"
    y <- safe_factor(e$r[, "label_idx"])
    split_col <- if ("split" %in% colnames(e$r)) trimws(tolower(as.character(e$r[, "split"]))) else rep("train", nrow(X))
    train_idx <- which(split_col == "train")
    test_idx <- which(split_col == "test")
    if (!length(train_idx) || !length(test_idx)) {
      sp <- half_split_idx(nrow(X))
      train_idx <- sp$train
      test_idx <- sp$test
    }
    return(list(name = name, Xtrain = X[train_idx, , drop = FALSE], Ytrain = y[train_idx], Xtest = X[test_idx, , drop = FALSE], Ytest = y[test_idx]))
  }

  if (name == "imagenet") {
    p <- file.path(base_dir, "imagenet.RData")
    e <- new.env(parent = emptyenv())
    objs <- load(p, envir = e)
    if (all(c("Xtrain", "Ytrain", "Xtest", "Ytest") %in% objs)) {
      Xall <- rbind(as.matrix(e$Xtrain), as.matrix(e$Xtest))
      yall <- safe_factor(c(as.character(e$Ytrain), as.character(e$Ytest)))
      sp <- fixed_train_split(nrow(Xall), imagenet_train_n)
      return(list(name = name, Xtrain = Xall[sp$train, , drop = FALSE], Ytrain = yall[sp$train], Xtest = Xall[sp$test, , drop = FALSE], Ytest = yall[sp$test]))
    }
    if ("r" %in% objs && is.data.frame(e$r) && "label_idx" %in% colnames(e$r)) {
      X <- e$r[, -c(1:3), drop = FALSE]
      X <- as.data.frame(lapply(X, function(x) suppressWarnings(as.numeric(as.character(x)))))
      keep <- vapply(X, function(v) any(is.finite(v)), logical(1))
      X <- as.matrix(X[, keep, drop = FALSE])
      y <- safe_factor(e$r[, "label_idx"])
      sp <- fixed_train_split(nrow(X), imagenet_train_n)
      return(list(name = name, Xtrain = X[sp$train, , drop = FALSE], Ytrain = y[sp$train], Xtest = X[sp$test, , drop = FALSE], Ytest = y[sp$test]))
    }
    if (all(c("data", "labels") %in% objs)) {
      X <- as.matrix(e$data)
      y <- safe_factor(e$labels)
      sp <- fixed_train_split(nrow(X), imagenet_train_n)
      return(list(name = name, Xtrain = X[sp$train, , drop = FALSE], Ytrain = y[sp$train], Xtest = X[sp$test, , drop = FALSE], Ytest = y[sp$test]))
    }
    stop("Unsupported imagenet.RData format")
  }

  if (name == "nmr") {
    p <- file.path(base_dir, "NMR.RData")
    e <- new.env(parent = emptyenv())
    load(p, envir = e)
    return(list(name = name, Xtrain = as.matrix(e$Xtrain), Ytrain = as.matrix(e$Ytrain), Xtest = as.matrix(e$Xtest), Ytest = as.matrix(e$Ytest)))
  }

  if (name == "singlecell") {
    p <- file.path(base_dir, "singlecell.RData")
    e <- new.env(parent = emptyenv())
    load(p, envir = e)
    X <- as.matrix(e$data)
    y <- safe_factor(e$labels)
    sp <- stratified_half_split(y)
    return(list(name = name, Xtrain = X[sp$train, , drop = FALSE], Ytrain = y[sp$train], Xtest = X[sp$test, , drop = FALSE], Ytest = y[sp$test]))
  }

  if (name == "metref") {
    suppressPackageStartupMessages(library(KODAMA))
    data("MetRef", package = "KODAMA")
    X <- MetRef$data
    X <- X[, colSums(X) != 0, drop = FALSE]
    X <- normalization(X)$newXtrain
    y <- safe_factor(MetRef$donor)
    ss <- sample(seq_len(nrow(X)), 100)
    return(list(name = name, Xtrain = as.matrix(X[-ss, , drop = FALSE]), Ytrain = y[-ss], Xtest = as.matrix(X[ss, , drop = FALSE]), Ytest = y[ss]))
  }

  stop("Unknown dataset: ", name)
}

subset_samples <- function(ds, frac) {
  ntr <- nrow(ds$Xtrain); nte <- nrow(ds$Xtest)
  ntr2 <- max(2L, as.integer(round(ntr * frac)))
  nte2 <- max(2L, as.integer(round(nte * frac)))
  it <- sample.int(ntr, size = min(ntr, ntr2))
  ie <- sample.int(nte, size = min(nte, nte2))
  list(
    Xtrain = ds$Xtrain[it, , drop = FALSE],
    Ytrain = if (is.factor(ds$Ytrain)) ds$Ytrain[it] else ds$Ytrain[it, , drop = FALSE],
    Xtest = ds$Xtest[ie, , drop = FALSE],
    Ytest = if (is.factor(ds$Ytest)) ds$Ytest[ie] else ds$Ytest[ie, , drop = FALSE]
  )
}

subset_xvars <- function(ds, frac) {
  p <- ncol(ds$Xtrain)
  p2 <- max(1L, as.integer(round(p * frac)))
  cols <- sample.int(p, size = min(p, p2))
  list(
    Xtrain = ds$Xtrain[, cols, drop = FALSE],
    Ytrain = ds$Ytrain,
    Xtest = ds$Xtest[, cols, drop = FALSE],
    Ytest = ds$Ytest
  )
}

subset_yvars <- function(ds, frac) {
  if (is.factor(ds$Ytrain)) return(ds)
  q <- ncol(ds$Ytrain)
  q2 <- max(1L, as.integer(round(q * frac)))
  cols <- sample.int(q, size = min(q, q2))
  list(
    Xtrain = ds$Xtrain,
    Ytrain = ds$Ytrain[, cols, drop = FALSE],
    Xtest = ds$Xtest,
    Ytest = ds$Ytest[, cols, drop = FALSE]
  )
}

method_grid <- function(cuda_ok, include_r = FALSE) {
  svd <- c("irlba", "arpack", "cpu_rsvd", if (cuda_ok) "cuda_rsvd")
  dt <- CJ(engine = "Rcpp", algorithm = c("simpls", "plssvd", "simpls_fast"), svd_method = svd, fast_profile = "default", unique = TRUE)
  if (isTRUE(include_simpls_fast_incremental)) {
    dt <- rbind(
      dt,
      CJ(engine = "Rcpp", algorithm = "simpls_fast", svd_method = svd, fast_profile = "incdefl", unique = TRUE),
      fill = TRUE
    )
  }
  if (include_r) {
    dt <- rbind(
      dt,
      CJ(engine = "R", algorithm = c("simpls", "plssvd", "simpls_fast"), svd_method = c("irlba", "arpack", "cpu_rsvd"), fast_profile = "default", unique = TRUE),
      fill = TRUE
    )
  }
  dt <- dt[!(algorithm == "simpls" & svd_method == "arpack")]
  dt[, method_id := paste(engine, algorithm, svd_method, fast_profile, sep = "_")]
  dt[]
}

methods_for_dataset <- function(dname, methods_all) {
  m <- copy(methods_all)
  if (tolower(dname) != "metref") {
    m <- m[!(engine %in% c("R", "pls_pkg"))]
  }
  if (tolower(dname) == "metref") {
    if (isTRUE(metref_include_r) && !any(m$engine == "R")) {
      m <- rbind(
        m,
        CJ(engine = "R", algorithm = c("simpls", "plssvd", "simpls_fast"), svd_method = c("irlba", "arpack", "cpu_rsvd"), fast_profile = "default", unique = TRUE),
        fill = TRUE
      )
    }
    if (isTRUE(metref_include_pls_pkg) && requireNamespace("pls", quietly = TRUE)) {
      m <- rbind(
        m,
        data.table(engine = "pls_pkg", algorithm = "simpls", svd_method = "none", fast_profile = "default"),
        fill = TRUE
      )
    }
    # The pure-R simpls_fast path is currently unstable on MetRef classification.
    # Keep R baselines (simpls/plssvd) and pls package baseline, but drop only R_simpls_fast.
    m <- m[!(engine == "R" & algorithm == "simpls_fast")]
    m <- m[!(algorithm == "simpls" & svd_method == "arpack")]
    m[, method_id := paste(engine, algorithm, svd_method, fast_profile, sep = "_")]
  }
  if (tolower(dname) == "nmr") {
    if (skip_arpack_on_nmr) {
      m <- m[svd_method != "arpack"]
    }
    if (skip_plssvd_on_nmr) {
      m <- m[algorithm != "plssvd"]
    }
  }
  m
}

fit_build <- function(ds, cfg, ncomp, param_cfg) {
  cfg <- as.list(cfg)

  if (identical(cfg$engine, "pls_pkg")) {
    if (!requireNamespace("pls", quietly = TRUE)) {
      stop("pls package not available")
    }
    if (is.factor(ds$Ytrain)) {
      class_lev <- levels(ds$Ytrain)
      Ymm <- model.matrix(~ ds$Ytrain - 1)
      colnames(Ymm) <- paste0("cls_", seq_len(ncol(Ymm)))
      class_map <- setNames(class_lev, colnames(Ymm))
      x_names <- paste0("x_", seq_len(ncol(ds$Xtrain)))
      Xtr <- ds$Xtrain
      colnames(Xtr) <- x_names
      df_train <- data.frame(Ymm, Xtr, check.names = FALSE)
      form <- as.formula(paste0("cbind(", paste(colnames(Ymm), collapse = ","), ") ~ ."))

      t0 <- proc.time()[3]
      mdl <- pls::plsr(form, data = df_train, ncomp = as.integer(ncomp), method = "simpls", scale = FALSE, validation = "none")
      train_ms <- (proc.time()[3] - t0) * 1000

      Xte <- as.data.frame(ds$Xtest)
      colnames(Xte) <- x_names
      pred_arr <- predict(mdl, newdata = Xte, ncomp = as.integer(ncomp))
      pred_mat <- pred_arr[, , 1, drop = FALSE]
      pred_idx <- apply(pred_mat, 1, which.max)
      lev <- colnames(Ymm)
      pred <- factor(class_map[lev[pred_idx]], levels = class_lev)
      m <- list(metric_name = "accuracy", metric_value = mean(as.character(pred) == as.character(ds$Ytest), na.rm = TRUE))
      return(list(train_ms = as.numeric(train_ms), metric_name = m$metric_name, metric_value = as.numeric(m$metric_value), model_size_mb = as.numeric(object.size(mdl)) / (1024^2)))
    }

    ymat <- as.matrix(ds$Ytrain)
    x_names <- paste0("x_", seq_len(ncol(ds$Xtrain)))
    colnames(ds$Xtrain) <- x_names
    df_train <- data.frame(ymat, ds$Xtrain, check.names = FALSE)
    y_cols <- colnames(df_train)[seq_len(ncol(ymat))]
    form <- as.formula(paste0("cbind(", paste(y_cols, collapse = ","), ") ~ ."))
    t0 <- proc.time()[3]
    mdl <- pls::plsr(form, data = df_train, ncomp = as.integer(ncomp), method = "simpls", scale = FALSE, validation = "none")
    train_ms <- (proc.time()[3] - t0) * 1000

    Xte <- as.data.frame(ds$Xtest)
    colnames(Xte) <- x_names
    pred_arr <- predict(mdl, newdata = Xte, ncomp = as.integer(ncomp))
    pred_mat <- pred_arr[, , 1, drop = FALSE]
    pred_num <- as.matrix(pred_mat)
    y_true <- as.matrix(ds$Ytest)
    if (!all(dim(pred_num) == dim(y_true))) {
      pred_num <- matrix(as.numeric(pred_num), nrow = nrow(y_true), ncol = ncol(y_true))
    }
    rmsd <- sqrt(mean((pred_num - y_true)^2))
    return(list(train_ms = as.numeric(train_ms), metric_name = "rmsd", metric_value = as.numeric(rmsd), model_size_mb = as.numeric(object.size(mdl)) / (1024^2)))
  }

  args <- list(
    Xtrain = ds$Xtrain,
    Ytrain = ds$Ytrain,
    ncomp = as.integer(ncomp),
    method = cfg$algorithm,
    svd.method = cfg$svd_method,
    scaling = "centering",
    rsvd_oversample = as.integer(param_cfg$rsvd_oversample),
    rsvd_power = as.integer(param_cfg$rsvd_power),
    svds_tol = as.numeric(param_cfg$svds_tol),
    irlba_svtol = as.numeric(irlba_svtol),
    rsvd_tol = as.numeric(rsvd_tol)
  )
  if (identical(cfg$algorithm, "simpls_fast") && identical(cfg$fast_profile, "incdefl")) {
    args <- c(args, list(
      fast_incremental = TRUE,
      fast_inc_iters = 2L,
      fast_defl_cache = TRUE,
      fast_center_t = FALSE,
      fast_reorth_v = FALSE,
      fast_block = 8L
    ))
  }

  if (identical(cfg$engine, "Rcpp")) {
    fn <- fastPLS::pls
  } else if (identical(cfg$engine, "R")) {
    fn <- fastPLS::pls_r
  } else {
    stop("Unsupported engine: ", cfg$engine)
  }

  call_args <- filter_call_args(fn, args)
  if (!length(call_args) || is.null(names(call_args)) || any(!nzchar(names(call_args)))) {
    stop("Internal benchmark error: unnamed model arguments after filtering")
  }

  t0 <- proc.time()[3]
  model <- do.call(fn, call_args)
  train_ms <- (proc.time()[3] - t0) * 1000

  pred <- predict(model, newdata = ds$Xtest, Ytest = ds$Ytest, proj = FALSE)
  m <- metric_from_pred(ds$Ytest, pred)

  list(train_ms = as.numeric(train_ms), metric_name = m$metric_name, metric_value = as.numeric(m$metric_value), model_size_mb = as.numeric(object.size(model)) / (1024^2))
}

run_analysis <- function(dname, ds0, methods, analysis, values, ncomp_fixed = 5L, reps_run = reps) {
  rows <- list()
  idx <- 0L
  for (i in seq_len(nrow(methods))) {
    cfg <- methods[i]
    for (v in values) {
      ncomp_preview <- if (analysis == "ncomp") as.integer(v) else as.integer(ncomp_fixed)
      log_msg(
        "[RUN] dataset=", dname,
        " analysis=", analysis,
        " value=", as.character(v),
        " method=", cfg$method_id,
        " ncomp=", ncomp_preview,
        " reps=", reps_run
      )
      for (r in seq_len(reps_run)) {
        ds <- ds0
        ncomp_run <- ncomp_fixed
        analysis_value <- as.character(v)

        if (analysis == "ncomp") ncomp_run <- as.integer(v)
        if (analysis == "sample_fraction") ds <- subset_samples(ds0, as.numeric(v))
        if (analysis == "xvar_fraction") ds <- subset_xvars(ds0, as.numeric(v))
        if (analysis == "yvar_fraction") ds <- subset_yvars(ds0, as.numeric(v))
        xtrain_nrow <- nrow(ds$Xtrain)
        xtrain_ncol <- ncol(ds$Xtrain)
        ytrain_ncol <- if (is.factor(ds$Ytrain)) 1L else ncol(as.matrix(ds$Ytrain))

        if (cfg$algorithm == "plssvd") {
          ycap <- if (is.factor(ds$Ytrain)) nlevels(ds$Ytrain) else ncol(as.matrix(ds$Ytrain))
          cap <- min(nrow(ds$Xtrain), ncol(ds$Xtrain), ycap)
          if (ncomp_run > cap) {
            idx <- idx + 1L
            rows[[idx]] <- data.table(
              dataset = dname, analysis = analysis, analysis_value = analysis_value,
              rep = r, engine = cfg$engine, algorithm = cfg$algorithm, svd_method = cfg$svd_method, fast_profile = cfg$fast_profile,
              method_id = cfg$method_id, param_set = NA_character_, ncomp = ncomp_run,
              xtrain_nrow = xtrain_nrow, xtrain_ncol = xtrain_ncol, ytrain_ncol = ytrain_ncol,
              train_ms = NA_real_, metric_name = if (is.factor(ds$Ytest)) "accuracy" else "rmsd", metric_value = NA_real_,
              model_size_mb = NA_real_, status = "skipped_plssvd_cap", msg = ""
            )
            next
          }
        }

        if (cfg$svd_method == "cuda_rsvd" && !include_cuda) {
          idx <- idx + 1L
          rows[[idx]] <- data.table(
            dataset = dname, analysis = analysis, analysis_value = analysis_value,
            rep = r, engine = cfg$engine, algorithm = cfg$algorithm, svd_method = cfg$svd_method, fast_profile = cfg$fast_profile,
              method_id = cfg$method_id, param_set = NA_character_, ncomp = ncomp_run,
            xtrain_nrow = xtrain_nrow, xtrain_ncol = xtrain_ncol, ytrain_ncol = ytrain_ncol,
            train_ms = NA_real_, metric_name = if (is.factor(ds$Ytest)) "accuracy" else "rmsd", metric_value = NA_real_,
            model_size_mb = NA_real_, status = "skipped_cuda_disabled", msg = ""
          )
          next
        }

        res <- tryCatch(
          fit_build(ds, cfg, ncomp_run, list(svds_tol = 0, rsvd_oversample = 10L, rsvd_power = 1L)),
          error = function(e) {
            e$benchmark_call <- paste(deparse(conditionCall(e)), collapse = " ")
            e
          }
        )
        idx <- idx + 1L
        if (inherits(res, "error")) {
          msg_txt <- conditionMessage(res)
          call_txt <- res$benchmark_call
          if (!is.null(call_txt) && nzchar(call_txt)) {
            msg_txt <- paste0(msg_txt, " | call=", call_txt)
          }
          log_msg("[ERR] dataset=", dname, " analysis=", analysis, " value=", analysis_value, " method=", cfg$method_id, " rep=", r, " msg=", msg_txt)
          rows[[idx]] <- data.table(
            dataset = dname, analysis = analysis, analysis_value = analysis_value,
            rep = r, engine = cfg$engine, algorithm = cfg$algorithm, svd_method = cfg$svd_method, fast_profile = cfg$fast_profile,
            method_id = cfg$method_id, param_set = NA_character_, ncomp = ncomp_run,
            xtrain_nrow = xtrain_nrow, xtrain_ncol = xtrain_ncol, ytrain_ncol = ytrain_ncol,
            train_ms = NA_real_, metric_name = if (is.factor(ds$Ytest)) "accuracy" else "rmsd", metric_value = NA_real_,
            model_size_mb = NA_real_, status = "error", msg = msg_txt
          )
        } else {
          rows[[idx]] <- data.table(
            dataset = dname, analysis = analysis, analysis_value = analysis_value,
            rep = r, engine = cfg$engine, algorithm = cfg$algorithm, svd_method = cfg$svd_method, fast_profile = cfg$fast_profile,
            method_id = cfg$method_id, param_set = NA_character_, ncomp = ncomp_run,
            xtrain_nrow = xtrain_nrow, xtrain_ncol = xtrain_ncol, ytrain_ncol = ytrain_ncol,
            train_ms = res$train_ms, metric_name = res$metric_name, metric_value = res$metric_value,
            model_size_mb = res$model_size_mb, status = "ok", msg = ""
          )
        }
      }
    }
  }
  rbindlist(rows, fill = TRUE)
}

dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)
if (!append_mode || !file.exists(out_log)) {
  cat("", file = out_log, append = FALSE)
} else {
  cat(paste0("\n[", stamp(), "] ---- append run start ----\n"), file = out_log, append = TRUE)
}

cuda_ok <- tryCatch(isTRUE(has_cuda()), error = function(e) FALSE)
if (!include_cuda) cuda_ok <- FALSE
methods <- method_grid(cuda_ok = cuda_ok, include_r = include_r_impl)

header <- data.table(
  dataset = character(), analysis = character(), analysis_value = character(), rep = integer(),
  engine = character(), algorithm = character(), svd_method = character(), fast_profile = character(), method_id = character(),
  param_set = character(), ncomp = integer(), xtrain_nrow = integer(), xtrain_ncol = integer(), ytrain_ncol = integer(),
  train_ms = numeric(), metric_name = character(), metric_value = numeric(),
  model_size_mb = numeric(), status = character(), msg = character()
)
if (!append_mode || !file.exists(out_csv)) {
  fwrite(header, out_csv)
}

log_msg("Multianalysis benchmark started")
log_msg("datasets=", paste(dataset_filter, collapse = ","), "; reps=", reps, "; cuda=", cuda_ok, "; include_r_impl=", include_r_impl)
log_msg("include_simpls_fast_incremental=", include_simpls_fast_incremental)
log_msg("metref_include_r=", metref_include_r, "; metref_include_pls_pkg=", metref_include_pls_pkg)
log_msg("threads=", threads)
log_msg("svd tuning: irlba_svtol=", irlba_svtol, "; rsvd_tol=", rsvd_tol)
log_msg("replicate policy: default reps=", reps, "; nmr_reps=", nmr_reps)
log_msg("replicate policy (dataset overrides): metref_reps=", metref_reps, "; singlecell_reps=", singlecell_reps, "; cifar100_reps=", cifar100_reps)
log_msg("NMR safety filters: skip_arpack_on_nmr=", skip_arpack_on_nmr, "; skip_plssvd_on_nmr=", skip_plssvd_on_nmr)
log_msg("ncomp(default datasets)=", paste(ncomp_vec, collapse = ","), "; ncomp(NMR)=", paste(ncomp_nmr_vec, collapse = ","), "; ncomp(SingleCell)=", paste(ncomp_singlecell_vec, collapse = ","), "; default_ncomp(non-ncomp analyses)=", default_ncomp, "; metref_default_ncomp=", metref_default_ncomp, "; singlecell_default_ncomp=", singlecell_default_ncomp, "; cifar100_default_ncomp=", cifar100_default_ncomp)
log_msg("sample_fracs=", paste(sample_fracs, collapse = ","), "; xvar_fracs=", paste(xvar_fracs, collapse = ","), "; yvar_fracs=", paste(yvar_fracs, collapse = ","))

all_rows <- list(); ai <- 0L
for (dname in dataset_filter) {
  log_msg("Loading dataset: ", dname)
  ds <- tryCatch(load_dataset(dname), error = function(e) e)
  if (inherits(ds, "error")) {
    log_msg("SKIP dataset ", dname, " error: ", conditionMessage(ds))
    next
  }
  methods_ds <- methods_for_dataset(dname, methods)
  reps_ds <- if (tolower(dname) == "nmr") nmr_reps else reps
  if (tolower(dname) == "metref") reps_ds <- metref_reps
  if (tolower(dname) == "singlecell") reps_ds <- singlecell_reps
  if (tolower(dname) == "cifar100") reps_ds <- cifar100_reps
  ncomp_ds <- if (tolower(dname) == "nmr") ncomp_nmr_vec else ncomp_vec
  if (tolower(dname) == "singlecell") ncomp_ds <- ncomp_singlecell_vec
  default_ncomp_ds <- if (tolower(dname) == "metref") metref_default_ncomp else default_ncomp
  if (tolower(dname) == "singlecell") default_ncomp_ds <- singlecell_default_ncomp
  if (tolower(dname) == "cifar100") default_ncomp_ds <- cifar100_default_ncomp
  log_msg("Methods for ", dname, ": ", nrow(methods_ds), " configs")
  log_msg("Replicates for ", dname, ": ", reps_ds)
  log_msg("ncomp grid for ", dname, ": ", paste(ncomp_ds, collapse = ","))
  log_msg("default ncomp for non-ncomp analyses on ", dname, ": ", default_ncomp_ds)

  log_msg("Running ncomp benchmark on ", dname)
  dt1 <- run_analysis(dname, ds, methods_ds, "ncomp", ncomp_ds, ncomp_fixed = default_ncomp_ds, reps_run = reps_ds)
  fwrite(dt1, out_csv, append = TRUE)
  ai <- ai + 1L; all_rows[[ai]] <- dt1

  log_msg("Running sample fraction benchmark on ", dname)
  dt2 <- run_analysis(dname, ds, methods_ds, "sample_fraction", sample_fracs, ncomp_fixed = default_ncomp_ds, reps_run = reps_ds)
  fwrite(dt2, out_csv, append = TRUE)
  ai <- ai + 1L; all_rows[[ai]] <- dt2

  log_msg("Running X-variable fraction benchmark on ", dname)
  dt3 <- run_analysis(dname, ds, methods_ds, "xvar_fraction", xvar_fracs, ncomp_fixed = default_ncomp_ds, reps_run = reps_ds)
  fwrite(dt3, out_csv, append = TRUE)
  ai <- ai + 1L; all_rows[[ai]] <- dt3

  if (tolower(dname) == "nmr") {
    log_msg("Running Y-variable fraction benchmark on NMR")
    dt4 <- run_analysis(dname, ds, methods_ds, "yvar_fraction", yvar_fracs, ncomp_fixed = default_ncomp_ds, reps_run = reps_ds)
    fwrite(dt4, out_csv, append = TRUE)
    ai <- ai + 1L; all_rows[[ai]] <- dt4
  }

}

res <- rbindlist(all_rows, fill = TRUE)
if (!nrow(res)) {
  log_msg("No results collected.")
  quit(save = "no", status = 1)
}

raw_all <- fread(out_csv)
sumtab <- raw_all[status == "ok", .(
  reps_ok = .N,
  train_ms_median = median(train_ms, na.rm = TRUE),
  train_ms_mean = mean(train_ms, na.rm = TRUE),
  train_ms_sd = sd(train_ms, na.rm = TRUE),
  metric_median = median(metric_value, na.rm = TRUE),
  metric_mean = mean(metric_value, na.rm = TRUE),
  model_size_mb_median = median(model_size_mb, na.rm = TRUE)
), by = .(dataset, analysis, analysis_value, engine, algorithm, svd_method, fast_profile, method_id, param_set, ncomp, metric_name)]

fwrite(sumtab, file.path(out_dir, "multianalysis_summary.csv"))

log_msg("Completed. Raw: ", out_csv)
log_msg("Summary: ", file.path(out_dir, "multianalysis_summary.csv"))
cat("Done\n")
