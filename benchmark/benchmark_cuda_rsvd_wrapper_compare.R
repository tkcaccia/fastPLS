#!/usr/bin/env Rscript

suppressPackageStartupMessages({
  library(fastPLS)
  library(data.table)
})

label <- Sys.getenv("FASTPLS_COMPARE_LABEL", "baseline")
data_root <- path.expand(Sys.getenv("FASTPLS_DATA_ROOT", "/home/chiamaka/Documents/fastpls/data"))
out_dir <- path.expand(Sys.getenv("FASTPLS_COMPARE_OUTDIR", file.path(getwd(), "benchmark_results_cuda_rsvd_compare")))
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

datasets <- strsplit(Sys.getenv("FASTPLS_COMPARE_DATASETS", "metref,singlecell,gtex_v8,tcga_pan_cancer,ccle,cifar100"), ",", fixed = TRUE)[[1]]
datasets <- trimws(datasets)
datasets <- datasets[nzchar(datasets)]
max_k <- as.integer(Sys.getenv("FASTPLS_COMPARE_MAX_K", "50"))
if (!is.finite(max_k) || is.na(max_k) || max_k < 1L) max_k <- 50L

load_task <- function(dataset_id) {
  dataset_id <- tolower(dataset_id)
  if (dataset_id == "metref") {
    for (p in c(file.path(data_root, "metref_remote_task.RData"),
                file.path(data_root, "metref.RData"))) {
      if (file.exists(p)) {
        e <- new.env(parent = emptyenv())
        load(p, envir = e)
        if ("out" %in% ls(e)) return(e$out)
        if (all(c("Xtrain", "Ytrain") %in% ls(e))) {
          return(list(Xtrain = as.matrix(e$Xtrain), Ytrain = e$Ytrain))
        }
      }
    }
  }
  if (dataset_id == "singlecell") {
    for (p in c(file.path(data_root, "singlecell.RData"),
                "/Users/stefano/HPC-firenze/image_analysis/dinoV2/Rdatasets/singlecell.RData")) {
      if (file.exists(p)) {
        e <- new.env(parent = emptyenv())
        load(p, envir = e)
        if (all(c("data", "labels") %in% ls(e))) {
          return(list(Xtrain = as.matrix(e$data), Ytrain = droplevels(e$labels)))
        }
      }
    }
  }
  path_map <- c(
    gtex_v8 = file.path(data_root, "gtex.RData"),
    tcga_pan_cancer = file.path(data_root, "tcga_pan_cancer.RData"),
    ccle = file.path(data_root, "ccle.RData"),
    cifar100 = file.path(data_root, "CIFAR100.RData")
  )
  p <- unname(path_map[[dataset_id]])
  if (is.null(p) || !file.exists(p)) stop("Missing dataset file for ", dataset_id)
  e <- new.env(parent = emptyenv())
  load(p, envir = e)
  if (all(c("Xtrain", "Ytrain") %in% ls(e))) {
    return(list(Xtrain = as.matrix(e$Xtrain), Ytrain = e$Ytrain))
  }
  if ("out" %in% ls(e) && is.list(e$out)) {
    return(list(Xtrain = as.matrix(e$out$Xtrain), Ytrain = e$out$Ytrain))
  }
  if ("r" %in% ls(e) && is.data.frame(e$r)) {
    feat_cols <- grep("^feat_", colnames(e$r), value = TRUE)
    if (!length(feat_cols)) {
      feat_cols <- setdiff(colnames(e$r), c("image_path", "split", "label_idx", "label_name"))
    }
    X <- as.matrix(e$r[, ..feat_cols])
    storage.mode(X) <- "double"
    y <- factor(e$r$label_idx)
    split_col <- if ("split" %in% colnames(e$r)) trimws(tolower(as.character(e$r$split))) else rep("train", nrow(X))
    train_idx <- which(split_col == "train")
    if (!length(train_idx)) train_idx <- seq_len(nrow(X))
    return(list(Xtrain = X[train_idx, , drop = FALSE], Ytrain = droplevels(y[train_idx])))
  }
  stop("Unsupported dataset structure for ", dataset_id)
}

make_y_numeric <- function(y) {
  if (is.factor(y)) {
    mm <- model.matrix(~ y - 1)
    colnames(mm) <- sub("^y", "", colnames(mm))
    return(mm)
  }
  as.matrix(y)
}

center_cols <- function(x) {
  scale(x, center = TRUE, scale = FALSE)
}

rows <- list()
artifacts <- list()
for (dataset_id in datasets) {
  task <- load_task(dataset_id)
  X <- as.matrix(task$Xtrain)
  Y <- make_y_numeric(task$Ytrain)
  storage.mode(X) <- "double"
  storage.mode(Y) <- "double"

  A <- crossprod(center_cols(X), center_cols(Y))
  k <- min(max_k, nrow(A), ncol(A))
  if (k < 1L) stop("Invalid k for dataset ", dataset_id)

  fit <- fastPLS::svd_run(
    A = A,
    k = k,
    method = "cuda_rsvd",
    rsvd_oversample = 10L,
    rsvd_power = 1L,
    seed = 123L,
    left_only = FALSE
  )

  artifact_path <- file.path(out_dir, sprintf("%s_%s.rds", label, dataset_id))
  saveRDS(fit, artifact_path)
  artifacts[[dataset_id]] <- artifact_path

  rows[[dataset_id]] <- data.table(
    label = label,
    dataset = dataset_id,
    m = nrow(A),
    n = ncol(A),
    k = k,
    elapsed = fit$elapsed,
    s1 = if (length(fit$s)) fit$s[1] else NA_real_,
    s_last = if (length(fit$s)) fit$s[length(fit$s)] else NA_real_,
    artifact = artifact_path
  )
}

res <- rbindlist(rows, fill = TRUE)
fwrite(res, file.path(out_dir, sprintf("%s_summary.csv", label)))
cat("Wrote:", file.path(out_dir, sprintf("%s_summary.csv", label)), "\n")
