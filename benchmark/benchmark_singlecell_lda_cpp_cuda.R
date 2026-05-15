#!/usr/bin/env Rscript

options(stringsAsFactors = FALSE)

arg <- function(name, default) {
  key <- paste0("--", name, "=")
  hit <- commandArgs(trailingOnly = TRUE)
  hit <- hit[startsWith(hit, key)]
  if (!length(hit)) return(default)
  sub(key, "", hit[[1L]], fixed = TRUE)
}

ncomp <- as.integer(arg("ncomp", Sys.getenv("FASTPLS_NCOMP", "50")))
if (!is.finite(ncomp) || is.na(ncomp) || ncomp < 1L) ncomp <- 50L
reps <- as.integer(arg("reps", Sys.getenv("FASTPLS_REPS", "3")))
if (!is.finite(reps) || is.na(reps) || reps < 1L) reps <- 3L
split_seed <- as.integer(arg("seed", Sys.getenv("FASTPLS_SEED", "123")))
if (!is.finite(split_seed) || is.na(split_seed)) split_seed <- 123L
outfile <- arg("out", Sys.getenv("FASTPLS_OUTFILE", "singlecell_fastpls_lda_cpp_cuda_ncomp50.csv"))

suppressPackageStartupMessages(library(fastPLS))

safe_factor <- function(x) droplevels(if (is.factor(x)) x else factor(x))

make_stratified_split <- function(y, train_frac = 0.5) {
  y <- safe_factor(y)
  idx <- seq_along(y)
  by_class <- split(idx, y)
  train_idx <- unlist(lapply(by_class, function(ii) {
    n_train <- max(1L, floor(length(ii) * train_frac))
    sample(ii, n_train)
  }), use.names = FALSE)
  list(train = sort(train_idx), test = sort(setdiff(idx, train_idx)))
}

find_singlecell <- function() {
  candidates <- unique(Filter(nzchar, c(
    Sys.getenv("FASTPLS_SINGLECELL_RDATA", ""),
    "/home/chiamaka/Documents/fastpls/data/singlecell.RData",
    "/home/chiamaka/Documents/fastpls/data/remote_fastpls_data/singlecell.RData",
    "/Users/stefano/Documents/GPUPLS/remote_fastpls_data/singlecell.RData",
    "/Users/stefano/Documents/GPUPLS/Data/singlecell.RData"
  )))
  candidates <- candidates[file.exists(candidates)]
  if (!length(candidates)) stop("singlecell.RData not found; set FASTPLS_SINGLECELL_RDATA.", call. = FALSE)
  normalizePath(candidates[[1L]], winslash = "/", mustWork = TRUE)
}

load_singlecell_task <- function(path, split_seed) {
  e <- new.env(parent = emptyenv())
  load(path, envir = e)
  if (!exists("data", e) || !exists("labels", e)) {
    stop("Expected singlecell.RData to contain objects named data and labels.", call. = FALSE)
  }
  X <- as.matrix(e$data)
  y <- safe_factor(e$labels)
  set.seed(as.integer(split_seed))
  sp <- make_stratified_split(y, train_frac = 0.5)
  list(
    dataset = "singlecell",
    dataset_path = path,
    split_seed = split_seed,
    Xtrain = X[sp$train, , drop = FALSE],
    Ytrain = droplevels(y[sp$train]),
    Xtest = X[sp$test, , drop = FALSE],
    Ytest = factor(y[sp$test], levels = levels(y[sp$train])),
    n_train = length(sp$train),
    n_test = length(sp$test),
    p = ncol(X),
    n_classes = nlevels(y[sp$train])
  )
}

task <- load_singlecell_task(find_singlecell(), split_seed)

accuracy <- function(pred) {
  mean(as.character(pred) == as.character(task$Ytest), na.rm = TRUE)
}

variants <- list(
  list(name = "cpp_plssvd_argmax", method = "plssvd", backend = "cpp", classifier = "argmax"),
  list(name = "cpp_plssvd_lda", method = "plssvd", backend = "cpu", classifier = "lda"),
  list(name = "cpp_simpls_argmax", method = "simpls", backend = "cpp", classifier = "argmax"),
  list(name = "cpp_simpls_lda", method = "simpls", backend = "cpu", classifier = "lda")
)
if (isTRUE(has_cuda())) {
  variants <- c(variants, list(
    list(name = "cuda_plssvd_argmax", method = "plssvd", backend = "cuda", classifier = "argmax"),
    list(name = "cuda_plssvd_lda", method = "plssvd", backend = "cuda", classifier = "lda"),
    list(name = "cuda_simpls_argmax", method = "simpls", backend = "cuda", classifier = "argmax"),
    list(name = "cuda_simpls_lda", method = "simpls", backend = "cuda", classifier = "lda")
  ))
}

rows <- list()
idx <- 1L
for (v in variants) {
  for (rep in seq_len(reps)) {
    gc()
    status <- "ok"
    err <- NA_character_
    fit <- NULL
    elapsed <- system.time({
      fit <- tryCatch({
        args <- list(
          Xtrain = task$Xtrain,
          Ytrain = task$Ytrain,
          Xtest = task$Xtest,
          Ytest = task$Ytest,
          ncomp = ncomp,
          method = v$method,
          backend = v$backend,
          classifier = v$classifier,
          fit = FALSE,
          proj = FALSE
        )
        if (!identical(v$backend, "cuda")) args$svd.method <- "cpu_rsvd"
        do.call(fastPLS::pls, args)
      }, error = function(e) {
        status <<- "error"
        err <<- conditionMessage(e)
        NULL
      })
    })

    rows[[idx]] <- data.frame(
      dataset = task$dataset,
      dataset_path = task$dataset_path,
      variant = v$name,
      method = v$method,
      backend = v$backend,
      classifier = v$classifier,
      ncomp = ncomp,
      replicate = rep,
      elapsed_ms = unname(elapsed[["elapsed"]]) * 1000,
      accuracy = if (is.null(fit)) NA_real_ else accuracy(fit$Ypred[[1L]]),
      classification_rule = if (is.null(fit) || is.null(fit$classification_rule)) NA_character_ else fit$classification_rule,
      n_train = task$n_train,
      n_test = task$n_test,
      p = task$p,
      n_classes = task$n_classes,
      has_cuda = isTRUE(has_cuda()),
      status = status,
      error = err,
      stringsAsFactors = FALSE
    )
    message(sprintf("[%s] rep=%d elapsed=%.1f ms acc=%s status=%s",
                    v$name, rep, rows[[idx]]$elapsed_ms,
                    format(rows[[idx]]$accuracy, digits = 5), status))
    idx <- idx + 1L
  }
}

res <- do.call(rbind, rows)
utils::write.csv(res, outfile, row.names = FALSE)
print(res)
cat("Saved:", normalizePath(outfile, winslash = "/", mustWork = FALSE), "\n")
