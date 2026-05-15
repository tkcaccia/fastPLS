#!/usr/bin/env Rscript

fastpls_lib <- Sys.getenv("FASTPLS_LIB", "")
if (nzchar(fastpls_lib)) {
  .libPaths(c(path.expand(fastpls_lib), .libPaths()))
}
suppressPackageStartupMessages(library(fastPLS))

accuracy <- function(obs, pred) {
  mean(as.character(obs) == as.character(pred))
}

predict_labels <- function(fit, Xtest, ncomp) {
  pred <- predict(fit, Xtest, ncomp = ncomp)
  if (is.list(pred) && !is.null(pred$Ypred)) return(pred$Ypred[[1]])
  if (is.list(pred) && !is.null(pred$predicted)) return(pred$predicted)
  pred
}

load_singlecell <- function(path = Sys.getenv("FASTPLS_SINGLECELL_RDATA", "")) {
  candidates <- c(
    path,
    "~/Documents/fastpls/data/singlecell.RData",
    "~/Documents/fastPLS/data/singlecell.RData"
  )
  candidates <- path.expand(candidates[nzchar(candidates)])
  file <- candidates[file.exists(candidates)][1]
  if (is.na(file)) stop("singlecell RData not found; set FASTPLS_SINGLECELL_RDATA", call. = FALSE)

  env <- new.env(parent = emptyenv())
  load(file, env)
  X <- as.matrix(env$data)
  y <- factor(env$labels)
  set.seed(123)
  train <- sample(seq_len(nrow(X)), floor(nrow(X) * 0.5))
  list(
    name = "singlecell",
    Xtrain = X[train, , drop = FALSE],
    Xtest = X[-train, , drop = FALSE],
    ytrain = y[train],
    ytest = y[-train]
  )
}

load_cifar100 <- function(path = Sys.getenv("FASTPLS_CIFAR100_RDATA", "")) {
  candidates <- c(
    path,
    "~/Documents/Rdatasets/CIFAR100.RData",
    "~/GPUPLS/Data/CIFAR100.RData"
  )
  candidates <- path.expand(candidates[nzchar(candidates)])
  file <- candidates[file.exists(candidates)][1]
  if (is.na(file)) stop("CIFAR100 RData not found; set FASTPLS_CIFAR100_RDATA", call. = FALSE)

  env <- new.env(parent = emptyenv())
  load(file, env)
  r <- env$r
  y <- factor(r$label_idx)
  Xdf <- r[, -(1:3), drop = FALSE]
  X <- as.matrix(data.frame(lapply(Xdf, as.numeric), check.names = FALSE))
  keep <- colSums(is.finite(X)) > 0
  X <- X[, keep, drop = FALSE]
  X[!is.finite(X)] <- 0
  set.seed(123)
  train <- sample(seq_len(nrow(X)), floor(nrow(X) * 0.5))
  list(
    name = "cifar100",
    Xtrain = X[train, , drop = FALSE],
    Xtest = X[-train, , drop = FALSE],
    ytrain = y[train],
    ytest = y[-train]
  )
}

run_head_benchmark <- function(task, method, ncomp = 50L, reps = 3L) {
  fit_start <- proc.time()[["elapsed"]]
  base <- pls(
    task$Xtrain,
    task$ytrain,
    ncomp = ncomp,
    method = method,
    backend = "cuda",
    classifier = "argmax",
    fit = FALSE,
    proj = FALSE
  )
  base_fit_ms <- 1000 * (proc.time()[["elapsed"]] - fit_start)
  Ttrain <- fastPLS:::.fastpls_latent_scores(base, task$Xtrain, ncomp = ncomp, backend = "cuda")
  Ttest <- fastPLS:::.fastpls_latent_scores(base, task$Xtest, ncomp = ncomp, backend = "cuda")
  y_codes <- as.integer(factor(task$ytrain, levels = base$lev))

  rows <- list()
  i <- 1L
  for (classifier in c("lda_cpp", "lda_cuda")) {
    for (replicate in seq_len(reps)) {
      gc()
      train_start <- proc.time()[["elapsed"]]
      models <- if (identical(classifier, "lda_cuda")) {
        fastPLS:::lda_train_prefix_cuda(Ttrain, y_codes, length(base$lev), ncomp, 1e-8)
      } else {
        fastPLS:::lda_train_prefix_cpp(Ttrain, y_codes, length(base$lev), ncomp, 1e-8)
      }
      train_ms <- 1000 * (proc.time()[["elapsed"]] - train_start)
      pred_start <- proc.time()[["elapsed"]]
      pred_code <- if (identical(classifier, "lda_cuda")) {
        fastPLS:::lda_predict_labels_cuda(Ttest, models[[as.character(ncomp)]])
      } else {
        fastPLS:::lda_predict_labels_cpp(Ttest, models[[as.character(ncomp)]])
      }
      predict_ms <- 1000 * (proc.time()[["elapsed"]] - pred_start)
      rows[[i]] <- data.frame(
        dataset = task$name,
        scope = "head_only",
        method = method,
        classifier = classifier,
        native_backend = if (identical(classifier, "lda_cuda")) "cuda_native" else "cpp",
        ncomp = ncomp,
        replicate = replicate,
        base_fit_ms = base_fit_ms,
        train_ms = train_ms,
        predict_ms = predict_ms,
        total_head_ms = train_ms + predict_ms,
        elapsed_ms = NA_real_,
        accuracy = accuracy(task$ytest, factor(base$lev[as.integer(pred_code)], levels = base$lev)),
        status = "ok",
        error = NA_character_
      )
      i <- i + 1L
    }
  }
  do.call(rbind, rows)
}

run_end_to_end <- function(task, method, ncomp = 50L) {
  rows <- list()
  i <- 1L
  for (classifier in c("argmax", "lda")) {
    gc()
    start <- proc.time()[["elapsed"]]
    rows[[i]] <- tryCatch({
      fit <- pls(
        task$Xtrain,
        task$ytrain,
        ncomp = ncomp,
        method = method,
        backend = "cuda",
        classifier = classifier,
        fit = FALSE,
        proj = FALSE
      )
      pred <- predict_labels(fit, task$Xtest, ncomp)
      data.frame(
        dataset = task$name,
        scope = "end_to_end",
        method = method,
        classifier = classifier,
        native_backend = if (classifier == "argmax") "argmax" else classifier,
        ncomp = ncomp,
        replicate = 1L,
        base_fit_ms = NA_real_,
        train_ms = NA_real_,
        predict_ms = NA_real_,
        total_head_ms = NA_real_,
        elapsed_ms = 1000 * (proc.time()[["elapsed"]] - start),
        accuracy = accuracy(task$ytest, pred),
        status = "ok",
        error = NA_character_
      )
    }, error = function(e) {
      data.frame(
        dataset = task$name,
        scope = "end_to_end",
        method = method,
        classifier = classifier,
        native_backend = NA_character_,
        ncomp = ncomp,
        replicate = 1L,
        base_fit_ms = NA_real_,
        train_ms = NA_real_,
        predict_ms = NA_real_,
        total_head_ms = NA_real_,
        elapsed_ms = 1000 * (proc.time()[["elapsed"]] - start),
        accuracy = NA_real_,
        status = "error",
        error = conditionMessage(e)
      )
    })
    i <- i + 1L
  }
  do.call(rbind, rows)
}

if (!has_cuda() || !fastPLS:::lda_cuda_native_available()) {
  stop("This benchmark requires CUDA and native CUDA LDA kernels.", call. = FALSE)
}

out_dir <- Sys.getenv("FASTPLS_CUDA_LDA_OUT", "benchmark_results/cuda_lda")
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)
ncomp <- as.integer(Sys.getenv("FASTPLS_CUDA_LDA_NCOMP", "50"))
reps <- as.integer(Sys.getenv("FASTPLS_CUDA_LDA_REPS", "3"))

tasks <- list(load_cifar100(), load_singlecell())
rows <- list()
i <- 1L
for (task in tasks) {
  for (method in c("plssvd", "simpls")) {
    rows[[i]] <- run_head_benchmark(task, method, ncomp, reps)
    i <- i + 1L
    rows[[i]] <- run_end_to_end(task, method, ncomp)
    i <- i + 1L
  }
}

results <- do.call(rbind, rows)
raw_path <- file.path(out_dir, "cuda_lda_cifar100_singlecell_raw.csv")
summary_path <- file.path(out_dir, "cuda_lda_cifar100_singlecell_summary.csv")
write.csv(results, raw_path, row.names = FALSE)

median_or_na <- function(x) {
  x <- x[is.finite(x)]
  if (!length(x)) return(NA_real_)
  median(x)
}

groups <- split(
  results,
  interaction(
    results$dataset,
    results$scope,
    results$method,
    results$classifier,
    results$native_backend,
    drop = TRUE,
    lex.order = TRUE
  )
)
summary <- do.call(rbind, lapply(groups, function(x) {
  data.frame(
    dataset = x$dataset[1],
    scope = x$scope[1],
    method = x$method[1],
    classifier = x$classifier[1],
    native_backend = x$native_backend[1],
    train_ms = median_or_na(x$train_ms),
    predict_ms = median_or_na(x$predict_ms),
    total_head_ms = median_or_na(x$total_head_ms),
    elapsed_ms = median_or_na(x$elapsed_ms),
    accuracy = median_or_na(x$accuracy)
  )
}))
row.names(summary) <- NULL
write.csv(summary, summary_path, row.names = FALSE)

cat("Wrote:\n")
cat(" -", normalizePath(raw_path), "\n")
cat(" -", normalizePath(summary_path), "\n")
print(summary, row.names = FALSE)
