#!/usr/bin/env Rscript

# Matched benchmark of the compiled CV engine against an explicit R fold loop.
# Both routes fit the same fastPLS estimator on the same prespecified folds.

options(stringsAsFactors = FALSE)

parse_args <- function(x = commandArgs(trailingOnly = TRUE)) {
  out <- list()
  for (item in x) {
    if (!startsWith(item, "--")) next
    kv <- strsplit(substring(item, 3L), "=", fixed = TRUE)[[1L]]
    out[[gsub("-", "_", kv[[1L]])]] <-
      if (length(kv) > 1L) paste(kv[-1L], collapse = "=") else "TRUE"
  }
  out
}

args <- parse_args()
arg <- function(name, default = NULL) {
  value <- args[[name]]
  if (is.null(value) || !nzchar(value)) default else value
}

task_path <- normalizePath(arg("task"), mustWork = TRUE)
output_path <- arg("output", "cv_compiled_vs_r_loop.csv")
dataset <- arg("dataset", sub("_task\\.rds$", "", basename(task_path)))
method <- arg("method", "simpls")
backend <- arg("backend", "cpu")
svd_method <- arg("svd_method", "irlba")
classifier <- arg("classifier", "argmax")
ncomp <- as.integer(arg("ncomp", "10"))
kfold <- as.integer(arg("kfold", "10"))
reps <- as.integer(arg("reps", "3"))
replicate_start <- as.integer(arg("replicate", "1"))
seed <- as.integer(arg("seed", "123"))

if (!method %in% c("plssvd", "simpls", "opls", "kernelpls")) {
  stop("Unsupported method: ", method)
}
if (!backend %in% c("cpu", "cuda")) stop("backend must be cpu or cuda")
if (!svd_method %in% c("irlba", "rsvd")) stop("svd_method must be irlba or rsvd")
if (backend == "cuda" && svd_method != "rsvd") {
  stop("CUDA supports rsvd only")
}
if (!classifier %in% c("argmax", "lda")) stop("Unsupported classifier")

suppressPackageStartupMessages(library(fastPLS))
if (backend == "cuda" && !isTRUE(has_cuda())) stop("CUDA is unavailable")

task <- readRDS(task_path)
X <- rbind(as.matrix(task$Xtrain), as.matrix(task$Xtest))
classification <- identical(task$task_type, "classification") ||
  is.factor(task$Ytrain)
if (classification) {
  lev <- levels(task$Ytrain)
  Y <- factor(
    c(as.character(task$Ytrain), as.character(task$Ytest)),
    levels = lev
  )
} else {
  Y <- rbind(as.matrix(task$Ytrain), as.matrix(task$Ytest))
}

make_folds <- getFromNamespace(".make_single_cv_folds", "fastPLS")
run_compiled <- getFromNamespace(".pls_cv_compiled", "fastPLS")
run_r_loop <- getFromNamespace(".pls_cv_via_pls", "fastPLS")

# Freeze one stratified/group-compatible partition. Passing its IDs as
# constraints forces both engines to use exactly these sample groups.
fold_groups <- make_folds(
  Ydata = if (classification) Y else Y[, 1L],
  constrain = seq_len(nrow(X)),
  kfold = kfold,
  seed = seed
)

coassignment_equal <- function(a, b) {
  a <- as.integer(a)
  b <- as.integer(b)
  if (length(a) != length(b)) return(FALSE)
  tab <- table(a, b)
  all(rowSums(tab > 0L) == 1L) && all(colSums(tab > 0L) == 1L)
}

extract_prediction <- function(result) {
  if (classification) {
    return(as.character(result$pred[[1L]]))
  }
  as.matrix(result$pred)
}

extract_metric <- function(result) {
  data.frame(
    metric_name = as.character(result$metrics$metric_name[[1L]]),
    metric_value = as.numeric(result$metrics$metric_value[[1L]]),
    stringsAsFactors = FALSE
  )
}

engine_call <- function(engine) {
  if (identical(engine, "compiled")) {
    run_compiled(
      Xdata = X,
      Ydata = Y,
      constrain = fold_groups,
      ncomp = ncomp,
      kfold = kfold,
      scaling = "centering",
      method = method,
      backend = if (backend == "cpu") "cpp" else "cuda",
      svd.method = svd_method,
      rsvd_oversample = 10L,
      rsvd_power = 1L,
      seed = seed,
      return_scores = TRUE,
      classifier = classifier,
      store_predictions = TRUE,
      selection_metric = if (classification) "accuracy" else "rmsd"
    )
  } else {
    run_r_loop(
      Xdata = X,
      Ydata = Y,
      constrain = fold_groups,
      ncomp = ncomp,
      kfold = kfold,
      scaling = "centering",
      method = method,
      backend = backend,
      svd.method = svd_method,
      rsvd_oversample = 10L,
      rsvd_power = 1L,
      seed = seed,
      classifier = classifier,
      return_scores = TRUE,
      store_predictions = TRUE,
      selection_metric = if (classification) "accuracy" else "rmsd"
    )
  }
}

rows <- vector("list", reps)
for (replicate_id in seq.int(replicate_start, length.out = reps)) {
  order <- if (replicate_id %% 2L) c("compiled", "r_loop") else c("r_loop", "compiled")
  result <- list()
  elapsed <- numeric()
  errors <- character()

  for (engine in order) {
    gc()
    timing <- system.time({
      value <- tryCatch(
        engine_call(engine),
        error = function(e) {
          errors[[engine]] <<- conditionMessage(e)
          NULL
        }
      )
    })
    result[[engine]] <- value
    elapsed[[engine]] <- unname(timing[["elapsed"]])
  }

  if (is.null(result$compiled) || is.null(result$r_loop)) {
    rows[[replicate_id]] <- data.frame(
      dataset, task_type = if (classification) "classification" else "regression",
      n = nrow(X), p = ncol(X),
      q = if (classification) nlevels(Y) else ncol(Y),
      method, backend, svd_method, classifier, ncomp, kfold, replicate_id,
      compiled_sec = elapsed[["compiled"]],
      r_loop_sec = elapsed[["r_loop"]],
      speedup_r_loop_over_compiled = NA_real_,
      metric_name = NA_character_,
      compiled_metric = NA_real_,
      r_loop_metric = NA_real_,
      metric_abs_diff = NA_real_,
      prediction_agreement = NA_real_,
      prediction_correlation = NA_real_,
      max_abs_prediction_diff = NA_real_,
      identical_fold_partition = NA,
      status = "failed",
      error = paste(unname(errors), collapse = " | "),
      stringsAsFactors = FALSE
    )
    next
  }

  pred_compiled <- extract_prediction(result$compiled)
  pred_r_loop <- extract_prediction(result$r_loop)
  metric_compiled <- extract_metric(result$compiled)
  metric_r_loop <- extract_metric(result$r_loop)

  if (classification) {
    agreement <- mean(pred_compiled == pred_r_loop, na.rm = TRUE)
    pred_cor <- NA_real_
    max_diff <- NA_real_
  } else {
    agreement <- NA_real_
    keep <- is.finite(pred_compiled) & is.finite(pred_r_loop)
    pred_cor <- if (sum(keep) > 1L) cor(pred_compiled[keep], pred_r_loop[keep]) else NA_real_
    max_diff <- if (any(keep)) max(abs(pred_compiled[keep] - pred_r_loop[keep])) else NA_real_
  }

  rows[[replicate_id]] <- data.frame(
    dataset,
    task_type = if (classification) "classification" else "regression",
    n = nrow(X),
    p = ncol(X),
    q = if (classification) nlevels(Y) else ncol(Y),
    method, backend, svd_method, classifier, ncomp, kfold, replicate_id,
    compiled_sec = elapsed[["compiled"]],
    r_loop_sec = elapsed[["r_loop"]],
    speedup_r_loop_over_compiled = elapsed[["r_loop"]] / elapsed[["compiled"]],
    metric_name = metric_compiled$metric_name,
    compiled_metric = metric_compiled$metric_value,
    r_loop_metric = metric_r_loop$metric_value,
    metric_abs_diff = abs(metric_compiled$metric_value - metric_r_loop$metric_value),
    prediction_agreement = agreement,
    prediction_correlation = pred_cor,
    max_abs_prediction_diff = max_diff,
    identical_fold_partition = coassignment_equal(result$compiled$fold, result$r_loop$fold),
    status = "success",
    error = "",
    stringsAsFactors = FALSE
  )
}

out <- do.call(rbind, rows)
dir.create(dirname(output_path), recursive = TRUE, showWarnings = FALSE)
write.csv(out, output_path, row.names = FALSE)
print(out)
