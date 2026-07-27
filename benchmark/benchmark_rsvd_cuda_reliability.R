#!/usr/bin/env Rscript

args <- commandArgs(trailingOnly = TRUE)
value_arg <- function(name, default) {
  hit <- grep(paste0("^--", name, "="), args, value = TRUE)
  if (!length(hit)) return(default)
  sub(paste0("^--", name, "="), "", hit[[1L]])
}

out_dir <- normalizePath(
  value_arg("out", "benchmark_results/rsvd_cuda_reliability"),
  mustWork = FALSE
)
lib <- value_arg("lib", "")
if (nzchar(lib)) .libPaths(c(lib, .libPaths()))
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

suppressPackageStartupMessages(library(fastPLS))
if (!has_cuda()) stop("This validation requires a CUDA-enabled fastPLS build.")

one_hot <- function(y) {
  y <- factor(y)
  out <- matrix(0, length(y), nlevels(y))
  out[cbind(seq_along(y), as.integer(y))] <- 1
  colnames(out) <- levels(y)
  out
}

relative_error <- function(x, ref) {
  sqrt(sum((x - ref)^2)) / max(sqrt(sum(ref^2)), .Machine$double.eps)
}

prediction_at <- function(fit, component, components) {
  value <- fit$Ypred
  if (is.list(value)) return(as.matrix(value[[paste0("ncomp=", component)]]))
  if (length(dim(value)) == 3L) {
    index <- match(component, as.integer(components))
    if (is.na(index)) stop("Requested component is not present in the prediction cube.")
    return(value[, , index, drop = FALSE][, , 1L])
  }
  as.matrix(value)
}

make_high_rank <- function(seed = 123L) {
  set.seed(seed)
  n_train <- 180L
  n_test <- 60L
  p <- 90L
  q <- 40L
  rank <- 20L
  x <- matrix(rnorm((n_train + n_test) * p), ncol = p)
  beta <- matrix(rnorm(p * rank), p, rank) %*%
    matrix(rnorm(rank * q), rank, q) / sqrt(p * rank)
  y <- x %*% beta + matrix(rnorm((n_train + n_test) * q, sd = 0.05), ncol = q)
  list(
    dataset = "synthetic_high_rank_response",
    Xtrain = x[seq_len(n_train), , drop = FALSE],
    Ytrain = y[seq_len(n_train), , drop = FALSE],
    Xtest = x[n_train + seq_len(n_test), , drop = FALSE],
    Ytest = y[n_train + seq_len(n_test), , drop = FALSE],
    ncomp = c(2L, 5L, 10L, 20L),
    classification = FALSE
  )
}

make_metref <- function() {
  if (!requireNamespace("KODAMA", quietly = TRUE)) return(NULL)
  env <- new.env(parent = emptyenv())
  data("MetRef", package = "KODAMA", envir = env)
  x <- env$MetRef$data
  x <- x[, colSums(x) != 0, drop = FALSE]
  x <- KODAMA::normalization(x)$newXtrain
  x <- KODAMA::scaling(x)$newXtrain
  labels <- factor(env$MetRef$donor)
  set.seed(123L)
  test <- sample(seq_len(nrow(x)), min(100L, floor(nrow(x) / 5L)))
  train <- setdiff(seq_len(nrow(x)), test)
  encoded <- one_hot(labels)
  list(
    dataset = "MetRef",
    Xtrain = as.matrix(x[train, , drop = FALSE]),
    Ytrain = encoded[train, , drop = FALSE],
    Xtest = as.matrix(x[test, , drop = FALSE]),
    Ytest = encoded[test, , drop = FALSE],
    ncomp = c(2L, 5L, 10L, 18L),
    classification = TRUE
  )
}

tasks <- Filter(Negate(is.null), list(make_high_rank(), make_metref()))
rows <- list()
for (task in tasks) {
  message(format(Sys.time()), " ", task$dataset)
  reference_time <- system.time({
    reference <- pls(
      task$Xtrain, task$Ytrain, task$Xtest, task$Ytest,
      ncomp = task$ncomp, method = "simpls", backend = "cpu",
      svd.method = "irlba", scaling = "none", fit = FALSE,
      return_variance = FALSE
    )
  })[["elapsed"]]

  for (oversample in c(10L, 20L)) {
  for (power in c(1L, 2L, 4L)) {
    cuda_time <- system.time({
      candidate <- pls(
        task$Xtrain, task$Ytrain, task$Xtest, task$Ytest,
        ncomp = task$ncomp, method = "simpls", backend = "cuda",
        svd.method = "rsvd", oversample = oversample, power = power,
        seed = 123L, scaling = "none", fit = FALSE,
        return_variance = FALSE
      )
    })[["elapsed"]]
    for (component in task$ncomp) {
      ref <- prediction_at(reference, component, task$ncomp)
      pred <- prediction_at(candidate, component, task$ncomp)
      correlation <- suppressWarnings(stats::cor(as.vector(pred), as.vector(ref)))
      label_agreement <- if (task$classification) {
        mean(max.col(pred, ties.method = "first") ==
               max.col(ref, ties.method = "first"))
      } else {
        NA_real_
      }
      metric_ref <- if (task$classification) {
        mean(max.col(ref) == max.col(task$Ytest))
      } else {
        sqrt(mean((ref - task$Ytest)^2))
      }
      metric_candidate <- if (task$classification) {
        mean(max.col(pred) == max.col(task$Ytest))
      } else {
        sqrt(mean((pred - task$Ytest)^2))
      }
      pred_error <- relative_error(pred, ref)
      metric_difference <- abs(metric_candidate - metric_ref)
      passed <- pred_error <= 0.05 + 1e-12 &&
        correlation >= 0.99 &&
        metric_difference <= 0.01 + 1e-12 &&
        (!task$classification || label_agreement >= 0.99)
      rows[[length(rows) + 1L]] <- data.frame(
        dataset = task$dataset,
        backend = "cuda",
        solver = "rsvd",
        oversample = oversample,
        power = power,
        ncomp = component,
        reference_time_sec = as.numeric(reference_time),
        candidate_time_sec = as.numeric(cuda_time),
        prediction_relative_error = pred_error,
        prediction_correlation = correlation,
        label_agreement = label_agreement,
        reference_metric = metric_ref,
        candidate_metric = metric_candidate,
        metric_absolute_difference = metric_difference,
        approximation_tolerance_pass = passed,
        diagnostic_status = candidate$diagnostics$status,
        gpu_resident = isTRUE(candidate$gpu_resident),
        stringsAsFactors = FALSE
      )
    }
    if (exists("cuda_reset_workspace", envir = asNamespace("fastPLS"), inherits = FALSE)) {
      get("cuda_reset_workspace", envir = asNamespace("fastPLS"))()
    }
  }
  }
}

results <- do.call(rbind, rows)
write.csv(results, file.path(out_dir, "rsvd_cuda_reliability.csv"), row.names = FALSE)
writeLines(capture.output(sessionInfo()), file.path(out_dir, "session_info.txt"))
print(results)
if (any(!results$approximation_tolerance_pass, na.rm = TRUE)) {
  message("One or more CUDA rSVD rows failed the prespecified approximation criteria.")
}
