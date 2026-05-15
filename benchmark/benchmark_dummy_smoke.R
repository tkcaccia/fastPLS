#!/usr/bin/env Rscript

# Fast local benchmark smoke test for the public fastPLS API.
# It intentionally uses synthetic data so it can run without the real benchmark
# datasets or optional third-party PLS packages.

options(stringsAsFactors = FALSE)
suppressPackageStartupMessages(library(fastPLS))

parse_args <- function(args = commandArgs(trailingOnly = TRUE)) {
  out <- list()
  for (arg in args) {
    if (!startsWith(arg, "--")) next
    kv <- substring(arg, 3L)
    bits <- strsplit(kv, "=", fixed = TRUE)[[1L]]
    key <- gsub("-", "_", bits[[1L]], fixed = TRUE)
    out[[key]] <- if (length(bits) > 1L) paste(bits[-1L], collapse = "=") else "TRUE"
  }
  out
}

args <- parse_args()
arg <- function(key, default = NULL) {
  val <- args[[key]]
  if (is.null(val) || !nzchar(val)) default else val
}

extract_last_prediction <- function(x) {
  if (is.data.frame(x)) return(x[[ncol(x)]])
  if (is.array(x) && length(dim(x)) == 3L) return(x[, , dim(x)[3L], drop = TRUE])
  x
}

make_classification <- function(n = 90L, p = 14L, k = 3L) {
  y <- factor(rep(seq_len(k), length.out = n), labels = paste0("C", seq_len(k)))
  X <- matrix(rnorm(n * p, sd = 0.4), n, p)
  for (i in seq_len(k)) {
    cols <- i:min(p, i + 3L)
    X[y == levels(y)[i], cols] <- X[y == levels(y)[i], cols] + 1.5
  }
  idx <- sample(seq_len(n), round(n * 0.25))
  list(
    Xtrain = X[-idx, , drop = FALSE],
    Ytrain = y[-idx],
    Xtest = X[idx, , drop = FALSE],
    Ytest = y[idx]
  )
}

make_regression <- function(n = 90L, p = 14L, q = 3L) {
  X <- matrix(rnorm(n * p), n, p)
  B <- matrix(rnorm(p * q), p, q)
  Y <- X %*% B + matrix(rnorm(n * q, sd = 0.1), n, q)
  idx <- sample(seq_len(n), round(n * 0.25))
  list(
    Xtrain = X[-idx, , drop = FALSE],
    Ytrain = Y[-idx, , drop = FALSE],
    Xtest = X[idx, , drop = FALSE],
    Ytest = Y[idx, , drop = FALSE]
  )
}

score_prediction <- function(pred, truth) {
  if (is.factor(truth)) {
    pred <- factor(pred, levels = levels(truth))
    return(list(metric_name = "accuracy",
                metric_value = mean(as.character(pred) == as.character(truth))))
  }
  pred <- as.matrix(pred)
  truth <- as.matrix(truth)
  if (!all(dim(pred) == dim(truth))) {
    stop(
      "Bad regression prediction dim: ",
      paste(dim(pred), collapse = "x"),
      " expected ",
      paste(dim(truth), collapse = "x")
    )
  }
  list(metric_name = "rmse", metric_value = sqrt(mean((pred - truth)^2)))
}

run_one <- function(task, method, svd_method, backend,
                    classifier = "argmax", gaussian_y = FALSE) {
  dat <- if (identical(task, "classification")) {
    make_classification()
  } else {
    make_regression()
  }

  started <- proc.time()[3L]
  fit <- fastPLS::pls(
    dat$Xtrain, dat$Ytrain, dat$Xtest, dat$Ytest,
    ncomp = 2L,
    method = method,
    svd.method = svd_method,
    backend = backend,
    classifier = classifier,
    gaussian_y = gaussian_y,
    fit = FALSE,
    return_variance = FALSE,
    proj = FALSE,
    seed = 123L
  )
  pred_obj <- stats::predict(fit, dat$Xtest, Ytest = dat$Ytest)
  elapsed_ms <- as.numeric(proc.time()[3L] - started) * 1000
  metric <- score_prediction(extract_last_prediction(pred_obj$Ypred), dat$Ytest)

  data.frame(
    task = task,
    method = method,
    svd_method = svd_method,
    backend = backend,
    classifier = classifier,
    gaussian_y = gaussian_y,
    runtime_ms = elapsed_ms,
    metric_name = metric$metric_name,
    metric_value = metric$metric_value,
    status = "ok",
    error = "",
    stringsAsFactors = FALSE
  )
}

safe_run_one <- function(...) {
  dots <- list(...)
  tryCatch(
    run_one(...),
    error = function(e) {
      data.frame(
        task = dots[[1L]],
        method = dots[[2L]],
        svd_method = dots[[3L]],
        backend = dots[[4L]],
        classifier = if (length(dots) >= 5L) dots[[5L]] else "argmax",
        gaussian_y = if (length(dots) >= 6L) dots[[6L]] else FALSE,
        runtime_ms = NA_real_,
        metric_name = NA_character_,
        metric_value = NA_real_,
        status = "error",
        error = conditionMessage(e),
        stringsAsFactors = FALSE
      )
    }
  )
}

set.seed(as.integer(arg("seed", "123")))
methods <- c("plssvd", "simpls", "opls", "kernelpls")
svd_methods <- c("irlba", "cpu_rsvd")
rows <- list()

append_row <- function(...) {
  rows[[length(rows) + 1L]] <<- safe_run_one(...)
}

for (method in methods) {
  for (svd_method in svd_methods) {
    append_row("regression", method, svd_method, "cpp", "argmax", FALSE)
    append_row("classification", method, svd_method, "cpp", "argmax", FALSE)
  }
}

for (method in methods) {
  for (svd_method in svd_methods) {
    append_row("classification", method, svd_method, "cpp", "lda", FALSE)
    append_row("classification", method, svd_method, "cpp", "cknn", FALSE)
  }
}

for (method in c("plssvd", "simpls")) {
  for (svd_method in svd_methods) {
    append_row("classification", method, svd_method, "cpp", "argmax", TRUE)
  }
}

if (isTRUE(fastPLS::has_metal())) {
  for (method in methods) {
    append_row("classification", method, "cpu_rsvd", "metal", "argmax", FALSE)
    append_row("classification", method, "cpu_rsvd", "metal", "lda", FALSE)
    append_row("classification", method, "cpu_rsvd", "metal", "cknn", FALSE)
  }
}

if (isTRUE(fastPLS::has_cuda())) {
  for (method in methods) {
    append_row("classification", method, "cpu_rsvd", "cuda", "cknn", FALSE)
  }
}

out <- do.call(rbind, rows)
out_dir <- normalizePath(arg("out_dir", "dummy_benchmark_results"), mustWork = FALSE)
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)
out_file <- file.path(out_dir, "dummy_benchmark_results.csv")
utils::write.csv(out, out_file, row.names = FALSE)

cat("Wrote:", out_file, "\n")
cat("OK rows:", sum(out$status == "ok"), " Errors:", sum(out$status != "ok"), "\n")
print(out)

if (any(out$status != "ok")) {
  cat("\nErrors:\n")
  print(out[out$status != "ok", ], row.names = FALSE)
  quit(status = 1L)
}
