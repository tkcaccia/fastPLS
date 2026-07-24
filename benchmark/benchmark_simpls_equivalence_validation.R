#!/usr/bin/env Rscript

# Controlled numerical validation of SIMPLS. This script compares the compiled
# implementation with pls::simpls.fit on fixed synthetic data. With --svd=irlba
# the comparison checks the deterministic de Jong estimator; --svd=rsvd is
# retained as a separate, explicitly approximate low-rank comparison.

library(fastPLS)

args <- commandArgs(trailingOnly = TRUE)
get_arg <- function(name, default) {
  key <- paste0("--", name, "=")
  hit <- args[startsWith(args, key)]
  if (!length(hit)) return(default)
  sub(key, "", hit[[1L]], fixed = TRUE)
}
svd_method <- match.arg(get_arg("svd", "irlba"), c("irlba", "rsvd"))

if (!requireNamespace("pls", quietly = TRUE)) {
  stop("The 'pls' package is required for this validation.", call. = FALSE)
}

out_dir <- Sys.getenv("FASTPLS_EQUIVALENCE_RESULTS", "benchmark_results/simpls_equivalence")
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

with_fast_optimized <- function(value, code) {
  old <- Sys.getenv("FASTPLS_FAST_OPTIMIZED", unset = NA_character_)
  on.exit({
    if (is.na(old)) Sys.unsetenv("FASTPLS_FAST_OPTIMIZED") else Sys.setenv(FASTPLS_FAST_OPTIMIZED = old)
  }, add = TRUE)
  Sys.setenv(FASTPLS_FAST_OPTIMIZED = as.integer(value))
  force(code)
}

principal_angle_degrees <- function(A, B) {
  qa <- qr.Q(qr(A))
  qb <- qr.Q(qr(B))
  singular <- svd(crossprod(qa, qb), nu = 0L, nv = 0L)$d
  acos(pmin(1, pmax(-1, singular))) * 180 / pi
}

make_task <- function(kind, seed = 123L, n_train = 180L, n_test = 60L, p = 60L, q = 4L) {
  set.seed(seed)
  n <- n_train + n_test
  latent <- matrix(rnorm(n * 5L), n, 5L)
  P <- matrix(rnorm(p * 5L), p, 5L)
  C <- matrix(rnorm(q * 5L), q, 5L)
  if (identical(kind, "ill_conditioned")) {
    P[, 2L] <- P[, 1L] + rnorm(p, sd = 1e-4)
    P[, 3L] <- P[, 1L] - P[, 2L] + rnorm(p, sd = 1e-4)
  }
  X <- latent %*% t(P) + matrix(rnorm(n * p, sd = 0.08), n, p)
  if (identical(kind, "rank_deficient")) {
    X[, (p - 9L):p] <- X[, 1:10] %*% matrix(rep(1, 100L), 10L, 10L)
  }
  Y <- latent %*% t(C) + matrix(rnorm(n * q, sd = 0.08), n, q)
  list(
    Xtrain = X[seq_len(n_train), , drop = FALSE],
    Ytrain = Y[seq_len(n_train), , drop = FALSE],
    Xtest = X[(n_train + 1L):n, , drop = FALSE],
    Ytest = Y[(n_train + 1L):n, , drop = FALSE]
  )
}

prediction_slice <- function(x, k) {
  if (is.list(x)) return(as.matrix(x[[paste0("ncomp=", k)]]))
  if (length(dim(x)) == 3L) return(x[, , dim(x)[3L], drop = FALSE][, , 1L])
  as.matrix(x)
}

coefficient_slice <- function(x, k) {
  if (is.null(x)) return(NULL)
  if (length(dim(x)) == 3L) return(x[, , dim(x)[3L], drop = FALSE][, , 1L])
  as.matrix(x)
}

fit_fastpls <- function(task, optimized, ncomp) {
  elapsed <- system.time({
    fit <- with_fast_optimized(optimized, pls(
      task$Xtrain, task$Ytrain,
      Xtest = task$Xtest, Ytest = task$Ytest,
      ncomp = ncomp, scaling = "centering", method = "simpls",
      backend = "cpu", svd.method = svd_method, fit = TRUE,
      return_variance = FALSE, seed = 123L
    ))
  })[["elapsed"]]
  list(
    label = if (optimized) "fastPLS_SIMPLS_accelerated" else "fastPLS_SIMPLS_reference_execution",
    fit = fit,
    prediction = prediction_slice(fit$Ypred, ncomp),
    elapsed = elapsed,
    subspace = fit$R[, seq_len(ncomp), drop = FALSE],
    coefficient = coefficient_slice(fit$B, ncomp)
  )
}

fit_reference <- function(task, ncomp) {
  elapsed <- system.time({
    fit <- pls::simpls.fit(task$Xtrain, task$Ytrain, ncomp = ncomp)
  })[["elapsed"]]
  B <- fit$coefficients[, , ncomp, drop = FALSE][, , 1L]
  xcentered <- sweep(task$Xtest, 2L, fit$Xmeans, "-")
  prediction <- sweep(xcentered %*% B, 2L, fit$Ymeans, "+")
  list(
    label = "pls::simpls.fit_reference",
    fit = fit,
    prediction = prediction,
    elapsed = elapsed,
    subspace = fit$projection[, seq_len(ncomp), drop = FALSE],
    coefficient = B
  )
}

rows <- list()
row_id <- 1L
for (scenario in c("well_conditioned", "ill_conditioned", "rank_deficient")) {
  task <- make_task(scenario)
  ncomp <- 5L
  models <- list(
    fit_fastpls(task, optimized = FALSE, ncomp = ncomp),
    fit_fastpls(task, optimized = TRUE, ncomp = ncomp),
    fit_reference(task, ncomp = ncomp)
  )
  reference <- models[[3L]]
  for (model in models) {
    err <- model$prediction - task$Ytest
    ref_err <- model$prediction - reference$prediction
    coef_delta <- if (!is.null(model$coefficient)) {
      sqrt(sum((model$coefficient - reference$coefficient)^2)) /
        max(sqrt(sum(reference$coefficient^2)), .Machine$double.eps)
    } else {
      NA_real_
    }
    rows[[row_id]] <- data.frame(
      scenario = scenario,
      implementation = model$label,
      svd_method = svd_method,
      n_train = nrow(task$Xtrain), p = ncol(task$Xtrain), q = ncol(task$Ytrain), ncomp = ncomp,
      elapsed_sec = model$elapsed,
      test_rmsd = sqrt(mean(err^2)),
      prediction_correlation = cor(as.vector(model$prediction), as.vector(reference$prediction)),
      relative_prediction_error = sqrt(sum(ref_err^2)) /
        max(sqrt(sum(reference$prediction^2)), .Machine$double.eps),
      coefficient_relative_error = coef_delta,
      max_principal_angle_degrees = max(principal_angle_degrees(model$subspace, reference$subspace)),
      status = "ok",
      stringsAsFactors = FALSE
    )
    row_id <- row_id + 1L
  }
}

results <- do.call(rbind, rows)
stamp <- format(Sys.time(), "%Y%m%d_%H%M%S")
csv <- file.path(out_dir, paste0("simpls_equivalence_", stamp, ".csv"))
utils::write.csv(results, csv, row.names = FALSE)
saveRDS(list(results = results, svd_method = svd_method, session = utils::sessionInfo()), sub("[.]csv$", ".rds", csv))
print(results)
cat("Saved: ", normalizePath(csv, winslash = "/", mustWork = FALSE), "\n", sep = "")
