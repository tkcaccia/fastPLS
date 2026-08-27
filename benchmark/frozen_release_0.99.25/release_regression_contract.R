#!/usr/bin/env Rscript

args <- commandArgs(trailingOnly = TRUE)
value_arg <- function(name, default = "") {
  hit <- grep(paste0("^--", name, "="), args, value = TRUE)
  if (!length(hit)) return(default)
  sub(paste0("^--", name, "="), "", hit[[1L]])
}
lib <- value_arg("lib")
out <- value_arg("out", "benchmark_results/frozen_release_0.99.25/regression_contract")
if (nzchar(lib)) .libPaths(c(lib, .libPaths()))
dir.create(out, recursive = TRUE, showWarnings = FALSE)
suppressPackageStartupMessages(library(fastPLS))

stopifnot(as.character(packageVersion("fastPLS")) == "0.99.25")
`%||%` <- function(x, y) if (is.null(x) || !length(x)) y else x
set.seed(20260825)
n <- 96L; p <- 24L; q <- 4L
X <- matrix(rnorm(n * p), n, p)
B <- matrix(rnorm(p * q), p, q)
Y <- X %*% B + matrix(rnorm(n * q, sd = 0.05), n, q)
train <- seq_len(72L)
test <- setdiff(seq_len(n), train)
components <- c(1L, 2L, 4L)

capture_fit <- function(solver) {
  warnings <- character()
  fit <- withCallingHandlers(
    pls(
      X[train, , drop = FALSE], Y[train, , drop = FALSE],
      X[test, , drop = FALSE], Y[test, , drop = FALSE],
      ncomp = components, method = "simpls", backend = "cpu",
      svd.method = solver, fit = FALSE, proj = TRUE,
      return_variance = FALSE, oversample = 20L, power = 2L, seed = 123L
    ),
    warning = function(w) {
      warnings <<- c(warnings, conditionMessage(w))
      invokeRestart("muffleWarning")
    }
  )
  list(fit = fit, warnings = warnings)
}

extract_prediction <- function(fit, component) {
  key <- paste0("ncomp=", component)
  value <- fit$Ypred
  if (is.list(value)) return(as.matrix(value[[key]]))
  if (length(dim(value)) == 3L) {
    index <- match(key, dimnames(value)[[3L]])
    if (is.na(index)) index <- match(component, components)
    return(value[, , index, drop = FALSE][, , 1L])
  }
  as.matrix(value)
}

rows <- list()
objects <- list()
for (solver in c("irlba", "rsvd")) {
  first <- capture_fit(solver)
  second <- capture_fit(solver)
  predictions_1 <- lapply(components, function(k) extract_prediction(first$fit, k))
  predictions_2 <- lapply(components, function(k) extract_prediction(second$fit, k))
  prediction_error <- max(vapply(seq_along(components), function(i) {
    max(abs(predictions_1[[i]] - predictions_2[[i]]))
  }, numeric(1L)))
  coefficient_error <- if (!is.null(first$fit$B) && !is.null(second$fit$B)) {
    max(abs(as.numeric(first$fit$B) - as.numeric(second$fit$B)))
  } else {
    NA_real_
  }
  diagnostics <- first$fit$diagnostics
  rows[[length(rows) + 1L]] <- data.frame(
    solver = solver,
    package_version = as.character(packageVersion("fastPLS")),
    prediction_repeat_max_abs_diff = prediction_error,
    coefficient_repeat_max_abs_diff = coefficient_error,
    diagnostics_status = diagnostics$status %||% NA_character_,
    approximation_audited = diagnostics$approximation_audited %||% NA,
    warning_count = length(first$warnings),
    component_names_ok = identical(names(first$fit$Q2Y), paste0("ncomp=", components)),
    status = if (isTRUE(prediction_error == 0) &&
                 (is.na(coefficient_error) || isTRUE(coefficient_error == 0))) "pass" else "fail",
    stringsAsFactors = FALSE
  )
  objects[[solver]] <- list(
    predictions = predictions_1,
    coefficients = first$fit$B,
    diagnostics = diagnostics,
    warnings = first$warnings
  )
}

result <- do.call(rbind, rows)
write.csv(result, file.path(out, "release_regression_contract.csv"), row.names = FALSE)
saveRDS(objects, file.path(out, "release_regression_contract.rds"), compress = "xz")
writeLines(capture.output(sessionInfo()), file.path(out, "session_info.txt"))
print(result)
if (any(result$status != "pass")) quit(save = "no", status = 1L)
