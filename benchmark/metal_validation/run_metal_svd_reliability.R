#!/usr/bin/env Rscript

suppressPackageStartupMessages(library(fastPLS))

repo_dir <- normalizePath(
  file.path(dirname(sub("^--file=", "", commandArgs()[grep("^--file=", commandArgs())])),
            "..", ".."),
  winslash = "/",
  mustWork = TRUE
)
out_dir <- Sys.getenv(
  "FASTPLS_METAL_SVD_OUT",
  file.path(repo_dir, "benchmark_results",
            paste0("metal_svd_reliability_", format(Sys.time(), "%Y%m%d_%H%M%S")))
)
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

matrix_specs <- list(
  tall = c(n = 1000L, p = 100L, k = 20L),
  wide = c(n = 400L, p = 1000L, k = 20L),
  balanced = c(n = 1000L, p = 500L, k = 50L)
)

subspace_angle <- function(reference, candidate) {
  values <- svd(crossprod(reference, candidate), nu = 0, nv = 0)$d
  values <- pmin(1, pmax(0, values))
  max(acos(values)) * 180 / pi
}

rows <- list()
index <- 0L
for (shape in names(matrix_specs)) {
  spec <- matrix_specs[[shape]]
  set.seed(880L + match(shape, names(matrix_specs)))
  A64 <- matrix(rnorm(spec[["n"]] * spec[["p"]]),
                nrow = spec[["n"]], ncol = spec[["p"]])
  exact_time <- system.time({
    exact <- svd(A64, nu = spec[["k"]], nv = spec[["k"]])
  })[["elapsed"]]
  exact_norm <- sqrt(sum(A64^2))

  for (precision in c("float64", "float32")) {
    A <- if (identical(precision, "float32")) float::fl(A64) else A64
    for (backend in c("cpu", "metal")) {
      for (oversample in c(10L, 20L)) {
        for (power in c(1L, 2L)) {
          for (seed in c(101L, 202L, 303L)) {
            warnings_seen <- character()
            elapsed <- system.time({
              fit <- tryCatch(
                withCallingHandlers(
                  fastsvd(
                    A, ncomp = spec[["k"]], backend = backend,
                    method = "rsvd", oversample = oversample,
                    power = power, seed = seed
                  ),
                  warning = function(w) {
                    warnings_seen <<- c(warnings_seen, conditionMessage(w))
                    invokeRestart("muffleWarning")
                  }
                ),
                error = function(e) e
              )
            })[["elapsed"]]
            index <- index + 1L
            if (inherits(fit, "error")) {
              rows[[index]] <- data.frame(
                shape = shape, n = spec[["n"]], p = spec[["p"]],
                k = spec[["k"]], precision = precision, backend = backend,
                oversample = oversample, power = power, seed = seed,
                elapsed_sec = NA_real_, exact_elapsed_sec = exact_time,
                singular_value_rel_error = NA_real_,
                reconstruction_rel_error = NA_real_,
                max_left_angle_deg = NA_real_,
                max_right_angle_deg = NA_real_,
                status = "failed", warnings = paste(unique(warnings_seen), collapse = " | "),
                error = conditionMessage(fit), stringsAsFactors = FALSE
              )
              next
            }
            u <- if (inherits(fit$u, "float32")) float::dbl(fit$u) else as.matrix(fit$u)
            v <- if (inherits(fit$v, "float32")) float::dbl(fit$v) else as.matrix(fit$v)
            d <- as.numeric(fit$d)
            approximation <- tcrossprod(sweep(u, 2L, d, `*`), v)
            rows[[index]] <- data.frame(
              shape = shape, n = spec[["n"]], p = spec[["p"]],
              k = spec[["k"]], precision = precision, backend = backend,
              oversample = oversample, power = power, seed = seed,
              elapsed_sec = unname(elapsed), exact_elapsed_sec = exact_time,
              singular_value_rel_error =
                sqrt(sum((d - exact$d[seq_len(spec[["k"]])])^2)) /
                sqrt(sum(exact$d[seq_len(spec[["k"]])]^2)),
              reconstruction_rel_error = sqrt(sum((A64 - approximation)^2)) / exact_norm,
              max_left_angle_deg =
                subspace_angle(exact$u[, seq_len(spec[["k"]]), drop = FALSE], u),
              max_right_angle_deg =
                subspace_angle(exact$v[, seq_len(spec[["k"]]), drop = FALSE], v),
              status = "success",
              warnings = paste(unique(warnings_seen), collapse = " | "),
              error = "",
              stringsAsFactors = FALSE
            )
          }
        }
      }
    }
  }
  rm(A, A64, exact)
  gc()
}

raw <- do.call(rbind, rows)
write.csv(raw, file.path(out_dir, "metal_svd_reliability_raw.csv"), row.names = FALSE)

group_columns <- c("shape", "n", "p", "k", "precision", "backend",
                   "oversample", "power")
key <- interaction(raw[group_columns], drop = TRUE, lex.order = TRUE)
summary_rows <- lapply(split(raw, key), function(x) {
  data.frame(
    x[1L, group_columns, drop = FALSE],
    successes = sum(x$status == "success"),
    failures = sum(x$status != "success"),
    median_elapsed_sec = median(x$elapsed_sec, na.rm = TRUE),
    median_singular_value_rel_error =
      median(x$singular_value_rel_error, na.rm = TRUE),
    max_singular_value_rel_error =
      max(x$singular_value_rel_error, na.rm = TRUE),
    median_reconstruction_rel_error =
      median(x$reconstruction_rel_error, na.rm = TRUE),
    max_left_angle_deg = max(x$max_left_angle_deg, na.rm = TRUE),
    max_right_angle_deg = max(x$max_right_angle_deg, na.rm = TRUE)
  )
})
summary <- do.call(rbind, summary_rows)
write.csv(summary, file.path(out_dir, "metal_svd_reliability_summary.csv"),
          row.names = FALSE)
writeLines(capture.output(sessionInfo()), file.path(out_dir, "session_info.txt"))
print(summary, row.names = FALSE)
