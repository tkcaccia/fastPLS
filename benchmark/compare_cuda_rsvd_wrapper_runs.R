#!/usr/bin/env Rscript

suppressPackageStartupMessages(library(data.table))

out_dir <- path.expand(Sys.getenv("FASTPLS_COMPARE_OUTDIR", file.path(getwd(), "benchmark_results_cuda_rsvd_compare")))

base <- fread(file.path(out_dir, "baseline_summary.csv"))
mod <- fread(file.path(out_dir, "modified_summary.csv"))
res <- merge(base, mod, by = "dataset", suffixes = c("_baseline", "_modified"))
res[, speedup := elapsed_baseline / elapsed_modified]

cos_mean <- function(A, B, by = c("col", "row")) {
  by <- match.arg(by)
  if (is.null(A) || is.null(B) || length(A) == 0L || length(B) == 0L) return(NA_real_)
  if (by == "col") {
    kk <- min(ncol(A), ncol(B))
    vals <- numeric(kk)
    for (j in seq_len(kk)) {
      a <- A[, j]
      b <- B[, j]
      vals[j] <- abs(sum(a * b) / sqrt(sum(a * a) * sum(b * b)))
    }
  } else {
    kk <- min(nrow(A), nrow(B))
    vals <- numeric(kk)
    for (j in seq_len(kk)) {
      a <- A[j, ]
      b <- B[j, ]
      vals[j] <- abs(sum(a * b) / sqrt(sum(a * a) * sum(b * b)))
    }
  }
  mean(vals)
}

extra <- rbindlist(lapply(res$dataset, function(ds) {
  b <- readRDS(file.path(out_dir, paste0("baseline_", ds, ".rds")))
  m <- readRDS(file.path(out_dir, paste0("modified_", ds, ".rds")))
  data.table(
    dataset = ds,
    s_max_abs_diff = max(abs(b$s - m$s)),
    s_rel_max = max(abs(b$s - m$s) / pmax(abs(b$s), 1e-12)),
    U_cos_mean = cos_mean(b$U, m$U, "col"),
    Vt_cos_mean = cos_mean(b$Vt, m$Vt, "row")
  )
}))

final <- merge(res, extra, by = "dataset")
fwrite(final, file.path(out_dir, "comparison_summary.csv"))
print(final)
