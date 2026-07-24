#!/usr/bin/env Rscript

# Summarize isolated CPU/CUDA repetitions of the fixed NMR final-validation
# protocol. The selected component count and held-out split are fixed upstream.

args <- commandArgs(trailingOnly = TRUE)
input_dir <- if (length(args)) args[[1L]] else "benchmark_results/review_nmr_replicates_100"
out_file <- if (length(args) >= 2L) args[[2L]] else file.path(input_dir, "nmr_replicates_summary.csv")

rss_mb <- function(log_file) {
  line <- grep("Maximum resident set size", readLines(log_file, warn = FALSE), value = TRUE)
  if (!length(line)) return(NA_real_)
  as.numeric(sub(".*: ", "", line[[1L]])) / 1024
}

rep_dirs <- list.dirs(input_dir, recursive = FALSE, full.names = TRUE)
rep_dirs <- rep_dirs[grepl("rep_[0-9]+$", rep_dirs)]
if (!length(rep_dirs)) stop("No rep_<number> directories found.", call. = FALSE)

rows <- do.call(rbind, lapply(rep_dirs, function(rep_dir) {
  rep_id <- as.integer(sub(".*rep_", "", rep_dir))
  do.call(rbind, lapply(c("cpu", "cuda"), function(backend) {
    summary_file <- file.path(rep_dir, paste0("nmr_final_", backend, "_summary.csv"))
    x <- utils::read.csv(summary_file, check.names = FALSE)
    x$backend <- NULL
    gpu_file <- file.path(rep_dir, "nmr_final_cuda_gpu_mem_peak_mb.txt")
    data.frame(
      repetition = rep_id,
      backend = toupper(backend),
      x,
      host_rss_mb = rss_mb(file.path(rep_dir, paste0("nmr_final_", backend, ".log"))),
      gpu_peak_mb = if (backend == "cuda" && file.exists(gpu_file)) as.numeric(readLines(gpu_file, warn = FALSE)[[1L]]) else 0,
      stringsAsFactors = FALSE
    )
  }))
}))

metric_cols <- c("fit_time_sec", "predict_time_sec", "total_time_sec", "R2", "Q2", "RMSD", "MAE", "median_sample_RMSD", "host_rss_mb", "gpu_peak_mb")
summary <- do.call(rbind, lapply(split(rows, rows$backend), function(x) {
  values <- vapply(metric_cols, function(metric) stats::median(x[[metric]], na.rm = TRUE), numeric(1L))
  iqr <- vapply(metric_cols, function(metric) stats::IQR(x[[metric]], na.rm = TRUE), numeric(1L))
  out <- data.frame(backend = x$backend[[1L]], n_repetitions = nrow(x), stringsAsFactors = FALSE)
  for (metric in metric_cols) {
    out[[paste0(metric, "_median")]] <- values[[metric]]
    out[[paste0(metric, "_iqr")]] <- iqr[[metric]]
  }
  out
}))

utils::write.csv(rows, file.path(input_dir, "nmr_replicates_raw.csv"), row.names = FALSE)
utils::write.csv(summary, out_file, row.names = FALSE)
print(rows)
print(summary)
