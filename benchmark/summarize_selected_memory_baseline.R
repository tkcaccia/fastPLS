#!/usr/bin/env Rscript

args <- commandArgs(trailingOnly = TRUE)
input <- if (length(args) >= 1L) args[[1L]] else
  "benchmark_results/manuscript_revision_cycle21_20260725/selected_memory_raw.csv"
output_dir <- if (length(args) >= 2L) args[[2L]] else dirname(input)
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

x <- utils::read.csv(input, stringsAsFactors = FALSE)
x <- x[x$status == "ok", , drop = FALSE]

required <- c(
  "peak_host_rss_mb", "fit_window_peak_host_rss_mb",
  "incremental_host_rss_mb", "rss_before_fit_mb",
  "peak_gpu_mem_mb", "gpu_before_fit_mb", "incremental_gpu_mem_mb"
)
missing <- setdiff(required, names(x))
if (length(missing)) {
  stop("Missing memory fields: ", paste(missing, collapse = ", "))
}

median_or_na <- function(value) {
  value <- suppressWarnings(as.numeric(value))
  if (!any(is.finite(value))) NA_real_ else stats::median(value, na.rm = TRUE)
}

q_or_na <- function(value, probability) {
  value <- suppressWarnings(as.numeric(value))
  if (!any(is.finite(value))) NA_real_ else
    unname(stats::quantile(value, probability, na.rm = TRUE))
}

keys <- c(
  "dataset", "task_type", "method_panel", "variant_name", "engine",
  "backend", "classifier", "effective_ncomp", "precision",
  "execution_precision", "metric_name"
)
groups <- interaction(x[keys], drop = TRUE, lex.order = TRUE)
summary <- do.call(rbind, lapply(split(x, groups), function(d) {
  data.frame(
    d[1L, keys, drop = FALSE],
    n_runs = nrow(d),
    metric_median = median_or_na(d$metric_value),
    total_time_sec_median = median_or_na(d$total_time_ms) / 1000,
    process_peak_host_rss_mb_median = median_or_na(d$peak_host_rss_mb),
    process_peak_host_rss_mb_q25 = q_or_na(d$peak_host_rss_mb, 0.25),
    process_peak_host_rss_mb_q75 = q_or_na(d$peak_host_rss_mb, 0.75),
    baseline_host_rss_mb_median = median_or_na(d$rss_before_fit_mb),
    baseline_host_rss_mb_q25 = q_or_na(d$rss_before_fit_mb, 0.25),
    baseline_host_rss_mb_q75 = q_or_na(d$rss_before_fit_mb, 0.75),
    fit_window_peak_host_rss_mb_median =
      median_or_na(d$fit_window_peak_host_rss_mb),
    incremental_host_rss_mb_median =
      median_or_na(d$incremental_host_rss_mb),
    incremental_host_rss_mb_q25 =
      q_or_na(d$incremental_host_rss_mb, 0.25),
    incremental_host_rss_mb_q75 =
      q_or_na(d$incremental_host_rss_mb, 0.75),
    baseline_gpu_mem_mb_median = median_or_na(d$gpu_before_fit_mb),
    peak_gpu_mem_mb_median = median_or_na(d$peak_gpu_mem_mb),
    incremental_gpu_mem_mb_median = median_or_na(d$incremental_gpu_mem_mb),
    status = "ok",
    stringsAsFactors = FALSE
  )
}))

summary$engine <- ifelse(
  toupper(summary$engine) %in% c("GPU", "CUDA"), "CUDA", "CPU"
)
summary <- summary[
  order(summary$dataset, summary$method_panel, summary$engine),
  ,
  drop = FALSE
]
utils::write.csv(
  summary,
  file.path(output_dir, "selected_memory_baseline_summary.csv"),
  row.names = FALSE
)

wide <- reshape(
  summary,
  idvar = c("dataset", "method_panel", "effective_ncomp"),
  timevar = "engine",
  direction = "wide"
)
utils::write.csv(
  wide,
  file.path(output_dir, "selected_memory_baseline_wide.csv"),
  row.names = FALSE
)

cat(file.path(output_dir, "selected_memory_baseline_summary.csv"), "\n")
