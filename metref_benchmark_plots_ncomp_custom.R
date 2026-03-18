#!/usr/bin/env Rscript

suppressPackageStartupMessages({
  library(data.table)
  library(ggplot2)
})

in_csv <- Sys.getenv("FASTPLS_METREF_IN", "metref_benchmark_results_ncomp_custom.csv")
out_dir <- Sys.getenv("FASTPLS_METREF_PLOT_DIR", "metref_benchmark_plots_ncomp_custom")

dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

res <- fread(in_csv)

ok <- res[
  !grepl("^error", status) &
    status != "skipped_cuda_unavailable" &
    status != "skipped_ncomp_above_plssvd_cap"
]

plot_metric <- function(df, ycol, ylab, filename) {
  p <- ggplot(df, aes(x = ncomp, y = get(ycol), color = config)) +
    geom_line(linewidth = 0.65) +
    theme_minimal(base_size = 11) +
    labs(x = "ncomp", y = ylab, color = "method")

  ggsave(file.path(out_dir, filename), p, width = 13, height = 7, dpi = 150)
}

plot_metric(ok, "acc_median_reps", "Accuracy (median over reps)", "accuracy_vs_ncomp.png")
plot_metric(ok, "time_ms_median_reps", "Time (ms, median over reps)", "time_vs_ncomp.png")
plot_metric(ok, "ram_alloc_mb", "RAM alloc (MB)", "ram_alloc_vs_ncomp.png")
plot_metric(ok, "gpu_mem_delta_mb_median", "GPU RAM delta (MB, median)", "gpu_ram_delta_vs_ncomp.png")
plot_metric(ok, "bench_median_ms", "bench::mark median (ms)", "bench_median_vs_ncomp.png")

# Engine/algorithm subsets for easier reading
sub_dir <- file.path(out_dir, "subsets")
dir.create(sub_dir, recursive = TRUE, showWarnings = FALSE)

save_subset <- function(expr, title_prefix) {
  d <- ok[eval(expr)]
  if (!nrow(d)) return(invisible(NULL))

  mk <- function(ycol, ylab, suffix) {
    p <- ggplot(d, aes(x = ncomp, y = get(ycol), color = config)) +
      geom_line(linewidth = 0.7) +
      theme_minimal(base_size = 11) +
      labs(title = title_prefix, x = "ncomp", y = ylab, color = "method")
    ggsave(file.path(sub_dir, paste0(title_prefix, "_", suffix, ".png")), p, width = 12, height = 6, dpi = 150)
  }

  mk("acc_median_reps", "Accuracy (median over reps)", "accuracy")
  mk("time_ms_median_reps", "Time (ms, median over reps)", "time")
  mk("ram_alloc_mb", "RAM alloc (MB)", "ram")
  mk("gpu_mem_delta_mb_median", "GPU RAM delta (MB)", "gpu")
}

save_subset(quote(engine == "Rcpp"), "rcpp")
save_subset(quote(engine == "R"), "r")
save_subset(quote(algorithm == "simpls"), "simpls")
save_subset(quote(algorithm == "plssvd"), "plssvd")
save_subset(quote(svd_method == "cuda_rsvd"), "cuda")

# Summary tables
summary_by_config <- ok[, .(
  best_acc = as.numeric(max(acc_median_reps, na.rm = TRUE)),
  ncomp_at_best_acc = as.numeric(ncomp[which.max(acc_median_reps)]),
  median_time_ms = as.numeric(median(time_ms_median_reps, na.rm = TRUE)),
  median_ram_mb = as.numeric(median(ram_alloc_mb, na.rm = TRUE)),
  median_gpu_delta_mb = as.numeric(median(gpu_mem_delta_mb_median, na.rm = TRUE))
), by = .(config)][order(-best_acc, median_time_ms)]

fwrite(summary_by_config, file.path(out_dir, "summary_by_config.csv"))

best_by_ncomp <- ok[, .SD[order(-acc_median_reps, time_ms_median_reps)][1], by = .(ncomp)]
fwrite(best_by_ncomp, file.path(out_dir, "best_method_by_ncomp.csv"))

cat("Done\n")
cat("Input:", normalizePath(in_csv), "\n")
cat("Plot dir:", normalizePath(out_dir), "\n")
cat("Rows (ok):", nrow(ok), "\n")
