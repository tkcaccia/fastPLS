#!/usr/bin/env Rscript

suppressPackageStartupMessages({
  library(data.table)
})

script_arg <- grep("^--file=", commandArgs(trailingOnly = FALSE), value = TRUE)
script_file <- if (length(script_arg)) sub("^--file=", "", script_arg[[1]]) else file.path(getwd(), "benchmark", "write_simulated_fastpls_summary.R")
script_dir <- dirname(normalizePath(script_file, winslash = "/", mustWork = FALSE))
repo_root <- normalizePath(file.path(script_dir, ".."), winslash = "/", mustWork = FALSE)

source(file.path(script_dir, "helpers_simulated_fastpls.R"), local = TRUE)

out_dir <- path.expand(Sys.getenv("FASTPLS_SIM_OUTDIR", file.path(repo_root, "benchmark_results_simulated_fastpls")))
raw_path <- file.path(out_dir, "simulated_fastpls_raw.csv")
if (!file.exists(raw_path)) stop("Missing raw synthetic benchmark CSV: ", raw_path)

raw <- fread(raw_path)
ok <- raw[status == "ok"]
if (!nrow(ok)) stop("No successful rows in ", raw_path)

ok[, predictive_metric := fifelse(task_type == "classification", accuracy, Q2)]
ok[, backend := fifelse(engine == "GPU", "gpu_native", fifelse(engine == "pls_pkg", "pls_pkg", svd_method))]

summary_dt <- ok[, .(
  reps_ok = .N,
  elapsed_ms_median = median(elapsed_ms, na.rm = TRUE),
  elapsed_ms_mean = mean(elapsed_ms, na.rm = TRUE),
  predictive_median = median(predictive_metric, na.rm = TRUE),
  predictive_mean = mean(predictive_metric, na.rm = TRUE),
  accuracy_median = median(accuracy, na.rm = TRUE),
  Q2_median = median(Q2, na.rm = TRUE),
  train_R2_median = median(train_R2, na.rm = TRUE),
  effective_ncomp_median = median(effective_ncomp, na.rm = TRUE),
  signal_var_X = unique(signal_var_X)[1],
  noise_var_X = unique(noise_var_X)[1],
  signal_var_Y = unique(signal_var_Y)[1],
  noise_var_Y = unique(noise_var_Y)[1],
  realized_snr_X = unique(realized_snr_X)[1],
  realized_snr_Y = unique(realized_snr_Y)[1],
  observed_zero_rate_X = unique(observed_zero_rate_X)[1],
  class_margin = unique(class_margin)[1]
), by = .(
  dataset, sim_family, task_type, analysis_type, analysis_value,
  spectrum_regime, noise_regime, dropout_regime,
  n, p, q, n_classes, r_true,
  requested_ncomp, sample_fraction, xvar_fraction, yvar_fraction,
  engine, method, fast_profile, method_id, method_label, svd_method, backend
)]

summary_path <- file.path(out_dir, "simulated_fastpls_summary.csv")
fwrite(summary_dt, summary_path)

spectrum_rows <- summary_dt[analysis_type == "spectrum_and_noise" & is.finite(predictive_median)]
hardest_regime <- if (nrow(spectrum_rows)) {
  spectrum_rows[order(method_id, predictive_median, -elapsed_ms_median)][, .SD[1], by = method_id]
} else {
  data.table()
}

ncomp_rows <- summary_dt[analysis_type == "ncomp" & backend != "pls_pkg"]
simpls_fast_gain <- merge(
  ncomp_rows[method == "simpls_fast", .(
    sim_family, task_type, requested_ncomp, engine, backend,
    elapsed_fast = elapsed_ms_median, pred_fast = predictive_median
  )],
  ncomp_rows[method == "simpls", .(
    sim_family, task_type, requested_ncomp, engine, backend,
    elapsed_simpls = elapsed_ms_median, pred_simpls = predictive_median
  )],
  by = c("sim_family", "task_type", "requested_ncomp", "engine", "backend"),
  all = FALSE
)
if (nrow(simpls_fast_gain)) {
  simpls_fast_gain[, speedup_vs_simpls := elapsed_simpls / elapsed_fast]
  simpls_fast_gain[, metric_delta := pred_fast - pred_simpls]
}

plssvd_cap <- summary_dt[method == "plssvd" & effective_ncomp_median < requested_ncomp]

spectral_effect <- if (nrow(spectrum_rows)) {
  spectrum_rows[backend %in% c("cpu_rsvd", "irlba", "gpu_native"),
                .(elapsed_ms_median = mean(elapsed_ms_median), predictive_median = mean(predictive_median, na.rm = TRUE)),
                by = .(method, backend, spectrum_regime)]
} else {
  data.table()
}

q_vs_p <- summary_dt[analysis_type == "ncomp" & requested_ncomp == max(requested_ncomp, na.rm = TRUE) &
                       sim_family %in% c("sim_reg_large_q", "sim_reg_pggn") &
                       method %in% c("simpls", "simpls_fast", "plssvd")]

md_path <- file.path(out_dir, "simulated_fastpls_summary.md")
lines <- c(
  "# Simulated fastPLS Summary",
  "",
  "## What Was Simulated",
  "",
  "- Latent-factor synthetic datasets with one-factor-at-a-time variations in requested components, sample count, X width, Y width, spectrum difficulty, noise level, and dropout.",
  "- Regression families emphasize multivariate-response scaling and extreme `p >> n` settings.",
  "- Classification families emphasize multiclass difficulty, class imbalance, and sparse single-cell-like structure.",
  "",
  "## Why These Regimes",
  "",
  "- `fast_decay`, `sharp_decay`, `slow_decay`, and `clustered_top` separate easy low-rank structure from hard poorly separated spectra.",
  "- `low_noise`, `medium_noise`, and `high_noise` map the methods across realistic signal-to-noise conditions.",
  "- Sparse dropout settings mimic the instability of high-dimensional single-cell features without mixing in the real-data confounders.",
  "",
  "## How This Maps To The Real-Data Story",
  "",
  "- `sim_reg_large_q` mirrors large multivariate-response problems such as NMR and PRISM.",
  "- `sim_reg_pggn` mirrors very wide omics settings like methylation or expression with `p >> n`.",
  "- `sim_sparse_singlecell_like` gives a controlled counterpart to single-cell benchmarks where sparsity and class count interact.",
  ""
)

if (nrow(hardest_regime)) {
  lines <- c(lines, "## Hardest Spectrum/Noise Regimes", "")
  for (i in seq_len(nrow(hardest_regime))) {
    rr <- hardest_regime[i]
    lines <- c(lines, sprintf(
      "- `%s`: hardest setting was `%s / %s` on `%s` (median metric %.4f, median runtime %.1f ms).",
      rr$method_id, rr$spectrum_regime, rr$noise_regime, rr$sim_family, rr$predictive_median, rr$elapsed_ms_median
    ))
  }
  lines <- c(lines, "")
}

if (nrow(simpls_fast_gain)) {
  best_gain <- simpls_fast_gain[order(-speedup_vs_simpls, -metric_delta)][1]
  lines <- c(lines, "## Simpls-fast vs Simpls", "", sprintf(
    "- The largest measured speedup of `simpls_fast` over `simpls` was `%.2fx` on `%s` with backend `%s` at requested `ncomp=%d` (metric delta %.4f).",
    best_gain$speedup_vs_simpls, best_gain$sim_family, best_gain$backend, best_gain$requested_ncomp, best_gain$metric_delta
  ), "")
}

if (nrow(plssvd_cap)) {
  cap_counts <- plssvd_cap[, .N, by = sim_family][order(-N)]
  lines <- c(lines, "## PLSSVD Effective-Rank Limits", "", sprintf(
    "- PLSSVD hit an effective-rank cap in `%d` benchmark rows. Families with the most capped rows: %s.",
    nrow(plssvd_cap),
    paste(sprintf("%s (%d)", cap_counts$sim_family, cap_counts$N), collapse = ", ")
  ), "")
}

if (nrow(spectral_effect)) {
  agg <- spectral_effect[spectrum_regime %in% c("slow_decay", "clustered_top"), .(
    elapsed_hard = mean(elapsed_ms_median),
    metric_hard = mean(predictive_median, na.rm = TRUE)
  ), by = .(method, backend)]
  easy <- spectral_effect[spectrum_regime %in% c("fast_decay", "sharp_decay"), .(
    elapsed_easy = mean(elapsed_ms_median),
    metric_easy = mean(predictive_median, na.rm = TRUE)
  ), by = .(method, backend)]
  effect <- merge(agg, easy, by = c("method", "backend"))
  effect[, runtime_ratio := elapsed_hard / pmax(elapsed_easy, 1e-8)]
  effect[, metric_delta := metric_hard - metric_easy]
  lines <- c(lines, "## Spectrum Difficulty Effect", "", sprintf(
    "- Averaged over spectrum/noise runs, the hardest spectra (`slow_decay`, `clustered_top`) changed runtime by a median factor of `%.2fx` and changed predictive score by a median of `%.4f` across method/backend pairs.",
    median(effect$runtime_ratio, na.rm = TRUE),
    median(effect$metric_delta, na.rm = TRUE)
  ), "")
}

if (nrow(q_vs_p)) {
  q_vs_p_sum <- q_vs_p[, .(
    elapsed_ms_median = mean(elapsed_ms_median),
    predictive_median = mean(predictive_median, na.rm = TRUE)
  ), by = .(sim_family, method, backend)]
  lines <- c(lines, "## Large-q vs Wide-p Contrast", "", "- Comparing `sim_reg_large_q` with `sim_reg_pggn` helps separate multivariate-response scaling from extreme-X-width behavior in the same benchmark format.", "")
}

writeLines(lines, md_path)

cat("Summary CSV written to:", summary_path, "\n")
cat("Summary markdown written to:", md_path, "\n")
