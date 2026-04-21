#!/usr/bin/env Rscript

suppressPackageStartupMessages({
  library(data.table)
})

script_arg <- grep("^--file=", commandArgs(trailingOnly = FALSE), value = TRUE)
script_file <- if (length(script_arg)) sub("^--file=", "", script_arg[[1]]) else file.path(getwd(), "benchmark", "write_synthetic_smoke_summary.R")
script_dir <- dirname(normalizePath(script_file, winslash = "/", mustWork = FALSE))
repo_root <- normalizePath(file.path(script_dir, ".."), winslash = "/", mustWork = FALSE)

source(file.path(repo_root, "R", "synthetic_smoke_generators.R"), local = TRUE)

out_dir <- path.expand(Sys.getenv("FASTPLS_SYNTH_SMOKE_OUTDIR", file.path(repo_root, "benchmark_results_synthetic_smoke_chiamaka")))
raw_path <- file.path(out_dir, "benchmark_results_synthetic_smoke_raw.csv")
summary_path <- file.path(out_dir, "benchmark_results_synthetic_smoke_summary.csv")
fair50_path <- file.path(out_dir, "benchmark_results_synthetic_smoke_fair50.csv")
capacity_path <- file.path(out_dir, "benchmark_results_synthetic_smoke_capacity_limited.csv")

families_override <- trimws(strsplit(Sys.getenv("FASTPLS_SYNTH_SMOKE_FAMILIES", ""), ",", fixed = TRUE)[[1]])
families_override <- families_override[nzchar(families_override)]
noise_override <- trimws(strsplit(Sys.getenv("FASTPLS_SYNTH_SMOKE_NOISES", ""), ",", fixed = TRUE)[[1]])
noise_override <- noise_override[nzchar(noise_override)]

if (!file.exists(raw_path)) {
  stop("Raw synthetic smoke CSV not found at: ", raw_path)
}

raw_dt <- fread(raw_path)
expected_families <- if (length(families_override)) families_override else names(smoke_family_specs())
expected_noise_levels <- if (length(noise_override)) noise_override else smoke_noise_levels()
synthetic_smoke_validate_raw(
  raw_dt,
  expected_families = expected_families,
  expected_noise_levels = expected_noise_levels
)

ok_dt <- raw_dt[status == "ok"]
if (!nrow(ok_dt)) {
  stop("No successful runs found in raw synthetic smoke output.")
}

summary_dt <- ok_dt[, .(
  reps_ok = .N,
  train_ms_median = median(train_ms, na.rm = TRUE),
  train_ms_iqr = IQR(train_ms, na.rm = TRUE),
  predict_ms_median = median(predict_ms, na.rm = TRUE),
  predict_ms_iqr = IQR(predict_ms, na.rm = TRUE),
  total_ms_median = median(total_ms, na.rm = TRUE),
  total_ms_iqr = IQR(total_ms, na.rm = TRUE),
  metric_median = median(metric_value, na.rm = TRUE),
  metric_iqr = IQR(metric_value, na.rm = TRUE),
  accuracy_median = median(accuracy, na.rm = TRUE),
  Q2_median = median(Q2, na.rm = TRUE),
  train_R2_median = median(train_R2, na.rm = TRUE),
  model_size_mb_median = median(model_size_mb, na.rm = TRUE),
  effective_ncomp_median = median(effective_ncomp, na.rm = TRUE)
), by = .(
  dataset, scenario_family, task_type, analysis, analysis_value,
  engine, algorithm, svd_method, fast_profile, method_id,
  n_train, n_test, p, q, K,
  requested_ncomp, capacity_limited,
  xtrain_nrow, xtrain_ncol, ytrain_ncol,
  noise_regime, noise_target_snr, noise_rank, metric_name
)]

summary_dt[, plot_subset := ifelse(capacity_limited, "full_only", "fair50")]

fair50_dt <- summary_dt[capacity_limited == FALSE]
capacity_dt <- raw_dt[capacity_limited == TRUE]

axis_checks <- list()
if (length(intersect(expected_families, c("sim_reg_n_p50", "sim_reg_n_p500", "sim_reg_n_p1000_q1000_ncomp500")))) {
  axis_checks$reg_n <- summary_dt[scenario_family %in% intersect(expected_families, c("sim_reg_n_p50", "sim_reg_n_p500", "sim_reg_n_p1000_q1000_ncomp500")), uniqueN(n_train)]
}
if ("sim_reg_p_sweep" %in% expected_families) {
  axis_checks$reg_p_full <- summary_dt[scenario_family == "sim_reg_p_sweep", uniqueN(p)]
  axis_checks$reg_p_fair <- fair50_dt[scenario_family == "sim_reg_p_sweep", uniqueN(p)]
}
if ("sim_reg_q_sweep" %in% expected_families) {
  axis_checks$reg_q_full <- summary_dt[scenario_family == "sim_reg_q_sweep", uniqueN(q)]
  axis_checks$reg_q_fair <- fair50_dt[scenario_family == "sim_reg_q_sweep", uniqueN(q)]
}
if ("sim_reg_noise_sweep" %in% expected_families) {
  axis_checks$reg_noise_full <- summary_dt[scenario_family == "sim_reg_noise_sweep", uniqueN(noise_rank)]
}

full_axes <- unlist(axis_checks[grepl("_full$|^reg_n$", names(axis_checks))], use.names = TRUE)
if (length(full_axes) && any(full_axes <= 1L)) {
  stop("Output summary has an invalid scaling axis with <= 1 unique x value.")
}

fair_axes <- unlist(axis_checks[grepl("_fair$", names(axis_checks))], use.names = TRUE)
if (length(fair_axes) && any(fair_axes <= 1L)) {
  stop("Fair50 summary cannot distinguish full vs fair50 plots.")
}

fwrite(summary_dt, summary_path)
fwrite(fair50_dt, fair50_path)
fwrite(capacity_dt, capacity_path)

message("Summary written to: ", summary_path)
message("Fair50 summary written to: ", fair50_path)
message("Capacity-limited rows written to: ", capacity_path)
