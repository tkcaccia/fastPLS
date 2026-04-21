#!/usr/bin/env Rscript

suppressPackageStartupMessages({
  library(data.table)
  library(fastPLS)
})

script_arg <- grep("^--file=", commandArgs(trailingOnly = FALSE), value = TRUE)
script_file <- if (length(script_arg)) sub("^--file=", "", script_arg[[1]]) else file.path(getwd(), "benchmark", "benchmark_synthetic_smoke_chiamaka.R")
script_dir <- dirname(normalizePath(script_file, winslash = "/", mustWork = FALSE))
repo_root <- normalizePath(file.path(script_dir, ".."), winslash = "/", mustWork = FALSE)

source(file.path(repo_root, "R", "synthetic_smoke_generators.R"), local = TRUE)

out_dir <- path.expand(Sys.getenv("FASTPLS_SYNTH_SMOKE_OUTDIR", file.path(repo_root, "benchmark_results_synthetic_smoke_chiamaka")))
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

threads <- suppressWarnings(as.integer(Sys.getenv("FASTPLS_THREADS", "1")))
if (!is.finite(threads) || is.na(threads) || threads < 1L) threads <- 1L
Sys.setenv(
  OMP_NUM_THREADS = as.character(threads),
  OPENBLAS_NUM_THREADS = as.character(threads),
  MKL_NUM_THREADS = as.character(threads),
  VECLIB_MAXIMUM_THREADS = as.character(threads),
  NUMEXPR_NUM_THREADS = as.character(threads)
)
if (requireNamespace("RhpcBLASctl", quietly = TRUE)) {
  RhpcBLASctl::blas_set_num_threads(threads)
  RhpcBLASctl::omp_set_num_threads(threads)
}

reps <- suppressWarnings(as.integer(Sys.getenv("FASTPLS_SYNTH_SMOKE_REPS", "5")))
if (!is.finite(reps) || is.na(reps) || reps < 1L) reps <- 5L
requested_ncomp <- suppressWarnings(as.integer(Sys.getenv("FASTPLS_SYNTH_SMOKE_REQUESTED_NCOMP", as.character(smoke_requested_ncomp_default()))))
if (!is.finite(requested_ncomp) || is.na(requested_ncomp) || requested_ncomp < 1L) requested_ncomp <- smoke_requested_ncomp_default()
base_seed <- suppressWarnings(as.integer(Sys.getenv("FASTPLS_SYNTH_SMOKE_SEED", "123")))
if (!is.finite(base_seed) || is.na(base_seed)) base_seed <- 123L
include_gpu <- tolower(Sys.getenv("FASTPLS_SYNTH_SMOKE_INCLUDE_GPU", "false")) %in% c("1", "true", "yes", "y")
include_r <- tolower(Sys.getenv("FASTPLS_SYNTH_SMOKE_INCLUDE_R", "false")) %in% c("1", "true", "yes", "y")
include_pls_pkg <- tolower(Sys.getenv("FASTPLS_SYNTH_SMOKE_INCLUDE_PLS_PKG", "false")) %in% c("1", "true", "yes", "y")

families_override <- trimws(strsplit(Sys.getenv("FASTPLS_SYNTH_SMOKE_FAMILIES", ""), ",", fixed = TRUE)[[1]])
families_override <- families_override[nzchar(families_override)]
noise_override <- trimws(strsplit(Sys.getenv("FASTPLS_SYNTH_SMOKE_NOISES", ""), ",", fixed = TRUE)[[1]])
noise_override <- noise_override[nzchar(noise_override)]

families <- smoke_family_specs()
if (length(families_override)) {
  unknown <- setdiff(families_override, names(families))
  if (length(unknown)) stop("Unknown synthetic smoke families: ", paste(unknown, collapse = ", "))
  families <- families[families_override]
}
noise_levels <- if (length(noise_override)) noise_override else smoke_noise_levels()
unknown_noise <- setdiff(noise_levels, smoke_noise_levels())
if (length(unknown_noise)) stop("Unknown noise regimes: ", paste(unknown_noise, collapse = ", "))

cuda_ok <- tryCatch(isTRUE(fastPLS::has_cuda()), error = function(e) FALSE)

raw_path <- file.path(out_dir, "benchmark_results_synthetic_smoke_raw.csv")
manifest_path <- file.path(out_dir, "benchmark_results_synthetic_smoke_manifest.txt")

message("Synthetic smoke benchmark started")
message("Output dir: ", out_dir)
message("Families: ", paste(names(families), collapse = ", "))
message("Noise regimes available: ", paste(noise_levels, collapse = ", "))
message("Requested ncomp: ", requested_ncomp)
message("Timing reps: ", reps)
message("CUDA available: ", cuda_ok)
message("Method scope: Rcpp always on; include_gpu=", include_gpu, "; include_r=", include_r, "; include_pls_pkg=", include_pls_pkg)

rows <- vector("list", length = 0L)
row_idx <- 0L

append_row <- function(dt_row) {
  row_idx <<- row_idx + 1L
  rows[[row_idx]] <<- dt_row
}

build_dataset <- function(family_name, family_cfg, analysis_value, noise_regime, seed_data) {
  requested_ncomp_local <- family_cfg$requested_ncomp %||% requested_ncomp
  if (identical(family_cfg$task_type, "regression")) {
    if (identical(family_cfg$analysis, "n_train")) {
      synthetic_smoke_generate_regression(
        n_train = as.integer(analysis_value),
        n_test = family_cfg$n_test,
        p = family_cfg$p,
        q = family_cfg$q,
        noise_regime = noise_regime,
        requested_ncomp = requested_ncomp_local,
        seed_data = seed_data
      )
    } else if (identical(family_cfg$analysis, "p")) {
      synthetic_smoke_generate_regression(
        n_train = family_cfg$n_train,
        n_test = family_cfg$n_test,
        p = as.integer(analysis_value),
        q = family_cfg$q,
        noise_regime = noise_regime,
        requested_ncomp = requested_ncomp_local,
        seed_data = seed_data
      )
    } else if (identical(family_cfg$analysis, "q")) {
      synthetic_smoke_generate_regression(
        n_train = family_cfg$n_train,
        n_test = family_cfg$n_test,
        p = family_cfg$p,
        q = as.integer(analysis_value),
        noise_regime = noise_regime,
        requested_ncomp = requested_ncomp_local,
        seed_data = seed_data
      )
    } else if (identical(family_cfg$analysis, "noise_regime")) {
      synthetic_smoke_generate_regression(
        n_train = family_cfg$n_train,
        n_test = family_cfg$n_test,
        p = family_cfg$p,
        q = family_cfg$q,
        noise_regime = as.character(analysis_value),
        requested_ncomp = requested_ncomp_local,
        seed_data = seed_data
      )
    } else {
      stop("Unsupported regression analysis for ", family_name)
    }
  } else {
    if (identical(family_cfg$analysis, "noise_regime")) {
      synthetic_smoke_generate_classification(
        n_train = family_cfg$n_train,
        n_test = family_cfg$n_test,
        p = family_cfg$p,
        K = family_cfg$K,
        noise_regime = as.character(analysis_value),
        requested_ncomp = requested_ncomp,
        seed_data = seed_data
      )
    } else {
      synthetic_smoke_generate_classification(
        n_train = family_cfg$n_train,
        n_test = family_cfg$n_test,
        p = family_cfg$p,
        K = as.integer(analysis_value),
        noise_regime = noise_regime,
        requested_ncomp = requested_ncomp,
        seed_data = seed_data
      )
    }
  }
}

for (family_idx in seq_along(families)) {
  family_name <- names(families)[[family_idx]]
  family_cfg <- families[[family_idx]]

  if (identical(family_cfg$analysis, "noise_regime")) {
    family_noise_levels <- if (length(noise_override)) noise_levels else family_cfg$values
    analysis_values <- family_noise_levels
    active_noise_levels <- NA_character_
  } else {
    family_noise_levels <- if (length(noise_override)) noise_levels else family_cfg$fixed_noise %||% smoke_default_noise_level()
    analysis_values <- family_cfg$values
    active_noise_levels <- family_noise_levels
  }

  for (noise_idx in seq_along(if (length(active_noise_levels)) active_noise_levels else 1L)) {
    noise_regime <- if (length(active_noise_levels)) active_noise_levels[[noise_idx]] else NA_character_

    for (value_idx in seq_along(analysis_values)) {
      analysis_value <- analysis_values[[value_idx]]
      noise_regime_value <- if (identical(family_cfg$analysis, "noise_regime")) as.character(analysis_value) else noise_regime

      for (rep_idx in seq_len(reps)) {
        seed_data <- as.integer(base_seed + family_idx * 100000L + noise_idx * 10000L + value_idx * 100L + rep_idx)
        ds <- build_dataset(family_name, family_cfg, analysis_value, noise_regime_value, seed_data)
        ds$scenario_family <- family_name
        synthetic_smoke_validate_dataset(ds)

        meta <- ds$meta
        methods <- synthetic_smoke_methods(
          cuda_ok = cuda_ok,
          n_train = meta$n_train,
          p = meta$p,
          q = meta$q,
          K = meta$K,
          include_gpu = include_gpu,
          include_r = include_r,
          include_pls_pkg = include_pls_pkg
        )

        message(
          sprintf(
            "[RUN] family=%s noise=%s %s=%s rep=%d methods=%d effective_ncomp=%d",
            family_name, noise_regime_value, family_cfg$analysis, as.character(analysis_value), rep_idx, nrow(methods), meta$effective_ncomp
          )
        )

        for (method_idx in seq_len(nrow(methods))) {
          cfg <- methods[method_idx]
          seed_fit <- as.integer(seed_data + 1000L + method_idx)

          fit_res <- tryCatch(
            synthetic_smoke_run_method(ds, cfg, effective_ncomp = meta$effective_ncomp, seed_fit = seed_fit),
            error = function(e) e
          )

          common_row <- data.table(
            dataset = family_name,
            scenario_family = family_name,
            task_type = ds$task_type,
            analysis = family_cfg$analysis,
            analysis_value = as.character(analysis_value),
            rep = rep_idx,
            engine = cfg$engine,
            algorithm = cfg$algorithm,
            svd_method = cfg$svd_method,
            fast_profile = cfg$fast_profile,
            method_id = cfg$method_id,
            n_train = meta$n_train,
            n_test = meta$n_test,
            p = meta$p,
            q = meta$q,
            K = meta$K,
            requested_ncomp = meta$requested_ncomp,
            effective_ncomp = meta$effective_ncomp,
            capacity_limited = meta$capacity_limited,
            ncomp = meta$effective_ncomp,
            xtrain_nrow = nrow(ds$Xtrain),
            xtrain_ncol = ncol(ds$Xtrain),
            ytrain_ncol = if (identical(ds$task_type, "classification")) nlevels(ds$Ytrain) else ncol(ds$Ytrain),
            noise_regime = meta$noise_regime,
            noise_target_snr = meta$noise_target_snr,
            noise_rank = meta$noise_rank,
            realized_snr_X = meta$realized_snr_X,
            realized_snr_Y = meta$realized_snr_Y,
            signal_var_X = meta$signal_var_X,
            noise_var_X = meta$noise_var_X,
            signal_var_Y = meta$signal_var_Y,
            noise_var_Y = meta$noise_var_Y,
            class_separation = meta$class_separation,
            seed_data = meta$seed_data,
            seed_fit = seed_fit
          )

          if (inherits(fit_res, "error")) {
            append_row(cbind(
              common_row,
              data.table(
                train_ms = NA_real_,
                predict_ms = NA_real_,
                total_ms = NA_real_,
                metric_name = synthetic_smoke_metric_name(ds$task_type),
                metric_value = NA_real_,
                accuracy = NA_real_,
                Q2 = NA_real_,
                train_R2 = NA_real_,
                model_size_mb = NA_real_,
                status = "error",
                msg = conditionMessage(fit_res)
              )
            ))
          } else {
            append_row(cbind(
              common_row,
              data.table(
                train_ms = fit_res$train_ms,
                predict_ms = fit_res$predict_ms,
                total_ms = fit_res$total_ms,
                metric_name = synthetic_smoke_metric_name(ds$task_type),
                metric_value = synthetic_smoke_metric_value(ds$task_type, fit_res),
                accuracy = fit_res$accuracy,
                Q2 = fit_res$Q2,
                train_R2 = fit_res$train_R2,
                model_size_mb = fit_res$model_size_mb,
                status = "ok",
                msg = ""
              )
            ))
          }
        }
      }
    }
  }
}

raw_dt <- rbindlist(rows, fill = TRUE)
synthetic_smoke_validate_raw(raw_dt, expected_families = names(families), expected_noise_levels = noise_levels)

fwrite(raw_dt, raw_path)
writeLines(synthetic_smoke_manifest_lines(raw_dt, cuda_ok = cuda_ok, timing_reps = reps, out_dir = out_dir), manifest_path)

if (any(raw_dt$capacity_limited, na.rm = TRUE)) {
  warning(
    "low p, low q, or low K values are CAPACITY-LIMITED regimes, not strict apples-to-apples 50-component comparisons",
    call. = FALSE
  )
}

message("Raw synthetic smoke results written to: ", raw_path)
message("Manifest written to: ", manifest_path)
