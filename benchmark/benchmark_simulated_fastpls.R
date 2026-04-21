#!/usr/bin/env Rscript

suppressPackageStartupMessages({
  library(data.table)
  library(fastPLS)
})

script_arg <- grep("^--file=", commandArgs(trailingOnly = FALSE), value = TRUE)
script_file <- if (length(script_arg)) sub("^--file=", "", script_arg[[1]]) else file.path(getwd(), "benchmark", "benchmark_simulated_fastpls.R")
script_dir <- dirname(normalizePath(script_file, winslash = "/", mustWork = FALSE))
repo_root <- normalizePath(file.path(script_dir, ".."), winslash = "/", mustWork = FALSE)

source(file.path(script_dir, "helpers_simulated_fastpls.R"), local = TRUE)

out_dir <- path.expand(Sys.getenv("FASTPLS_SIM_OUTDIR", file.path(repo_root, "benchmark_results_simulated_fastpls")))
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

panel <- tolower(Sys.getenv("FASTPLS_SIM_PANEL", "main"))
if (!panel %in% c("main", "supplement", "all")) {
  stop("FASTPLS_SIM_PANEL must be one of: main, supplement, all")
}

family_override <- simfast_parse_csv(Sys.getenv("FASTPLS_SIM_FAMILIES", ""))
analysis_override <- simfast_parse_csv(Sys.getenv("FASTPLS_SIM_ANALYSES", ""))
mode <- tolower(Sys.getenv("FASTPLS_SIM_MODE", "paper"))
if (!mode %in% c("paper", "stability")) stop("FASTPLS_SIM_MODE must be paper or stability")

explicit_reps <- suppressWarnings(as.integer(Sys.getenv("FASTPLS_REPS", NA_character_)))
if (!is.finite(explicit_reps) || is.na(explicit_reps) || explicit_reps < 1L) explicit_reps <- NA_integer_
reps <- simfast_mode_reps(mode = mode, explicit_reps = explicit_reps)
include_r_impl <- simfast_bool_env("FASTPLS_INCLUDE_R_IMPL", FALSE)
include_pls_pkg <- simfast_bool_env("FASTPLS_INCLUDE_PLS_PKG", TRUE)
include_cuda <- simfast_bool_env("FASTPLS_INCLUDE_CUDA", TRUE)
include_pls_pkg_pggn <- simfast_bool_env("FASTPLS_SIM_INCLUDE_PLS_PKG_PGGN", FALSE)
train_fraction <- simfast_num_env("FASTPLS_SIM_TRAIN_FRACTION", 0.8)
base_seed <- simfast_int_env("FASTPLS_SEED", 123)
default_requested_ncomp <- simfast_int_env("FASTPLS_SIM_DEFAULT_NCOMP", 20L)
threads <- simfast_int_env("FASTPLS_THREADS", 1L)

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

grids <- simfast_default_grids()

ncomp_grid <- suppressWarnings(as.integer(simfast_parse_csv(Sys.getenv("FASTPLS_NCOMP_LIST", paste(grids$ncomp, collapse = ",")))))
ncomp_grid <- sort(unique(ncomp_grid[is.finite(ncomp_grid) & !is.na(ncomp_grid) & ncomp_grid >= 1L]))
if (!length(ncomp_grid)) ncomp_grid <- grids$ncomp

sample_fractions <- suppressWarnings(as.numeric(simfast_parse_csv(Sys.getenv("FASTPLS_SAMPLE_FRACS", paste(grids$sample_fraction, collapse = ",")))))
sample_fractions <- sort(unique(sample_fractions[is.finite(sample_fractions) & !is.na(sample_fractions) & sample_fractions > 0 & sample_fractions <= 1]))
if (!length(sample_fractions)) sample_fractions <- grids$sample_fraction

xvar_fractions <- suppressWarnings(as.numeric(simfast_parse_csv(Sys.getenv("FASTPLS_XVAR_FRACS", paste(grids$xvar_fraction, collapse = ",")))))
xvar_fractions <- sort(unique(xvar_fractions[is.finite(xvar_fractions) & !is.na(xvar_fractions) & xvar_fractions > 0 & xvar_fractions <= 1]))
if (!length(xvar_fractions)) xvar_fractions <- grids$xvar_fraction

yvar_fractions <- suppressWarnings(as.numeric(simfast_parse_csv(Sys.getenv("FASTPLS_YVAR_FRACS", paste(grids$yvar_fraction, collapse = ",")))))
yvar_fractions <- sort(unique(yvar_fractions[is.finite(yvar_fractions) & !is.na(yvar_fractions) & yvar_fractions > 0 & yvar_fractions <= 1]))
if (!length(yvar_fractions)) yvar_fractions <- grids$yvar_fraction

spectra <- simfast_parse_csv(Sys.getenv("FASTPLS_SIM_SPECTRA", paste(grids$spectrum_regime, collapse = ",")))
if (!length(spectra)) spectra <- grids$spectrum_regime
noise_regimes <- simfast_parse_csv(Sys.getenv("FASTPLS_SIM_NOISES", paste(grids$noise_regime, collapse = ",")))
if (!length(noise_regimes)) noise_regimes <- grids$noise_regime

families <- simfast_select_families(panel = panel, families = if (length(family_override)) family_override else NULL)
analyses <- simfast_select_analyses(if (length(analysis_override)) analysis_override else NULL)
catalog <- simfast_family_catalog()
methods_all <- simfast_method_grid(include_cuda = include_cuda, include_r_impl = include_r_impl)

manifest_all <- simfast_manifest_dt()
manifest_used <- manifest_all[sim_family %in% families]
fwrite(manifest_used, file.path(out_dir, "simulated_fastpls_manifest_used.csv"))
manifest_repo <- file.path(script_dir, "simulated_fastpls_manifest.csv")
if (file.exists(manifest_repo)) {
  file.copy(manifest_repo, file.path(out_dir, "simulated_fastpls_manifest.csv"), overwrite = TRUE)
}

message("Synthetic families: ", paste(families, collapse = ", "))
message("Analyses: ", paste(analyses, collapse = ", "))
message("Reps: ", reps, " | panel: ", panel, " | include_cuda: ", include_cuda, " | include_pls_pkg: ", include_pls_pkg)

fit_one_method <- function(ds, cfg, requested_ncomp, seed_fit) {
  effective_ncomp <- max(1L, min(as.integer(requested_ncomp), simfast_legal_ncomp(ds)))

  if (identical(cfg$engine, "pls_pkg")) {
    res <- simfast_pls_pkg_fit(ds, task_type = ds$task_type, effective_ncomp = effective_ncomp)
  } else {
    res <- simfast_fastpls_fit(ds, cfg, task_type = ds$task_type, effective_ncomp = effective_ncomp, seed_fit = seed_fit)
  }

  list(
    elapsed_ms = as.numeric(res$elapsed_ms),
    accuracy = as.numeric(res$accuracy),
    Q2 = as.numeric(res$Q2),
    train_R2 = as.numeric(res$train_R2),
    model_size_mb = as.numeric(res$model_size_mb),
    effective_ncomp = effective_ncomp
  )
}

record_row <- function(rows, idx, ds, cfg, analysis_type, analysis_value, replicate, requested_ncomp,
                       sample_fraction, xvar_fraction, yvar_fraction, seed_data, seed_split, seed_fit,
                       status, msg = "", fit_res = NULL) {
  meta <- ds$meta
  idx <- idx + 1L
  rows[[idx]] <- data.table(
    dataset = ds$sim_family,
    sim_family = ds$sim_family,
    task_type = ds$task_type,
    analysis_type = analysis_type,
    analysis_value = as.character(analysis_value),
    spectrum_regime = meta$spectrum_regime,
    noise_regime = meta$noise_regime,
    dropout_regime = meta$dropout_regime,
    n = nrow(ds$Xtrain) + nrow(ds$Xtest),
    p = ncol(ds$Xtrain),
    q = if (is.factor(ds$Ytrain)) nlevels(ds$Ytrain) else ncol(ds$Ytrain),
    n_classes = meta$n_classes,
    r_true = meta$r_true,
    requested_ncomp = as.integer(requested_ncomp),
    effective_ncomp = if (is.null(fit_res)) NA_integer_ else as.integer(fit_res$effective_ncomp),
    sample_fraction = sample_fraction,
    xvar_fraction = xvar_fraction,
    yvar_fraction = yvar_fraction,
    replicate = replicate,
    engine = cfg$engine,
    method = cfg$method,
    fast_profile = cfg$fast_profile,
    method_id = cfg$method_id,
    method_label = simfast_method_label(cfg$engine, cfg$method, cfg$svd_method),
    svd_method = cfg$svd_method,
    elapsed_ms = if (is.null(fit_res)) NA_real_ else fit_res$elapsed_ms,
    accuracy = if (is.null(fit_res)) NA_real_ else fit_res$accuracy,
    Q2 = if (is.null(fit_res)) NA_real_ else fit_res$Q2,
    train_R2 = if (is.null(fit_res)) NA_real_ else fit_res$train_R2,
    model_size_mb = if (is.null(fit_res)) NA_real_ else fit_res$model_size_mb,
    seed_data = seed_data,
    seed_split = seed_split,
    seed_fit = seed_fit,
    signal_var_X = meta$signal_var_X,
    noise_var_X = meta$noise_var_X,
    signal_var_Y = meta$signal_var_Y,
    noise_var_Y = meta$noise_var_Y,
    realized_snr_X = meta$realized_snr_X,
    realized_snr_Y = meta$realized_snr_Y,
    observed_zero_rate_X = meta$observed_zero_rate_X %||% NA_real_,
    class_margin = meta$class_margin,
    status = status,
    msg = msg
  )
  list(rows = rows, idx = idx)
}

run_setting <- function(rows, idx, ds, methods, analysis_type, analysis_value, requested_ncomp,
                        sample_fraction, xvar_fraction, yvar_fraction,
                        replicate, seed_data, seed_split, seed_fit_base) {
  simfast_validate_dataset(ds)
  for (m_idx in seq_len(nrow(methods))) {
    cfg <- methods[m_idx]
    seed_fit <- as.integer(seed_fit_base + m_idx)
    fit_res <- tryCatch(
      fit_one_method(ds, cfg, requested_ncomp = requested_ncomp, seed_fit = seed_fit),
      error = function(e) e
    )

    if (inherits(fit_res, "error")) {
      out <- record_row(
        rows, idx, ds, cfg,
        analysis_type = analysis_type,
        analysis_value = analysis_value,
        replicate = replicate,
        requested_ncomp = requested_ncomp,
        sample_fraction = sample_fraction,
        xvar_fraction = xvar_fraction,
        yvar_fraction = yvar_fraction,
        seed_data = seed_data,
        seed_split = seed_split,
        seed_fit = seed_fit,
        status = "error",
        msg = conditionMessage(fit_res),
        fit_res = NULL
      )
    } else {
      out <- record_row(
        rows, idx, ds, cfg,
        analysis_type = analysis_type,
        analysis_value = analysis_value,
        replicate = replicate,
        requested_ncomp = requested_ncomp,
        sample_fraction = sample_fraction,
        xvar_fraction = xvar_fraction,
        yvar_fraction = yvar_fraction,
        seed_data = seed_data,
        seed_split = seed_split,
        seed_fit = seed_fit,
        status = "ok",
        msg = "",
        fit_res = fit_res
      )
    }
    rows <- out$rows
    idx <- out$idx
  }
  list(rows = rows, idx = idx)
}

rows <- list()
idx <- 0L

for (f_idx in seq_along(families)) {
  family <- families[[f_idx]]
  family_cfg <- catalog[[family]]
  methods_family <- simfast_methods_for_family(
    sim_family = family,
    methods_all = methods_all,
    include_pls_pkg = isTRUE(include_pls_pkg) && isTRUE(family_cfg$small_pls_pkg),
    include_pls_pkg_pggn = isTRUE(include_pls_pkg_pggn)
  )

  for (replicate in seq_len(reps)) {
    seed_data_base <- as.integer(base_seed + f_idx * 100000L + replicate * 1000L)
    seed_split_base <- as.integer(seed_data_base + 1L)
    base_ds <- simfast_generate_dataset(family_cfg, seed_data = seed_data_base, seed_split = seed_split_base, train_fraction = train_fraction)

    if ("ncomp" %in% analyses) {
      for (v_idx in seq_along(ncomp_grid)) {
        req_ncomp <- as.integer(ncomp_grid[[v_idx]])
        seed_fit_base <- as.integer(seed_data_base + 10000L + v_idx * 100L)
        out <- run_setting(
          rows = rows,
          idx = idx,
          ds = base_ds,
          methods = methods_family,
          analysis_type = "ncomp",
          analysis_value = req_ncomp,
          requested_ncomp = req_ncomp,
          sample_fraction = 1.0,
          xvar_fraction = 1.0,
          yvar_fraction = if (identical(base_ds$task_type, "regression")) 1.0 else NA_real_,
          replicate = replicate,
          seed_data = seed_data_base,
          seed_split = seed_split_base,
          seed_fit_base = seed_fit_base
        )
        rows <- out$rows
        idx <- out$idx
      }
    }

    if ("sample_fraction" %in% analyses) {
      for (v_idx in seq_along(sample_fractions)) {
        frac <- sample_fractions[[v_idx]]
        seed_analysis <- as.integer(seed_data_base + 20000L + v_idx)
        prepared <- simfast_prepare_analysis_dataset(base_ds, "sample_fraction", frac, seed_analysis = seed_analysis)
        out <- run_setting(
          rows = rows,
          idx = idx,
          ds = prepared$dataset,
          methods = methods_family,
          analysis_type = "sample_fraction",
          analysis_value = frac,
          requested_ncomp = default_requested_ncomp,
          sample_fraction = prepared$sample_fraction,
          xvar_fraction = prepared$xvar_fraction,
          yvar_fraction = prepared$yvar_fraction,
          replicate = replicate,
          seed_data = seed_data_base,
          seed_split = seed_split_base,
          seed_fit_base = seed_analysis + 100L
        )
        rows <- out$rows
        idx <- out$idx
      }
    }

    if ("xvar_fraction" %in% analyses) {
      for (v_idx in seq_along(xvar_fractions)) {
        frac <- xvar_fractions[[v_idx]]
        seed_analysis <- as.integer(seed_data_base + 30000L + v_idx)
        prepared <- simfast_prepare_analysis_dataset(base_ds, "xvar_fraction", frac, seed_analysis = seed_analysis)
        out <- run_setting(
          rows = rows,
          idx = idx,
          ds = prepared$dataset,
          methods = methods_family,
          analysis_type = "xvar_fraction",
          analysis_value = frac,
          requested_ncomp = default_requested_ncomp,
          sample_fraction = prepared$sample_fraction,
          xvar_fraction = prepared$xvar_fraction,
          yvar_fraction = prepared$yvar_fraction,
          replicate = replicate,
          seed_data = seed_data_base,
          seed_split = seed_split_base,
          seed_fit_base = seed_analysis + 100L
        )
        rows <- out$rows
        idx <- out$idx
      }
    }

    if ("yvar_fraction" %in% analyses && identical(base_ds$task_type, "regression")) {
      for (v_idx in seq_along(yvar_fractions)) {
        frac <- yvar_fractions[[v_idx]]
        seed_analysis <- as.integer(seed_data_base + 40000L + v_idx)
        prepared <- simfast_prepare_analysis_dataset(base_ds, "yvar_fraction", frac, seed_analysis = seed_analysis)
        out <- run_setting(
          rows = rows,
          idx = idx,
          ds = prepared$dataset,
          methods = methods_family,
          analysis_type = "yvar_fraction",
          analysis_value = frac,
          requested_ncomp = default_requested_ncomp,
          sample_fraction = prepared$sample_fraction,
          xvar_fraction = prepared$xvar_fraction,
          yvar_fraction = prepared$yvar_fraction,
          replicate = replicate,
          seed_data = seed_data_base,
          seed_split = seed_split_base,
          seed_fit_base = seed_analysis + 100L
        )
        rows <- out$rows
        idx <- out$idx
      }
    }

    if ("spectrum_and_noise" %in% analyses) {
      combo_idx <- 0L
      for (spec in spectra) {
        for (noise in noise_regimes) {
          combo_idx <- combo_idx + 1L
          cfg_alt <- utils::modifyList(family_cfg, list(spectrum_regime = spec, noise_regime = noise))
          seed_data_alt <- as.integer(seed_data_base + 50000L + combo_idx * 10L)
          seed_split_alt <- as.integer(seed_data_alt + 1L)
          ds_alt <- simfast_generate_dataset(cfg_alt, seed_data = seed_data_alt, seed_split = seed_split_alt, train_fraction = train_fraction)
          out <- run_setting(
            rows = rows,
            idx = idx,
            ds = ds_alt,
            methods = methods_family,
            analysis_type = "spectrum_and_noise",
            analysis_value = paste(spec, noise, sep = "__"),
            requested_ncomp = default_requested_ncomp,
            sample_fraction = 1.0,
            xvar_fraction = 1.0,
            yvar_fraction = if (identical(ds_alt$task_type, "regression")) 1.0 else NA_real_,
            replicate = replicate,
            seed_data = seed_data_alt,
            seed_split = seed_split_alt,
            seed_fit_base = seed_data_alt + 100L
          )
          rows <- out$rows
          idx <- out$idx
        }
      }
    }
  }
}

if (!length(rows)) stop("No synthetic benchmark rows produced")
raw <- rbindlist(rows, fill = TRUE)
raw_path <- file.path(out_dir, "simulated_fastpls_raw.csv")
fwrite(raw, raw_path)

cat("Synthetic benchmark written to:", raw_path, "\n")
