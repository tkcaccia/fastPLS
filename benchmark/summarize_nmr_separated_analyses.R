#!/usr/bin/env Rscript

# Separate NMR predictive model selection from backend-only benchmarking.

options(stringsAsFactors = FALSE)

root <- normalizePath(
  Sys.getenv("FASTPLS_REPO", unset = getwd()),
  winslash = "/",
  mustWork = TRUE
)
out_dir <- file.path(
  root,
  "benchmark_results",
  "manuscript_revision_cycle64_20260726"
)
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

required <- c("ggplot2", "patchwork", "scales")
missing <- required[
  !vapply(required, requireNamespace, logical(1), quietly = TRUE)
]
if (length(missing)) {
  stop("Missing packages: ", paste(missing, collapse = ", "))
}

plssvd_prediction_path <- file.path(
  root,
  "benchmark_results",
  "manuscript_revision_cycle16_20260725",
  "nmr_plssvd_extended_lower_grid",
  "heldout",
  "predictions",
  "fastpls_plssvd_cuda_rsvd__rep1.rds"
)
simpls_prediction_path <- file.path(
  root,
  "benchmark_results",
  "manuscript_revision_cycle17_20260725",
  "nmr_simpls_one_se",
  "heldout",
  "predictions",
  "fastpls_simpls_cuda_rsvd__rep1.rds"
)
nmr_task_path <- file.path(
  root,
  "benchmark_results",
  "manuscript_revision_cycle13_20260725",
  "kernel_suite",
  "pipeline1",
  "real_datasets",
  "nmr_task.rds"
)
plssvd_selection_path <- file.path(
  root,
  "benchmark_results",
  "manuscript_revision_cycle17_20260725",
  "nmr_plssvd_one_se_summary.csv"
)
simpls_selection_path <- file.path(
  root,
  "benchmark_results",
  "manuscript_revision_cycle17_20260725",
  "nmr_simpls_one_se_summary.csv"
)
memory_path <- file.path(
  root,
  "benchmark_results",
  "manuscript_revision_cycle21_20260725",
  "selected_memory_baseline_summary.csv"
)
cpu_prediction_paths <- c(
  "PLS-SVD" = file.path(
    root,
    "benchmark_results",
    "manuscript_revision_cycle20_20260725",
    "remote_paired_backend",
    "nmr_plssvd_k5_cpu",
    "predictions",
    "fastpls_plssvd_cpu_rsvd__rep1.rds"
  ),
  "SIMPLS" = file.path(
    root,
    "benchmark_results",
    "manuscript_revision_cycle20_20260725",
    "remote_paired_backend",
    "nmr_simpls_k50_cpu",
    "predictions",
    "fastpls_simpls_cpu_rsvd__rep1.rds"
  )
)
cuda_prediction_paths <- c(
  "PLS-SVD" = plssvd_prediction_path,
  "SIMPLS" = simpls_prediction_path
)
historical_path <- file.path(
  dirname(root),
  "reviewer_experiments",
  "nmr_165",
  "results_remote_float64",
  "nmr_165_summary.csv"
)

all_paths <- c(
  plssvd_prediction_path,
  simpls_prediction_path,
  nmr_task_path,
  plssvd_selection_path,
  simpls_selection_path,
  memory_path,
  cpu_prediction_paths,
  cuda_prediction_paths,
  historical_path
)
missing_paths <- all_paths[!file.exists(all_paths)]
if (length(missing_paths)) {
  stop("Missing required files:\n", paste(missing_paths, collapse = "\n"))
}

method_colours <- c("PLS-SVD" = "#0072B2", "SIMPLS" = "#D55E00")
backend_colours <- c("CPU" = "#0072B2", "CUDA" = "#E69F00")
theme_publication <- function() {
  ggplot2::theme_bw(base_size = 10) +
    ggplot2::theme(
      panel.grid.minor = ggplot2::element_blank(),
      strip.background = ggplot2::element_rect(fill = "#E8E8E8"),
      strip.text = ggplot2::element_text(face = "bold"),
      plot.title = ggplot2::element_text(face = "bold", size = 11),
      legend.position = "bottom"
    )
}

selection_plssvd <- utils::read.csv(
  plssvd_selection_path,
  check.names = FALSE
)
selection_simpls <- utils::read.csv(
  simpls_selection_path,
  check.names = FALSE
)
selection_plssvd$family <- "PLS-SVD"
selection_simpls$family <- "SIMPLS"
selection <- rbind(selection_plssvd, selection_simpls)
selection$selected <- (
  selection$family == "PLS-SVD" & selection$ncomp == 5
) | (
  selection$family == "SIMPLS" & selection$ncomp == 50
)
selection$within_one_se <- factor(
  selection$within_one_se,
  levels = c("False", "True"),
  labels = c("Outside one-SE set", "Within one-SE set")
)
selection$family <- factor(
  selection$family,
  levels = c("PLS-SVD", "SIMPLS")
)
utils::write.csv(
  selection,
  file.path(out_dir, "nmr_family_selected_component_paths.csv"),
  row.names = FALSE
)

prediction_objects <- list(
  "PLS-SVD" = readRDS(plssvd_prediction_path),
  "SIMPLS" = readRDS(simpls_prediction_path)
)
nmr_task <- readRDS(nmr_task_path)

reference_observed <- prediction_objects[[1]]$observed
stopifnot(
  identical(dim(reference_observed), c(321L, 28355L)),
  max(abs(reference_observed - nmr_task$Ytest)) < 1e-12,
  max(abs(
    prediction_objects[[1]]$observed -
      prediction_objects[[2]]$observed
  )) < 1e-12
)

training_intensity <- colMeans(abs(nmr_task$Ytrain))
intensity_breaks <- stats::quantile(
  training_intensity,
  probs = c(0, 0.5, 0.9, 0.99, 1),
  na.rm = TRUE,
  names = FALSE
)
intensity_breaks <- unique(intensity_breaks)
if (length(intensity_breaks) != 5L) {
  stop("Training-intensity quantiles are not unique.")
}
intensity_stratum <- cut(
  training_intensity,
  breaks = intensity_breaks,
  include.lowest = TRUE,
  labels = c(
    "Low (0-50%)",
    "Moderate (50-90%)",
    "High (90-99%)",
    "Peak (top 1%)"
  )
)

bootstrap_metrics <- function(observed, predicted, reps = 10000L, seed = 123L) {
  n <- nrow(observed)
  q <- ncol(observed)
  error <- observed - predicted
  row_sse <- rowSums(error^2)
  row_sae <- rowSums(abs(error))
  row_y <- rowSums(observed)
  row_p <- rowSums(predicted)
  row_y2 <- rowSums(observed^2)
  row_p2 <- rowSums(predicted^2)
  row_yp <- rowSums(observed * predicted)

  metric_from_index <- function(index) {
    count <- length(index) * q
    sy <- sum(row_y[index])
    sp <- sum(row_p[index])
    sy2 <- sum(row_y2[index])
    sp2 <- sum(row_p2[index])
    syp <- sum(row_yp[index])
    covariance <- syp - sy * sp / count
    variance_y <- sy2 - sy^2 / count
    variance_p <- sp2 - sp^2 / count
    correlation <- covariance / sqrt(variance_y * variance_p)
    c(
      RMSD = sqrt(sum(row_sse[index]) / count),
      MAE = sum(row_sae[index]) / count,
      Q2 = correlation^2
    )
  }

  set.seed(seed)
  point <- metric_from_index(seq_len(n))
  boot <- replicate(
    reps,
    metric_from_index(sample.int(n, n, replace = TRUE))
  )
  data.frame(
    metric = names(point),
    estimate = unname(point),
    lower = apply(boot, 1, stats::quantile, probs = 0.025),
    upper = apply(boot, 1, stats::quantile, probs = 0.975),
    confidence_level = 0.95,
    bootstrap_reps = reps,
    uncertainty_scope = paste(
      "Held-out-sample percentile bootstrap conditional on the fixed",
      "321-spectrum outer test set"
    )
  )
}

sample_rows <- list()
response_rows <- list()
intensity_rows <- list()
bootstrap_rows <- list()
summary_rows <- list()

for (family in names(prediction_objects)) {
  object <- prediction_objects[[family]]
  observed <- object$observed
  predicted <- object$predicted
  error <- observed - predicted
  sample_rmsd <- sqrt(rowMeans(error^2))
  response_rmsd <- sqrt(colMeans(error^2))
  response_mae <- colMeans(abs(error))

  sample_rows[[family]] <- data.frame(
    family = family,
    sample = seq_len(nrow(observed)),
    RMSD = sample_rmsd
  )
  response_rows[[family]] <- data.frame(
    family = family,
    response = seq_len(ncol(observed)),
    chemical_shift_ppm = suppressWarnings(
      as.numeric(colnames(observed))
    ),
    training_mean_absolute_intensity = training_intensity,
    intensity_stratum = intensity_stratum,
    RMSD = response_rmsd,
    MAE = response_mae
  )

  for (level in levels(intensity_stratum)) {
    columns <- which(intensity_stratum == level)
    stratum_observed <- observed[, columns, drop = FALSE]
    stratum_predicted <- predicted[, columns, drop = FALSE]
    stratum_error <- stratum_observed - stratum_predicted
    intensity_rows[[paste(family, level)]] <- data.frame(
      family = family,
      intensity_stratum = level,
      n_responses = length(columns),
      aggregate_RMSD = sqrt(mean(stratum_error^2)),
      aggregate_MAE = mean(abs(stratum_error)),
      median_response_RMSD = stats::median(response_rmsd[columns]),
      response_RMSD_q25 = stats::quantile(
        response_rmsd[columns],
        0.25
      ),
      response_RMSD_q75 = stats::quantile(
        response_rmsd[columns],
        0.75
      )
    )
  }

  bootstrap <- bootstrap_metrics(observed, predicted)
  bootstrap$family <- family
  bootstrap$ncomp <- object$row$ncomp
  bootstrap_rows[[family]] <- bootstrap

  summary_rows[[family]] <- data.frame(
    family = family,
    ncomp = object$row$ncomp,
    global_RMSD = sqrt(mean(error^2)),
    Q2_correlation_squared = stats::cor(
      as.vector(observed),
      as.vector(predicted)
    )^2,
    global_MAE = mean(abs(error)),
    sample_RMSD_median = stats::median(sample_rmsd),
    sample_RMSD_q25 = stats::quantile(sample_rmsd, 0.25),
    sample_RMSD_q75 = stats::quantile(sample_rmsd, 0.75),
    sample_RMSD_q975 = stats::quantile(sample_rmsd, 0.975),
    response_RMSD_median = stats::median(response_rmsd),
    response_RMSD_q25 = stats::quantile(response_rmsd, 0.25),
    response_RMSD_q75 = stats::quantile(response_rmsd, 0.75),
    response_RMSD_q975 = stats::quantile(response_rmsd, 0.975)
  )
}

per_sample <- do.call(rbind, sample_rows)
per_response <- do.call(rbind, response_rows)
intensity_summary <- do.call(rbind, intensity_rows)
bootstrap_summary <- do.call(rbind, bootstrap_rows)
family_summary <- do.call(rbind, summary_rows)
row.names(per_sample) <- NULL
row.names(per_response) <- NULL
row.names(intensity_summary) <- NULL
row.names(bootstrap_summary) <- NULL
row.names(family_summary) <- NULL

utils::write.csv(
  per_sample,
  file.path(out_dir, "nmr_family_selected_per_spectrum_rmsd.csv"),
  row.names = FALSE
)
utils::write.csv(
  per_response,
  file.path(out_dir, "nmr_family_selected_response_wise_error.csv"),
  row.names = FALSE
)
utils::write.csv(
  intensity_summary,
  file.path(out_dir, "nmr_family_selected_intensity_stratified_error.csv"),
  row.names = FALSE
)
utils::write.csv(
  bootstrap_summary,
  file.path(out_dir, "nmr_family_selected_bootstrap_uncertainty.csv"),
  row.names = FALSE
)
utils::write.csv(
  family_summary,
  file.path(out_dir, "nmr_family_selected_error_summary.csv"),
  row.names = FALSE
)

memory <- utils::read.csv(memory_path, check.names = FALSE)
paired_backend <- memory[
  memory$dataset == "nmr" &
    memory$method_panel %in% c("plssvd", "simpls"),
  ,
  drop = FALSE
]
paired_backend$family <- ifelse(
  paired_backend$method_panel == "plssvd",
  "PLS-SVD",
  "SIMPLS"
)
paired_backend$implementation_backend <- paired_backend$engine
paired_backend$solver <- "rSVD"
paired_backend$precision_fixed <- "float64"

agreement <- lapply(names(cpu_prediction_paths), function(family) {
  cpu <- readRDS(cpu_prediction_paths[[family]])
  cuda <- readRDS(cuda_prediction_paths[[family]])
  stopifnot(
    max(abs(cpu$observed - cuda$observed)) < 1e-12,
    cpu$row$ncomp == cuda$row$ncomp,
    cpu$row$svd_method == cuda$row$svd_method,
    cpu$row$precision == cuda$row$precision
  )
  data.frame(
    family = family,
    prediction_correlation = stats::cor(
      as.vector(cpu$predicted),
      as.vector(cuda$predicted)
    ),
    prediction_RMSD = sqrt(mean(
      (cpu$predicted - cuda$predicted)^2
    ))
  )
})
agreement <- do.call(rbind, agreement)
paired_backend <- merge(
  paired_backend,
  agreement,
  by = "family",
  all.x = TRUE,
  sort = FALSE
)
paired_backend <- paired_backend[
  order(
    match(paired_backend$family, c("PLS-SVD", "SIMPLS")),
    match(paired_backend$engine, c("CPU", "CUDA"))
  ),
]
utils::write.csv(
  paired_backend,
  file.path(out_dir, "nmr_paired_backend_only_summary.csv"),
  row.names = FALSE
)

historical <- utils::read.csv(historical_path, check.names = FALSE)
historical$comparison_scope <- paste(
  "Historical 165-component workflow rerun using the original",
  "centering-only protocol; not pooled with family-selected analysis"
)
utils::write.csv(
  historical,
  file.path(out_dir, "nmr_historical_reference_165_summary.csv"),
  row.names = FALSE
)

p_selection <- ggplot2::ggplot(
  selection,
  ggplot2::aes(ncomp, mean, colour = family, group = family)
) +
  ggplot2::geom_ribbon(
    ggplot2::aes(ymin = mean - se, ymax = mean + se, fill = family),
    alpha = 0.16,
    colour = NA
  ) +
  ggplot2::geom_line(linewidth = 0.7) +
  ggplot2::geom_point(
    ggplot2::aes(shape = within_one_se),
    size = 2
  ) +
  ggplot2::geom_point(
    data = selection[selection$selected, ],
    shape = 21,
    fill = "white",
    stroke = 1.2,
    size = 3.2
  ) +
  ggplot2::facet_wrap(~family, scales = "free_y") +
  ggplot2::scale_colour_manual(values = method_colours) +
  ggplot2::scale_fill_manual(values = method_colours) +
  ggplot2::scale_shape_manual(
    values = c(
      "Outside one-SE set" = 16,
      "Within one-SE set" = 15
    )
  ) +
  ggplot2::labs(
    title = "A  Training-only component selection",
    subtitle = "Mean +/- SE over five paired training splits; open circle is selected",
    x = "Components",
    y = "Validation RMSD",
    colour = NULL,
    fill = NULL,
    shape = NULL
  ) +
  theme_publication() +
  ggplot2::theme(legend.position = "none")

p_sample <- ggplot2::ggplot(
  per_sample,
  ggplot2::aes(family, RMSD, fill = family)
) +
  ggplot2::geom_violin(
    scale = "width",
    trim = TRUE,
    alpha = 0.7,
    colour = "black",
    linewidth = 0.25
  ) +
  ggplot2::geom_boxplot(
    width = 0.18,
    outlier.shape = NA,
    fill = "white",
    linewidth = 0.35
  ) +
  ggplot2::scale_fill_manual(values = method_colours) +
  ggplot2::scale_y_log10(labels = scales::label_scientific()) +
  ggplot2::labs(
    title = "B  Held-out per-spectrum errors",
    subtitle = "321 spectra at family-selected components",
    x = NULL,
    y = "Per-spectrum RMSD (log scale)",
    fill = NULL
  ) +
  theme_publication() +
  ggplot2::theme(legend.position = "none")

p_response <- ggplot2::ggplot(
  per_response,
  ggplot2::aes(family, RMSD, fill = family)
) +
  ggplot2::geom_violin(
    scale = "width",
    trim = TRUE,
    alpha = 0.7,
    colour = "black",
    linewidth = 0.25
  ) +
  ggplot2::geom_boxplot(
    width = 0.18,
    outlier.shape = NA,
    fill = "white",
    linewidth = 0.35
  ) +
  ggplot2::scale_fill_manual(values = method_colours) +
  ggplot2::scale_y_log10(labels = scales::label_scientific()) +
  ggplot2::labs(
    title = "C  Response-wise errors",
    subtitle = "28,355 spectral coordinates",
    x = NULL,
    y = "Response-wise RMSD (log scale)",
    fill = NULL
  ) +
  theme_publication() +
  ggplot2::theme(legend.position = "none")

intensity_summary$intensity_stratum <- factor(
  intensity_summary$intensity_stratum,
  levels = c(
    "Low (0-50%)",
    "Moderate (50-90%)",
    "High (90-99%)",
    "Peak (top 1%)"
  )
)
p_intensity <- ggplot2::ggplot(
  intensity_summary,
  ggplot2::aes(
    intensity_stratum,
    aggregate_RMSD,
    colour = family,
    group = family
  )
) +
  ggplot2::geom_point(
    position = ggplot2::position_dodge(width = 0.8),
    size = 3
  ) +
  ggplot2::geom_line(
    position = ggplot2::position_dodge(width = 0.8),
    linewidth = 0.55
  ) +
  ggplot2::scale_colour_manual(values = method_colours) +
  ggplot2::scale_y_log10(labels = scales::label_scientific()) +
  ggplot2::labs(
    title = "D  Intensity-stratified prediction error",
    subtitle = "Strata fixed from mean absolute training-response intensity",
    x = NULL,
    y = "Aggregate RMSD (log scale)",
    colour = NULL
  ) +
  theme_publication() +
  ggplot2::theme(
    axis.text.x = ggplot2::element_text(angle = 22, hjust = 1),
    legend.position = "bottom"
  )

resource_rows <- rbind(
  data.frame(
    family = paired_backend$family,
    backend = paired_backend$engine,
    ncomp = paired_backend$effective_ncomp,
    metric = "Total fit + prediction (s)",
    value = paired_backend$total_time_sec_median
  ),
  data.frame(
    family = paired_backend$family,
    backend = paired_backend$engine,
    ncomp = paired_backend$effective_ncomp,
    metric = "Incremental host RSS (MB)",
    value = paired_backend$incremental_host_rss_mb_median
  ),
  data.frame(
    family = paired_backend$family,
    backend = paired_backend$engine,
    ncomp = paired_backend$effective_ncomp,
    metric = "Incremental GPU memory (MB)",
    value = paired_backend$incremental_gpu_mem_mb_median
  )
)
resource_rows$label <- paste0(
  resource_rows$family,
  "\nA=",
  resource_rows$ncomp
)
resource_rows$metric <- factor(
  resource_rows$metric,
  levels = c(
    "Total fit + prediction (s)",
    "Incremental host RSS (MB)",
    "Incremental GPU memory (MB)"
  )
)
p_resource <- ggplot2::ggplot(
  resource_rows,
  ggplot2::aes(label, value, fill = backend)
) +
  ggplot2::geom_col(
    position = ggplot2::position_dodge(width = 0.78),
    width = 0.7,
    colour = "black",
    linewidth = 0.25
  ) +
  ggplot2::facet_wrap(~metric, scales = "free_y", nrow = 1) +
  ggplot2::scale_fill_manual(values = backend_colours) +
  ggplot2::labs(
    title = "E  Paired backend-only computational benchmark",
    subtitle = paste(
      "Within each family: same split, rSVD, float64, component count,",
      "and prediction target; only CPU/CUDA implementation changes"
    ),
    x = NULL,
    y = NULL,
    fill = "Backend"
  ) +
  theme_publication()

main_figure <- (
  p_selection | p_sample
) / (
  p_response | p_intensity
) / p_resource +
  patchwork::plot_layout(heights = c(1, 1, 1.05)) +
  patchwork::plot_annotation(
    title = "NMR prediction: predictive selection and implementation benchmarking",
    subtitle = paste(
      "Predictive inference (A-D) is separated from the backend-only",
      "resource comparison (E)"
    ),
    theme = ggplot2::theme(
      plot.title = ggplot2::element_text(face = "bold", size = 14),
      plot.subtitle = ggplot2::element_text(size = 10)
    )
  )

ggplot2::ggsave(
  file.path(out_dir, "nmr_separated_predictive_and_backend_benchmark.png"),
  main_figure,
  width = 11,
  height = 11.2,
  dpi = 320,
  bg = "white"
)
ggplot2::ggsave(
  file.path(out_dir, "nmr_separated_predictive_and_backend_benchmark.pdf"),
  main_figure,
  width = 11,
  height = 11.2,
  device = grDevices::cairo_pdf
)

historical_plot_data <- historical[
  historical$variant_name %in% c(
    "nature_fastsimpls_plssvd",
    "cpp_plssvd_cpu_rsvd",
    "cpp_simpls_cpu_rsvd",
    "gpu_plssvd_rsvd",
    "gpu_simpls_rsvd"
  ),
]
historical_labels <- c(
  "nature_fastsimpls_plssvd" = "Deposited PLS-SVD/IRLBA",
  "cpp_plssvd_cpu_rsvd" = "fastPLS CPU PLS-SVD/rSVD",
  "cpp_simpls_cpu_rsvd" = "fastPLS CPU SIMPLS/rSVD",
  "gpu_plssvd_rsvd" = "fastPLS CUDA PLS-SVD/rSVD",
  "gpu_simpls_rsvd" = "fastPLS CUDA SIMPLS/rSVD"
)
historical_plot_data$workflow <- unname(
  historical_labels[historical_plot_data$variant_name]
)
historical_plot_data$workflow <- factor(
  historical_plot_data$workflow,
  levels = rev(unname(historical_labels))
)
historical_long <- rbind(
  data.frame(
    workflow = historical_plot_data$workflow,
    metric = "Total time (s)",
    value = historical_plot_data$total_time_sec_median
  ),
  data.frame(
    workflow = historical_plot_data$workflow,
    metric = "Global RMSD",
    value = historical_plot_data$global_rmsd
  ),
  data.frame(
    workflow = historical_plot_data$workflow,
    metric = "Incremental host RSS (MB)",
    value = historical_plot_data$incremental_peak_host_rss_mb_median
  )
)
historical_figure <- ggplot2::ggplot(
  historical_long,
  ggplot2::aes(workflow, value, fill = workflow)
) +
  ggplot2::geom_col(colour = "black", linewidth = 0.25) +
  ggplot2::facet_wrap(~metric, scales = "free_x", nrow = 1) +
  ggplot2::coord_flip() +
  ggplot2::scale_y_log10(labels = scales::label_scientific()) +
  ggplot2::labs(
    title = "Historical NMR workflow comparison at the published 165 components",
    subtitle = paste(
      "Composite workflow context: family, solver, implementation, and",
      "hardware are not held fixed across every row"
    ),
    x = NULL,
    y = NULL,
    fill = NULL
  ) +
  theme_publication() +
  ggplot2::theme(legend.position = "none")
ggplot2::ggsave(
  file.path(out_dir, "nmr_historical_reference_165.png"),
  historical_figure,
  width = 11,
  height = 4.8,
  dpi = 320,
  bg = "white"
)
ggplot2::ggsave(
  file.path(out_dir, "nmr_historical_reference_165.pdf"),
  historical_figure,
  width = 11,
  height = 4.8,
  device = grDevices::cairo_pdf
)

report <- c(
  "# Separated NMR analyses",
  "",
  "## Predictive model selection",
  "",
  paste(
    "- PLS-SVD: 5 components, held-out RMSD",
    sprintf("%.7f.", family_summary$global_RMSD[
      family_summary$family == "PLS-SVD"
    ])
  ),
  paste(
    "- SIMPLS: 50 components, held-out RMSD",
    sprintf("%.7f.", family_summary$global_RMSD[
      family_summary$family == "SIMPLS"
    ])
  ),
  paste(
    "- Prediction uncertainty uses 10,000 held-out-sample bootstrap",
    "resamples and is conditional on the fixed outer split."
  ),
  paste(
    "- Intensity strata were fixed from mean absolute Ytrain intensity:",
    "0-50%, 50-90%, 90-99%, and top 1%."
  ),
  "",
  "## Backend-only implementation benchmark",
  "",
  paste(
    "- CPU and CUDA are compared separately within PLS-SVD (A=5) and",
    "SIMPLS (A=50), holding solver, precision, split, and target fixed."
  ),
  sprintf(
    "- CPU/CUDA prediction correlation: PLS-SVD %.8f; SIMPLS %.8f.",
    agreement$prediction_correlation[agreement$family == "PLS-SVD"],
    agreement$prediction_correlation[agreement$family == "SIMPLS"]
  ),
  "",
  "## Historical workflow",
  "",
  paste(
    "- The original diffusion-edited scientific workflow used 165",
    "components. The deposited fastsimpls() function was therefore rerun at",
    "165 components for historical context."
  ),
  paste(
    "- The earlier 100-component deposited-reference display was an",
    "artificial equal-size sensitivity analysis, not the original scientific",
    "setting and not an implementation-only comparison."
  )
)
writeLines(
  report,
  file.path(out_dir, "NMR_SEPARATED_ANALYSES_NOTE.md")
)

capture.output(
  sessionInfo(),
  file = file.path(out_dir, "session_info.txt")
)

message("Wrote NMR separated-analysis outputs to: ", out_dir)
