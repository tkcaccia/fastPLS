#!/usr/bin/env Rscript

options(stringsAsFactors = FALSE)

`%||%` <- function(x, y) if (is.null(x) || !length(x)) y else x

args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 2L) {
  stop(
    "Usage: plot_nmr_spectrum_prediction.R TASK_RDS OUTPUT_DIR ",
    "[NCOMP=100] [BACKEND=cuda] [LIB_LOC='']",
    call. = FALSE
  )
}

task_file <- normalizePath(path.expand(args[[1L]]), mustWork = TRUE)
output_dir <- path.expand(args[[2L]])
ncomp <- if (length(args) >= 3L) as.integer(args[[3L]]) else 100L
backend <- if (length(args) >= 4L) tolower(args[[4L]]) else "cuda"
lib_loc <- if (length(args) >= 5L) path.expand(args[[5L]]) else ""

if (!is.finite(ncomp) || is.na(ncomp) || ncomp < 1L) {
  stop("NCOMP must be a positive integer.", call. = FALSE)
}
if (!backend %in% c("cpu", "cuda", "metal")) {
  stop("BACKEND must be cpu, cuda, or metal.", call. = FALSE)
}
if (nzchar(lib_loc)) {
  .libPaths(unique(c(normalizePath(lib_loc, mustWork = TRUE), .libPaths())))
}

suppressPackageStartupMessages(library(fastPLS))
if (!requireNamespace("ggplot2", quietly = TRUE)) {
  stop("The ggplot2 package is required to create the NMR figure.", call. = FALSE)
}

dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

is_float32 <- function(x) inherits(x, "float32") || methods::is(x, "float32")
as_double_matrix <- function(x) {
  if (is_float32(x)) return(float::dbl(x))
  as.matrix(x)
}

last_prediction <- function(x) {
  if (is.null(x)) stop("Prediction output is empty.", call. = FALSE)
  if (is.list(x) && !is.data.frame(x)) return(last_prediction(x[[length(x)]]))
  if (length(dim(x)) == 3L) return(x[, , dim(x)[3L], drop = TRUE])
  x
}

task <- readRDS(task_file)
required <- c("Xtrain", "Ytrain", "Xtest", "Ytest")
if (!all(required %in% names(task))) {
  stop("TASK_RDS must contain Xtrain, Ytrain, Xtest, and Ytest.", call. = FALSE)
}
if (is.factor(task$Ytrain) || is.factor(task$Ytest)) {
  stop("The NMR spectrum figure requires a numeric regression task.", call. = FALSE)
}

if (identical(backend, "cuda") && !isTRUE(has_cuda())) {
  stop("The installed fastPLS build does not provide CUDA.", call. = FALSE)
}
if (identical(backend, "metal") && !isTRUE(has_metal())) {
  stop("The installed fastPLS build does not provide Metal.", call. = FALSE)
}

set.seed(123)
fit_elapsed <- system.time({
  model <- pls(
    task$Xtrain,
    task$Ytrain,
    ncomp = ncomp,
    method = "simpls",
    svd.method = "rsvd",
    backend = backend,
    fit = FALSE,
    return_variance = FALSE,
    seed = 123
  )
})[["elapsed"]]

predict_elapsed <- system.time({
  prediction <- predict(model, task$Xtest, backend = backend)
})[["elapsed"]]

observed <- as_double_matrix(task$Ytest)
predicted <- as_double_matrix(last_prediction(prediction$Ypred))
if (!identical(dim(observed), dim(predicted))) {
  stop(
    "Observed and predicted response dimensions differ: ",
    paste(dim(observed), collapse = "x"), " versus ",
    paste(dim(predicted), collapse = "x"), ".",
    call. = FALSE
  )
}

sample_rmsd <- sqrt(rowMeans((observed - predicted)^2, na.rm = TRUE))
target_rmsd <- stats::median(sample_rmsd, na.rm = TRUE)
sample_index <- which.min(abs(sample_rmsd - target_rmsd))
observed_one <- observed[sample_index, ]
predicted_one <- predicted[sample_index, ]
residual_one <- observed_one - predicted_one

response_names <- colnames(observed)
axis_numeric <- suppressWarnings(as.numeric(response_names))
has_numeric_axis <- length(axis_numeric) == ncol(observed) && all(is.finite(axis_numeric))
if (has_numeric_axis) {
  spectral_axis <- axis_numeric
  axis_label <- if (diff(range(spectral_axis)) <= 30) "Chemical shift (ppm)" else "Spectral variable"
} else {
  spectral_axis <- seq_len(ncol(observed))
  axis_label <- "Spectral variable"
}

overlay <- rbind(
  data.frame(x = spectral_axis, intensity = observed_one, series = "Observed"),
  data.frame(x = spectral_axis, intensity = predicted_one, series = "SIMPLS-rSVD prediction")
)
residual <- data.frame(x = spectral_axis, residual = residual_one)

common_theme <- ggplot2::theme_classic(base_size = 13) +
  ggplot2::theme(
    plot.title = ggplot2::element_text(face = "bold", size = 14),
    legend.position = "bottom",
    legend.title = ggplot2::element_blank(),
    axis.title = ggplot2::element_text(face = "bold")
  )

p_overlay <- ggplot2::ggplot(
  overlay,
  ggplot2::aes(x = x, y = intensity, colour = series, linewidth = series)
) +
  ggplot2::geom_line(alpha = 0.88) +
  ggplot2::scale_colour_manual(values = c(
    "Observed" = "#111111",
    "SIMPLS-rSVD prediction" = "#D55E00"
  )) +
  ggplot2::scale_linewidth_manual(values = c(
    "Observed" = 0.45,
    "SIMPLS-rSVD prediction" = 0.35
  )) +
  ggplot2::labs(
    title = "Observed and predicted NMR spectrum",
    subtitle = sprintf(
      "Median-RMSD representative test sample %d; SIMPLS-rSVD, %d components, %s backend",
      sample_index, ncomp, toupper(backend)
    ),
    x = NULL,
    y = "Intensity"
  ) +
  common_theme

p_residual <- ggplot2::ggplot(residual, ggplot2::aes(x = x, y = residual)) +
  ggplot2::geom_hline(yintercept = 0, colour = "#777777", linewidth = 0.3) +
  ggplot2::geom_line(colour = "#0072B2", linewidth = 0.35) +
  ggplot2::labs(x = axis_label, y = "Residual") +
  common_theme +
  ggplot2::theme(legend.position = "none", plot.title = ggplot2::element_blank())

if (has_numeric_axis && identical(axis_label, "Chemical shift (ppm)")) {
  p_overlay <- p_overlay + ggplot2::scale_x_reverse()
  p_residual <- p_residual + ggplot2::scale_x_reverse()
}

if (requireNamespace("patchwork", quietly = TRUE)) {
  combined <- patchwork::wrap_plots(
    p_overlay,
    p_residual,
    ncol = 1,
    heights = c(3, 1)
  )
  ggplot2::ggsave(
    file.path(output_dir, "nmr_observed_predicted_spectrum.png"),
    combined, width = 11, height = 6.8, dpi = 320
  )
  ggplot2::ggsave(
    file.path(output_dir, "nmr_observed_predicted_spectrum.pdf"),
    combined, width = 11, height = 6.8, device = grDevices::cairo_pdf
  )
} else {
  ggplot2::ggsave(
    file.path(output_dir, "nmr_observed_predicted_spectrum.png"),
    p_overlay, width = 11, height = 5.4, dpi = 320
  )
  ggplot2::ggsave(
    file.path(output_dir, "nmr_observed_predicted_residual.png"),
    p_residual, width = 11, height = 2.6, dpi = 320
  )
}

figure_data <- data.frame(
  spectral_axis = spectral_axis,
  observed = observed_one,
  predicted = predicted_one,
  residual = residual_one
)
utils::write.csv(
  figure_data,
  file.path(output_dir, "nmr_observed_predicted_spectrum_data.csv"),
  row.names = FALSE
)

metadata <- data.frame(
  task_file = task_file,
  split_seed = task$split_seed %||% 123L,
  test_sample_index = sample_index,
  selection_rule = "sample RMSD closest to median test-sample RMSD",
  n_train = nrow(task$Xtrain),
  n_test = nrow(task$Xtest),
  p = ncol(task$Xtrain),
  q = ncol(observed),
  method = "simpls",
  svd_method = "rsvd",
  backend = backend,
  ncomp = ncomp,
  fit_time_sec = unname(fit_elapsed),
  predict_time_sec = unname(predict_elapsed),
  sample_rmsd = sample_rmsd[[sample_index]],
  median_test_sample_rmsd = target_rmsd,
  sample_correlation = stats::cor(observed_one, predicted_one, use = "pairwise.complete.obs"),
  stringsAsFactors = FALSE
)
utils::write.csv(
  metadata,
  file.path(output_dir, "nmr_observed_predicted_spectrum_metadata.csv"),
  row.names = FALSE
)

message("NMR spectrum figure written to: ", normalizePath(output_dir))
