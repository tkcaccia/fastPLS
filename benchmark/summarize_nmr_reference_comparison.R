#!/usr/bin/env Rscript

args <- commandArgs(trailingOnly = TRUE)
input_dir <- args[[1L]]
output_dir <- args[[2L]]
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

row_files <- list.files(
  file.path(input_dir, "rows"), pattern = "\\.csv$", full.names = TRUE
)
if (!length(row_files)) stop("No benchmark rows found.", call. = FALSE)
raw <- do.call(rbind, lapply(row_files, utils::read.csv, check.names = FALSE))

reference_predictions <- list.files(
  file.path(input_dir, "predictions"),
  pattern = "^deposited_fastsimpls_irlba__rep[0-9]+\\.rds$",
  full.names = TRUE
)
if (!length(reference_predictions)) stop("Reference predictions are missing.", call. = FALSE)
reference <- readRDS(reference_predictions[[1L]])$predicted

prediction_files <- list.files(
  file.path(input_dir, "predictions"), pattern = "\\.rds$", full.names = TRUE
)
agreement <- do.call(rbind, lapply(prediction_files, function(path) {
  object <- readRDS(path)
  data.frame(
    variant = object$row$variant,
    repetition = as.integer(sub(".*__rep([0-9]+)\\.rds$", "\\1", path)),
    prediction_correlation_vs_reference = stats::cor(
      as.vector(reference), as.vector(object$predicted),
      use = "pairwise.complete.obs"
    ),
    prediction_rmsd_vs_reference = sqrt(mean(
      (reference - object$predicted)^2
    )),
    stringsAsFactors = FALSE
  )
}))
raw <- merge(raw, agreement, by = c("variant", "repetition"), all.x = TRUE)

median_iqr <- function(x) {
  c(median = stats::median(x, na.rm = TRUE), iqr = stats::IQR(x, na.rm = TRUE))
}
variants <- unique(raw$variant)
summary <- do.call(rbind, lapply(variants, function(id) {
  x <- raw[raw$variant == id, , drop = FALSE]
  data.frame(
    variant = id,
    algorithm = x$algorithm[[1L]],
    backend = x$backend[[1L]],
    svd_method = x$svd_method[[1L]],
    precision = x$precision[[1L]],
    n_repetitions = nrow(x),
    total_time_sec_median = median_iqr(x$total_time_sec)[["median"]],
    total_time_sec_iqr = median_iqr(x$total_time_sec)[["iqr"]],
    host_rss_mb_median = median_iqr(x$host_rss_mb)[["median"]],
    host_rss_mb_iqr = median_iqr(x$host_rss_mb)[["iqr"]],
    gpu_peak_mb_median = if (all(is.na(x$gpu_peak_mb))) NA_real_ else
      median_iqr(x$gpu_peak_mb)[["median"]],
    gpu_peak_mb_iqr = if (all(is.na(x$gpu_peak_mb))) NA_real_ else
      median_iqr(x$gpu_peak_mb)[["iqr"]],
    R2_median = median_iqr(x$R2)[["median"]],
    Q2_median = median_iqr(x$Q2)[["median"]],
    RMSD_median = median_iqr(x$RMSD)[["median"]],
    RMSD_iqr = median_iqr(x$RMSD)[["iqr"]],
    prediction_correlation_vs_reference = median_iqr(
      x$prediction_correlation_vs_reference
    )[["median"]],
    prediction_rmsd_vs_reference = median_iqr(
      x$prediction_rmsd_vs_reference
    )[["median"]],
    stringsAsFactors = FALSE
  )
}))

utils::write.csv(raw, file.path(output_dir, "nmr_reference_comparison_raw.csv"),
                 row.names = FALSE)
utils::write.csv(summary, file.path(output_dir, "nmr_reference_comparison_summary.csv"),
                 row.names = FALSE)
print(summary)
