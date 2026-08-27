#!/usr/bin/env Rscript

args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 2L) {
  stop("Usage: summarize_nmr_predictions.R <nmr-result-dir> <output-dir>")
}
input <- normalizePath(args[[1L]], mustWork = TRUE)
output <- args[[2L]]
dir.create(output, recursive = TRUE, showWarnings = FALSE)
`%||%` <- function(x, y) if (is.null(x) || !length(x)) y else x

routes <- c(
  selected_plssvd_cpu_irlba_n5 = "PLS-SVD CPU / IRLBA (5)",
  selected_plssvd_cpu_rsvd_n5 = "PLS-SVD CPU / rSVD (5)",
  selected_plssvd_cuda_rsvd_n5 = "PLS-SVD CUDA / rSVD (5)",
  selected_simpls_cpu_irlba_n50 = "SIMPLS CPU / IRLBA (50)",
  selected_simpls_cpu_rsvd_n50 = "SIMPLS CPU / rSVD (50)",
  selected_simpls_cuda_rsvd_n50 = "SIMPLS CUDA / rSVD (50)"
)

sample_rows <- list()
response_rows <- list()
representative <- NULL
selection <- NULL

for (key in names(routes)) {
  path <- file.path(input, "predictions", paste0(key, ".rds"))
  if (!file.exists(path)) stop("Missing prediction artifact: ", path)
  object <- readRDS(path)
  observed <- as.matrix(object$observed)
  predicted <- as.matrix(object$predicted)
  if (!identical(dim(observed), dim(predicted))) {
    stop("Observed and predicted dimensions differ for ", key)
  }
  sample_rmsd <- sqrt(rowMeans((observed - predicted)^2))
  response_rmsd <- sqrt(colMeans((observed - predicted)^2))
  response_mae <- colMeans(abs(observed - predicted))
  sample_rows[[key]] <- data.frame(
    route = routes[[key]], route_id = key,
    sample_index = seq_len(nrow(observed)),
    sample_id = rownames(observed) %||% as.character(seq_len(nrow(observed))),
    RMSD = sample_rmsd
  )
  response_rows[[key]] <- data.frame(
    route = routes[[key]], route_id = key,
    response_index = seq_len(ncol(observed)),
    ppm = suppressWarnings(as.numeric(colnames(observed))),
    RMSD = response_rmsd, MAE = response_mae
  )
  if (key == "selected_simpls_cuda_rsvd_n50") {
    target <- median(sample_rmsd)
    index <- which.min(abs(sample_rmsd - target))
    representative <- data.frame(
      response_index = seq_len(ncol(observed)),
      ppm = suppressWarnings(as.numeric(colnames(observed))),
      observed = observed[index, ], predicted = predicted[index, ]
    )
    selection <- data.frame(
      route = routes[[key]], sample_index = index,
      sample_id = (rownames(observed) %||% as.character(seq_len(nrow(observed))))[[index]],
      sample_RMSD = sample_rmsd[[index]], median_sample_RMSD = target,
      selection_rule = "closest held-out sample to median per-spectrum RMSD"
    )
  }
  rm(object, observed, predicted)
  gc(FALSE)
}

write.csv(do.call(rbind, sample_rows),
          file.path(output, "nmr_frozen_per_sample.csv"), row.names = FALSE)
write.csv(do.call(rbind, response_rows),
          file.path(output, "nmr_frozen_per_response.csv"), row.names = FALSE)
write.csv(representative,
          file.path(output, "nmr_frozen_representative_spectrum.csv"), row.names = FALSE)
write.csv(selection,
          file.path(output, "nmr_frozen_representative_selection.csv"), row.names = FALSE)

raw <- read.csv(file.path(input, "nmr_all_runs.csv"), check.names = FALSE)
raw$oversample_label <- ifelse(is.na(raw$oversample), "not_applicable",
                               as.character(raw$oversample))
raw$power_label <- ifelse(is.na(raw$power), "not_applicable",
                          as.character(raw$power))
keys <- c("family", "backend", "solver", "precision", "ncomp",
          "oversample_label", "power_label")
summary <- aggregate(
  raw[c("fit_time_sec", "predict_time_sec", "total_time_sec", "RMSD", "Q2",
        "MAE", "median_sample_RMSD", "p95_sample_RMSD", "baseline_rss_mb",
        "after_fit_rss_mb")],
  raw[keys],
  function(x) c(median = median(x, na.rm = TRUE),
                q25 = unname(quantile(x, 0.25, na.rm = TRUE)),
                q75 = unname(quantile(x, 0.75, na.rm = TRUE)))
)
flat <- summary[keys]
names(flat)[names(flat) == "oversample_label"] <- "oversample"
names(flat)[names(flat) == "power_label"] <- "power"
for (name in setdiff(names(summary), keys)) {
  value <- summary[[name]]
  if (!is.matrix(value)) value <- do.call(rbind, value)
  colnames(value) <- paste(name, colnames(value), sep = "_")
  flat <- cbind(flat, value)
}
write.csv(flat, file.path(output, "nmr_frozen_route_summary.csv"), row.names = FALSE)

writeLines(c(
  "fastPLS version: 0.99.25",
  "source archive SHA-256: 604f74d72f5e9540f7378efd37f2ca6ed87ccb9e73b912211331a8cd97233481",
  "Representative spectrum: closest held-out sample to the median per-spectrum RMSD under family-selected SIMPLS CUDA/rSVD (50 components)."
), file.path(output, "README.txt"))
