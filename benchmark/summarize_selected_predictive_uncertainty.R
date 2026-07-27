#!/usr/bin/env Rscript

root <- normalizePath(
  Sys.getenv("FASTPLS_REPO_ROOT", getwd()),
  winslash = "/",
  mustWork = TRUE
)
selected_file <- file.path(
  root,
  "benchmark_results/manuscript_revision_cycle17_20260725",
  "selected_backend_cycle17_chosen.csv"
)
out_dir <- file.path(
  root,
  "benchmark_results/manuscript_revision_cycle19_20260725"
)
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

bootstrap_reps <- 10000L
set.seed(123)

n_test <- c(
  metref = 100L,
  ccle = 71L,
  tcga_brca = 88L,
  tcga_hnsc_methylation = 58L,
  gtex_v8 = 797L,
  tcga_pan_cancer = 982L,
  retina = 22406L,
  tabula = 50059L,
  cifar100 = 10000L,
  cbmc_citeseq = 862L,
  prism = 54L,
  nmr = 321L
)

wilson_interval <- function(successes, n, z = 1.959963984540054) {
  p <- successes / n
  denominator <- 1 + z^2 / n
  center <- (p + z^2 / (2 * n)) / denominator
  half <- z * sqrt((p * (1 - p) + z^2 / (4 * n)) / n) / denominator
  c(max(0, center - half), min(1, center + half))
}

read_regression_prediction <- function(path) {
  object <- readRDS(path)
  if (!is.null(object$pred) && !is.null(object$truth)) {
    predicted <- as.matrix(object$pred)
    observed <- as.matrix(object$truth)
  } else if (!is.null(object$predicted) && !is.null(object$observed)) {
    predicted <- as.matrix(object$predicted)
    observed <- as.matrix(object$observed)
  } else {
    stop("Unsupported prediction object: ", path)
  }
  if (!identical(dim(predicted), dim(observed))) {
    stop("Prediction and truth dimensions differ: ", path)
  }
  row_mse <- rowMeans((predicted - observed)^2)
  list(
    point = sqrt(mean(row_mse)),
    row_mse = row_mse,
    n_test = length(row_mse)
  )
}

bootstrap_rmsd <- function(row_mse, reps) {
  n <- length(row_mse)
  draws <- vapply(
    seq_len(reps),
    function(i) sqrt(mean(row_mse[sample.int(n, n, replace = TRUE)])),
    numeric(1)
  )
  unname(stats::quantile(draws, c(0.025, 0.975), names = FALSE))
}

regression_prediction_path <- function(dataset, variant) {
  base <- file.path(
    root,
    "benchmark_results/manuscript_revision_cycle19_20260725",
    "regression_selected_predictions",
    "predictions"
  )
  if (dataset == "nmr" && variant == "fastpls_plssvd_cuda_rsvd") {
    return(file.path(
      root,
      "benchmark_results/manuscript_revision_cycle16_20260725",
      "nmr_plssvd_extended_lower_grid/heldout/predictions",
      "fastpls_plssvd_cuda_rsvd__rep1.rds"
    ))
  }
  if (dataset == "nmr" && variant == "fastpls_simpls_cuda_rsvd") {
    return(file.path(
      root,
      "benchmark_results/manuscript_revision_cycle17_20260725",
      "nmr_simpls_one_se/heldout/predictions",
      "fastpls_simpls_cuda_rsvd__rep1.rds"
    ))
  }
  k <- c(
    cpp_plssvd_cpu_rsvd = 10L,
    gpu_simpls_rsvd = 50L,
    gpu_opls_rsvd = 50L,
    gpu_kernelpls_rsvd = 50L,
    cpp_simpls_cpu_rsvd = 5L,
    cpp_opls_cpu_rsvd = 2L,
    cpp_kernelpls_cpu_rsvd = 5L
  )[[variant]]
  file.path(base, sprintf("%s__%s__k%d.rds", dataset, variant, k))
}

selected <- utils::read.csv(selected_file, stringsAsFactors = FALSE)
selected <- selected[
  selected$status == "ok" & is.finite(selected$metric_median),
  ,
  drop = FALSE
]

rows <- vector("list", nrow(selected))
for (i in seq_len(nrow(selected))) {
  row <- selected[i, ]
  dataset <- row$dataset
  n <- unname(n_test[[dataset]])
  if (row$metric_name == "accuracy") {
    successes <- as.integer(round(row$metric_median * n))
    point <- successes / n
    interval <- wilson_interval(successes, n)
    method <- "Wilson score interval"
    reps <- NA_integer_
  } else {
    prediction_path <- regression_prediction_path(
      dataset,
      row$variant_name
    )
    prediction <- read_regression_prediction(prediction_path)
    point <- prediction$point
    interval <- bootstrap_rmsd(prediction$row_mse, bootstrap_reps)
    method <- "held-out-sample percentile bootstrap"
    reps <- bootstrap_reps
    n <- prediction$n_test
  }
  rows[[i]] <- data.frame(
    dataset = dataset,
    method_panel = row$method_panel,
    variant_name = row$variant_name,
    effective_ncomp = as.integer(row$effective_ncomp),
    metric_name = row$metric_name,
    point_estimate = point,
    ci_lower = interval[[1L]],
    ci_upper = interval[[2L]],
    ci_level = 0.95,
    ci_method = method,
    n_test = as.integer(n),
    bootstrap_reps = reps,
    uncertainty_scope = paste(
      "Conditional on the prespecified held-out split;",
      "does not include training-split variability"
    ),
    stringsAsFactors = FALSE
  )
}

result <- do.call(rbind, rows)
utils::write.csv(
  result,
  file.path(out_dir, "selected_predictive_uncertainty.csv"),
  row.names = FALSE
)

compact <- result
compact$estimate_ci <- ifelse(
  compact$metric_name == "accuracy",
  sprintf(
    "%.3f [%.3f, %.3f]",
    compact$point_estimate,
    compact$ci_lower,
    compact$ci_upper
  ),
  sprintf(
    "%.4g [%.4g, %.4g]",
    compact$point_estimate,
    compact$ci_lower,
    compact$ci_upper
  )
)
utils::write.csv(
  compact[
    ,
    c(
      "dataset", "method_panel", "effective_ncomp", "metric_name",
      "estimate_ci", "n_test", "ci_method"
    )
  ],
  file.path(out_dir, "selected_predictive_uncertainty_compact.csv"),
  row.names = FALSE
)

imagenet <- data.frame(
  representation = c(
    "Raw DINOv2",
    rep("PCA-rSVD", 3L),
    rep("PLS-SVD/rSVD", 3L)
  ),
  dimension = c(1024L, 50L, 100L, 200L, 50L, 100L, 200L),
  top1 = c(0.6556, 0.5929, 0.6260, 0.6430, 0.6085, 0.6370, 0.6516),
  top5 = c(0.9392, 0.9215, 0.9341, 0.9383, 0.9259, 0.9364, 0.9397),
  stringsAsFactors = FALSE
)
imagenet_n <- 281167L
imagenet_rows <- do.call(
  rbind,
  lapply(seq_len(nrow(imagenet)), function(i) {
    top1_success <- round(imagenet$top1[[i]] * imagenet_n)
    top5_success <- round(imagenet$top5[[i]] * imagenet_n)
    top1_ci <- wilson_interval(top1_success, imagenet_n)
    top5_ci <- wilson_interval(top5_success, imagenet_n)
    data.frame(
      representation = imagenet$representation[[i]],
      dimension = imagenet$dimension[[i]],
      top1 = top1_success / imagenet_n,
      top1_lower = top1_ci[[1L]],
      top1_upper = top1_ci[[2L]],
      top5 = top5_success / imagenet_n,
      top5_lower = top5_ci[[1L]],
      top5_upper = top5_ci[[2L]],
      n_test = imagenet_n,
      ci_method = "Wilson score interval",
      stringsAsFactors = FALSE
    )
  })
)
utils::write.csv(
  imagenet_rows,
  file.path(out_dir, "imagenet_predictive_uncertainty.csv"),
  row.names = FALSE
)

cat(file.path(out_dir, "selected_predictive_uncertainty.csv"), "\n")
