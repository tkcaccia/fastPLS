#!/usr/bin/env Rscript

root <- normalizePath(
  Sys.getenv("FASTPLS_REPO_ROOT", getwd()),
  winslash = "/",
  mustWork = TRUE
)
out_dir <- file.path(
  root,
  "benchmark_results/manuscript_revision_cycle20_20260725"
)
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

source_all <- file.path(
  root,
  "benchmark_results/manuscript_revision_cycle13_20260725",
  "selected_backend_cycle13_all.csv"
)
source_chosen <- file.path(
  root,
  "benchmark_results/manuscript_revision_cycle17_20260725",
  "selected_backend_cycle17_chosen.csv"
)
remote_root <- file.path(out_dir, "remote_paired_backend")
bootstrap_reps <- 10000L
set.seed(123)

dataset_order <- c(
  "metref", "ccle", "tcga_brca", "tcga_hnsc_methylation",
  "gtex_v8", "tcga_pan_cancer", "retina", "tabula", "cifar100",
  "cbmc_citeseq", "prism", "nmr"
)
method_order <- c("plssvd", "simpls", "opls", "kernelpls")
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

wilson_interval <- function(successes, n, z = 1.95996398454005) {
  p <- successes / n
  denominator <- 1 + z^2 / n
  center <- (p + z^2 / (2 * n)) / denominator
  half <- z * sqrt((p * (1 - p) + z^2 / (4 * n)) / n) /
    denominator
  c(max(0, center - half), min(1, center + half))
}

read_prediction <- function(path) {
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

prediction_path <- function(dataset, variant, ncomp) {
  if (dataset %in% c("cbmc_citeseq", "prism")) {
    return(file.path(
      remote_root,
      "selected_regression",
      "predictions",
      sprintf("%s__%s__k%d.rds", dataset, variant, ncomp)
    ))
  }
  if (dataset == "nmr" && variant == "fastpls_plssvd_cpu_rsvd") {
    return(file.path(
      remote_root,
      "nmr_plssvd_k5_cpu",
      "predictions",
      "fastpls_plssvd_cpu_rsvd__rep1.rds"
    ))
  }
  if (dataset == "nmr" && variant == "fastpls_simpls_cpu_rsvd") {
    return(file.path(
      remote_root,
      "nmr_simpls_k50_cpu",
      "predictions",
      "fastpls_simpls_cpu_rsvd__rep1.rds"
    ))
  }
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
  stop("No prediction path for ", dataset, " / ", variant)
}

summarize_nmr_rows <- function(paths, family, engine) {
  rows <- do.call(
    rbind,
    lapply(paths, utils::read.csv, stringsAsFactors = FALSE)
  )
  data.frame(
    dataset = "nmr",
    task_type = "regression",
    method_panel = family,
    variant_name = rows$variant[[1L]],
    engine = engine,
    backend = paste0(tolower(engine), "_rsvd"),
    classifier = "not_applicable",
    requested_ncomp = rows$ncomp[[1L]],
    effective_ncomp = rows$ncomp[[1L]],
    precision = "float64",
    execution_precision = "float64",
    metric_name = "rmsd",
    n_runs = nrow(rows),
    metric_median = stats::median(rows$RMSD),
    metric_q25 = unname(stats::quantile(rows$RMSD, 0.25)),
    metric_q75 = unname(stats::quantile(rows$RMSD, 0.75)),
    total_time_sec_median = stats::median(rows$total_time_sec),
    total_time_sec_q25 = unname(
      stats::quantile(rows$total_time_sec, 0.25)
    ),
    total_time_sec_q75 = unname(
      stats::quantile(rows$total_time_sec, 0.75)
    ),
    host_rss_mb_median = stats::median(rows$host_rss_mb),
    host_rss_mb_q25 = unname(
      stats::quantile(rows$host_rss_mb, 0.25)
    ),
    host_rss_mb_q75 = unname(
      stats::quantile(rows$host_rss_mb, 0.75)
    ),
    gpu_mem_mb_median = if (all(is.na(rows$gpu_peak_mb))) {
      NA_real_
    } else {
      stats::median(rows$gpu_peak_mb, na.rm = TRUE)
    },
    gpu_mem_mb_q25 = if (all(is.na(rows$gpu_peak_mb))) {
      NA_real_
    } else {
      unname(stats::quantile(rows$gpu_peak_mb, 0.25, na.rm = TRUE))
    },
    gpu_mem_mb_q75 = if (all(is.na(rows$gpu_peak_mb))) {
      NA_real_
    } else {
      unname(stats::quantile(rows$gpu_peak_mb, 0.75, na.rm = TRUE))
    },
    status = if (all(rows$status == "ok")) "ok" else "failed",
    stringsAsFactors = FALSE
  )
}

paired <- utils::read.csv(source_all, stringsAsFactors = FALSE)
paired$status <- "ok"
paired <- paired[paired$dataset != "nmr", , drop = FALSE]

nmr_rows <- rbind(
  summarize_nmr_rows(
    list.files(
      file.path(remote_root, "nmr_plssvd_k5_cpu", "rows"),
      pattern = "\\.csv$",
      full.names = TRUE
    ),
    "plssvd",
    "CPU"
  ),
  summarize_nmr_rows(
    list.files(
      file.path(
        root,
        "benchmark_results/manuscript_revision_cycle16_20260725",
        "nmr_plssvd_extended_lower_grid/heldout/rows"
      ),
      pattern = "\\.csv$",
      full.names = TRUE
    ),
    "plssvd",
    "CUDA"
  ),
  summarize_nmr_rows(
    list.files(
      file.path(remote_root, "nmr_simpls_k50_cpu", "rows"),
      pattern = "\\.csv$",
      full.names = TRUE
    ),
    "simpls",
    "CPU"
  ),
  summarize_nmr_rows(
    list.files(
      file.path(
        root,
        "benchmark_results/manuscript_revision_cycle17_20260725",
        "nmr_simpls_one_se/heldout/rows"
      ),
      pattern = "\\.csv$",
      full.names = TRUE
    ),
    "simpls",
    "CUDA"
  )
)
paired <- rbind(paired, nmr_rows)

chosen <- utils::read.csv(source_chosen, stringsAsFactors = FALSE)
selection <- unique(
  chosen[, c("dataset", "method_panel", "selection_status")]
)
paired <- merge(
  paired,
  selection,
  by = c("dataset", "method_panel"),
  all.x = TRUE
)
paired$engine <- ifelse(
  toupper(paired$engine) %in% c("GPU", "CUDA"),
  "CUDA",
  "CPU"
)

uncertainty <- vector("list", nrow(paired))
for (i in seq_len(nrow(paired))) {
  row <- paired[i, ]
  n <- unname(n_test[[row$dataset]])
  if (row$metric_name == "accuracy") {
    successes <- as.integer(round(row$metric_median * n))
    point <- successes / n
    interval <- wilson_interval(successes, n)
    method <- "Wilson score interval"
    reps <- NA_integer_
  } else {
    path <- prediction_path(
      row$dataset,
      row$variant_name,
      as.integer(row$effective_ncomp)
    )
    prediction <- read_prediction(path)
    point <- prediction$point
    interval <- bootstrap_rmsd(prediction$row_mse, bootstrap_reps)
    method <- "held-out-sample percentile bootstrap"
    reps <- bootstrap_reps
    n <- prediction$n_test
    rm(prediction)
    gc()
  }
  uncertainty[[i]] <- data.frame(
    dataset = row$dataset,
    method_panel = row$method_panel,
    engine = row$engine,
    point_estimate = point,
    ci_lower = interval[[1L]],
    ci_upper = interval[[2L]],
    ci_method = method,
    n_test = as.integer(n),
    bootstrap_reps = reps,
    stringsAsFactors = FALSE
  )
}
uncertainty <- do.call(rbind, uncertainty)
paired <- merge(
  paired,
  uncertainty,
  by = c("dataset", "method_panel", "engine"),
  all.x = TRUE
)

not_evaluated <- paired[rep(1L, 4L), , drop = FALSE]
not_evaluated[,] <- NA
not_evaluated$dataset <- "nmr"
not_evaluated$method_panel <- rep(c("opls", "kernelpls"), each = 2L)
not_evaluated$engine <- rep(c("CPU", "CUDA"), 2L)
not_evaluated$status <- "not evaluated in NMR protocol"
not_evaluated$selection_status <- "not evaluated"
paired <- rbind(
  paired,
  not_evaluated
)

paired$dataset <- factor(
  paired$dataset,
  levels = dataset_order,
  ordered = TRUE
)
paired$method_panel <- factor(
  paired$method_panel,
  levels = method_order,
  ordered = TRUE
)
paired$engine <- factor(
  paired$engine,
  levels = c("CPU", "CUDA"),
  ordered = TRUE
)
paired <- paired[order(
  paired$dataset,
  paired$method_panel,
  paired$engine
), ]

utils::write.csv(
  paired,
  file.path(out_dir, "paired_backend_selected_summary.csv"),
  row.names = FALSE
)

agreement <- paired[
  paired$status == "ok",
  c(
    "dataset", "method_panel", "engine", "effective_ncomp",
    "point_estimate", "ci_lower", "ci_upper",
    "total_time_sec_median", "host_rss_mb_median",
    "gpu_mem_mb_median", "n_runs", "status"
  )
]
wide <- reshape(
  agreement,
  idvar = c("dataset", "method_panel", "effective_ncomp"),
  timevar = "engine",
  direction = "wide"
)
wide$metric_difference_cuda_minus_cpu <-
  wide$point_estimate.CUDA - wide$point_estimate.CPU
wide$time_speedup_cpu_over_cuda <-
  wide$total_time_sec_median.CPU / wide$total_time_sec_median.CUDA
wide$host_rss_ratio_cuda_over_cpu <-
  wide$host_rss_mb_median.CUDA / wide$host_rss_mb_median.CPU
utils::write.csv(
  wide,
  file.path(out_dir, "paired_backend_selected_wide.csv"),
  row.names = FALSE
)

cat(file.path(out_dir, "paired_backend_selected_summary.csv"), "\n")
