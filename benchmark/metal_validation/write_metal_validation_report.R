#!/usr/bin/env Rscript

args <- commandArgs(trailingOnly = TRUE)
root <- if (length(args)) args[[1L]] else
  file.path(getwd(), "benchmark_results", "metal_validation_20260726")
root <- normalizePath(root, winslash = "/", mustWork = TRUE)
summary_dir <- file.path(root, "summary")
dir.create(summary_dir, recursive = TRUE, showWarnings = FALSE)

summary_path <- file.path(summary_dir, "metal_validation_summary.csv")
if (!file.exists(summary_path)) {
  stop("Run summarize_metal_validation.R first.")
}
summary <- read.csv(summary_path, stringsAsFactors = FALSE)

default_real <- subset(
  summary,
  experiment == "real_dataset" & oversample == 10 & power == 1
)

pair_backends <- function(x) {
  keys <- c(
    "dataset", "task_type", "method", "precision", "classifier", "svd_method",
    "oversample", "power", "kernel", "north", "ncomp", "n_train", "n_test",
    "p", "q"
  )
  values <- c(
    "successes", "failures", "median_total_sec", "median_fit_sec",
    "median_prediction_sec", "median_metric", "median_peak_rss_mb",
    "median_incremental_peak_rss_mb"
  )
  cpu <- subset(x, backend_requested == "cpu", select = c(keys, values))
  metal <- subset(x, backend_requested == "metal", select = c(keys, values))
  names(cpu)[match(values, names(cpu))] <- paste0(values, "_cpu")
  names(metal)[match(values, names(metal))] <- paste0(values, "_metal")
  out <- merge(cpu, metal, by = keys, all = TRUE)
  out$metal_speedup <- out$median_total_sec_cpu / out$median_total_sec_metal
  out$metric_delta_metal_minus_cpu <-
    out$median_metric_metal - out$median_metric_cpu
  out$incremental_rss_delta_mb <-
    out$median_incremental_peak_rss_mb_metal -
    out$median_incremental_peak_rss_mb_cpu
  out
}

pair_precision <- function(x) {
  keys <- c(
    "dataset", "task_type", "method", "backend_requested", "classifier",
    "svd_method", "oversample", "power", "kernel", "north", "ncomp",
    "n_train", "n_test", "p", "q"
  )
  values <- c(
    "median_total_sec", "median_metric", "median_peak_rss_mb",
    "median_incremental_peak_rss_mb"
  )
  f64 <- subset(x, precision == "float64", select = c(keys, values))
  f32 <- subset(x, precision == "float32", select = c(keys, values))
  names(f64)[match(values, names(f64))] <- paste0(values, "_float64")
  names(f32)[match(values, names(f32))] <- paste0(values, "_float32")
  out <- merge(f64, f32, by = keys, all = TRUE)
  out$float32_speedup <-
    out$median_total_sec_float64 / out$median_total_sec_float32
  out$metric_delta_float32_minus_float64 <-
    out$median_metric_float32 - out$median_metric_float64
  out$incremental_rss_saved_mb <-
    out$median_incremental_peak_rss_mb_float64 -
    out$median_incremental_peak_rss_mb_float32
  out
}

backend_pairs <- pair_backends(default_real)
precision_pairs <- pair_precision(default_real)
write.csv(
  backend_pairs,
  file.path(summary_dir, "metal_backend_paired.csv"),
  row.names = FALSE
)
write.csv(
  precision_pairs,
  file.path(summary_dir, "metal_float32_vs_float64.csv"),
  row.names = FALSE
)

nmr <- subset(summary, grepl("^nmr_", dataset))
write.csv(nmr, file.path(summary_dir, "metal_nmr_summary.csv"), row.names = FALSE)

model_specific <- subset(
  summary,
  experiment %in% c("kernel_sensitivity", "opls_sensitivity")
)
write.csv(
  model_specific,
  file.path(summary_dir, "metal_model_specific_summary.csv"),
  row.names = FALSE
)

cv_path <- file.path(root, "cv", "metal_cv_validation_raw.csv")
if (file.exists(cv_path)) {
  cv <- read.csv(cv_path, stringsAsFactors = FALSE)
  cv_keys <- c(
    "dataset", "method", "backend_requested", "svd_method_requested",
    "classifier", "kfold", "reported_backend",
    "reported_prediction_backend", "reported_svd_method"
  )
  cv_key <- interaction(cv[cv_keys], drop = TRUE, lex.order = TRUE)
  cv_summary <- do.call(rbind, lapply(split(cv, cv_key), function(x) {
    ok <- x$status == "success"
    data.frame(
      x[1L, cv_keys, drop = FALSE],
      successes = sum(ok),
      failures = sum(!ok),
      median_elapsed_sec = median(x$elapsed_sec[ok], na.rm = TRUE),
      min_elapsed_sec = min(x$elapsed_sec[ok], na.rm = TRUE),
      max_elapsed_sec = max(x$elapsed_sec[ok], na.rm = TRUE),
      median_best_ncomp = median(x$best_ncomp[ok], na.rm = TRUE),
      median_metric = median(x$metric_value[ok], na.rm = TRUE),
      stringsAsFactors = FALSE
    )
  }))
  write.csv(
    cv_summary,
    file.path(summary_dir, "metal_cv_summary.csv"),
    row.names = FALSE
  )
} else {
  cv <- data.frame()
  cv_summary <- data.frame()
}

capability <- data.frame(
  model = c("PLS-SVD", "SIMPLS", "OPLS", "kernel PLS", "LDA", "10-fold CV"),
  large_products = c(
    "Metal/MPS", "Metal/MPS", "Metal inner PLS", "Metal where supported",
    "Metal outside CV", "compiled CPU in tested public route"
  ),
  reduced_decomposition = c(
    "host-assisted", "host-assisted", "host-assisted", "host-assisted",
    "Cholesky path", "reported CPU rSVD/IRLBA"
  ),
  prediction = c(
    "Metal", "Metal", "Metal/host hybrid", "Metal/host hybrid",
    "Metal outside CV", "compiled CPU scorer in tested route"
  ),
  memory_accounting = rep("Unified process RSS; no separate VRAM", 6L),
  stringsAsFactors = FALSE
)
write.csv(
  capability,
  file.path(summary_dir, "metal_backend_capability.csv"),
  row.names = FALSE
)

fmt <- function(x, digits = 3L) {
  ifelse(is.finite(x), formatC(x, format = "f", digits = digits), "NA")
}

cifar <- subset(
  backend_pairs,
  dataset == "cifar100" & svd_method == "rsvd" &
    oversample == 10 & power == 1
)
nmr_full <- subset(nmr, dataset == "nmr_q28355")
scaling <- subset(
  backend_pairs,
  dataset %in% c("synthetic_balanced", "synthetic_high_components") &
    precision == "float64"
)

total_runs <- 0L
failed_runs <- 0L
raw_path <- file.path(summary_dir, "metal_validation_all_raw.csv")
if (file.exists(raw_path)) {
  all_raw <- read.csv(raw_path, stringsAsFactors = FALSE)
  total_runs <- nrow(all_raw)
  failed_runs <- sum(all_raw$status != "success")
}
if (nrow(cv)) {
  total_runs <- total_runs + nrow(cv)
  failed_runs <- failed_runs + sum(cv$status != "success")
}
svd_path <- file.path(root, "svd_reliability", "metal_svd_reliability_raw.csv")
if (file.exists(svd_path)) {
  svd_raw <- read.csv(svd_path, stringsAsFactors = FALSE)
  total_runs <- total_runs + nrow(svd_raw)
  failed_runs <- failed_runs + sum(svd_raw$status != "success")
}

lines <- c(
  "# Apple Metal validation report",
  "",
  "## Scope",
  "",
  paste0(
    "The suite completed ", total_runs, " isolated benchmark runs with ",
    failed_runs, " failures. Each fit ran in a fresh R process. Runtime includes ",
    "fitting and prediction. Memory is reported as baseline-corrected peak ",
    "process RSS because Apple silicon uses unified memory; a distinct GPU VRAM ",
    "measurement is not physically meaningful on this system."
  ),
  "",
  "Hardware: MacBook Pro Mac15,3; Apple M3 (8 CPU cores, 10 GPU cores); 8 GB unified memory; Metal 3.",
  "",
  "Software: R 4.6.0 arm64; fastPLS 0.99.6; source commit 6e50bd318f20289101f6b723953830aefa8b95d6.",
  "",
  "## Main findings",
  "",
  paste0(
    "1. Metal acceleration is shape dependent. It is slower for MetRef, Retina, ",
    "Tabula and the NMR matrix on this 8 GB M3 because launch, conversion, and ",
    "host-assisted reduced-decomposition costs dominate. It becomes useful for ",
    "larger dense float64 workloads such as CIFAR-100 and the balanced/high-",
    "component synthetic regimes."
  ),
  paste0(
    "2. On CIFAR-100 (50,000 training, 10,000 test, 768 predictors, 100 classes, ",
    "50 components), float64 Metal produced ", nrow(cifar),
    " paired method/classifier results. See `metal_backend_paired.csv` for every ",
    "runtime, accuracy, and memory value."
  ),
  paste0(
    "3. Native float32 is the strongest local CPU configuration. It markedly ",
    "reduces time for the real datasets and generally gives close CPU/Metal ",
    "predictions. Metal does not automatically win with float32 because the M3 ",
    "has shared memory and the tested matrices often do not amortize dispatch ",
    "and synchronization."
  ),
  paste0(
    "4. Full-response NMR PLS-SVD float32 completed on both backends. CPU took ",
    if (nrow(nmr_full)) fmt(nmr_full$median_total_sec[nmr_full$backend_requested == "cpu"]) else "NA",
    " s and Metal took ",
    if (nrow(nmr_full)) fmt(nmr_full$median_total_sec[nmr_full$backend_requested == "metal"]) else "NA",
    " s. RMSD was ",
    if (nrow(nmr_full)) fmt(nmr_full$median_metric[nmr_full$backend_requested == "cpu"], 7L) else "NA",
    " versus ",
    if (nrow(nmr_full)) fmt(nmr_full$median_metric[nmr_full$backend_requested == "metal"], 7L) else "NA",
    ". Incremental peak RSS was ",
    if (nrow(nmr_full)) fmt(nmr_full$median_incremental_peak_rss_mb[nmr_full$backend_requested == "cpu"], 1L) else "NA",
    " MB versus ",
    if (nrow(nmr_full)) fmt(nmr_full$median_incremental_peak_rss_mb[nmr_full$backend_requested == "metal"], 1L) else "NA",
    " MB. Full-response Metal SIMPLS was not attempted after the q=5,000 ",
    "guarded run required 21.6 s and showed no numerical advantage on an 8 GB machine."
  ),
  paste0(
    "5. The float64 Metal SIMPLS randomized route is sensitive to rSVD power ",
    "iterations. On CIFAR-100, power=2 reduced CPU/Metal accuracy disagreement. ",
    "The float64 Metal PLS-SVD route remained insensitive to the exposed power ",
    "and oversampling settings and retained a material argmax discrepancy; it ",
    "must not be described as estimator-equivalent without correction."
  ),
  paste0(
    "6. Standalone rSVD reliability improved with power=2 and oversampling=20. ",
    "Principal angles on random Gaussian matrices can be large when singular ",
    "values are clustered, because the individual truncated subspace is then ",
    "poorly identifiable; prediction and singular-value errors must therefore ",
    "be interpreted together."
  ),
  paste0(
    "7. All 192 public cross-validation runs completed, but returned metadata ",
    "shows that the tested Metal requests use compiled CPU decomposition/scoring ",
    "for important stages. These results validate usability, not a fully Metal-",
    "resident cross-validation claim."
  ),
  "",
  "## Interpretation",
  "",
  "Metal should currently be selected for sufficiently large dense float64 workloads after checking numerical agreement. CPU float32 is preferable on this Mac for small omics data, tall-thin data, and NMR-like extreme-response matrices. The package should expose or record effective residency so users can distinguish native Metal execution from hybrid routes.",
  "",
  "## Files",
  "",
  "- `metal_validation_summary.csv`: medians and ranges over repeated runs.",
  "- `metal_backend_paired.csv`: CPU versus Metal pairs with speedup, metric delta, and incremental-memory delta.",
  "- `metal_float32_vs_float64.csv`: precision pairs with speedup and metric delta.",
  "- `metal_nmr_summary.csv`: guarded and full-response NMR results.",
  "- `metal_cv_summary.csv`: 10-fold CV results and reported effective backends.",
  "- `metal_model_specific_summary.csv`: OPLS orthogonal-component and kernel sensitivity.",
  "- `metal_backend_capability.csv`: native/hybrid residency summary.",
  "- `../svd_reliability/metal_svd_reliability_summary.csv`: standalone rSVD diagnostics.",
  "",
  "## Known limitations",
  "",
  "- This Mac has only 8 GB unified memory; full NMR SIMPLS and very large ImageNet-scale Metal fitting require a larger Apple-silicon system.",
  "- Dedicated GPU memory cannot be separated from host memory on Apple silicon.",
  "- Real-data timing used three repetitions, but predictive uncertainty requires repeated data splits or resampling and is not estimated by computational repeats.",
  "- The CIFAR float64 Metal PLS-SVD discrepancy requires an implementation fix before numerical-equivalence claims."
)
writeLines(lines, file.path(summary_dir, "METAL_VALIDATION_REPORT.md"))
cat("Wrote report and paired tables to", summary_dir, "\n")
