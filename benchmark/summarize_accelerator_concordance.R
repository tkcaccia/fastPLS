#!/usr/bin/env Rscript

root <- normalizePath(
  Sys.getenv("FASTPLS_REPO", unset = "."),
  winslash = "/",
  mustWork = TRUE
)
out_dir <- file.path(
  root,
  "benchmark_results",
  "manuscript_revision_cycle62_20260726"
)
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

metric_tolerance <- 0.005
prediction_tolerance <- 0.995

read_optional <- function(path) {
  if (!file.exists(path)) return(NULL)
  utils::read.csv(path, stringsAsFactors = FALSE, check.names = FALSE)
}

paired_cuda <- utils::read.csv(
  file.path(
    root,
    "benchmark_results",
    "manuscript_revision_cycle20_20260725",
    "paired_backend_selected_summary.csv"
  ),
  stringsAsFactors = FALSE,
  check.names = FALSE
)
paired_cuda <- paired_cuda[paired_cuda$dataset != "nmr", , drop = FALSE]

cuda_keys <- unique(paired_cuda[, c("dataset", "method_panel")])
cuda <- do.call(
  rbind,
  lapply(seq_len(nrow(cuda_keys)), function(i) {
    dataset <- cuda_keys$dataset[[i]]
    family <- cuda_keys$method_panel[[i]]
    cpu <- paired_cuda[
      paired_cuda$dataset == dataset &
        paired_cuda$method_panel == family &
        paired_cuda$engine == "CPU",
      ,
      drop = FALSE
    ]
    accelerator <- paired_cuda[
      paired_cuda$dataset == dataset &
        paired_cuda$method_panel == family &
        paired_cuda$engine == "CUDA",
      ,
      drop = FALSE
    ]
    if (nrow(cpu) != 1L || nrow(accelerator) != 1L) return(NULL)
    data.frame(
      accelerator = "CUDA",
      dataset = dataset,
      family = family,
      ncomp = accelerator$requested_ncomp,
      metric_name = accelerator$metric_name,
      metric_cpu = cpu$metric_median,
      metric_accelerator = accelerator$metric_median,
      metric_delta = accelerator$metric_median - cpu$metric_median,
      prediction_agreement = NA_real_,
      time_cpu_sec = cpu$total_time_sec_median,
      time_accelerator_sec = accelerator$total_time_sec_median,
      speedup = cpu$total_time_sec_median /
        accelerator$total_time_sec_median,
      host_rss_cpu_mb = cpu$host_rss_mb_median,
      host_rss_accelerator_mb = accelerator$host_rss_mb_median,
      accelerator_memory_mb = accelerator$gpu_mem_mb_median,
      accelerator_memory_scope = "absolute process GPU memory",
      stringsAsFactors = FALSE
    )
  })
)

cuda_agreement <- read_optional(file.path(
  out_dir,
  "cuda_prediction_agreement",
  "cuda_prediction_agreement.csv"
))
if (!is.null(cuda_agreement)) {
  cuda <- merge(
    cuda[, setdiff(names(cuda), "prediction_agreement"), drop = FALSE],
    cuda_agreement[, c(
      "dataset", "method_panel", "prediction_agreement"
    )],
    by.x = c("dataset", "family"),
    by.y = c("dataset", "method_panel"),
    all.x = TRUE,
    sort = FALSE
  )
}

metal <- utils::read.csv(
  file.path(
    root,
    "benchmark_results",
    "manuscript_revision_cycle57_20260726",
    "internal_metal_speedup.csv"
  ),
  stringsAsFactors = FALSE,
  check.names = FALSE
)
metal <- data.frame(
  accelerator = "Metal",
  dataset = metal$dataset,
  family = metal$method,
  ncomp = metal$ncomp,
  metric_name = ifelse(metal$task_type == "classification", "accuracy", "rmsd"),
  metric_cpu = metal$median_metric_cpu,
  metric_accelerator = metal$median_metric_metal,
  metric_delta = metal$metric_delta_metal_minus_cpu,
  prediction_agreement = NA_real_,
  time_cpu_sec = metal$median_total_sec_cpu,
  time_accelerator_sec = metal$median_total_sec_metal,
  speedup = metal$metal_speedup,
  host_rss_cpu_mb = metal$median_peak_rss_mb_cpu,
  host_rss_accelerator_mb = metal$median_peak_rss_mb_metal,
  accelerator_memory_mb = metal$median_incremental_peak_rss_mb_metal,
  accelerator_memory_scope = "incremental unified-process RSS",
  stringsAsFactors = FALSE
)

metal_agreement <- read_optional(file.path(
  out_dir,
  "metal_prediction_agreement",
  "metal_prediction_agreement.csv"
))
if (!is.null(metal_agreement)) {
  metal <- merge(
    metal[, setdiff(names(metal), "prediction_agreement"), drop = FALSE],
    metal_agreement[, c("dataset", "method", "prediction_agreement")],
    by.x = c("dataset", "family"),
    by.y = c("dataset", "method"),
    all.x = TRUE,
    sort = FALSE
  )
}

audit <- rbind(cuda, metal)
audit$family <- factor(
  audit$family,
  levels = c("plssvd", "simpls", "opls", "kernelpls")
)
audit <- audit[order(
  factor(audit$accelerator, levels = c("CUDA", "Metal")),
  audit$dataset,
  audit$family
), , drop = FALSE]
audit$family <- as.character(audit$family)
audit$metric_concordant <- abs(audit$metric_delta) <= metric_tolerance
audit$prediction_concordant <- !is.na(audit$prediction_agreement) &
  audit$prediction_agreement >= prediction_tolerance
audit$evidence_status <- ifelse(
  !audit$metric_concordant,
  "discordant_metric",
  ifelse(
    is.na(audit$prediction_agreement),
    "prediction_not_archived",
    ifelse(
      !audit$prediction_concordant,
      "discordant_prediction",
      "concordant"
    )
  )
)
audit$speed_eligible <- audit$evidence_status == "concordant"

utils::write.csv(
  audit,
  file.path(out_dir, "accelerator_paired_concordance_audit.csv"),
  row.names = FALSE
)

summary_rows <- do.call(
  rbind,
  lapply(split(audit, audit$accelerator), function(x) {
    eligible <- x$speed_eligible
    data.frame(
      accelerator = x$accelerator[[1L]],
      paired_routes = nrow(x),
      concordant_routes = sum(eligible),
      metric_discordant_routes = sum(!x$metric_concordant),
      prediction_discordant_routes = sum(
        x$metric_concordant &
          !is.na(x$prediction_agreement) &
          !x$prediction_concordant
      ),
      prediction_not_archived = sum(is.na(x$prediction_agreement)),
      accelerator_faster_concordant = sum(eligible & x$speedup > 1),
      median_speedup_concordant = if (any(eligible)) {
        stats::median(x$speedup[eligible])
      } else {
        NA_real_
      },
      maximum_speedup_concordant = if (any(eligible)) {
        max(x$speedup[eligible])
      } else {
        NA_real_
      },
      stringsAsFactors = FALSE
    )
  })
)
utils::write.csv(
  summary_rows,
  file.path(out_dir, "accelerator_concordance_summary.csv"),
  row.names = FALSE
)

print(summary_rows)
