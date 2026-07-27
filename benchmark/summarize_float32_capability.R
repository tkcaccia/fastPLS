#!/usr/bin/env Rscript

args <- commandArgs(trailingOnly = TRUE)
out_dir <- if (length(args)) {
  args[[1L]]
} else {
  file.path(
    "benchmark_results", "manuscript_revision_cycle13_20260725"
  )
}
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

capability_file <- file.path(out_dir, "float32_capability_matrix.csv")
paired_file <- file.path(out_dir, "float32_float64_paired_resources.csv")
agreement_file <- file.path(out_dir, "float32_float64_controlled_agreement.csv")

families <- data.frame(
  family = c("PLS-SVD", "SIMPLS", "OPLS", "kernel PLS", "kernel PLS"),
  method = c("plssvd", "simpls", "opls", "kernelpls", "kernelpls"),
  kernel_scope = c("not applicable", "not applicable", "not applicable", "linear", "nonlinear"),
  stringsAsFactors = FALSE
)
backends <- c("CPU", "CUDA", "Metal")
endpoints <- c("regression", "classification: argmax", "classification: LDA")

capability_row <- function(family, method, kernel_scope, backend, endpoint) {
  status <- "validated"
  execution <- switch(
    backend,
    CPU = "compiled CPU",
    CUDA = "device-accelerated",
    Metal = "device-accelerated with host-assisted reduced decomposition"
  )
  evidence <- "small controlled regression/classification plus matched real-data tests"
  limitation <- "float32 input storage is smaller, but runtime and peak memory are route dependent"
  api <- "allow"

  if (backend == "CPU") {
    solver <- "rSVD validated; IRLBA experimental"
    platform <- paste(
      "Unix-like compiled route; Windows has an experimental portable",
      "rSVD fallback only for PLS-SVD, SIMPLS, and linear kernel PLS with argmax"
    )
  } else if (backend == "CUDA") {
    solver <- "rSVD only"
    platform <- "CUDA-enabled build and NVIDIA GPU required; Windows and Linux supported when compiled"
  } else {
    solver <- "rSVD and IRLBA experimental"
    platform <- "macOS with Apple Metal required"
    status <- "experimental"
    evidence <- "small controlled tests and selected MetRef/CIFAR-100 portability tests"
    api <- "warn"
  }

  if (endpoint != "regression" &&
      method %in% c("simpls", "kernelpls") &&
      kernel_scope != "nonlinear") {
    status <- if (backend == "Metal") "experimental" else "experimental"
    limitation <- paste(
      "matched classification tests showed route-dependent float64/float32",
      "accuracy differences; held-out agreement must be checked"
    )
    api <- "warn"
  }

  if (endpoint == "classification: LDA" && backend == "Metal") {
    status <- "hybrid"
    execution <- "Metal PLS scores plus compiled CPU float32 LDA"
    limitation <- "the LDA head is not Metal-native"
    api <- "warn"
  }

  if (method == "opls" && backend %in% c("CUDA", "Metal")) {
    status <- "hybrid"
    execution <- paste(backend, "inner PLS with host-resident OPLS orchestration")
    limitation <- "orthogonal filtering and reduced decomposition are not fully device-resident"
    api <- "warn"
  }

  if (kernel_scope == "nonlinear") {
    status <- if (backend == "CPU") "experimental" else "hybrid"
    execution <- if (backend == "CPU") {
      "compiled CPU with explicit Gram matrix"
    } else {
      paste(backend, "kernel products with host-resident Gram centering/orchestration")
    }
    evidence <- "limited kernel-specific float32 tests"
    limitation <- "materializes an n-by-n Gram matrix; no general memory advantage is established"
    api <- "warn"
  }

  if (endpoint == "classification: LDA" && .Platform$OS.type == "windows") {
    # This does not change the cross-platform status; the constraint is recorded
    # explicitly below so the authoritative table remains platform readable.
    platform <- paste(platform, "Float32 LDA is unavailable on Windows.")
  }

  data.frame(
    family = family,
    kernel_scope = kernel_scope,
    backend = backend,
    endpoint = endpoint,
    status = status,
    execution_residency = execution,
    supported_solver = solver,
    validation_evidence = evidence,
    observed_limitation = limitation,
    public_api_behavior = api,
    platform_constraints = platform,
    windows_status = if (backend != "CPU") {
      "unavailable"
    } else if (method == "opls" ||
               (method == "kernelpls" && kernel_scope == "nonlinear") ||
               endpoint == "classification: LDA") {
      "unavailable"
    } else {
      "experimental"
    },
    windows_behavior = if (backend != "CPU" ||
                           method == "opls" ||
                           (method == "kernelpls" && kernel_scope == "nonlinear") ||
                           endpoint == "classification: LDA") {
      "error before allocation"
    } else {
      "portable float-package CPU rSVD; warn"
    },
    extreme_response_status = if (endpoint != "regression") {
      "not applicable"
    } else if (method == "plssvd") {
      "experimental"
    } else {
      "failed"
    },
    extreme_response_policy = if (method == "plssvd") {
      "q >= 10000 and ncomp >= 50: warn (measured performance/memory risk)"
    } else {
      "q >= 10000 and ncomp >= 50: warn as a measured failed/unsafe regime"
    },
    stringsAsFactors = FALSE
  )
}

capability <- do.call(
  rbind,
  lapply(seq_len(nrow(families)), function(i) {
    do.call(
      rbind,
      lapply(backends, function(backend) {
        do.call(
          rbind,
          lapply(endpoints, function(endpoint) {
            capability_row(
              families$family[[i]],
              families$method[[i]],
              families$kernel_scope[[i]],
              backend,
              endpoint
            )
          })
        )
      })
    )
  })
)
write.csv(capability, capability_file, row.names = FALSE, na = "")

evidence_root <- Sys.getenv(
  "FASTPLS_PRECISION_EVIDENCE_ROOT",
  unset = file.path(
    dirname(normalizePath(".", mustWork = FALSE)),
    "reviewer_experiments", "precision_validation_20260724"
  )
)

raw_specs <- data.frame(
  dataset = rep(c("metref", "prism"), each = 2L),
  precision = rep(c("float64", "float32"), 2L),
  path = c(
    file.path(evidence_root, "metref_k20_replicated_local", "float64_raw.csv"),
    file.path(evidence_root, "metref_k20_replicated_local", "float32_raw.csv"),
    file.path(evidence_root, "prism_k5_replicated_local", "float64_raw.csv"),
    file.path(evidence_root, "prism_k5_replicated_local", "float32_raw.csv")
  ),
  stringsAsFactors = FALSE
)

median_or_na <- function(x) {
  x <- suppressWarnings(as.numeric(x))
  if (!length(x) || all(is.na(x))) return(NA_real_)
  stats::median(x, na.rm = TRUE)
}

if (all(file.exists(raw_specs$path))) {
  raw <- do.call(
    rbind,
    lapply(seq_len(nrow(raw_specs)), function(i) {
      x <- read.csv(raw_specs$path[[i]], stringsAsFactors = FALSE)
      x$precision <- raw_specs$precision[[i]]
      x$incremental_host_rss_mb <- x$peak_host_rss_mb - x$rss_before_fit_mb
      x
    })
  )
  group_key <- interaction(
    raw$dataset, raw$method_family, raw$backend, raw$classifier, raw$precision,
    drop = TRUE
  )
  summaries <- lapply(split(raw, group_key), function(x) {
    data.frame(
      dataset = x$dataset[[1L]],
      task_type = x$task_type[[1L]],
      family = x$method_family[[1L]],
      backend = if (x$backend[[1L]] == "cpu_rsvd") "CPU" else "CUDA",
      classifier = x$classifier[[1L]],
      precision = x$precision[[1L]],
      n_runs = sum(x$status == "success", na.rm = TRUE),
      total_time_sec = median_or_na(x$total_time_ms) / 1000,
      metric_name = x$metric_name[[1L]],
      metric = median_or_na(x$metric_value),
      input_storage_mb = median_or_na(x$input_storage_mb),
      baseline_host_rss_mb = median_or_na(x$rss_before_fit_mb),
      peak_host_rss_mb = median_or_na(x$peak_host_rss_mb),
      incremental_host_rss_mb = median_or_na(x$incremental_host_rss_mb),
      sampled_peak_gpu_used_mb = median_or_na(x$peak_gpu_mem_mb),
      gpu_baseline_mb = NA_real_,
      incremental_gpu_workspace_mb = NA_real_,
      stringsAsFactors = FALSE
    )
  })
  summaries <- do.call(rbind, summaries)
  id <- c("dataset", "task_type", "family", "backend", "classifier", "metric_name")
  f32 <- summaries[summaries$precision == "float32", ]
  f64 <- summaries[summaries$precision == "float64", ]
  names(f32)[!names(f32) %in% id] <- paste0(
    names(f32)[!names(f32) %in% id], "_float32"
  )
  names(f64)[!names(f64) %in% id] <- paste0(
    names(f64)[!names(f64) %in% id], "_float64"
  )
  paired <- merge(f32, f64, by = id, all = TRUE, sort = TRUE)
  paired$metric_delta_float32_minus_float64 <-
    paired$metric_float32 - paired$metric_float64
  paired$time_ratio_float32_over_float64 <-
    paired$total_time_sec_float32 / paired$total_time_sec_float64
  paired$input_storage_saved_mb <-
    paired$input_storage_mb_float64 - paired$input_storage_mb_float32
  paired$incremental_host_rss_delta_mb <-
    paired$incremental_host_rss_mb_float32 -
    paired$incremental_host_rss_mb_float64
  paired$sampled_peak_gpu_delta_mb <-
    paired$sampled_peak_gpu_used_mb_float32 -
    paired$sampled_peak_gpu_used_mb_float64
  write.csv(paired, paired_file, row.names = FALSE, na = "")
} else {
  warning("Raw paired precision files were not found; paired resource table was not regenerated.")
}

agreement_dir <- file.path(
  "benchmark_results", "float32_backend_agreement_cycle5"
)
agreement_paths <- c(
  file.path(agreement_dir, "float32_backend_agreement_cpu_20260724_175303.csv"),
  file.path(agreement_dir, "float32_backend_agreement_cuda_20260724_175723.csv"),
  file.path(agreement_dir, "float32_backend_agreement_metal_20260724_175557.csv")
)
if (all(file.exists(agreement_paths))) {
  agreement <- do.call(
    rbind,
    lapply(agreement_paths, read.csv, stringsAsFactors = FALSE)
  )
  write.csv(agreement, agreement_file, row.names = FALSE, na = "")
}

message("Wrote ", normalizePath(capability_file, mustWork = FALSE))
if (file.exists(paired_file)) {
  message("Wrote ", normalizePath(paired_file, mustWork = FALSE))
}
if (file.exists(agreement_file)) {
  message("Wrote ", normalizePath(agreement_file, mustWork = FALSE))
}
