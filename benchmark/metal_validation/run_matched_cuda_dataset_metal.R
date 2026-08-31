#!/usr/bin/env Rscript

# Repeats the selected-point CUDA dataset panel on Apple CPU and Metal. The
# benchmark uses fixed task objects, paired seeds, and identical output policy.

args <- commandArgs(trailingOnly = TRUE)
script_arg <- commandArgs()[grep("^--file=", commandArgs())]
script_path <- normalizePath(sub("^--file=", "", script_arg[[1L]]))
repo_dir <- normalizePath(file.path(dirname(script_path), "..", ".."))
worker <- file.path(repo_dir, "benchmark", "metal_validation", "metal_worker.R")
accelerator <- tolower(Sys.getenv("FASTPLS_MATCHED_ACCELERATOR", "metal"))
if (!accelerator %in% c("metal", "cuda")) {
  stop("FASTPLS_MATCHED_ACCELERATOR must be 'metal' or 'cuda'.", call. = FALSE)
}
task_root <- normalizePath(
  Sys.getenv(
    "FASTPLS_METAL_MATCHED_TASK_ROOT",
    "/Users/stefano/Documents/GPUPLS/Data/metal_matched"
  ),
  mustWork = TRUE
)
out_dir <- if (length(args)) args[[1L]] else file.path(
  repo_dir,
  "benchmark_results",
  paste0("metal_matched_cuda_datasets_", format(Sys.time(), "%Y%m%d_%H%M%S"))
)
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

selected <- data.frame(
  dataset = rep(c(
    "cbmc_citeseq", "ccle", "cifar100", "gtex_v8", "metref", "prism",
    "retina", "tabula", "tcga_brca", "tcga_hnsc_methylation",
    "tcga_pan_cancer"
  ), each = 4L),
  method = rep(c("plssvd", "simpls", "opls", "kernelpls"), 11L),
  ncomp = c(
    10, 50, 50, 50,
    17, 18, 18, 18,
    99, 200, 200, 200,
    31, 100, 100, 100,
    21, 100, 100, 100,
    10, 5, 2, 5,
    11, 50, 20, 50,
    31, 50, 50, 50,
    4, 10, 5, 10,
    1, 2, 2, 2,
    31, 100, 100, 100
  ),
  stringsAsFactors = FALSE
)

task_path <- function(dataset) {
  path <- file.path(task_root, paste0(dataset, "_task.rds"))
  if (!file.exists(path)) stop("Missing task: ", path, call. = FALSE)
  normalizePath(path)
}

make_config <- function(dataset, method, ncomp, backend, replicate) {
  task <- readRDS(task_path(dataset))
  classification <- is.factor(task$Ytrain) || is.character(task$Ytrain)
  list(
    run_id = paste(
      "metal_matched", dataset, method, backend, paste0("r", replicate),
      sep = "__"
    ),
    experiment = "metal_matched_cuda_dataset_panel",
    dataset = dataset,
    task_type = if (classification) "classification" else "regression",
    method = method,
    backend = backend,
    precision = "float64",
    classifier = "argmax",
    ncomp = as.integer(ncomp),
    replicate = as.integer(replicate),
    seed = as.integer(1000L + replicate),
    data_seed = 777L,
    oversample = 20L,
    power = 2L,
    svd_method = "rsvd",
    task_path = task_path(dataset),
    n_train = nrow(task$Xtrain),
    n_test = nrow(task$Xtest),
    p = ncol(task$Xtrain),
    q = if (classification) nlevels(factor(task$Ytrain)) else ncol(task$Ytrain),
    latent_rank = NA_integer_,
    noise = NA_real_,
    train_n = NULL,
    test_n = NULL,
    p_limit = NULL,
    q_limit = NULL,
    kernel = "linear",
    gamma = NULL,
    degree = 3L,
    coef0 = 1,
    north = 1L,
    scaling = "centering",
    save_diagnostics = FALSE,
    save_prediction = TRUE
  )
}

configs <- list()
for (i in seq_len(nrow(selected))) {
  for (backend in c("cpu", accelerator)) {
    for (replicate in 1:3) {
      configs[[length(configs) + 1L]] <- make_config(
        selected$dataset[[i]], selected$method[[i]], selected$ncomp[[i]],
        backend, replicate
      )
    }
  }
}
saveRDS(configs, file.path(out_dir, "configurations.rds"))

parse_peak_rss <- function(path) {
  if (!file.exists(path)) return(NA_real_)
  hit <- grep("maximum resident set size", readLines(path, warn = FALSE), value = TRUE)
  if (!length(hit)) return(NA_real_)
  bytes <- suppressWarnings(as.numeric(gsub("[^0-9]", "", tail(hit, 1L))))
  if (is.finite(bytes)) bytes / 1024^2 else NA_real_
}

rows <- list()
time_flag <- if (identical(Sys.info()[["sysname"]], "Darwin")) "-l" else "-v"
for (i in seq_along(configs)) {
  cfg <- configs[[i]]
  cfg_path <- file.path(out_dir, paste0(cfg$run_id, "_config.rds"))
  result_path <- file.path(out_dir, paste0(cfg$run_id, "_result.rds"))
  stdout_path <- file.path(out_dir, paste0(cfg$run_id, ".out"))
  time_path <- file.path(out_dir, paste0(cfg$run_id, ".time"))
  if (file.exists(result_path)) {
    existing <- readRDS(result_path)
    if (identical(existing$status[[1L]], "success")) {
      existing$peak_rss_mb <- parse_peak_rss(time_path)
      existing$incremental_peak_rss_mb <- if (
        is.finite(existing$peak_rss_mb) && is.finite(existing$baseline_rss_mb)
      ) max(0, existing$peak_rss_mb - existing$baseline_rss_mb) else NA_real_
      rows[[length(rows) + 1L]] <- existing
      cat(sprintf("[%d/%d] %s [reused]\n", i, length(configs), cfg$run_id))
      next
    }
  }
  saveRDS(cfg, cfg_path)
  cat(sprintf("[%d/%d] %s\n", i, length(configs), cfg$run_id))
  status <- system2(
    "/usr/bin/time",
    c(time_flag, file.path(R.home("bin"), "Rscript"), worker, cfg_path, result_path),
    stdout = stdout_path,
    stderr = time_path
  )
  if (file.exists(result_path)) {
    row <- readRDS(result_path)
  } else {
    row <- data.frame(
      run_id = cfg$run_id, experiment = cfg$experiment, dataset = cfg$dataset,
      task_type = cfg$task_type, method = cfg$method,
      backend_requested = cfg$backend, backend_reported = NA_character_,
      prediction_backend = NA_character_, svd_method = cfg$svd_method,
      classifier = cfg$classifier, precision = cfg$precision,
      ncomp = cfg$ncomp, n_train = cfg$n_train, n_test = cfg$n_test,
      p = cfg$p, q = cfg$q, seed = cfg$seed, replicate = cfg$replicate,
      oversample = cfg$oversample, power = cfg$power, kernel = cfg$kernel,
      north = cfg$north, fit_sec = NA_real_, prediction_sec = NA_real_,
      total_sec = NA_real_, baseline_rss_mb = NA_real_,
      rss_after_fit_mb = NA_real_, rss_after_prediction_mb = NA_real_,
      peak_rss_mb = NA_real_, incremental_peak_rss_mb = NA_real_,
      metric_name = NA_character_, metric_value = NA_real_, accuracy = NA_real_,
      q2 = NA_real_, rmsd = NA_real_, prediction_checksum = NA_real_,
      prediction_length = NA_integer_, status = "process_failed", warnings = "",
      error = paste("exit status", status), stringsAsFactors = FALSE
    )
  }
  row$peak_rss_mb <- parse_peak_rss(time_path)
  row$incremental_peak_rss_mb <- if (
    is.finite(row$peak_rss_mb) && is.finite(row$baseline_rss_mb)
  ) max(0, row$peak_rss_mb - row$baseline_rss_mb) else NA_real_
  rows[[length(rows) + 1L]] <- row
  write.csv(do.call(rbind, rows), file.path(out_dir, "matched_metal_raw.csv"),
            row.names = FALSE)
  unlink(cfg_path)
}

raw <- do.call(rbind, rows)
keys <- unique(raw[c("dataset", "method", "backend_requested")])
summaries <- lapply(seq_len(nrow(keys)), function(i) {
  key <- keys[i, ]
  x <- raw[
    raw$dataset == key$dataset & raw$method == key$method &
      raw$backend_requested == key$backend_requested,
    , drop = FALSE
  ]
  ok <- x$status == "success"
  data.frame(
    dataset = key$dataset,
    method = key$method,
    backend = key$backend_requested,
    ncomp = x$ncomp[[1L]],
    n_ok = sum(ok),
    n_failed = sum(!ok),
    median_total_sec = if (any(ok)) median(x$total_sec[ok]) else NA_real_,
    iqr_total_sec = if (any(ok)) IQR(x$total_sec[ok]) else NA_real_,
    median_metric = if (any(ok)) median(x$metric_value[ok]) else NA_real_,
    median_peak_rss_mb = if (any(ok)) median(x$peak_rss_mb[ok]) else NA_real_,
    median_incremental_rss_mb = if (any(ok)) {
      median(x$incremental_peak_rss_mb[ok])
    } else NA_real_,
    error = paste(unique(x$error[!ok & nzchar(x$error)]), collapse = " | "),
    stringsAsFactors = FALSE
  )
})
summary <- do.call(rbind, summaries)
write.csv(summary, file.path(out_dir, "matched_metal_summary.csv"), row.names = FALSE)

prediction_agreement <- function(cpu, metal) {
  a <- as.vector(cpu$prediction)
  b <- as.vector(metal$prediction)
  if (length(a) != length(b)) return(NA_real_)
  if (is.character(a) || is.factor(a) || is.character(b) || is.factor(b)) {
    return(mean(as.character(a) == as.character(b), na.rm = TRUE))
  }
  a <- as.numeric(a)
  b <- as.numeric(b)
  denominator <- sqrt(sum(a^2))
  if (!is.finite(denominator) || denominator == 0) return(NA_real_)
  1 - sqrt(sum((a - b)^2)) / denominator
}

pairs <- lapply(seq_len(nrow(selected)), function(i) {
  dataset <- selected$dataset[[i]]
  method <- selected$method[[i]]
  cpu <- summary[summary$dataset == dataset & summary$method == method &
                   summary$backend == "cpu", , drop = FALSE]
  accelerated <- summary[
    summary$dataset == dataset & summary$method == method &
      summary$backend == accelerator,
    , drop = FALSE
  ]
  agreements <- vapply(1:3, function(replicate) {
    prefix <- paste("metal_matched", dataset, method, sep = "__")
    cpu_path <- file.path(out_dir, paste0(
      prefix, "__cpu__r", replicate, "_result_diagnostic.rds"
    ))
    accelerator_path <- file.path(out_dir, paste0(
      prefix, "__", accelerator, "__r", replicate, "_result_diagnostic.rds"
    ))
    if (!file.exists(cpu_path) || !file.exists(accelerator_path)) return(NA_real_)
    prediction_agreement(readRDS(cpu_path), readRDS(accelerator_path))
  }, numeric(1))
  data.frame(
    accelerator = accelerator,
    dataset = dataset,
    method = method,
    ncomp = selected$ncomp[[i]],
    cpu_total_sec = cpu$median_total_sec,
    accelerator_total_sec = accelerated$median_total_sec,
    cpu_accelerator_ratio = cpu$median_total_sec / accelerated$median_total_sec,
    metric_cpu = cpu$median_metric,
    metric_accelerator = accelerated$median_metric,
    metric_delta = accelerated$median_metric - cpu$median_metric,
    prediction_agreement = median(agreements, na.rm = TRUE),
    cpu_peak_rss_mb = cpu$median_peak_rss_mb,
    accelerator_peak_rss_mb = accelerated$median_peak_rss_mb,
    cpu_incremental_rss_mb = cpu$median_incremental_rss_mb,
    accelerator_incremental_rss_mb = accelerated$median_incremental_rss_mb,
    cpu_ok = cpu$n_ok,
    accelerator_ok = accelerated$n_ok,
    stringsAsFactors = FALSE
  )
})
paired <- do.call(rbind, pairs)
write.csv(
  paired,
  file.path(out_dir, paste0("matched_", accelerator, "_paired.csv")),
  row.names = FALSE
)

writeLines(c(
  paste("created:", format(Sys.time(), tz = "UTC", usetz = TRUE)),
  paste("repo_commit:", system2("git", c("-C", repo_dir, "rev-parse", "HEAD"),
                                stdout = TRUE)),
  paste("fastPLS:", as.character(packageVersion("fastPLS"))),
  paste("accelerator:", accelerator),
  paste("accelerator_available:", if (accelerator == "metal") {
    fastPLS::has_metal()
  } else fastPLS::has_cuda()),
  paste("task_root:", task_root),
  capture.output(sessionInfo())
), file.path(out_dir, "session_info.txt"))

cat("Results:", normalizePath(out_dir), "\n")
