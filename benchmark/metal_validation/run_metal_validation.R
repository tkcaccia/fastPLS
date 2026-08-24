#!/usr/bin/env Rscript

args <- commandArgs(trailingOnly = TRUE)
stage <- if (length(args)) args[[1L]] else "smoke"

repo_dir <- normalizePath(
  file.path(dirname(sub("^--file=", "", commandArgs()[grep("^--file=", commandArgs())])),
            "..", ".."),
  winslash = "/",
  mustWork = TRUE
)
worker <- file.path(repo_dir, "benchmark", "metal_validation", "metal_worker.R")
timestamp <- format(Sys.time(), "%Y%m%d_%H%M%S")
out_dir <- Sys.getenv(
  "FASTPLS_METAL_VALIDATION_OUT",
  file.path(repo_dir, "benchmark_results", paste0("metal_validation_", timestamp))
)
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

log_msg <- function(...) {
  cat("[", format(Sys.time(), "%Y-%m-%d %H:%M:%S"), "] ",
      paste(..., collapse = ""), "\n", sep = "")
  flush.console()
}

cfg <- function(experiment, dataset, task_type, method, backend,
                precision = "float64", classifier = "argmax", ncomp = 10L,
                replicate = 1L, seed = 123L, oversample = 10L, power = 2L,
                svd_method = "rsvd",
                task_path = NULL, n_train = 600L, n_test = 200L,
                p = 120L, q = 8L, latent_rank = 8L, noise = 0.25,
                train_n = NULL, test_n = NULL, p_limit = NULL, q_limit = NULL,
                kernel = "linear", gamma = NULL, degree = 3L, coef0 = 1,
                north = 1L, scaling = "centering", save_diagnostics = FALSE) {
  list(
    run_id = paste(
      experiment, dataset, method, backend, precision, classifier,
      paste0("k", ncomp), paste0("kernel-", kernel),
      paste0("north-", north), paste0("r", replicate), paste0("s", seed),
      sep = "__"
    ),
    experiment = experiment,
    dataset = dataset,
    task_type = task_type,
    method = method,
    backend = backend,
    precision = precision,
    classifier = classifier,
    ncomp = as.integer(ncomp),
    replicate = as.integer(replicate),
    seed = as.integer(seed),
    data_seed = 777L,
    oversample = as.integer(oversample),
    power = as.integer(power),
    svd_method = svd_method,
    task_path = task_path,
    n_train = as.integer(n_train),
    n_test = as.integer(n_test),
    p = as.integer(p),
    q = as.integer(q),
    latent_rank = as.integer(latent_rank),
    noise = noise,
    train_n = train_n,
    test_n = test_n,
    p_limit = p_limit,
    q_limit = q_limit,
    kernel = kernel,
    gamma = gamma,
    degree = as.integer(degree),
    coef0 = coef0,
    north = as.integer(north),
    scaling = scaling,
    save_diagnostics = save_diagnostics
  )
}

find_task <- function(...) {
  candidates <- c(...)
  found <- candidates[file.exists(candidates)]
  if (length(found)) normalizePath(found[[1L]], winslash = "/") else NULL
}

task_paths <- list(
  metref = find_task(
    file.path(repo_dir, "benchmark_results", "manuscript_revision_cycle13_20260725",
              "kernel_suite", "pipeline1", "real_datasets", "metref_task.rds")
  ),
  retina = find_task(
    file.path(repo_dir, "benchmark_results", "manuscript_revision_cycle13_20260725",
              "retina_tabula_selected_outer", "runs", "retina_simpls",
              "retina_task.rds")
  ),
  tabula = find_task(
    file.path(repo_dir, "benchmark_results", "manuscript_revision_cycle13_20260725",
              "retina_tabula_selected_outer", "runs", "tabula_simpls",
              "tabula_task.rds")
  ),
  cifar100 = find_task(
    "/Users/stefano/Documents/GPUPLS/fastPLS_publication_from_chiamaka_20260722/tasks/cifar100_task.rds",
    "/Users/stefano/Documents/GPUPLS/fastPLS_publication_from_chiamaka_20260722/tasks/cifar100_task_float64.rds"
  ),
  nmr = find_task(
    file.path(repo_dir, "benchmark_results", "manuscript_revision_cycle13_20260725",
              "kernel_suite", "pipeline1", "real_datasets", "nmr_task.rds")
  )
)

make_smoke <- function() {
  runs <- list()
  i <- 0L
  for (task_type in c("classification", "regression")) {
    for (method in c("plssvd", "simpls", "opls", "kernelpls")) {
      for (backend in c("cpu", "metal")) {
        for (precision in c("float64", "float32")) {
          i <- i + 1L
          runs[[i]] <- cfg(
            "smoke", paste0("synthetic_", task_type), task_type,
            method, backend, precision = precision,
            classifier = "argmax", ncomp = 5L,
            save_diagnostics = backend == "metal" && precision == "float64"
          )
        }
      }
    }
  }
  for (method in c("plssvd", "simpls", "opls", "kernelpls")) {
    for (backend in c("cpu", "metal")) {
      for (precision in c("float64", "float32")) {
        i <- i + 1L
        runs[[i]] <- cfg(
          "smoke_lda", "synthetic_classification", "classification",
          method, backend, precision = precision, classifier = "lda",
          ncomp = 5L
        )
      }
    }
  }
  runs
}

make_synthetic_scaling <- function() {
  shapes <- list(
    wide = c(n_train = 400, n_test = 100, p = 2000, q = 20, k = 10),
    tall_thin = c(n_train = 5000, n_test = 1000, p = 50, q = 20, k = 10),
    high_response = c(n_train = 1000, n_test = 250, p = 300, q = 500, k = 50),
    balanced = c(n_train = 5000, n_test = 1000, p = 500, q = 50, k = 50),
    high_components = c(n_train = 3000, n_test = 600, p = 768, q = 200, k = 100)
  )
  runs <- list()
  i <- 0L
  for (shape_name in names(shapes)) {
    s <- shapes[[shape_name]]
    for (method in c("plssvd", "simpls")) {
      for (backend in c("cpu", "metal")) {
        for (replicate in 1:3) {
          i <- i + 1L
          runs[[i]] <- cfg(
            "synthetic_scaling", paste0("synthetic_", shape_name),
            "regression", method, backend, precision = "float64",
            ncomp = s[["k"]], replicate = replicate,
            seed = 100L + replicate,
            n_train = s[["n_train"]], n_test = s[["n_test"]],
            p = s[["p"]], q = s[["q"]],
            latent_rank = min(20L, s[["p"]], s[["q"]]),
            save_diagnostics = replicate == 1L
          )
        }
      }
    }
  }
  runs
}

make_real <- function() {
  runs <- list()
  i <- 0L
  specs <- list(
    metref = list(task_type = "classification", k = 22L,
                  methods = c("plssvd", "simpls", "opls", "kernelpls"),
                  classifiers = c("argmax", "lda"), precisions = c("float64", "float32")),
    retina = list(task_type = "classification", k = 20L,
                  methods = c("plssvd", "simpls", "opls", "kernelpls"),
                  classifiers = c("argmax", "lda"), precisions = c("float64", "float32")),
    tabula = list(task_type = "classification", k = 20L,
                  methods = c("plssvd", "simpls"),
                  classifiers = c("argmax", "lda"), precisions = c("float64", "float32")),
    cifar100 = list(task_type = "classification", k = 50L,
                    methods = c("plssvd", "simpls"),
                    classifiers = c("argmax", "lda"), precisions = c("float64", "float32"))
  )
  for (dataset in names(specs)) {
    if (is.null(task_paths[[dataset]])) next
    s <- specs[[dataset]]
    for (method in s$methods) {
      for (backend in c("cpu", "metal")) {
        for (precision in s$precisions) {
          for (classifier in s$classifiers) {
            for (replicate in 1:3) {
              i <- i + 1L
              runs[[i]] <- cfg(
                "real_dataset", dataset, s$task_type, method, backend,
                precision = precision, classifier = classifier,
                ncomp = s$k, replicate = replicate,
                seed = 200L + replicate, task_path = task_paths[[dataset]]
              )
            }
          }
        }
      }
    }
  }
  runs
}

make_nmr_guarded <- function() {
  if (is.null(task_paths$nmr)) return(list())
  q_limits <- suppressWarnings(as.integer(trimws(strsplit(
    Sys.getenv("FASTPLS_NMR_Q_LIMITS", "500,2000"),
    ",",
    fixed = TRUE
  )[[1L]])))
  q_limits <- q_limits[is.finite(q_limits) & q_limits > 0L]
  runs <- list()
  i <- 0L
  for (q_limit in q_limits) {
    for (method in c("plssvd", "simpls")) {
      for (backend in c("cpu", "metal")) {
        for (precision in c("float64", "float32")) {
          for (replicate in 1:3) {
            i <- i + 1L
            runs[[i]] <- cfg(
              "nmr_guarded", paste0("nmr_q", q_limit), "regression",
              method, backend, precision = precision, ncomp = 25L,
              replicate = replicate, seed = 300L + replicate,
              task_path = task_paths$nmr, q_limit = q_limit
            )
          }
        }
      }
    }
  }
  runs
}

make_cv <- function() {
  list()
}

make_model_specific <- function() {
  runs <- list()
  i <- 0L
  datasets <- list(
    metref = list(
      task_type = "classification",
      task_path = task_paths$metref,
      ncomp = 10L,
      classifier = "lda"
    ),
    synthetic_regression = list(
      task_type = "regression",
      task_path = NULL,
      ncomp = 10L,
      classifier = "argmax"
    )
  )
  for (dataset in names(datasets)) {
    spec <- datasets[[dataset]]
    if (identical(dataset, "metref") && is.null(spec$task_path)) next
    for (backend in c("cpu", "metal")) {
      for (precision in c("float64", "float32")) {
        for (replicate in 1:3) {
          for (kernel in c("linear", "rbf", "poly")) {
            i <- i + 1L
            runs[[i]] <- cfg(
              "kernel_sensitivity", dataset, spec$task_type,
              "kernelpls", backend, precision = precision,
              classifier = spec$classifier, ncomp = spec$ncomp,
              replicate = replicate, seed = 400L + replicate,
              task_path = spec$task_path, kernel = kernel,
              gamma = if (identical(kernel, "linear")) NULL else 1 / 120,
              degree = 2L
            )
          }
          for (north in c(1L, 2L, 3L)) {
            i <- i + 1L
            runs[[i]] <- cfg(
              "opls_sensitivity", dataset, spec$task_type,
              "opls", backend, precision = precision,
              classifier = spec$classifier, ncomp = spec$ncomp,
              replicate = replicate, seed = 500L + replicate,
              task_path = spec$task_path, north = north
            )
          }
        }
      }
    }
  }
  runs
}

runs <- switch(
  stage,
  smoke = make_smoke(),
  scaling = make_synthetic_scaling(),
  real = make_real(),
  nmr = make_nmr_guarded(),
  cv = make_cv(),
  model_specific = make_model_specific(),
  all = c(make_smoke(), make_synthetic_scaling(), make_real(), make_nmr_guarded()),
  stop("Unknown stage: ", stage)
)

dataset_filter <- trimws(strsplit(
  Sys.getenv("FASTPLS_METAL_DATASETS", ""),
  ",",
  fixed = TRUE
)[[1L]])
dataset_filter <- dataset_filter[nzchar(dataset_filter)]
if (length(dataset_filter)) {
  runs <- Filter(function(x) x$dataset %in% dataset_filter, runs)
}
apply_filter <- function(runs, env_name, field) {
  values <- trimws(strsplit(Sys.getenv(env_name, ""), ",", fixed = TRUE)[[1L]])
  values <- values[nzchar(values)]
  if (!length(values)) return(runs)
  Filter(function(x) as.character(x[[field]]) %in% values, runs)
}
runs <- apply_filter(runs, "FASTPLS_METAL_METHODS", "method")
runs <- apply_filter(runs, "FASTPLS_METAL_BACKENDS", "backend")
runs <- apply_filter(runs, "FASTPLS_METAL_PRECISIONS", "precision")
runs <- apply_filter(runs, "FASTPLS_METAL_CLASSIFIERS", "classifier")
svd_method_override <- trimws(Sys.getenv("FASTPLS_METAL_SVD_METHOD", ""))
oversample_override <- suppressWarnings(as.integer(
  Sys.getenv("FASTPLS_METAL_OVERSAMPLE", "")
))
power_override <- suppressWarnings(as.integer(
  Sys.getenv("FASTPLS_METAL_POWER", "")
))
runs <- lapply(runs, function(x) {
  if (nzchar(svd_method_override)) x$svd_method <- svd_method_override
  if (is.finite(oversample_override)) x$oversample <- oversample_override
  if (is.finite(power_override)) x$power <- power_override
  x
})
max_replicate <- suppressWarnings(as.integer(
  Sys.getenv("FASTPLS_METAL_MAX_REPLICATE", "3")
))
if (is.finite(max_replicate)) {
  runs <- Filter(function(x) x$replicate <= max_replicate, runs)
}

if (!length(runs)) stop("No configurations were generated for stage: ", stage)

writeLines(capture.output({
  cat("stage:", stage, "\n")
  cat("created:", format(Sys.time(), "%Y-%m-%d %H:%M:%S %z"), "\n")
  cat("repo_commit:", system2("git", c("-C", repo_dir, "rev-parse", "HEAD"),
                              stdout = TRUE), "\n")
  cat("R:", R.version.string, "\n")
  cat("fastPLS:", as.character(utils::packageVersion("fastPLS")), "\n")
  cat("Metal:", fastPLS::has_metal(), "\n")
  cat("task_paths:\n")
  print(task_paths)
  print(sessionInfo())
}), file.path(out_dir, "session_info.txt"))
saveRDS(runs, file.path(out_dir, "configurations.rds"))

parse_peak_rss <- function(log_file) {
  if (!file.exists(log_file)) return(NA_real_)
  lines <- readLines(log_file, warn = FALSE)
  hit <- grep("maximum resident set size", lines, value = TRUE)
  if (!length(hit)) return(NA_real_)
  bytes <- suppressWarnings(as.numeric(gsub("[^0-9]", "", hit[[length(hit)]])))
  if (is.finite(bytes)) bytes / 1024^2 else NA_real_
}

all_rows <- list()
summary_csv <- file.path(out_dir, "metal_validation_raw.csv")
for (index in seq_along(runs)) {
  one <- runs[[index]]
  config_file <- file.path(out_dir, paste0(one$run_id, "_config.rds"))
  result_file <- file.path(out_dir, paste0(one$run_id, "_result.rds"))
  stdout_file <- file.path(out_dir, paste0(one$run_id, ".out"))
  stderr_file <- file.path(out_dir, paste0(one$run_id, ".time"))
  saveRDS(one, config_file)
  log_msg(index, "/", length(runs), " ", one$run_id)
  status <- system2(
    "/usr/bin/time",
    c("-l", file.path(R.home("bin"), "Rscript"), worker,
      config_file, result_file),
    stdout = stdout_file,
    stderr = stderr_file
  )
  if (file.exists(result_file)) {
    row <- readRDS(result_file)
  } else {
    row <- data.frame(
      run_id = one$run_id, experiment = one$experiment,
      dataset = one$dataset, task_type = one$task_type,
      method = one$method, backend_requested = one$backend,
      backend_reported = NA_character_, prediction_backend = NA_character_,
      svd_method = one$svd_method, classifier = one$classifier,
      precision = one$precision, ncomp = one$ncomp,
      n_train = NA_integer_, n_test = NA_integer_, p = NA_integer_, q = NA_integer_,
      seed = one$seed, replicate = one$replicate,
      oversample = one$oversample, power = one$power,
      kernel = one$kernel, north = one$north,
      fit_sec = NA_real_, prediction_sec = NA_real_, total_sec = NA_real_,
      baseline_rss_mb = NA_real_, rss_after_fit_mb = NA_real_,
      rss_after_prediction_mb = NA_real_, peak_rss_mb = NA_real_,
      incremental_peak_rss_mb = NA_real_, metric_name = NA_character_,
      metric_value = NA_real_, accuracy = NA_real_, q2 = NA_real_,
      rmsd = NA_real_, prediction_checksum = NA_real_,
      prediction_length = NA_integer_, status = "process_failed",
      warnings = "", error = paste("exit status", status),
      stringsAsFactors = FALSE
    )
  }
  row$peak_rss_mb <- parse_peak_rss(stderr_file)
  row$incremental_peak_rss_mb <- if (
    is.finite(row$peak_rss_mb) && is.finite(row$baseline_rss_mb)
  ) {
    max(0, row$peak_rss_mb - row$baseline_rss_mb)
  } else {
    NA_real_
  }
  all_rows[[length(all_rows) + 1L]] <- row
  result <- do.call(rbind, all_rows)
  write.csv(result, summary_csv, row.names = FALSE)
  unlink(config_file)
  log_msg("status=", row$status,
          if (is.finite(row$total_sec)) paste0(" total=", round(row$total_sec, 3), "s") else "",
          if (is.finite(row$metric_value)) paste0(" metric=", signif(row$metric_value, 6)) else "")
}

log_msg("Completed stage ", stage, ". Results: ", summary_csv)
