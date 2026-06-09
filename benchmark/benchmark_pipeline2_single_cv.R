#!/usr/bin/env Rscript

script_arg <- grep("^--file=", commandArgs(trailingOnly = FALSE), value = TRUE)
script_file <- if (length(script_arg)) sub("^--file=", "", script_arg[[1L]]) else file.path(getwd(), "benchmark_pipeline2_single_cv.R")
script_dir <- dirname(normalizePath(script_file, winslash = "/", mustWork = FALSE))
source(file.path(script_dir, "helpers_dataset_memory_compare.R"))

`%||%` <- function(x, y) {
  if (is.null(x) || length(x) == 0L) y else x
}

timestamp <- function() format(Sys.time(), "%Y-%m-%d %H:%M:%S")
log_msg <- function(...) cat("[", timestamp(), "] ", sprintf(...), "\n", sep = "")

split_csv <- function(x, default) {
  if (is.null(x) || !nzchar(x)) x <- default
  trimws(strsplit(x, ",", fixed = TRUE)[[1L]])
}

default_best_ncomp <- function() {
  data.frame(
    dataset = c(
      "metref", "ccle", "cifar100", "prism", "gtex_v8",
      "tcga_pan_cancer", "singlecell", "tcga_brca",
      "tcga_hnsc_methylation", "nmr", "cbmc_citeseq"
    ),
    best_ncomp = c(100L, 50L, 200L, 5L, 100L, 100L, 50L, 5L, 2L, 50L, 50L),
    source = "embedded_pipeline1_best_single_run",
    stringsAsFactors = FALSE
  )
}

find_dataset_rdata_pipeline2 <- function(dataset_id) {
  fname <- dataset_filename(dataset_id)
  env_name <- sprintf("FASTPLS_%s_RDATA", toupper(dataset_id))
  candidates <- c(
    Sys.getenv(env_name, ""),
    file.path(path.expand("~"), "Documents", "GPUPLS", "Data", fname),
    file.path(path.expand("~"), "Documents", "fastPLS", "data", fname),
    file.path(path.expand("~"), "Documents", "Rdatasets", fname),
    file.path(path.expand("~"), "GPUPLS", "Data", fname),
    file.path(dirname(script_dir), "Data", fname),
    file.path(dirname(dirname(script_dir)), "Data", fname)
  )
  candidates <- unique(Filter(nzchar, vapply(candidates, normalize_path_if_exists, character(1))))
  for (cand in candidates) {
    if (file.exists(cand)) return(cand)
  }
  allow_recursive <- tolower(Sys.getenv("PIPELINE2_ALLOW_RECURSIVE_SEARCH", "0")) %in% c("1", "true", "yes", "y")
  if (allow_recursive) {
    return(find_dataset_rdata(dataset_id))
  }
  stop(
    "Dataset RData not found for ", dataset_id,
    ". Set FASTPLS_", toupper(dataset_id), "_RDATA, PIPELINE2_TASK_DIR, or PIPELINE2_ALLOW_RECURSIVE_SEARCH=1."
  )
}

find_task_rds_pipeline2 <- function(dataset_id, task_dir = "") {
  env_name <- sprintf("FASTPLS_%s_TASK_RDS", toupper(dataset_id))
  candidates <- c(
    Sys.getenv(env_name, ""),
    if (nzchar(task_dir)) file.path(task_dir, paste0(dataset_id, "_task.rds")) else "",
    Sys.glob(file.path(dirname(dirname(script_dir)), "*", "real_datasets", paste0(dataset_id, "_task.rds"))),
    Sys.glob(file.path(dirname(dirname(script_dir)), "*", paste0(dataset_id, "_task.rds")))
  )
  candidates <- unique(Filter(nzchar, vapply(candidates, normalize_path_if_exists, character(1))))
  for (cand in candidates) {
    if (file.exists(cand)) return(cand)
  }
  ""
}

load_task_pipeline2 <- function(dataset_id, split_seed, task_dir = "") {
  task_rds <- find_task_rds_pipeline2(dataset_id, task_dir = task_dir)
  if (nzchar(task_rds)) {
    task <- readRDS(task_rds)
    task$dataset <- dataset_id
    task$dataset_path <- task$dataset_path %||% task_rds
    return(task)
  }
  as_task(find_dataset_rdata_pipeline2(dataset_id), dataset_id = dataset_id, split_seed = split_seed)
}

best_ncomp_from_pipeline1 <- function(path) {
  if (is.null(path) || !nzchar(path) || !file.exists(path)) {
    return(default_best_ncomp())
  }
  d <- read.csv(path, stringsAsFactors = FALSE, check.names = FALSE)
  required <- c("dataset", "requested_ncomp", "metric_name", "metric_value")
  if (!all(required %in% names(d))) {
    warning("Best-ncomp source lacks raw metric columns; using embedded defaults.", call. = FALSE)
    return(default_best_ncomp())
  }
  if ("status" %in% names(d)) {
    d <- d[d$status %in% c("ok", "capped"), , drop = FALSE]
  }
  d <- d[is.finite(d$metric_value) & is.finite(d$requested_ncomp), , drop = FALSE]
  if (!nrow(d)) return(default_best_ncomp())
  best <- do.call(rbind, lapply(split(d, d$dataset), function(x) {
    metric <- tolower(x$metric_name[[which(!is.na(x$metric_name))[1L]]])
    loss <- metric %in% c("rmsd", "rmse", "mae", "mse")
    hit <- if (loss) x[which.min(x$metric_value), , drop = FALSE] else x[which.max(x$metric_value), , drop = FALSE]
    data.frame(
      dataset = hit$dataset[[1L]],
      best_ncomp = as.integer(hit$requested_ncomp[[1L]]),
      source = normalizePath(path, winslash = "/", mustWork = FALSE),
      stringsAsFactors = FALSE
    )
  }))
  defaults <- default_best_ncomp()
  missing <- setdiff(defaults$dataset, best$dataset)
  if (length(missing)) best <- rbind(best, defaults[defaults$dataset %in% missing, , drop = FALSE])
  best[order(best$dataset), , drop = FALSE]
}

combined_task_data <- function(task) {
  X <- rbind(as.matrix(task$Xtrain), as.matrix(task$Xtest))
  if (identical(task$task_type, "classification")) {
    y <- factor(
      c(as.character(task$Ytrain), as.character(task$Ytest)),
      levels = levels(task$Ytrain)
    )
  } else {
    y <- rbind(as.matrix(task$Ytrain), as.matrix(task$Ytest))
  }
  list(X = X, Y = y)
}

make_pipeline2_grid <- function(task_type, methods, backends, cpu_svd_methods, gpu_backend, classifiers) {
  rows <- list()
  add <- function(method, backend, svd_method, classifier, native) {
    rows[[length(rows) + 1L]] <<- data.frame(
      method = method,
      backend = backend,
      svd.method = svd_method,
      classifier = classifier,
      native_backend = native,
      stringsAsFactors = FALSE
    )
  }
  cls <- if (identical(task_type, "classification")) classifiers else "argmax"
  for (method in methods) {
    if ("cpu" %in% backends) {
      for (svd_method in cpu_svd_methods) {
        for (classifier in cls) add(method, "cpu", svd_method, classifier, "cpp")
      }
    }
    if ("gpu" %in% backends) {
      for (classifier in cls) add(method, gpu_backend, "rsvd", classifier, gpu_backend)
    }
  }
  do.call(rbind, rows)
}

metric_from_cv <- function(cv, classification) {
  metrics <- cv$metrics
  if (!is.data.frame(metrics) || !nrow(metrics)) {
    return(list(metric_name = NA_character_, metric_value = NA_real_))
  }
  metric_name <- as.character(metrics$metric_name[[1L]])
  metric_value <- as.numeric(metrics$metric_value[[1L]])
  list(metric_name = metric_name, metric_value = metric_value)
}

args <- parse_kv_args()
out_dir <- normalizePath(arg_value(args, "out_dir", default = file.path("benchmark_results", "pipeline2_single_cv")), winslash = "/", mustWork = FALSE)
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

host_label <- arg_value(args, "host_label", default = Sys.info()[["nodename"]])
datasets <- split_csv(arg_value(args, "datasets", default = Sys.getenv("PIPELINE2_DATASETS", "")),
                      "metref,ccle,cifar100,prism,gtex_v8,tcga_pan_cancer,singlecell,tcga_brca,tcga_hnsc_methylation,nmr,cbmc_citeseq")
methods <- split_csv(arg_value(args, "methods", default = Sys.getenv("PIPELINE2_METHODS", "")),
                     "plssvd,simpls,opls,kernelpls")
backends <- split_csv(arg_value(args, "backends", default = Sys.getenv("PIPELINE2_BACKENDS", "")), "cpu,gpu")
cpu_svd_methods <- split_csv(arg_value(args, "cpu_svd_methods", default = Sys.getenv("PIPELINE2_CPU_SVD_METHODS", "")), "rsvd,irlba")
classifiers <- split_csv(arg_value(args, "classifiers", default = Sys.getenv("PIPELINE2_CLASSIFIERS", "")), "argmax,lda,cknn")
split_seed <- suppressWarnings(as.integer(arg_value(args, "split_seed", default = Sys.getenv("PIPELINE2_SPLIT_SEED", "123"))))
kfold <- arg_value(args, "kfold", default = Sys.getenv("PIPELINE2_KFOLD", "5"))
timeout_sec <- suppressWarnings(as.numeric(arg_value(args, "timeout_sec", default = Sys.getenv("PIPELINE2_TIMEOUT_SEC", "3600"))))
best_source <- arg_value(args, "best_source", default = Sys.getenv("PIPELINE2_BEST_SOURCE", ""))
task_dir <- arg_value(args, "task_dir", default = Sys.getenv("PIPELINE2_TASK_DIR", ""))
lib_loc <- arg_value(args, "lib_loc", default = Sys.getenv("PIPELINE2_LIB_LOC", ""))
k_cknn <- suppressWarnings(as.integer(arg_value(args, "k", default = Sys.getenv("PIPELINE2_CKNN_K", "10"))))
tau_cknn <- suppressWarnings(as.numeric(arg_value(args, "tau", default = Sys.getenv("PIPELINE2_CKNN_TAU", "0.2"))))
alpha_cknn <- suppressWarnings(as.numeric(arg_value(args, "alpha", default = Sys.getenv("PIPELINE2_CKNN_ALPHA", "0.75"))))
top_m_cknn <- suppressWarnings(as.integer(arg_value(args, "top_m", default = Sys.getenv("PIPELINE2_CKNN_TOP_M", "20"))))
gpu_backend <- benchmark_gpu_backend()
if (!nzchar(lib_loc)) lib_loc <- .libPaths()[[1L]]

if (!is.finite(split_seed) || is.na(split_seed)) split_seed <- 123L
if (!is.finite(timeout_sec) || is.na(timeout_sec) || timeout_sec <= 0) timeout_sec <- 3600
if (!is.finite(k_cknn) || is.na(k_cknn) || k_cknn < 1L) k_cknn <- 10L
if (!is.finite(tau_cknn) || is.na(tau_cknn) || tau_cknn <= 0) tau_cknn <- 0.2
if (!is.finite(alpha_cknn) || is.na(alpha_cknn)) alpha_cknn <- 0.75
if (!is.finite(top_m_cknn) || is.na(top_m_cknn) || top_m_cknn < 1L) top_m_cknn <- 20L
kfold_arg <- if (tolower(kfold) %in% c("loocv", "loo")) "loocv" else suppressWarnings(as.integer(kfold))
if (length(kfold_arg) != 1L || (is.numeric(kfold_arg) && (!is.finite(kfold_arg) || is.na(kfold_arg) || kfold_arg < 2L))) {
  kfold_arg <- 5L
}
best_table <- best_ncomp_from_pipeline1(best_source)
write.csv(best_table, file.path(out_dir, "pipeline2_best_ncomp.csv"), row.names = FALSE)

raw_path <- file.path(out_dir, "pipeline2_single_cv_raw.csv")
manifest_path <- file.path(out_dir, "pipeline2_single_cv_manifest.txt")
writeLines(c(
  paste("created:", timestamp()),
  paste("host_label:", host_label),
  paste("gpu_backend:", gpu_backend),
  paste("datasets:", paste(datasets, collapse = ",")),
  paste("methods:", paste(methods, collapse = ",")),
  paste("backends:", paste(backends, collapse = ",")),
  paste("cpu_svd_methods:", paste(cpu_svd_methods, collapse = ",")),
  paste("classifiers:", paste(classifiers, collapse = ",")),
  paste("kfold:", as.character(kfold_arg)),
  paste("cknn_k:", k_cknn),
  paste("cknn_tau:", tau_cknn),
  paste("cknn_alpha:", alpha_cknn),
  paste("cknn_top_m:", top_m_cknn),
  paste("timeout_sec:", timeout_sec),
  paste("best_source:", if (nzchar(best_source)) best_source else "embedded defaults"),
  paste("task_dir:", if (nzchar(task_dir)) task_dir else "auto"),
  paste("lib_loc:", lib_loc)
), manifest_path)

suppressPackageStartupMessages(library("fastPLS", lib.loc = lib_loc, character.only = TRUE))

append_row <- function(row) {
  write.table(
    row,
    raw_path,
    sep = ",",
    row.names = FALSE,
    col.names = !file.exists(raw_path),
    append = file.exists(raw_path),
    qmethod = "double"
  )
}

for (dataset_id in datasets) {
  dataset_id <- tolower(dataset_id)
  hit <- best_table[best_table$dataset == dataset_id, , drop = FALSE]
  requested_ncomp <- if (nrow(hit)) as.integer(hit$best_ncomp[[1L]]) else 5L
  log_msg("Preparing dataset=%s ncomp=%s", dataset_id, requested_ncomp)
  task <- tryCatch({
    load_task_pipeline2(dataset_id, split_seed = split_seed, task_dir = task_dir)
  }, error = function(e) e)
  if (inherits(task, "error")) {
    append_row(data.frame(
      host = host_label, dataset = dataset_id, task_type = NA_character_,
      method = NA_character_, backend = NA_character_, svd.method = NA_character_,
      classifier = NA_character_, requested_ncomp = requested_ncomp, effective_ncomp = NA_integer_,
      kfold = as.character(kfold_arg), n = NA_integer_, p = NA_integer_, q = NA_integer_,
      elapsed_sec = NA_real_, metric_name = NA_character_, metric_value = NA_real_,
      status = "dataset_error", msg = conditionMessage(task), stringsAsFactors = FALSE
    ))
    next
  }
  dat <- combined_task_data(task)
  X <- dat$X
  Y <- dat$Y
  q <- if (identical(task$task_type, "classification")) nlevels(Y) else ncol(as.matrix(Y))
  grid <- make_pipeline2_grid(task$task_type, methods, backends, cpu_svd_methods, gpu_backend, classifiers)

  for (i in seq_len(nrow(grid))) {
    spec <- grid[i, , drop = FALSE]
    effective_ncomp <- safe_effective_ncomp(
      list(n_train = nrow(X), p = ncol(X), n_classes = q),
      requested_ncomp,
      method_family = spec$method
    )
    status <- "ok"
    msg <- ""
    metric_name <- NA_character_
    metric_value <- NA_real_
    elapsed_sec <- NA_real_

    if (identical(spec$backend, "cuda") && !isTRUE(fastPLS::has_cuda())) {
      status <- "skipped_no_cuda"
      msg <- "CUDA backend not available"
    } else if (identical(spec$backend, "metal") && !isTRUE(fastPLS::has_metal())) {
      status <- "skipped_no_metal"
      msg <- "Metal backend not available"
    } else if (identical(spec$backend, "metal") && !identical(spec$classifier, "argmax")) {
      status <- "skipped_not_native_metal_cv"
      msg <- "Metal pls.single.cv currently has native compiled CV for argmax only"
    }

    if (identical(status, "ok")) {
      log_msg(
        "RUN dataset=%s method=%s backend=%s svd=%s classifier=%s ncomp=%s",
        dataset_id, spec$method, spec$backend, spec$svd.method, spec$classifier, effective_ncomp
      )
      res <- tryCatch({
        setTimeLimit(elapsed = timeout_sec, transient = TRUE)
        t0 <- proc.time()[["elapsed"]]
        cv <- fastPLS::pls.single.cv(
          Xdata = X,
          Ydata = Y,
          ncomp = effective_ncomp,
          kfold = kfold_arg,
          method = spec$method,
          backend = spec$backend,
          svd.method = spec$svd.method,
          classifier = spec$classifier,
          k = k_cknn,
          tau = tau_cknn,
          alpha = alpha_cknn,
          top_m = top_m_cknn,
          scaling = "centering",
          seed = 123L,
          return_scores = FALSE,
          xprod = NULL
        )
        elapsed <- proc.time()[["elapsed"]] - t0
        setTimeLimit(cpu = Inf, elapsed = Inf, transient = FALSE)
        met <- metric_from_cv(cv, identical(task$task_type, "classification"))
        list(elapsed = as.numeric(elapsed), metric_name = met$metric_name, metric_value = met$metric_value)
      }, error = function(e) {
        setTimeLimit(cpu = Inf, elapsed = Inf, transient = FALSE)
        list(error = conditionMessage(e))
      })
      if (!is.null(res$error)) {
        status <- if (grepl("time limit", res$error, ignore.case = TRUE)) "timeout" else "error"
        msg <- res$error
      } else {
        elapsed_sec <- res$elapsed
        metric_name <- res$metric_name
        metric_value <- res$metric_value
      }
    }

    append_row(data.frame(
      host = host_label,
      dataset = dataset_id,
      task_type = task$task_type,
      method = spec$method,
      backend = spec$backend,
      svd.method = spec$svd.method,
      classifier = spec$classifier,
      requested_ncomp = as.integer(requested_ncomp),
      effective_ncomp = as.integer(effective_ncomp),
      kfold = as.character(kfold_arg),
      n = as.integer(nrow(X)),
      p = as.integer(ncol(X)),
      q = as.integer(q),
      elapsed_sec = elapsed_sec,
      metric_name = metric_name,
      metric_value = metric_value,
      status = status,
      msg = msg,
      stringsAsFactors = FALSE
    ))
  }
}

if (file.exists(raw_path)) {
  raw <- read.csv(raw_path, stringsAsFactors = FALSE)
  ok <- raw[raw$status == "ok" & is.finite(raw$metric_value), , drop = FALSE]
  if (nrow(ok)) {
    summary <- aggregate(
      cbind(elapsed_sec, metric_value) ~ host + dataset + task_type + method + backend + svd.method + classifier + requested_ncomp + effective_ncomp + metric_name,
      ok,
      median,
      na.rm = TRUE
    )
    write.csv(summary, file.path(out_dir, "pipeline2_single_cv_summary.csv"), row.names = FALSE)
  }
}

log_msg("Pipeline 2 finished. Results: %s", raw_path)
