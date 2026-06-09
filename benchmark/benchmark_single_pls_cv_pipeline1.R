#!/usr/bin/env Rscript

suppressPackageStartupMessages(library(fastPLS))

timestamp <- function() format(Sys.time(), "%Y-%m-%d %H:%M:%S")
log_msg <- function(...) cat("[", timestamp(), "] ", paste(..., collapse = ""), "\n", sep = "")

arg_value <- function(name, default = NULL) {
  args <- commandArgs(trailingOnly = TRUE)
  key <- paste0("--", name, "=")
  hit <- grep(paste0("^", key), args, value = TRUE)
  if (length(hit)) sub(paste0("^", key), "", hit[[1L]]) else default
}

split_csv <- function(x, default) {
  if (is.null(x) || !nzchar(x)) x <- default
  out <- trimws(strsplit(x, ",", fixed = TRUE)[[1L]])
  out[nzchar(out)]
}

task_dir <- arg_value(
  "task-dir",
  Sys.getenv(
    "FASTPLS_REAL_TASK_DIR",
    "/Users/stefano/Documents/GPUPLS/local_usual_pipeline_metal_20260515_230543/real_datasets"
  )
)
out_dir <- arg_value(
  "out-dir",
  Sys.getenv(
    "FASTPLS_SINGLE_CV_OUT",
    file.path(getwd(), "benchmark_results", paste0("single_pls_cv_pipeline1_", format(Sys.time(), "%Y%m%d_%H%M%S")))
  )
)
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

ncomp_grid <- list(
  metref = c(2L, 5L, 10L, 22L, 50L, 100L),
  ccle = c(2L, 5L, 10L, 18L, 50L, 100L),
  cifar100 = c(2L, 5L, 10L, 20L, 50L, 100L, 200L),
  nmr = c(2L, 5L, 10L, 20L, 50L, 100L, 200L, 500L),
  singlecell = c(2L, 5L, 10L, 20L, 50L),
  tcga_brca = c(2L, 5L, 10L, 20L, 50L),
  tcga_hnsc_methylation = c(2L, 5L, 10L, 20L, 50L),
  tcga_pan_cancer = c(2L, 5L, 10L, 20L, 50L, 100L),
  gtex_v8 = c(2L, 5L, 10L, 20L, 32L, 50L, 100L),
  prism = c(2L, 5L, 10L, 20L, 50L, 100L),
  cbmc_citeseq = c(2L, 5L, 10L, 20L, 50L)
)

datasets <- split_csv(
  arg_value("datasets", Sys.getenv("FASTPLS_SINGLE_CV_DATASETS", "")),
  paste(c("metref", "ccle", "prism", "singlecell", "tcga_hnsc_methylation"), collapse = ",")
)
methods <- split_csv(
  arg_value("methods", Sys.getenv("FASTPLS_SINGLE_CV_METHODS", "")),
  "plssvd,simpls,opls,kernelpls"
)
backends <- split_csv(
  arg_value("backends", Sys.getenv("FASTPLS_SINGLE_CV_BACKENDS", "")),
  "cpu,metal"
)
classifiers <- split_csv(
  arg_value("classifiers", Sys.getenv("FASTPLS_SINGLE_CV_CLASSIFIERS", "")),
  "argmax,lda,cknn"
)
cpu_svd_methods <- split_csv(
  arg_value("cpu-svd-methods", Sys.getenv("FASTPLS_SINGLE_CV_CPU_SVD_METHODS", "")),
  "rsvd,irlba"
)
kfold <- as.integer(arg_value("kfold", Sys.getenv("FASTPLS_SINGLE_CV_KFOLD", "5")))
host_label <- arg_value("host-label", Sys.info()[["nodename"]])

metric_label <- function(task_type) {
  if (identical(task_type, "classification")) "accuracy" else "rmsd"
}

run_one <- function(dataset, method, backend, svd_method, classifier) {
  task_path <- file.path(task_dir, paste0(dataset, "_task.rds"))
  if (!file.exists(task_path)) {
    return(data.frame(
      host = host_label, dataset = dataset, task_type = NA_character_,
      method = method, backend = backend, svd_method = svd_method,
      classifier = classifier, ncomp_grid = NA_character_, kfold = kfold,
      elapsed_sec = NA_real_, best_ncomp = NA_integer_,
      best_metric_name = NA_character_, best_metric_value = NA_real_,
      status = "missing_task", message = task_path
    ))
  }
  task <- readRDS(task_path)
  task_type <- task$task_type
  if (!identical(task_type, "classification")) classifier <- "argmax"
  grid <- ncomp_grid[[dataset]]
  if (is.null(grid)) stop("No ncomp grid for dataset: ", dataset, call. = FALSE)
  if (identical(method, "plssvd")) {
    q <- if (identical(task_type, "classification")) length(levels(factor(task$Ytrain))) else ncol(as.matrix(task$Ytrain))
    grid <- pmin(grid, q)
    grid <- sort(unique(pmax(1L, as.integer(grid))))
  }
  if (!identical(backend, "cpu")) svd_method <- "rsvd"
  selection_metric <- metric_label(task_type)

  log_msg("RUN dataset=", dataset, " method=", method, " backend=", backend,
          " svd=", svd_method, " classifier=", classifier,
          " metric=", selection_metric, " ncomp=", paste(grid, collapse = "/"))

  fit <- NULL
  tm <- system.time({
    fit <- tryCatch(
      fastPLS::pls.single.cv(
        Xdata = as.matrix(task$Xtrain),
        Ydata = if (identical(task_type, "classification")) factor(task$Ytrain) else as.matrix(task$Ytrain),
        ncomp = grid,
        kfold = kfold,
        method = method,
        backend = backend,
        svd.method = svd_method,
        classifier = classifier,
        selection_metric = selection_metric,
        seed = 123
      ),
      error = function(e) e
    )
  })
  if (inherits(fit, "error")) {
    log_msg("ERR dataset=", dataset, " method=", method, " backend=", backend,
            " msg=", conditionMessage(fit))
    return(data.frame(
      host = host_label, dataset = dataset, task_type = task_type,
      method = method, backend = backend, svd_method = svd_method,
      classifier = classifier, ncomp_grid = paste(grid, collapse = ";"),
      kfold = kfold, elapsed_sec = unname(tm[["elapsed"]]),
      best_ncomp = NA_integer_, best_metric_name = NA_character_,
      best_metric_value = NA_real_, status = "error",
      message = conditionMessage(fit)
    ))
  }
  out <- data.frame(
    host = host_label,
    dataset = dataset,
    task_type = task_type,
    method = method,
    backend = backend,
    svd_method = svd_method,
    classifier = classifier,
    ncomp_grid = paste(grid, collapse = ";"),
    kfold = kfold,
    elapsed_sec = unname(tm[["elapsed"]]),
    best_ncomp = as.integer(fit$best_ncomp),
    best_metric_name = as.character(fit$best_metric_name),
    best_metric_value = as.numeric(fit$best_metric_value),
    status = "ok",
    message = "",
    stringsAsFactors = FALSE
  )
  log_msg("OK dataset=", dataset, " method=", method, " backend=", backend,
          " elapsed=", round(out$elapsed_sec, 3),
          " best_ncomp=", out$best_ncomp, " ", out$best_metric_name, "=",
          signif(out$best_metric_value, 6))
  out
}

rows <- data.frame()
out_csv <- file.path(out_dir, "single_pls_cv_pipeline1_raw.csv")
for (dataset in datasets) {
  for (method in methods) {
    for (backend in backends) {
      svd_list <- if (identical(backend, "cpu")) cpu_svd_methods else "rsvd"
      for (svd_method in svd_list) {
        task_path <- file.path(task_dir, paste0(dataset, "_task.rds"))
        task_type <- if (file.exists(task_path)) readRDS(task_path)$task_type else NA_character_
        cls_list <- if (identical(task_type, "classification")) classifiers else "argmax"
        for (classifier in cls_list) {
          rows <- rbind(rows, run_one(dataset, method, backend, svd_method, classifier))
          write.csv(rows, out_csv, row.names = FALSE)
        }
      }
    }
  }
}

ok <- rows[rows$status == "ok", , drop = FALSE]
summary_csv <- file.path(out_dir, "single_pls_cv_pipeline1_summary.csv")
write.csv(ok[order(ok$dataset, ok$method, ok$backend, ok$elapsed_sec), ], summary_csv, row.names = FALSE)
writeLines(capture.output(sessionInfo()), file.path(out_dir, "sessionInfo.txt"))
log_msg("Saved raw: ", out_csv)
log_msg("Saved summary: ", summary_csv)
print(rows)
