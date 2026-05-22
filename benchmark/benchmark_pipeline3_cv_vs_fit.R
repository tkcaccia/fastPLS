#!/usr/bin/env Rscript

# Pipeline 3: compare ordinary fastPLS fit+prediction speed with 10-fold CV
# speed for the same method/backend/classifier combinations on real datasets.

options(stringsAsFactors = FALSE)

bench_lib <- Sys.getenv("FASTPLS_BENCH_LIB", "")
if (nzchar(bench_lib) && dir.exists(bench_lib)) {
  .libPaths(unique(c(normalizePath(bench_lib, winslash = "/", mustWork = TRUE), .libPaths())))
}

parse_args <- function(args = commandArgs(trailingOnly = TRUE)) {
  out <- list()
  for (arg in args) {
    if (!startsWith(arg, "--")) next
    kv <- substring(arg, 3L)
    bits <- strsplit(kv, "=", fixed = TRUE)[[1L]]
    key <- gsub("-", "_", bits[[1L]], fixed = TRUE)
    val <- if (length(bits) > 1L) paste(bits[-1L], collapse = "=") else "TRUE"
    out[[key]] <- val
  }
  out
}

args <- parse_args()
arg <- function(key, default = NULL) {
  val <- args[[key]]
  if (is.null(val) || !nzchar(val)) default else val
}

`%||%` <- function(a, b) if (is.null(a)) b else a

script_file <- grep("^--file=", commandArgs(FALSE), value = TRUE)
script_file <- if (length(script_file)) sub("^--file=", "", script_file[[1L]]) else "benchmark_pipeline3_cv_vs_fit.R"
repo_root <- normalizePath(file.path(dirname(script_file), ".."), winslash = "/", mustWork = FALSE)
source(file.path(repo_root, "benchmark", "helpers_dataset_memory_compare.R"))

mode <- arg("mode", "run_one")
dataset_id <- tolower(arg("dataset", "metref"))
ncomp_requested <- suppressWarnings(as.integer(arg("ncomp", "50")))
if (!is.finite(ncomp_requested) || is.na(ncomp_requested) || ncomp_requested < 1L) ncomp_requested <- 50L
method_id <- arg("method_id", "")
benchmark_mode <- arg("benchmark_mode", "fit_predict")
replicate_id <- suppressWarnings(as.integer(arg("replicate", "1")))
if (!is.finite(replicate_id) || is.na(replicate_id)) replicate_id <- 1L
split_seed <- suppressWarnings(as.integer(arg("seed", "123")))
if (!is.finite(split_seed) || is.na(split_seed)) split_seed <- 123L
kfold <- suppressWarnings(as.integer(arg("kfold", "10")))
if (!is.finite(kfold) || is.na(kfold) || kfold < 2L) kfold <- 10L
row_out <- arg("row_out", "")
results_dir <- arg("results_dir", getwd())
status_override <- arg("status", "")
message_override <- arg("message", "")

cknn_k <- suppressWarnings(as.integer(arg("k", Sys.getenv("FASTPLS_PIPELINE3_CKNN_K", "10"))))
cknn_tau <- suppressWarnings(as.numeric(arg("tau", Sys.getenv("FASTPLS_PIPELINE3_CKNN_TAU", "0.2"))))
cknn_alpha <- suppressWarnings(as.numeric(arg("alpha", Sys.getenv("FASTPLS_PIPELINE3_CKNN_ALPHA", "0.75"))))
cknn_top_m <- suppressWarnings(as.integer(arg("top_m", Sys.getenv("FASTPLS_PIPELINE3_CKNN_TOP_M", "20"))))
if (!is.finite(cknn_k) || is.na(cknn_k) || cknn_k < 1L) cknn_k <- 10L
if (!is.finite(cknn_tau) || is.na(cknn_tau) || cknn_tau <= 0) cknn_tau <- 0.2
if (!is.finite(cknn_alpha) || is.na(cknn_alpha)) cknn_alpha <- 0.75
if (!is.finite(cknn_top_m) || is.na(cknn_top_m) || cknn_top_m < 1L) cknn_top_m <- 20L

classification_dataset_ids <- c(
  "ccle", "cifar100", "gtex_v8", "imagenet", "metref",
  "singlecell", "tcga_brca", "tcga_hnsc_methylation", "tcga_pan_cancer"
)
regression_dataset_ids <- c("cbmc_citeseq", "nmr", "prism")

fastpls_method_ids <- function(dataset_id) {
  is_classification <- dataset_id %in% classification_dataset_ids
  if (dataset_id %in% regression_dataset_ids) is_classification <- FALSE
  methods <- c("plssvd", "simpls", "opls", "kernelpls")
  cpu_svd <- c("rsvd", "irlba")
  gpu_backend <- benchmark_gpu_backend()
  ids <- character()
  for (method in methods) {
    for (svd_method in cpu_svd) {
      base <- paste("fastPLS", method, "cpu", svd_method, sep = "_")
      ids <- c(ids, base)
      if (is_classification) {
        ids <- c(ids, paste0(base, "_lda"), paste0(base, "_cknn"))
      }
    }
    base_gpu <- paste("fastPLS", method, gpu_backend, "rsvd", sep = "_")
    ids <- c(ids, base_gpu)
    if (is_classification) {
      ids <- c(ids, paste0(base_gpu, "_lda"), paste0(base_gpu, "_cknn"))
    }
  }
  ids
}

if (identical(mode, "list_methods")) {
  cat(paste(fastpls_method_ids(dataset_id), collapse = "\n"))
  cat("\n")
  quit(save = "no", status = 0)
}

parse_method_id <- function(method_id) {
  x <- sub("^fastPLS_", "", method_id)
  classifier <- "argmax"
  if (grepl("_lda$", x)) {
    classifier <- "lda"
    x <- sub("_lda$", "", x)
  } else if (grepl("_cknn$", x)) {
    classifier <- "cknn"
    x <- sub("_cknn$", "", x)
  }
  parts <- strsplit(x, "_", fixed = TRUE)[[1L]]
  if (length(parts) != 3L) stop("Unsupported fastPLS method_id: ", method_id)
  list(method = parts[[1L]], backend = parts[[2L]], svd_method = parts[[3L]], classifier = classifier)
}

write_row <- function(row) {
  if (!nzchar(row_out)) {
    print(row)
    return(invisible(row))
  }
  dir.create(dirname(row_out), recursive = TRUE, showWarnings = FALSE)
  write.csv(row, file = row_out, row.names = FALSE, quote = TRUE)
  invisible(row)
}

empty_row <- function(status, msg, task = NULL, spec = NULL) {
  if (is.null(spec) && nzchar(method_id)) {
    spec <- tryCatch(parse_method_id(method_id), error = function(e) NULL)
  }
  data.frame(
    dataset = dataset_id,
    task_type = if (!is.null(task)) task$task_type else NA_character_,
    dataset_path = if (!is.null(task)) task$dataset_path else NA_character_,
    split_seed = split_seed,
    n_train = if (!is.null(task)) task$n_train else NA_integer_,
    n_test = if (!is.null(task)) task$n_test else NA_integer_,
    n_total = if (!is.null(task)) task$n_train + task$n_test else NA_integer_,
    p = if (!is.null(task)) task$p else NA_integer_,
    q = if (!is.null(task)) task$n_classes else NA_integer_,
    ncomp_requested = ncomp_requested,
    effective_ncomp = NA_integer_,
    kfold = kfold,
    benchmark_mode = benchmark_mode,
    method_id = method_id,
    method = if (!is.null(spec)) spec$method else NA_character_,
    backend = if (!is.null(spec)) spec$backend else NA_character_,
    svd.method = if (!is.null(spec)) spec$svd_method else NA_character_,
    classifier = if (!is.null(spec)) spec$classifier else NA_character_,
    fit_time_sec = NA_real_,
    predict_time_sec = NA_real_,
    total_time_sec = NA_real_,
    cv_time_sec = NA_real_,
    metric_name = NA_character_,
    metric_value = NA_real_,
    status = status,
    error_message = msg,
    stringsAsFactors = FALSE
  )
}

if (identical(mode, "missing_row")) {
  write_row(empty_row(status_override %||% "missing_row", message_override %||% "Rscript did not produce a row"))
  quit(save = "no", status = 0)
}

load_pipeline3_task <- function(dataset_id, split_seed) {
  path <- find_dataset_rdata(dataset_id)
  task <- as_task(path, dataset_id = dataset_id, split_seed = split_seed)
  task$Xtrain <- as.matrix(task$Xtrain)
  task$Xtest <- as.matrix(task$Xtest)
  if (identical(task$task_type, "classification")) {
    task$Ytrain <- droplevels(as.factor(task$Ytrain))
    task$Ytest <- factor(task$Ytest, levels = levels(task$Ytrain))
    keep <- !is.na(task$Ytest)
    task$Xtest <- task$Xtest[keep, , drop = FALSE]
    task$Ytest <- droplevels(task$Ytest[keep])
    task$n_test <- nrow(task$Xtest)
    task$n_classes <- nlevels(task$Ytrain)
  } else {
    task$Ytrain <- as.matrix(task$Ytrain)
    task$Ytest <- as.matrix(task$Ytest)
    task$n_classes <- ncol(task$Ytrain)
  }
  task
}

combined_task_data <- function(task) {
  X <- rbind(task$Xtrain, task$Xtest)
  if (identical(task$task_type, "classification")) {
    Y <- factor(c(as.character(task$Ytrain), as.character(task$Ytest)), levels = levels(task$Ytrain))
  } else {
    Y <- rbind(as.matrix(task$Ytrain), as.matrix(task$Ytest))
  }
  list(X = X, Y = Y)
}

cv_metric <- function(cv) {
  if (!is.null(cv$best_metric_name) && length(cv$best_metric_name)) {
    return(list(metric_name = as.character(cv$best_metric_name[[1L]]),
                metric_value = as.numeric(cv$best_metric_value[[1L]])))
  }
  metrics <- cv$selection_metrics %||% cv$metrics
  if (is.data.frame(metrics) && nrow(metrics)) {
    return(list(metric_name = as.character(metrics$metric_name[[1L]]),
                metric_value = as.numeric(metrics$metric_value[[1L]])))
  }
  list(metric_name = NA_character_, metric_value = NA_real_)
}

format_pipeline3_metric <- function(value, metric_name = NA_character_) {
  if (length(value) == 0L || is.na(value) || !is.finite(value)) return("")
  metric_name <- tolower(as.character(metric_name %||% ""))
  if (metric_name %in% c("accuracy", "q2")) {
    sprintf("%.4f", value)
  } else if (identical(metric_name, "rmsd")) {
    if (abs(value) >= 100) sprintf("%.1f", value) else sprintf("%.4f", value)
  } else {
    sprintf("%.4f", value)
  }
}

format_pipeline3_time <- function(value) {
  if (length(value) == 0L || is.na(value) || !is.finite(value)) return("")
  if (value < 10) sprintf("%.3f s", value) else sprintf("%.1f s", value)
}

format_pipeline3_error <- function(status, msg) {
  status <- as.character(status %||% "")
  msg <- as.character(msg %||% "")
  if (!nzchar(status) || identical(status, "ok")) return("")
  label <- switch(
    status,
    error = "ERROR",
    killed_timeout = "TIMEOUT",
    killed_sig9 = "KILLED",
    toupper(status)
  )
  if (nzchar(msg)) paste0(label, ": ", msg) else label
}

pipeline3_function_label <- function(row) {
  parts <- c("fastPLS", row$backend[[1L]], row$svd.method[[1L]])
  if (!identical(row$classifier[[1L]], "argmax")) {
    parts <- c(parts, row$classifier[[1L]])
  }
  paste(parts, collapse = " / ")
}

write_pipeline3_wide_tables <- function(raw, results_dir, benchmark_mode = "cv10") {
  d <- raw[raw$benchmark_mode == benchmark_mode, , drop = FALSE]
  if (!nrow(d)) return(invisible(NULL))

  out_dir <- file.path(results_dir, "rearranged_tables")
  dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

  method_order <- c("plssvd", "simpls", "opls", "kernelpls")
  dataset_order <- c(
    "metref", "ccle", "tcga_brca", "tcga_hnsc_methylation",
    "gtex_v8", "tcga_pan_cancer", "singlecell", "cifar100",
    "cbmc_citeseq", "prism", "nmr", "imagenet"
  )
  dataset_order <- dataset_order[dataset_order %in% unique(d$dataset)]
  if (!length(dataset_order)) dataset_order <- sort(unique(d$dataset))

  metric_map <- do.call(rbind, lapply(dataset_order, function(ds) {
    dd <- d[d$dataset == ds, , drop = FALSE]
    task_type <- unique(dd$task_type[!is.na(dd$task_type) & nzchar(dd$task_type)])
    metric_name <- unique(dd$metric_name[!is.na(dd$metric_name) & nzchar(dd$metric_name)])
    data.frame(
      dataset = ds,
      task_type = if (length(task_type)) task_type[[1L]] else NA_character_,
      metric_name = if (length(metric_name)) metric_name[[1L]] else NA_character_,
      stringsAsFactors = FALSE
    )
  }))
  write.csv(
    metric_map,
    file.path(out_dir, sprintf("pipeline3_%s_dataset_metric_map.csv", benchmark_mode)),
    row.names = FALSE,
    na = ""
  )

  manifest <- data.frame(
    source = file.path(results_dir, "pipeline3_cv_vs_fit_raw.csv"),
    benchmark_mode = benchmark_mode,
    output_dir = out_dir,
    note = "Each method table has datasets in columns; every function/package has one accuracy/RMSD/Q2 row and one time row. Error cells retain the execution error.",
    stringsAsFactors = FALSE
  )
  write.csv(manifest, file.path(out_dir, "pipeline3_rearranged_manifest.csv"), row.names = FALSE)

  for (meth in method_order) {
    dm <- d[d$method == meth, , drop = FALSE]
    if (!nrow(dm)) next

    ids <- unique(dm$method_id)
    rows <- list()
    for (id in ids) {
      di <- dm[dm$method_id == id, , drop = FALSE]
      label <- pipeline3_function_label(di[1L, , drop = FALSE])
      metric_row <- data.frame(function_package = label, row_type = "accuracy/RMSD/Q2", stringsAsFactors = FALSE)
      time_row <- data.frame(function_package = label, row_type = "time", stringsAsFactors = FALSE)

      for (ds in dataset_order) {
        cell <- di[di$dataset == ds, , drop = FALSE]
        if (!nrow(cell)) {
          metric_row[[ds]] <- ""
          time_row[[ds]] <- ""
          next
        }
        cell <- cell[1L, , drop = FALSE]
        if (!identical(cell$status[[1L]], "ok")) {
          err <- format_pipeline3_error(cell$status[[1L]], cell$error_message[[1L]])
          metric_row[[ds]] <- err
          time_row[[ds]] <- err
        } else {
          metric_row[[ds]] <- format_pipeline3_metric(cell$metric_value[[1L]], cell$metric_name[[1L]])
          time_row[[ds]] <- format_pipeline3_time(cell$cv_time_sec[[1L]])
        }
      }

      rows[[length(rows) + 1L]] <- metric_row
      rows[[length(rows) + 1L]] <- time_row
    }

    table_out <- do.call(rbind, rows)
    csv_path <- file.path(out_dir, sprintf("pipeline3_%s_%s_wide_table.csv", meth, benchmark_mode))
    tsv_path <- file.path(out_dir, sprintf("pipeline3_%s_%s_wide_table.tsv", meth, benchmark_mode))
    write.csv(table_out, csv_path, row.names = FALSE, na = "")
    write.table(table_out, tsv_path, sep = "\t", row.names = FALSE, quote = FALSE, na = "")
  }

  invisible(out_dir)
}

run_fit_predict <- function(task, spec, effective_ncomp) {
  suppressPackageStartupMessages(library(fastPLS))
  if (identical(spec$backend, "cuda") && !isTRUE(fastPLS::has_cuda())) {
    stop("CUDA backend not available")
  }
  if (identical(spec$backend, "metal") && !isTRUE(fastPLS::has_metal())) {
    stop("Metal backend not available")
  }
  t0 <- proc.time()[["elapsed"]]
  fit <- fastPLS::pls(
    Xtrain = task$Xtrain,
    Ytrain = task$Ytrain,
    ncomp = effective_ncomp,
    method = spec$method,
    backend = spec$backend,
    svd.method = spec$svd_method,
    scaling = "centering",
    classifier = spec$classifier,
    k = cknn_k,
    tau = cknn_tau,
    alpha = cknn_alpha,
    top_m = cknn_top_m,
    north = if (identical(spec$method, "opls")) 1L else 1L,
    fit = FALSE,
    proj = FALSE,
    return_variance = FALSE,
    seed = 123L + replicate_id
  )
  t1 <- proc.time()[["elapsed"]]
  pred <- stats::predict(fit, task$Xtest, Ytest = task$Ytest)
  t2 <- proc.time()[["elapsed"]]
  met <- metric_from_pred(task$Ytest, pred, y_train = task$Ytrain)
  list(
    fit_time_sec = as.numeric(t1 - t0),
    predict_time_sec = as.numeric(t2 - t1),
    total_time_sec = as.numeric(t2 - t0),
    cv_time_sec = NA_real_,
    metric_name = met$metric_name,
    metric_value = met$metric_value
  )
}

run_cv10 <- function(task, spec, effective_ncomp) {
  suppressPackageStartupMessages(library(fastPLS))
  if (identical(spec$backend, "cuda") && !isTRUE(fastPLS::has_cuda())) {
    stop("CUDA backend not available")
  }
  if (identical(spec$backend, "metal") && !isTRUE(fastPLS::has_metal())) {
    stop("Metal backend not available")
  }
  dat <- combined_task_data(task)
  selection_metric <- if (identical(task$task_type, "classification")) {
    "accuracy"
  } else if (is.matrix(dat$Y) && ncol(dat$Y) == 1L) {
    "q2"
  } else {
    "rmsd"
  }
  t0 <- proc.time()[["elapsed"]]
  cv <- fastPLS::single.pls.cv(
    Xdata = dat$X,
    Ydata = dat$Y,
    constrain = seq_len(nrow(dat$X)),
    ncomp = effective_ncomp,
    kfold = kfold,
    method = spec$method,
    backend = spec$backend,
    svd.method = spec$svd_method,
    scaling = "centering",
    classifier = spec$classifier,
    k = cknn_k,
    tau = cknn_tau,
    alpha = cknn_alpha,
    top_m = cknn_top_m,
    north = if (identical(spec$method, "opls")) 1L else 1L,
    return_scores = FALSE,
    seed = 123L + replicate_id,
    selection_metric = selection_metric
  )
  elapsed <- proc.time()[["elapsed"]] - t0
  met <- cv_metric(cv)
  list(
    fit_time_sec = NA_real_,
    predict_time_sec = NA_real_,
    total_time_sec = NA_real_,
    cv_time_sec = as.numeric(elapsed),
    metric_name = met$metric_name,
    metric_value = met$metric_value
  )
}

summarize_results <- function(results_dir) {
  raw_path <- file.path(results_dir, "pipeline3_cv_vs_fit_raw.csv")
  if (!file.exists(raw_path)) stop("Raw results not found: ", raw_path)
  d <- read.csv(raw_path, stringsAsFactors = FALSE, check.names = FALSE)
  ok <- d[d$status == "ok", , drop = FALSE]
  if (!nrow(ok)) return(invisible(NULL))
  ok$speed_sec <- ifelse(ok$benchmark_mode == "cv10", ok$cv_time_sec, ok$total_time_sec)
  summary <- aggregate(
    cbind(speed_sec, metric_value) ~ dataset + task_type + method_id + method + backend + svd.method + classifier + benchmark_mode + ncomp_requested + effective_ncomp + metric_name,
    ok,
    median,
    na.rm = TRUE
  )
  write.csv(summary, file.path(results_dir, "pipeline3_cv_vs_fit_summary.csv"), row.names = FALSE)

  fit <- ok[ok$benchmark_mode == "fit_predict", c("dataset", "method_id", "speed_sec", "metric_value", "metric_name"), drop = FALSE]
  cv <- ok[ok$benchmark_mode == "cv10", c("dataset", "method_id", "speed_sec", "metric_value", "metric_name"), drop = FALSE]
  names(fit)[3:5] <- c("fit_predict_sec", "fit_predict_metric", "fit_metric_name")
  names(cv)[3:5] <- c("cv10_sec", "cv10_metric", "cv_metric_name")
  comp <- merge(fit, cv, by = c("dataset", "method_id"), all = TRUE)
  comp$cv_over_fit_ratio <- comp$cv10_sec / comp$fit_predict_sec
  write.csv(comp, file.path(results_dir, "pipeline3_cv_vs_fit_comparison.csv"), row.names = FALSE)
  write_pipeline3_wide_tables(d, results_dir, benchmark_mode = "cv10")

  if (requireNamespace("ggplot2", quietly = TRUE)) {
    dir.create(file.path(results_dir, "plots"), recursive = TRUE, showWarnings = FALSE)
    library(ggplot2)
    plot_ok <- ok[is.finite(ok$speed_sec) & ok$speed_sec > 0, , drop = FALSE]
    plot_ok$variant <- paste(plot_ok$backend, plot_ok$svd.method, plot_ok$classifier, sep = " / ")
    p1 <- ggplot(plot_ok, aes(x = variant, y = speed_sec, fill = benchmark_mode)) +
      geom_col(position = "dodge", width = 0.75) +
      facet_grid(dataset ~ method, scales = "free_y") +
      scale_y_log10() +
      coord_flip() +
      labs(x = NULL, y = "Elapsed time (s, log scale)", title = "Pipeline 3: fit+predict speed versus 10-fold CV speed") +
      theme_bw(base_size = 12) +
      theme(legend.position = "bottom", strip.text = element_text(face = "bold"))
    ggsave(file.path(results_dir, "plots", "pipeline3_fit_vs_cv10_time.png"), p1, width = 18, height = max(9, 0.75 * length(unique(plot_ok$dataset))), dpi = 160)

    comp_ok <- comp[is.finite(comp$cv_over_fit_ratio) & comp$cv_over_fit_ratio > 0, , drop = FALSE]
    if (nrow(comp_ok)) {
      spec <- do.call(rbind, lapply(comp_ok$method_id, function(x) as.data.frame(parse_method_id(x), stringsAsFactors = FALSE)))
      comp_ok <- cbind(comp_ok, spec)
      comp_ok$variant <- paste(comp_ok$backend, comp_ok$svd_method, comp_ok$classifier, sep = " / ")
      p2 <- ggplot(comp_ok, aes(x = variant, y = cv_over_fit_ratio, fill = backend)) +
        geom_col(width = 0.75) +
        facet_grid(dataset ~ method, scales = "free_y") +
        scale_y_log10() +
        coord_flip() +
        labs(x = NULL, y = "CV / fit+predict time ratio (log scale)", title = "Pipeline 3: 10-fold CV overhead") +
        theme_bw(base_size = 12) +
        theme(legend.position = "bottom", strip.text = element_text(face = "bold"))
      ggsave(file.path(results_dir, "plots", "pipeline3_cv10_overhead_ratio.png"), p2, width = 18, height = max(9, 0.75 * length(unique(comp_ok$dataset))), dpi = 160)
    }
  }
  invisible(summary)
}

if (identical(mode, "summarize")) {
  summarize_results(results_dir)
  quit(save = "no", status = 0)
}

spec <- parse_method_id(method_id)
task <- tryCatch(load_pipeline3_task(dataset_id, split_seed), error = function(e) e)
if (inherits(task, "error")) {
  write_row(empty_row("dataset_error", conditionMessage(task), spec = spec))
  quit(save = "no", status = 0)
}

effective_ncomp <- safe_effective_ncomp(task, ncomp_requested, method_family = spec$method)
if (!identical(task$task_type, "classification") && !identical(spec$classifier, "argmax")) {
  write_row(empty_row("skipped_classifier_regression", "lda/cknn classifiers require classification data", task, spec))
  quit(save = "no", status = 0)
}

res <- tryCatch({
  if (identical(benchmark_mode, "fit_predict")) {
    run_fit_predict(task, spec, effective_ncomp)
  } else if (identical(benchmark_mode, "cv10")) {
    run_cv10(task, spec, effective_ncomp)
  } else {
    stop("Unknown benchmark_mode: ", benchmark_mode)
  }
}, error = function(e) {
  list(error = conditionMessage(e))
})

if (!is.null(res$error)) {
  row <- empty_row("error", res$error, task, spec)
  row$effective_ncomp <- effective_ncomp
  write_row(row)
  quit(save = "no", status = 0)
}

row <- data.frame(
  dataset = dataset_id,
  task_type = task$task_type,
  dataset_path = task$dataset_path,
  split_seed = split_seed,
  n_train = task$n_train,
  n_test = task$n_test,
  n_total = task$n_train + task$n_test,
  p = task$p,
  q = task$n_classes,
  ncomp_requested = ncomp_requested,
  effective_ncomp = effective_ncomp,
  kfold = kfold,
  benchmark_mode = benchmark_mode,
  method_id = method_id,
  method = spec$method,
  backend = spec$backend,
  svd.method = spec$svd_method,
  classifier = spec$classifier,
  fit_time_sec = res$fit_time_sec,
  predict_time_sec = res$predict_time_sec,
  total_time_sec = res$total_time_sec,
  cv_time_sec = res$cv_time_sec,
  metric_name = res$metric_name,
  metric_value = res$metric_value,
  status = "ok",
  error_message = "",
  stringsAsFactors = FALSE
)
write_row(row)
