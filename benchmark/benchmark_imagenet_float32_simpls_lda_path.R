#!/usr/bin/env Rscript

# Fit one qualified float32 SIMPLS component path and evaluate held-out CUDA
# LDA predictions one prefix at a time. The shared fit time must not be read as
# an independent fit time for every prefix.

options(stringsAsFactors = FALSE)

env <- function(name, default = "") {
  value <- Sys.getenv(name, unset = default)
  if (nzchar(value)) value else default
}
stamp <- function(...) {
  message(
    "[", format(Sys.time(), "%Y-%m-%d %H:%M:%S"), "] ",
    paste0(..., collapse = "")
  )
}
rss_mb <- function() {
  if (!file.exists("/proc/self/status")) return(NA_real_)
  line <- grep(
    "^VmRSS:",
    readLines("/proc/self/status", warn = FALSE),
    value = TRUE
  )
  if (!length(line)) return(NA_real_)
  as.numeric(sub("^VmRSS:\\s*([0-9.]+).*", "\\1", line[[1L]])) / 1024
}
component_value <- function(x) {
  if (is.list(x) && !is.data.frame(x)) return(x[[1L]])
  if (is.data.frame(x)) return(x[[1L]])
  dims <- dim(x)
  if (length(dims) == 3L) return(x[, , 1L, drop = TRUE])
  if (length(dims) == 2L && ncol(x) == 1L) return(x[, 1L])
  x
}
classification_metrics_from_counts <- function(tab, top1_n, top5_n, total_n) {
  support <- rowSums(tab)
  predicted_n <- colSums(tab)
  recall <- ifelse(support > 0, diag(tab) / support, NA_real_)
  precision <- ifelse(predicted_n > 0, diag(tab) / predicted_n, 0)
  f1 <- ifelse(
    is.finite(precision + recall) & precision + recall > 0,
    2 * precision * recall / (precision + recall),
    0
  )
  c(
    top1_accuracy = top1_n / total_n,
    top5_accuracy = top5_n / total_n,
    balanced_accuracy = mean(recall, na.rm = TRUE),
    macro_f1 = mean(f1, na.rm = TRUE)
  )
}
subset_prefix <- function(model, k) {
  out <- model
  out$ncomp <- as.integer(k)
  out$lda$ncomp <- as.integer(k)
  out$lda$models <- model$lda$models[as.character(k)]
  out$lda$ridge <- model$lda$ridge[
    match(k, model$lda$ncomp)
  ]
  out
}

lib <- env("FASTPLS_LIB")
if (nzchar(lib)) .libPaths(unique(c(lib, .libPaths())))
suppressPackageStartupMessages(library(fastPLS))

task_file <- path.expand(env(
  "TASK_RDS",
  "~/Documents/fastpls/data/imagenet_float32_seed123_train1000000_task.rds"
))
output_csv <- path.expand(env(
  "OUTPUT_CSV",
  "imagenet_float32_simpls_lda_path.csv"
))
grid <- sort(unique(as.integer(strsplit(
  env("NCOMP_GRID", "100,200,300,400,500,600,700,800,900,1000"),
  ",",
  fixed = TRUE
)[[1L]])))
oversample <- as.integer(env("OVERSAMPLE", "20"))
power <- as.integer(env("POWER", "2"))
seed <- as.integer(env("SEED", "123"))
block_size <- as.integer(env("BLOCK_SIZE", "10000"))
archive_sha <- env("SOURCE_ARCHIVE_SHA256", "")

dir.create(dirname(output_csv), recursive = TRUE, showWarnings = FALSE)
task <- readRDS(task_file)
stopifnot(all(c("Xtrain_rds", "Ytrain", "Xtest_rds", "Ytest") %in% names(task)))
if (!isTRUE(has_cuda())) stop("CUDA backend is unavailable")

stamp("Loading stored float32 ImageNet matrices")
Xtrain <- readRDS(task$Xtrain_rds)
Xtest <- readRDS(task$Xtest_rds)
if (!inherits(Xtrain, "float32") || !inherits(Xtest, "float32")) {
  stop("This qualified path requires stored float32 matrices")
}
rss_before_fit <- rss_mb()

stamp(
  "Fitting shared SIMPLS path: ", paste(grid, collapse = ","),
  "; oversample=", oversample, "; power=", power
)
set.seed(seed)
fit_time <- unname(system.time({
  fit <- pls(
    Xtrain,
    task$Ytrain,
    ncomp = grid,
    method = "simpls",
    svd.method = "rsvd",
    backend = "cuda",
    classifier = "lda",
    scaling = "centering",
    fit = FALSE,
    return_variance = FALSE,
    oversample = oversample,
    power = power,
    seed = seed
  )
})[["elapsed"]])
internal <- attr(fit, "fastPLS_internal", exact = TRUE)
if (!identical(as.character(internal$pls_method)[1L], "simpls")) {
  stop("Requested SIMPLS but executed ", as.character(internal$pls_method)[1L])
}
if (!identical(as.character(internal$predict_backend)[1L], "float32_cuda")) {
  stop(
    "Expected float32_cuda metadata but observed ",
    as.character(internal$predict_backend)[1L]
  )
}

rows <- vector("list", length(grid))
class_levels <- levels(factor(task$Ytrain))
truth <- factor(task$Ytest, levels = class_levels)
for (i in seq_along(grid)) {
  k <- grid[[i]]
  stamp(
    "Held-out blocked CUDA LDA prediction at ncomp=", k,
    "; block_size=", block_size
  )
  prefix <- subset_prefix(fit, k)
  prediction_time <- unname(system.time({
    confusion <- matrix(
      0,
      nrow = length(class_levels),
      ncol = length(class_levels),
      dimnames = list(class_levels, class_levels)
    )
    top1_n <- 0
    top5_n <- 0
    for (start in seq.int(1L, nrow(Xtest), by = block_size)) {
      end <- min(nrow(Xtest), start + block_size - 1L)
      index <- start:end
      pred <- predict(
        prefix,
        Xtest[index, , drop = FALSE],
        top = 5L,
        top5 = TRUE
      )
      predicted <- factor(
        as.character(component_value(pred$Ypred)),
        levels = class_levels
      )
      top_labels <- as.matrix(component_value(pred$Ypred_top))
      truth_block <- truth[index]
      top1_n <- top1_n + sum(predicted == truth_block, na.rm = TRUE)
      top5_n <- top5_n + sum(vapply(seq_along(index), function(j) {
        as.character(truth_block[[j]]) %in%
          as.character(top_labels[j, seq_len(min(5L, ncol(top_labels)))])
      }, logical(1L)))
      confusion <- confusion + unclass(table(truth_block, predicted))
      rm(pred, predicted, top_labels, truth_block)
    }
  })[["elapsed"]])
  metrics <- classification_metrics_from_counts(
    confusion,
    top1_n,
    top5_n,
    length(truth)
  )
  rows[[i]] <- data.frame(
    dataset = "imagenet",
    train_n = task$n_train,
    test_n = task$n_test,
    p = task$p,
    q = task$n_classes,
    method = "simpls",
    solver = "rsvd",
    backend = "cuda",
    classifier = "lda",
    precision = "float32",
    ncomp_requested = k,
    ncomp_effective = k,
    oversample = oversample,
    power = power,
    seed = seed,
    shared_path_fit_time_sec = fit_time,
    prediction_time_sec = prediction_time,
    top1_accuracy = metrics[["top1_accuracy"]],
    top5_accuracy = metrics[["top5_accuracy"]],
    balanced_accuracy = metrics[["balanced_accuracy"]],
    macro_f1 = metrics[["macro_f1"]],
    rss_before_fit_mb = rss_before_fit,
    rss_after_path_fit_mb = rss_mb(),
    executed_estimator = "simpls",
    fit_residency = "hybrid_host_simpls_cuda_rsvd",
    prediction_residency = "host_score_projection_cuda_lda",
    model_gpu_resident = isTRUE(internal$gpu_resident),
    timing_scope = "one_shared_100_to_1000_component_fit_plus_prefix_prediction",
    prediction_block_size = block_size,
    audit_status = "exploratory_controls_recorded_not_independently_qualified",
    loaded_package_path = normalizePath(
      system.file(package = "fastPLS"),
      winslash = "/",
      mustWork = TRUE
    ),
    loaded_package_version = as.character(packageVersion("fastPLS")),
    source_archive_sha256 = archive_sha,
    status = "success",
    stringsAsFactors = FALSE
  )
  write.csv(do.call(rbind, rows[seq_len(i)]), output_csv, row.names = FALSE)
  rm(prefix, confusion)
  gc(FALSE)
}

print(do.call(rbind, rows))
