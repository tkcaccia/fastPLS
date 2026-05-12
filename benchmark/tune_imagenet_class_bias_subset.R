#!/usr/bin/env Rscript

suppressPackageStartupMessages({
  library(fastPLS)
})

`%||%` <- function(x, y) if (is.null(x) || length(x) == 0L || is.na(x[1L]) || !nzchar(as.character(x[1L]))) y else x

cmd_args_full <- commandArgs(FALSE)
file_arg <- sub("^--file=", "", cmd_args_full[grep("^--file=", cmd_args_full)][1L])
script_dir <- dirname(normalizePath(file_arg %||% "benchmark/tune_imagenet_class_bias_subset.R", mustWork = FALSE))
if (!nzchar(script_dir) || is.na(script_dir)) {
  script_dir <- getwd()
}
helpers <- file.path(script_dir, "helpers_dataset_memory_compare.R")
if (!file.exists(helpers)) {
  helpers <- file.path(getwd(), "benchmark", "helpers_dataset_memory_compare.R")
}
source(helpers)

csv_vec <- function(x, default, type = c("character", "integer", "numeric")) {
  type <- match.arg(type)
  x <- x %||% paste(default, collapse = ",")
  out <- trimws(strsplit(x, ",", fixed = TRUE)[[1L]])
  out <- out[nzchar(out)]
  switch(
    type,
    character = out,
    integer = as.integer(out),
    numeric = as.numeric(out)
  )
}

as_bool <- function(x, default = FALSE) {
  if (is.null(x)) return(default)
  tolower(as.character(x)[1L]) %in% c("1", "true", "t", "yes", "y")
}

timestamp <- function() format(Sys.time(), "%Y-%m-%d %H:%M:%S")
log_msg <- function(...) message("[", timestamp(), "] ", paste0(..., collapse = ""))

top1_accuracy <- function(pred, truth) {
  mean(as.character(pred) == as.character(truth), na.rm = TRUE)
}

topk_accuracy <- function(top_mat, truth) {
  truth_chr <- as.character(truth)
  mean(vapply(seq_along(truth_chr), function(i) truth_chr[[i]] %in% as.character(top_mat[i, ]), logical(1)), na.rm = TRUE)
}

fit_peak_mb <- function(expr) {
  gc()
  before <- sum(gc()[, "used"])
  value <- force(expr)
  after <- sum(gc()[, "used"])
  list(value = value, delta_gc_cells = after - before)
}

args <- parse_kv_args()
out_root <- normalize_path_if_exists(arg_value(args, "out_dir", ""))
if (!nzchar(out_root)) {
  out_root <- file.path(
    getwd(),
    "benchmark_results",
    "imagenet_class_bias_tuning",
    format(Sys.time(), "%Y%m%d_%H%M%S")
  )
}
dir.create(out_root, recursive = TRUE, showWarnings = FALSE)

train_n <- as.integer(arg_value(args, "train_n", Sys.getenv("FASTPLS_IMAGENET_TRAIN_N", "10000")))
test_n <- as.integer(arg_value(args, "test_n", Sys.getenv("FASTPLS_IMAGENET_TEST_N", "3000")))
split_seed <- as.integer(arg_value(args, "seed", "123"))
Sys.setenv(
  FASTPLS_IMAGENET_TRAIN_N = train_n,
  FASTPLS_IMAGENET_TEST_N = test_n
)

ncomp_grid <- csv_vec(arg_value(args, "ncomp", "100,200,300"), c(100L, 200L, 300L), "integer")
method_grid <- csv_vec(arg_value(args, "methods", "plssvd,simpls"), c("plssvd", "simpls"), "character")
backend_grid <- csv_vec(arg_value(args, "backends", "cuda"), c("cuda"), "character")
method_bias_grid <- csv_vec(arg_value(args, "class_bias_method", "iter_count_ratio,count_ratio"), c("iter_count_ratio", "count_ratio"), "character")
lambda_grid <- csv_vec(arg_value(args, "lambda", "0,0.01,0.025,0.05,0.1,0.2"), c(0, 0.01, 0.025, 0.05, 0.1, 0.2), "numeric")
iter_grid <- csv_vec(arg_value(args, "iter", "1,2"), c(1L, 2L), "integer")
clip_grid <- csv_vec(arg_value(args, "clip", "Inf,0.5,1"), c(Inf, 0.5, 1), "numeric")
eps_grid <- csv_vec(arg_value(args, "eps", "1"), 1, "numeric")
fraction_grid <- csv_vec(arg_value(args, "class_bias_fraction", "1,0.5"), c(1, 0.5), "numeric")
scaling_grid <- csv_vec(arg_value(args, "scaling", "centering"), "centering", "character")
top_k <- as.integer(arg_value(args, "top", "5"))
max_runs <- as.integer(arg_value(args, "max_runs", "0"))
return_variance <- as_bool(arg_value(args, "return_variance", "false"), FALSE)

log_msg("Output: ", out_root)
log_msg("Loading ImageNet subset: train_n=", train_n, ", test_n=", test_n, ", seed=", split_seed)
imagenet_path <- find_dataset_rdata("imagenet")
task <- as_task(imagenet_path, "imagenet", split_seed = split_seed)
log_msg("Loaded ", task$n_train, " train x ", task$p, " features; ", task$n_test, " test; classes=", task$n_classes)

params <- expand.grid(
  method = method_grid,
  backend = backend_grid,
  ncomp = ncomp_grid,
  scaling = scaling_grid,
  class_bias_method = method_bias_grid,
  class_bias_lambda = lambda_grid,
  class_bias_iter = iter_grid,
  class_bias_clip = clip_grid,
  class_bias_eps = eps_grid,
  class_bias_calibration_fraction = fraction_grid,
  stringsAsFactors = FALSE
)
params <- params[order(params$method, params$backend, params$ncomp, params$class_bias_method, params$class_bias_lambda, params$class_bias_iter, params$class_bias_clip, params$class_bias_calibration_fraction), ]
if (max_runs > 0L && nrow(params) > max_runs) {
  params <- params[seq_len(max_runs), , drop = FALSE]
}

write.csv(params, file.path(out_root, "imagenet_class_bias_tuning_grid.csv"), row.names = FALSE)
writeLines(capture.output(sessionInfo()), file.path(out_root, "session_info.txt"))
writeLines(c(
  paste("imagenet_path=", imagenet_path, sep = ""),
  paste("train_n=", task$n_train, sep = ""),
  paste("test_n=", task$n_test, sep = ""),
  paste("p=", task$p, sep = ""),
  paste("n_classes=", task$n_classes, sep = ""),
  paste("top_k=", top_k, sep = ""),
  paste("return_variance=", return_variance, sep = "")
), file.path(out_root, "parameters.txt"))

raw_file <- file.path(out_root, "imagenet_class_bias_tuning_raw.csv")
rows <- vector("list", nrow(params))

for (i in seq_len(nrow(params))) {
  p <- params[i, ]
  log_msg(
    "Run ", i, "/", nrow(params),
    ": method=", p$method,
    ", backend=", p$backend,
    ", ncomp=", p$ncomp,
    ", class_bias_method=", p$class_bias_method,
    ", lambda=", p$class_bias_lambda,
    ", iter=", p$class_bias_iter,
    ", clip=", p$class_bias_clip,
    ", eps=", p$class_bias_eps,
    ", fraction=", p$class_bias_calibration_fraction
  )
  row <- data.frame(
    status = "ok",
    method = p$method,
    backend = p$backend,
    ncomp = p$ncomp,
    scaling = p$scaling,
    classifier = "class_bias_cuda",
    class_bias_method = p$class_bias_method,
    class_bias_lambda = p$class_bias_lambda,
    class_bias_iter = p$class_bias_iter,
    class_bias_clip = p$class_bias_clip,
    class_bias_eps = p$class_bias_eps,
    class_bias_calibration_fraction = p$class_bias_calibration_fraction,
    train_n = task$n_train,
    test_n = task$n_test,
    p = task$p,
    n_classes = task$n_classes,
    fit_time_sec = NA_real_,
    predict_time_sec = NA_real_,
    total_time_sec = NA_real_,
    top1_accuracy = NA_real_,
    top5_accuracy = NA_real_,
    peak_host_rss_mb = NA_real_,
    notes = "",
    stringsAsFactors = FALSE
  )
  t0 <- proc.time()[["elapsed"]]
  res <- tryCatch({
    gc()
    fit_t <- system.time({
      model <- fastPLS::pls(
        task$Xtrain,
        task$Ytrain,
        ncomp = p$ncomp,
        method = p$method,
        backend = p$backend,
        svd.method = "cpu_rsvd",
        scaling = p$scaling,
        classifier = "class_bias_cuda",
        class_bias_method = p$class_bias_method,
        class_bias_lambda = p$class_bias_lambda,
        class_bias_iter = p$class_bias_iter,
        class_bias_clip = p$class_bias_clip,
        class_bias_eps = p$class_bias_eps,
        class_bias_calibration_fraction = p$class_bias_calibration_fraction,
        return_variance = return_variance,
        fit = FALSE,
        proj = FALSE
      )
    })
    pred_t <- system.time({
      pred <- predict(model, task$Xtest, top = top_k)
    })
    yhat <- pred$Ypred[[1L]]
    acc1 <- top1_accuracy(yhat, task$Ytest)
    acc5 <- NA_real_
    if (!is.null(pred$Ypred_top) && length(pred$Ypred_top)) {
      acc5 <- topk_accuracy(pred$Ypred_top[[1L]], task$Ytest)
    }
    list(fit_time = fit_t[["elapsed"]], pred_time = pred_t[["elapsed"]], top1 = acc1, top5 = acc5)
  }, error = function(e) {
    row$status <<- "error"
    row$notes <<- conditionMessage(e)
    NULL
  })
  row$total_time_sec <- proc.time()[["elapsed"]] - t0
  if (!is.null(res)) {
    row$fit_time_sec <- res$fit_time
    row$predict_time_sec <- res$pred_time
    row$top1_accuracy <- res$top1
    row$top5_accuracy <- res$top5
  }
  rows[[i]] <- row
  out <- do.call(rbind, rows[!vapply(rows, is.null, logical(1))])
  write.csv(out, raw_file, row.names = FALSE)
}

raw <- read.csv(raw_file, stringsAsFactors = FALSE)
best <- raw[order(-raw$top1_accuracy, -raw$top5_accuracy, raw$total_time_sec, na.last = TRUE), ]
write.csv(best, file.path(out_root, "imagenet_class_bias_tuning_best.csv"), row.names = FALSE)

if (requireNamespace("ggplot2", quietly = TRUE)) {
  library(ggplot2)
  ok <- raw[raw$status == "ok", , drop = FALSE]
  if (nrow(ok)) {
    ok$clip_label <- ifelse(is.infinite(ok$class_bias_clip), "Inf", as.character(ok$class_bias_clip))
    p1 <- ggplot(ok, aes(class_bias_lambda, top1_accuracy, color = factor(class_bias_iter), linetype = clip_label)) +
      geom_line() + geom_point(size = 2) +
      facet_grid(class_bias_method + method ~ ncomp, scales = "free_y") +
      theme_bw(base_size = 13) +
      labs(x = "class_bias_lambda", y = "top-1 accuracy", color = "iter", linetype = "clip")
    ggsave(file.path(out_root, "imagenet_class_bias_top1.png"), p1, width = 12, height = 7, dpi = 150)
    p2 <- ggplot(ok, aes(total_time_sec, top1_accuracy, color = factor(class_bias_iter), shape = method)) +
      geom_point(size = 2.5) +
      facet_wrap(~ ncomp) +
      theme_bw(base_size = 13) +
      labs(x = "total time (s)", y = "top-1 accuracy", color = "iter")
    ggsave(file.path(out_root, "imagenet_class_bias_time_accuracy.png"), p2, width = 11, height = 6, dpi = 150)
  }
}

log_msg("Top configurations:")
print(utils::head(best[, c("method", "backend", "ncomp", "class_bias_method", "class_bias_lambda", "class_bias_iter", "class_bias_clip", "class_bias_eps", "class_bias_calibration_fraction", "top1_accuracy", "top5_accuracy", "total_time_sec", "status", "notes")], 10), row.names = FALSE)
log_msg("Done: ", out_root)
