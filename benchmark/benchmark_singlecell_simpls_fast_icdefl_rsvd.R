#!/usr/bin/env Rscript

suppressPackageStartupMessages({
  library(fastPLS)
})

`%||%` <- function(x, y) if (is.null(x) || !length(x) || is.na(x) || !nzchar(x)) y else x

script_arg <- grep("^--file=", commandArgs(trailingOnly = FALSE), value = TRUE)
script_file <- if (length(script_arg)) sub("^--file=", "", script_arg[[1]]) else file.path(getwd(), "benchmark", "benchmark_singlecell_simpls_fast_icdefl_rsvd.R")
script_dir <- dirname(normalizePath(script_file, winslash = "/", mustWork = FALSE))
repo_root <- normalizePath(file.path(script_dir, ".."), winslash = "/", mustWork = FALSE)

find_data_root <- function() {
  candidates <- c(
    Sys.getenv("FASTPLS_DATA_ROOT", unset = NA_character_),
    file.path(repo_root, "Rdataset"),
    file.path(dirname(repo_root), "GPUPLS", "Rdataset"),
    "/Users/stefano/HPC-firenze/image_analysis/dinoV2/Rdatasets"
  )
  candidates <- unique(candidates[!is.na(candidates) & nzchar(candidates)])
  for (path in candidates) {
    if (file.exists(file.path(path, "singlecell.RData"))) return(path)
  }
  stop("singlecell.RData not found. Set FASTPLS_DATA_ROOT to the folder containing the dataset.")
}

stratified_half_split <- function(y) {
  idx_by <- split(seq_along(y), y)
  test_idx <- unlist(lapply(idx_by, function(ix) {
    n <- length(ix)
    if (n <= 1L) return(integer(0))
    sample(ix, size = max(1L, floor(n / 2L)))
  }), use.names = FALSE)
  test_idx <- sort(unique(test_idx))
  list(train = setdiff(seq_along(y), test_idx), test = test_idx)
}

metric_accuracy <- function(truth, pred) {
  mean(as.character(pred) == as.character(truth), na.rm = TRUE)
}

with_fast_flag <- function(value, code) {
  old <- Sys.getenv("FASTPLS_FAST_OPTIMIZED", unset = NA_character_)
  on.exit({
    if (length(old) != 1L || is.na(old)) {
      Sys.unsetenv("FASTPLS_FAST_OPTIMIZED")
    } else {
      Sys.setenv(FASTPLS_FAST_OPTIMIZED = old)
    }
  }, add = TRUE)
  Sys.setenv(FASTPLS_FAST_OPTIMIZED = as.character(as.integer(value)))
  force(code)
}

load_singlecell <- function(data_root) {
  e <- new.env(parent = emptyenv())
  load(file.path(data_root, "singlecell.RData"), envir = e)
  X <- as.matrix(e$data)
  storage.mode(X) <- "double"
  y <- as.factor(e$labels)
  list(X = X, y = y)
}

plot_metric <- function(summary_df, metric_col, ylab, file) {
  methods <- unique(summary_df$method)
  cols <- c("#1b9e77", "#d95f02", "#7570b3", "#e7298a")
  png(file, width = 1400, height = 900, res = 160)
  on.exit(dev.off(), add = TRUE)
  plot(
    NA, NA,
    xlim = range(summary_df$ncomp, na.rm = TRUE),
    ylim = range(summary_df[[metric_col]], na.rm = TRUE),
    xlab = "Requested components",
    ylab = ylab,
    main = paste("singlecell:", ylab, "baseline vs optimized")
  )
  for (i in seq_along(methods)) {
    m <- methods[[i]]
    sub <- summary_df[summary_df$method == m, , drop = FALSE]
    ord <- order(sub$ncomp)
    lines(sub$ncomp[ord], sub[[metric_col]][ord], type = "b", lwd = 2, pch = 19, col = cols[[i]])
  }
  legend("topright", legend = methods, col = cols[seq_along(methods)], lwd = 2, pch = 19, bty = "n")
}

set.seed(as.integer(Sys.getenv("FASTPLS_SEED", "20260321")))
data_root <- find_data_root()
out_dir <- normalizePath(
  Sys.getenv("FASTPLS_SINGLECELL_BENCH_OUTDIR", file.path(dirname(repo_root), "fastPLS_benchmark_results", "singlecell_simpls_fast_icdefl_rsvd")),
  winslash = "/",
  mustWork = FALSE
)
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

ncomp_list <- as.integer(strsplit(Sys.getenv("FASTPLS_SINGLECELL_NCOMP_LIST", "2,5,10,20,50"), ",", fixed = TRUE)[[1]])
ncomp_list <- sort(unique(ncomp_list[is.finite(ncomp_list) & ncomp_list >= 1L]))
if (!length(ncomp_list)) ncomp_list <- c(2L, 5L, 10L, 20L, 50L)
reps <- as.integer(Sys.getenv("FASTPLS_SINGLECELL_REPS", "3"))
if (!is.finite(reps) || is.na(reps) || reps < 1L) reps <- 3L

ds <- load_singlecell(data_root)
methods <- data.frame(
  method = c("simpls_fast_icdefl_baseline", "simpls_fast_icdefl_optimized"),
  fast_flag = c(0L, 1L),
  stringsAsFactors = FALSE
)

results <- vector("list", length = reps * length(ncomp_list) * nrow(methods))
idx_out <- 0L

for (rep_id in seq_len(reps)) {
  set.seed(1000L + rep_id)
  split <- stratified_half_split(ds$y)
  Xtrain <- ds$X[split$train, , drop = FALSE]
  Ytrain <- ds$y[split$train]
  Xtest <- ds$X[split$test, , drop = FALSE]
  Ytest <- ds$y[split$test]

  for (ncomp in ncomp_list) {
    for (i in seq_len(nrow(methods))) {
      cfg <- methods[i, , drop = FALSE]
      message(sprintf("[singlecell] rep=%d ncomp=%d method=%s", rep_id, ncomp, cfg$method))
      fit_res <- tryCatch(
        with_fast_flag(cfg$fast_flag, {
          elapsed <- system.time({
            model <- pls(
              Xtrain = Xtrain,
              Ytrain = Ytrain,
              ncomp = as.integer(ncomp),
              method = "simpls_fast",
              svd.method = "cpu_rsvd",
              scaling = "centering",
              seed = 5000L + rep_id,
              rsvd_oversample = 10L,
              rsvd_power = 1L,
              fast_incremental = TRUE,
              fast_inc_iters = 2L,
              fast_defl_cache = TRUE,
              fast_center_t = FALSE,
              fast_reorth_v = FALSE,
              fast_block = 8L
            )
          })["elapsed"]
          pred <- predict(model, newdata = Xtest, Ytest = Ytest, proj = FALSE)
          list(
            elapsed = as.numeric(elapsed),
            accuracy = metric_accuracy(Ytest, pred$Ypred[[1]]),
            effective_ncomp = max(model$ncomp),
            status = "ok",
            msg = ""
          )
        }),
        error = function(e) list(
          elapsed = NA_real_,
          accuracy = NA_real_,
          effective_ncomp = NA_integer_,
          status = "error",
          msg = conditionMessage(e)
        )
      )

      idx_out <- idx_out + 1L
      results[[idx_out]] <- data.frame(
        method = cfg$method,
        dataset = "singlecell",
        rep = rep_id,
        ncomp = as.integer(ncomp),
        elapsed_time_seconds = fit_res$elapsed,
        accuracy_metric_name = "accuracy",
        accuracy_value = fit_res$accuracy,
        effective_ncomp = fit_res$effective_ncomp,
        status = fit_res$status,
        msg = fit_res$msg,
        stringsAsFactors = FALSE
      )
    }
  }
}

res <- do.call(rbind, results)
write.csv(res, file.path(out_dir, "singlecell_simpls_fast_icdefl_rsvd_replicates.csv"), row.names = FALSE)

ok <- res[res$status == "ok", , drop = FALSE]
if (!nrow(ok)) {
  stop("All benchmark runs failed. See replicates CSV for details.")
}

summary_df <- aggregate(
  cbind(elapsed_time_seconds, accuracy_value, effective_ncomp) ~ method + dataset + ncomp + accuracy_metric_name,
  data = ok,
  FUN = median
)
colnames(summary_df)[colnames(summary_df) == "elapsed_time_seconds"] <- "elapsed_time_seconds_median"
colnames(summary_df)[colnames(summary_df) == "accuracy_value"] <- "accuracy_value_median"
colnames(summary_df)[colnames(summary_df) == "effective_ncomp"] <- "effective_ncomp_median"

baseline_ref <- summary_df[summary_df$method == "simpls_fast_icdefl_baseline", c("ncomp", "elapsed_time_seconds_median", "accuracy_value_median")]
colnames(baseline_ref) <- c("ncomp", "baseline_elapsed", "baseline_accuracy")
summary_df <- merge(summary_df, baseline_ref, by = "ncomp", all.x = TRUE, sort = TRUE)
summary_df$percent_accuracy_change_vs_baseline <- 100 * (summary_df$accuracy_value_median - summary_df$baseline_accuracy) / pmax(abs(summary_df$baseline_accuracy), .Machine$double.eps)
summary_df$speedup_vs_baseline <- summary_df$baseline_elapsed / summary_df$elapsed_time_seconds_median
summary_df$passed_1pct_accuracy_constraint <- summary_df$percent_accuracy_change_vs_baseline >= -1

write.csv(summary_df, file.path(out_dir, "singlecell_simpls_fast_icdefl_rsvd_summary.csv"), row.names = FALSE)

plot_metric(
  summary_df,
  metric_col = "elapsed_time_seconds_median",
  ylab = "Median build time (seconds)",
  file = file.path(out_dir, "singlecell_simpls_fast_icdefl_rsvd_time.png")
)
plot_metric(
  summary_df,
  metric_col = "accuracy_value_median",
  ylab = "Median accuracy",
  file = file.path(out_dir, "singlecell_simpls_fast_icdefl_rsvd_accuracy.png")
)

summary_txt <- file.path(out_dir, "singlecell_simpls_fast_icdefl_rsvd_summary.txt")
sink(summary_txt)
cat("singlecell simpls_fast (icdefl, cpu_rsvd) baseline vs optimized\n\n")
print(summary_df[order(summary_df$ncomp, summary_df$method), ], row.names = FALSE)
cat("\nSession info\n")
print(sessionInfo())
sink()

writeLines(capture.output(sessionInfo()), file.path(out_dir, "sessionInfo.txt"))
message("Benchmark complete. Results written to: ", out_dir)
