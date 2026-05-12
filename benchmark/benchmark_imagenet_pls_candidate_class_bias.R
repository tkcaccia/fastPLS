#!/usr/bin/env Rscript

suppressPackageStartupMessages({
  if (requireNamespace("data.table", quietly = TRUE)) {
    library(data.table)
  }
})

arg <- function(name, default = NULL) {
  key <- paste0("--", name, "=")
  hit <- grep(paste0("^", key), commandArgs(TRUE), value = TRUE)
  if (length(hit)) sub(key, "", hit[[length(hit)]], fixed = TRUE) else default
}

timestamp <- function() format(Sys.time(), "%Y-%m-%d %H:%M:%S")
log_msg <- function(...) message("[", timestamp(), "] ", paste0(..., collapse = ""))

parse_num <- function(x) {
  as.numeric(strsplit(x, ",", fixed = TRUE)[[1L]])
}

safe_norm <- function(x) {
  x <- sqrt(pmax(x, 0))
  x[!is.finite(x) | x == 0] <- 1
  x
}

row_l2 <- function(X) {
  X / safe_norm(rowSums(X * X))
}

.rss_state <- new.env(parent = emptyenv())
.rss_state$peak_mb <- NA_real_

rss_mb <- function() {
  path <- file.path("/proc", Sys.getpid(), "status")
  if (!file.exists(path)) return(NA_real_)
  lines <- readLines(path, warn = FALSE)
  val <- sub("^VmRSS:\\s*([0-9]+).*", "\\1", lines[grepl("^VmRSS:", lines)])
  out <- if (!length(val)) NA_real_ else as.numeric(val[[1L]]) / 1024
  if (is.finite(out) && (is.na(.rss_state$peak_mb) || out > .rss_state$peak_mb)) {
    .rss_state$peak_mb <- out
  }
  out
}

peak_rss_mb <- function() {
  rss_mb()
  .rss_state$peak_mb
}

read_int32 <- function(file, n) {
  con <- file(file, "rb")
  on.exit(close(con), add = TRUE)
  readBin(con, what = "integer", n = n, size = 4L, endian = "little")
}

read_f32_rowmajor <- function(file, n, p) {
  con <- file(file, "rb")
  on.exit(close(con), add = TRUE)
  matrix(
    readBin(con, what = "numeric", n = n * p, size = 4L, endian = "little"),
    nrow = n,
    ncol = p,
    byrow = TRUE
  )
}

storage_order <- function(file) {
  order_file <- file.path(dirname(file), "storage_order.txt")
  if (file.exists(order_file)) trimws(readLines(order_file, warn = FALSE)[1L]) else "colmajor"
}

read_f32_matrix <- function(file, n, p) {
  con <- file(file, "rb")
  on.exit(close(con), add = TRUE)
  x <- readBin(con, what = "numeric", n = n * p, size = 4L, endian = "little")
  if (identical(storage_order(file), "rowmajor")) {
    t(matrix(x, nrow = p, ncol = n))
  } else {
    matrix(x, nrow = n, ncol = p)
  }
}

read_f32_block <- function(file, n, p, start, stop, keep_cols = NULL) {
  start <- as.integer(start)
  stop <- as.integer(stop)
  if (start < 1L || stop < start || stop > n) {
    stop("Invalid f32 block: start=", start, ", stop=", stop, ", n=", n)
  }
  nr <- stop - start + 1L
  keep_cols <- keep_cols %||% seq_len(p)
  keep_cols <- as.integer(keep_cols)
  con <- file(file, "rb")
  on.exit(close(con), add = TRUE)

  if (identical(storage_order(file), "rowmajor")) {
    seek(con, where = as.double(start - 1L) * as.double(p) * 4, origin = "start", rw = "read")
    vals <- readBin(con, what = "numeric", n = nr * p, size = 4L, endian = "little")
    if (length(vals) != nr * p) stop("Short row-major read from ", file)
    X <- matrix(vals, nrow = nr, ncol = p, byrow = TRUE)
    return(X[, keep_cols, drop = FALSE])
  }

  out <- matrix(NA_real_, nrow = nr, ncol = length(keep_cols))
  for (jj in seq_along(keep_cols)) {
    col <- keep_cols[[jj]]
    seek(con, where = as.double((col - 1L) * n + (start - 1L)) * 4, origin = "start", rw = "read")
    vals <- readBin(con, what = "numeric", n = nr, size = 4L, endian = "little")
    if (length(vals) != nr) stop("Short column-major read from ", file, " column ", col)
    out[, jj] <- vals
  }
  out
}

write_i32_rowmajor <- function(x, file) {
  con <- file(file, "wb")
  on.exit(close(con), add = TRUE)
  writeBin(as.integer(t(x)), con, size = 4L, endian = "little")
}

write_i32_vector <- function(x, file) {
  con <- file(file, "wb")
  on.exit(close(con), add = TRUE)
  writeBin(as.integer(x), con, size = 4L, endian = "little")
}

append_row <- function(row, file) {
  row$peak_rss_mb <- peak_rss_mb()
  write.table(
    as.data.frame(row, stringsAsFactors = FALSE),
    file = file,
    append = file.exists(file),
    sep = ",",
    row.names = FALSE,
    col.names = !file.exists(file),
    quote = TRUE
  )
}

`%||%` <- function(x, y) {
  if (is.null(x) || length(x) == 0L || is.na(x[1L]) || !nzchar(as.character(x[1L]))) y else x
}

file_size <- function(path) {
  if (!file.exists(path)) return(NA_real_)
  as.numeric(file.info(path)$size)
}

complete_file <- function(path, bytes) {
  file.exists(path) && isTRUE(file_size(path) == as.double(bytes))
}

append_f32_rowmajor <- function(X, file) {
  con <- file(file, "ab")
  on.exit(close(con), add = TRUE)
  writeBin(as.numeric(t(X)), con, size = 4L, endian = "little")
}

copy_binary_file <- function(from, to, chunk_bytes = 64L * 1024L * 1024L) {
  input <- file(from, "rb")
  on.exit(close(input), add = TRUE)
  output <- file(to, "ab")
  on.exit(close(output), add = TRUE)
  repeat {
    raw <- readBin(input, what = "raw", n = chunk_bytes)
    if (!length(raw)) break
    writeBin(raw, output)
  }
}

write_grouped_l2_cache_stream <- function(score_file,
                                          y_train,
                                          train_n,
                                          max_p,
                                          ncomp,
                                          n_classes,
                                          out_file,
                                          offsets_file,
                                          counts_file,
                                          tmp_dir,
                                          block_size = 50000L) {
  dir.create(tmp_dir, recursive = TRUE, showWarnings = FALSE)
  unlink(file.path(tmp_dir, sprintf("class_%04d.bin", seq_len(n_classes))), force = TRUE)
  counts <- tabulate(y_train, nbins = n_classes)
  keep_cols <- seq_len(ncomp)
  starts <- seq.int(1L, train_n, by = block_size)
  for (start in starts) {
    stop <- min(train_n, start + block_size - 1L)
    rows <- start:stop
    X <- read_f32_block(score_file, train_n, max_p, start, stop, keep_cols = keep_cols)
    X <- row_l2(X)
    cls <- y_train[rows]
    for (cc in sort(unique(cls))) {
      append_f32_rowmajor(X[cls == cc, , drop = FALSE], file.path(tmp_dir, sprintf("class_%04d.bin", cc)))
    }
    rm(X)
    gc(FALSE)
    log_msg("Grouped train-score rows ", stop, "/", train_n, "; rss_mb=", round(rss_mb(), 1))
  }

  tmp_out <- paste0(out_file, ".tmp")
  unlink(tmp_out, force = TRUE)
  offsets <- integer(n_classes)
  cursor <- 0L
  for (cc in seq_len(n_classes)) {
    offsets[[cc]] <- cursor
    class_file <- file.path(tmp_dir, sprintf("class_%04d.bin", cc))
    expected <- as.double(counts[[cc]]) * as.double(ncomp) * 4
    if (!complete_file(class_file, expected)) {
      stop("Grouped class cache is incomplete for class ", cc, ": ", class_file)
    }
    copy_binary_file(class_file, tmp_out)
    cursor <- cursor + counts[[cc]]
  }
  file.rename(tmp_out, out_file)
  write_i32_vector(offsets, offsets_file)
  write_i32_vector(counts, counts_file)
  unlink(tmp_dir, recursive = TRUE, force = TRUE)
  invisible(list(offsets = offsets, counts = counts))
}

prepare_grouped_l2_cache <- function(pls_dir,
                                     train_n,
                                     max_p,
                                     ncomp,
                                     y_train,
                                     n_classes,
                                     grouped_cache_dir,
                                     block_size) {
  dir.create(grouped_cache_dir, recursive = TRUE, showWarnings = FALSE)
  train_file <- file.path(grouped_cache_dir, "Ttrain_l2_grouped_f32.bin")
  offsets_file <- file.path(grouped_cache_dir, "class_offsets_i32.bin")
  counts_file <- file.path(grouped_cache_dir, "class_counts_i32.bin")
  expected_train_bytes <- as.double(train_n) * as.double(ncomp) * 4
  expected_i32_bytes <- as.double(n_classes) * 4
  if (complete_file(train_file, expected_train_bytes) &&
      complete_file(offsets_file, expected_i32_bytes) &&
      complete_file(counts_file, expected_i32_bytes)) {
    log_msg("Reusing grouped L2 training-score cache: ", grouped_cache_dir)
    return(list(train_file = train_file, offsets_file = offsets_file, counts_file = counts_file))
  }

  log_msg("Building grouped L2 training-score cache in blocks: ", grouped_cache_dir)
  tmp_dir <- file.path(grouped_cache_dir, paste0("tmp_classes_", Sys.getpid()))
  write_grouped_l2_cache_stream(
    score_file = file.path(pls_dir, "Ttrain_f32.bin"),
    y_train = y_train,
    train_n = train_n,
    max_p = max_p,
    ncomp = ncomp,
    n_classes = n_classes,
    out_file = train_file,
    offsets_file = offsets_file,
    counts_file = counts_file,
    tmp_dir = tmp_dir,
    block_size = block_size
  )
  list(train_file = train_file, offsets_file = offsets_file, counts_file = counts_file)
}

top_idx_matrix <- function(scores, top_m) {
  t(apply(scores, 1L, function(z) order(z, decreasing = TRUE)[seq_len(top_m)]))
}

top5_full <- function(scores, y, idx, bias) {
  S <- sweep(scores[idx, , drop = FALSE], 2L, bias, "+")
  true_score <- S[cbind(seq_along(idx), y[idx])]
  rowSums(S > true_score) < 5L
}

predict_full <- function(scores, idx, bias) {
  max.col(sweep(scores[idx, , drop = FALSE], 2L, bias, "+"), ties.method = "first")
}

metrics_full <- function(scores, y, idx, bias) {
  pred <- predict_full(scores, idx, bias)
  c(
    top1_accuracy = mean(pred == y[idx]),
    top5_accuracy = mean(top5_full(scores, y, idx, bias))
  )
}

candidate_pred_hit5 <- function(S, cand, y, idx, gated, base_scores, base_bias, cand_bias) {
  base_pred <- predict_full(base_scores, idx, base_bias)
  base_hit5 <- top5_full(base_scores, y, idx, base_bias)
  pred <- base_pred
  hit5 <- base_hit5
  use <- intersect(idx, gated)
  if (length(use)) {
    pos <- match(use, idx)
    Sg <- S[use, , drop = FALSE]
    Cg <- cand[use, , drop = FALSE]
    Sg <- Sg + matrix(
      cand_bias[as.vector(t(Cg))],
      nrow = nrow(Sg),
      ncol = ncol(Sg),
      byrow = TRUE
    )
    pred[pos] <- Cg[cbind(seq_along(use), max.col(Sg, ties.method = "first"))]
    hit5[pos] <- vapply(seq_along(use), function(i) {
      y[[use[[i]]]] %in% Cg[i, order(Sg[i, ], decreasing = TRUE)[seq_len(min(5L, ncol(Cg)))]]
    }, logical(1L))
  }
  list(pred = pred, hit5 = hit5)
}

metrics_candidate <- function(S, cand, y, idx, gated, base_scores, base_bias, cand_bias) {
  ph <- candidate_pred_hit5(S, cand, y, idx, gated, base_scores, base_bias, cand_bias)
  c(top1_accuracy = mean(ph$pred == y[idx]), top5_accuracy = mean(ph$hit5))
}

stratified_split <- function(y, frac = 0.5, seed = 123L) {
  set.seed(seed)
  cal <- logical(length(y))
  for (cc in sort(unique(y))) {
    idx <- which(y == cc)
    n <- max(1L, floor(length(idx) * frac))
    cal[sample(idx, n)] <- TRUE
  }
  list(calibration = which(cal), holdout = which(!cal))
}

count_bias <- function(y_true, y_pred, n_classes, lambda, eps = 1) {
  true_counts <- tabulate(y_true, nbins = n_classes)
  pred_counts <- tabulate(y_pred, nbins = n_classes)
  bias <- lambda * log((true_counts + eps) / (pred_counts + eps))
  bias - mean(bias)
}

iter_bias_full <- function(scores, y, idx, n_classes, eta, n_iter, clip) {
  bias <- numeric(n_classes)
  for (it in seq_len(n_iter)) {
    pred <- predict_full(scores, idx, bias)
    step <- count_bias(y[idx], pred, n_classes, eta)
    bias <- pmax(pmin(bias + step, clip), -clip)
    bias <- bias - mean(bias)
  }
  bias
}

iter_bias_candidate <- function(S, cand, y, idx, gated, base_scores, base_bias,
                                n_classes, eta, n_iter, clip) {
  bias <- numeric(n_classes)
  for (it in seq_len(n_iter)) {
    pred <- candidate_pred_hit5(S, cand, y, idx, gated, base_scores, base_bias, bias)$pred
    step <- count_bias(y[idx], pred, n_classes, eta)
    bias <- pmax(pmin(bias + step, clip), -clip)
    bias <- bias - mean(bias)
  }
  bias
}

run_cuda <- function(exe, args) {
  if (!file.exists(exe)) stop("CUDA executable not found: ", exe)
  out <- system2(exe, args, stdout = TRUE, stderr = TRUE)
  log_msg(paste(out, collapse = " | "))
}

score_candidate_knn <- function(exe, x_file, train_file, offsets_file, counts_file,
                                cand_file, out_file, n_test, d, n_classes,
                                candidate_stride, top_m, knn_k, tau) {
  elapsed <- system.time({
    run_cuda(exe, c(
      x_file, train_file, offsets_file, counts_file, cand_file, out_file,
      n_test, d, n_classes, candidate_stride, top_m, knn_k, 1L, tau
    ))
  })[["elapsed"]]
  unname(elapsed)
}

pls_dir <- path.expand(arg("pls-score-dir", Sys.getenv("FASTPLS_IMAGENET_PLS_SCORE_DIR", "")))
source_run <- path.expand(arg("source-run", Sys.getenv("FASTPLS_IMAGENET_PROTOTYPE_RUN", "")))
prototype_scorer <- path.expand(arg("prototype-scorer", Sys.getenv("FASTPLS_CUDA_PROTOTYPE_SCORER", "")))
candidate_knn <- path.expand(arg("candidate-knn", Sys.getenv("FASTPLS_CUDA_CANDIDATE_KNN", "")))
out_root <- path.expand(arg("out-root", file.path(getwd(), "benchmark_results", "imagenet_candidate_class_bias")))

train_n <- as.integer(arg("train-n", "1000000"))
test_n <- as.integer(arg("test-n", "50000"))
n_classes <- as.integer(arg("n-classes", "1000"))
max_p <- as.integer(arg("max-p", "300"))
ncomp <- as.integer(arg("ncomp", "300"))
top_m <- as.integer(arg("top-m", "20"))
knn_k <- as.integer(arg("knn-k", "3"))
tau <- as.numeric(arg("tau", "0.1"))
alpha <- as.numeric(arg("alpha", "0.75"))
gate_fracs <- parse_num(arg("gate-fracs", "0.6"))
lambdas <- parse_num(arg("lambdas", "0"))
etas <- parse_num(arg("etas", "0.0025"))
iters <- as.integer(parse_num(arg("iters", "5")))
clips <- parse_num(arg("clips", "0.05"))
chunk <- as.integer(arg("chunk", "2500"))
split_seed <- as.integer(arg("split-seed", "123"))
train_block_size <- as.integer(arg("train-block-size", "50000"))
grouped_cache_dir <- path.expand(arg(
  "grouped-cache-dir",
  Sys.getenv(
    "FASTPLS_IMAGENET_GROUPED_CACHE_DIR",
    file.path(out_root, paste0("grouped_l2_ncomp", ncomp, "_train", train_n))
  )
))

if (!nzchar(pls_dir)) stop("--pls-score-dir is required")
if (!nzchar(source_run)) stop("--source-run is required")
if (!nzchar(prototype_scorer)) stop("--prototype-scorer is required")
if (!nzchar(candidate_knn)) stop("--candidate-knn is required")

run_dir <- file.path(out_root, paste0("run_", format(Sys.time(), "%Y%m%d_%H%M%S")))
dir.create(run_dir, recursive = TRUE, showWarnings = FALSE)
out_csv <- file.path(run_dir, "imagenet_pls_candidate_class_bias.csv")
best_csv <- file.path(run_dir, "imagenet_pls_candidate_class_bias_best.csv")

writeLines(c(
  paste("date", timestamp()),
  paste("pls_score_dir", pls_dir),
  paste("source_run", source_run),
  paste("prototype_scorer", prototype_scorer),
  paste("candidate_knn", candidate_knn),
  paste("train_n", train_n),
  paste("test_n", test_n),
  paste("ncomp", ncomp),
  paste("top_m", top_m),
  paste("knn_k", knn_k),
  paste("tau", tau),
  paste("alpha", alpha),
  paste("gate_fracs", paste(gate_fracs, collapse = ",")),
  paste("lambdas", paste(lambdas, collapse = ",")),
  paste("etas", paste(etas, collapse = ",")),
  paste("iters", paste(iters, collapse = ",")),
  paste("clips", paste(clips, collapse = ",")),
  paste("train_block_size", train_block_size),
  paste("grouped_cache_dir", grouped_cache_dir)
), file.path(run_dir, "parameters.txt"))

log_msg("Loading labels")
y_train <- read_int32(file.path(pls_dir, "y_train_i32.bin"), train_n) + 1L
y_test <- read_int32(file.path(pls_dir, "y_test_i32.bin"), test_n) + 1L
split <- stratified_split(y_test, frac = 0.5, seed = split_seed)
cal <- split$calibration
hold <- split$holdout
rows <- seq_len(test_n)

log_msg("Scoring prototype baseline")
x_file <- file.path(source_run, "Ttest_l2_f32.bin")
manifest <- file.path(source_run, "manifest_avg_k100.tsv")
base_scores_file <- file.path(run_dir, "base_scores_f32.bin")
base_sec <- system.time({
  run_cuda(prototype_scorer, c(x_file, manifest, base_scores_file, test_n, ncomp, n_classes, chunk))
})[["elapsed"]]
base_scores <- read_f32_rowmajor(base_scores_file, test_n, n_classes)
base0 <- numeric(n_classes)

for (sp in c("calibration", "holdout", "full")) {
  idx <- switch(sp, calibration = cal, holdout = hold, full = rows)
  m <- metrics_full(base_scores, y_test, idx, base0)
  append_row(data.frame(
    model = "prototype", bias_type = "none", split = sp,
    lambda = NA_real_, eta = NA_real_, n_iter = NA_integer_,
    clip = NA_real_, gate_fraction = 0,
    top1_accuracy = m[[1L]], top5_accuracy = m[[2L]],
    prototype_time_sec = base_sec, knn_time_sec = 0,
    total_score_time_sec = base_sec, rss_mb = rss_mb(),
    status = "success", notes = "prototype baseline"
  ), out_csv)
}

pred_base_cal <- predict_full(base_scores, cal, base0)
for (lambda in lambdas) {
  bias <- count_bias(y_test[cal], pred_base_cal, n_classes, lambda)
  for (sp in c("calibration", "holdout", "full")) {
    idx <- switch(sp, calibration = cal, holdout = hold, full = rows)
    m <- metrics_full(base_scores, y_test, idx, bias)
    append_row(data.frame(
      model = "prototype", bias_type = "count_ratio", split = sp,
      lambda = lambda, eta = NA_real_, n_iter = NA_integer_,
      clip = NA_real_, gate_fraction = 0,
      top1_accuracy = m[[1L]], top5_accuracy = m[[2L]],
      prototype_time_sec = base_sec, knn_time_sec = 0,
      total_score_time_sec = base_sec, rss_mb = rss_mb(),
      status = "success",
      notes = "bias=lambda*log(true_count/pred_count) learned on calibration split"
    ), out_csv)
  }
}

for (eta in etas) for (n_iter in iters) for (clip in clips) {
  bias <- iter_bias_full(base_scores, y_test, cal, n_classes, eta, n_iter, clip)
  for (sp in c("calibration", "holdout", "full")) {
    idx <- switch(sp, calibration = cal, holdout = hold, full = rows)
    m <- metrics_full(base_scores, y_test, idx, bias)
    append_row(data.frame(
      model = "prototype", bias_type = "iter_count_ratio", split = sp,
      lambda = NA_real_, eta = eta, n_iter = n_iter,
      clip = clip, gate_fraction = 0,
      top1_accuracy = m[[1L]], top5_accuracy = m[[2L]],
      prototype_time_sec = base_sec, knn_time_sec = 0,
      total_score_time_sec = base_sec, rss_mb = rss_mb(),
      status = "success",
      notes = "iterative class count balancing learned on calibration split"
    ), out_csv)
  }
}

log_msg("Preparing candidate local kNN scores")
cand <- top_idx_matrix(base_scores, top_m)
cand_file <- file.path(run_dir, "candidates_i32.bin")
write_i32_rowmajor(cand - 1L, cand_file)

ord2 <- t(apply(base_scores, 1L, function(z) order(z, decreasing = TRUE)[seq_len(2L)]))
margins <- base_scores[cbind(rows, ord2[, 1L])] - base_scores[cbind(rows, ord2[, 2L])]
base_cand <- matrix(
  base_scores[cbind(rep(rows, each = top_m), as.vector(t(cand)))],
  nrow = test_n,
  ncol = top_m,
  byrow = TRUE
)
S_base_part <- alpha * base_cand

cache <- prepare_grouped_l2_cache(
  pls_dir = pls_dir,
  train_n = train_n,
  max_p = max_p,
  ncomp = ncomp,
  y_train = y_train,
  n_classes = n_classes,
  grouped_cache_dir = grouped_cache_dir,
  block_size = train_block_size
)
train_file <- cache$train_file
offsets_file <- cache$offsets_file
counts_file <- cache$counts_file
gc()

log_msg("Scoring CUDA candidate kNN")
local_scores_file <- file.path(run_dir, "local_scores_f32.bin")
knn_sec <- score_candidate_knn(
  candidate_knn,
  x_file,
  train_file,
  offsets_file,
  counts_file,
  cand_file,
  local_scores_file,
  test_n,
  ncomp,
  n_classes,
  top_m,
  top_m,
  knn_k,
  tau
)
local_scores <- read_f32_rowmajor(local_scores_file, test_n, n_classes)
local_cand <- matrix(
  local_scores[cbind(rep(rows, each = top_m), as.vector(t(cand)))],
  nrow = test_n,
  ncol = top_m,
  byrow = TRUE
)
S <- local_cand + S_base_part
rm(local_scores)
gc()

for (gf in gate_fracs) {
  threshold <- as.numeric(stats::quantile(margins, probs = gf, names = FALSE, type = 7))
  gated <- which(margins <= threshold)
  zero <- numeric(n_classes)
  for (sp in c("calibration", "holdout", "full")) {
    idx <- switch(sp, calibration = cal, holdout = hold, full = rows)
    m <- metrics_candidate(S, cand, y_test, idx, gated, base_scores, zero, zero)
    append_row(data.frame(
      model = "gated_candidate_knn", bias_type = "none", split = sp,
      lambda = NA_real_, eta = NA_real_, n_iter = NA_integer_,
      clip = NA_real_, gate_fraction = gf,
      top1_accuracy = m[[1L]], top5_accuracy = m[[2L]],
      prototype_time_sec = base_sec,
      knn_time_sec = knn_sec * length(gated) / test_n,
      total_score_time_sec = base_sec + knn_sec * length(gated) / test_n,
      rss_mb = rss_mb(),
      status = "success", notes = "base gated candidate kNN"
    ), out_csv)
  }

  ph0 <- candidate_pred_hit5(S, cand, y_test, cal, gated, base_scores, zero, zero)
  for (lambda in lambdas) {
    bias <- count_bias(y_test[cal], ph0$pred, n_classes, lambda)
    for (sp in c("calibration", "holdout", "full")) {
      idx <- switch(sp, calibration = cal, holdout = hold, full = rows)
      m <- metrics_candidate(S, cand, y_test, idx, gated, base_scores, zero, bias)
      append_row(data.frame(
        model = "gated_candidate_knn", bias_type = "count_ratio", split = sp,
        lambda = lambda, eta = NA_real_, n_iter = NA_integer_,
        clip = NA_real_, gate_fraction = gf,
        top1_accuracy = m[[1L]], top5_accuracy = m[[2L]],
        prototype_time_sec = base_sec,
        knn_time_sec = knn_sec * length(gated) / test_n,
        total_score_time_sec = base_sec + knn_sec * length(gated) / test_n,
        rss_mb = rss_mb(),
        status = "success", notes = "candidate class bias learned on calibration split"
      ), out_csv)
    }
  }

  for (eta in etas) for (n_iter in iters) for (clip in clips) {
    bias <- iter_bias_candidate(S, cand, y_test, cal, gated, base_scores, zero, n_classes, eta, n_iter, clip)
    for (sp in c("calibration", "holdout", "full")) {
      idx <- switch(sp, calibration = cal, holdout = hold, full = rows)
      m <- metrics_candidate(S, cand, y_test, idx, gated, base_scores, zero, bias)
      append_row(data.frame(
        model = "gated_candidate_knn", bias_type = "iter_count_ratio", split = sp,
        lambda = NA_real_, eta = eta, n_iter = n_iter,
        clip = clip, gate_fraction = gf,
        top1_accuracy = m[[1L]], top5_accuracy = m[[2L]],
        prototype_time_sec = base_sec,
        knn_time_sec = knn_sec * length(gated) / test_n,
        total_score_time_sec = base_sec + knn_sec * length(gated) / test_n,
        rss_mb = rss_mb(),
        status = "success",
        notes = "iterative candidate class count balancing learned on calibration split"
      ), out_csv)
    }
  }
}

out <- read.csv(out_csv, stringsAsFactors = FALSE)
out$top1_accuracy <- as.numeric(out$top1_accuracy)
out$top5_accuracy <- as.numeric(out$top5_accuracy)
out <- out[order(out$split, -out$top1_accuracy, -out$top5_accuracy, out$total_score_time_sec), ]
write.csv(out, best_csv, row.names = FALSE)
writeLines(capture.output(sessionInfo()), file.path(run_dir, "sessionInfo.txt"))

if (requireNamespace("ggplot2", quietly = TRUE)) {
  suppressPackageStartupMessages(library(ggplot2))
  plot_data <- out[out$split %in% c("holdout", "full"), , drop = FALSE]
  p <- ggplot(plot_data, aes(total_score_time_sec, top1_accuracy, color = model, shape = bias_type)) +
    geom_point(size = 3) +
    facet_wrap(~ split) +
    theme_bw(base_size = 14) +
    labs(x = "scoring time (s)", y = "top-1 accuracy")
  ggsave(file.path(run_dir, "imagenet_candidate_class_bias_time_accuracy.png"), p, width = 11, height = 6.5, dpi = 160)
}

unlink(c(base_scores_file, local_scores_file, cand_file), force = TRUE)
log_msg("Finished: ", run_dir)
print(head(out, 30), row.names = FALSE)
