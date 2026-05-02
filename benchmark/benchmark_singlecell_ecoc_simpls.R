#!/usr/bin/env Rscript

script_arg <- grep("^--file=", commandArgs(trailingOnly = FALSE), value = TRUE)
script_file <- if (length(script_arg)) sub("^--file=", "", script_arg[[1L]]) else file.path(getwd(), "benchmark_singlecell_ecoc_simpls.R")
script_dir <- dirname(normalizePath(script_file, winslash = "/", mustWork = FALSE))
source(file.path(script_dir, "helpers_dataset_memory_compare.R"))

args <- parse_kv_args()
out_dir <- normalizePath(arg_value(args, "out_dir", default = "benchmark_results_singlecell_ecoc"), winslash = "/", mustWork = FALSE)
lib_loc <- normalizePath(arg_value(args, "lib_loc", default = Sys.getenv("FASTPLS_BENCH_LIB", .libPaths()[[1L]])), winslash = "/", mustWork = FALSE)
ncomp <- suppressWarnings(as.integer(arg_value(args, "ncomp", default = "50")))
code_dim <- suppressWarnings(as.integer(arg_value(args, "code_dim", default = "50")))
reps <- suppressWarnings(as.integer(arg_value(args, "reps", default = "1")))
split_seed <- suppressWarnings(as.integer(arg_value(args, "split_seed", default = "123")))
timeout_note <- arg_value(args, "timeout_note", default = "")

if (!is.finite(ncomp) || is.na(ncomp) || ncomp < 1L) ncomp <- 50L
if (!is.finite(code_dim) || is.na(code_dim) || code_dim < 2L) code_dim <- 50L
if (!is.finite(reps) || is.na(reps) || reps < 1L) reps <- 1L
if (!is.finite(split_seed) || is.na(split_seed)) split_seed <- 123L

dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)
suppressPackageStartupMessages(library("fastPLS", lib.loc = lib_loc, character.only = TRUE))

normalize_code <- function(codes) {
  codes <- as.matrix(codes)
  storage.mode(codes) <- "double"
  codes <- sweep(codes, 2L, colMeans(codes), "-", check.margin = FALSE)
  sds <- sqrt(colSums(codes^2))
  sds[sds == 0] <- 1
  codes <- sweep(codes, 2L, sds, "/", check.margin = FALSE)
  row_norm <- sqrt(rowSums(codes^2))
  row_norm[row_norm == 0] <- 1
  sweep(codes, 1L, row_norm, "/", check.margin = FALSE)
}

make_sylvester_hadamard <- function(n) {
  H <- matrix(1, 1L, 1L)
  while (nrow(H) < n) {
    H <- rbind(cbind(H, H), cbind(H, -H))
  }
  H
}

deduplicate_code_rows <- function(codes, seed = 123L, values = c(-1, 1)) {
  set.seed(as.integer(seed))
  seen <- character(0)
  for (i in seq_len(nrow(codes))) {
    key <- paste(sign(codes[i, ]), collapse = "")
    tries <- 0L
    while (key %in% seen && tries < 1000L) {
      codes[i, ] <- sample(values, ncol(codes), replace = TRUE)
      key <- paste(sign(codes[i, ]), collapse = "")
      tries <- tries + 1L
    }
    seen <- c(seen, key)
  }
  codes
}

make_response_code <- function(task, response_mode, code_dim = 50L, seed = 123L) {
  y <- droplevels(as.factor(task$Ytrain))
  response_mode <- tolower(response_mode)
  y <- droplevels(as.factor(y))
  classes <- levels(y)
  set.seed(as.integer(seed))

  if (response_mode %in% c("ecoc50", "random_ecoc50", "random_ecoc")) {
    codes <- matrix(sample(c(-1, 1), length(classes) * code_dim, replace = TRUE), nrow = length(classes), ncol = code_dim)
  } else if (response_mode %in% c("balanced_ecoc50", "balanced_ecoc")) {
    codes <- matrix(NA_real_, nrow = length(classes), ncol = code_dim)
    for (j in seq_len(code_dim)) {
      z <- rep(c(-1, 1), length.out = length(classes))
      codes[, j] <- sample(z, length(classes), replace = FALSE)
    }
  } else if (response_mode %in% c("hadamard_ecoc50", "hadamard_ecoc", "hadamard50")) {
    hdim <- 2L
    while (hdim < max(length(classes), code_dim + 1L)) hdim <- hdim * 2L
    H <- make_sylvester_hadamard(hdim)
    row_idx <- sample(seq_len(nrow(H)), length(classes), replace = FALSE)
    col_pool <- setdiff(seq_len(ncol(H)), 1L)
    col_idx <- sample(col_pool, code_dim, replace = FALSE)
    codes <- H[row_idx, col_idx, drop = FALSE]
  } else if (response_mode %in% c("gaussian50", "gaussian")) {
    codes <- matrix(rnorm(length(classes) * code_dim), nrow = length(classes), ncol = code_dim)
  } else if (response_mode %in% c("orthogonal_random50", "orthogonal_random", "orthogonal50")) {
    z <- matrix(rnorm(length(classes) * code_dim), nrow = length(classes), ncol = code_dim)
    codes <- qr.Q(qr(z), complete = FALSE)
  } else if (response_mode %in% c("centroid_pca50", "centroid_pca")) {
    X <- as.matrix(task$Xtrain)
    centers <- rowsum(X, y, reorder = FALSE) / as.numeric(table(y))
    centers <- scale(centers, center = TRUE, scale = FALSE)
    sv <- svd(centers, nu = 0L, nv = min(code_dim, ncol(centers)))
    codes <- centers %*% sv$v[, seq_len(min(code_dim, ncol(sv$v))), drop = FALSE]
    if (ncol(codes) < code_dim) {
      codes <- cbind(codes, matrix(0, nrow(codes), code_dim - ncol(codes)))
    }
  } else {
    stop("Unknown compressed response mode: ", response_mode)
  }

  rownames(codes) <- classes
  colnames(codes) <- sprintf("ecoc_%03d", seq_len(code_dim))
  # Remove degenerate all-equal columns; they carry no class information.
  keep <- apply(codes, 2L, function(z) length(unique(z)) > 1L)
  if (!all(keep)) {
    codes <- codes[, keep, drop = FALSE]
  }
  if (all(codes %in% c(-1, 1))) {
    codes <- deduplicate_code_rows(codes, seed = seed)
  }
  normalize_code(codes)
}

encode_ecoc <- function(y, codes) {
  y <- as.character(y)
  out <- codes[y, , drop = FALSE]
  storage.mode(out) <- "double"
  out
}

decode_ecoc <- function(pred_scores, codes, decode = c("correlation", "euclidean")) {
  decode <- match.arg(decode)
  codes <- as.matrix(codes)
  pred_dim <- dim(pred_scores)
  if (length(pred_dim) == 3L) {
    if (pred_dim[[2L]] == ncol(codes)) {
      pred_scores <- pred_scores[, , pred_dim[[3L]], drop = TRUE]
    } else if (pred_dim[[3L]] == ncol(codes)) {
      pred_scores <- pred_scores[, pred_dim[[2L]], , drop = TRUE]
    } else {
      stop("Cannot decode ECOC predictions with dimensions: ", paste(pred_dim, collapse = " x "))
    }
  }
  pred_scores <- as.matrix(pred_scores)
  if (ncol(pred_scores) != ncol(codes)) {
    stop(
      "ECOC prediction/code dimension mismatch: pred has ",
      ncol(pred_scores), " columns; code has ", ncol(codes), " columns"
    )
  }
  if (identical(decode, "correlation")) {
    pred_centered <- pred_scores - rowMeans(pred_scores)
    code_centered <- codes - rowMeans(codes)
    pred_norm <- sqrt(rowSums(pred_centered^2))
    code_norm <- sqrt(rowSums(code_centered^2))
    pred_norm[pred_norm == 0] <- 1
    code_norm[code_norm == 0] <- 1
    sim <- tcrossprod(pred_centered / pred_norm, code_centered / code_norm)
    return(factor(rownames(codes)[max.col(sim, ties.method = "first")], levels = rownames(codes)))
  }
  pred_sq <- rowSums(pred_scores^2)
  code_sq <- rowSums(codes^2)
  dist <- outer(pred_sq, code_sq, "+") - 2 * tcrossprod(pred_scores, codes)
  factor(rownames(codes)[max.col(-dist, ties.method = "first")], levels = rownames(codes))
}

macro_f1 <- function(truth, pred) {
  truth <- factor(truth)
  pred <- factor(pred, levels = levels(truth))
  f1 <- vapply(levels(truth), function(cls) {
    tp <- sum(truth == cls & pred == cls, na.rm = TRUE)
    fp <- sum(truth != cls & pred == cls, na.rm = TRUE)
    fn <- sum(truth == cls & pred != cls, na.rm = TRUE)
    denom <- 2 * tp + fp + fn
    if (denom == 0) NA_real_ else 2 * tp / denom
  }, numeric(1))
  mean(f1, na.rm = TRUE)
}

fit_predict_one <- function(task, variant, response_mode, rep_id, codes = NULL) {
  method <- "simpls"
  svd_method <- if (grepl("irlba", variant, fixed = TRUE)) "irlba" else "cpu_rsvd"
  implementation <- if (startsWith(variant, "cpp")) "Cpp" else if (startsWith(variant, "r_")) "R" else "CUDA"
  engine <- if (identical(implementation, "CUDA")) "GPU" else "CPU"

  if (!identical(response_mode, "onehot")) {
    Yfit <- encode_ecoc(task$Ytrain, codes)
  } else {
    Yfit <- task$Ytrain
  }

  fit_call <- switch(
    variant,
    cpp_rsvd = function() fastPLS::pls(task$Xtrain, Yfit, ncomp = ncomp, method = method, svd.method = "cpu_rsvd", fit = FALSE, seed = 1000L + rep_id),
    cpp_irlba = function() fastPLS::pls(task$Xtrain, Yfit, ncomp = ncomp, method = method, svd.method = "irlba", fit = FALSE, seed = 1000L + rep_id),
    r_rsvd = function() fastPLS::pls_r(task$Xtrain, Yfit, ncomp = ncomp, method = method, svd.method = "cpu_rsvd", fit = FALSE, seed = 1000L + rep_id),
    r_irlba = function() fastPLS::pls_r(task$Xtrain, Yfit, ncomp = ncomp, method = method, svd.method = "irlba", fit = FALSE, seed = 1000L + rep_id),
    cuda = function() fastPLS::simpls_gpu(task$Xtrain, Yfit, ncomp = ncomp, fit = FALSE, seed = 1000L + rep_id),
    stop("Unknown variant: ", variant)
  )

  if (identical(variant, "cuda") && !isTRUE(fastPLS::has_cuda())) {
    return(data.frame(status = "skipped_no_cuda", msg = "CUDA not available", stringsAsFactors = FALSE))
  }

  gc()
  fit_time <- system.time(model <- fit_call())[["elapsed"]] * 1000
  pred_time <- system.time(pred <- predict(model, task$Xtest, proj = FALSE))[["elapsed"]] * 1000

  if (!identical(response_mode, "onehot")) {
    pred_labels <- decode_ecoc(pred$Ypred, codes, decode = "correlation")
    truth <- factor(task$Ytest, levels = rownames(codes))
  } else {
    decoded <- metric_from_pred(task$Ytest, pred, y_train = task$Ytrain)
    pred_labels <- factor(decoded$pred, levels = levels(task$Ytest))
    truth <- task$Ytest
  }

  acc <- mean(as.character(pred_labels) == as.character(truth), na.rm = TRUE)
  data.frame(
    dataset = "singlecell",
    variant = variant,
    response_mode = response_mode,
    method = "simpls",
    implementation = implementation,
    engine = engine,
    backend = if (identical(variant, "cuda")) "gpu_native" else svd_method,
    replicate = as.integer(rep_id),
    ncomp = as.integer(ncomp),
    code_dim = if (!identical(response_mode, "onehot")) ncol(codes) else NA_integer_,
    n_train = as.integer(task$n_train),
    n_test = as.integer(task$n_test),
    p = as.integer(task$p),
    n_classes = as.integer(task$n_classes),
    fit_time_ms = as.numeric(fit_time),
    predict_time_ms = as.numeric(pred_time),
    total_time_ms = as.numeric(fit_time + pred_time),
    accuracy = as.numeric(acc),
    macro_f1 = as.numeric(macro_f1(truth, pred_labels)),
    status = "ok",
    msg = "",
    stringsAsFactors = FALSE
  )
}

task_path <- find_dataset_rdata("singlecell")
task <- as_task(task_path, dataset_id = "singlecell", split_seed = split_seed)
variants <- strsplit(arg_value(args, "variants", default = "cpp_rsvd,cpp_irlba,r_rsvd,r_irlba,cuda"), ",", fixed = TRUE)[[1L]]
variants <- trimws(variants)
response_modes <- strsplit(arg_value(args, "response_modes", default = "onehot,random_ecoc50,balanced_ecoc50,hadamard_ecoc50,gaussian50,orthogonal_random50,centroid_pca50"), ",", fixed = TRUE)[[1L]]
response_modes <- trimws(response_modes)
code_list <- setNames(vector("list", length(response_modes)), response_modes)
for (mode in response_modes) {
  if (!identical(mode, "onehot")) {
    code_list[[mode]] <- make_response_code(task, mode, code_dim = code_dim, seed = split_seed)
  }
}

manifest <- c(
  sprintf("dataset_path=%s", task$dataset_path),
  sprintf("split_seed=%s", split_seed),
  sprintf("n_train=%s", task$n_train),
  sprintf("n_test=%s", task$n_test),
  sprintf("p=%s", task$p),
  sprintf("n_classes=%s", task$n_classes),
  sprintf("ncomp=%s", ncomp),
  sprintf("requested_code_dim=%s", code_dim),
  sprintf(
    "effective_code_dims=%s",
    paste(vapply(response_modes, function(mode) {
      if (identical(mode, "onehot")) return("onehot=NA")
      sprintf("%s=%s", mode, ncol(code_list[[mode]]))
    }, character(1)), collapse = ",")
  ),
  sprintf("reps=%s", reps),
  sprintf("variants=%s", paste(variants, collapse = ",")),
  sprintf("response_modes=%s", paste(response_modes, collapse = ",")),
  sprintf("lib_loc=%s", lib_loc),
  sprintf("timeout_note=%s", timeout_note)
)
writeLines(manifest, file.path(out_dir, "singlecell_ecoc_manifest.txt"))
saveRDS(
  list(
    task_meta = task[c("dataset", "task_type", "dataset_path", "split_seed", "n_train", "n_test", "p", "n_classes")],
    codes = code_list
  ),
  file.path(out_dir, "singlecell_ecoc_codes.rds")
)

rows <- list()
idx <- 1L
for (variant in variants) {
  for (response_mode in response_modes) {
    codes <- code_list[[response_mode]]
    for (rep_id in seq_len(reps)) {
      message(sprintf("[RUN] variant=%s response=%s rep=%s", variant, response_mode, rep_id))
      rows[[idx]] <- tryCatch(
        fit_predict_one(task, variant, response_mode, rep_id, codes = codes),
        error = function(e) {
          data.frame(
            dataset = "singlecell", variant = variant, response_mode = response_mode,
            method = "simpls", implementation = NA_character_, engine = NA_character_,
            backend = NA_character_, replicate = as.integer(rep_id), ncomp = as.integer(ncomp),
            code_dim = if (!identical(response_mode, "onehot") && !is.null(codes)) ncol(codes) else NA_integer_,
            n_train = as.integer(task$n_train), n_test = as.integer(task$n_test), p = as.integer(task$p),
            n_classes = as.integer(task$n_classes), fit_time_ms = NA_real_, predict_time_ms = NA_real_,
            total_time_ms = NA_real_, accuracy = NA_real_, macro_f1 = NA_real_,
            status = "error", msg = conditionMessage(e), stringsAsFactors = FALSE
          )
        }
      )
      utils::write.csv(do.call(rbind, rows), file.path(out_dir, "singlecell_ecoc_simpls_raw.csv"), row.names = FALSE)
      idx <- idx + 1L
    }
  }
}

raw <- do.call(rbind, rows)
ok_raw <- raw[raw$status == "ok", , drop = FALSE]
summary <- if (nrow(ok_raw)) {
  aggregate(
    cbind(fit_time_ms, predict_time_ms, total_time_ms, accuracy, macro_f1) ~ variant + response_mode + implementation + engine + backend + status,
    ok_raw,
    function(x) median(x, na.rm = TRUE)
  )
} else {
  raw[0, c("variant", "response_mode", "implementation", "engine", "backend", "status",
           "fit_time_ms", "predict_time_ms", "total_time_ms", "accuracy", "macro_f1"), drop = FALSE]
}
utils::write.csv(summary, file.path(out_dir, "singlecell_ecoc_simpls_summary.csv"), row.names = FALSE)
print(summary)
