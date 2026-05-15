#!/usr/bin/env Rscript

fastpls_lib <- Sys.getenv("FASTPLS_LIB", "")
if (nzchar(fastpls_lib)) {
  .libPaths(c(path.expand(fastpls_lib), .libPaths()))
}
suppressPackageStartupMessages(library(fastPLS))

timestamp <- function() format(Sys.time(), "%Y-%m-%d %H:%M:%S")
message_ts <- function(...) message("[", timestamp(), "] ", paste0(..., collapse = ""))

csv_arg <- function(name, default) {
  value <- Sys.getenv(name, default)
  value <- trimws(strsplit(value, ",", fixed = TRUE)[[1L]])
  value[nzchar(value)]
}

int_env <- function(name, default) {
  value <- suppressWarnings(as.integer(Sys.getenv(name, as.character(default))))
  if (length(value) != 1L || is.na(value)) default else value
}

num_env <- function(name, default) {
  value <- suppressWarnings(as.numeric(Sys.getenv(name, as.character(default))))
  if (length(value) != 1L || is.na(value)) default else value
}

heap_mb <- function() {
  g <- gc(verbose = FALSE)
  sum(g[, "used"] * c(56, 8) / 1024^2)
}

load_cifar100 <- function(path = Sys.getenv("FASTPLS_CIFAR100_RDATA", "")) {
  candidates <- c(
    path,
    "/Users/stefano/Documents/GPUPLS/Data/CIFAR100.RData",
    "/Users/stefano/Documents/GPUPLS/Data/cifar100.RData",
    "~/Documents/Rdatasets/CIFAR100.RData",
    "~/GPUPLS/Data/CIFAR100.RData"
  )
  candidates <- path.expand(candidates[nzchar(candidates)])
  file <- candidates[file.exists(candidates)][1L]
  if (is.na(file)) {
    stop("CIFAR100 RData not found; set FASTPLS_CIFAR100_RDATA", call. = FALSE)
  }

  env <- new.env(parent = emptyenv())
  load(file, envir = env)
  if (!exists("r", envir = env, inherits = FALSE)) {
    stop("CIFAR100 RData must contain object `r`.", call. = FALSE)
  }
  r <- get("r", envir = env)
  feature_cols <- grep("^feat_", names(r), value = TRUE)
  if (!length(feature_cols)) {
    numeric_cols <- vapply(r, is.numeric, logical(1))
    feature_cols <- setdiff(names(r)[numeric_cols], c("label_idx"))
  }
  if (!length(feature_cols)) stop("No numeric CIFAR100 feature columns found.", call. = FALSE)
  label_col <- if ("label_idx" %in% names(r)) "label_idx" else {
    candidates <- grep("label|class", names(r), value = TRUE, ignore.case = TRUE)
    candidates[1L]
  }
  if (is.na(label_col) || !nzchar(label_col)) stop("No CIFAR100 label column found.", call. = FALSE)

  X <- as.matrix(data.frame(r[, feature_cols, drop = FALSE], check.names = FALSE))
  storage.mode(X) <- "double"
  keep <- colSums(is.finite(X)) > 0
  X <- X[, keep, drop = FALSE]
  X[!is.finite(X)] <- 0
  y <- factor(r[[label_col]])
  list(file = normalizePath(file), X = X, y = y)
}

balanced_subset <- function(X, y, per_class, seed = 123L) {
  if (!is.finite(per_class) || per_class <= 0L) {
    return(list(X = X, y = y, note = "full"))
  }
  set.seed(seed)
  idx <- unlist(lapply(split(seq_along(y), y), function(ii) {
    sample(ii, min(length(ii), per_class))
  }), use.names = FALSE)
  idx <- sort(idx)
  list(
    X = X[idx, , drop = FALSE],
    y = droplevels(y[idx]),
    note = paste0("balanced_", per_class, "_per_class")
  )
}

make_profiles <- function(names) {
  all <- list(
    accurate = list(oversample = 10L, power = 1L),
    balanced = list(oversample = 5L, power = 1L),
    fast0 = list(oversample = 5L, power = 0L),
    tiny0 = list(oversample = 2L, power = 0L)
  )
  missing <- setdiff(names, names(all))
  if (length(missing)) stop("Unknown profiles: ", paste(missing, collapse = ", "), call. = FALSE)
  all[names]
}

extract_accuracy <- function(out) {
  if (is.null(out$metrics)) return(NA_real_)
  ii <- which(out$metrics$metric_name == "accuracy")
  if (!length(ii)) return(NA_real_)
  as.numeric(out$metrics$metric_value[ii[1L]])
}

run_one <- function(X, y, method, classifier, backend, svd_method, profile_name,
                    profile, ncomp, kfold, seed, timeout_sec,
                    candidate_knn_k, candidate_tau, candidate_alpha,
                    candidate_top_m) {
  if (backend %in% c("cuda", "metal") && identical(svd_method, "irlba")) {
    return(data.frame(
      method = method,
      classifier = classifier,
      backend = backend,
      svd_method = svd_method,
      profile = profile_name,
      ncomp = ncomp,
      kfold = kfold,
      oversample = profile$oversample,
      power = profile$power,
      elapsed_sec = NA_real_,
      accuracy = NA_real_,
      heap_after_mb = NA_real_,
      status = "skipped",
      error = "IRLBA is CPU-only; CUDA/Metal use rSVD.",
      stringsAsFactors = FALSE
    ))
  }
  if (identical(backend, "cuda") && !isTRUE(has_cuda())) {
    return(data.frame(
      method = method,
      classifier = classifier,
      backend = backend,
      svd_method = svd_method,
      profile = profile_name,
      ncomp = ncomp,
      kfold = kfold,
      oversample = profile$oversample,
      power = profile$power,
      elapsed_sec = NA_real_,
      accuracy = NA_real_,
      heap_after_mb = NA_real_,
      status = "skipped",
      error = "CUDA backend unavailable.",
      stringsAsFactors = FALSE
    ))
  }
  if (identical(backend, "metal") && !isTRUE(has_metal())) {
    return(data.frame(
      method = method,
      classifier = classifier,
      backend = backend,
      svd_method = svd_method,
      profile = profile_name,
      ncomp = ncomp,
      kfold = kfold,
      oversample = profile$oversample,
      power = profile$power,
      elapsed_sec = NA_real_,
      accuracy = NA_real_,
      heap_after_mb = NA_real_,
      status = "skipped",
      error = "Metal backend unavailable.",
      stringsAsFactors = FALSE
    ))
  }

  gc(verbose = FALSE)
  elapsed <- NA_real_
  accuracy <- NA_real_
  status <- "ok"
  error <- NA_character_
  start <- proc.time()[["elapsed"]]
  old_limit <- setTimeLimit(elapsed = timeout_sec, transient = TRUE)
  on.exit({
    setTimeLimit(cpu = Inf, elapsed = Inf, transient = FALSE)
  }, add = TRUE)
  tryCatch({
    out <- pls.single.cv(
      X,
      y,
      ncomp = ncomp,
      kfold = kfold,
      method = method,
      backend = backend,
      svd.method = svd_method,
      classifier = classifier,
      scaling = "centering",
      seed = seed,
      oversample = profile$oversample,
      power = profile$power,
      candidate_knn_k = candidate_knn_k,
      candidate_tau = candidate_tau,
      candidate_alpha = candidate_alpha,
      candidate_top_m = candidate_top_m,
      return_scores = FALSE
    )
    accuracy <- extract_accuracy(out)
  }, error = function(e) {
    status <<- if (grepl("reached elapsed time limit", conditionMessage(e), fixed = TRUE)) {
      "timeout"
    } else {
      "error"
    }
    error <<- conditionMessage(e)
  })
  elapsed <- proc.time()[["elapsed"]] - start
  setTimeLimit(cpu = Inf, elapsed = Inf, transient = FALSE)
  data.frame(
    method = method,
    classifier = classifier,
    backend = backend,
    svd_method = svd_method,
    profile = profile_name,
    ncomp = ncomp,
    kfold = kfold,
    oversample = profile$oversample,
    power = profile$power,
    elapsed_sec = elapsed,
    accuracy = accuracy,
    heap_after_mb = heap_mb(),
    status = status,
    error = error,
    stringsAsFactors = FALSE
  )
}

summarize_results <- function(raw) {
  ok <- raw[raw$status == "ok", , drop = FALSE]
  if (!nrow(ok)) return(raw[0, , drop = FALSE])
  groups <- split(ok, interaction(ok$method, ok$classifier, ok$backend, ok$svd_method, ok$profile, drop = TRUE))
  do.call(rbind, lapply(groups, function(x) {
    data.frame(
      method = x$method[1L],
      classifier = x$classifier[1L],
      backend = x$backend[1L],
      svd_method = x$svd_method[1L],
      profile = x$profile[1L],
      ncomp = x$ncomp[1L],
      kfold = x$kfold[1L],
      oversample = x$oversample[1L],
      power = x$power[1L],
      median_elapsed_sec = median(x$elapsed_sec, na.rm = TRUE),
      median_accuracy = median(x$accuracy, na.rm = TRUE),
      median_heap_after_mb = median(x$heap_after_mb, na.rm = TRUE),
      runs = nrow(x),
      stringsAsFactors = FALSE
    )
  }))
}

write_plots <- function(summary, out_dir) {
  if (!requireNamespace("ggplot2", quietly = TRUE) || !nrow(summary)) return(invisible(FALSE))
  p1 <- ggplot2::ggplot(
    summary,
    ggplot2::aes(
      x = median_elapsed_sec,
      y = median_accuracy,
      colour = backend,
      shape = classifier
    )
  ) +
    ggplot2::geom_point(size = 3) +
    ggplot2::facet_grid(method ~ profile, scales = "free_x") +
    ggplot2::scale_x_log10() +
    ggplot2::labs(x = "Median CV time (s, log10)", y = "CV accuracy", title = "CIFAR100 CV accuracy/speed tradeoff") +
    ggplot2::theme_bw(base_size = 12)
  ggplot2::ggsave(file.path(out_dir, "cifar100_cv_accuracy_speed.png"), p1, width = 12, height = 8, dpi = 160)

  p2 <- ggplot2::ggplot(
    summary,
    ggplot2::aes(
      x = interaction(classifier, backend, sep = " / "),
      y = median_elapsed_sec,
      fill = profile
    )
  ) +
    ggplot2::geom_col(position = "dodge") +
    ggplot2::facet_wrap(~ method, scales = "free_y") +
    ggplot2::scale_y_log10() +
    ggplot2::labs(x = "Classifier / backend", y = "Median CV time (s, log10)", title = "CIFAR100 CV runtime") +
    ggplot2::theme_bw(base_size = 12) +
    ggplot2::theme(axis.text.x = ggplot2::element_text(angle = 40, hjust = 1))
  ggplot2::ggsave(file.path(out_dir, "cifar100_cv_runtime.png"), p2, width = 12, height = 8, dpi = 160)
  invisible(TRUE)
}

set.seed(123)
out_base <- Sys.getenv("FASTPLS_CIFAR100_CV_OUT", "benchmark_results/cifar100_cv_classifiers")
out_dir <- file.path(out_base, format(Sys.time(), "%Y%m%d_%H%M%S"))
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

ncomp <- int_env("FASTPLS_CIFAR100_CV_NCOMP", 50L)
kfold <- int_env("FASTPLS_CIFAR100_CV_KFOLD", 3L)
reps <- int_env("FASTPLS_CIFAR100_CV_REPS", 1L)
seed <- int_env("FASTPLS_CIFAR100_CV_SEED", 123L)
timeout_sec <- num_env("FASTPLS_CIFAR100_CV_TIMEOUT", 1800)
per_class <- int_env("FASTPLS_CIFAR100_CV_PER_CLASS", 0L)
candidate_knn_k <- int_env("FASTPLS_CIFAR100_CV_CKNN_K", 10L)
candidate_top_m <- int_env("FASTPLS_CIFAR100_CV_CKNN_TOP_M", 20L)
candidate_tau <- num_env("FASTPLS_CIFAR100_CV_CKNN_TAU", 0.2)
candidate_alpha <- num_env("FASTPLS_CIFAR100_CV_CKNN_ALPHA", 0.75)
methods <- csv_arg("FASTPLS_CIFAR100_CV_METHODS", "simpls,plssvd,opls,kernelpls")
classifiers <- csv_arg("FASTPLS_CIFAR100_CV_CLASSIFIERS", "argmax,lda,cknn")
backends <- csv_arg("FASTPLS_CIFAR100_CV_BACKENDS", "cpu")
svd_methods <- csv_arg("FASTPLS_CIFAR100_CV_SVDS", "rsvd")
profiles <- make_profiles(csv_arg("FASTPLS_CIFAR100_CV_PROFILES", "accurate,balanced,fast0"))

message_ts("Loading CIFAR100")
data <- load_cifar100()
subset <- balanced_subset(data$X, data$y, per_class = per_class, seed = seed)
X <- subset$X
y <- subset$y
rm(data)
gc(verbose = FALSE)

params <- data.frame(
  parameter = c(
    "dataset_file", "subset", "n", "p", "classes", "ncomp", "kfold", "reps",
    "methods", "classifiers", "backends", "svd_methods", "profiles", "timeout_sec",
    "candidate_knn_k", "candidate_top_m", "candidate_tau", "candidate_alpha"
  ),
  value = c(
    Sys.getenv("FASTPLS_CIFAR100_RDATA", "auto"),
    subset$note,
    nrow(X),
    ncol(X),
    nlevels(y),
    ncomp,
    kfold,
    reps,
    paste(methods, collapse = ","),
    paste(classifiers, collapse = ","),
    paste(backends, collapse = ","),
    paste(svd_methods, collapse = ","),
    paste(names(profiles), collapse = ","),
    timeout_sec,
    candidate_knn_k,
    candidate_top_m,
    candidate_tau,
    candidate_alpha
  )
)
write.csv(params, file.path(out_dir, "parameters.csv"), row.names = FALSE)
capture.output(sessionInfo(), file = file.path(out_dir, "sessionInfo.txt"))

raw_path <- file.path(out_dir, "cifar100_cv_classifiers_raw.csv")
summary_path <- file.path(out_dir, "cifar100_cv_classifiers_summary.csv")
rows <- list()
row_i <- 1L
total <- length(methods) * length(classifiers) * length(backends) * length(svd_methods) * length(profiles) * reps
run_i <- 0L

for (replicate in seq_len(reps)) {
  for (method in methods) {
    for (classifier in classifiers) {
      for (backend in backends) {
        for (svd_method in svd_methods) {
          for (profile_name in names(profiles)) {
            run_i <- run_i + 1L
            message_ts(
              "Run ", run_i, "/", total,
              " rep=", replicate,
              " method=", method,
              " classifier=", classifier,
              " backend=", backend,
              " svd=", svd_method,
              " profile=", profile_name
            )
            res <- run_one(
              X = X,
              y = y,
              method = method,
              classifier = classifier,
              backend = backend,
              svd_method = svd_method,
              profile_name = profile_name,
              profile = profiles[[profile_name]],
              ncomp = ncomp,
              kfold = kfold,
              seed = seed + replicate - 1L,
              timeout_sec = timeout_sec,
              candidate_knn_k = candidate_knn_k,
              candidate_tau = candidate_tau,
              candidate_alpha = candidate_alpha,
              candidate_top_m = candidate_top_m
            )
            res$replicate <- replicate
            res$n <- nrow(X)
            res$p <- ncol(X)
            res$classes <- nlevels(y)
            res$subset <- subset$note
            rows[[row_i]] <- res
            row_i <- row_i + 1L
            raw <- do.call(rbind, rows)
            write.csv(raw, raw_path, row.names = FALSE)
            summary <- summarize_results(raw)
            write.csv(summary, summary_path, row.names = FALSE)
            write_plots(summary, out_dir)
            if (identical(res$status[1L], "ok")) {
              message_ts(
                "  ok: time=", sprintf("%.2f", res$elapsed_sec[1L]),
                "s accuracy=", sprintf("%.4f", res$accuracy[1L])
              )
            } else {
              message_ts("  ", res$status[1L], ": ", res$error[1L])
            }
          }
        }
      }
    }
  }
}

raw <- do.call(rbind, rows)
summary <- summarize_results(raw)
write.csv(raw, raw_path, row.names = FALSE)
write.csv(summary, summary_path, row.names = FALSE)
write_plots(summary, out_dir)

message_ts("Finished. Results:")
message_ts("  raw: ", normalizePath(raw_path))
message_ts("  summary: ", normalizePath(summary_path))
if (nrow(summary)) {
  print(summary[order(-summary$median_accuracy, summary$median_elapsed_sec), ], row.names = FALSE)
}
