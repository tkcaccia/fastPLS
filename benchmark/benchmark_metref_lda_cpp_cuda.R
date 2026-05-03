#!/usr/bin/env Rscript

options(stringsAsFactors = FALSE)

arg <- function(name, default) {
  key <- paste0("--", name, "=")
  hit <- commandArgs(trailingOnly = TRUE)
  hit <- hit[startsWith(hit, key)]
  if (!length(hit)) return(default)
  sub(key, "", hit[[1L]], fixed = TRUE)
}

ncomp <- as.integer(arg("ncomp", Sys.getenv("FASTPLS_NCOMP", "22")))
if (!is.finite(ncomp) || is.na(ncomp) || ncomp < 1L) ncomp <- 22L
reps <- as.integer(arg("reps", Sys.getenv("FASTPLS_REPS", "3")))
if (!is.finite(reps) || is.na(reps) || reps < 1L) reps <- 3L
outfile <- arg("out", Sys.getenv("FASTPLS_OUTFILE", "metref_fastpls_lda_cpp_cuda_ncomp22.csv"))

suppressPackageStartupMessages(library(fastPLS))

load_metref_task <- function() {
  candidates <- unique(Filter(nzchar, c(
    Sys.getenv("FASTPLS_METREF_RDATA", ""),
    "/home/chiamaka/Documents/fastpls/data/metref.RData",
    "/home/chiamaka/Documents/fastpls/data/remote_fastpls_data/metref.RData",
    "/Users/stefano/Documents/GPUPLS/Data/metref_remote_task.RData",
    "/Users/stefano/Documents/GPUPLS/remote_fastpls_data/metref.RData"
  )))
  candidates <- candidates[file.exists(candidates)]
  if (!length(candidates)) {
    stop("Could not find a MetRef task RData file; set FASTPLS_METREF_RDATA.", call. = FALSE)
  }
  e <- new.env(parent = emptyenv())
  load(candidates[[1L]], envir = e)
  if (exists("out", e) && is.list(e$out) &&
      all(c("Xtrain", "Ytrain", "Xtest", "Ytest") %in% names(e$out))) {
    return(list(
      path = normalizePath(candidates[[1L]], winslash = "/", mustWork = TRUE),
      Xtrain = as.matrix(e$out$Xtrain),
      Ytrain = droplevels(factor(e$out$Ytrain)),
      Xtest = as.matrix(e$out$Xtest),
      Ytest = factor(e$out$Ytest, levels = levels(droplevels(factor(e$out$Ytrain))))
    ))
  }
  if (all(c("Xtrain", "Ytrain", "Xtest", "Ytest") %in% ls(e))) {
    return(list(
      path = normalizePath(candidates[[1L]], winslash = "/", mustWork = TRUE),
      Xtrain = as.matrix(e$Xtrain),
      Ytrain = droplevels(factor(e$Ytrain)),
      Xtest = as.matrix(e$Xtest),
      Ytest = factor(e$Ytest, levels = levels(droplevels(factor(e$Ytrain))))
    ))
  }
  stop("MetRef RData did not contain out$Xtrain/out$Ytrain/out$Xtest/out$Ytest.", call. = FALSE)
}

task <- load_metref_task()
Xtrain <- task$Xtrain
Ytrain <- task$Ytrain
Xtest <- task$Xtest
Ytest <- task$Ytest

accuracy <- function(pred) {
  mean(as.character(pred) == as.character(Ytest), na.rm = TRUE)
}

variants <- list(
  list(name = "cpp_plssvd_argmax", method = "plssvd", backend = "cpp", classifier = "argmax"),
  list(name = "cpp_plssvd_lda_cpp", method = "plssvd", backend = "cpp", classifier = "lda_cpp"),
  list(name = "cpp_simpls_argmax", method = "simpls", backend = "cpp", classifier = "argmax"),
  list(name = "cpp_simpls_lda_cpp", method = "simpls", backend = "cpp", classifier = "lda_cpp")
)
if (isTRUE(has_cuda())) {
  variants <- c(variants, list(
    list(name = "cuda_plssvd_argmax", method = "plssvd", backend = "cuda", classifier = "argmax"),
    list(name = "cuda_plssvd_lda_cuda", method = "plssvd", backend = "cuda", classifier = "lda_cuda"),
    list(name = "cuda_simpls_argmax", method = "simpls", backend = "cuda", classifier = "argmax"),
    list(name = "cuda_simpls_lda_cuda", method = "simpls", backend = "cuda", classifier = "lda_cuda")
  ))
}

rows <- list()
idx <- 1L
for (v in variants) {
  for (rep in seq_len(reps)) {
    gc()
    status <- "ok"
    err <- NA_character_
    fit <- NULL
    elapsed <- system.time({
      fit <- tryCatch(
        {
          args <- list(
            Xtrain = Xtrain,
            Ytrain = Ytrain,
            Xtest = Xtest,
            Ytest = Ytest,
            ncomp = ncomp,
            method = v$method,
            backend = v$backend,
            classifier = v$classifier,
            fit = FALSE,
            proj = FALSE
          )
          if (!identical(v$backend, "cuda")) args$svd.method <- "cpu_rsvd"
          do.call(fastPLS::pls, args)
        },
        error = function(e) {
          status <<- "error"
          err <<- conditionMessage(e)
          NULL
        }
      )
    })
    rows[[idx]] <- data.frame(
      dataset = "MetRef",
      dataset_path = task$path,
      variant = v$name,
      method = v$method,
      backend = v$backend,
      classifier = v$classifier,
      ncomp = ncomp,
      replicate = rep,
      elapsed_ms = unname(elapsed[["elapsed"]]) * 1000,
      accuracy = if (is.null(fit)) NA_real_ else accuracy(fit$Ypred[[1L]]),
      classification_rule = if (is.null(fit) || is.null(fit$classification_rule)) NA_character_ else fit$classification_rule,
      has_cuda = isTRUE(has_cuda()),
      status = status,
      error = err,
      stringsAsFactors = FALSE
    )
    idx <- idx + 1L
    message(sprintf("[%s] rep=%d elapsed=%.1f ms acc=%s status=%s",
                    v$name, rep, rows[[idx - 1L]]$elapsed_ms,
                    format(rows[[idx - 1L]]$accuracy, digits = 4), status))
  }
}

res <- do.call(rbind, rows)
utils::write.csv(res, outfile, row.names = FALSE)
print(res)
cat("Saved:", normalizePath(outfile, winslash = "/", mustWork = FALSE), "\n")
