#!/usr/bin/env Rscript

suppressPackageStartupMessages(library(fastPLS))

timestamp <- function() format(Sys.time(), "%F %T")

message(sprintf("[%s] MetRef gpu_qless_qr focused benchmark", timestamp()))

metref_path <- Sys.getenv(
  "FASTPLS_METREF_RDATA",
  "/home/chiamaka/Documents/fastpls/data/metref.RData"
)
out_root <- Sys.getenv(
  "FASTPLS_OUT_ROOT",
  path.expand("~/fastPLS_gpu_qless_qr_metref")
)
reps <- as.integer(Sys.getenv("FASTPLS_REPS", "3"))
ncomp <- as.integer(Sys.getenv("FASTPLS_NCOMP", "22"))

e <- new.env(parent = emptyenv())
load(metref_path, envir = e)
if (!exists("out", envir = e)) {
  stop("MetRef RData must contain an object named 'out'.", call. = FALSE)
}

task <- e$out
Xtrain <- as.matrix(task$Xtrain)
Ytrain <- task$Ytrain
Xtest <- as.matrix(task$Xtest)
Ytest <- task$Ytest

if (!is.factor(Ytrain) || !is.factor(Ytest)) {
  stop("This focused benchmark expects factor MetRef labels.", call. = FALSE)
}

lev <- levels(Ytrain)
Ytest <- factor(as.character(Ytest), levels = lev)

extract_pred <- function(fit) {
  yp <- fit$Ypred
  if (is.data.frame(yp)) {
    return(factor(as.character(yp[[ncol(yp)]]), levels = lev))
  }
  if (is.matrix(yp)) {
    return(factor(as.character(yp[, ncol(yp)]), levels = lev))
  }
  if (length(dim(yp)) == 3L) {
    mat <- yp[, , dim(yp)[3L], drop = TRUE]
    return(factor(lev[max.col(mat, ties.method = "first")], levels = lev))
  }
  factor(as.character(yp), levels = lev)
}

reset_cuda <- function() {
  if (exists("cuda_reset_workspace", envir = asNamespace("fastPLS"))) {
    try(fastPLS:::cuda_reset_workspace(), silent = TRUE)
  }
}

run_one <- function(method, qless, rep) {
  gc()
  reset_cuda()
  fun <- switch(
    method,
    plssvd = fastPLS:::plssvd_gpu,
    simpls = fastPLS:::simpls_gpu
  )
  args <- list(
    Xtrain = Xtrain,
    Ytrain = Ytrain,
    Xtest = Xtest,
    Ytest = Ytest,
    ncomp = ncomp,
    scaling = "centering",
    rsvd_oversample = 10L,
    rsvd_power = 1L,
    svds_tol = 0,
    seed = 123L + rep,
    fit = FALSE,
    proj = FALSE,
    gpu_qr = TRUE,
    gpu_eig = TRUE,
    gpu_qless_qr = qless,
    gpu_finalize_threshold = 32L,
    gaussian_y = FALSE,
    classifier = "argmax",
    return_variance = FALSE
  )
  if (identical(method, "simpls")) {
    args$gpu_device_state <- TRUE
  }

  err <- NULL
  fit <- NULL
  tm <- system.time({
    fit <- tryCatch(
      do.call(fun, args),
      error = function(e) {
        err <<- conditionMessage(e)
        NULL
      }
    )
  })

  if (!is.null(err)) {
    return(data.frame(
      method = method,
      gpu_qless_qr = qless,
      rep = rep,
      status = "error",
      total_sec = unname(tm[["elapsed"]]),
      accuracy = NA_real_,
      notes = err,
      stringsAsFactors = FALSE
    ))
  }

  pred <- extract_pred(fit)
  acc <- mean(as.character(pred) == as.character(Ytest), na.rm = TRUE)
  reset_cuda()
  data.frame(
    method = method,
    gpu_qless_qr = qless,
    rep = rep,
    status = "ok",
    total_sec = unname(tm[["elapsed"]]),
    accuracy = acc,
    notes = "",
    stringsAsFactors = FALSE
  )
}

rows <- list()
i <- 0L
for (method in c("plssvd", "simpls")) {
  for (qless in c(FALSE, TRUE)) {
    for (rep in seq_len(reps)) {
      i <- i + 1L
      message(sprintf(
        "[%s] RUN method=%s gpu_qless_qr=%s rep=%d",
        timestamp(), method, qless, rep
      ))
      rows[[i]] <- run_one(method, qless, rep)
      print(rows[[i]])
    }
  }
}

raw <- do.call(rbind, rows)
outdir <- file.path(out_root, format(Sys.time(), "%Y%m%d_%H%M%S"))
dir.create(outdir, recursive = TRUE, showWarnings = FALSE)
write.csv(raw, file.path(outdir, "metref_gpu_qless_qr_raw.csv"), row.names = FALSE)

summary <- aggregate(
  cbind(total_sec, accuracy) ~ method + gpu_qless_qr + status,
  raw,
  function(x) median(x, na.rm = TRUE)
)
n <- aggregate(rep ~ method + gpu_qless_qr + status, raw, length)
summary$n <- n$rep
write.csv(summary, file.path(outdir, "metref_gpu_qless_qr_summary.csv"), row.names = FALSE)

cat("\nSUMMARY\n")
print(summary)
cat("RESULT_DIR=", outdir, "\n", sep = "")
