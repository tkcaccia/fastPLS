#!/usr/bin/env Rscript

suppressPackageStartupMessages(library(fastPLS))

timestamp <- function() format(Sys.time(), "%Y-%m-%d %H:%M:%S")
log_msg <- function(...) cat("[", timestamp(), "] ", paste(..., collapse = ""), "\n", sep = "")

task_path <- Sys.getenv(
  "FASTPLS_NMR_TASK_RDS",
  "/Users/stefano/Documents/GPUPLS/local_usual_pipeline_metal_20260515_230543/real_datasets/nmr_task.rds"
)
out_dir <- Sys.getenv(
  "FASTPLS_NMR_METAL_FOCUS_OUT",
  file.path(getwd(), "benchmark_results", paste0("nmr_metal_simpls_focus_", format(Sys.time(), "%Y%m%d_%H%M%S")))
)
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

log_msg("Loading task: ", task_path)
task <- readRDS(task_path)
Xtrain <- as.matrix(task$Xtrain)
Ytrain <- as.matrix(task$Ytrain)
Xtest <- as.matrix(task$Xtest)
Ytest <- as.matrix(task$Ytest)
log_msg("Dimensions: Xtrain=", paste(dim(Xtrain), collapse = "x"),
        " Ytrain=", paste(dim(Ytrain), collapse = "x"),
        " Xtest=", paste(dim(Xtest), collapse = "x"),
        " Ytest=", paste(dim(Ytest), collapse = "x"))

extract_pred <- function(pred) {
  yp <- pred$Ypred
  if (is.null(yp)) {
    yp <- pred$Prediction
  }
  if (length(dim(yp)) == 3L) {
    yp <- yp[, , dim(yp)[3L], drop = FALSE]
    yp <- matrix(yp, nrow = dim(yp)[1L], ncol = dim(yp)[2L])
  }
  as.matrix(yp)
}

rmsd <- function(pred, truth) {
  sqrt(mean((as.matrix(pred) - as.matrix(truth))^2, na.rm = TRUE))
}

bench_one <- function(label, method, backend) {
  log_msg("RUN ", label)
  gc()
  fit_time <- system.time({
    fit <- tryCatch(
      fastPLS::pls(
        Xtrain, Ytrain,
        ncomp = 50,
        method = method,
        backend = backend,
        svd.method = "rsvd",
        scaling = "centering",
        fit = FALSE,
        return_variance = FALSE,
        proj = FALSE,
        seed = 123
      ),
      error = function(e) e
    )
  })
  if (inherits(fit, "error")) {
    log_msg("ERR fit ", label, ": ", conditionMessage(fit))
    return(data.frame(
      variant = label, method = method, backend = backend,
      fit_sec = NA_real_, predict_sec = NA_real_, total_sec = NA_real_,
      rmsd = NA_real_, status = "error", message = conditionMessage(fit)
    ))
  }
  gc()
  pred_time <- system.time({
    pred <- tryCatch(
      predict(fit, Xtest, Ytest = Ytest),
      error = function(e) e
    )
  })
  if (inherits(pred, "error")) {
    log_msg("ERR predict ", label, ": ", conditionMessage(pred))
    return(data.frame(
      variant = label, method = method, backend = backend,
      fit_sec = unname(fit_time[["elapsed"]]), predict_sec = NA_real_,
      total_sec = NA_real_, rmsd = NA_real_,
      status = "predict_error", message = conditionMessage(pred)
    ))
  }
  pred_mat <- extract_pred(pred)
  out <- data.frame(
    variant = label,
    method = method,
    backend = backend,
    fit_sec = unname(fit_time[["elapsed"]]),
    predict_sec = unname(pred_time[["elapsed"]]),
    total_sec = unname(fit_time[["elapsed"]] + pred_time[["elapsed"]]),
    rmsd = rmsd(pred_mat, Ytest),
    status = "ok",
    message = "",
    stringsAsFactors = FALSE
  )
  log_msg("OK ", label, " fit=", round(out$fit_sec, 3),
          " pred=", round(out$predict_sec, 3),
          " total=", round(out$total_sec, 3),
          " rmsd=", signif(out$rmsd, 6))
  rm(fit, pred, pred_mat)
  gc()
  out
}

variants <- list(
  list(label = "cpp_plssvd_rsvd", method = "plssvd", backend = "cpu"),
  list(label = "metal_plssvd_rsvd", method = "plssvd", backend = "metal"),
  list(label = "cpp_simpls_rsvd", method = "simpls", backend = "cpu"),
  list(label = "metal_simpls_rsvd", method = "simpls", backend = "metal")
)

res <- do.call(rbind, lapply(variants, function(v) {
  bench_one(v$label, v$method, v$backend)
}))

out_csv <- file.path(out_dir, "nmr_ncomp50_cpp_metal_rsvd_compare.csv")
write.csv(res, out_csv, row.names = FALSE)
writeLines(capture.output(sessionInfo()), file.path(out_dir, "sessionInfo.txt"))
log_msg("Saved: ", out_csv)
print(res)
