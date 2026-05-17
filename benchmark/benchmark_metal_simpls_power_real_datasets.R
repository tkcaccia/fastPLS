#!/usr/bin/env Rscript

suppressPackageStartupMessages(library(fastPLS))

timestamp <- function() format(Sys.time(), "%Y-%m-%d %H:%M:%S")
log_msg <- function(...) cat("[", timestamp(), "] ", paste(..., collapse = ""), "\n", sep = "")

task_dir <- Sys.getenv(
  "FASTPLS_REAL_TASK_DIR",
  "/Users/stefano/Documents/GPUPLS/local_usual_pipeline_metal_20260515_230543/real_datasets"
)
out_dir <- Sys.getenv(
  "FASTPLS_METAL_POWER_OUT",
  file.path(getwd(), "benchmark_results", paste0("metal_simpls_power_real_", format(Sys.time(), "%Y%m%d_%H%M%S")))
)
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

best_ncomp <- c(
  metref = 100L,
  ccle = 50L,
  cifar100 = 200L,
  prism = 5L,
  gtex_v8 = 100L,
  tcga_pan_cancer = 100L,
  singlecell = 50L,
  tcga_brca = 5L,
  tcga_hnsc_methylation = 2L,
  cbmc_citeseq = 50L
)

datasets <- strsplit(
  Sys.getenv("FASTPLS_POWER_DATASETS", paste(names(best_ncomp), collapse = ",")),
  ",",
  fixed = TRUE
)[[1]]
datasets <- trimws(datasets[nzchar(trimws(datasets))])

extract_pred_matrix <- function(pred) {
  yp <- pred$Ypred
  if (is.null(yp)) yp <- pred$Prediction
  if (length(dim(yp)) == 3L) {
    yp <- matrix(yp[, , dim(yp)[3L]], nrow = dim(yp)[1L], ncol = dim(yp)[2L])
  }
  as.matrix(yp)
}

classification_accuracy <- function(pred, truth) {
  if (!is.null(pred$Ypred_index)) {
    idx <- as.integer(pred$Ypred_index[, 1L])
    lev <- levels(as.factor(truth))
    if (length(lev) >= max(idx, na.rm = TRUE)) {
      return(mean(lev[idx] == as.character(truth), na.rm = TRUE))
    }
  }
  yhat <- pred$Ypred
  if (is.data.frame(yhat)) yhat <- yhat[[1L]]
  mean(as.character(yhat) == as.character(truth), na.rm = TRUE)
}

rmsd <- function(pred, truth) {
  sqrt(mean((as.matrix(pred) - as.matrix(truth))^2, na.rm = TRUE))
}

q2 <- function(pred, truth, train_y) {
  truth <- as.matrix(truth)
  pred <- as.matrix(pred)
  center <- matrix(colMeans(as.matrix(train_y), na.rm = TRUE),
                   nrow = nrow(truth), ncol = ncol(truth), byrow = TRUE)
  1 - sum((truth - pred)^2, na.rm = TRUE) / sum((truth - center)^2, na.rm = TRUE)
}

run_one <- function(dataset, power) {
  task_path <- file.path(task_dir, paste0(dataset, "_task.rds"))
  if (!file.exists(task_path)) {
    return(data.frame(
      dataset = dataset, task_type = NA_character_, ncomp = best_ncomp[[dataset]],
      power = power, fit_sec = NA_real_, predict_sec = NA_real_, total_sec = NA_real_,
      metric_name = NA_character_, metric_value = NA_real_, rmsd = NA_real_, q2 = NA_real_,
      accuracy = NA_real_, status = "missing_task", message = task_path
    ))
  }
  task <- readRDS(task_path)
  Xtrain <- as.matrix(task$Xtrain)
  Xtest <- as.matrix(task$Xtest)
  Ytrain <- task$Ytrain
  Ytest <- task$Ytest
  ncomp <- as.integer(best_ncomp[[dataset]])

  log_msg("RUN dataset=", dataset, " task=", task$task_type,
          " ncomp=", ncomp, " power=", power)
  gc()
  fit_time <- system.time({
    fit <- tryCatch(
      fastPLS::pls(
        Xtrain, Ytrain,
        ncomp = ncomp,
        method = "simpls",
        backend = "metal",
        svd.method = "rsvd",
        scaling = "centering",
        fit = FALSE,
        return_variance = FALSE,
        classifier = "argmax",
        power = power,
        seed = 123
      ),
      error = function(e) e
    )
  })
  if (inherits(fit, "error")) {
    log_msg("ERR fit dataset=", dataset, " power=", power, " msg=", conditionMessage(fit))
    return(data.frame(
      dataset = dataset, task_type = task$task_type, ncomp = ncomp,
      power = power, fit_sec = NA_real_, predict_sec = NA_real_, total_sec = NA_real_,
      metric_name = NA_character_, metric_value = NA_real_, rmsd = NA_real_, q2 = NA_real_,
      accuracy = NA_real_, status = "fit_error", message = conditionMessage(fit)
    ))
  }

  gc()
  pred_time <- system.time({
    pred <- tryCatch(predict(fit, Xtest, Ytest = Ytest), error = function(e) e)
  })
  if (inherits(pred, "error")) {
    log_msg("ERR predict dataset=", dataset, " power=", power, " msg=", conditionMessage(pred))
    return(data.frame(
      dataset = dataset, task_type = task$task_type, ncomp = ncomp,
      power = power, fit_sec = unname(fit_time[["elapsed"]]),
      predict_sec = NA_real_, total_sec = NA_real_,
      metric_name = NA_character_, metric_value = NA_real_, rmsd = NA_real_, q2 = NA_real_,
      accuracy = NA_real_, status = "predict_error", message = conditionMessage(pred)
    ))
  }

  if (identical(task$task_type, "classification")) {
    acc <- classification_accuracy(pred, Ytest)
    metric_name <- "accuracy"
    metric_value <- acc
    rr <- NA_real_
    qq <- NA_real_
  } else {
    pred_mat <- extract_pred_matrix(pred)
    rr <- rmsd(pred_mat, Ytest)
    qq <- q2(pred_mat, Ytest, Ytrain)
    acc <- NA_real_
    metric_name <- "rmsd"
    metric_value <- rr
  }

  out <- data.frame(
    dataset = dataset,
    task_type = task$task_type,
    ncomp = ncomp,
    power = power,
    fit_sec = unname(fit_time[["elapsed"]]),
    predict_sec = unname(pred_time[["elapsed"]]),
    total_sec = unname(fit_time[["elapsed"]] + pred_time[["elapsed"]]),
    metric_name = metric_name,
    metric_value = metric_value,
    rmsd = rr,
    q2 = qq,
    accuracy = acc,
    status = "ok",
    message = "",
    stringsAsFactors = FALSE
  )
  log_msg("OK dataset=", dataset, " power=", power,
          " total=", round(out$total_sec, 3), " ", metric_name, "=",
          signif(metric_value, 6))
  rm(fit, pred)
  gc()
  out
}

all_results <- data.frame()
out_csv <- file.path(out_dir, "metal_simpls_power_real_datasets.csv")
for (dataset in datasets) {
  for (power in c(1L, 2L)) {
    row <- run_one(dataset, power)
    all_results <- rbind(all_results, row)
    write.csv(all_results, out_csv, row.names = FALSE)
  }
}

writeLines(capture.output(sessionInfo()), file.path(out_dir, "sessionInfo.txt"))
log_msg("Saved: ", out_csv)
print(all_results)
