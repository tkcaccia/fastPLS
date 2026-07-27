#!/usr/bin/env Rscript

suppressPackageStartupMessages(library(fastPLS))

repo_dir <- normalizePath(
  file.path(dirname(sub("^--file=", "", commandArgs()[grep("^--file=", commandArgs())])),
            "..", ".."),
  winslash = "/",
  mustWork = TRUE
)
out_dir <- Sys.getenv(
  "FASTPLS_METAL_CV_OUT",
  file.path(repo_dir, "benchmark_results",
            paste0("metal_cv_validation_", format(Sys.time(), "%Y%m%d_%H%M%S")))
)
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

tasks <- list(
  metref = list(
    path = file.path(
      repo_dir, "benchmark_results", "manuscript_revision_cycle13_20260725",
      "kernel_suite", "pipeline1", "real_datasets", "metref_task.rds"
    ),
    ncomp = c(5L, 10L, 15L, 20L, 22L)
  ),
  retina = list(
    path = file.path(
      repo_dir, "benchmark_results", "manuscript_revision_cycle13_20260725",
      "retina_tabula_selected_outer", "runs", "retina_simpls",
      "retina_task.rds"
    ),
    ncomp = c(5L, 10L, 15L, 20L)
  )
)
tasks <- tasks[vapply(tasks, function(x) file.exists(x$path), logical(1L))]

rss_mb <- function() {
  if (!requireNamespace("ps", quietly = TRUE)) return(NA_real_)
  as.numeric(ps::ps_memory_info(ps::ps_handle())[["rss"]]) / 1024^2
}

rows <- list()
index <- 0L
for (dataset in names(tasks)) {
  task <- readRDS(tasks[[dataset]]$path)
  for (method in c("plssvd", "simpls", "opls", "kernelpls")) {
    for (backend in c("cpu", "metal")) {
      for (svd_method in c("irlba", "rsvd")) {
        for (classifier in c("argmax", "lda")) {
          for (replicate in 1:3) {
            index <- index + 1L
            seed <- 600L + replicate
            gc()
            baseline <- rss_mb()
            warnings_seen <- character()
            elapsed <- system.time({
              fit <- tryCatch(
                withCallingHandlers(
                  pls.single.cv(
                    task$Xtrain,
                    task$Ytrain,
                    ncomp = tasks[[dataset]]$ncomp,
                    kfold = 10L,
                    method = method,
                    backend = backend,
                    svd.method = svd_method,
                    classifier = classifier,
                    seed = seed,
                    fit = FALSE
                  ),
                  warning = function(w) {
                    warnings_seen <<- c(warnings_seen, conditionMessage(w))
                    invokeRestart("muffleWarning")
                  }
                ),
                error = function(e) e
              )
            })[["elapsed"]]
            after <- rss_mb()
            if (inherits(fit, "error")) {
              rows[[index]] <- data.frame(
                dataset = dataset, method = method, backend_requested = backend,
                svd_method_requested = svd_method, classifier = classifier,
                replicate = replicate, seed = seed, kfold = 10L,
                ncomp_grid = paste(tasks[[dataset]]$ncomp, collapse = ";"),
                best_ncomp = NA_integer_, metric_name = NA_character_,
                metric_value = NA_real_, elapsed_sec = NA_real_,
                baseline_rss_mb = baseline, rss_after_mb = after,
                reported_backend = NA_character_,
                reported_prediction_backend = NA_character_,
                reported_svd_method = NA_character_,
                status = "failed",
                warnings = paste(unique(warnings_seen), collapse = " | "),
                error = conditionMessage(fit), stringsAsFactors = FALSE
              )
            } else {
              rows[[index]] <- data.frame(
                dataset = dataset, method = method, backend_requested = backend,
                svd_method_requested = svd_method, classifier = classifier,
                replicate = replicate, seed = seed, kfold = 10L,
                ncomp_grid = paste(tasks[[dataset]]$ncomp, collapse = ";"),
                best_ncomp = fit$best_ncomp,
                metric_name = fit$best_metric_name,
                metric_value = fit$best_metric_value,
                elapsed_sec = unname(elapsed),
                baseline_rss_mb = baseline, rss_after_mb = after,
                reported_backend = fit$backend,
                reported_prediction_backend = fit$prediction_backend,
                reported_svd_method = fit$tuning_config$svd.method,
                status = "success",
                warnings = paste(unique(warnings_seen), collapse = " | "),
                error = "", stringsAsFactors = FALSE
              )
            }
            write.csv(do.call(rbind, rows),
                      file.path(out_dir, "metal_cv_validation_raw.csv"),
                      row.names = FALSE)
            cat(
              sprintf(
                "[%s] %d dataset=%s method=%s backend=%s svd=%s head=%s status=%s time=%.3f\n",
                format(Sys.time(), "%Y-%m-%d %H:%M:%S"), index, dataset,
                method, backend, svd_method, classifier,
                rows[[index]]$status, rows[[index]]$elapsed_sec
              )
            )
          }
        }
      }
    }
  }
  rm(task)
  gc()
}

raw <- do.call(rbind, rows)
write.csv(raw, file.path(out_dir, "metal_cv_validation_raw.csv"), row.names = FALSE)
writeLines(capture.output(sessionInfo()), file.path(out_dir, "session_info.txt"))
