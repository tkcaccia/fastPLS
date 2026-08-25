#!/usr/bin/env Rscript

# Unrestricted end-to-end example covering data generation, a fixed split,
# training-only selection, final fitting, prediction, evaluation, and export.
suppressPackageStartupMessages(library(fastPLS))

set.seed(123)
out_dir <- file.path("benchmark_results", "synthetic_end_to_end_example")
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

n <- 180L
p <- 30L
X <- matrix(rnorm(n * p), n, p)
latent <- X[, 1L] - 0.8 * X[, 2L] + 0.5 * X[, 3L]
y_class <- factor(ifelse(latent + rnorm(n, sd = 0.7) > 0, "case", "control"))
y_reg <- 2 * latent + rnorm(n, sd = 0.5)

test_id <- sample.int(n, 45L)
train_id <- setdiff(seq_len(n), test_id)
Xtrain <- X[train_id, , drop = FALSE]
Xtest <- X[test_id, , drop = FALSE]

classification_time <- system.time({
  cv_class <- pls.single.cv(
    Xtrain, y_class[train_id],
    ncomp = 1:5,
    kfold = 3,
    scaling = "autoscaling",
    method = "simpls",
    svd.method = "rsvd",
    classifier = c("argmax", "lda"),
    selection_metric = "balanced_accuracy",
    seed = 123
  )
  fit_class <- pls(
    cv_class,
    Xtest = Xtest,
    Ytest = y_class[test_id],
    return_variance = FALSE
  )
})
class_metrics <- evaluate(
  observed = y_class[test_id],
  predicted = fit_class$Ypred[[length(fit_class$Ypred)]]
)

regression_time <- system.time({
  cv_reg <- pls.single.cv(
    Xtrain, y_reg[train_id],
    ncomp = 1:5,
    kfold = 3,
    scaling = "autoscaling",
    method = "simpls",
    svd.method = "rsvd",
    selection_metric = "rmsd",
    seed = 123
  )
  fit_reg <- pls(
    cv_reg,
    Xtest = Xtest,
    Ytest = y_reg[test_id],
    return_variance = FALSE
  )
})
reg_metrics <- evaluate(
  observed = y_reg[test_id],
  predicted = fit_reg$Ypred[, 1L, dim(fit_reg$Ypred)[3L], drop = TRUE],
  ytrain = y_reg[train_id]
)

summary <- data.frame(
  task = c("classification", "regression"),
  selected_ncomp = c(cv_class$best_ncomp, cv_reg$best_ncomp),
  selection_metric = c("balanced_accuracy", "rmsd"),
  heldout_metric = c(
    class_metrics$metrics$balanced_accuracy,
    reg_metrics$metrics$RMSD
  ),
  elapsed_sec = c(classification_time[["elapsed"]], regression_time[["elapsed"]])
)

write.csv(summary, file.path(out_dir, "summary.csv"), row.names = FALSE)
saveRDS(
  list(
    split = list(train = train_id, test = test_id),
    classification = list(cv = cv_class, fit = fit_class, metrics = class_metrics),
    regression = list(cv = cv_reg, fit = fit_reg, metrics = reg_metrics)
  ),
  file.path(out_dir, "complete_results.rds")
)
writeLines(capture.output(sessionInfo()), file.path(out_dir, "session_info.txt"))
print(summary)
