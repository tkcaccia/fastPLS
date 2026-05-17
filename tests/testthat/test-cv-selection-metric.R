test_that("single.pls.cv can optimize explicit regression metrics", {
  set.seed(2101)
  X <- matrix(rnorm(48 * 7), nrow = 48, ncol = 7)
  y <- matrix(0.6 * X[, 1] - 0.3 * X[, 2] + rnorm(48, sd = 0.2), ncol = 1)

  opt_r2 <- single.pls.cv(
    Xdata = X,
    Ydata = y,
    ncomp = 1:2,
    kfold = 3,
    method = "simpls",
    backend = "cpu",
    svd.method = "rsvd",
    seed = 11,
    selection_metric = "r2"
  )
  expect_identical(opt_r2$selection_metric, "r2")
  expect_identical(opt_r2$best_metric_name, "r2")
  expect_true(opt_r2$best_ncomp %in% 1:2)

  opt_rmsd <- single.pls.cv(
    Xdata = X,
    Ydata = y,
    ncomp = 1:2,
    kfold = 3,
    method = "simpls",
    backend = "cpu",
    svd.method = "rsvd",
    seed = 11,
    selection_metric = "rmsd"
  )
  expect_identical(opt_rmsd$selection_metric, "rmsd")
  expect_identical(opt_rmsd$best_metric_name, "rmsd")
  expect_true(opt_rmsd$best_ncomp %in% 1:2)
  expect_null(opt_rmsd$Ypred)
  expect_null(opt_rmsd$Ypred_optim)
})

test_that("classification CV selects by accuracy and nested CV forwards the rule", {
  set.seed(2102)
  X <- matrix(rnorm(54 * 6), nrow = 54, ncol = 6)
  y <- factor(rep(c("A", "B", "C"), each = 18))
  X[, 1] <- X[, 1] + as.numeric(y)

  opt <- single.pls.cv(
    Xdata = X,
    Ydata = y,
    ncomp = 1:2,
    kfold = 3,
    method = "plssvd",
    backend = "cpu",
    svd.method = "rsvd",
    seed = 12,
    selection_metric = "accuracy"
  )
  expect_identical(opt$selection_metric, "accuracy")
  expect_identical(opt$best_metric_name, "accuracy")
  expect_true(opt$best_ncomp %in% 1:2)
  expect_null(opt$class_pred)
  expect_null(opt$Ypred_optim)

  nested <- pls.double.cv(
    Xdata = X,
    Ydata = y,
    ncomp = 1:2,
    runn = 1,
    kfold_inner = 3,
    kfold_outer = 3,
    method = "plssvd",
    backend = "cpu",
    svd.method = "rsvd",
    seed = 13,
    selection_metric = "accuracy"
  )
  expect_identical(nested$selection_metric, "accuracy")
  expect_true(all(vapply(nested$results[[1]]$inner, function(x) {
    is.null(x) || identical(x$selection_metric, "accuracy")
  }, logical(1))))
})

test_that("SIMPLS metric-only CV matches stored prediction CV", {
  set.seed(2103)
  X <- matrix(rnorm(60 * 8), nrow = 60, ncol = 8)
  y <- matrix(0.7 * X[, 1] - 0.5 * X[, 3] + rnorm(60, sd = 0.25), ncol = 1)

  metric_only <- single.pls.cv(
    Xdata = X,
    Ydata = y,
    ncomp = 1:3,
    kfold = 3,
    method = "simpls",
    backend = "cpu",
    svd.method = "rsvd",
    seed = 21,
    selection_metric = "rmsd"
  )
  stored <- single.pls.cv(
    Xdata = X,
    Ydata = y,
    ncomp = 1:3,
    kfold = 3,
    method = "simpls",
    backend = "cpu",
    svd.method = "rsvd",
    seed = 21,
    return_scores = TRUE,
    selection_metric = "rmsd"
  )

  expect_null(metric_only$Ypred)
  expect_false(is.null(stored$Ypred))
  expect_equal(
    metric_only$selection_metrics$metric_value,
    stored$selection_metrics$metric_value,
    tolerance = 1e-8
  )
  expect_identical(metric_only$best_ncomp, stored$best_ncomp)
})
