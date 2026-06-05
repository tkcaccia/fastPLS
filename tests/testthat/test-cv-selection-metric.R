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

test_that("regression CV reports distinct training R2 and held-out Q2", {
  set.seed(2104)
  X <- matrix(rnorm(80 * 12), nrow = 80, ncol = 12)
  beta <- matrix(rnorm(12 * 2), nrow = 12, ncol = 2)
  Y <- X %*% beta + matrix(rnorm(80 * 2, sd = 2), nrow = 80, ncol = 2)

  scalar_fit <- pls(
    Xtrain = X,
    Ytrain = Y,
    ncomp = 3,
    method = "simpls",
    backend = "cpu",
    svd.method = "rsvd",
    fit = TRUE,
    return_variance = FALSE,
    seed = 31
  )
  path_fit <- pls(
    Xtrain = X,
    Ytrain = Y,
    ncomp = 1:3,
    method = "simpls",
    backend = "cpu",
    svd.method = "rsvd",
    fit = TRUE,
    return_variance = FALSE,
    seed = 31
  )
  expect_equal(scalar_fit$R2Y[[1]], path_fit$R2Y[[3]], tolerance = 1e-10)

  single <- single.pls.cv(
    Xdata = X,
    Ydata = Y,
    ncomp = 1:3,
    kfold = 4,
    method = "simpls",
    backend = "cpu",
    svd.method = "rsvd",
    seed = 32,
    selection_metric = "q2"
  )
  expect_false(isTRUE(all.equal(single$Q2Y, single$R2Y)))
  expect_false(isTRUE(all.equal(single$Q2Y, single$RMSD)))
  expect_false(isTRUE(all.equal(single$R2Y, single$RMSD)))

  nested <- pls.double.cv(
    Xdata = X,
    Ydata = Y,
    ncomp = 1:3,
    runn = 1,
    kfold_inner = 3,
    kfold_outer = 3,
    method = "simpls",
    backend = "cpu",
    svd.method = "rsvd",
    seed = 33,
    selection_metric = "q2"
  )
  expect_false(isTRUE(all.equal(nested$Q2Y, nested$R2Y)))
  expect_false(isTRUE(all.equal(nested$Q2Y, nested$RMSD)))
  expect_false(isTRUE(all.equal(nested$R2Y, nested$RMSD)))
})

test_that("single.pls.cv can skip the extra full-data R2 fit", {
  set.seed(21045)
  X <- matrix(rnorm(48 * 7), nrow = 48, ncol = 7)
  beta <- matrix(rnorm(7 * 2), nrow = 7, ncol = 2)
  Y <- X %*% beta + matrix(rnorm(48 * 2, sd = 0.25), nrow = 48, ncol = 2)

  with_r2 <- single.pls.cv(
    Xdata = X,
    Ydata = Y,
    ncomp = 1:3,
    kfold = 4,
    method = "simpls",
    backend = "cpu",
    svd.method = "rsvd",
    seed = 21045,
    return_r2 = TRUE
  )
  without_r2 <- single.pls.cv(
    Xdata = X,
    Ydata = Y,
    ncomp = 1:3,
    kfold = 4,
    method = "simpls",
    backend = "cpu",
    svd.method = "rsvd",
    seed = 21045,
    return_r2 = FALSE
  )

  expect_true(any(is.finite(with_r2$R2Y)))
  expect_true(all(is.na(without_r2$R2Y)))
  expect_true(any(is.finite(without_r2$Q2Y)))
  expect_true(any(is.finite(without_r2$RMSD)))
  expect_equal(without_r2$best_ncomp, with_r2$best_ncomp)
})

test_that("RMSD selection does not overwrite Q2Y", {
  set.seed(2105)
  X <- matrix(rnorm(72 * 9), nrow = 72, ncol = 9)
  beta <- matrix(rnorm(9 * 3), nrow = 9, ncol = 3)
  Y <- X %*% beta + matrix(rnorm(72 * 3, sd = 1.5), nrow = 72, ncol = 3)

  single <- single.pls.cv(
    Xdata = X,
    Ydata = Y,
    ncomp = 1:3,
    kfold = 4,
    method = "simpls",
    backend = "cpu",
    svd.method = "rsvd",
    seed = 34,
    selection_metric = "rmsd"
  )
  expect_identical(single$best_metric_name, "rmsd")
  expect_false(isTRUE(all.equal(single$Q2Y, single$RMSD)))
  expect_equal(single$selection_values, single$RMSD, tolerance = 1e-10)

  nested <- pls.double.cv(
    Xdata = X,
    Ydata = Y,
    ncomp = 1:3,
    runn = 1,
    kfold_inner = 3,
    kfold_outer = 3,
    method = "simpls",
    backend = "cpu",
    svd.method = "rsvd",
    seed = 35,
    selection_metric = "rmsd"
  )
  expect_identical(nested$metric_name[[1]], "rmsd")
  expect_false(isTRUE(all.equal(nested$Q2Y, nested$RMSD)))
  expect_equal(nested$results[[1]]$metric_value, nested$RMSD[[1]], tolerance = 1e-10)
})
