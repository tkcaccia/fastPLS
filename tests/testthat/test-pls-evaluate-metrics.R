test_that("pls includes complete evaluate metrics for classification", {
  idx <- c(1:12, 51:62, 101:112)
  X <- as.matrix(iris[idx, 1:4])
  y <- factor(iris[idx, 5])

  fit <- pls(
    X, y, X, y,
    ncomp = 1:2,
    method = "simpls",
    backend = "cpu",
    svd.method = "rsvd",
    fit = TRUE,
    return_variance = FALSE
  )

  expect_named(fit$metrics, c("definitions", "fitted", "test"))
  expect_named(fit$metrics$test[["ncomp=1"]],
               c("task", "metrics", "metric_definitions", "per_class", "confusion", "topk"))
  expect_true("lift_accuracy" %in% names(fit$metrics$test[["ncomp=1"]]$metrics))
})

test_that("single and double CV expose evaluate metrics separately", {
  idx <- c(1:12, 51:62, 101:112)
  X <- as.matrix(iris[idx, 1:4])
  y <- factor(iris[idx, 5])

  single <- pls.single.cv(
    X, y, ncomp = 1:2, kfold = 3,
    method = "simpls", backend = "cpu", svd.method = "rsvd", seed = 1
  )
  expect_named(single$metrics, c("definitions", "cross_validated", "fitted"))
  expect_true("balanced_accuracy" %in%
    names(single$metrics$cross_validated[["ncomp=1"]]$metrics))
  expect_true(is.data.frame(single$selection_metrics))

  nested <- pls.double.cv(
    X, y, ncomp = 1:2, runn = 1,
    kfold_inner = 2, kfold_outer = 2,
    method = "simpls", backend = "cpu", svd.method = "rsvd", seed = 1
  )
  expect_named(nested$metrics, c("definitions", "cross_validated", "aggregate"))
  expect_true("macro_f1" %in% names(nested$metrics$aggregate$metrics))
})

test_that("PLS metric paths honor bycol and retain permutation metrics", {
  set.seed(2)
  X <- matrix(rnorm(60), nrow = 15, ncol = 4)
  Y <- cbind(X[, 1] + rnorm(15, sd = 0.1), X[, 2] + rnorm(15, sd = 0.1))

  aggregate <- pls(
    X, Y, X, Y, ncomp = 1, method = "simpls", backend = "cpu",
    svd.method = "rsvd", return_variance = FALSE
  )
  detailed <- pls(
    X, Y, X, Y, ncomp = 1, method = "simpls", backend = "cpu",
    svd.method = "rsvd", bycol = TRUE, return_variance = FALSE
  )
  expect_null(aggregate$metrics$test[["ncomp=1"]]$per_response)
  expect_equal(nrow(detailed$metrics$test[["ncomp=1"]]$per_response), ncol(Y))

  single <- pls.single.cv(
    X, Y, ncomp = 1, kfold = 2, method = "simpls", backend = "cpu",
    svd.method = "rsvd", bycol = TRUE, seed = 1
  )
  expect_equal(nrow(single$metrics$cross_validated[["ncomp=1"]]$per_response), ncol(Y))

  nested <- pls.double.cv(
    X, Y, ncomp = 1, runn = 1, kfold_inner = 2, kfold_outer = 2,
    method = "simpls", backend = "cpu", svd.method = "rsvd",
    bycol = TRUE, seed = 1
  )
  expect_equal(nrow(nested$metrics$aggregate$per_response), ncol(Y))

  permuted <- pls(
    X, Y, X, Y, ncomp = 1, method = "simpls", backend = "cpu",
    svd.method = "rsvd", perm.test = TRUE, times = 2,
    return_variance = FALSE
  )
  expect_true(all(
    c("results", "p_value", "requested", "completed", "failed") %in%
      names(permuted$metrics$permutation)
  ))
})

test_that("single-split permutation p-values are calculated per component", {
  set.seed(204)
  X <- matrix(rnorm(72), nrow = 18, ncol = 4)
  Y <- cbind(X[, 1] + rnorm(18, sd = 0.2), X[, 2] + rnorm(18, sd = 0.2))

  fit <- pls(
    X, Y, X, Y, ncomp = 1:2, method = "simpls", backend = "cpu",
    svd.method = "rsvd", perm.test = TRUE, times = 4, seed = 17,
    return_variance = FALSE
  )

  perm_q2 <- subset(
    fit$permutation,
    type == "permutation" & metric == "Q2"
  )
  component_values <- sort(unique(perm_q2$ncomp))
  expected <- vapply(seq_along(fit$Q2Y), function(j) {
    values <- perm_q2$value[perm_q2$ncomp == component_values[[j]]]
    valid <- is.finite(values)
    (sum(values[valid] >= fit$Q2Y[[j]]) + 1) / (sum(valid) + 1)
  }, numeric(1L))
  names(expected) <- names(fit$Q2Y)

  expect_equal(fit$pval, expected)
  expect_true(all(fit$pval > 0))
  expect_identical(names(fit$pval), names(fit$Q2Y))
})
