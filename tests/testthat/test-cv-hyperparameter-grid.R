library(fastPLS)

test_that("single.pls.cv tunes prediction hyperparameters", {
  idx <- c(1:12, 51:62, 101:112)
  X <- as.matrix(iris[idx, 1:4])
  y <- factor(iris[idx, 5])

  opt <- single.pls.cv(
    X,
    y,
    ncomp = 1:2,
    kfold = 3,
    method = "kernelpls",
    backend = "cpu",
    svd.method = "rsvd",
    kernel = c("linear", "rbf"),
    gamma = c(0.1, 1),
    seed = 1
  )

  expect_true(opt$best_ncomp %in% 1:2)
  expect_true(all(c("kernel", "gamma", "ncomp") %in% names(opt$best_parameters)))
  expect_false("method" %in% names(opt$best_parameters))
  expect_true(nrow(opt$tuning_summary) >= 2L)
  expect_true(all(c("kernel", "gamma", "best_metric_value") %in% names(opt$tuning_summary)))
})

test_that("pls.double.cv uses inner selected hyperparameters", {
  idx <- c(1:10, 51:60, 101:110)
  X <- as.matrix(iris[idx, 1:4])
  y <- factor(iris[idx, 5])

  nested <- pls.double.cv(
    X,
    y,
    ncomp = 1:2,
    runn = 1,
    kfold_inner = 2,
    kfold_outer = 2,
    method = "kernelpls",
    backend = "cpu",
    svd.method = "rsvd",
    kernel = c("linear", "rbf"),
    gamma = c(0.1, 1),
    seed = 1
  )

  expect_length(nested$results, 1L)
  expect_true(all(nested$results[[1]]$best_ncomp %in% 1:2))
  expect_true(all(vapply(nested$results[[1]]$best_parameters, function(x) {
    is.list(x) && all(c("kernel", "gamma", "ncomp") %in% names(x)) &&
      !("method" %in% names(x))
  }, logical(1L))))
})

test_that("single.pls.cv reports only optimized best parameters", {
  idx <- c(1:12, 51:62, 101:112)
  X <- as.matrix(iris[idx, 1:4])
  y <- factor(iris[idx, 5])

  opt <- single.pls.cv(
    X,
    y,
    ncomp = 2:4,
    kfold = 3,
    classifier = c("argmax", "lda"),
    seed = 1
  )

  expect_setequal(names(opt$best_parameters), c("ncomp", "classifier"))
  expect_true(opt$best_parameters$ncomp %in% 2:4)
  expect_true(opt$best_parameters$classifier %in% c("argmax", "lda"))
})
