library(fastPLS)

test_that("Gaussian response compression decodes regression predictions", {
  set.seed(120)
  X <- matrix(rnorm(70 * 15), nrow = 70, ncol = 15)
  Y <- matrix(rnorm(70 * 24), nrow = 70, ncol = 24)
  Xtest <- matrix(rnorm(9 * 15), nrow = 9, ncol = 15)
  Ytest <- matrix(rnorm(9 * 24), nrow = 9, ncol = 24)

  fit <- pls(
    X,
    Y,
    Xtest,
    Ytest,
    ncomp = 1:3,
    method = "simpls",
    svd.method = "cpu_rsvd",
    gaussian_y = TRUE,
    gaussian_y_dim = 7,
    seed = 120L
  )

  expect_true(isTRUE(fit$gaussian_y))
  expect_identical(fit$gaussian_y_task, "regression")
  expect_equal(fit$gaussian_y_dim, 7)
  expect_equal(dim(fit$Ypred), c(nrow(Xtest), ncol(Y), 3L))
  expect_length(fit$Q2Y, 3L)
  expect_true(all(is.finite(fit$Ypred)))

  default_fit <- pls(
    X,
    Y,
    Xtest,
    Ytest,
    ncomp = 2,
    method = "plssvd",
    svd.method = "cpu_rsvd",
    gaussian_y = TRUE,
    seed = 120L
  )
  expect_equal(default_fit$gaussian_y_dim, min(ncol(X), 100L))
  expect_equal(dim(default_fit$Ypred), c(nrow(Xtest), ncol(Y), 1L))

  small_y <- Y[, 1:3, drop = FALSE]
  small_ytest <- Ytest[, 1:3, drop = FALSE]
  default_small_y <- pls(
    X,
    small_y,
    Xtest,
    small_ytest,
    ncomp = 2,
    method = "plssvd",
    svd.method = "cpu_rsvd",
    gaussian_y = TRUE,
    seed = 120L
  )
  expect_equal(default_small_y$gaussian_y_dim, min(ncol(X), 100L))
  expect_equal(dim(default_small_y$Ypred), c(nrow(Xtest), ncol(small_y), 1L))

  r_fit <- pls(
    X,
    Y,
    Xtest,
    Ytest,
    ncomp = 2,
    method = "simpls",
    backend = "r",
    svd.method = "cpu_rsvd",
    gaussian_y = TRUE,
    gaussian_y_dim = 7,
    seed = 120L
  )
  expect_equal(dim(r_fit$Ypred), c(nrow(Xtest), ncol(Y), 1L))
  expect_true(isTRUE(r_fit$gaussian_y))
})

test_that("Gaussian response compression decodes classification labels", {
  set.seed(121)
  X <- matrix(rnorm(90 * 12), nrow = 90, ncol = 12)
  y <- factor(sample(letters[1:6], nrow(X), replace = TRUE))
  Xtest <- matrix(rnorm(11 * 12), nrow = 11, ncol = 12)
  ytest <- factor(sample(levels(y), nrow(Xtest), replace = TRUE), levels = levels(y))

  fit <- pls(
    X,
    y,
    Xtest,
    ytest,
    ncomp = 2,
    method = "simpls",
    svd.method = "cpu_rsvd",
    gaussian_y = TRUE,
    gaussian_y_dim = 5,
    seed = 121L
  )

  expect_true(isTRUE(fit$gaussian_y))
  expect_identical(fit$gaussian_y_task, "classification")
  expect_s3_class(fit$Ypred[[1]], "factor")
  expect_equal(levels(fit$Ypred[[1]]), levels(y))
  expect_equal(nrow(fit$Ypred), nrow(Xtest))
})

test_that("Gaussian response compression is available through OPLS and kernelPLS", {
  set.seed(122)
  X <- matrix(rnorm(55 * 10), nrow = 55, ncol = 10)
  Y <- matrix(rnorm(55 * 18), nrow = 55, ncol = 18)
  Xtest <- matrix(rnorm(8 * 10), nrow = 8, ncol = 10)
  Ytest <- matrix(rnorm(8 * 18), nrow = 8, ncol = 18)

  opls_fit <- pls(
    X,
    Y,
    Xtest,
    Ytest,
    ncomp = 2,
    method = "opls",
    backend = "cpp",
    gaussian_y = TRUE,
    gaussian_y_dim = 6
  )
  opls_r_fit <- pls(
    X,
    Y,
    Xtest,
    Ytest,
    ncomp = 2,
    method = "opls",
    backend = "r",
    gaussian_y = TRUE,
    gaussian_y_dim = 6
  )
  kernel_fit <- pls(
    X,
    Y,
    Xtest,
    Ytest,
    ncomp = 2,
    method = "kernelpls",
    backend = "cpp",
    kernel = "linear",
    gaussian_y = TRUE,
    gaussian_y_dim = 6
  )
  kernel_r_fit <- pls(
    X,
    Y,
    Xtest,
    Ytest,
    ncomp = 2,
    method = "kernelpls",
    backend = "r",
    kernel = "linear",
    gaussian_y = TRUE,
    gaussian_y_dim = 6
  )
  kernel_svd_fit <- pls(
    X,
    Y,
    Xtest,
    Ytest,
    ncomp = 2,
    method = "kernelpls",
    backend = "cpp",
    kernel = "linear",
    inner.method = "plssvd",
    gaussian_y = TRUE,
    gaussian_y_dim = 6
  )

  expect_equal(dim(opls_fit$Ypred), c(nrow(Xtest), ncol(Y), 1L))
  expect_equal(dim(opls_r_fit$Ypred), c(nrow(Xtest), ncol(Y), 1L))
  expect_equal(dim(kernel_fit$Ypred), c(nrow(Xtest), ncol(Y), 1L))
  expect_equal(dim(kernel_r_fit$Ypred), c(nrow(Xtest), ncol(Y), 1L))
  expect_equal(dim(kernel_svd_fit$Ypred), c(nrow(Xtest), ncol(Y), 1L))
  expect_true(isTRUE(opls_fit$inner_model$gaussian_y))
  expect_true(isTRUE(opls_r_fit$inner_model$gaussian_y))
  expect_true(isTRUE(kernel_fit$gaussian_y))
  expect_true(isTRUE(kernel_r_fit$gaussian_y))
  expect_true(isTRUE(kernel_svd_fit$gaussian_y))
})
