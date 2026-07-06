test_that("vendored fastPLS-cpp rSVD bridge returns valid decompositions", {
  set.seed(2201)
  A <- matrix(rnorm(90 * 18), 90, 18)

  out64 <- fastPLS:::fastpls_cpp_core_rsvd(A, 5L, oversample = 8L, power = 1L, seed = 11L)
  out32 <- fastPLS:::fastpls_cpp_core_rsvd(A, 5L, oversample = 8L, power = 1L, seed = 11L, use_float = TRUE)

  expect_equal(dim(out64$u), c(90L, 5L))
  expect_equal(dim(out64$v), c(18L, 5L))
  expect_length(out64$d, 5L)
  expect_true(all(is.finite(out64$d)))
  expect_identical(out64$precision, "double64")

  expect_equal(dim(out32$u), c(90L, 5L))
  expect_equal(dim(out32$v), c(18L, 5L))
  expect_length(out32$d, 5L)
  expect_true(all(is.finite(out32$d)))
  expect_identical(out32$precision, "float32")

  expect_equal(out32$d, out64$d, tolerance = 2e-1)
})

test_that("vendored fastPLS-cpp PLS bridge produces sensible regression fits", {
  set.seed(2202)
  X <- matrix(rnorm(80 * 12), 80, 12)
  Y <- cbind(
    0.9 * X[, 1] - 0.4 * X[, 3] + rnorm(80, sd = 0.02),
    -0.3 * X[, 2] + 0.7 * X[, 4] + rnorm(80, sd = 0.02)
  )

  simpls_core <- fastPLS:::fastpls_cpp_core_simpls(X, Y, 2L, oversample = 6L, power = 1L, seed = 12L)
  plssvd_core <- fastPLS:::fastpls_cpp_core_plssvd(X, Y, 2L, oversample = 6L, power = 1L, seed = 12L)

  expect_equal(dim(simpls_core$B), c(12L, 2L))
  expect_equal(dim(simpls_core$scores), c(80L, 2L))
  expect_lt(sqrt(mean((simpls_core$Ypred_train - Y)^2)), 0.30)

  expect_equal(dim(plssvd_core$B), c(12L, 2L))
  expect_equal(dim(plssvd_core$scores), c(80L, 2L))
  expect_lt(sqrt(mean((plssvd_core$Ypred_train - Y)^2)), 0.35)

  package_fit <- fastPLS::pls(
    X,
    Y,
    ncomp = 2L,
    method = "simpls",
    backend = "cpu",
    svd.method = "rsvd",
    fit = TRUE,
    return_variance = FALSE,
    seed = 12L
  )
  expect_s3_class(package_fit, "fastPLS")
  expect_true(any(is.finite(package_fit$Yfit[[1]])))
})

test_that("vendored fastPLS-cpp float bridge follows double bridge", {
  set.seed(2203)
  X <- matrix(rnorm(60 * 8), 60, 8)
  Y <- matrix(0.6 * X[, 1] - 0.2 * X[, 5] + rnorm(60, sd = 0.1), ncol = 1)

  fit64 <- fastPLS:::fastpls_cpp_core_simpls(X, Y, 2L, oversample = 5L, power = 1L, seed = 13L)
  fit32 <- fastPLS:::fastpls_cpp_core_simpls(X, Y, 2L, oversample = 5L, power = 1L, seed = 13L, use_float = TRUE)

  expect_equal(fit32$Ypred_train, fit64$Ypred_train, tolerance = 1e-3)
  expect_lt(sqrt(mean((fit32$Ypred_train - Y)^2)), 0.25)
})
