test_that("fastsvd returns decomposition outputs from public backends", {
  set.seed(1)
  A <- matrix(rnorm(120 * 20), 120, 20)
  out <- fastsvd(A, ncomp = 5, method = "cpu_rsvd")
  expect_true(is.list(out))
  expect_true(all(c("u", "d", "v", "method", "elapsed") %in% names(out)))
  expect_equal(ncol(out$u), 5)
  expect_equal(length(out$d), 5)
  expect_equal(ncol(out$v), 5)
})

test_that("removed arpack and dc labels are rejected by fastsvd", {
  set.seed(11)
  A <- matrix(rnorm(80 * 12), 80, 12)
  expect_error(fastsvd(A, ncomp = 4, method = "arpack"), "removed")
  expect_error(fastsvd(A, ncomp = 4, method = "dc"), "removed")
})

test_that("small SVD inputs use exact fallback for iterative public backends", {
  set.seed(42)
  A <- matrix(rnorm(40 * 5), 40, 5)
  ref <- base::svd(A, nu = 3, nv = 3)

  for (method in c("irlba", "cpu_rsvd")) {
    out <- fastsvd(
      A,
      ncomp = 3,
      method = method,
      rsvd_oversample = 0L,
      rsvd_power = 0L,
      seed = 99L
    )
    expect_equal(out$d, ref$d[1:3], tolerance = 1e-8)
    expect_equal(abs(out$u), abs(ref$u[, 1:3, drop = FALSE]), tolerance = 1e-6)
    expect_equal(abs(out$v), abs(ref$v[, 1:3, drop = FALSE]), tolerance = 1e-6)
  }
})

test_that("pca uses public SVD backends and returns plottable scores", {
  set.seed(9)
  X <- matrix(rnorm(60 * 8), 60, 8)
  fit <- pca(X, ncomp = 3, svd.method = "cpu_rsvd", seed = 12)
  expect_s3_class(fit, "fastPLSPCA")
  expect_equal(dim(fit$scores), c(60L, 3L))
  expect_equal(dim(fit$loadings), c(8L, 3L))
  expect_true(all(is.finite(fit$variance_explained)))
})

test_that("fastPLS does not mask base svd", {
  expect_identical(svd, base::svd)
})
