test_that("the qualified rSVD controls are package-wide defaults", {
  defaults <- fastPLS:::.svd_control_defaults()
  expect_identical(defaults$rsvd_oversample, 10L)
  expect_identical(defaults$rsvd_power, 2L)
  expect_identical(formals(fastsvd)$oversample, 10L)
  expect_identical(formals(fastsvd)$power, 2L)

  resolved <- fastPLS:::.resolve_svd_control(context = "test")
  expect_identical(resolved$rsvd_oversample, 10L)
  expect_identical(resolved$rsvd_power, 2L)
})

test_that("pls records the qualified rSVD controls unless explicitly overridden", {
  set.seed(2201)
  X <- matrix(rnorm(80 * 12), 80, 12)
  Y <- matrix(rnorm(80 * 3), 80, 3)

  fit_default <- pls(X, Y, ncomp = 2, method = "simpls")
  expect_identical(fit_default$diagnostics$rsvd$oversample, 10L)
  expect_identical(fit_default$diagnostics$rsvd$power, 2L)

  fit_explicit <- pls(
    X, Y, ncomp = 2, method = "simpls",
    oversample = 6L, power = 1L
  )
  expect_identical(fit_explicit$diagnostics$rsvd$oversample, 6L)
  expect_identical(fit_explicit$diagnostics$rsvd$power, 1L)
})
