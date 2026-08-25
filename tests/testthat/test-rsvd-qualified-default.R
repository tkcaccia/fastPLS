test_that("the qualified rSVD controls are package-wide defaults", {
  defaults <- fastPLS:::.svd_control_defaults()
  expect_identical(defaults$rsvd_oversample, 20L)
  expect_identical(defaults$rsvd_power, 2L)
  expect_identical(formals(fastsvd)$oversample, 20L)
  expect_identical(formals(fastsvd)$power, 2L)

  resolved <- fastPLS:::.resolve_svd_control(context = "test")
  expect_identical(resolved$rsvd_oversample, 20L)
  expect_identical(resolved$rsvd_power, 2L)
})

test_that("backend defaults use only audited rSVD configurations", {
  base <- fastPLS:::.resolve_svd_control(context = "test")

  cpu <- fastPLS:::.apply_backend_rsvd_controls(base, "cpu", "test")
  expect_identical(cpu$rsvd_oversample, 20L)
  expect_identical(cpu$rsvd_power, 2L)
  expect_true(cpu$rsvd_qualification$qualified_on_prespecified_panel)

  cuda <- fastPLS:::.apply_backend_rsvd_controls(base, "cuda", "test")
  expect_identical(cuda$rsvd_oversample, 20L)
  expect_identical(cuda$rsvd_power, 2L)
  expect_true(cuda$rsvd_qualification$qualified_on_prespecified_panel)

  expect_warning(
    metal <- fastPLS:::.apply_backend_rsvd_controls(base, "metal", "test"),
    "not qualified"
  )
  expect_identical(metal$rsvd_oversample, 20L)
  expect_false(metal$rsvd_qualification$qualified_on_prespecified_panel)
})

test_that("pls records the qualified rSVD controls unless explicitly overridden", {
  set.seed(2201)
  X <- matrix(rnorm(80 * 12), 80, 12)
  Y <- matrix(rnorm(80 * 3), 80, 3)

  fit_default <- pls(X, Y, ncomp = 2, method = "simpls")
  expect_identical(fit_default$diagnostics$rsvd$oversample, 20L)
  expect_identical(fit_default$diagnostics$rsvd$power, 2L)

  expect_warning(
    fit_explicit <- pls(
      X, Y, ncomp = 2, method = "simpls",
      oversample = 6L, power = 1L
    ),
    "not qualified"
  )
  expect_identical(fit_explicit$diagnostics$rsvd$oversample, 6L)
  expect_identical(fit_explicit$diagnostics$rsvd$power, 1L)
})
