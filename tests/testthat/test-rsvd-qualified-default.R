test_that("the audited rSVD starting controls are package-wide defaults", {
  defaults <- fastPLS:::.svd_control_defaults()
  expect_identical(defaults$rsvd_oversample, 20L)
  expect_identical(defaults$rsvd_power, 2L)
  expect_identical(formals(fastsvd)$oversample, 20L)
  expect_identical(formals(fastsvd)$power, 2L)

  resolved <- fastPLS:::.resolve_svd_control(context = "test")
  expect_identical(resolved$rsvd_oversample, 20L)
  expect_identical(resolved$rsvd_power, 2L)
})

test_that("backend controls distinguish panel evidence from general certification", {
  base <- fastPLS:::.resolve_svd_control(context = "test")

  cpu <- fastPLS:::.apply_backend_rsvd_controls(base, "cpu", "test")
  expect_identical(cpu$rsvd_oversample, 20L)
  expect_identical(cpu$rsvd_power, 2L)
  expect_true(cpu$rsvd_qualification$qualified_on_prespecified_panel)
  expect_true(cpu$rsvd_qualification$met_prespecified_panel)
  expect_false(cpu$rsvd_qualification$general_use_certified)

  expect_warning(
    cuda <- fastPLS:::.apply_backend_rsvd_controls(base, "cuda", "test"),
    "raised CUDA rSVD controls"
  )
  expect_identical(cuda$rsvd_oversample, 48L)
  expect_identical(cuda$rsvd_power, 4L)
  expect_true(cuda$rsvd_qualification$qualified_on_prespecified_panel)
  expect_false(cuda$rsvd_qualification$general_use_certified)

  explicit_weak <- fastPLS:::.resolve_svd_control(
    dots = list(oversample = 10L, power = 1L),
    context = "test"
  )
  expect_warning(
    cuda_weak <- fastPLS:::.apply_backend_rsvd_controls(
      explicit_weak, "cuda", "test"
    ),
    "raised CUDA rSVD controls"
  )
  expect_identical(cuda_weak$rsvd_oversample, 48L)
  expect_identical(cuda_weak$rsvd_power, 4L)

  expect_warning(
    metal <- fastPLS:::.apply_backend_rsvd_controls(base, "metal", "test"),
    "no prespecified Metal qualification panel"
  )
  expect_identical(metal$rsvd_oversample, 20L)
  expect_false(metal$rsvd_qualification$qualified_on_prespecified_panel)
})

test_that("accelerated SIMPLS uses task-aware lean randomized controls", {
  set.seed(2201)
  X <- matrix(rnorm(80 * 12), 80, 12)
  Y <- matrix(rnorm(80 * 3), 80, 3)

  fit_default <- pls(X, Y, ncomp = 2, method = "simpls")
  expect_identical(fit_default$diagnostics$rsvd$oversample, 10L)
  expect_identical(fit_default$diagnostics$rsvd$power, 1L)
  expect_true(fit_default$diagnostics$simpls_direction$approximate_execution)

  y <- factor(rep(c("a", "b"), each = 40L))
  fit_class <- pls(X, y, ncomp = 2, method = "simpls")
  expect_identical(fit_class$diagnostics$rsvd$oversample, 10L)
  expect_identical(fit_class$diagnostics$rsvd$power, 2L)

  fit_explicit <- pls(
    X, Y, ncomp = 2, method = "simpls",
    oversample = 6L, power = 1L
  )
  expect_identical(fit_explicit$diagnostics$rsvd$oversample, 6L)
  expect_identical(fit_explicit$diagnostics$rsvd$power, 1L)
})
