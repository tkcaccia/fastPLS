test_that("float32 capability assessment is shape based", {
  safe <- fastPLS:::.float32_capability_assessment(
    method = "plssvd",
    backend = "cpu",
    svd_method = "cpu_rsvd",
    q = 22L,
    ncomp = 20L,
    classification = TRUE
  )
  expect_identical(safe$status, "validated")
  expect_identical(safe$action, "allow")
  expect_length(safe$warnings, 0L)

  risky <- fastPLS:::.float32_capability_assessment(
    method = "simpls",
    backend = "cuda",
    svd_method = "cpu_rsvd",
    q = 28355L,
    ncomp = 100L,
    classification = FALSE
  )
  expect_identical(risky$status, "failed")
  expect_identical(risky$action, "warn")
  expect_match(risky$warnings, "numerical-risk")

  plssvd <- fastPLS:::.float32_capability_assessment(
    method = "plssvd",
    backend = "cuda",
    svd_method = "cpu_rsvd",
    q = 28355L,
    ncomp = 100L,
    classification = FALSE
  )
  expect_identical(plssvd$status, "experimental")
  expect_match(plssvd$warnings, "performance-risk")
})

test_that("float32 classification and nonlinear kernels are flagged", {
  classification <- fastPLS:::.float32_capability_assessment(
    method = "simpls",
    backend = "cpu",
    svd_method = "cpu_rsvd",
    q = 100L,
    ncomp = 50L,
    classification = TRUE
  )
  expect_identical(classification$status, "experimental")
  expect_match(classification$warnings, "five percentage points")

  nonlinear <- fastPLS:::.float32_capability_assessment(
    method = "kernelpls",
    backend = "metal",
    svd_method = "irlba",
    q = 1L,
    ncomp = 5L,
    classification = FALSE,
    kernel = "rbf"
  )
  expect_identical(nonlinear$status, "hybrid")
  expect_identical(nonlinear$execution, "hybrid_host_device")
  expect_true(any(grepl("n-by-n Gram matrix", nonlinear$warnings)))
})

test_that("float32 unavailable and hybrid routes are explicit", {
  cuda_irlba <- fastPLS:::.float32_capability_assessment(
    method = "plssvd",
    backend = "cuda",
    svd_method = "irlba",
    q = 3L,
    ncomp = 2L,
    classification = TRUE
  )
  expect_identical(cuda_irlba$status, "unavailable")
  expect_identical(cuda_irlba$action, "error")
  expect_match(cuda_irlba$errors, "CUDA supports")

  metal_lda <- fastPLS:::.float32_capability_assessment(
    method = "simpls",
    backend = "metal",
    svd_method = "cpu_rsvd",
    q = 3L,
    ncomp = 2L,
    classification = TRUE,
    classifier = "lda"
  )
  expect_identical(metal_lda$status, "hybrid")
  expect_identical(metal_lda$execution, "hybrid_device_cpu_lda")
  expect_true(any(grepl("LDA is hybrid", metal_lda$warnings)))

  windows_opls <- fastPLS:::.float32_capability_assessment(
    method = "opls",
    backend = "cpu",
    svd_method = "cpu_rsvd",
    q = 1L,
    ncomp = 2L,
    classification = FALSE,
    os_type = "windows"
  )
  expect_identical(windows_opls$status, "unavailable")
  expect_identical(windows_opls$action, "error")
})

test_that("float32 risk warnings are emitted once per route", {
  state <- fastPLS:::.float32_warning_state
  rm(list = ls(envir = state, all.names = TRUE), envir = state)
  on.exit(rm(list = ls(envir = state, all.names = TRUE), envir = state))

  Y <- matrix(0, nrow = 1L, ncol = 10000L)
  expect_warning(
    fastPLS:::.warn_float32_capability(
      method = "simpls",
      backend = "cuda",
      svd_method = "cpu_rsvd",
      Ytrain = Y,
      ncomp = 50L
    ),
    "numerical-risk"
  )
  expect_silent(
    fastPLS:::.warn_float32_capability(
      method = "simpls",
      backend = "cuda",
      svd_method = "cpu_rsvd",
      Ytrain = Y,
      ncomp = 50L
    )
  )
})
