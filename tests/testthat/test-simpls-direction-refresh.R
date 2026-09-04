test_that("SIMPLS reports the release direction-refresh rule", {
  set.seed(41)
  X <- matrix(rnorm(72 * 14), 72, 14)
  Y <- cbind(
    0.7 * X[, 1] - 0.2 * X[, 3] + rnorm(72, sd = 0.05),
    -0.4 * X[, 2] + 0.3 * X[, 5] + rnorm(72, sd = 0.05)
  )

  fit <- pls(
    X,
    Y,
    ncomp = 1:3,
    method = "simpls",
    backend = "cpu",
    svd.method = "rsvd",
    return_variance = FALSE,
    seed = 17
  )

  rule <- fit$diagnostics$simpls_direction
  expect_identical(rule$rule, "fresh_oversampled_sketch_per_component")
  expect_identical(rule$directions_per_solve, 1L)
  expect_false(rule$candidate_block_refresh)
  expect_identical(rule$seed_rule, "seed_plus_component_index")
  expect_true("cached_rank_one_deflation_product" %in% rule$active_optimizations)
  expect_true(rule$approximate_execution)
})

test_that("IRLBA starts a new direction solve for each component", {
  set.seed(45)
  X <- matrix(rnorm(72 * 14), 72, 14)
  Y <- cbind(X[, 1] + rnorm(72, sd = 0.1), X[, 2] + rnorm(72, sd = 0.1))

  fit <- pls(
    X, Y, ncomp = 1:3, method = "simpls", backend = "cpu",
    svd.method = "irlba", return_variance = FALSE, seed = 17
  )

  rule <- fit$diagnostics$simpls_direction
  expect_identical(rule$rule, "fresh_per_component")
  expect_identical(rule$directions_per_solve, 1L)
  expect_false(rule$candidate_block_refresh)
  expect_true(rule$fresh_start)
  expect_false(rule$approximate_execution)
})

test_that("rejected refresh environment variables no longer affect SIMPLS", {
  set.seed(42)
  X <- matrix(rnorm(64 * 12), 64, 12)
  Y <- cbind(X[, 1] + rnorm(64, sd = 0.1), X[, 2] + rnorm(64, sd = 0.1))

  fit_reference <- pls(
    X, Y, ncomp = 1:3, method = "simpls", backend = "cpu",
    svd.method = "rsvd", return_variance = FALSE, seed = 19
  )

  old <- Sys.getenv(
    c("FASTPLS_FAST_INCREMENTAL", "FASTPLS_FAST_ADAPTIVE_RSVD"),
    unset = NA_character_
  )
  on.exit({
    for (name in names(old)) {
      if (is.na(old[[name]])) Sys.unsetenv(name) else do.call(Sys.setenv, setNames(list(old[[name]]), name))
    }
  }, add = TRUE)
  Sys.setenv(FASTPLS_FAST_INCREMENTAL = "1", FASTPLS_FAST_ADAPTIVE_RSVD = "1")

  fit_obsolete_env <- pls(
    X, Y, ncomp = 1:3, method = "simpls", backend = "cpu",
    svd.method = "rsvd", return_variance = FALSE, seed = 19
  )

  expect_equal(fit_obsolete_env$R, fit_reference$R, tolerance = 0)
  expect_equal(fit_obsolete_env$Q, fit_reference$Q, tolerance = 0)
  expect_identical(
    fit_obsolete_env$diagnostics$simpls_direction$rule,
    "fresh_oversampled_sketch_per_component"
  )
})

test_that("available accelerator dispatches expose their SIMPLS rule", {
  set.seed(43)
  X <- matrix(rnorm(48 * 10), 48, 10)
  Y <- cbind(X[, 1] + rnorm(48, sd = 0.1), X[, 2] + rnorm(48, sd = 0.1))

  available <- c(cpu = TRUE, cuda = has_cuda(), metal = has_metal())
  for (backend in names(available)[available]) {
    fit <- suppressWarnings(pls(
      X, Y, ncomp = 1:2, method = "simpls", backend = backend,
      svd.method = "rsvd", return_variance = FALSE, seed = 23
    ))
    expect_equal(ncol(fit$R), 2L, info = backend)
    expect_equal(ncol(fit$Q), 2L, info = backend)
    expect_true(all(is.finite(fit$R)), info = backend)
    expect_true(all(is.finite(fit$Q)), info = backend)
    rule <- fit$diagnostics$simpls_direction
    expect_identical(rule$directions_per_solve, 1L, info = backend)
    expect_false(rule$candidate_block_refresh, info = backend)
    expect_true(rule$fresh_start, info = backend)
    expect_true(is.na(rule$refresh_width), info = backend)
    expect_true(is.na(rule$refresh_iterations), info = backend)
    expect_identical(rule$seed_rule, "seed_plus_component_index", info = backend)
    expect_identical(rule$rule, "fresh_oversampled_sketch_per_component")
  }
})

test_that("Metal source does not reuse a preceding component direction", {
  source_path <- testthat::test_path("..", "..", "src", "svd_metal_backend.mm.in")
  skip_if_not(file.exists(source_path), "source tree unavailable after installation")
  source <- paste(readLines(source_path, warn = FALSE), collapse = "\n")
  expect_false(grepl("has_rr_prev|rr_prev", source))
  expect_match(source, "A fresh direction avoids propagating approximation")
})

test_that("CUDA classification diagnostics report batched refresh", {
  rule <- fastPLS:::.simpls_direction_diagnostics(
    randomized = TRUE,
    backend = "cuda",
    classification = TRUE,
    training_samples = 50000L,
    response_dimension = 100L,
    predictor_dimension = 768L,
    requested_components = 100L
  )
  expect_identical(rule$rule, "fresh_cuda_candidate_block")
  expect_identical(rule$directions_per_solve, 8L)
  expect_true(rule$candidate_block_refresh)
})

test_that("CUDA moderate classification retains per-component refresh", {
  rule <- fastPLS:::.simpls_direction_diagnostics(
    randomized = TRUE,
    backend = "cuda",
    classification = TRUE,
    training_samples = 5000L,
    predictor_dimension = 128L,
    response_dimension = 50L,
    requested_components = 50L
  )
  expect_identical(rule$rule, "fresh_oversampled_sketch_per_component")
  expect_false(rule$candidate_block_refresh)
})

test_that("CUDA massive cross-covariance diagnostics report rank-one refresh", {
  rule <- fastPLS:::.simpls_direction_diagnostics(
    randomized = TRUE,
    backend = "cuda",
    classification = FALSE,
    training_samples = 1200L,
    predictor_dimension = 13000L,
    response_dimension = 28355L,
    requested_components = 165L,
    power = 2L
  )
  expect_identical(rule$rule, "fresh_cuda_rank_one_refresh")
  expect_identical(rule$refresh_width, 1L)
  expect_identical(rule$refresh_iterations, 2L)
  expect_true(rule$fresh_start)
})

test_that("CPU massive cross-covariance diagnostics report rank-one refresh", {
  rule <- fastPLS:::.simpls_direction_diagnostics(
    randomized = TRUE,
    backend = "cpu",
    classification = FALSE,
    training_samples = 1200L,
    predictor_dimension = 13000L,
    response_dimension = 28355L,
    requested_components = 50L,
    power = 2L
  )
  expect_identical(rule$rule, "fresh_cpu_rank_one_refresh")
  expect_identical(rule$refresh_width, 1L)
  expect_identical(rule$refresh_iterations, 2L)
  expect_identical(rule$seed_rule, "seed_plus_component_index")
  expect_true(rule$fresh_start)
})

test_that("Metal resident SIMPLS diagnostics report the native rank-one route", {
  rule <- fastPLS:::.simpls_direction_diagnostics(
    randomized = TRUE,
    backend = "metal",
    classification = FALSE,
    training_samples = 1200L,
    predictor_dimension = 13000L,
    response_dimension = 28355L,
    requested_components = 50L,
    route_mode = "metal_resident_rank_one_simpls",
    power = 1L
  )
  expect_identical(rule$rule, "fresh_metal_rank_one_refresh")
  expect_identical(rule$refresh_width, 1L)
  expect_identical(rule$refresh_iterations, 1L)
  expect_identical(
    rule$seed_rule,
    "single_seed_fresh_component_sequence"
  )
  expect_true(rule$fresh_start)
})

test_that("CUDA rSVD repeats exactly when the seed is fixed", {
  skip_if_not(isTRUE(has_cuda()), "CUDA backend is unavailable")
  set.seed(20260902)
  X <- matrix(rnorm(180 * 24), 180, 24)
  Y <- matrix(rnorm(180 * 5), 180, 5)
  fit_once <- function() {
    pls(
      X, Y,
      ncomp = 4L,
      method = "simpls",
      backend = "cuda",
      svd.method = "rsvd",
      seed = 77L,
      fit = TRUE,
      return_variance = FALSE
    )
  }
  first <- fit_once()
  second <- fit_once()
  expect_equal(first$Yfit, second$Yfit, tolerance = 0)
})

test_that("SIMPLS norm guard rejects unusable directions", {
  expect_true(fastPLS:::.is_usable_simpls_norm(1))
  expect_false(fastPLS:::.is_usable_simpls_norm(0))
  expect_false(fastPLS:::.is_usable_simpls_norm(NA_real_))
  expect_false(fastPLS:::.is_usable_simpls_norm(Inf))
})

test_that("massive CUDA diagnostics report the executed refresh iterations", {
  direction <- fastPLS:::.simpls_direction_diagnostics(
    randomized = TRUE,
    backend = "cuda",
    classification = FALSE,
    training_samples = 1200L,
    predictor_dimension = 13000L,
    response_dimension = 28355L,
    requested_components = 50L,
    route_mode = "cuda_resident_rank_one_simpls",
    power = 6L
  )
  expect_identical(direction$rule, "fresh_cuda_rank_one_refresh")
  expect_true(direction$fresh_start)
  expect_identical(direction$refresh_iterations, 6L)
})

test_that("SIMPLS-family backends attach fresh-start diagnostics", {
  set.seed(44)
  X <- matrix(rnorm(54 * 9), 54, 9)
  Y <- cbind(X[, 1] + rnorm(54, sd = 0.1), X[, 2] + rnorm(54, sd = 0.1))

  available <- c(cpu = TRUE, cuda = has_cuda(), metal = has_metal())
  for (backend in names(available)[available]) {
    for (method in c("simpls", "opls", "kernelpls")) {
      fit <- expect_no_warning(pls(
        X, Y,
        ncomp = 1:2,
        method = method,
        kernel = "rbf",
        backend = backend,
        svd.method = "rsvd",
        return_variance = FALSE,
        seed = 29
      ))
      direction <- fit$diagnostics$simpls_direction
      expected_optimizations <- c(
        "cached_rank_one_deflation_product",
        "incremental_coefficient_path",
        "conditional_crossproduct_cache",
        "compact_prediction"
      )
      expect_identical(
        direction$rule,
        "fresh_oversampled_sketch_per_component",
        info = paste(method, backend)
      )
      expect_true(direction$fresh_start, info = paste(method, backend))
      expect_identical(
        direction$directions_per_solve,
        1L,
        info = paste(method, backend)
      )
      expect_true(
        all(expected_optimizations %in% direction$active_optimizations),
        info = paste(method, backend)
      )
      expect_identical(
        fit$diagnostics$rsvd$backend,
        backend,
        info = paste(method, backend)
      )
    }
  }
})
