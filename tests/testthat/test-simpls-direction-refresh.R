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
  expect_identical(rule$rule, "warm_started_rank_one_randomized_refresh")
  expect_identical(rule$directions_per_solve, 1L)
  expect_true(rule$warm_start)
  expect_false(rule$adaptive_block_refresh)
  expect_identical(rule$seed_rule, "seed_plus_component_index")
  expect_true("cached_rank_one_deflation_product" %in% rule$active_optimizations)
  expect_true(rule$approximate_execution)
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
    "warm_started_rank_one_randomized_refresh"
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
    expect_false(rule$adaptive_block_refresh, info = backend)
    expect_identical(rule$seed_rule, "seed_plus_component_index", info = backend)
    if (identical(backend, "cpu")) {
      expect_identical(rule$rule, "warm_started_rank_one_randomized_refresh")
      expect_true(rule$warm_start)
    } else {
      expect_identical(rule$rule, "resident_rank_one_randomized_refresh")
      expect_false(rule$warm_start)
    }
  }
})

test_that("Metal source retains fresh resident rank-one refresh", {
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
    requested_components = 100L
  )
  expect_identical(rule$rule, "resident_batched_randomized_refresh")
  expect_identical(rule$directions_per_solve, 8L)
  expect_true(rule$adaptive_block_refresh)
  expect_false(rule$warm_start)
})
