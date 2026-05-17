test_that("pls records active backend-control settings", {
  old_store_b <- Sys.getenv("FASTPLS_STORE_B", unset = NA_character_)
  old_deflcache <- Sys.getenv("FASTPLS_FAST_DEFLCACHE", unset = NA_character_)
  on.exit({
    if (is.na(old_store_b)) Sys.unsetenv("FASTPLS_STORE_B") else Sys.setenv(FASTPLS_STORE_B = old_store_b)
    if (is.na(old_deflcache)) Sys.unsetenv("FASTPLS_FAST_DEFLCACHE") else Sys.setenv(FASTPLS_FAST_DEFLCACHE = old_deflcache)
  }, add = TRUE)

  Sys.setenv(FASTPLS_STORE_B = "0", FASTPLS_FAST_DEFLCACHE = "0")
  set.seed(11)
  X <- matrix(rnorm(60 * 8), 60, 8)
  y <- factor(sample(letters[1:3], 60, replace = TRUE))

  fit <- pls(
    X,
    y,
    ncomp = 2,
    method = "simpls",
    backend = "cpu",
    svd.method = "rsvd",
    return_variance = FALSE
  )

  expect_s3_class(fit, "fastPLS")
  expect_s3_class(fit$backend_control, "fastPLSBackendControl")
  expect_equal(fit$backend_control$overrides$backend, "cpu")
  expect_equal(fit$backend_control$overrides$svd.method, "cpu_rsvd")

  env <- fit$backend_control$env
  store_b <- env[env$name == "FASTPLS_STORE_B", , drop = FALSE]
  deflcache <- env[env$name == "FASTPLS_FAST_DEFLCACHE", , drop = FALSE]
  expect_equal(store_b$value, "0")
  expect_true(store_b$overridden)
  expect_equal(deflcache$value, "0")
  expect_true(deflcache$overridden)
})
