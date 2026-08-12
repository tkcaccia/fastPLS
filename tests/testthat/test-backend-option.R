test_that("fastPLS backend precedence is explicit, option, environment, CPU", {
  old_option <- getOption("fastPLS.backend", NULL)
  old_env <- Sys.getenv("FASTPLS_BACKEND", unset = NA_character_)
  on.exit({
    options(fastPLS.backend = old_option)
    if (is.na(old_env)) Sys.unsetenv("FASTPLS_BACKEND") else Sys.setenv(FASTPLS_BACKEND = old_env)
  }, add = TRUE)
  options(fastPLS.backend = NULL); Sys.unsetenv("FASTPLS_BACKEND")
  expect_identical(fastPLS_backend(), "cpu")
  Sys.setenv(FASTPLS_BACKEND = "metal"); expect_identical(fastPLS_backend(), "metal")
  options(fastPLS.backend = "cuda"); expect_identical(fastPLS_backend(), "cuda")
  expect_identical(fastPLS:::.fastpls_resolve_backend("cpu"), "cpu")
  expect_error(fastPLS:::.fastpls_resolve_backend("auto"), "must be one of")
})

test_that("unrelated global backend settings do not affect fastPLS", {
  old_global_option <- getOption("backend", NULL)
  old_global_env <- Sys.getenv("BACKEND", unset = NA_character_)
  old_option <- getOption("fastPLS.backend", NULL)
  old_env <- Sys.getenv("FASTPLS_BACKEND", unset = NA_character_)
  on.exit({
    options(backend = old_global_option, fastPLS.backend = old_option)
    if (is.na(old_global_env)) Sys.unsetenv("BACKEND") else Sys.setenv(BACKEND = old_global_env)
    if (is.na(old_env)) Sys.unsetenv("FASTPLS_BACKEND") else Sys.setenv(FASTPLS_BACKEND = old_env)
  }, add = TRUE)
  options(backend = "cuda", fastPLS.backend = NULL)
  Sys.setenv(BACKEND = "metal")
  Sys.unsetenv("FASTPLS_BACKEND")
  expect_identical(fastPLS_backend(), "cpu")
})

test_that("public fitting functions defer omitted backends", {
  expect_null(formals(fastsvd)$backend)
  expect_null(formals(pls)$backend)
  expect_null(formals(pls.single.cv)$backend)
  expect_null(formals(pls.double.cv)$backend)
  expect_null(formals(getS3method("predict", "fastPLS"))$backend)
})
