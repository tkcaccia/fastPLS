test_that("fastPLS backend precedence is explicit, option, environment, CPU", {
  old_option <- getOption("backend", NULL)
  old_env <- Sys.getenv("FASTPLS_BACKEND", unset = NA_character_)
  on.exit({
    options(backend = old_option)
    if (is.na(old_env)) Sys.unsetenv("FASTPLS_BACKEND") else Sys.setenv(FASTPLS_BACKEND = old_env)
  }, add = TRUE)
  options(backend = NULL); Sys.unsetenv("FASTPLS_BACKEND")
  expect_identical(fastPLS_backend(), "cpu")
  Sys.setenv(FASTPLS_BACKEND = "metal"); expect_identical(fastPLS_backend(), "metal")
  options(backend = "cuda"); expect_identical(fastPLS_backend(), "cuda")
  expect_identical(fastPLS:::.fastpls_resolve_backend("cpu"), "cpu")
  expect_error(fastPLS:::.fastpls_resolve_backend("auto"), "must be one of")
})

test_that("generic backend option controls fastPLS and explicit values win", {
  old_option <- getOption("backend", NULL)
  old_env <- Sys.getenv("FASTPLS_BACKEND", unset = NA_character_)
  on.exit({
    options(backend = old_option)
    if (is.na(old_env)) Sys.unsetenv("FASTPLS_BACKEND") else Sys.setenv(FASTPLS_BACKEND = old_env)
  }, add = TRUE)
  Sys.setenv(FASTPLS_BACKEND = "metal")
  options(backend = "cuda")
  expect_identical(fastPLS_backend(), "cuda")
  expect_identical(fastPLS:::.fastpls_resolve_backend("cpu"), "cpu")
})

test_that("CPU core option is validated and applied to thread runtimes", {
  old_cores <- getOption("cores", NULL)
  old_blas_cores <- if (requireNamespace("RhpcBLASctl", quietly = TRUE)) {
    RhpcBLASctl::blas_get_num_procs()
  } else {
    NULL
  }
  variables <- c("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "GOTO_NUM_THREADS",
                 "MKL_NUM_THREADS", "BLIS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")
  old_environment <- Sys.getenv(variables, unset = NA_character_)
  on.exit({
    options(cores = old_cores)
    if (!is.null(old_blas_cores)) {
      RhpcBLASctl::blas_set_num_threads(old_blas_cores)
      RhpcBLASctl::omp_set_num_threads(old_blas_cores)
    }
    for (variable in variables) {
      value <- old_environment[[variable]]
      if (is.na(value)) Sys.unsetenv(variable) else do.call(Sys.setenv,
        stats::setNames(list(value), variable))
    }
  }, add = TRUE)
  options(cores = 3L)
  expect_identical(fastPLS:::.fastpls_apply_cpu_cores(), 3L)
  expect_true(all(Sys.getenv(variables) == "3"))
  options(cores = 1.5)
  expect_error(fastPLS:::.fastpls_apply_cpu_cores(), "positive integer")
})

test_that("public fitting functions defer omitted backends", {
  expect_null(formals(fastsvd)$backend)
  expect_null(formals(pls)$backend)
  expect_null(formals(pls.single.cv)$backend)
  expect_null(formals(pls.double.cv)$backend)
  expect_null(formals(getS3method("predict", "fastPLS"))$backend)
})
