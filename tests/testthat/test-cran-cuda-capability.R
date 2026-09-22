test_that("CUDA capability metadata is internally consistent", {
    information <- cuda_info()
    expect_named(
        information,
        c(
            "status", "compiled", "available", "diagnostic_only",
            "device_count", "runtime_version", "driver_version",
            "no_cpu_fallback"
        )
    )
    expect_true(
        information$status %in% c(
            "available", "unavailable", "diagnostic-only"
        )
    )
    expect_identical(information$available, has_cuda())
    expect_true(isTRUE(information$no_cpu_fallback))
    if (isTRUE(information$available)) {
        expect_true(isTRUE(information$compiled))
        expect_gt(information$device_count, 0L)
    }
})

test_that("an unavailable CUDA backend never falls back to CPU", {
    skip_if(has_cuda(), "CUDA is available in this check environment")
    set.seed(901L)
    X <- matrix(rnorm(24L * 5L), nrow = 24L)
    y <- factor(rep(c("a", "b"), each = 12L))
    expect_error(
        pls(X, y, ncomp = 1L, backend = "cuda"),
        "No CPU fallback is performed",
        fixed = TRUE
    )
})
