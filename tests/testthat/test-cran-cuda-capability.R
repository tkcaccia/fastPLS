test_that("CUDA capability metadata is internally consistent", {
    information <- cuda_info()
    expect_named(
        information,
        c(
            "status", "compiled", "available", "diagnostic_only",
            "device_count", "runtime_version", "driver_version",
            "no_cpu_fallback", "hardware_available", "libraries"
        )
    )
    expect_true(
        information$status %in% c(
            "available", "unavailable", "diagnostic-only"
        )
    )
    expect_identical(
        information$available,
        fastPLS:::.fastpls_backend_available("cuda")
    )
    expect_true(isTRUE(information$no_cpu_fallback))
    expect_s3_class(information$libraries, "data.frame")
    expect_true(all(c(
        "component", "loadable", "required_by_fastPLS"
    ) %in% names(information$libraries)))
    expect_setequal(
        information$libraries$component[
            information$libraries$required_by_fastPLS
        ],
        c("driver", "runtime", "cublas", "cusolver", "curand")
    )
    if (isTRUE(information$available)) {
        expect_true(isTRUE(information$compiled))
        expect_gt(information$device_count, 0L)
    }
})

test_that("an unavailable CUDA backend never falls back to CPU", {
    expect_type(fastPLS:::set_cpu_threads_cpp(1L), "character")
    skip_if(
        fastPLS:::.fastpls_backend_available("cuda"),
        "CUDA is available in this check environment"
    )
    set.seed(901L)
    X <- matrix(rnorm(24L * 5L), nrow = 24L)
    y <- factor(rep(c("a", "b"), each = 12L))
    expect_error(
        pls(X, y, ncomp = 1L, backend = "cuda", n.cores = 1L),
        "No CPU fallback is performed",
        fixed = TRUE
    )
})

test_that("native accelerator probes return scalar availability", {
    expect_true(fastPLS:::.fastpls_backend_available("cpu"))
    expect_type(fastPLS:::.fastpls_backend_available("cuda"), "logical")
    expect_type(fastPLS:::.fastpls_backend_available("metal"), "logical")
})

test_that("system discovery is absent from numerical backend checks", {
    local_mocked_bindings(
        has_cuda = function() stop("unexpected CUDA discovery"),
        has_metal = function() stop("unexpected Metal discovery"),
        cuda_info = function(...) stop("unexpected library discovery"),
        .package = "gpuinfo"
    )
    local_mocked_bindings(
        .fastpls_native_has_cuda = function() TRUE,
        .fastpls_native_has_metal = function() TRUE,
        .package = "fastPLS"
    )
    expect_true(fastPLS:::.fastpls_backend_available("cuda"))
    expect_true(fastPLS:::.fastpls_backend_available("metal"))
    expect_true(fastPLS:::.fastpls_backend_available("cpu"))
})

test_that("hardware discovery cannot enable a missing native backend", {
    local_mocked_bindings(
        has_cuda = function() TRUE,
        has_metal = function() TRUE,
        .package = "gpuinfo"
    )
    local_mocked_bindings(
        .fastpls_native_has_cuda = function() FALSE,
        .fastpls_native_has_metal = function() FALSE,
        .package = "fastPLS"
    )
    for (backend in c("cuda", "metal")) {
        expect_false(fastPLS:::.fastpls_backend_available(backend))
        expect_error(
            fastPLS:::.fastpls_require_backend_available(backend),
            "No CPU fallback is performed", fixed = TRUE
        )
    }
})

test_that("library diagnostics never override the linked CUDA runtime", {
    components <- c(
        "driver", "runtime", "cublas", "cufft", "cusolver", "curand", "nccl"
    )
    hardware <- list(
        driver_usable = FALSE,
        libraries = data.frame(component = components, loadable = FALSE)
    )
    native <- list(
        status = "available", compiled = TRUE, available = TRUE,
        diagnostic_only = FALSE, device_count = 1L,
        runtime_version = 13000L, driver_version = 13000L,
        no_cpu_fallback = TRUE
    )
    local_mocked_bindings(
        cuda_info = function(...) hardware, .package = "gpuinfo"
    )
    local_mocked_bindings(
        cuda_info_cpp = function() native, .package = "fastPLS"
    )
    information <- fastPLS::cuda_info()
    expect_identical(information$status, "available")
    expect_true(information$available)
    expect_false(information$hardware_available)
    expect_false(any(information$libraries$loadable))
    expect_false(any(information$libraries$required_by_fastPLS[
        information$libraries$component %in% c("cufft", "nccl")
    ]))

    hardware$driver_usable <- TRUE
    native$status <- "unavailable"
    native$compiled <- native$available <- FALSE
    native$device_count <- 0L
    information <- fastPLS::cuda_info()
    expect_identical(information$status, "unavailable")
    expect_false(information$available)
    expect_true(information$hardware_available)

    native$status <- "diagnostic-only"
    native$diagnostic_only <- TRUE
    information <- fastPLS::cuda_info()
    expect_identical(information$status, "diagnostic-only")
    expect_false(information$available)
})

test_that("BLAS diagnostics identify this package's linked library", {
    linked_backend <- "Accelerate"
    local_mocked_bindings(
        blas_info_cpp = function() {
            list(backend = linked_backend, library = NA_character_)
        },
        .package = "fastPLS"
    )
    expect_identical(fastPLS_blas()$library, NA_character_)
    linked_backend <- "OpenBLAS"
    expect_identical(fastPLS_blas()$library, NA_character_)
    linked_backend <- "R BLAS/LAPACK"
    expect_identical(
        fastPLS_blas()$library, unname(extSoftVersion()["BLAS"])
    )
})
