test_that("the compiled CPU numerical library is reported", {
    backend <- fastPLS_blas(details = FALSE)

    expect_type(backend, "character")
    expect_length(backend, 1L)
    expect_true(backend %in% c("Accelerate", "OpenBLAS", "R BLAS/LAPACK"))

    information <- fastPLS_blas()
    expect_named(
        information,
        c(
            "backend", "version", "configuration", "core", "parallel",
            "threads", "library"
        )
    )
    expect_identical(information$backend, backend)
    expect_length(information$version, 1L)
    if (identical(backend, "OpenBLAS")) {
        expect_match(information$version, "^[0-9]+[.][0-9]+[.][0-9]+")
        expect_match(information$configuration, "OpenBLAS")
        expect_true(nzchar(information$core))
        expect_true(information$threads >= 1L)
    }
})
