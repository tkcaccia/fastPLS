configure_win_source <- function() {
    root <- normalizePath(
        file.path(testthat::test_path(), "..", ".."),
        mustWork = FALSE
    )
    list(
        script = file.path(root, "configure.win"),
        makevars = file.path(root, "src", "Makevars.win.in")
    )
}

run_configure_win_probe <- function(target_arch, openblas_arch,
                                    use_openblas = "auto") {
    source <- configure_win_source()
    testthat::skip_if_not(file.exists(source$script))
    testthat::skip_if_not(file.exists(source$makevars))
    shell <- Sys.which("sh")
    testthat::skip_if(shell == "", "A POSIX shell is required")

    stage <- tempfile("fastpls-configure-win-")
    dir.create(file.path(stage, "src"), recursive = TRUE)
    file.copy(source$script, file.path(stage, "configure.win"))
    file.copy(source$makevars, file.path(stage, "src", "Makevars.win.in"))

    root <- file.path(stage, paste0("rtools-", openblas_arch))
    include <- file.path(root, "include", "openblas")
    library <- file.path(root, "lib")
    dir.create(include, recursive = TRUE)
    dir.create(library, recursive = TRUE)
    writeLines("/* configure probe */", file.path(include, "openblas_config.h"))
    writeBin(raw(), file.path(library, "libopenblas.a"))

    old <- setwd(stage)
    on.exit(setwd(old), add = TRUE)
    output <- suppressWarnings(
        system2(
            shell,
            "configure.win",
            stdout = TRUE,
            stderr = TRUE,
            env = c(
                paste0("FASTPLS_CONFIGURE_TARGET_ARCH=", target_arch),
                "FASTPLS_USE_CUDA=0",
                paste0("FASTPLS_USE_OPENBLAS=", use_openblas),
                paste0("R_TOOLS_SOFT=", root),
                "OPENBLAS_ROOT=",
                "PATH=/usr/bin:/bin"
            )
        )
    )
    status <- attr(output, "status")
    if (is.null(status)) status <- 0L
    makevars <- file.path(stage, "src", "Makevars.win")
    list(
        output = output,
        status = status,
        makevars = if (file.exists(makevars)) {
            readLines(makevars)
        } else {
            character()
        }
    )
}

test_that("Windows ARM64 rejects an x64 OpenBLAS archive", {
    result <- run_configure_win_probe("aarch64", "x86_64")

    expect_identical(result$status, 0L)
    output <- paste(result$output, collapse = "\n")
    expect_match(output, "target architecture aarch64")
    expect_match(output, "using the BLAS/LAPACK supplied by R")
    expect_false(any(grepl("FASTPLS_USE_OPENBLAS", result$makevars)))
})

test_that("Windows x64 accepts an x64 OpenBLAS archive", {
    result <- run_configure_win_probe("x86_64", "x86_64")

    expect_identical(result$status, 0L)
    expect_match(paste(result$output, collapse = "\n"), "using OpenBLAS")
    expect_true(any(grepl("FASTPLS_USE_OPENBLAS", result$makevars)))
})

test_that("required OpenBLAS rejects a cross-architecture archive", {
    result <- run_configure_win_probe("aarch64", "x86_64", "1")

    expect_identical(result$status, 1L)
    expect_match(
        paste(result$output, collapse = "\n"),
        "matching aarch64"
    )
})
