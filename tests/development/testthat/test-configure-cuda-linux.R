cuda_configure_fixture <- function(target_layout = TRUE, library_name = NULL) {
    source_root <- normalizePath(testthat::test_path(), mustWork = TRUE)
    while (
        !file.exists(file.path(source_root, "DESCRIPTION")) &&
            dirname(source_root) != source_root
    ) {
        source_root <- dirname(source_root)
    }
    stopifnot(file.exists(file.path(source_root, "DESCRIPTION")))
    root <- tempfile("fastpls-cuda-root-")
    project <- tempfile("fastpls-configure-project-")
    dir.create(root, recursive = TRUE)
    dir.create(file.path(project, "src"), recursive = TRUE)
    file.copy(file.path(source_root, "configure"), project)
    file.copy(
        file.path(source_root, "src", "Makevars.in"),
        file.path(project, "src")
    )

    if (target_layout) {
        include <- file.path(root, "targets", "test-linux", "include")
        if (is.null(library_name)) library_name <- "lib64"
        library <- file.path(root, "targets", "test-linux", library_name)
    } else {
        include <- file.path(root, "include")
        if (is.null(library_name)) library_name <- "lib"
        library <- file.path(root, library_name)
    }
    dir.create(include, recursive = TRUE)
    dir.create(library, recursive = TRUE)
    dir.create(file.path(root, "bin"), recursive = TRUE)
    file.create(file.path(
        include,
        c("cuda_runtime.h", "cublas_v2.h", "cusolverDn.h", "curand.h")
    ))
    file.create(file.path(
        library,
        paste0("lib", c("cudart", "cublas", "cusolver", "curand"), ".so")
    ))
    nvcc <- file.path(root, "bin", "nvcc")
    writeLines(
        c(
            "#!/bin/sh",
            "output=''",
            "while [ \"$#\" -gt 0 ]; do",
            "  if [ \"$1\" = '-o' ]; then shift; output=$1; fi",
            "  shift",
            "done",
            "[ -n \"$output\" ] && : > \"$output\"",
            "exit 0"
        ),
        nvcc
    )
    Sys.chmod(nvcc, mode = "0755")
    list(root = root, project = project, include = include, library = library)
}

run_cuda_configure <- function(fixture, extra = character()) {
    environment <- c(
        "FASTPLS_USE_OPENBLAS=0",
        "FASTPLS_USE_METAL=0",
        "FASTPLS_USE_CUDA=1",
        "FASTPLS_REQUIRE_CUDA=1",
        paste0("CUDA_HOME=", fixture$root),
        extra
    )
    old <- setwd(fixture$project)
    on.exit(setwd(old), add = TRUE)
    suppressWarnings(system2(
        Sys.which("sh"), "./configure", env = environment,
        stdout = TRUE, stderr = TRUE
    ))
}

command_status <- function(output) {
    status <- attr(output, "status")
    if (is.null(status)) 0L else status
}

configure_output <- function(project, environment) {
    old <- setwd(project)
    on.exit(setwd(old), add = TRUE)
    suppressWarnings(system2(
        Sys.which("sh"), "./configure", env = environment,
        stdout = TRUE, stderr = TRUE
    ))
}

test_that("a POSIX shell is available for configure tests", {
    skip_if(Sys.which("sh") == "", "A POSIX shell is required")
    succeed()
})

test_that("configure discovers target-specific CUDA include and libraries", {
    skip_on_os("windows")
    skip_if(Sys.which("sh") == "", "A POSIX shell is required")
    for (library_name in c("lib", "lib64")) {
        fixture <- cuda_configure_fixture(
            target_layout = TRUE,
            library_name = library_name
        )
        output <- run_cuda_configure(fixture)
        expect_identical(command_status(output), 0L)
        makevars <- readLines(file.path(fixture$project, "src", "Makevars"))
        expect_true(any(grepl(fixture$include, makevars, fixed = TRUE)))
        expect_true(any(grepl(fixture$library, makevars, fixed = TRUE)))
        expect_true(any(grepl("-Wl,-rpath", makevars, fixed = TRUE)))
        expect_true(any(grepl(
            "CUDA_HOST_CXX = /usr/bin/", makevars, fixed = TRUE
        )))
        unlink(c(fixture$root, fixture$project), recursive = TRUE)
    }
})

test_that("strict CUDA configuration rejects an incomplete toolkit", {
    skip_on_os("windows")
    skip_if(Sys.which("sh") == "", "A POSIX shell is required")
    fixture <- cuda_configure_fixture(target_layout = FALSE)
    on.exit(unlink(c(fixture$root, fixture$project), recursive = TRUE))
    unlink(file.path(fixture$library, "libcusolver.so"))
    output <- run_cuda_configure(fixture)
    expect_false(is.null(attr(output, "status")))
    expect_true(any(grepl("CUDA support cannot be built", output, fixed = TRUE)))
})

test_that("diagnostic-only configuration is explicit", {
    skip_on_os("windows")
    skip_if(Sys.which("sh") == "", "A POSIX shell is required")
    fixture <- cuda_configure_fixture(target_layout = FALSE)
    on.exit(unlink(c(fixture$root, fixture$project), recursive = TRUE))
    output <- configure_output(
        fixture$project,
        c(
            "FASTPLS_USE_OPENBLAS=0", "FASTPLS_USE_METAL=0",
            "FASTPLS_CUDA_DIAGNOSTIC_ONLY=1"
        )
    )
    expect_identical(command_status(output), 0L)
    makevars <- readLines(file.path(fixture$project, "src", "Makevars"))
    expect_true(any(grepl("FASTPLS_CUDA_DIAGNOSTIC_ONLY", makevars)))
})
