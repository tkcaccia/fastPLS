#' Configure the default fastPLS execution backend
#'
#' An explicit function argument takes precedence over
#' `options(backend = ...)`, then `FASTPLS_BACKEND`; CPU is the final
#' default.
#'
#' For CPU execution, `options(cores = n)` requests `n` threads from the
#' linked BLAS and OpenMP runtimes. Matrix operations can use those threads
#' when the installed numerical library supports runtime thread control;
#' sequential PLS deflation steps remain serial.
#'
#' @param backend Optional backend: `"cpu"`, `"cuda"`, or `"metal"`.
#' @return The active backend. Setting returns the previous option invisibly.
#' @examples
#' current <- fastPLS_backend()
#' current
#' previous <- fastPLS_backend("cpu")
#' fastPLS_backend(previous)
#' @export
fastPLS_backend <- function(backend = NULL) {
    if (is.null(backend)) {
        return(.fastpls_resolve_backend(NULL))
    }
    backend <- .fastpls_validate_backend(backend, "backend")
    old <- getOption("backend", NULL)
    options(backend = backend)
    invisible(old)
}

.fastpls_validate_backend <- function(backend, label = "backend") {
    backend <- tolower(as.character(backend))
    if (
        length(backend) != 1L ||
            is.na(backend) ||
            !nzchar(backend) ||
            !backend %in% c("cpu", "cuda", "metal")
    ) {
        stop(
            "`",
            label,
            "` must be one of \"cpu\", \"cuda\", or \"metal\".",
            call. = FALSE
        )
    }
    backend
}

.fastpls_resolve_backend <- function(backend = NULL, allow_auto = FALSE) {
    if (!is.null(backend) && length(backend) == 1L) {
        value <- tolower(as.character(backend))
        if (allow_auto && identical(value, "auto")) {
            return("auto")
        }
        return(.fastpls_validate_backend(value))
    }
    option <- getOption("backend", NULL)
    if (!is.null(option)) {
        return(.fastpls_validate_backend(option, "option backend"))
    }
    environment <- Sys.getenv("FASTPLS_BACKEND", unset = "")
    if (nzchar(environment)) {
        return(.fastpls_validate_backend(environment, "FASTPLS_BACKEND"))
    }
    "cpu"
}

.fastpls_validate_cores <- function(cores) {
    if (
        length(cores) != 1L ||
            !is.numeric(cores) ||
            is.na(cores) ||
            !is.finite(cores) ||
            cores < 1 ||
            cores != floor(cores)
    ) {
        stop("`options(cores = ...)` must contain one positive integer.",
            call. = FALSE)
    }
    as.integer(cores)
}

.fastpls_cpu_cores <- function() {
    cores <- getOption("cores", NULL)
    if (is.null(cores)) {
        return(NULL)
    }
    .fastpls_validate_cores(cores)
}

.fastpls_apply_cpu_cores <- function() {
    cores <- .fastpls_cpu_cores()
    if (is.null(cores)) {
        return(invisible(NULL))
    }
    value <- as.character(cores)
    do.call(
        Sys.setenv,
        as.list(stats::setNames(
            rep(value, 6L),
            c(
                "OMP_NUM_THREADS",
                "OPENBLAS_NUM_THREADS",
                "GOTO_NUM_THREADS",
                "MKL_NUM_THREADS",
                "BLIS_NUM_THREADS",
                "VECLIB_MAXIMUM_THREADS"
            )
        ))
    )
    if (requireNamespace("RhpcBLASctl", quietly = TRUE)) {
        try(RhpcBLASctl::blas_set_num_threads(cores), silent = TRUE)
        try(RhpcBLASctl::omp_set_num_threads(cores), silent = TRUE)
    }
    invisible(cores)
}
