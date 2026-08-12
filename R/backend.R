#' Configure the default fastPLS execution backend
#'
#' An explicit function argument takes precedence over
#' `options(backend = ...)`, then `BACKEND`; CPU is the final default.
#' Package-specific settings remain supported as compatibility fallbacks.
#'
#' @param backend Optional backend: `"cpu"`, `"cuda"`, or `"metal"`.
#' @return The active backend. Setting returns the previous option invisibly.
#' @export
fastPLS_backend <- function(backend = NULL) {
  if (is.null(backend)) return(.fastpls_resolve_backend(NULL))
  backend <- .fastpls_validate_backend(backend, "backend")
  old <- getOption("backend", NULL)
  options(backend = backend)
  invisible(old)
}

.fastpls_validate_backend <- function(backend, label = "backend") {
  backend <- tolower(as.character(backend))
  if (length(backend) != 1L || is.na(backend) || !nzchar(backend) ||
      !backend %in% c("cpu", "cuda", "metal")) {
    stop("`", label, "` must be one of \"cpu\", \"cuda\", or \"metal\".", call. = FALSE)
  }
  backend
}

.fastpls_resolve_backend <- function(backend = NULL, allow_auto = FALSE) {
  if (!is.null(backend) && length(backend) == 1L) {
    value <- tolower(as.character(backend))
    if (allow_auto && identical(value, "auto")) return("auto")
    return(.fastpls_validate_backend(value))
  }
  option <- getOption("backend", NULL)
  if (!is.null(option)) return(.fastpls_validate_backend(option, "option backend"))
  legacy_option <- getOption("fastPLS.backend", NULL)
  if (!is.null(legacy_option)) return(.fastpls_validate_backend(legacy_option, "option fastPLS.backend"))
  environment <- Sys.getenv("BACKEND", unset = "")
  if (nzchar(environment)) return(.fastpls_validate_backend(environment, "BACKEND"))
  legacy_environment <- Sys.getenv("FASTPLS_BACKEND", unset = "")
  if (nzchar(legacy_environment)) return(.fastpls_validate_backend(legacy_environment, "FASTPLS_BACKEND"))
  "cpu"
}
