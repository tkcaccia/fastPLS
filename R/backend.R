#' Configure the default fastPLS execution backend
#'
#' An explicit function argument takes precedence over
#' `options(fastPLS.backend = ...)`, then `FASTPLS_BACKEND`; CPU is the final
#' default.
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
  if (is.null(backend)) return(.fastpls_resolve_backend(NULL))
  backend <- .fastpls_validate_backend(backend, "backend")
  old <- getOption("fastPLS.backend", NULL)
  options(fastPLS.backend = backend)
  invisible(old)
}

.fastpls_validate_backend <- function(backend, label = "backend") {
  backend <- tolower(as.character(backend))
  if (length(backend) != 1L || is.na(backend) || !nzchar(backend) ||
      !backend %in% c("cpu", "cuda", "metal")) {
    stop(
      "`", label, "` must be one of \"cpu\", \"cuda\", or \"metal\".",
      call. = FALSE
    )
  }
  backend
}

.fastpls_resolve_backend <- function(backend = NULL, allow_auto = FALSE) {
  if (!is.null(backend) && length(backend) == 1L) {
    value <- tolower(as.character(backend))
    if (allow_auto && identical(value, "auto")) return("auto")
    return(.fastpls_validate_backend(value))
  }
  option <- getOption("fastPLS.backend", NULL)
  if (!is.null(option)) {
    return(.fastpls_validate_backend(option, "option fastPLS.backend"))
  }
  environment <- Sys.getenv("FASTPLS_BACKEND", unset = "")
  if (nzchar(environment)) {
    return(.fastpls_validate_backend(environment, "FASTPLS_BACKEND"))
  }
  "cpu"
}
