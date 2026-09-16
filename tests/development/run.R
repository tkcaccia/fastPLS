library(testthat)
library(fastPLS)

options(n.cores = 1L)

test_dir(
    file.path("tests", "development", "testthat"),
    reporter = "progress",
    stop_on_failure = TRUE,
    stop_on_warning = FALSE
)
