library(testthat)
library(fastPLS)

# Keep package checks deterministic and avoid BLAS/OpenMP oversubscription on
# shared CRAN workers. Dedicated parallel tests override this option locally.
options(n.cores = 1L)

test_check("fastPLS")
