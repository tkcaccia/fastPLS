library(fastPLS)

information <- cuda_info()
stopifnot(
    identical(information$status, "available"),
    isTRUE(information$compiled),
    isTRUE(information$available),
    !isTRUE(information$diagnostic_only),
    information$device_count > 0L,
    isTRUE(information$no_cpu_fallback),
    isTRUE(has_cuda())
)

# This internal primitive has no CPU implementation in a CUDA build. Agreement
# with base R therefore verifies both genuine device execution and its result.
set.seed(20260922L)
left <- matrix(rnorm(35L), nrow = 7L)
right <- matrix(rnorm(20L), nrow = 5L)
gpu_product <- fastPLS:::.cuda_matmul(left, right)
cpu_reference <- left %*% right
stopifnot(
    isTRUE(all.equal(gpu_product, cpu_reference, tolerance = 1e-10))
)

X <- matrix(rnorm(72L * 10L), nrow = 72L)
Y <- cbind(
    0.8 * X[, 1L] - 0.3 * X[, 2L],
    -0.5 * X[, 3L] + 0.2 * X[, 4L]
)
cuda_fit <- pls(
    X, Y, ncomp = 1:3, method = "simpls", backend = "cuda",
    seed = 20260922L
)
cuda_prediction <- predict(cuda_fit, X, backend = "cuda")$Ypred
stopifnot(
    identical(cuda_fit$diagnostics$rsvd$backend, "cuda"),
    all(is.finite(cuda_prediction))
)

cat("Strict CUDA smoke test passed.\n")
print(information)
