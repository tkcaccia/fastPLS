# fastPLS SVD Acceleration Design (CPU RSVD + Optional CUDA RSVD)

## 1) SVD Call-Site Map (authoritative: `tkcaccia/fastPLS`)

| File | Function | Matrix at SVD site | Requested rank | SVD outputs used downstream |
|---|---|---:|---:|---|
| `src/fastPLS.cpp` | `pls_model1()` | `S = t(Xtrain) %*% Ytrain` with shape `(p x m)` | `max(ncomp)` | `U(:,1:k)` becomes `R`; `V(:,1:k)` becomes `Q`; `s` not used; then `T = Xtrain %*% R` and `B = R * (...) * t(Q)` |
| `src/fastPLS.cpp` | `pls_model2()` (inside component loop) | deflated `S` with shape `(p x m)` | `1` (top singular vector only) | first left singular vector `u1` only (`rr = u1`); no singular values/right vectors used |
| `inst/include/fastPLS.h` | `fastPLS::pls_light()` | `S = t(Xtrain) %*% Ytrain` with shape `(p x m)` | `ncomp` | same use pattern as `pls_model1()` (`R`, `Q`, then regression coefficients) |

Notes:
- The statistical procedure (centering/scaling, deflation, normalization, component extraction) is unchanged.
- Only the SVD kernel path is abstracted and accelerated.

## 2) New SVD Abstraction Layer

### New files
- `src/svd_iface.h`
- `src/svd_iface.cpp`
- `src/svd_cpu_exact.cpp`
- `src/svd_cpu_rsvd.cpp`
- `src/svd_cuda_rsvd.h`
- `src/svd_cuda_rsvd.cpp` (CUDA code guarded by `FASTPLS_HAS_CUDA`)

### API
```cpp
struct SVDResult { arma::mat U; arma::vec s; arma::mat Vt; };

SVDResult truncated_svd(
    const arma::mat& A,
    int k,
    const SVDOptions& opt,
    Backend backend
);
```

`SVDOptions` includes:
- `method`: exact or rsvd
- `oversample`
- `power_iters`
- `seed`
- `left_only` (for `pls_model2` top-left-vector extraction)
- `use_full_svd` (to preserve legacy exact behavior where needed)

### Method IDs wired to R `svd.method`
- `irlba` -> legacy IRLBA path (default, unchanged)
- `dc` -> CPU exact SVD (legacy exact)
- `cpu_exact` -> CPU exact SVD
- `cpu_rsvd` -> CPU randomized SVD
- `cuda_rsvd` -> CUDA randomized SVD

## 3) RSVD Implementation

Implemented dense block RSVD:
1. `Omega ~ N(0, 1)` with shape `(n x (k+p))`
2. `Y = A * Omega`
3. `q` power iterations: `Y = A * (A^T * Y)` (with QR stabilization)
4. `Q = qr(Y)`
5. `B = Q^T * A`
6. `B = Uhat * Sigma * V^T`
7. `U = Q * Uhat`
8. return top-`k`

CPU path uses Armadillo QR/SVD.

## 4) CUDA Strategy

`cuda_rsvd` path (guarded by `FASTPLS_HAS_CUDA`):
- Uses cuBLAS GEMM for:
  - `A * Omega`
  - `A^T * Y`
  - `A * Z`
- Keeps RSVD sampling multiplies on device.
- Transfers sampled basis `Y` back once for QR + small SVD finalization on CPU.
- This is a hybrid RSVD implementation (GPU sampling + CPU finalization), not a full cuSOLVER end-to-end SVD yet.
- If `svd.method = "cuda_rsvd"` is requested and CUDA is unavailable, `pls()` errors explicitly (no silent fallback to CPU).

Runtime check from R:
- `has_cuda()` -> TRUE only if package has CUDA build and at least one CUDA device is visible.

## 5) Build System

### CPU-only (default, Mac M1 compatible)
- No CUDA required.
- `src/Makevars` builds only C/C++ sources by default.

### CUDA-enabled (NVIDIA machine)
- Set compile/link flags so the package is built with `FASTPLS_HAS_CUDA` and CUDA libraries, for example:
  - `PKG_CPPFLAGS='-DFASTPLS_HAS_CUDA -I${CUDA_HOME}/include'`
  - `PKG_LIBS='-L${CUDA_HOME}/lib64 -lcudart -lcublas -lcusolver'`

## 6) R Interface Changes (default behavior unchanged)

Functions updated:
- `pls()`
- `optim.pls.cv()`
- `pls.double.cv()`

New/extended parameters:
- `svd.method = c("irlba", "dc", "cpu_exact", "cpu_rsvd", "cuda_rsvd")`
- `rsvd_oversample`
- `rsvd_power`
- `seed`

Default remains `svd.method = "irlba"` to preserve existing behavior.

## 7) Validation and Testing

Added tests in `tests/testthat/`:
- RSVD vs exact SVD on random dense matrices
- default behavior parity (`pls(... )` vs explicit `svd.method="irlba"`)
- `cpu_exact` vs `cpu_rsvd` PLS output closeness
- deterministic `cpu_rsvd` behavior with fixed seed
- `has_cuda()` behavior and CUDA guard path

## 8) Benchmarking

### Microbenchmark SVD kernel
```r
A <- matrix(rnorm(2000 * 300), 2000, 300)
truncated_svd_debug(A, 20, svd_method = 3L, rsvd_oversample = 10L, rsvd_power = 1L, seed = 1L, left_only = FALSE)
truncated_svd_debug(A, 20, svd_method = 4L, rsvd_oversample = 20L, rsvd_power = 2L, seed = 1L, left_only = FALSE)
```

### End-to-end PLS benchmark
```r
X <- matrix(rnorm(800 * 200), 800, 200)
Y <- matrix(rnorm(800 * 20), 800, 20)

system.time(pls(X, Y, ncomp = 1:10, svd.method = "dc"))
system.time(pls(X, Y, ncomp = 1:10, svd.method = "cpu_rsvd", rsvd_oversample = 20L, rsvd_power = 2L, seed = 1L))
```
