# fastPLS

`fastPLS` provides C++ and pure-R implementations of partial least squares (PLS)
with interchangeable SVD backends.

## Implemented PLS Algorithms

- `method = "plssvd"`:
  single truncated SVD of cross-covariance `S = X^T Y`, then coefficient
  reconstruction per requested component count.
- `method = "simpls"`:
  iterative SIMPLS-style deflation of `S`, extracting one latent direction per
  component and updating coefficients cumulatively.
- `method = "simpls_fast"` (experimental):
  SIMPLS-like deflation with block direction refresh and optional incremental
  block-power updates.

## SVD Backends

- `irlba`:
  legacy label; in `pls_model1/2`, IRLB is used only for sufficiently large
  cross-covariance matrices. In `svd_run`/`svd_benchmark`, this label currently
  follows exact CPU dispatch.
- `arpack`:
  ARPACK-truncated SVD path through Armadillo `svds()`. 
- `cpu_rsvd`:
  randomized SVD on CPU (Gaussian sketch + optional power iterations + reduced
  exact SVD).
- `cuda_rsvd`:
  randomized SVD with GPU GEMM sampling (cuBLAS), then CPU QR/reduced SVD
  finalization.

Check runtime availability using `has_cuda()` and `svd_methods()`.

## Method/Backend Summary

| Outer algorithm | Inner backend | Exact/approximate | Iterative/direct (inner) | CPU/GPU | Supported functions |
|---|---|---|---|---|---|
| `plssvd` | `arpack` | ARPACK-truncated | Direct SVD (`svds`) | CPU | `pls`, `pls_r`, `optim.pls.cv`, `pls.double.cv` |
| `plssvd` | `irlba` | Typically exact fallback for small matrices; legacy IRLB branch for larger `S` in PLS C++ loops | Iterative Lanczos branch in selected PLS paths | CPU | `pls`, `optim.pls.cv`, `pls.double.cv` |
| `plssvd` | `cpu_rsvd` | Approximate | Iterative randomized range/power + reduced exact SVD | CPU | `pls`, `pls_r`, `optim.pls.cv`, `pls.double.cv` |
| `plssvd` | `cuda_rsvd` | Approximate | Iterative randomized sampling/power GEMM on GPU + CPU finalization | Hybrid GPU+CPU | `pls`, `optim.pls.cv`, `pls.double.cv` |
| `simpls` | `arpack` | ARPACK-truncated | Direct SVD (`svds`) per component on deflated `S` | CPU | `pls`, `pls_r`, `optim.pls.cv`, `pls.double.cv` |
| `simpls` | `irlba` | Legacy truncated branch in selected C++ paths | Iterative Lanczos in selected paths | CPU | `pls`, `optim.pls.cv`, `pls.double.cv` |
| `simpls` | `cpu_rsvd` | Approximate | Iterative randomized range/power + reduced exact SVD | CPU | `pls`, `pls_r`, `optim.pls.cv`, `pls.double.cv` |
| `simpls` | `cuda_rsvd` | Approximate | GPU randomized sampling/power + CPU finalization | Hybrid GPU+CPU | `pls`, `optim.pls.cv`, `pls.double.cv` |
| `simpls_fast` | `arpack`/`irlba`/`cpu_rsvd`/`cuda_rsvd` | Depends on backend (`arpack` exact, RSVD approximate) | Block refresh + optional incremental updates in outer loop | CPU or Hybrid | `pls`, `optim.pls.cv`, `pls.double.cv` |

Notes:
- `svd_run()`/`svd_benchmark()` are backend utility wrappers; their `irlba` label currently follows exact dispatch in utility mode.

## Public API (current)

Main modeling functions:
- `pls()` (C++ backends)
- `pls_r()` (pure-R reference path)
- `predict.fastPLS()`

Model selection and diagnostics:
- `optim.pls.cv()`
- `pls.double.cv()`
- `ViP()`
- `fastcor()`

SVD utilities:
- `svd_methods()`
- `svd_run()`
- `svd_benchmark()`
- `has_cuda()`

## Notes for Reproducible Benchmarks

- For randomized backends (`cpu_rsvd`, `cuda_rsvd`), set `seed`,
  `rsvd_oversample`, and `rsvd_power` explicitly.
- For `arpack`, `svds_tol` controls ARPACK convergence tolerance in C++ paths.
  Higher values can improve speed at the cost of looser convergence.
- For legacy `irlba` paths, tune `irlba_work`, `irlba_maxit`, `irlba_tol`,
  `irlba_eps`, and `irlba_svtol` from the R API.
- `simpls_fast` now permanently uses the former incremental-deflation
  configuration. Legacy `fast_*` tuning arguments are accepted for backward
  compatibility but are deprecated and ignored.

## References

- de Jong, S. (1993). SIMPLS. *Chemometrics and Intelligent Laboratory Systems*.
- Halko, N., Martinsson, P.-G., Tropp, J. A. (2011). *SIAM Review*.
- Musco, C., Musco, C. (2015). Randomized block Krylov SVD. *arXiv:1504.05477*.
- Baglama, J., Reichel, L. (2005). IRLBA. *SIAM Journal on Scientific Computing*.
