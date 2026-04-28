# fastPLS

`fastPLS` provides C++, GPU-native, and pure-R implementations of partial least
squares (PLS) with interchangeable CPU SVD backends and compact latent
prediction for large response spaces.

## Implemented PLS Algorithms

- `method = "plssvd"`:
  single truncated SVD of cross-covariance `S = X^T Y`, then coefficient
  reconstruction per requested component count.
- `method = "simpls"`:
  the optimized SIMPLS path formerly exposed as `simpls_fast`, using
  incremental deflation and compact latent prediction when coefficient storage
  would be large.
- `method = "simpls_fast"`:
  accepted as a legacy alias for `method = "simpls"`.
- `opls_*()` and `kernel_pls_*()`:
  OPLS filtering and kernel-PLS wrappers built on the same PLSSVD/SIMPLS cores.

## SVD Backends

- `irlba`:
  bundled internal IRLBA wrapper. Matrix-free xprod is intentionally disabled
  for IRLBA routes; use `cpu_rsvd` for the xprod-enabled randomized path.
- `cpu_rsvd`:
  randomized SVD on CPU (Gaussian sketch + optional power iterations + reduced
  exact SVD).
- The former hybrid `cuda_rsvd` route through `pls()` has been removed.
- GPU-native fitting is now exposed separately through:
  - `simpls_gpu()`
  - `plssvd_gpu()`

Check runtime availability using `has_cuda()` and `svd_methods()`.

## Method/Backend Summary

| Outer algorithm | Inner backend | Exact/approximate | Iterative/direct (inner) | CPU/GPU | Supported functions |
|---|---|---|---|---|---|
| `plssvd` | `irlba` | Truncated | Bundled IRLBA wrapper | CPU | `pls`, `pls_r`, `optim.pls.cv`, `pls.double.cv` |
| `plssvd` | `cpu_rsvd` | Approximate | Iterative randomized range/power + reduced exact SVD | CPU | `pls`, `pls_r`, `optim.pls.cv`, `pls.double.cv` |
| `simpls` | `irlba` | Truncated | Bundled IRLBA wrapper inside optimized SIMPLS loop | CPU | `pls`, `pls_r`, `optim.pls.cv`, `pls.double.cv` |
| `simpls` | `cpu_rsvd` | Approximate | Iterative randomized range/power + reduced exact SVD inside optimized SIMPLS loop | CPU | `pls`, `pls_r`, `optim.pls.cv`, `pls.double.cv` |
| `opls` / `kernel_pls` | `irlba` / `cpu_rsvd` | Depends on backend | Shared fastPLS core after OPLS filtering or kernel construction | CPU | `opls_*`, `kernel_pls_*` |
| `plssvd` | GPU-native | Approximate | Dedicated CUDA path with device-resident training buffers | GPU | `plssvd_gpu` |
| `simpls` | GPU-native | Approximate | Dedicated CUDA path with device-resident training buffers | GPU | `simpls_gpu` |

## Public API (current)

Main modeling functions:
- `pls()` (C++ backends)
- `pls_r()` (pure-R reference path)
- `simpls_gpu()` (GPU-native SIMPLS-fast)
- `plssvd_gpu()` (GPU-native PLSSVD)
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
- For legacy `irlba` paths, tune `irlba_work`, `irlba_maxit`, `irlba_tol`,
  `irlba_eps`, and `irlba_svtol` from the R API.
- `simpls_fast` now permanently uses the former incremental-deflation
  configuration. Legacy `fast_*` tuning arguments are accepted for backward
  compatibility but are deprecated and ignored.
- For high-dimensional multivariate responses, fastPLS can omit the full
  coefficient cube `B` and predict from compact latent factors instead. Set
  `FASTPLS_STORE_B=always` to force legacy coefficient storage,
  `FASTPLS_STORE_B=never` to force compact storage, or tune the automatic
  threshold with `FASTPLS_STORE_B_MAX_MB` (default: `256`).
- The GPU-native APIs reset their CUDA workspace after each fit and keep
  training buffers in double precision.

## References

- de Jong, S. (1993). SIMPLS. *Chemometrics and Intelligent Laboratory Systems*.
- Halko, N., Martinsson, P.-G., Tropp, J. A. (2011). *SIAM Review*.
- Musco, C., Musco, C. (2015). Randomized block Krylov SVD. *arXiv:1504.05477*.
- Baglama, J., Reichel, L. (2005). IRLBA. *SIAM Journal on Scientific Computing*.
