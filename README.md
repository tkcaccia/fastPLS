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
- `dc`:
  exact CPU SVD path through Armadillo `svd_econ()`.
- `cpu_rsvd`:
  randomized SVD on CPU (Gaussian sketch + optional power iterations + reduced
  exact SVD).
- `cuda_rsvd`:
  randomized SVD with GPU GEMM sampling (cuBLAS), then CPU QR/reduced SVD
  finalization.

Check runtime availability using `has_cuda()` and `svd_methods()`.

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
- For `simpls_fast`, tuning knobs are exposed as high-level arguments and are
  propagated to C++ via environment variables:
  `fast_block`, `fast_center_t`, `fast_reorth_v`, `fast_incremental`,
  `fast_inc_iters`, `fast_defl_cache`.

## References

- de Jong, S. (1993). SIMPLS. *Chemometrics and Intelligent Laboratory Systems*.
- Halko, N., Martinsson, P.-G., Tropp, J. A. (2011). *SIAM Review*.
- Musco, C., Musco, C. (2015). Randomized block Krylov SVD. *arXiv:1504.05477*.
- Baglama, J., Reichel, L. (2005). IRLBA. *SIAM Journal on Scientific Computing*.
