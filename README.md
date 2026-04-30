# fastPLS

`fastPLS` provides R, C++, and CUDA implementations of partial least squares
models for high-dimensional regression and classification. The current standard
pipeline compares four model families:

- `plssvd`
- `simpls`
- `opls`
- `kernelpls`

The `simpls` implementation is the optimized fastPLS SIMPLS core. Older
low-level tuning arguments are kept only for source compatibility; new analyses
should use `method = "simpls"`.

## Algorithms

- `plssvd`: computes the dominant subspace of the cross-covariance
  `S = X^T Y` and reuses it for the requested component path.
- `simpls`: optimized SIMPLS with compact latent prediction and automatic
  matrix-free `xprod` selection when it reduces cross-covariance work.
- `opls_*()`: supervised orthogonal filtering followed by the selected PLS core.
- `kernel_pls_*()`: linear, RBF, or polynomial kernel construction followed by
  the selected PLS core.

For classification, factor responses are handled as PLS-DA responses. Large
response spaces use compact prediction where possible so the full coefficient
cube does not need to be stored.

## Backends

CPU backends:

- `irlba`: bundled internal IRLBA wrapper.
- `cpu_rsvd`: randomized SVD with Gaussian sketching and optional power
  iterations.
- `exact`: exact dense SVD fallback, used automatically for very small SVD
  inputs.

CUDA backends:

- `plssvd_gpu()`
- `simpls_gpu()`
- `opls_cuda()`
- `kernel_pls_cuda()`

FlashSVD-style CUDA prediction wrappers:

- `plssvd_flash_gpu()`
- `simpls_flash_gpu()`
- `opls_flash_gpu()`
- `kernel_pls_flash_gpu()`

The flash wrappers use the same fitted GPU model as the standard CUDA functions,
but apply predictions through low-rank CUDA products instead of materializing and
multiplying by the full coefficient matrix. This primarily reduces prediction
time; fit memory is still governed by the fitting backend.

The removed hybrid `svd.method = "cuda_rsvd"` route through `pls()` is no longer
supported. Use the GPU-native functions above for CUDA runs.

## Current API

Main model fitting:

- `pls()`
- `pls_r()`
- `plssvd_gpu()`
- `simpls_gpu()`
- `opls_r()`, `opls_cpp()`, `opls_cuda()`
- `kernel_pls_r()`, `kernel_pls_cpp()`, `kernel_pls_cuda()`

Prediction and utilities:

- `predict.fastPLS()`
- `transformy()`
- `ViP()`
- `fastcor()`
- `has_cuda()`
- `svd_methods()`
- `svd_run()`
- `svd_benchmark()`

Cross-validation:

- `optim.pls.cv()`
- `pls.double.cv()`
- `plssvd_cv_cpp()`, `simpls_cv_cpp()`
- `plssvd_cv_cuda()`, `simpls_cv_cuda()`

## Reproducible Benchmark Pipeline

The standard real-dataset benchmark is:

```sh
scripts/remote_run_dataset_memory_compare.sh
```

It writes one raw row per run and regenerates 4x4 plots with:

- columns: `plssvd`, `simpls`, `opls`, `kernelpls`
- rows: total time, predictive metric, peak host RSS, peak GPU memory
- color: SVD/backend (`irlba`, `rsvd`, `flash_svd`, `pls_pkg`)
- line type: implementation (`R`, `cpp`, `cuda`, `pls_pkg`)

The standard simulated variable-sweep benchmark is:

```sh
benchmark/workflow_synthetic_variable_sweeps.sh
```

or directly:

```sh
Rscript benchmark/benchmark_synthetic_variable_sweeps.R
Rscript benchmark/plot_synthetic_variable_sweeps.R <results_dir>
```

Important environment controls:

- `FASTPLS_RUN_TIMEOUT_SEC`: per-run timeout for real datasets.
- `FASTPLS_COMPARE_REPS`: number of replicates for real datasets.
- `FASTPLS_STORE_B`: `auto`, `always`, or `never`.
- `FASTPLS_STORE_B_MAX_MB`: automatic coefficient-cube storage threshold.
- `FASTPLS_SYNTH_VAR_TIMEOUT_SEC`: per-run timeout for simulated sweeps.
- `FASTPLS_SYNTH_VAR_MAX_HOST_RSS_MB`: RAM cap for simulated sweeps.

## References

- de Jong, S. (1993). SIMPLS. *Chemometrics and Intelligent Laboratory Systems*.
- Baglama, J. and Reichel, L. (2005). IRLBA. *SIAM Journal on Scientific Computing*.
- Halko, N., Martinsson, P.-G. and Tropp, J. A. (2011). Randomized algorithms
  for matrix decompositions. *SIAM Review*.
- Musco, C. and Musco, C. (2015). Randomized block Krylov methods for stronger
  and faster approximate singular value decomposition.
