# fastPLS Developer Notes

This repository now treats the dataset-memory comparison pipeline as the
standard benchmark path. Older one-off benchmark scripts and obsolete notes have
been removed to keep the repository focused.

## Current Implementation

The package exposes four model families:

- `plssvd`
- `simpls`
- `opls`
- `kernelpls`

`simpls` is the optimized fastPLS SIMPLS implementation used in current
documentation, benchmarks, and plots. Old local script compatibility should stay
unexported and out of benchmark labels.

## SVD and xprod Policy

Supported SVD choices through `pls()` and `pls_r()` are:

- `irlba`
- `cpu_rsvd`
- `exact`

The former hybrid `svd.method = "cuda_rsvd"` path is intentionally removed.
CUDA fitting is exposed through the GPU-native wrappers.

The default matrix-free `xprod` policy is implemented in `R/main.R`:

- `cpu_rsvd`: use `xprod` when `X^T Y` would exceed 32 MB, or when
  `q >= 100` and `max(ncomp) <= 10`.
- `irlba`: use `xprod` only for much larger response spaces, currently when
  `X^T Y` would exceed 32 MB and `min(p, q) >= 1000`.

The C++ implementation rejects removed FP32/mixed-precision `xprod_precision`
values. The remaining implicit paths are double precision.

## Gaussian Response Compression

The user-facing option is `gaussian_y = TRUE`, with
`gaussian_y_dim = NULL` resolving to `min(ncol(Xtrain), 100)`. This option must
remain opt-in. Do not make it part of the default benchmark unless the benchmark
is explicitly testing response compression.

Implementation location:

- R orchestration and decoding helpers live in `R/main.R`.
- CUDA matrix multiplication helpers are exported internally from
  `src/svd_cuda_rsvd.cpp` through `src/fastPLS.cpp`.
- Tests for regression, classification, R, C++, OPLS, and kernelPLS coverage are
  in `tests/testthat/test-gaussian-y.R`.

Regression path:

- center `Y`
- draw a Gaussian projection with `gaussian_y_seed`
- fit the PLS model to `Y_centered %*% G`
- store a small ridge decoder from compressed predictions back to the original
  response columns
- decode `Yfit` and `Ypred` before reporting metrics

Classification path:

- avoid dense one-hot `transformy()` when `gaussian_y = TRUE`
- generate reproducible Gaussian class codes
- fit to the per-sample class code matrix
- decode predictions by nearest-code scores and return the original factor
  levels

CUDA path:

- regression projection and decoder RHS use the CUDA matrix-multiply helper when
  CUDA is available
- classification codebook construction stays on the host because the codebook is
  small and the expensive PLS fit remains GPU-native

## CUDA Paths

GPU-native fitting wrappers:

- `plssvd_gpu()`
- `simpls_gpu()`
- `opls_cuda()`
- `kernel_pls_cuda()`

FlashSVD-style prediction is integrated into the standard compact prediction
path instead of being treated as a separate benchmark algorithm. It keeps
fitting identical to the selected model family and applies predictions with
streamed low-rank products when compact factors are available. This is expected
to improve prediction time most visibly when `q`, `ncomp`, or the number of
requested component slices is large. It is not expected to reduce fit peak
memory unless fit workspaces are changed separately.

## Standard Benchmark Files

Real datasets:

- `scripts/remote_run_dataset_memory_compare.sh`
- `benchmark/benchmark_dataset_memory_compare.R`
- `benchmark/helpers_dataset_memory_compare.R`
- `benchmark/plot_dataset_memory_compare.R`

Simulated variable sweeps:

- `benchmark/benchmark_synthetic_variable_sweeps.R`
- `benchmark/plot_synthetic_variable_sweeps.R`
- `benchmark/workflow_synthetic_variable_sweeps.sh`

These scripts generate the current 4x4 plots and CSV summaries. New benchmark
work should extend these scripts instead of adding one-off benchmark files.
The standard simulated families are `reg_n`, `reg_p`, `reg_q`, `class_n`, and
`class_p`; noise sweeps are intentionally excluded from the default workflow.

## Build Hygiene

Do not commit generated benchmark outputs, R library folders, compiled objects,
or remote run logs. Clean source trees should not contain:

- `benchmark_results*`
- `Library/`
- `Outputs/`
- `*.o`, `*.so`, `*.dll`, `*.dylib`
- `*.Rcheck`
- `.Rproj.user/`
