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

An experimental paper-inspired randomized block-Krylov range finder is available
for comparison with the standard rSVD path:

```sh
FASTPLS_RSVD_VARIANT=block_krylov
```

It keeps the intermediate Krylov blocks instead of using only the final
power-iteration sample. This can slightly improve the sampled subspace, but it
also increases the sketch width and therefore can be slower or use more temporary
workspace. It is intentionally not the default.

## CUDA Paths

GPU-native fitting wrappers:

- `plssvd_gpu()`
- `simpls_gpu()`
- `opls_cuda()`
- `kernel_pls_cuda()`

FlashSVD-style prediction wrappers:

- `plssvd_flash_gpu()`
- `simpls_flash_gpu()`
- `opls_flash_gpu()`
- `kernel_pls_flash_gpu()`

The flash wrappers keep fitting identical to the GPU-native wrappers and replace
prediction with low-rank CUDA products. This is expected to improve prediction
time most visibly when `q`, `ncomp`, or the number of requested component slices
is large. It is not expected to reduce fit peak memory unless fit workspaces are
changed separately.

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

## Build Hygiene

Do not commit generated benchmark outputs, R library folders, compiled objects,
or remote run logs. Clean source trees should not contain:

- `benchmark_results*`
- `Library/`
- `Outputs/`
- `*.o`, `*.so`, `*.dll`, `*.dylib`
- `*.Rcheck`
- `.Rproj.user/`
