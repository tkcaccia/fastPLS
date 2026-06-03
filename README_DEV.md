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

Supported CPU SVD choices through `pls()` are:

- `irlba`
- `cpu_rsvd`
- `exact`

The former hybrid `svd.method = "cuda_rsvd"` path is intentionally removed from
PLS fitting. CUDA fitting is exposed only through `pls(..., backend = "cuda")`.

The default matrix-free `xprod` policy is implemented in `R/main.R`:

- `cpu_rsvd`: use `xprod` when `X^T Y` would exceed 32 MB, or when
  `q >= 100` and `max(ncomp) <= 10`.
- `irlba`: use `xprod` only for much larger response spaces, currently when
  `X^T Y` would exceed 32 MB, `n >= 10000`, and `min(p, q) >= 1000`.
  This intentionally keeps medium-size synthetic response sweeps such as
  `n = 5000`, `p = 1000`, and `q > 100` on the explicit cross-product route.

The C++ implementation rejects removed FP32/mixed-precision `xprod_precision`
values. The remaining implicit paths are double precision.

## CUDA Paths

GPU-native fitting is selected through `pls(..., backend = "cuda")` with
`method = "plssvd"`, `"simpls"`, `"opls"`, or `"kernelpls"`.

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
