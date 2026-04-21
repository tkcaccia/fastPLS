# `plssvd_gpu()` implementation note

## Scope

This note documents the new GPU-native PLSSVD entry point:

- `/Users/stefano/Documents/fastPLS-src/R/main.R`
  - `plssvd_gpu()`

and the supporting CUDA backend:

- `/Users/stefano/Documents/fastPLS-src/src/fastPLS.cpp`
- `/Users/stefano/Documents/fastPLS-src/src/svd_cuda_rsvd.cpp`
- `/Users/stefano/Documents/fastPLS-src/src/svd_cuda_rsvd.h`

The goal is to make explicit:

- how `plssvd_gpu()` differs from CPU `plssvd`,
- which parts of the fit are now GPU-native,
- how it compares to the removed hybrid `cuda_rsvd` route,
- and why the public API now prefers explicit GPU functions over mixed CPU/GPU dispatch.

## Public API position

`plssvd_gpu()` is a separate GPU-native entry point.

It exists alongside:

- `pls(..., method = "plssvd", svd.method = "cpu_rsvd")`
- `pls(..., method = "plssvd", svd.method = "arpack")`
- `pls(..., method = "plssvd", svd.method = "irlba")`

The former mixed route through:

- `pls(..., svd.method = "cuda_rsvd")`

has been removed from the public CPU-side API because it was slower and harder
to reason about than an explicit GPU path.

So the current split is:

- CPU-side `pls()` for standard `plssvd`, `simpls`, and `simpls_fast`
- GPU-side `plssvd_gpu()` for GPU-native PLSSVD
- GPU-side `simpls_gpu()` for GPU-native `simpls_fast`

## Algorithmic goal

The GPU implementation is not a new estimator.

It computes the same PLSSVD model family:

1. preprocess `X`
2. center `Y`
3. form the cross-covariance
   - `S = X'Y`
4. compute a low-rank decomposition of `S`
5. derive latent quantities `R`, `Q`, `T`
6. build the coefficient tensor component by component

The improvement target is implementation speed, not model semantics.

## CPU `plssvd` reference path

The CPU-side path:

- preprocesses `X` and `Y` on host
- forms `S = X'Y`
- runs a truncated SVD backend on CPU
- builds `R`, `Q`, `T`, and `B`
- optionally computes fitted values and predictions

When `svd.method = "cpu_rsvd"`, only the SVD step is approximate; the rest of
the fit remains standard PLSSVD algebra on CPU.

## `plssvd_gpu()` design

`plssvd_gpu()` keeps the PLSSVD decomposition and downstream latent linear
algebra on GPU as much as possible.

### Current flow

1. preprocess `X` and `Y` exactly as in the high-level R wrapper
2. upload training matrices to the CUDA workspace
3. use resident device cross-covariance machinery to obtain the truncated
   PLSSVD factorization on GPU
4. compute:
   - `R`
   - `T = XR`
   - the reduced projected solve used to recover coefficient checkpoints
5. build coefficient slices on GPU at requested `ncomp` checkpoints
6. copy back only the final host outputs needed to assemble the standard
   `fastPLS` object

### GPU-resident linear algebra

The current implementation keeps these pieces on device in the PLSSVD path:

- cross-covariance factorization machinery
- reduced projected matrices
- the projected Gram system used at checkpoint recovery
- coefficient checkpoint construction
- optional fitted-value construction

### Reduced solve strategy

For each requested component checkpoint:

1. form the score-space Gram matrix on device
2. form the right-hand side corresponding to the projected PLSSVD coefficients
3. solve the symmetric positive-definite system on device
4. recover the coefficient block on device

This avoids the old pattern where only the SVD itself was accelerated but the
rest of the PLSSVD fit moved back to CPU.

## Difference from the former hybrid CUDA route

The removed hybrid route mixed:

- GPU-accelerated randomized SVD pieces
- host-side model assembly and finalization

That design was slower in practice for the package use case because:

- host/device transfers remained inside the fit path,
- finalization still paid CPU-side costs,
- and the user-visible API suggested a GPU option that was not truly GPU-native.

`plssvd_gpu()` replaces that with a clearer contract:

- explicit GPU entry point,
- explicit CUDA requirement,
- explicit GPU-native fit path.

## Returned object structure

`plssvd_gpu()` returns the same high-level `fastPLS` object style as the CPU
path, including:

- `B`
- `Q`
- `Ttrain`
- `R`
- `mX`
- `vX`
- `mY`
- optional `Yfit`
- optional test-side prediction outputs through `predict.fastPLS()`

This keeps downstream usage stable for:

- `predict()`
- plotting
- model summaries
- benchmark code

## Validation strategy

The repo includes:

- `/Users/stefano/Documents/fastPLS-src/tests/testthat/test-plssvd-gpu.R`

The test checks:

- CUDA requirement behavior when no GPU backend is available
- valid `fastPLS` return structure when CUDA is available
- dimensional agreement with CPU `plssvd`
- finite `B` and `R`
- classification accuracy remaining in the same general range

Additional CPU-side API checks also confirm that:

- `svd_methods()` no longer exposes `cuda_rsvd`
- `svd_run()` and `svd_benchmark()` no longer advertise the mixed CUDA route
- `pls(..., svd.method = "cuda_rsvd")` errors with guidance toward
  `simpls_gpu()` or `plssvd_gpu()`

## Benchmark summary

The dedicated comparison script is:

- `/Users/stefano/Documents/fastPLS-src/benchmark/benchmark_plssvd_gpu_vs_cpu.R`

It compares:

- CPU `plssvd` with `svd.method = "cpu_rsvd"`
- `plssvd_gpu()`

on:

- `metref`
- `singlecell`
- `gtex_v8`
- `tcga_pan_cancer`
- `ccle`
- `cifar100`

In the current comparison run at `ncomp = 50`, the GPU path was faster on every
dataset tested, with accuracy matched exactly on `singlecell` and remaining
very close on the others.

## Current caveats

`plssvd_gpu()` should still be treated as experimental.

Reasons:

- it depends on a CUDA-enabled build and runtime device availability
- the GPU path has been validated on the current benchmark set, but not yet on
  every package use case
- numerical equality to CPU randomized PLSSVD is not expected bit-for-bit

The practical acceptance criterion is:

- materially better fit speed
- accuracy in the same practical range
- cleaner public API than the former hybrid route

## Recommendation

Use:

- CPU `plssvd` via `pls(..., method = "plssvd", svd.method = "cpu_rsvd")`
  for the stable CPU path
- `plssvd_gpu()` when explicitly benchmarking or using the CUDA-native PLSSVD
  implementation

Do not use:

- `pls(..., svd.method = "cuda_rsvd")`

because that mixed public route has been intentionally retired.
