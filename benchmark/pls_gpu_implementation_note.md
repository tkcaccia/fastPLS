# `simpls_gpu()` and PLS algorithm implementation note

## Scope

This note summarizes the current implementation status of the main `fastPLS`
algorithms and, in particular, the new experimental GPU-native entry point
`simpls_gpu()`.

The goal is to make explicit:

- which algorithm each public entry point computes,
- which parts are exact vs approximate,
- which parts run on CPU vs GPU,
- what changed when the old hybrid CUDA route was removed from `pls()`,
- and how the current benchmark behavior looks on the main reference datasets.

## High-level API split

The package now exposes two distinct execution families:

### 1. Standard high-level PLS API

Implemented through `pls()` in:

- `/Users/stefano/Documents/fastPLS-src/R/main.R`

Available methods:

- `method = "plssvd"`
- `method = "simpls"`
- `method = "simpls_fast"`

Available public SVD backends in `pls()`:

- `irlba`
- `cpu_rsvd`

The former hybrid CUDA route through:

- `pls(..., svd.method = "cuda_rsvd")`

has been removed from the high-level `pls()`, `optim.pls.cv()`, and
`pls.double.cv()` APIs.

### 2. Experimental GPU-native API

Implemented through `simpls_gpu()` in:

- `/Users/stefano/Documents/fastPLS-src/R/main.R`

This path is separate by design, so the standard `pls()` behavior remains
stable while GPU work can evolve independently.

## Algorithm descriptions

## `plssvd`

Implementation entry points:

- R wrapper:
  - `/Users/stefano/Documents/fastPLS-src/R/main.R`
- C++ core:
  - `/Users/stefano/Documents/fastPLS-src/src/fastPLS.cpp`

Core idea:

1. preprocess `X` by centering or autoscaling;
2. center `Y`;
3. form cross-covariance:
   - `S = X'Y`
4. compute a truncated SVD of `S`;
5. use the left/right singular subspaces to construct:
   - `R`
   - `Q`
   - `T = XR`
6. reconstruct component-wise regression coefficients.

Properties:

- one-shot low-rank decomposition of `X'Y`
- not an iterative SIMPLS deflation algorithm
- effective component count is bounded by the rank of `X'Y`

Approximation behavior:

- `cpu_rsvd`: approximate randomized SVD
- `irlba`: legacy IRLB/exact fallback path depending on code path and matrix size

## `simpls`

Implementation entry points:

- R wrapper:
  - `/Users/stefano/Documents/fastPLS-src/R/main.R`
- C++ core:
  - `/Users/stefano/Documents/fastPLS-src/src/fastPLS.cpp`

Core idea:

1. preprocess `X`, center `Y`;
2. initialize cross-covariance:
   - `S = X'Y`
3. for each component:
   - extract the current leading left singular direction of `S`
   - build latent quantities `r`, `t`, `p`, `q`
   - orthogonalize the deflation vector `v`
   - deflate:
     - `S <- S - v (v' S)`
4. accumulate coefficients from the sequence of extracted directions.

Properties:

- iterative SIMPLS-style deflation
- one leading direction is extracted per component
- coefficient tensor is built cumulatively over components

## `simpls_fast`

Implementation entry points:

- R wrapper:
  - `/Users/stefano/Documents/fastPLS-src/R/main.R`
- C++ core:
  - `/Users/stefano/Documents/fastPLS-src/src/fastPLS.cpp`

Core idea:

`simpls_fast` preserves SIMPLS-like outer deflation semantics but accelerates
the leading-direction extraction step.

The current implementation uses:

1. the same outer deflated cross-covariance object:
   - `S`
2. a refreshed block subspace instead of recomputing a fresh full truncated SVD
   every component;
3. incremental / block-refresh logic to update a left subspace `Ublock`;
4. a reduced projected problem:
   - `Bsmall = Ublock' S`
5. extraction of leading left directions from the reduced problem;
6. standard SIMPLS-like outer updates:
   - component stats,
   - coefficient accumulation,
   - rank-1 deflation.

Important design point:

- `simpls_fast` is not a different estimator in spirit
- it is an implementation optimization of the SIMPLS-style fit loop
- the user-facing fixed profile corresponds to the former incremental-deflation
  optimized variant

### CPU-side improvements currently included

Recent `simpls_fast` engineering in the repo includes:

- tighter heuristics for the `X'X` cache path so wide problems do not pay for a
  dense `p x p` route when it is not beneficial;
- improved leading-left-vector handling;
- reduced-stage square-matrix finalization:
  - use of `Bsmall Bsmall'` when only left directions are needed;
- faster prediction path in the general `predict.fastPLS()` machinery;
- conservative internal tuning for randomized SVD oversampling / power settings.

## `simpls_gpu()`

Implementation entry points:

- R wrapper:
  - `/Users/stefano/Documents/fastPLS-src/R/main.R`
- exported C++ entry point:
  - `/Users/stefano/Documents/fastPLS-src/src/fastPLS.cpp`
- CUDA backend:
  - `/Users/stefano/Documents/fastPLS-src/src/svd_cuda_rsvd.cpp`
  - `/Users/stefano/Documents/fastPLS-src/src/svd_cuda_rsvd.h`

`simpls_gpu()` is an experimental GPU-native implementation of the `simpls_fast`
fit loop.

It is separate from `pls()` because:

- the standard CPU/hybrid PLS API should remain stable,
- the GPU path still evolves rapidly,
- benchmarking and validation are easier when the GPU engine is explicit.

### Current `simpls_gpu()` flow

1. preprocess `X` and encode `Y` exactly like the standard high-level API;
2. upload training `X`, `Y`, and the initial cross-covariance to the GPU;
3. keep the evolving deflated operator resident on device;
4. refresh the left block on GPU;
5. compute reduced projected quantities on GPU;
6. update per-component quantities on GPU when possible;
7. apply rank-1 deflation on GPU;
8. copy back only the outputs needed to assemble the standard `fastPLS` object.

### GPU-resident pieces currently implemented

The current GPU engine includes:

- resident device storage for:
  - training `X`
  - training `Y`
  - evolving deflated cross-covariance
- device-side Gaussian random initialization with cuRAND
- GPU block power iterations
- GPU-native QR / orthonormalization with cuSOLVER
- GPU-side reduced square-matrix eigendecomposition
- GPU-side projected row extraction
- GPU-side rank-1 deflation
- GPU-side rank-1 fitted-response accumulation

### What `simpls_gpu()` still is not

It is a real separate GPU engine, but it is still experimental.

It is not yet a final “all logic permanently on device with zero fallback”
implementation. The current code is best described as:

- GPU-resident hot loop,
- GPU-native refresh / QR / reduced solve,
- standard `fastPLS` output reconstruction on the host side.

## CUDA backend details

Main files:

- `/Users/stefano/Documents/fastPLS-src/src/svd_cuda_rsvd.cpp`
- `/Users/stefano/Documents/fastPLS-src/src/svd_cuda_rsvd.h`

The CUDA backend now contains:

- a persistent workspace with reusable allocations,
- resident-matrix helpers for the evolving deflated operator,
- refresh helpers for the left block,
- GPU projection helpers,
- GPU rank-1 deflation helpers,
- GPU QR based on cuSOLVER,
- GPU small symmetric eigensolve for the reduced projected matrix,
- device-side random initialization based on cuRAND.

This is a substantial shift from the earlier hybrid approach, where the GPU
mainly accelerated the randomized sampling multiplies while important pieces
still returned to CPU immediately.

## Removal of the old hybrid CUDA route

The removed high-level route was:

- `pls(..., svd.method = "cuda_rsvd")`

This was a hybrid path because:

- it kept the standard high-level `pls()` logic,
- but delegated only selected inner linear algebra pieces to CUDA,
- without being a truly separate GPU model engine.

The package now uses a cleaner split:

- `pls()` for standard CPU-side algorithms and CPU-side randomized SVD,
- `simpls_gpu()` for the experimental GPU-native path.

This reduces API ambiguity and makes benchmark interpretation much clearer.

## Benchmark script included in the repo

The dedicated benchmark script for the GPU-engine comparison is:

- `/Users/stefano/Documents/fastPLS-src/benchmark/benchmark_simpls_fast_gpu_vs_hybrid.R`

This benchmark compares:

- `hybrid_cpu`
  - standard `simpls_fast` through `pls(..., svd.method = "cpu_rsvd")`
- `full_gpu`
  - experimental `simpls_gpu()`

Datasets currently used in that script:

- `metref`
- `singlecell`
- `cifar100`
- `gtex`
- `tcga_pan_cancer`
- `ccle`

The script uses:

- `ncomp = 50`
- three replicates
- classification accuracy as the prediction metric

## Current benchmark readout

The latest reported remote benchmark medians for `ncomp = 50` were:

### MetRef

- hybrid CPU: `0.020s`, accuracy `0.91`
- full GPU: `0.014s`, accuracy `0.92`

### singlecell

- hybrid CPU: `0.125s`
- full GPU: `0.051s`
- accuracy:
  - hybrid CPU: `0.5462`
  - full GPU: `0.5449`

### gtex_v8

- hybrid CPU: `0.240s`
- full GPU: `0.044s`
- accuracy:
  - hybrid CPU: `0.9423`
  - full GPU: `0.9460`

### tcga_pan_cancer

- hybrid CPU: `0.210s`
- full GPU: `0.043s`
- accuracy:
  - hybrid CPU: `0.9053`
  - full GPU: `0.9073`

### ccle

- hybrid CPU: `0.035s`
- full GPU: `0.020s`
- accuracy:
  - hybrid CPU: `0.6620`
  - full GPU: `0.6761`

### cifar100

- hybrid CPU: `5.366s`
- full GPU: `0.884s`
- accuracy:
  - hybrid CPU: `0.7498`
  - full GPU: `0.7485`

## Interpretation of the current benchmark

The present GPU engine is already much faster on the benchmarked datasets.

The main caveat is not speed anymore; it is validation confidence:

- the GPU path is still experimental,
- predictive performance stays in the same general range,
- but it is not guaranteed to be numerically identical to the CPU/hybrid path,
- and dataset-specific tradeoffs, especially on `cifar100`, still need to be
  monitored carefully.

## Tests included

Relevant tests currently covering this work include:

- `/Users/stefano/Documents/fastPLS-src/tests/testthat/test-pls-rsvd.R`
- `/Users/stefano/Documents/fastPLS-src/tests/testthat/test-simpls-fast-optimized.R`

The current test intent is:

- preserve standard `pls()` semantics on CPU backends,
- ensure the removed hybrid CUDA route errors clearly in high-level APIs,
- ensure `simpls_gpu()` returns a valid `fastPLS` object structure,
- ensure the GPU fit stays reasonably close to the supported CPU reference on a
  controlled problem when CUDA is available.

## Main caveats

1. `simpls_gpu()` is experimental.
2. The high-level hybrid CUDA path in `pls()` is intentionally gone.
3. Low-level CUDA/SVD machinery still exists internally because `simpls_gpu()`
   uses it.
4. `R CMD check --as-cran` should always be rerun after further GPU-interface
   changes, because the API split affects documentation and exported surface.

## Recommended usage today

Use:

- `pls()` for the stable package API and CPU-side work;
- `simpls_gpu()` only when explicitly benchmarking or experimenting with the
  GPU-native fit path.

That split reflects the current maturity of the two execution families.
