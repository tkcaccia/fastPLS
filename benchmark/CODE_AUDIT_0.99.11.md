# fastPLS 0.99.11 native ABI audit

Audit date: 2026-08-08

## Scope

This audit followed the public candidate-kNN removal completed in version
0.99.10. It covered native CPU and CUDA source, generated Rcpp wrappers and
registration entries, compiled cross-validation routing, compact top-k argmax
prediction, active benchmark generators, and installed-library symbols.

## Changes

- Removed candidate-kNN fitting and prediction helpers from C++ and CUDA.
- Removed class-bias offsets, calibration helpers, kernels, wrappers, and
  generated registrations.
- Removed the unreachable candidate-kNN branch from compiled cross-validation;
  its classifier identifier now accepts only argmax and latent-space LDA.
- Retained top-k argmax prediction through bias-free compact CPU and CUDA
  routines, preserving top-1 and optional top-5 output.
- Removed retired classifier variants and tuning controls from active benchmark
  generators without changing archived result files.

## Verification

- Source search contains no candidate-kNN or class-bias identifiers in Rcpp or
  native CPU/CUDA sources.
- The installed CPU/Metal shared library exports no candidate-kNN, class-bias,
  or cKNN symbol.
- Package installation succeeds with the CPU and Apple Metal backends.
- An isolated CUDA build on an NVIDIA GeForce RTX 5060 Ti compiles, links,
  installs, and passes native SIMPLS top-k prediction.
- Public classifier routing remains restricted to `argmax` and `lda`.
- The complete local testthat suite passes; unavailable platform-specific paths
  are skipped and documented precision-risk routes emit their expected warnings.
- `R CMD check --as-cran` completes with zero errors and zero warnings. Its sole
  NOTE is the expected development-version jump from the former CRAN 0.2
  release.
- Source archive: `fastPLS_0.99.11.tar.gz`.
- SHA-256: `7f287a0ba5c79b07b9077c04fd78dc0e99b28b394d7a115014876cead96b8588`.

Quantitative manuscript benchmarks remain tied to their archived source and
are not relabelled as version 0.99.11 reruns.
