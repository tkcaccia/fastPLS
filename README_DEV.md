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

`simpls` is the public method name for the optimized fastPLS SIMPLS-family
estimator used in current documentation, benchmarks, and plots. A route that
consumes a bounded candidate block from one deflated state is an approximate
SIMPLS-family estimator, not unqualified classical de Jong SIMPLS. Old local
script compatibility should stay unexported and out of benchmark labels.

## Solver and Cross-Product Policy

The public package uses native randomized SVD. Users can tune only the
documented randomized controls (`oversample`, `power`, and `seed`); removed
solver names and legacy cross-product arguments are not compatibility options.

The implementation chooses automatically between an explicit
predictor-response cross-product and matrix-free products such as
`X^T (Y Omega)`. The decision uses matrix dimensions, precision, backend,
requested rank, and estimated storage. It is an internal execution choice and
must not change the requested PLS family, preprocessing, component count, or
prediction head.

Do not reintroduce IRLBA or user-facing `svd.method`, `xprod`, or
`xprod_precision` arguments in this repository. Historical behavior remains
documented in `NEWS.md` only.

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

## External Benchmarks

Publication-scale analyses do not belong in this package repository. The
companion `tkcaccia/fastPLS-extra` repository contains the reproducible
benchmark workflows:

- `Phase1/` contains the CMPB analyses, acquisition scripts, formal Lean
  project, tables, and figure builders.
- `Phase2/` contains the separate software-interface and portability work for
  the future JSS article.

Keep only package examples, unit tests, and small deterministic smoke data in
this repository. Extend the appropriate companion phase instead of adding a
new benchmark or generated result directory here.

## Build Hygiene

Do not commit generated benchmark outputs, R library folders, compiled objects,
or remote run logs. Clean source trees should not contain:

- `benchmark_results*`
- `Library/`
- `Outputs/`
- `*.o`, `*.so`, `*.dll`, `*.dylib`
- `*.Rcheck`
- `.Rproj.user/`

## Test Suites

Routine `R CMD check` runs the compact deterministic suite in
`tests/testthat`. It covers all four PLS families, regression, argmax and LDA
classification, float32 prediction, single cross-validation, nested
cross-validation, and explicit failure of unavailable accelerator backends.

The exhaustive validation suite is retained in
`tests/development/testthat` and excluded from source archives. After installing
the current source, run it explicitly from the repository root with:

```r
source("tests/development/run.R")
```

Run the exhaustive suite in dedicated continuous-integration jobs rather than
as part of CRAN checks. Publication-scale benchmarks remain outside the package
repository.
