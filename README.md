# fastPLS

`fastPLS` provides compiled C++ and CUDA implementations of partial least squares
models for high-dimensional regression and classification. The user-facing API
is intentionally small: algorithms and implementation backends are selected
through `pls()`, `pls.single.cv()`, `pls.double.cv()`, and
`fastsvd()` instead of through low-level implementation wrappers.
The current benchmark suite compares four model families:

- `plssvd`
- `simpls`
- `opls`
- `kernelpls`

The `simpls` implementation is the optimized fastPLS SIMPLS core. Older
low-level SIMPLS tuning arguments are not part of the public API; new analyses
should use `method = "simpls"`.

The explicit package exports are `pls()`, `pls.single.cv()`, `pls.double.cv()`,
`evaluate()`, `plot.permutation()`, `ViP()`, `fastsvd()`, `fastcor()`,
`fastPLS_backend()`, `has_cuda()`, and `has_metal()`. Standard `predict()` and
`plot()` generics dispatch to registered fastPLS methods. There is no exported PCA API. The supported model families are `plssvd`,
`simpls`, `opls`, and `kernelpls`; classification uses `argmax` or latent-space
`lda`. The deprecated `lda_ridge` compatibility argument is ignored and warns
when supplied because LDA uses a fixed scale-normalized Cholesky fallback.

## Installation

Install the released package from Bioconductor:

```r
if (!requireNamespace("BiocManager", quietly = TRUE))
  install.packages("BiocManager")
BiocManager::install("fastPLS")
```

The GitHub repository contains the development source and the optional
CUDA/Metal build instructions below.

## Bundled data

The package includes two small, fixed example datasets that can be loaded with
`data()`: `colon` and `breast`. Larger real benchmark matrices such as CCLE,
GTEx, and TCGA subsets are intentionally kept outside the source package to keep
installation lightweight; the benchmark scripts load those matrices from the
local benchmark data directories instead. Source attribution and data-use notes
for bundled examples are provided in the dataset help pages and in
`inst/DATA_SOURCES.md`.

## Algorithms

- `plssvd`: computes the dominant subspace of the cross-covariance
  `S = X^T Y` and reuses it for the requested component path.
- `simpls`: accelerated sequential SIMPLS. The default rSVD route is an
  explicitly approximate high-speed profile: ordinary CPU and Metal routes
  generate a new oversampled sketch for each component, while CUDA can generate up to eight
  fresh candidates together for large dummy-coded classification paths to
  amortize GPU launches. Each candidate is accepted only after the sequential
  SIMPLS orthogonalization and deflation update. Compact
  prediction, cached deflation products, incremental coefficient updates, and
  automatic matrix-free `xprod` further reduce computation and storage.
  `svd.method = "irlba"` retains a fixed-control component-wise comparator.
- `opls`: supervised orthogonal filtering followed by the selected PLS core.
- `kernelpls`: linear, RBF, or polynomial kernel construction followed by the
  selected PLS core.

The CPU backend uses the BLAS/LAPACK library linked when the package is built.
Set `options(cores = 4L)` to request four CPU threads. Eligible matrix
operations can use multiple cores when linked to a multithreaded BLAS,
for example by installing with `FASTPLS_USE_OPENBLAS=1` and a valid
`OPENBLAS_ROOT`. SIMPLS deflation remains sequential, so multicore gains depend
on matrix shape; on the evaluated Apple M3 tasks, two and four OpenBLAS threads
did not improve runtime over one thread.

For classification, factor responses are handled as PLS-DA responses. Large
response spaces use compact prediction where possible so the full coefficient
cube does not need to be stored.

For PLS-DA with LDA classification, the recommended high-accuracy/high-speed
configuration is `method = "plssvd", backend = "cuda", classifier = "lda"`.
This uses the optimized standard CUDA path for latent projection, LDA training,
and discriminant scoring. On systems without CUDA, users can explicitly select
`method = "plssvd", backend = "cpu", classifier = "lda"` for compiled CPU
execution. An experimental fused CUDA PLS+LDA path is available with
`FASTPLS_FUSED_CUDA_LDA=1`, but benchmark results currently keep it opt-in
rather than the default.

For large classification problems, such as ImageNet-scale DINOv2 feature
matrices, `method = "plssvd", backend = "cuda"` automatically switches to a
  label-aware PLS-SVD route when the dense one-hot response would exceed the memory
threshold. This route streams class-wise cross-products from the label vector,
never materializes the dense `n x classes` response matrix, stores only compact
low-rank prediction factors. The default threshold is controlled by
`FASTPLS_LABEL_AWARE_Y_THRESHOLD_MB`.

## Backends

Set the fastPLS session default with `options(backend = "cuda")`, or use
`Sys.setenv(FASTPLS_BACKEND = "cuda")`. An explicit function argument always
takes precedence.

CPU backends:

- `irlba`: bundled internal IRLBA wrapper.
- `rsvd`: randomized SVD with Gaussian sketching and optional power
  iterations.

`rsvd` is a stochastic approximation, not a deterministic replacement for
IRLBA. It remains the primary starting solver. For CPU float64 fits, every
randomized decomposition is checked using normalized singular-triplet
residuals and an omitted-direction audit. The solver strengthens the sketch
automatically when needed. A weak spectral boundary must either agree with an
independent strengthened sketch or recover with IRLBA; consensus and recovery
are recorded in `diagnostics`, not hidden.
CUDA, Metal, and float32 routes record their exact controls and structural
diagnostics. The controlled validation uses matrix-shape-specific automatic
controls and reports numerical agreement separately from successful execution.
For confirmatory coefficient or subspace interpretation, use
`svd.method = "irlba"` on the CPU.
The validation suite places an rSVD fit
outside the numerical screen relative to a matched CPU IRLBA fit if prediction
or score relative error exceeds 0.01, the corresponding correlation is below
0.995, a latent-subspace angle exceeds 0.1 degrees, classification-label
agreement is below 0.995, or the predictive metric differs by more than 0.005.
PLS-SVD and standalone `fastsvd()` use 32 oversampling directions and five
power iterations by default. Accelerated SIMPLS, OPLS, and kernel-PLS use
32/5 for ordinary shapes. Numeric responses with at least 64 columns and a
response-to-sample ratio of at least 0.2 use 48/6. Classification with at least
32 classes and no more than 20 samples per class uses 64/7. When the explicit
predictor-response cross-covariance would exceed 512 MiB, the massive-matrix
profile records `oversample = 12` and `power = 2`; execution advances with one
new rank-one randomized direction per component.
CPU, Metal, and ordinary CUDA SIMPLS-family routes use seeded sketches of the
current deflated operator. For a massive cross-covariance, CPU, CUDA, and Metal
use a fresh rank-one randomized direction for every component; the CUDA state
remains device resident. Large dummy-coded
classification can instead refresh a small candidate block. Effective
controls are recorded in model diagnostics.
Very small SVD inputs automatically use a full dense decomposition inside the
selected compiled backend when the truncated route is not meaningful, but
`exact` is no longer exposed as a user-selectable PLS benchmark option.

CUDA backend:

- use `pls(..., backend = "cuda")` with `method = "plssvd"`, `"simpls"`,
  `"opls"`, or `"kernelpls"`.

On Linux and Windows, CUDA support is optional. If the CUDA Toolkit is not
available, the package builds CPU-only and CUDA requests give a clear runtime
error without running the requested operation on CPU. A CPU-only build is also
produced if an old environment has
`FASTPLS_USE_CUDA = "1"` but the toolkit is missing. To force a CPU-only build
on any machine:

```r
Sys.setenv(FASTPLS_USE_CUDA = "0")
remotes::install_github("tkcaccia/fastPLS", force = TRUE, upgrade = "never")
```

On Windows, if installation prints `package 'fastPLS' is in use and will not be
installed`, restart R before reinstalling. Windows keeps the loaded package DLL
locked, so `force = TRUE` cannot replace it while `library(fastPLS)` is active.
For a Windows machine without the CUDA Toolkit, use the CPU-only command above
and do not set `CUDA_ROOT`.

On an NVIDIA workstation, install the NVIDIA CUDA Toolkit and build with CUDA
enabled by setting `FASTPLS_USE_CUDA = "1"` and `CUDA_ROOT`.

Linux example:

```r
Sys.setenv(
  FASTPLS_USE_CUDA = "1",
  CUDA_ROOT = "/usr/local/cuda"
)
remotes::install_github("tkcaccia/fastPLS", force = TRUE, upgrade = "never")
```

Windows example:

```r
Sys.setenv(
  FASTPLS_USE_CUDA = "1",
  CUDA_ROOT = "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.6"
)
remotes::install_github("tkcaccia/fastPLS", force = TRUE, upgrade = "never")
```

After installation, `has_cuda()` reports whether the package was compiled with
CUDA support and can see a CUDA device at runtime.

Selecting or configuring `backend = "cuda"` without an available CUDA
build/device raises an error. fastPLS never silently changes a CUDA or Metal
request to CPU; `fastPLS_backend()` also rejects an unavailable configured
accelerator.

On macOS, the Metal backend is compiled automatically when the macOS SDK or
system Metal frameworks are available. The CUDA message "building without
CUDA" does not mean that Metal was disabled. A successful configuration prints
`Apple Metal backend enabled`. After restarting R, verify both compilation and
runtime device access with:

```r
library(fastPLS)
has_metal()
```

Likewise, selecting `backend = "metal"` when Metal support is unavailable raises
an error; choose `backend = "cpu"` explicitly when CPU execution is intended.

To require Metal explicitly during a GitHub installation, use:

```r
Sys.setenv(FASTPLS_USE_METAL = "1")
remotes::install_github("tkcaccia/fastPLS", force = TRUE, upgrade = "never")
```

For automated CUDA build tests, set `FASTPLS_REQUIRE_CUDA = "1"` in addition to
`FASTPLS_USE_CUDA = "1"` if installation should fail when the CUDA Toolkit is
not found.

FlashSVD-style low-rank prediction is integrated into the standard prediction
path. When compact latent factors are available, `predict.fastPLS()` can apply
predictions through streamed low-rank products instead of materializing and
multiplying by the full coefficient matrix. This primarily reduces prediction
time and RAM pressure during prediction; fit memory is still governed by the
fitting backend.

Use `backend = "cuda"` for supported CUDA PLS runs, or
`fastsvd(..., backend = "cuda", method = "rsvd")` for stand-alone GPU rSVD
when CUDA is available.

## Current API

Main model fitting:

- `pls()`

Prediction and utilities:

- `predict()`
- `ViP()`
- `fastcor()`
- `has_cuda()`
- `has_metal()`
- `fastsvd()`
- `plot()` for `fastPLS` score plots with optional confidence
  or Hotelling's T2 ellipses

Cross-validation:

- `pls.single.cv()`
- `pls.double.cv()`

All lower-level C++, CUDA, OPLS, kernel PLS, SVD-dispatch, and KODAMA-oriented
helpers are internal implementation details. Benchmarks should use the same
public API as package users.

## Reproducible Benchmarks

Benchmark runners, numerical-validation studies, and the code that generates
manuscript tables and figures are maintained in the separate
`tkcaccia/fastPLS-extra` repository. This repository contains only the
installable package, its tests, and user documentation. Generated benchmark
results are not tracked here; a frozen evidence archive will be published
separately for the manuscript release.

## References

- de Jong, S. (1993). SIMPLS. *Chemometrics and Intelligent Laboratory Systems*.
- Baglama, J. and Reichel, L. (2005). IRLBA. *SIAM Journal on Scientific Computing*.
- Halko, N., Martinsson, P.-G. and Tropp, J. A. (2011). Randomized algorithms
  for matrix decompositions. *SIAM Review*.
- Musco, C. and Musco, C. (2015). Randomized block Krylov methods for stronger
  and faster approximate singular value decomposition.
