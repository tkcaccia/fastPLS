# fastPLS

`fastPLS` provides compiled C++ and CUDA implementations of partial least squares
models for high-dimensional regression and classification. The user-facing API
is intentionally small: algorithms and implementation backends are selected
through `pls()`, `pls.single.cv()`, `pls.double.cv()`,
`fastsvd()`, and `pca()` instead of through low-level implementation wrappers.
The current standard pipeline compares four model families:

- `plssvd`
- `simpls`
- `opls`
- `kernelpls`

The `simpls` implementation is the optimized fastPLS SIMPLS core. Older
low-level SIMPLS tuning arguments are not part of the public API; new analyses
should use `method = "simpls"`.

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
- `simpls`: optimized SIMPLS with compact latent prediction and automatic
  matrix-free `xprod` selection when it reduces cross-covariance work.
- `opls`: supervised orthogonal filtering followed by the selected PLS core.
- `kernelpls`: linear, RBF, or polynomial kernel construction followed by the
  selected PLS core.

For classification, factor responses are handled as PLS-DA responses. Large
response spaces use compact prediction where possible so the full coefficient
cube does not need to be stored.

For PLS-DA with LDA classification, the recommended high-accuracy/high-speed
configuration is `method = "plssvd", backend = "cuda", classifier = "lda"`.
This uses the optimized standard CUDA path for latent projection, LDA training,
and discriminant scoring. If CUDA is unavailable, use
`method = "plssvd", backend = "cpu", classifier = "lda"` as the compiled CPU
fallback. An experimental fused CUDA PLS+LDA path is available with
`FASTPLS_FUSED_CUDA_LDA=1`, but benchmark results currently keep it opt-in
rather than the default.

For PLS-DA latent-score prediction, `classifier = "cknn"` uses cKNN, the short
public name for the optional PLS-score candidate-kNN classifier.
The model first ranks classes by centroids in the supervised PLS score space and
then reranks every sample by same-class kNN among the top candidate classes. The
defaults are `k = 10`, `tau = 0.2`, and `alpha = 0.75`; `top_m` controls the
number of candidate classes. cKNN is a proof-of-concept downstream classifier
for sample-rich PLS score spaces, not the default PLS-DA decoder; compare it
with argmax or LDA for the dataset being analysed. `predict()` also supports `top` and
`top5 = TRUE` to return ranked class labels and scores for ImageNet-style top-5
evaluation.

For large classification problems, such as ImageNet-scale DINOv2 feature
matrices, `method = "plssvd", backend = "cuda"` automatically switches to a
label-aware PLSSVD route when the dense one-hot response would exceed the memory
threshold. This route streams class-wise cross-products from the label vector,
never materializes the dense `n x classes` response matrix, stores only compact
low-rank prediction factors, and can train the candidate-kNN classifier from
compact latent scores.
The default threshold is controlled by `FASTPLS_LABEL_AWARE_Y_THRESHOLD_MB`
and the candidate-kNN cache is stored in compact latent-score form.

For the ImageNet/DINOv2 experiments, the latent-score candidate-kNN case study
uses PLS scores to build CUDA prototype scores and reranks each sample by CUDA
candidate kNN within the PLS score space. The obsolete gated/calibrated
class-bias variants have been removed; tune only `k`, `tau`, `alpha`, and
`top_m`. The generic `pls()` API exposes the same decision rule through
`classifier = "cknn"` and chooses the backend-specific route. CPU uses compiled
C++, CUDA accelerates supported scoring operations, and Metal projects scores
on Metal before CPU neighbour scoring.

## Backends

CPU backends:

- `irlba`: bundled internal IRLBA wrapper.
- `rsvd`: randomized SVD with Gaussian sketching and optional power
  iterations.
Very small SVD inputs automatically fall back to a full dense decomposition
inside the compiled backends when the truncated route is not meaningful, but
`exact` is no longer exposed as a user-selectable PLS benchmark option.

CUDA backend:

- use `pls(..., backend = "cuda")` with `method = "plssvd"`, `"simpls"`,
  `"opls"`, or `"kernelpls"`.

On Linux and Windows, CUDA support is optional. If the CUDA Toolkit is not
available, the package builds CPU-only and CUDA requests give a clear runtime
error. This CPU-only fallback is also used if an old environment has
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

For automated CUDA build tests, set `FASTPLS_REQUIRE_CUDA = "1"` in addition to
`FASTPLS_USE_CUDA = "1"` if installation should fail when the CUDA Toolkit is
not found.

FlashSVD-style low-rank prediction is integrated into the standard prediction
path. When compact latent factors are available, `predict.fastPLS()` can apply
predictions through streamed low-rank products instead of materializing and
multiplying by the full coefficient matrix. This primarily reduces prediction
time and RAM pressure during prediction; fit memory is still governed by the
fitting backend.

Use `backend = "cuda"` for GPU-native PLS runs, or
`fastsvd(..., backend = "cuda", method = "rsvd")` /
`pca(..., backend = "cuda", method = "rsvd")` for stand-alone GPU SVD/PCA when CUDA is
available.

## Current API

Main model fitting:

- `pls()`

Prediction and utilities:

- `predict()`
- `ViP()`
- `fastcor()`
- `has_cuda()`
- `fastsvd()`
- `pca()`
- `plot()` for `fastPLS` and `fastPLSPCA` score plots with optional confidence
  or Hotelling's T2 ellipses

Cross-validation:

- `pls.single.cv()`
- `pls.double.cv()`

All lower-level C++, CUDA, OPLS, kernel PLS, SVD-dispatch, and KODAMA-oriented
helpers are internal implementation details. Benchmarks should use the same
public API as package users.

## Reproducible Benchmark Pipeline

The standard real-dataset benchmark is:

```sh
scripts/remote_run_dataset_memory_compare.sh
```

It writes one raw row per run and regenerates 4x4 plots with:

- columns: `plssvd`, `simpls`, `opls`, `kernelpls`
- rows: total time, predictive metric, peak host RSS, peak GPU memory
- color: SVD/backend (`irlba`, `rsvd_cpu`, `rsvd_cuda`, `pls_pkg`)
- line type: prediction rule (`argmax`, `LDA`)

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

By default, simulated sweeps vary sample size, predictor dimension, and response
dimension/classes. Noise sweeps are not part of the standard simulated
benchmark.

## References

- de Jong, S. (1993). SIMPLS. *Chemometrics and Intelligent Laboratory Systems*.
- Baglama, J. and Reichel, L. (2005). IRLBA. *SIAM Journal on Scientific Computing*.
- Halko, N., Martinsson, P.-G. and Tropp, J. A. (2011). Randomized algorithms
  for matrix decompositions. *SIAM Review*.
- Musco, C. and Musco, C. (2015). Randomized block Krylov methods for stronger
  and faster approximate singular value decomposition.
