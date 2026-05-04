# fastPLS

`fastPLS` provides compiled C++ and CUDA implementations of partial least squares
models for high-dimensional regression and classification. The user-facing API
is intentionally small: algorithms and implementation backends are selected
through `pls()`, `pls.single.cv()`, `pls.double.cv()`, `optim.pls.cv()`,
`fastsvd()`, and `pca()` instead of through low-level implementation wrappers.
The current standard pipeline compares four model families:

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
- `opls`: supervised orthogonal filtering followed by the selected PLS core.
- `kernelpls`: linear, RBF, or polynomial kernel construction followed by the
  selected PLS core.

For classification, factor responses are handled as PLS-DA responses. Large
response spaces use compact prediction where possible so the full coefficient
cube does not need to be stored.

## Gaussian Response Compression

All four model families can optionally fit a compressed Gaussian representation
of the response by setting `gaussian_y = TRUE` in `pls()`. This is available
through the compiled C++ and CUDA backends for `plssvd`, `simpls`, `opls`,
and `kernelpls`. The option is disabled by default, so existing analyses are
unchanged unless it is requested explicitly.

When `gaussian_y_dim = NULL`, fastPLS uses `min(ncol(Xtrain), 100)` compressed
response dimensions. A positive integer can be passed to `gaussian_y_dim` to
test a smaller or larger sketch. The projection is reproducible through
`gaussian_y_seed`, which defaults to the model `seed`.

For regression, the centered response matrix is multiplied by a Gaussian random
projection before fitting. The fitted model stores a small ridge decoder that
maps predictions from the compressed response space back to the original
response columns, so `predict()` and in-fit predictions still return values on
the original scale.

For classification, Gaussian compression avoids fitting to a dense one-hot
response when requested. Class labels are represented by reproducible Gaussian
class codes, the PLS model is fit to those codes, and predictions are decoded by
nearest-code scores back to the original factor levels.

CUDA wrappers use the CUDA matrix-multiply helper for the regression projection
and decoder construction when CUDA is available. The classification codebook is
small and is built on the host before the GPU-native PLS fit.

For PLS-DA with LDA classification, the recommended high-accuracy/high-speed
configuration is `method = "plssvd", backend = "cuda", classifier = "lda_cuda"`.
This uses the optimized standard CUDA path for latent projection, LDA training,
and discriminant scoring. If CUDA is unavailable, use
`method = "plssvd", backend = "cpp", classifier = "lda_cpp"` as the compiled CPU
fallback. An experimental fused CUDA PLS+LDA path is available with
`FASTPLS_FUSED_CUDA_LDA=1`, but benchmark results currently keep it opt-in
rather than the default.

Example:

```r
fit <- pls(
  Xtrain,
  Ytrain,
  Xtest,
  Ytest,
  method = "simpls",
  svd.method = "cpu_rsvd",
  ncomp = 50,
  gaussian_y = TRUE,
  gaussian_y_dim = 50
)
```

## Backends

CPU backends:

- `irlba`: bundled internal IRLBA wrapper.
- `cpu_rsvd`: randomized SVD with Gaussian sketching and optional power
  iterations.
Very small SVD inputs automatically fall back to a full dense decomposition
inside the compiled backends when the truncated route is not meaningful, but
`exact` is no longer exposed as a user-selectable PLS benchmark option.

CUDA backend:

- use `pls(..., backend = "cuda")` with `method = "plssvd"`, `"simpls"`,
  `"opls"`, or `"kernelpls"`.

FlashSVD-style low-rank prediction is integrated into the standard prediction
path. When compact latent factors are available, `predict.fastPLS()` can apply
predictions through streamed low-rank products instead of materializing and
multiplying by the full coefficient matrix. This primarily reduces prediction
time and RAM pressure during prediction; fit memory is still governed by the
fitting backend.

The removed hybrid `svd.method = "cuda_rsvd"` route through the CPU PLS fitter is
no longer supported. Use `backend = "cuda"` for GPU-native PLS runs, or
`fastsvd(..., method = "cuda_rsvd")` / `pca(..., svd.method = "cuda_rsvd")` for
stand-alone GPU SVD/PCA when CUDA is available.

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

- `optim.pls.cv()`
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
