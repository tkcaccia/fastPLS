# simpls_fast (icdefl) singlecell optimization

## What changed

- Kept the public `simpls_fast` API unchanged.
- Added an internal legacy-vs-optimized switch with `FASTPLS_FAST_OPTIMIZED` so the current codebase can benchmark the pre-optimization path against the optimized path on the same split.
- Made the incremental randomized refresh deterministic by seeding the Gaussian sketch from the per-fit `seed`.
- Added an exact cross-product shortcut for the optimized path:
  - cache `X'X`
  - cache the initial `X'Y`
  - reuse those cached matrices to compute `p = X' t` and `q = Y' t` from `r` without recomputing full `X' t` / `Y' t` products every iteration
- Enabled that cache only when the requested component count is large enough to amortize it (`FASTPLS_FAST_CROSSPROD_MIN_NCOMP`, default `20`).
- Left the riskier top-1 RSVD path behind an internal opt-in flag (`FASTPLS_FAST_RSVD_TOP1=1`) instead of making it the default optimized route.

## Bottlenecks found

- In `pls_model2_fast()`, the per-component work was dominated by repeated products involving the full `n x p` training matrix even though the singlecell benchmark has `n = 23822` and `p = 50`.
- The randomized incremental block refresh used an unseeded Gaussian draw, which made baseline-vs-optimized benchmarking noisier than necessary.
- Small `ncomp` fits are dominated by fixed overhead, so heavier caching only helps once enough components are requested.

## Benchmark summary

Dataset: `singlecell`

Metric: classification accuracy

Timing: model-building wall-clock time only

Output folder:

- `/Users/stefano/Documents/fastPLS_benchmark_results/singlecell_simpls_fast_icdefl_rsvd`

Compact results:

| method | ncomp | elapsed_time_seconds | accuracy | speedup_vs_baseline | accuracy_change_vs_baseline | passed_1pct |
|---|---:|---:|---:|---:|---:|---|
| baseline | 2 | 0.041 | 0.12053 | 1.000 | 0.000% | yes |
| optimized | 2 | 0.044 | 0.12053 | 0.932 | 0.000% | yes |
| baseline | 5 | 0.043 | 0.19291 | 1.000 | 0.000% | yes |
| optimized | 5 | 0.044 | 0.19291 | 0.977 | 0.000% | yes |
| baseline | 10 | 0.047 | 0.28987 | 1.000 | 0.000% | yes |
| optimized | 10 | 0.049 | 0.28987 | 0.959 | 0.000% | yes |
| baseline | 20 | 0.058 | 0.42791 | 1.000 | 0.000% | yes |
| optimized | 20 | 0.051 | 0.42791 | 1.137 | 0.000% | yes |
| baseline | 50 | 0.084 | 0.54928 | 1.000 | 0.000% | yes |
| optimized | 50 | 0.051 | 0.54928 | 1.647 | 0.000% | yes |

## Main effect

The most effective optimization was the exact cross-product cache in the higher-component regime. On the singlecell benchmark it produced:

- `1.14x` speedup at `ncomp = 20`
- `1.65x` speedup at `ncomp = 50`

with no observed accuracy loss relative to the baseline median on this benchmark run.

## Commands used

Install locally:

```bash
R CMD INSTALL /Users/stefano/Documents/fastPLS-src
```

Run tests:

```bash
Rscript -e 'library(testthat); library(fastPLS); test_dir("/Users/stefano/Documents/fastPLS-src/tests/testthat")'
```

Run the singlecell benchmark:

```bash
Rscript /Users/stefano/Documents/fastPLS-src/benchmark/benchmark_singlecell_simpls_fast_icdefl_rsvd.R
```

## Caveats

- On very small `ncomp`, the optimized path is not faster because the cache setup cost dominates. That is why the cache is enabled only from `ncomp >= 20` by default.
- The top-1 randomized refresh experiment is still available internally but is not the default optimized path because it was less stable than the exact cross-product shortcut.
