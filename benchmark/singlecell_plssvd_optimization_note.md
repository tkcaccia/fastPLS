# plssvd singlecell optimization

## What changed

- Kept the public `plssvd` API unchanged.
- Added an internal `FASTPLS_PLSSVD_OPTIMIZED` switch so the same codebase can benchmark the legacy and optimized implementations side by side.
- Replaced the repeated legacy coefficient construction
  - `Y %*% V`
  - `t(T) %*% (Y %*% V)`
  - `inv(t(T) %*% T)`
  - `X %*% B`
  with exact algebra based on the PLSSVD SVD:
  - `S = X'Y = U D V'`
  - `T = XU`
  - `t(T) %*% (YV) = D`
- Precomputed `t(T) %*% T` once and reused leading principal blocks for each requested `ncomp`.
- Replaced explicit inversion with linear solves.
- Computed fitted values in latent space as `T_a %*% solve(G_a, D_a) %*% V_a'` instead of recomputing `X %*% B`.

## Bottlenecks found

- The old `plssvd` loop rebuilt `Y %*% V`, `t(T) %*% (YV)`, and `X %*% B` for every requested component count.
- It also used `inv(T'T)` explicitly.
- On the singlecell benchmark that repeated work dominated the post-SVD cost.

## Main optimization effect

The biggest win came from using the singular values directly in coefficient construction together with cached `t(T) %*% T` blocks. That removed most of the per-`ncomp` repeated matrix work without changing the method mathematically.

## Commands

```bash
R CMD INSTALL /Users/stefano/Documents/fastPLS-src
Rscript -e 'library(testthat); library(fastPLS); test_dir("/Users/stefano/Documents/fastPLS-src/tests/testthat")'
Rscript /Users/stefano/Documents/fastPLS-src/benchmark/benchmark_singlecell_plssvd.R
```
