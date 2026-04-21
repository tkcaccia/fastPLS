# Synthetic Smoke Benchmark Note

This synthetic smoke benchmark is the first paper-facing synthetic benchmark for `fastPLS`. It is intentionally simple: every run requests **50 components**, and each task changes **one axis at a time** while the others stay fixed.

## How the data are built

The generators use one shared moderate latent-signal design.

For regression:

- `X = T diag(sx) P^T + sigma_x E_x`
- `Y = T diag(sy) C^T + sigma_y E_y`

where:

- `T` is the latent score matrix
- `P` and `C` are orthonormal loading matrices
- `sx` and `sy` decay smoothly from strong to weaker latent directions
- `E_x` and `E_y` are Gaussian noise matrices
- `X` and `Y` are centered using train-set means before fitting

Noise is controlled through target signal-to-noise ratios. The benchmark now uses five ordered noise points:

- `very_low_noise` with target SNR `40`
- `low_noise` with target SNR `20`
- `medium_noise` with target SNR `10`
- `high_noise` with target SNR `5`
- `very_high_noise` with target SNR `1`

The raw output stores both:

- the nominal noise label
- the realized SNR for `X` and `Y`

## What each synthetic task means

- `sim_reg_n_p50`
  - Regression sample-scaling test with `p = 50`, `q = 50`
  - This asks how training time and predictive quality change when the number of training samples grows, while the predictor width stays modest.

- `sim_reg_n_p500`
  - Regression sample-scaling test with `p = 500`, `q = 50`
  - This shows whether the same sample-growth pattern changes once the predictor matrix is much wider.

- `sim_reg_n_p1000_q1000_ncomp500`
  - Large regression sample-scaling test with `p = 1000`, `q = 1000`, and requested `ncomp = 500`
  - This is the first intentionally high-capacity example in the smoke benchmark.
  - It is included to show behavior when both `X` and `Y` are wide and the requested component count is much larger than `50`.

- `sim_reg_p_sweep`
  - Regression predictor-width test with fixed `n_train = 1000`, `q = 50`
  - This isolates the cost of making `X` wider.

- `sim_reg_q_sweep`
  - Regression response-width test with fixed `n_train = 1000`, `p = 500`
  - This isolates the cost of making `Y` wider, which is especially relevant for multivariate-response problems.

- `sim_reg_noise_sweep`
  - Regression noise-difficulty test with fixed `n_train = 1000`, `p = 500`, `q = 50`
  - This asks how runtime and predictive quality move when the signal becomes progressively noisier, without changing matrix size.

## Capacity-limited regimes

The benchmark always requests `50` components, but some settings cannot actually support `50`.

Capacity-limited cases occur when:

- `p < 50`
- `q < 50`
- or the train-set rank is smaller than `50`

The large `sim_reg_n_p1000_q1000_ncomp500` example follows the same rule, but with a requested `500` components instead of `50`.

In those cases the benchmark does **not** fail. Instead it records:

- `requested_ncomp = 50`
- `effective_ncomp = min(50, n_train - 1, p, q_or_K)`
- `capacity_limited = TRUE`

That is why the benchmark writes both:

- full results including all scenarios
- `fair50` results restricted to non-capacity-limited settings

## Why this is useful for the manuscript

This smoke benchmark is meant to answer the most interpretable scaling questions first:

- How do methods scale with more samples?
- How do they scale with wider `X`?
- How do they scale with wider `Y`?
- How do they behave as noise increases?

Only after this Phase 1 benchmark is stable should Phase 2 add signal-difficulty profiles such as:

- `signal_profile = easy`
- `signal_profile = medium`
- `signal_profile = hard`
