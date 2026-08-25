# Controlled SIMPLS scaling benchmark

This benchmark varies one dimension at a time while holding the remaining
synthetic-data parameters fixed. It measures the shape-dependent execution
claims made for fastPLS SIMPLS rather than treating heterogeneous real datasets
as a scaling experiment.

The publication profile sweeps sample count, predictor dimension, response
dimension, retained components, requested component prefixes, effective
cross-covariance rank, class count, and explicit cross-covariance size. Every
scenario is compared with a deterministic float64 CPU-IRLBA fit generated from
the same synthetic matrices. Candidate runs use rSVD with oversampling 20, two
power iterations, and fixed seeds. The explicit/implicit crossover panel forces
both routes; all other panels test automatic routing.

The rank sweep uses a noise-free low-rank response so that the rank of the
population cross-covariance is controlled. Other regression sweeps add fixed
Gaussian response noise, and the class-count sweep uses labels generated from
a fixed-dimensional latent score model.

Each run uses a fresh R process. Fit and prediction time are recorded
separately. An external 20-ms sampler records process RSS and, on CUDA systems,
per-process and total GPU memory. Incremental host RSS is the sampled peak minus
RSS immediately before fitting. Numerical agreement is evaluated only after
the memory-sampling window closes, so loading the deterministic reference does
not contaminate candidate memory measurements.

Run a local smoke test:

```sh
bash scripts/run_controlled_scaling.sh \
  benchmark_results/controlled_scaling_smoke smoke cpu,metal 1
```

Run the publication CPU/CUDA grid on a CUDA host:

```sh
FASTPLS_SCALING_TIMEOUT_SEC=600 \
bash scripts/run_controlled_scaling.sh \
  benchmark_results/controlled_scaling_cuda publication cpu,cuda 3
```

Primary outputs are `controlled_scaling_raw.csv`,
`controlled_scaling_summary.csv`, `explicit_implicit_crossover.csv`,
`failures_and_numerical_discordance.csv`, and four PDF figures. Failed and
timed-out routes remain in the raw and failure tables.
