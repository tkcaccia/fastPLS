# Pipeline 1: real and simulated benchmark

Pipeline 1 is the main benchmark used for the real and simulated datasets.
It evaluates PLSSVD, SIMPLS, OPLS, and kernel-PLS over dataset-specific
component grids for real datasets, and over n/p/q sweeps for simulated
datasets.

## Run

```sh
bash benchmark/run_pipeline1_real_and_simulated.sh
```

## Real datasets

The real-dataset part records total fitting plus prediction time, predictive
metric, peak host RAM, peak GPU memory where available, requested/effective
number of components, and execution status.

Classification datasets are evaluated by accuracy.  Univariate regression is
evaluated by Q2, and multivariate regression by RMSD.

## Simulated datasets

The simulated part uses the same reporting format, but the x-axis represents
the swept simulation variable rather than the number of components.  The
default sweeps are sample size, predictor dimension, and response dimension
for both classification and regression.

## Main outputs

- `dataset_memory_compare_raw.csv`
- `dataset_memory_compare_summary.csv`
- `pipeline1_best_by_dataset_method.csv`
- one 4-row plot per real dataset
- `synthetic_variable_sweeps_raw.csv`
- `synthetic_variable_sweeps_summary.csv`
- simulated n/p/q sweep plots
