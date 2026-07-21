# fastPLS benchmark pipelines

This folder contains the benchmark workflows used to evaluate `fastPLS`.
The four main pipelines can be launched from this directory through small
wrapper scripts.  The wrappers call the lower-level scripts in `scripts/`
and keep the benchmark entry points visible in one place.

## Pipeline 1: real and simulated datasets

Purpose: benchmark PLSSVD, SIMPLS, OPLS, and kernel-PLS on the real dataset
panel and on simulated n/p/q sweeps.

Run:

```sh
bash benchmark/run_pipeline1_real_and_simulated.sh
```

Core scripts:

- `benchmark/benchmark_dataset_memory_compare.R`
- `benchmark/plot_dataset_memory_compare.R`
- `benchmark/workflow_synthetic_variable_sweeps.sh`
- `benchmark/benchmark_synthetic_variable_sweeps.R`
- `benchmark/plot_synthetic_variable_sweeps.R`

Default outputs:

- `dataset_memory_compare_raw.csv`
- `dataset_memory_compare_summary.csv`
- one 4-row method plot per real dataset
- `synthetic_variable_sweeps_raw.csv`
- `synthetic_variable_sweeps_summary.csv`
- simulated n/p/q sweep plots

## Pipeline 2: comparison with external R packages

Purpose: compare `fastPLS` against independent PLS implementations available
in R packages on real datasets at dataset-specific component counts.

Run:

```sh
bash benchmark/run_pipeline2_package_comparison.sh
```

Core scripts:

- `benchmark/benchmark_pls_package_comparison.R`
- `benchmark/rearrange_pipeline2_package_tables.R`

Default outputs:

- `pls_package_comparison_raw.csv`
- `pls_package_comparison_summary.csv`
- package-comparison speed and prediction plots
- `rearranged_tables/pipeline2_plssvd_package_wide_table.csv`
- `rearranged_tables/pipeline2_simpls_package_wide_table.csv`
- `rearranged_tables/pipeline2_opls_package_wide_table.csv`
- `rearranged_tables/pipeline2_kernelpls_package_wide_table.csv`

The four wide tables have datasets as columns and function/package
implementations as rows.  Each implementation has two rows: predictive metric
and total fitting plus prediction time.  Failed runs, timeouts, memory kills,
and package limitations are retained in the table cells.

## Pipeline 3: single fit versus 10-fold cross-validation

Purpose: compare the cost of a single fit/prediction workflow with the
compiled 10-fold cross-validation workflow.

Run:

```sh
bash benchmark/run_pipeline3_cv_vs_fit.sh
```

Core script:

- `benchmark/benchmark_pipeline3_cv_vs_fit.R`

Default outputs:

- `pipeline3_cv_vs_fit_raw.csv`
- `pipeline3_cv_vs_fit_summary.csv`
- `pipeline3_cv_vs_fit_comparison.csv`
- `rearranged_tables/pipeline3_plssvd_cv10_wide_table.csv`
- `rearranged_tables/pipeline3_simpls_cv10_wide_table.csv`
- `rearranged_tables/pipeline3_opls_cv10_wide_table.csv`
- `rearranged_tables/pipeline3_kernelpls_cv10_wide_table.csv`

## Pipeline 4: ImageNet DINOv2 requested-SIMPLS classifier scaling

Purpose: request ImageNet-scale SIMPLS with randomized SVD on DINOv2 features,
including argmax, LDA, and candidate-kNN heads. Outputs retain both requested
and executed estimators. Large-class CUDA rows routed by the memory guard are
labelled as label-aware PLS-SVD rather than sequential SIMPLS.

Run:

```sh
bash benchmark/run_pipeline4_imagenet_simpls_rsvd.sh
```

Core script:

- `benchmark/benchmark_imagenet_simpls_rsvd_classifiers.R`

Default outputs:

- `imagenet_simpls_rsvd_classifiers_raw.csv`
- `imagenet_simpls_rsvd_classifiers_time.csv`
- `imagenet_simpls_rsvd_classifiers_joined.csv`
- run manifest and logs

Pipeline 4 expects a prepared ImageNet/DINOv2 task RDS on the remote machine.
The path can be overridden with `TASK_RDS`.

## Common environment variables

The launch scripts are controlled through environment variables so the same
code can run locally or on the CUDA workstation.  Frequently used variables
include:

- `FASTPLS_BENCH_LIB`: isolated R library for benchmark package installs.
- `FASTPLS_DATASETS`: comma-separated dataset list for pipeline 1.
- `FASTPLS_RUN_TIMEOUT_SEC`: per-run timeout for pipeline 1.
- `FASTPLS_PKG_COMPARE_RESULTS_DIR`: output directory for pipeline 2.
- `FASTPLS_PIPELINE3_RESULTS_DIR`: output directory for pipeline 3.
- `RUN_ROOT`, `TASK_RDS`, `NCOMP`, `BACKENDS`, `CLASSIFIERS`: pipeline 4 controls.

Datasets are not stored in this repository.  The loaders search the configured
benchmark data locations and stop with an explicit error if a dataset is not
available.
