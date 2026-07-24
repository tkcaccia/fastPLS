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

Successful configurations use all requested replicates. If an attempted
replicate ends in a timeout, process kill, package limitation, or deterministic
error, later replicates are recorded as `skipped_after_previous_failure`
instead of repeating the same failed computation.

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

## Pipeline 4: ImageNet DINOv2 SIMPLS classifier scaling

Purpose: benchmark ImageNet-scale SIMPLS with randomized SVD on float32 DINOv2
features, including argmax, LDA, and candidate-kNN heads. Outputs retain both
requested and executed estimators so any documented large-class memory routing
remains explicit. Candidate-kNN rows also identify their mixed-precision score
cache and memory mode rather than being described as end-to-end float32.
Candidate-kNN is treated as an optional downstream case study showing how the
supervised PLS score space can support a nonparametric classifier; it is not the
default decoder or part of the core PLS algorithm.

Run:

```sh
bash benchmark/run_pipeline4_imagenet_simpls_rsvd.sh
```

Core script:

- `benchmark/benchmark_imagenet_simpls_rsvd_classifiers.R`
- `benchmark/prepare_imagenet_float32_task.R`
- `benchmark/plot_imagenet_simpls_rsvd_classifiers.R`

Default outputs:

- `pipeline4_imagenet_raw.csv`
- `pipeline4_imagenet_summary.csv`
- accuracy, runtime, host-memory, and GPU-memory plots
- run manifest and logs

Pipeline 4 expects a prepared ImageNet/DINOv2 task RDS on the remote machine.
The path can be overridden with `TASK_RDS`.

## Publication rerun

`scripts/run_publication_benchmarks.sh` runs Pipelines 1-4 sequentially in an
isolated output tree. It uses float32 for the main real, simulated, package,
and ImageNet benchmarks; adds matched float64 runs for the precision-memory
comparison; and retains per-stage status files so a failed optional method does
not erase completed results. The matched precision summary only includes rows
whose recorded execution precision equals the requested comparison precision.
It reports both absolute peak RSS and incremental RSS above the isolated
process baseline, together with the input-storage reduction, prediction delta,
and execution scope. This avoids diluting the measured float32 memory benefit
with the fixed cost of loading R and the package.

The current float32 CUDA path keeps model arithmetic in float32 and accelerates
the randomized-SVD range products and supported classifier kernels on CUDA.
The outer PLS recurrence is still executed by the compiled host float32 core;
therefore publication summaries describe this path as GPU-accelerated rather
than claiming that the complete float32 fit is device-resident. Candidate-kNN
also records its mixed-precision latent-score cache explicitly.

`benchmark/plot_publication_backend_overview.R` converts the completed raw
real-dataset table into estimator-matched CPU-rSVD/CUDA-rSVD manuscript panels.
It excludes cKNN from the core backend comparison, requires matching executed
estimators and effective component counts, and reports runtime, CUDA speedup,
predictive deviation, and peak host memory with one shared external legend.

`benchmark/plot_publication_cknn_case_study.R` evaluates cKNN separately as an
optional downstream classifier on PLS scores. It compares cKNN with matched
argmax/LDA configurations across the ordinary classification datasets and then
shows ImageNet top-1, top-5, and prediction-time trajectories. This separation
prevents the cKNN case study from being interpreted as evidence about the core
PLS estimators.

`benchmark/plot_publication_precision_overview.R` creates a compact matched
float32/float64 figure for input storage, incremental host RSS, peak GPU memory,
and predictive deviation. It excludes cKNN because that classifier currently
uses a mixed-precision latent-score cache.

## Representative NMR spectrum figure

`benchmark/plot_nmr_spectrum_prediction.R` fits float32 SIMPLS with randomized
SVD to the prepared NMR task and overlays one observed test spectrum with its
prediction. To avoid selecting an unusually favorable example, the script uses
the test sample whose RMSD is closest to the median test-set RMSD. It saves the
observed and predicted spectrum, a residual panel, the plotted values, and
metadata containing the selected sample, RMSD, correlation, and timings.

`scripts/run_nmr_publication_figure_after_suite.sh` can wait for the publication
benchmark suite to finish and then generate the NMR figure without competing
with the benchmark for memory or compute resources.

`scripts/run_backend_publication_overview_after_suite.sh` applies the same
wait-until-complete rule to the estimator-matched backend overview.

`scripts/run_cknn_publication_case_study_after_suite.sh` waits for both the
real-dataset table and Pipeline 4 ImageNet summary before creating the cKNN
case-study tables and figure.

`scripts/run_precision_publication_overview_after_suite.sh` waits for the
matched precision table and creates the float32/float64 manuscript figure.

## Supplementary kernel sensitivity

`benchmark/benchmark_kernel_sensitivity.R` evaluates linear, RBF, and
polynomial kernel PLS on representative classification (MetRef and CCLE) and
multivariate-regression (PRISM and NMR) tasks. Kernel and component settings
are selected by five-fold cross-validation using the training data only. The
selected configuration for each kernel family is then refitted on the full
training set and evaluated on the unchanged test set with both CPU-rSVD and
CUDA-rSVD. Classification uses argmax throughout so the comparison isolates
the kernel rather than a downstream LDA or cKNN effect.

The RBF search uses `0.25`, `1`, and `4` times a median-distance scale estimated
from at most 512 training observations. Polynomial models use the same three
scale multipliers around `1 / p`, degrees 2 and 3, and intercepts 0 and 1.
Nonlinear kernels are deliberately not run on the sample-rich image and
single-cell tasks because their required `n x n` Gram matrix would make those
runs a quadratic-storage stress test rather than a useful kernel comparison.

`scripts/run_kernel_sensitivity_after_suite.sh` waits for the publication
suite, reuses its prepared task objects and package installation, records
isolated-process runtime, peak host RSS, and PID-specific CUDA memory, and then
runs `benchmark/plot_kernel_sensitivity.R`. Outputs include complete tuning,
selected-configuration, raw, summary, and failure tables plus separate
classification and regression figures whose facets share a y-axis within each
metric row.

`scripts/local_copy_publication_results_after_suite.sh` waits for the suite and
post-processing figures, then copies publication tables and figures locally.
It deliberately excludes task matrices, installed libraries, per-run row files,
and memory-sampling logs.

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
