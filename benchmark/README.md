# fastPLS benchmark pipelines

This folder contains the benchmark workflows used to evaluate `fastPLS`.
The four main pipelines can be launched from this directory through small
wrapper scripts.  The wrappers call the lower-level scripts in `scripts/`
and keep the benchmark entry points visible in one place.

Dataset redistribution and reproducible acquisition are documented in
[`DATA_ACQUISITION.md`](DATA_ACQUISITION.md). Run
`Rscript benchmark/acquire_publication_datasets.R --list` to see the prepared
object names and access classes expected by the benchmark loader. Prepared
real-data matrices are not stored in this repository.

The submission manuscript uses a compact, authoritative evidence set.
[`MANUSCRIPT_EVIDENCE_ARCHIVE.md`](MANUSCRIPT_EVIDENCE_ARCHIVE.md) maps each
definitive Supplementary table to its result archive and indexes expanded
component paths, sensitivity analyses, diagnostics, and superseded review-cycle
material that are retained for audit but not duplicated in the Supplement.

## Run provenance

Publication runs must record source provenance before computation. Use:

```sh
Rscript benchmark/write_run_provenance.R \
  --analysis=analysis_id \
  --output=RESULTS_DIR/run_provenance.csv \
  --script=benchmark/the_benchmark_script.R \
  --data=prepared_task_identifier \
  --split=split_identifier \
  --seed=123
```

The record contains the package repository commit and dirty state, benchmark
script checksum, installed `fastPLS` version, reusable-core commit when a core
repository is supplied, and data/split identifiers. The accompanying
`run_provenance.csv.session_info.txt` captures the R environment. A package
version is not a substitute for a Git commit, and a manuscript-generation
commit must never be assigned retrospectively to an older result archive.

Unless a sensitivity analysis explicitly states otherwise, randomized-SVD
publication runs use `oversample = 10`, `power = 2`, and `seed = 123`. Scripts
must write these effective values into every result row.

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
compiled 10-fold cross-validation workflow. This quantifies cross-validation
overhead relative to one fit; it is not an acceleration benchmark.

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

For a matched acceleration benchmark, use
`benchmark/benchmark_cv_compiled_vs_r_loop.R` through
`scripts/run_cv_compiled_vs_r_loop_remote.sh`. It compares the compiled engine
with an explicit R-level fold loop that calls the same `fastPLS::pls()`
estimator on the same prespecified folds. The output records paired runtime,
fold-partition identity, predictive metrics, and out-of-fold prediction
agreement.

## SIMPLS implementation ablation

`benchmark/benchmark_simpls_multidataset_ablation.R` and
`scripts/run_simpls_multidataset_ablation_remote.sh` isolate five execution
optimizations in deterministic CPU SIMPLS/IRLBA: cached `X'X`, incremental
coefficient-path updates, cached deflation products, compact prediction, and
matrix-free cross-covariance products. Each reference/optimized configuration
runs in a fresh R process on the same stored train/test task, seed, scaling,
component path, and classifier.

The shell driver records RSS immediately after data loading and garbage
collection, then samples process RSS only during fitting and prediction.
Consequently, the reported incremental peak is the fit-window peak minus the
pre-fit baseline rather than absolute process RSS. The summarizer pairs
replicates, verifies prediction agreement, and reports median runtime speedup,
runtime IQR, incremental-RSS reduction, RSS IQR, and predictive-metric
difference. Cached `X'X` is explicitly marked not applicable when its
shape-based production condition is not met. The matrix-free route is treated
as a memory/runtime trade-off, not assumed to be faster.

Run and summarize:

```sh
bash scripts/run_simpls_multidataset_ablation_remote.sh RESULTS_DIR
Rscript benchmark/summarize_simpls_multidataset_ablation.R RESULTS_DIR
Rscript benchmark/plot_simpls_multidataset_ablation.R RESULTS_DIR
```

## Pipeline 4: ImageNet DINOv2 SIMPLS classifier scaling

Purpose: benchmark ImageNet-scale SIMPLS with randomized SVD on float32 DINOv2
features using the native argmax and LDA classification heads. Outputs retain
both requested and executed estimators and explicitly distinguish fully
resident from hybrid accelerator routes.

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
than claiming that the complete float32 fit is device-resident.

`benchmark/plot_publication_backend_overview.R` converts the completed raw
real-dataset table into estimator-matched CPU-rSVD/CUDA-rSVD manuscript panels.
It evaluates the native argmax and LDA heads, requires matching executed
estimators and effective component counts, and reports runtime, CUDA speedup,
predictive deviation, and peak host memory with one shared external legend.

`benchmark/benchmark_imagenet_faiss_matched_retrieval.R` is a separate external
retrieval experiment. It compares exact FAISS search on raw DINOv2, PCA scores,
and PLS scores while including representation fitting, held-out transformation,
index construction, and query time. PCA is computed directly in the benchmark;
it is not a fastPLS package function. This experiment is not part of the native
PLS classification API.

`benchmark/plot_publication_precision_overview.R` creates a compact matched
float32/float64 figure for input storage, incremental host RSS, peak GPU memory,
and predictive deviation.

## Qualified NMR and ImageNet evidence

The current manuscript controls are generated by:

- `benchmark/benchmark_nmr_qualified_solver.R`
- `scripts/run_nmr_qualified_solver_remote.sh`
- `scripts/run_nmr_gpu_memory_probe_remote.sh`
- `benchmark/plot_nmr_cycle80_qualified.R`
- `benchmark/benchmark_imagenet_current_fused_lda.R`
- `benchmark/benchmark_imagenet_float32_simpls_lda_path.R`
- `scripts/run_imagenet_float32_simpls_lda_path_remote.sh`
- `benchmark/plot_imagenet_cycle80_qualified.R`

The NMR solver comparison fixes the data split, preprocessing, family,
component count, float64 precision, and randomized-SVD controls. The ImageNet
component path uses stored float32 features, label-aware responses, and blocked
held-out top-5 prediction with online metric accumulation. This avoids
materializing a full held-out score matrix while preserving the fitted model
and class decisions. Both workflows record source-archive SHA-256, requested
and executed estimators, solver controls, host RSS, device memory, and failures.

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
the kernel rather than a downstream classification-head effect.

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

## OPLS and kernel-PLS settings

New pipeline-1 rows record the OPLS total component budget, predictive
component count, and `north`, plus the kernel family and kernel parameters for
kernel-PLS. The principal multi-dataset benchmark prespecifies one orthogonal
OPLS component and uses a linear kernel. Thus its kernel-PLS rows are
linear-kernel implementation controls, not nonlinear-kernel results.

`benchmark/summarize_main_benchmark_model_settings.R` reconstructs the same
metadata for the archived selected-point benchmark. Nonlinear RBF and
polynomial models are evaluated separately by
`benchmark/benchmark_kernel_sensitivity.R`, with training-only selection of
the kernel parameters and component count.

## Independent OPLS and nonlinear kernel-PLS validation

`benchmark/benchmark_opls_kernel_estimator_validation.R` validates
deterministic OPLS and nonlinear RBF/polynomial kernel PLS without calling the
fastPLS filtering or kernel-construction helpers for the reference path. The
OPLS reference implements the Trygg-Wold orthogonal-filter equations directly,
and the nonlinear reference independently constructs and centres the Gram
matrices. Both references then use `pls::simpls.fit`.

The fixed design covers regression and classification, `p < n`, `p > n`, an
ill-conditioned synthetic design, gasoline spectroscopy, and breast molecular
classification. Outputs report operator, coefficient, prediction, score
subspace, decoded-label, predictive-metric, failure, and fixed five-fold
component-selection agreement. Deterministic IRLBA evidence is kept separate
from approximate rSVD workflow results.

Run:

```sh
Rscript benchmark/benchmark_opls_kernel_estimator_validation.R \
  --root=. \
  --out=benchmark_results/opls_kernel_estimator_validation
```

The output directory contains endpoint, tolerance, failure, fold-level,
component-path, selected-component, session, and Markdown report files.

## Float32 capability summary

`benchmark/summarize_float32_capability.R` writes the implementation and
validation status of each float32 method/backend combination to
`float32_capability_table.csv`. The table separates supported execution from
the strength of available numerical evidence and records the automatic
shape-based warnings used by `pls()`.

## Repeated outer-partition uncertainty

`benchmark/benchmark_repeated_outer_selection.R` repeats training-only
component selection and outer-test evaluation across fixed outer-partition
seeds. It was used for representative small, medium, and large biomedical
classification tasks and for NMR regression. The script records every
successful or failed route, the selected component count, boundary status,
predictive metric, and elapsed time.

`benchmark/summarize_repeated_outer_selection.R` combines the dataset-level
outputs, reports selection frequencies and predictive dispersion, and creates
the supplementary figures. Endpoint selections are described as best within
the evaluated grid; response-rank-constrained PLS-SVD endpoints remain marked
as structural boundaries.

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
