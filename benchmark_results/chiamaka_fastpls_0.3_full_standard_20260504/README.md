# Chiamaka full standard benchmark, fastPLS 0.3

This directory stores the completed benchmark run produced on the Chiamaka
remote workstation from fastPLS commit `4c4896d9899f3da74d31cc5c775d18fb2ac12dfc`
(`Version: 0.3`).

Run root on Chiamaka:

```text
/home/chiamaka/fastPLS_usual_latest_return_variance_20260504_211939
```

The benchmark used the public `fastPLS::pls()` API with
`return_variance = FALSE` for timing fairness. This avoids including optional
predictor-space variance metadata calculation in the reported fit time while
leaving the default user-facing plotting behavior unchanged.

## Contents

```text
real_datasets/dataset_memory_compare_raw.csv
real_datasets/dataset_memory_compare_summary.csv
real_datasets/plots/*_4x4_methods_memory.png
simulated_datasets/synthetic_variable_sweeps_raw.csv
simulated_datasets/synthetic_variable_sweeps_summary.csv
simulated_datasets/synthetic_variable_sweeps_manifest.txt
simulated_datasets/plots/*_4x4_synthetic_variable_sweep.png
```

## Real datasets

The real-dataset benchmark contains 3955 raw rows and plots for:

```text
metref
ccle
cifar100
prism
gtex_v8
tcga_pan_cancer
singlecell
tcga_brca
tcga_hnsc_methylation
nmr
cbmc_citeseq
```

Each plot uses the 4-row benchmark layout:

```text
time
predictive metric
peak host RSS
peak GPU memory
```

## Simulated datasets

The simulated benchmark contains 3285 raw rows and 1095 summary rows. It sweeps:

```text
class_n
class_p
class_q
reg_n
reg_p
reg_q
```

## Failed or skipped-heavy rows

The run completed. The recorded non-OK rows were expected heavy-path failures or
timeouts:

```text
cifar100 pls_pkg_opls ncomp=200: killed_timeout
nmr pls_pkg_simpls ncomp=2: killed_timeout
nmr pls_pkg_kernelpls ncomp=2: killed_timeout
nmr cpp_opls_cpu_rsvd ncomp=2: killed_timeout
nmr cpp_opls_irlba ncomp=2: killed_timeout
nmr gpu_opls_fp64 ncomp=2: killed_timeout
nmr pls_pkg_opls ncomp=5,10: killed_sig9
nmr pls_pkg_opls ncomp=20,50,100,200,500: allocation errors from 52.2 GB to 1370.5 GB
```

No simulated benchmark rows failed.
