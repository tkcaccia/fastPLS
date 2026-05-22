# Pipeline 2: comparison with external R PLS packages

Pipeline 2 compares `fastPLS` with independent PLS implementations available
in R packages.  It uses the same train/test split and dataset-specific number
of components for each method.

## Run

```sh
bash benchmark/run_pipeline2_package_comparison.sh
```

## Method families

The output is organized into four model-family tables:

- PLSSVD
- SIMPLS/PLS
- OPLS
- kernel-PLS

PLSSVD is available only in `fastPLS`.  SIMPLS/PLS, OPLS, and kernel-PLS are
compared with external packages where the corresponding method is available.

## Main outputs

- `pls_package_comparison_raw.csv`
- `pls_package_comparison_summary.csv`
- package-comparison speed plots
- package-comparison prediction plots
- `rearranged_tables/pipeline2_plssvd_package_wide_table.csv`
- `rearranged_tables/pipeline2_simpls_package_wide_table.csv`
- `rearranged_tables/pipeline2_opls_package_wide_table.csv`
- `rearranged_tables/pipeline2_kernelpls_package_wide_table.csv`

The wide tables place datasets in columns and implementations in rows.  Each
implementation has a metric row and a runtime row.  Package limitations,
timeouts, killed processes, and implementation errors are reported explicitly.
