# Pipeline 3: fit/predict versus cross-validation

Pipeline 3 compares ordinary fitting plus prediction with the internal
10-fold cross-validation workflow.  It is intended to quantify the overhead
and speed benefit of compiled cross-validation for each PLS family.

## Run

```sh
bash benchmark/run_pipeline3_cv_vs_fit.sh
```

## Benchmark modes

- `fit_predict`: single model fit followed by test-set prediction.
- `cv10`: 10-fold cross-validation using `pls.single.cv`.

The default launch script runs both modes.  Use
`FASTPLS_PIPELINE3_BENCHMARK_MODES=cv10` to run only cross-validation.

## Main outputs

- `pipeline3_cv_vs_fit_raw.csv`
- `pipeline3_cv_vs_fit_summary.csv`
- `pipeline3_cv_vs_fit_comparison.csv`
- `rearranged_tables/pipeline3_plssvd_cv10_wide_table.csv`
- `rearranged_tables/pipeline3_simpls_cv10_wide_table.csv`
- `rearranged_tables/pipeline3_opls_cv10_wide_table.csv`
- `rearranged_tables/pipeline3_kernelpls_cv10_wide_table.csv`

The wide tables report one metric row and one runtime row per method/backend
configuration, with failed runs kept in the table cells.
