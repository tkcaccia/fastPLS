# Manuscript evidence archive

This file indexes detailed analyses that support the fastPLS manuscript but are
not repeated in the compact submission supplement. The compact supplement is
the authoritative presentation. This archive preserves full component paths,
review-cycle sensitivity analyses, diagnostic plots, and machine-readable
outputs for reproducibility.

## Authoritative submission evidence

| Claim | Authoritative source |
|---|---|
| Backend residency | Supplementary Table S1 |
| Deterministic estimator validation | Supplementary Table S7 |
| rSVD reliability | Supplementary Table S8 |
| Float32 capability | Supplementary Table S9 |
| External software comparison | Supplementary Table S10 |
| Selected-point performance | Supplementary Table S11 |
| Analysis provenance | Supplementary Table S15 |

The claim-to-evidence map in Supplementary Table S6 identifies the single
authoritative source for every central main-text claim.

## Archived detailed evidence

| Topic removed from the compact supplement | Preserved result archive or script |
|---|---|
| Full cycle-66 review document | `artifacts/CMPB_rewrite_20260726_cycle66/fastPLS_CMPB_supplement_cycle66_0.99.6_20260726.docx` |
| SIMPLS component-level deterministic and rSVD endpoints | `benchmark_results/simpls_estimator_preservation_reliable_power2_final_20260725/` |
| OPLS and kernel-PLS equation-level validation | `benchmark_results/opls_kernel_setting_reliability_20260726/` |
| CPU rSVD reliability endpoints | `benchmark_results/simpls_estimator_preservation_reliable_power2_final_20260725/simpls_estimator_approximation_rsvd.csv` |
| CUDA rSVD reliability grid | `benchmark_results/rsvd_cuda_reliability_20260725.csv` |
| Full selected-component paths for all datasets | `benchmark_results/manuscript_revision_cycle48_20260726/` |
| Selected CPU/CUDA rows and predictive intervals | `benchmark_results/manuscript_revision_cycle20_20260725/` |
| Baseline and incremental host/GPU memory | `benchmark_results/manuscript_revision_cycle21_20260725/` |
| Float32/float64 paired resource measurements | `benchmark_results/manuscript_revision_cycle13_20260725/` |
| External-package method-level results | `benchmark_results/manuscript_revision_cycle57_20260726/` |
| NMR component paths, errors, and matched backend contrasts | `benchmark_results/manuscript_revision_cycle64_20260726/` |
| ImageNet classification and retrieval paths | `benchmark_results/manuscript_revision_cycle54_20260726/` |
| Repeated outer-partition uncertainty | `benchmark_results/manuscript_revision_cycle66_20260726/` |
| SIMPLS versus PLS-SVD matched-shape experiment | `benchmark_results/simpls_vs_plssvd_shapes_20260726_cuda/` |
| Compiled versus R-level cross-validation | `benchmark_results/cv_compiled_vs_r_loop_20260725/` |
| Simulated sample, predictor, and response sweeps | `benchmark_results/manuscript_revision_cycle33_20260725/` |
| Analysis-to-source provenance ledger | `benchmark_results/manuscript_revision_cycle63_20260726/analysis_commit_provenance.csv` |

## Archival policy

- The files above remain available for audit and reanalysis.
- Superseded table and figure numbers are not cited by the main manuscript.
- New manuscript claims must be added first to the claim-to-evidence map.
- Historical archives lacking a recorded Git commit remain labelled
  `not recoverable; not inferred`; later commits are not assigned
  retrospectively.
