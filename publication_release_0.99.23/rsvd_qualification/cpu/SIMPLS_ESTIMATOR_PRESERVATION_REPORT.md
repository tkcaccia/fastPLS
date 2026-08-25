# SIMPLS estimator-preservation validation

The deterministic IRLBA and approximate rSVD results are intentionally separated.

## Prespecified deterministic tolerances
- Relative prediction error <= 1.0e-04
- Relative coefficient error <= 1.0e-03 where X has full column rank
- Maximum score, projection, and loading subspace angle <= 0.100 degrees
- Classification label agreement >= 0.995
- Classification accuracy difference <= 0.005
- Regression RMSD difference <= 1.0e-04
- Cross-validation selected-component agreement is reported exactly.

## Prespecified rSVD approximation criteria
- Relative prediction error <= 0.05
- Prediction correlation >= 0.99
- Maximum score, projection, and loading subspace angle <= 10.0 degrees
- Classification label agreement >= 0.99
- Predictive-metric difference <= 0.01
- An rSVD row that violates any criterion is labelled failed_approximation_criteria.

## Aggregate results
 endpoint_runs endpoint_failures cv_runs cv_failures
           168                 0      72           0
 deterministic_endpoint_rows deterministic_endpoint_tolerance_passes
                         117                                     117
 deterministic_endpoint_tolerance_failures rsvd_endpoint_rows
                                         0                585
 rsvd_approximation_passes rsvd_approximation_failures
                       585                           0
 deterministic_cv_selection_agreement
                                    1

## rSVD variability across prespecified seeds
 randomized_seed rows passes failures maximum_prediction_relative_error
               1  117    117        0                      1.239524e-09
               7  117    117        0                      1.239524e-09
              19  117    117        0                      1.239524e-09
              43  117    117        0                      1.239524e-09
             123  117    117        0                      1.239524e-09
 minimum_prediction_correlation minimum_classification_label_agreement
                              1                                      1
                              1                                      1
                              1                                      1
                              1                                      1
                              1                                      1
 maximum_metric_absolute_difference
                       5.315811e-10
                       5.315811e-10
                       5.315811e-10
                       5.315811e-10
                       5.315811e-10

## Interpretation
IRLBA rows test the estimator-preservation claim against pls::simpls.fit.
rSVD rows characterize an explicitly approximate direction solver and are not used as equivalence evidence.
IRLBA is preferred for confirmatory inference, ill-conditioned or rank-deficient data, slowly decaying singular spectra, and any task that fails the rSVD approximation criteria.
