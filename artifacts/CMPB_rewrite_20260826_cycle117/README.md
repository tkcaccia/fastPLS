# CMPB manuscript cycle 117

This cycle adds the IKPLS 6.1.2 large-case float32 feasibility experiment to
the Methods, Results, Discussion, availability statement, and Supplementary
Material.

The comparison is explicitly not estimator matched. ImageNet IKPLS results are
reported at 100, 200, 500, and 1,000 components with preprocessing separated
from model timing. The NMR section records the degenerate one-component fit,
the guarded five-component memory failure, and the analytical 68.66-GiB
coefficient-path requirement at 50 components.

Benchmark scripts and compact reference results are in
`benchmark/ikpls_cross_language/`.
