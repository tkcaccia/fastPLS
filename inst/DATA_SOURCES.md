# Example And Benchmark Data

fastPLS does not redistribute third-party biomedical datasets. Package
vignettes use base R datasets for concise examples and do not bundle a
synthetic or processed benchmark dataset.

Prepared real-data benchmark matrices are not bundled or redistributed, even
when their upstream source is publicly downloadable. This avoids repackaging a
processed copy under the fastPLS package licence when the source provider's
terms may apply separately. Users acquire the authoritative source, retain its
terms, and generate the prepared benchmark object locally.

The companion benchmark repository may contain scripts that generate
simulation inputs, together with aggregate result tables, plots, run manifests,
split indices, and checksums that contain no source matrices or
participant-level records.

Dataset-by-dataset access classes, authoritative source links, release
requirements, and executable acquisition or local-validation commands are
maintained with the benchmark workflows in the companion fastPLS-extra
repository. ImageNet, NMR, historical CCLE, and release-specific PRISM inputs
require a user-authorized local copy and are never downloaded or substituted
automatically.
