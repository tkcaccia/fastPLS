# Bundled Example Data

fastPLS includes two small example datasets so the vignette can run without
requiring extra omics-data packages in `Suggests`.

- `colon` is derived from the `Colon` dataset distributed by the
  `plsgenomics` R package, version 1.5-3, licensed as GPL (>= 2).
- `breast` is derived from the `breast.TCGA` dataset distributed
  by the `mixOmics` R package, version 6.26.0, licensed as GPL (>= 2).
The source package licenses for `colon` and `breast` are GPL-compatible with
fastPLS, which is licensed as GPL-3. The dataset help pages contain the
scientific references, source attribution, and dimensions.

Large real benchmark matrices such as CCLE/DepMap, GTEx v8, TCGA-BRCA,
TCGA-HNSC methylation, and TCGA pan-cancer are not bundled in the source package
to keep installation lightweight. The benchmark scripts load those matrices from
local benchmark data directories, where users can retain the license and
data-use terms attached to the exact releases they downloaded.

CBMC CITE-seq and PRISM are not bundled in this source tree because the prepared
feature matrices were not present locally at packaging time. They should be
added only from the original public source files, with the applicable license or
data-use terms retained in this file and in the corresponding help pages.
