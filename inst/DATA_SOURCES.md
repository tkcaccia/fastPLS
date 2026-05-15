# Bundled Example Data

fastPLS includes two small example datasets so the vignette can run without
requiring extra omics-data packages in `Suggests`.

- `colon` is derived from the `Colon` dataset distributed by the
  `plsgenomics` R package, version 1.5-3, licensed as GPL (>= 2).
- `breast` is derived from the `breast.TCGA` dataset distributed
  by the `mixOmics` R package, version 6.26.0, licensed as GPL (>= 2).
- `ccle` is a prepared CCLE/DepMap-derived expression benchmark subset.
- `gtex_v8` is a prepared GTEx v8 tissue-expression benchmark subset.
- `tcga_brca` is a prepared TCGA-BRCA expression subtype benchmark subset.
- `tcga_hnsc_methylation` is a prepared TCGA-HNSC methylation tumour/normal
  benchmark subset.
- `tcga_pan_cancer` is a prepared TCGA pan-cancer expression benchmark subset.

The source package licenses for `colon` and `breast` are GPL-compatible with
fastPLS, which is licensed as GPL-3. The TCGA and GTEx subsets are derived from
open-access, de-identified public research data and are bundled with source
attribution. The CCLE subset is derived from a local public CCLE/DepMap
benchmark matrix and is bundled with DepMap/CCLE attribution; users should
verify the license and terms attached to the exact DepMap/CCLE release before
redistribution outside this research package. The dataset help pages contain
the scientific references, source attribution, and dimensions.

CBMC CITE-seq and PRISM are not bundled in this source tree because the prepared
feature matrices were not present locally at packaging time. They should be
added only from the original public source files, with the applicable license or
data-use terms retained in this file and in the corresponding help pages.
