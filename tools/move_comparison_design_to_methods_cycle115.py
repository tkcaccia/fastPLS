from pathlib import Path

from docx import Document


ROOT = Path("/Users/stefano/Documents/GPUPLS/fastPLS_fresh_github")
SOURCE_DIR = ROOT / "artifacts/CMPB_rewrite_20260826_cycle114"
OUTDIR = ROOT / "artifacts/CMPB_rewrite_20260826_cycle115"


EXTERNAL_METHODS = (
    "To distinguish numerical computation from the cost of constructing ordinary model "
    "objects, external timing was assessed under two predefined output profiles. The "
    "minimum-output profile compared deterministic float64 SIMPLS while requiring the "
    "complete coefficient path, centering quantities, and final held-out predictions; "
    "scores, loadings, fitted-value arrays, and variance summaries were suppressed, and "
    "pls::simpls.fit was called with stripped = TRUE. The public-workflow profile retained "
    "each implementation's ordinary model object together with the final predictions. "
    "The strict paired comparison comprised nine classification datasets, two output "
    "profiles, fastPLS and pls::simpls.fit, and three fresh-process repetitions, yielding "
    "108 planned runs. A broader workflow panel compared fastPLS argmax and LDA with the "
    "available classification workflows from independent R PLS packages on the same nine "
    "datasets; package-specific model objects and unsupported configurations were retained "
    "and interpreted as an end-to-end software comparison. Package and data loading were "
    "completed before timing, no numerical warm-up was applied, and each repetition used "
    "one effective BLAS thread and a common 10,000-s timeout. The analysis retained median "
    "time, interquartile range, completed repetitions, failures, model-object size, baseline "
    "and peak process resident-set size, and final held-out predictions."
)


IKPLS_METHODS = (
    "Numerical equivalence and software-level performance were assessed separately. "
    "Deterministic float64 fastPLS SIMPLS was first compared with the de Jong SIMPLS "
    "implementation in pls::simpls.fit to evaluate the numerical kernel. A separate "
    "archived-release CPU experiment compared fastPLS 0.99.25 with the two NumPy "
    "formulations provided by IKPLS 6.1.2. The score-explicit formulation, termed "
    "Algorithm 1 in IKPLS, computes and retains the training-score matrix, whereas the "
    "cross-product formulation, termed Algorithm 2, fits from the predictor cross-product "
    "and predictor-response cross-product without retaining training scores. The experiment "
    "used Breast, MetRef, and CIFAR-100 with 10, 22, and 50 components, respectively. Both "
    "IKPLS formulations, deterministic fastPLS IRLBA, and approximate fastPLS rSVD were "
    "executed three times for each dataset, giving 36 planned runs. All routes used identical "
    "stored splits, externally training-centred float64 predictors, centred one-hot "
    "responses, component counts, and final held-out predictions, and each run used a fresh "
    "process with one effective thread. Because IKPLS and SIMPLS implement different "
    "estimators and retain different internal state, this experiment was interpreted as an "
    "end-to-end software comparison rather than an estimator-matched benchmark. An earlier "
    "JAX/CUDA development comparison could not be reproduced from the frozen environment "
    "and was therefore excluded from the central evidence."
)


NMR_METHODS = (
    "The NMR case study comprised 1,200 training and 321 held-out spectra, with 13,000 "
    "predictors and 28,355 response intensities. The 4.6-4.8 ppm predictor interval was "
    "defined a priori from the chemical-shift labels and set to zero in both training and "
    "test predictors before any training-only split; no response coordinate was masked and "
    "no held-out response information informed preprocessing. Component selection used five "
    "paired training-only splits, family-specific grids spanning 1-300 components, and the "
    "one-standard-error rule. Family-selected implementation comparisons held the split, "
    "precision, response target, and component count fixed. Approximate rSVD used "
    "oversampling 20, two power iterations, and seed 123. A separate implementation-only "
    "analysis fixed both families at 100 components, and the deposited 165-component "
    "PLS-SVD/IRLBA workflow was treated only as historical scientific context."
)


IMAGENET_METHODS = (
    "The historical ImageNet/DINOv2 feasibility analysis used 1,281,167 stored "
    "1,024-dimensional embeddings, partitioned noncanonically into 1,000,000 training and "
    "281,167 held-out rows. The exact DINOv2 checkpoint, pooling rule, feature-extraction "
    "script, and independently auditable image-to-row mapping were unavailable. Downstream "
    "label-aware float32 CUDA-rSVD SIMPLS fitting and blocked prediction were rerun with "
    "fastPLS 0.99.25. Argmax and LDA were evaluated at the shared component prefixes 100, "
    "200, 300, 400, 500, 600, 700, 800, 900, and 1,000; one maximal fit supplied every "
    "prefix. rSVD used oversampling 20, two power iterations, and seed 123. Because the split "
    "had informed earlier component choices and each configuration was measured once, this "
    "analysis was predefined as a partially reproducible engineering stress test rather than "
    "a comparative or biomedical predictive evaluation."
)


RESULT_EXTERNAL = (
    "All 108 planned paired runs completed, and held-out accuracy was identical for every "
    "fastPLS-pls::simpls.fit pair. With minimum common prediction outputs, fastPLS was "
    "faster on four datasets and pls::simpls.fit on five; the largest fastPLS advantage was "
    "1.48-fold on GTEx v8. Under ordinary public workflows, fastPLS was faster than "
    "pls::simpls.fit on five datasets, including 2.39-fold on CIFAR-100, 3.35-fold on "
    "Retina, and 4.85-fold on Tabula Muris. Corresponding exact held-out counts were "
    "8,739/10,000, 21,684/22,406, and 40,077/50,059. These are conditional row-level "
    "computational endpoints; because Retina and Tabula Muris rows are cells rather than "
    "independent biological replicates, binomial intervals are not interpreted as biological "
    "generalization intervals. In the broader R-package panel, fastPLS had the lowest "
    "observed total time on seven of nine classification datasets. The exceptions were "
    "TCGA-BRCA and TCGA-HNSC, where plsgenomics was faster (Figure 2; Supplementary Figure "
    "S3 and Tables S10a-S10e)."
)


RESULT_CONTROLLED = (
    "Every automatic route completed execution, but numerical qualification was not uniform: "
    "agreement fell outside the predefined tolerances in parts of the retained-component, "
    "class-count, cross-covariance-rank, and CUDA response-dimension sweeps. Runtime and "
    "memory summaries therefore exclude those discordant routes rather than treating "
    "successful execution as numerical validation."
)


RESULT_ACCELERATOR = (
    "The computational benefit of hardware acceleration depended on both the execution route "
    "and workload dimensions. Among the 44 CPU/CUDA pairs, 28 met the predefined numerical-"
    "concordance criteria and seven favored CUDA, including an 8.90-fold PLS-SVD runtime "
    "ratio on CIFAR-100 (Figure 3). Six of the 12 CPU/Metal pairs met the same criteria, but "
    "none favored Metal. Discordant routes remain visible as gray cells and were not converted "
    "into acceleration claims. A separate frozen-release SIMPLS-rSVD comparison is reported "
    "in Supplementary Figure S4 and Table S11. CPU IRLBA remains the deterministic numerical "
    "reference, and float32 did not uniformly improve runtime, process memory, or numerical "
    "agreement (Supplementary Tables S8-S9)."
)


RESULT_NMR_SELECTION = (
    "Training-only selection by the one-standard-error rule retained five PLS-SVD components "
    "and 50 SIMPLS components, both interior to their evaluated grids. At these settings, "
    "PLS-SVD achieved RMSD 0.001043 and Q² 0.98916 across CPU IRLBA, CPU rSVD, and CUDA "
    "rSVD. SIMPLS achieved RMSD 0.00075608, 0.00075595, and 0.00075606 and Q² 0.994299, "
    "0.994301, and 0.994299, respectively. Figure 4 displays held-out sample AMI-0030-9 "
    "(index 38), selected by the predefined rule of closest per-spectrum RMSD to the median "
    "under 50-component SIMPLS CUDA rSVD rather than by visual concordance."
)


RESULT_NMR_TIME = (
    "At five PLS-SVD components, CPU IRLBA, CPU rSVD, and CUDA rSVD required 262.86, 4.57, "
    "and 0.422 s, respectively. At 50 SIMPLS components, the corresponding times were "
    "692.21, 152.09, and 3.91 s. Every approximate row met the predefined numerical "
    "tolerances. The separate 100-component implementation comparison is reported in the "
    "Supplement because it addresses a different question from family-specific predictive "
    "selection."
)


RESULT_IMAGENET = (
    "Archived fastPLS 0.99.25 completed the million-row SIMPLS fit and blocked prediction of "
    "all 281,167 held-out embeddings (Figure 5). The resulting top-1 and top-5 trajectories, "
    "runtime, and memory measurements demonstrate downstream matrix-processing feasibility. "
    "They do not establish representation-level reproducibility, biomedical utility, or an "
    "optimized ImageNet classifier, and the 1,000-component endpoint remains a boundary "
    "stress point. Full values and provenance limitations are reported in Supplementary "
    "Section S18 and Table S13."
)


def find_start(document: Document, prefix: str):
    matches = [p for p in document.paragraphs if p.text.startswith(prefix)]
    if len(matches) != 1:
        raise RuntimeError(f"Expected one paragraph starting with {prefix!r}, found {len(matches)}")
    return matches[0]


def delete_paragraph(paragraph) -> None:
    element = paragraph._element
    element.getparent().remove(element)
    paragraph._p = paragraph._element = None


def insert_before(reference, text: str, style: str):
    paragraph = reference.insert_paragraph_before(text)
    paragraph.style = style
    return paragraph


def main() -> None:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    document = Document(SOURCE_DIR / "fastPLS_CMPB_main_cycle114_0.99.25_20260826.docx")

    find_start(document, "To distinguish numerical computation").text = EXTERNAL_METHODS
    find_start(document, "Numerical equivalence and software-level performance").text = IKPLS_METHODS

    results_heading = find_start(document, "3. Results")
    insert_before(results_heading, "2.8 Large-scale case-study protocols", "Heading 2")
    insert_before(results_heading, NMR_METHODS, "First Paragraph")
    insert_before(results_heading, IMAGENET_METHODS, "Body Text")

    delete_paragraph(find_start(document, "Component counts are described"))
    find_start(document, "The strict comparison completed all 108").text = RESULT_EXTERNAL
    find_start(document, "The archived controlled study isolated").text = RESULT_CONTROLLED
    delete_paragraph(find_start(document, "Development-stage same-code ablations"))
    find_start(document, "The computational benefit of hardware acceleration").text = RESULT_ACCELERATOR
    find_start(document, "NMR comprised 1,200 training").text = RESULT_NMR_SELECTION
    delete_paragraph(find_start(document, "At the family-selected settings"))
    find_start(document, "A matched family-selected analysis").text = RESULT_NMR_TIME
    find_start(document, "Foundation-model embeddings are increasingly relevant").text = RESULT_IMAGENET

    document.save(OUTDIR / "fastPLS_CMPB_main_cycle115_0.99.25_20260826.docx")


if __name__ == "__main__":
    main()
