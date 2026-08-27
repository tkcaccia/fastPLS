from pathlib import Path

from docx import Document


ROOT = Path("/Users/stefano/Documents/GPUPLS/fastPLS_fresh_github")
SOURCE = ROOT / "artifacts/CMPB_rewrite_20260826_cycle109/fastPLS_CMPB_main_cycle109_0.99.25_20260826.docx"
OUTDIR = ROOT / "artifacts/CMPB_rewrite_20260826_cycle110"
OUTPUT = OUTDIR / "fastPLS_CMPB_main_cycle110_0.99.25_20260826.docx"


INTRO_OLD = [
    "Partial least squares (PLS) combines supervised dimension reduction with prediction and is widely used when biomedical predictors are correlated, high dimensional, or accompanied by multivariate responses [1-3]. Applications include cancer metabolomics and prediction of multiple NMR spectra from one acquisition [4-7].",
    "Increasing spectral resolution, single-cell sample counts, foundation-model embeddings, and repeated validation make PLS computationally demanding. KODAMA, for example, repeatedly fits cross-validated PLS discriminant models [8,9]. In medical imaging, foundation models such as UNI and Prov-GigaPath generate large dense tile or slide representations for downstream supervised analysis [31,32]. Historical ImageNet/DINOv2 embeddings were therefore retained only as a partially reproducible engineering proxy for this post-extraction matrix regime, not as biomedical validation [28,29].",
    "Among PLS formulations, SIMPLS constructs sequential components without explicitly deflating the predictor matrix [11]. PLS-SVD provides a one-shot cross-covariance comparator [10], while OPLS and kernel PLS extend the framework to orthogonal filtering and nonlinear relations [12-14]. Reference software such as pls::simpls.fit already returns coefficient and fitted-value arrays for every component prefix; fastPLS does not claim otherwise. Its distinct execution features are compact latent prediction, shape-dependent intermediate storage, optional implicit cross-covariance products, compiled low-rank solvers, and route-qualified CUDA and Metal support. Supplementary Table S1a defines this feature comparison explicitly.",
    "1.1 Related high-performance PLS software",
    "A direct software comparator is IKPLS, which implements the two Improved Kernel PLS algorithms of Dayal and MacGregor using NumPy for CPU execution and JAX for CPU, GPU, and TPU execution [33,34]. IKPLS also combines fold-wise cross-product updates with vectorized JAX execution for fast validation [35]. It is not de Jong SIMPLS and therefore cannot establish estimator preservation, but it is an appropriate high-performance end-to-end comparator. Other accelerated PLS work includes iterative or randomized low-rank direction extraction, compressed or implicit products, parallel dense linear algebra, and GPU execution; these approaches trade different estimators, storage policies, precision, or compilation overhead against runtime [15,16,33-35].",
    "fastPLS addresses a different software regime: an R interface spanning PLS-SVD, de Jong SIMPLS, OPLS, and kernel PLS; compact latent prediction; optional implicit predictor-response products for large multivariate responses; compiled single and nested validation; and explicit solver/backend diagnostics. We therefore separate numerical-kernel validation against pls::simpls.fit from end-to-end software comparisons against IKPLS and other independent implementations.",
    "We present fastPLS, whose principal methodological contribution is a compiled, shape-dependent execution of de Jong SIMPLS. Sequential deflation, orthogonalization, and component definitions are retained, while avoidable intermediate products and dense outputs are reduced according to matrix shape and requested output. Low-rank solvers, implicit products, float32, cross-validation, LDA, and CPU/CUDA/Metal execution are supporting options around this core estimator. OPLS and kernel PLS reuse the same optimized SIMPLS engine. The GPL-3 R package uses reusable components from the MIT-licensed kodama-cpp project.",
]


INTRO_NEW = [
    "Partial least squares (PLS) combines supervised dimension reduction with prediction and is widely used when biomedical predictors are correlated, high dimensional, or accompanied by multivariate responses [1-3]. Applications include cancer metabolomics and prediction of multiple nuclear magnetic resonance (NMR) spectra from one acquisition [4-7].",
    "NMR spectrum prediction illustrates a particularly demanding multivariate setting. In the previous scientific workflow that motivated this work, 1,200 spectra represented by 13,000 NOESY predictor bins were used to predict 28,355 diffusion-edited spectral intensities [7]. Unlike prediction of a single clinical endpoint, this task requires fitting and evaluating tens of thousands of correlated responses simultaneously. The predictor-response cross-covariance alone contains approximately 369 million values, or 2.75 GiB in float64, before latent factors, coefficients, predictions, and validation workspaces are considered. Repeated component selection further multiplies this cost, making both execution time and memory central methodological constraints.",
    "The same computational pressure arises from increasing spectral resolution, single-cell sample counts, foundation-model embeddings, and repeated validation. KODAMA, for example, repeatedly fits cross-validated PLS discriminant models [8,9]. In medical imaging, foundation models such as UNI and Prov-GigaPath generate large dense tile or slide representations for downstream supervised analysis [31,32]. Historical ImageNet/DINOv2 embeddings were therefore retained only as a partially reproducible engineering proxy for this post-extraction matrix regime, not as biomedical validation [28,29].",
    "Several PLS formulations and software implementations address different parts of this problem. SIMPLS constructs sequential components without explicitly deflating the predictor matrix [11]; PLS-SVD provides a one-shot cross-covariance formulation [10]; and OPLS and kernel PLS extend the framework to orthogonal filtering and nonlinear relations [12-14]. Established R packages provide widely used reference and application workflows, while accelerated approaches use iterative or randomized low-rank solvers, implicit products, compiled linear algebra, or accelerator execution [15,16]. The IKPLS software implements Improved Kernel PLS with NumPy and JAX [33-35]; because it is not de Jong SIMPLS, we evaluate it as a separate end-to-end software comparator rather than as evidence of estimator equivalence.",
    "We present fastPLS, whose principal methodological contribution is a compiled, shape-dependent execution of de Jong SIMPLS. Sequential deflation, orthogonalization, and component definitions are retained, while avoidable intermediate products and dense outputs are reduced according to matrix shape and requested output. The R interface also provides PLS-SVD, OPLS, kernel PLS, compact latent prediction, optional implicit predictor-response products, compiled single and nested validation, and explicit solver and backend diagnostics. Low-rank solvers, float32 execution, linear discriminant analysis, and CPU, NVIDIA CUDA, and Apple Metal routes support the central SIMPLS implementation. The benchmark therefore separates numerical validation against pls::simpls.fit, comparison with available R PLS workflows, and a distinct cross-language comparison with IKPLS. The GPL-3 R package uses reusable components from the MIT-licensed kodama-cpp project.",
]


DISCUSSION_OLD = [
    "The computational results support a shape-dependent choice rather than one universally preferred PLS family. On one CPU, PLS-SVD was as fast as or faster than SIMPLS across the five matched synthetic shapes (SIMPLS/PLS-SVD time ratio 1.00-3.84), as expected for a one-shot decomposition. On the qualified CUDA shapes, SIMPLS approached or marginally exceeded PLS-SVD speed (ratio 0.92-0.98), while retaining sequential orthogonalization and support for component counts not restricted by response rank. Family selection should nevertheless be based on training-only predictive validation, because PLS-SVD and SIMPLS are different estimators. Compact prediction matters most when the test set, response dimension, or number of requested prefixes is large: it reduced incremental RSS by up to 77.7% by avoiding dense coefficient and fitted-response paths, but offered little benefit when those outputs were intrinsically small.",
    "The frozen IKPLS comparison places the contribution in the high-performance PLS landscape. Improved Kernel PLS was substantially faster in the tested single-thread CPU workflows, whereas fastPLS offered an R-native de Jong SIMPLS path, multivariate-response storage controls, nested validation, multiple PLS families, and route diagnostics. This does not establish universal superiority of either software. The cross-language result is interpreted as end-to-end workflow evidence, separate from deterministic estimator validation.",
]


DISCUSSION_NEW = (
    "The computational results support a shape-dependent choice rather than one universally preferred PLS family or implementation. On one CPU, PLS-SVD was as fast as or faster than SIMPLS across the five matched synthetic shapes (SIMPLS/PLS-SVD time ratio 1.00-3.84), as expected for a one-shot decomposition. On the qualified CUDA shapes, SIMPLS approached or marginally exceeded PLS-SVD speed (ratio 0.92-0.98), while retaining sequential orthogonalization and support for component counts not restricted by response rank. Family selection should nevertheless be based on training-only predictive validation, because PLS-SVD and SIMPLS are different estimators. Compact prediction matters most when the test set, response dimension, or number of requested prefixes is large: it reduced incremental RSS by up to 77.7% by avoiding dense coefficient and fitted-response paths, but offered little benefit when those outputs were intrinsically small. In the broader R-package panel, fastPLS had the lowest observed total time on seven of nine classification datasets, although the matched minimal-output comparison with pls::simpls.fit showed smaller, dataset-dependent differences. IKPLS was faster in the separate single-thread cross-language experiment, emphasizing that fastPLS contributes an R-native de Jong SIMPLS workflow, multivariate-response storage controls, nested validation, multiple PLS families, and route diagnostics rather than universal superiority over every PLS implementation."
)


def find_exact(document: Document, text: str):
    matches = [p for p in document.paragraphs if p.text == text]
    if len(matches) != 1:
        raise RuntimeError(f"Expected one exact match, found {len(matches)} for: {text[:80]}")
    return matches[0]


def delete_paragraph(paragraph) -> None:
    element = paragraph._element
    element.getparent().remove(element)
    paragraph._p = paragraph._element = None


def main() -> None:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    document = Document(SOURCE)

    intro_paragraphs = [find_exact(document, text) for text in INTRO_OLD]
    for paragraph, replacement in zip(intro_paragraphs[:5], INTRO_NEW):
        paragraph.text = replacement
        paragraph.style = document.styles["First Paragraph"] if paragraph is intro_paragraphs[0] else document.styles["Body Text"]
    for paragraph in intro_paragraphs[5:]:
        delete_paragraph(paragraph)

    discussion_paragraphs = [find_exact(document, text) for text in DISCUSSION_OLD]
    discussion_paragraphs[0].text = DISCUSSION_NEW
    discussion_paragraphs[0].style = document.styles["First Paragraph"]
    delete_paragraph(discussion_paragraphs[1])

    find_exact(document, "2.7 High-performance cross-language comparison").text = (
        "2.7 Cross-language software comparison"
    )
    find_exact(document, "3.2 Cross-language high-performance PLS comparison").text = (
        "3.2 Cross-language software comparison"
    )

    document.save(OUTPUT)


if __name__ == "__main__":
    main()
