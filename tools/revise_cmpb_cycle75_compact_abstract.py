#!/usr/bin/env python3
"""Condense the structured abstract around the primary SIMPLS contribution."""

from pathlib import Path
import shutil

from docx import Document


ROOT = Path(__file__).resolve().parents[1]
SOURCE_DIR = ROOT / "artifacts" / "CMPB_rewrite_20260726_cycle74"
OUTPUT_DIR = ROOT / "artifacts" / "CMPB_rewrite_20260726_cycle75"
MAIN_SOURCE = (
    SOURCE_DIR / "fastPLS_CMPB_main_cycle74_0.99.6_20260726.docx"
)
SUPP_SOURCE = (
    SOURCE_DIR / "fastPLS_CMPB_supplement_cycle74_0.99.6_20260726.docx"
)
MAIN_OUTPUT = (
    OUTPUT_DIR / "fastPLS_CMPB_main_cycle75_0.99.6_20260726.docx"
)
SUPP_OUTPUT = (
    OUTPUT_DIR / "fastPLS_CMPB_supplement_cycle75_0.99.6_20260726.docx"
)


ABSTRACT = {
    "Background and objective:": (
        "Background and objective: High-dimensional biomedical PLS workflows, "
        "including multivariate NMR prediction and repeated validation, can be "
        "limited by the sequential cost and storage of SIMPLS. We developed "
        "fastPLS to accelerate SIMPLS while retaining de Jong's component equations."
    ),
    "Methods:": (
        "Methods: The implementation reuses deflation products, latent quantities, "
        "coefficients, and predictions along one maximal component path, with compact "
        "prediction and optional implicit cross-covariance products. Deterministic "
        "float64 CPU IRLBA was used for estimator-matched validation against de Jong "
        "SIMPLS and independent R software. Approximate rSVD and accelerator routes "
        "were evaluated separately; the qualified rSVD setting used oversampling 10 "
        "and two power iterations."
    ),
    "Results:": (
        "Results: fastPLS SIMPLS met the prespecified numerical tolerances in all 117 "
        "deterministic component-level comparisons. In matched single-CPU comparisons, "
        "it was faster than pls::simpls.fit on seven of nine datasets, with identical "
        "accuracy and speed-up up to 8.90-fold. The two-power rSVD setting also met the "
        "prespecified tolerances in all 117 checks. Exploratory one-power rSVD stress "
        "tests gave CUDA SIMPLS an NMR RMSD of 0.000759 with a 5.94-fold CUDA/CPU "
        "speed-up; CUDA SIMPLS-LDA processed 1,000,000 training and 281,167 held-out "
        "ImageNet/DINOv2 embeddings, reaching top-1 accuracy 0.8093 at 1,000 components. "
        "These stress tests demonstrate workflow feasibility rather than confirmatory "
        "equivalence."
    ),
    "Conclusions:": (
        "Conclusions: fastPLS reduces the computational barrier to sequential SIMPLS "
        "in large biomedical workflows while making solver and backend qualification "
        "explicit."
    ),
}


def revise_abstract(document):
    replaced = set()
    for paragraph in document.paragraphs:
        for prefix, replacement in ABSTRACT.items():
            if paragraph.text.startswith(prefix):
                paragraph.text = replacement
                replaced.add(prefix)
                break
    missing = set(ABSTRACT) - replaced
    if missing:
        raise RuntimeError(f"Abstract paragraphs not found: {sorted(missing)}")


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    document = Document(MAIN_SOURCE)
    revise_abstract(document)
    document.save(MAIN_OUTPUT)
    shutil.copy2(SUPP_SOURCE, SUPP_OUTPUT)

    abstract_words = sum(len(text.split()) for text in ABSTRACT.values())
    print(f"abstract_words={abstract_words}")
    print(MAIN_OUTPUT)
    print(SUPP_OUTPUT)


if __name__ == "__main__":
    main()
