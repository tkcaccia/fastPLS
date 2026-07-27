#!/usr/bin/env python3
"""Justify rSVD approximation screens and state their scientific limits."""

from pathlib import Path

from docx import Document

from revise_cmpb_cycle67_consolidate_evidence import (
    normalize_submission_terminology,
    replace_paragraph,
)


ROOT = Path(__file__).resolve().parents[1]
MAIN_SOURCE = (
    ROOT
    / "artifacts"
    / "CMPB_rewrite_20260726_cycle71"
    / "fastPLS_CMPB_main_cycle71_0.99.6_20260726.docx"
)
SUPP_SOURCE = (
    ROOT
    / "artifacts"
    / "CMPB_rewrite_20260726_cycle68"
    / "fastPLS_CMPB_supplement_cycle68_0.99.6_20260726.docx"
)
OUTPUT_DIR = ROOT / "artifacts" / "CMPB_rewrite_20260726_cycle72"
MAIN_OUTPUT = OUTPUT_DIR / "fastPLS_CMPB_main_cycle72_0.99.6_20260726.docx"
SUPP_OUTPUT = (
    OUTPUT_DIR / "fastPLS_CMPB_supplement_cycle72_0.99.6_20260726.docx"
)


def revise_main(document):
    replace_paragraph(
        document,
        "The primary evidence tested whether deterministic IRLBA SIMPLS",
        (
            "The primary evidence tested whether deterministic IRLBA SIMPLS "
            "preserved de Jong's estimator and improved runtime. rSVD was assessed "
            "separately using prespecified computational non-inferiority screens: "
            "relative Frobenius prediction error at most 0.05, prediction "
            "correlation at least 0.99, each score/projection/loading subspace angle "
            "at most 10 degrees, decoded-label agreement at least 0.99, and absolute "
            "predictive-metric difference at most 0.01. These limits are engineering "
            "guardrails, not clinical-equivalence margins. The norm-based criterion "
            "was required jointly with an outcome-scale metric; for classification, "
            "0.99 agreement limits changed decisions to 1% and, because counts are "
            "discrete, permits no discordance when n < 100 and at most one when "
            "n = 100. Exact discordant counts and metric changes were therefore "
            "examined alongside pass/fail status. Deterministic IRLBA remains the "
            "confirmatory route for small cohorts or decisions in which one changed "
            "case is consequential. Supporting designs and provenance are in "
            "Supplementary Tables S3-S15."
        ),
    )
    replace_paragraph(
        document,
        "Deterministic fastPLS SIMPLS met all prespecified tolerances",
        (
            "Deterministic fastPLS SIMPLS met all prespecified tolerances in 117 "
            "component-level comparisons with de Jong SIMPLS. In the separate rSVD "
            "audit, oversampling 10 with one power iteration passed 101/117 checks; "
            "oversampling 10 with two power iterations passed 117/117 across seeds "
            "101, 202, 303, and 123 for the real datasets. In the qualified "
            "two-power analysis, the worst relative prediction error was 0.0332 "
            "(MetRef, 18 components) with identical decoded labels and accuracy. "
            "Minimum label agreement was 0.99 in three MetRef component settings, "
            "each representing exactly one discordant prediction among 100; the "
            "associated accuracy difference was zero or 0.01. The largest regression "
            "metric difference was 0.000693. These observations satisfy the "
            "computational screens but do not establish scientific equivalence for "
            "an individual case. OPLS and kernel PLS extensions passed all 66 "
            "setting/task comparisons and all 1,540 fold-component fits under "
            "deterministic float64 CPU settings (Supplementary Tables S7-S8)."
        ),
    )
    replace_paragraph(
        document,
        "rSVD, implicit products, float32, CUDA, and Metal are optional",
        (
            "rSVD, implicit products, float32, CUDA, and Metal are optional "
            "implementation mechanisms around accelerated SIMPLS. Only oversampling "
            "10 with two power iterations passed all 117 broad-audit checks. Passing "
            "denotes numerical non-inferiority within the stated prediction, "
            "subspace, label, and metric guardrails; it does not imply estimator or "
            "clinical equivalence. This distinction matters in small biomedical "
            "test sets, where one discordant label can represent at least 1% of the "
            "sample. The faster one-power NMR and ImageNet workflows passed 101/117 "
            "at setting level or were not separately qualified; they demonstrate "
            "feasibility but cannot support confirmatory claims. CPU IRLBA remains "
            "the deterministic reference."
        ),
    )


def revise_supplement(document):
    replace_paragraph(
        document,
        "rSVD uses a fixed seed, Gaussian range sketch",
        (
            "rSVD uses a fixed seed, Gaussian range sketch, oversampling, and power "
            "iterations. Qualification required all of the following: relative "
            "Frobenius prediction error <= 0.05, prediction correlation >= 0.99, "
            "each score/projection/loading subspace angle <= 10 degrees, decoded-"
            "label agreement >= 0.99, and absolute predictive-metric difference "
            "<= 0.01. These prespecified values were computational non-inferiority "
            "screens rather than clinical or estimator-equivalence margins. The "
            "relative error is scale normalized but can conceal localized errors, "
            "so it was never interpreted without the endpoint-specific metric and, "
            "for classification, exact discordant counts. An agreement threshold of "
            "0.99 permits no changed prediction for test sets smaller than 100 and "
            "at most one for n = 100. In the qualified CPU two-power analysis, the "
            "maximum relative error of 0.0332 occurred on MetRef at 18 components "
            "with zero label or accuracy change; the three 0.99-agreement endpoints "
            "each contained one discordant MetRef prediction among 100 and changed "
            "accuracy by zero or 0.01. The largest regression metric difference was "
            "0.000693. Thus, qualification supports approximate computational use, "
            "not a claim that individual predictions are interchangeable. "
            "Deterministic IRLBA is recommended for confirmatory small-cohort or "
            "case-level inference."
        ),
    )


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    main_document = Document(MAIN_SOURCE)
    revise_main(main_document)
    normalize_submission_terminology(main_document)
    main_document.save(MAIN_OUTPUT)

    supplement_document = Document(SUPP_SOURCE)
    revise_supplement(supplement_document)
    normalize_submission_terminology(supplement_document)
    supplement_document.save(SUPP_OUTPUT)

    print(MAIN_OUTPUT)
    print(SUPP_OUTPUT)


if __name__ == "__main__":
    main()
