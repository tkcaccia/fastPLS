#!/usr/bin/env python3

from pathlib import Path
import shutil

from docx import Document


ROOT = Path("/Users/stefano/Documents/GPUPLS/fastPLS_fresh_github")
SOURCE = ROOT / "artifacts" / "CMPB_rewrite_20260725_cycle27"
OUT = ROOT / "artifacts" / "CMPB_rewrite_20260725_cycle28"

MAIN_SOURCE = SOURCE / "fastPLS_CMPB_main_cycle27_0.99.6_20260725.docx"
SUPP_SOURCE = SOURCE / "fastPLS_CMPB_supplement_cycle27_0.99.6_20260725.docx"
RESPONSE_SOURCE = (
    SOURCE / "fastPLS_CMPB_response_to_reviewers_cycle27_20260725.docx"
)

MAIN_OUT = OUT / "fastPLS_CMPB_main_cycle28_0.99.6_20260725.docx"
SUPP_OUT = OUT / "fastPLS_CMPB_supplement_cycle28_0.99.6_20260725.docx"
RESPONSE_OUT = OUT / "fastPLS_CMPB_response_to_reviewers_cycle28_20260725.docx"


def find_paragraph(document, prefix):
    for paragraph in document.paragraphs:
        if paragraph.text.strip().startswith(prefix):
            return paragraph
    raise RuntimeError(f"Paragraph not found: {prefix}")


def main():
    OUT.mkdir(parents=True, exist_ok=True)

    document = Document(MAIN_SOURCE)
    paragraph = find_paragraph(
        document,
        "Table 1 and Figure 2 show both matched CPU and CUDA backends",
    )
    marker = "these values are not claimed as global optima. "
    correction = (
        "Retina and Tabula Muris, the two single-cell resources, are reported "
        "separately under their specific names. "
    )
    if correction not in paragraph.text:
        if marker not in paragraph.text:
            raise RuntimeError("Insertion point not found in Results paragraph")
        paragraph.text = paragraph.text.replace(marker, marker + correction, 1)

    document.core_properties.title = (
        "fastPLS CMPB manuscript - corrected single-cell wording"
    )
    document.save(MAIN_OUT)

    shutil.copy2(SUPP_SOURCE, SUPP_OUT)
    shutil.copy2(RESPONSE_SOURCE, RESPONSE_OUT)


if __name__ == "__main__":
    main()
