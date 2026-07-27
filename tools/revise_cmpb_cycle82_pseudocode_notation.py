#!/usr/bin/env python3
"""Make Algorithm S1 component-prefix notation match the notation table."""

from pathlib import Path
from shutil import copy2

from docx import Document


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "artifacts" / "CMPB_rewrite_20260727_cycle81"
OUTPUT = ROOT / "artifacts" / "CMPB_rewrite_20260727_cycle82"
MAIN_IN = SOURCE / "fastPLS_CMPB_main_cycle81_0.99.6_20260727.docx"
SUPP_IN = SOURCE / "fastPLS_CMPB_supplement_cycle81_0.99.6_20260727.docx"
MAIN_OUT = OUTPUT / "fastPLS_CMPB_main_cycle82_0.99.6_20260727.docx"
SUPP_OUT = OUTPUT / "fastPLS_CMPB_supplement_cycle82_0.99.6_20260727.docx"


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    copy2(MAIN_IN, MAIN_OUT)

    document = Document(SUPP_IN)
    matches = [
        paragraph
        for paragraph in document.paragraphs
        if paragraph.text.startswith(
            "Form the compact latent prediction factors from prefixes"
        )
    ]
    if len(matches) != 1:
        raise RuntimeError(
            f"Expected one Algorithm S1 prefix line, found {len(matches)}"
        )
    matches[0].text = (
        "For each requested a \u2208 C, form the compact latent prediction "
        "factors from prefixes 1,...,a."
    )
    document.save(SUPP_OUT)
    print(MAIN_OUT)
    print(SUPP_OUT)


if __name__ == "__main__":
    main()
