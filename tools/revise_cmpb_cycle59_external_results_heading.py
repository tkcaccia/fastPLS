#!/usr/bin/env python3

from pathlib import Path
import shutil

from docx import Document


ROOT = Path("/Users/stefano/Documents/GPUPLS/fastPLS_fresh_github")
SOURCE_DIR = ROOT / "artifacts" / "CMPB_rewrite_20260726_cycle58"
OUTPUT_DIR = ROOT / "artifacts" / "CMPB_rewrite_20260726_cycle59"
MAIN_SOURCE = (
    SOURCE_DIR / "fastPLS_CMPB_main_cycle58_0.99.6_20260726.docx"
)
SUPP_SOURCE = (
    SOURCE_DIR / "fastPLS_CMPB_supplement_cycle58_0.99.6_20260726.docx"
)
SUPP_SOURCE_PDF = (
    SOURCE_DIR / "fastPLS_CMPB_supplement_cycle58_0.99.6_20260726.pdf"
)
MAIN_OUTPUT = (
    OUTPUT_DIR / "fastPLS_CMPB_main_cycle59_0.99.6_20260726.docx"
)
SUPP_OUTPUT = (
    OUTPUT_DIR / "fastPLS_CMPB_supplement_cycle59_0.99.6_20260726.docx"
)
SUPP_OUTPUT_PDF = (
    OUTPUT_DIR / "fastPLS_CMPB_supplement_cycle59_0.99.6_20260726.pdf"
)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    document = Document(MAIN_SOURCE)
    heading = next(
        paragraph
        for paragraph in document.paragraphs
        if paragraph.text.strip()
        == "3.1 Single-CPU comparison with independent R implementations"
    )
    heading.text = "3.1 Comparison with independent R implementations"
    heading.style = document.styles["Heading 2"]
    document.save(MAIN_OUTPUT)

    shutil.copy2(SUPP_SOURCE, SUPP_OUTPUT)
    shutil.copy2(SUPP_SOURCE_PDF, SUPP_OUTPUT_PDF)
    print(MAIN_OUTPUT)
    print(SUPP_OUTPUT)


if __name__ == "__main__":
    main()
