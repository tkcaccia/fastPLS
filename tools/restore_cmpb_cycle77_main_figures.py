#!/usr/bin/env python3
"""Restore the five agreed main-text figure anchors lost after cycle 67."""

from copy import deepcopy
from pathlib import Path
import shutil
from zipfile import ZIP_DEFLATED, ZipFile

from lxml import etree


ROOT = Path(__file__).resolve().parents[1]
FIGURE_SOURCE = (
    ROOT
    / "artifacts"
    / "CMPB_rewrite_20260726_cycle67"
    / "fastPLS_CMPB_main_cycle67_0.99.6_20260726.docx"
)
SOURCE_DIR = ROOT / "artifacts" / "CMPB_rewrite_20260726_cycle76"
OUTPUT_DIR = ROOT / "artifacts" / "CMPB_rewrite_20260726_cycle77"
MAIN_SOURCE = SOURCE_DIR / "fastPLS_CMPB_main_cycle76_0.99.6_20260726.docx"
SUPP_SOURCE = (
    SOURCE_DIR / "fastPLS_CMPB_supplement_cycle76_0.99.6_20260726.docx"
)
MAIN_OUTPUT = OUTPUT_DIR / "fastPLS_CMPB_main_cycle77_0.99.6_20260726.docx"
SUPP_OUTPUT = (
    OUTPUT_DIR / "fastPLS_CMPB_supplement_cycle77_0.99.6_20260726.docx"
)


W_NS = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
NS = {"w": W_NS}


def paragraph_text(paragraph):
    return "".join(paragraph.xpath(".//w:t/text()", namespaces=NS)).strip()


def extract_figure_paragraphs(document_xml):
    root = etree.fromstring(document_xml)
    figures = {}
    paragraphs = root.xpath(".//w:body/w:p", namespaces=NS)
    for index, paragraph in enumerate(paragraphs):
        if not paragraph.xpath(".//w:drawing", namespaces=NS):
            continue
        for following in paragraphs[index + 1 :]:
            text = paragraph_text(following)
            if not text:
                continue
            for number in range(1, 6):
                if text.startswith(f"Figure {number}."):
                    figures[number] = deepcopy(paragraph)
            break
    if set(figures) != set(range(1, 6)):
        raise RuntimeError(f"Expected five source figures, found {sorted(figures)}")
    return figures


def restore_figure_paragraphs(document_xml, figures):
    root = etree.fromstring(document_xml)
    paragraphs = root.xpath(".//w:body/w:p", namespaces=NS)
    restored = set()
    for paragraph in paragraphs:
        text = paragraph_text(paragraph)
        for number in range(1, 6):
            if text.startswith(f"Figure {number}."):
                paragraph.addprevious(deepcopy(figures[number]))
                restored.add(number)
                break
    if restored != set(range(1, 6)):
        raise RuntimeError(f"Expected five target captions, found {sorted(restored)}")
    return etree.tostring(
        root,
        xml_declaration=True,
        encoding="UTF-8",
        standalone="yes",
    )


def write_docx_with_document_xml(source, destination, document_xml):
    with ZipFile(source, "r") as zin, ZipFile(
        destination, "w", compression=ZIP_DEFLATED
    ) as zout:
        for item in zin.infolist():
            data = (
                document_xml
                if item.filename == "word/document.xml"
                else zin.read(item.filename)
            )
            zout.writestr(item, data)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with ZipFile(FIGURE_SOURCE) as archive:
        figures = extract_figure_paragraphs(archive.read("word/document.xml"))
    with ZipFile(MAIN_SOURCE) as archive:
        restored_xml = restore_figure_paragraphs(
            archive.read("word/document.xml"), figures
        )
    write_docx_with_document_xml(MAIN_SOURCE, MAIN_OUTPUT, restored_xml)
    shutil.copy2(SUPP_SOURCE, SUPP_OUTPUT)
    print(MAIN_OUTPUT)
    print(SUPP_OUTPUT)


if __name__ == "__main__":
    main()
