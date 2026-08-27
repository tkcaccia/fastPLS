from pathlib import Path

from docx import Document


ROOT = Path("/Users/stefano/Documents/GPUPLS/fastPLS_fresh_github")
SOURCE = ROOT / "artifacts/CMPB_rewrite_20260826_cycle110/fastPLS_CMPB_main_cycle110_0.99.25_20260826.docx"
OUTDIR = ROOT / "artifacts/CMPB_rewrite_20260826_cycle111"
OUTPUT = OUTDIR / "fastPLS_CMPB_main_cycle111_0.99.25_20260826.docx"


OLD = (
    "The broader benchmark design covered biomedical and computational tasks spanning "
    "metabolomics, NMR, CITE-seq, tissue and cancer omics, single-cell transcriptomics, "
    "drug response, and image embeddings [7,20-30]. Archived central evidence comprises "
    "deterministic SIMPLS validation, external comparison, controlled scaling and solver "
    "qualification, selected CPU/CUDA routes, and the NMR case study. Methods used "
    "identical stored splits and training-only component grids within each analysis. "
    "Cross-dataset selected points are treated as fixed computational workloads when "
    "their grids were boundary or rank constrained. Runtime included fitting and "
    "prediction; memory definitions, split units, endpoint definitions, and uncertainty "
    "limitations are detailed in the Supplement."
)


NEW = (
    "The broader benchmark design covered biomedical and computational tasks spanning "
    "metabolomics, NMR, CITE-seq, tissue and cancer omics, single-cell transcriptomics, "
    "drug response, and image embeddings [7,20-30]. Dataset provenance, acquisition or "
    "access requirements, construction of the prepared analysis matrices, preprocessing, "
    "response encoding, dimensions, split units and seeds, and component grids are "
    "documented for every dataset in Supplementary Section S5 and Table S3. "
    "Redistribution restrictions and executable acquisition instructions are provided in "
    "Supplementary Section S5.6 and the repository file benchmark/DATA_ACQUISITION.md. "
    "Archived central evidence comprises deterministic SIMPLS validation, external "
    "comparison, controlled scaling and solver qualification, selected CPU/CUDA routes, "
    "and the NMR case study. Methods used identical stored splits and training-only "
    "component grids within each analysis. Cross-dataset selected points are treated as "
    "fixed computational workloads when their grids were boundary or rank constrained. "
    "Runtime included fitting and prediction; memory definitions, endpoint definitions, "
    "and uncertainty limitations are detailed in Supplementary Sections S4 and S7."
)


def main() -> None:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    document = Document(SOURCE)
    matches = [paragraph for paragraph in document.paragraphs if paragraph.text == OLD]
    if len(matches) != 1:
        raise RuntimeError(f"Expected one benchmark-design paragraph, found {len(matches)}")
    matches[0].text = NEW
    document.save(OUTPUT)


if __name__ == "__main__":
    main()
