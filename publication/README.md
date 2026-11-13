# Publication Build (`publication/`)

This folder holds the scripts, Typst template, and vector figures used to typeset `../CEG-JOURNAL-MANUSCRIPT-V3.2.md` into the 36-page PDF (`CEG-JOURNAL-MANUSCRIPT-V3.2.pdf` and `../context-evidence-graph.pdf`).

## Contents

| File | Purpose |
| :--- | :--- |
| `CEG-JOURNAL-MANUSCRIPT-V3.2.pdf` | Typeset article (36 pages, A4). Identical to `../CEG-JOURNAL-MANUSCRIPT-V3.2.pdf` and `../context-evidence-graph.pdf`. |
| `CEG-JOURNAL-MANUSCRIPT-V3.2.typ` | Generated Typst source. Re-run `build.py` rather than editing by hand. |
| `build.py` | Converts the Markdown manuscript with `pandoc`, applies layout rules, and compiles with `typst` (Python standard library only). |
| `template.typ` | Page geometry, typography, table rules, and figure styling. |
| `make_figures.py` | Generates Figures 1 to 3 as vector graphics (`matplotlib`). |
| `figures/` | Figures 1 to 3 in PDF (embedded in the article), SVG, and PNG preview formats. |
| `audit_pdf.py` | Automated layout, font, character, and metadata checks on the compiled PDF (`pymupdf`). |
| `measure_record_size.py` | Measures the canonical JSON byte size of the 31 `EvidenceAtom` records in the 12-study corpus (§7.3). |
| `record_size.json` | Stored output of `measure_record_size.py`. |
| `SHA256SUMS` | SHA-256 digests of the files in this folder. |

## Rebuild Commands

Run from this folder:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -B make_figures.py
PYTHONDONTWRITEBYTECODE=1 python3 -B build.py
PYTHONDONTWRITEBYTECODE=1 python3 -B audit_pdf.py
PYTHONDONTWRITEBYTECODE=1 python3 -B measure_record_size.py
PYTHONDONTWRITEBYTECODE=1 python3 -B measure_record_size.py --check ../CEG-JOURNAL-MANUSCRIPT-V3.2.md
```

Toolchain used for the delivered PDF: `pandoc` 3.13, `typst` 0.15.1 (`--ignore-system-fonts`), Python 3.11.18, `matplotlib` 3.11.2, and `pymupdf` 2.1. Bundled fonts only (`New Computer Modern`, `New Computer Modern Math`, and `DejaVu Sans` / `DejaVu Sans Mono`).

## Automated PDF Checks (`audit_pdf.py`)

1. **Fonts:** Only bundled text and figure fonts are embedded.
2. **Geometry:** Zero text lines fall outside the type area across all 36 pages.
3. **Characters:** Zero replacement, soft-hyphen, or zero-width characters.
4. **Content & symbols:** Accented author names and mathematical symbols render cleanly.
5. **Number parity:** Digit runs in the PDF body text match the Markdown source exactly.
6. **Word parity:** Every prose word of the Markdown source appears at least as often in the PDF body text.
7. **Metadata:** Title and author (`agent-kev`) set cleanly with zero local filesystem paths.
