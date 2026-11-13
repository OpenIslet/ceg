# Context–Evidence Graph (CEG)

An open method, reference Python verifier, benchmark suite and worked examples for reading and synthesising scientific studies with large language models without losing experimental context.

- **Plain-English overview:** https://openislet.org/ceg
- **Typeset paper (PDF):** [`context-evidence-graph.pdf`](context-evidence-graph.pdf) (`CEG-JOURNAL-MANUSCRIPT-V3.2.pdf`, 36 pages)
- **Practical guide for any field:** [`HOW-TO-USE-CEG.md`](HOW-TO-USE-CEG.md)

---

## Why a Context–Evidence Graph

Studies on the same question often reach different answers because the experimental context differed: cells in a dish versus a living graft, mouse versus human tissue, resting versus inflamed conditions, or different doses and endpoints. Standard retrieval and summarisation pipelines strip those boundary conditions away when compressing papers into short snippets.

A **Context–Evidence Graph (CEG)** keeps every measured result attached to its experimental and control groups, measurement units, partially ordered experimental setting and exact source pointer, while pairing **free exploration in an AI context window** with **deterministic verification before saving to a shared graph**.

---

## Quickstart (`Python 3.9+`, Standard Library Only)

No external Python packages are required to run the reference verifier, the worked examples, the synthetic benchmark or the PubMed primary-source evaluation.

```bash
git clone https://github.com/openislet/ceg.git
cd ceg

# Run the synthetic worked examples (antifibrotic drug development & sport science)
PYTHONDONTWRITEBYTECODE=1 python3 -B domain_vignettes.py --domain both

# Run the real-world type 1 diabetes worked example (8 studies, 10 findings)
PYTHONDONTWRITEBYTECODE=1 python3 -B domain_vignettes.py --domain t1d_islet_replacement

# Run the 1,931-case synthetic benchmark across 5 suites and 5 ablation regimes
PYTHONDONTWRITEBYTECODE=1 python3 -B ceg_synthetic_benchmark.py

# Run the 12-study PubMed evaluation (IPF, heart failure & thoracic oncology)
PYTHONDONTWRITEBYTECODE=1 python3 -B ceg_real_text_evaluation.py
```

Verify file integrity against the published SHA-256 digests:

```bash
shasum -a 256 -c SHA256SUMS
```

---

## Repository Contents

| Path | Description |
| :--- | :--- |
| [`context-evidence-graph.pdf`](context-evidence-graph.pdf) | Typeset manuscript PDF (`36` pages; identical to `CEG-JOURNAL-MANUSCRIPT-V3.2.pdf`). |
| [`CEG-JOURNAL-MANUSCRIPT-V3.2.md`](CEG-JOURNAL-MANUSCRIPT-V3.2.md) | Markdown source (`v3.2`) of *Context–Evidence Graph (CEG): In-Context Multi-Study Scientific Synthesis with Large Language Models*. |
| [`HOW-TO-USE-CEG.md`](HOW-TO-USE-CEG.md) | Step-by-step practical guide, minimal JSON template and AI prompt patterns for building or querying a Context–Evidence Graph in any field. |
| [`ceg_reference.py`](ceg_reference.py) | Zero-dependency Python 3 reference verifier (`v3.2`): invariants `I1`–`I6`, bijective `JSONPatch` ledger replay, partial-order context transfer classifier, Unicode-screen / SI-case / negated-bound prose lint, 7-coordinate typed layer, non-prose surface locks and two-layer package verifier. |
| [`domain_vignettes.py`](domain_vignettes.py) | Three worked examples: translational antifibrotic drug development (`NX-410`), exercise physiology (`sport_science`), and type 1 diabetes islet replacement (`t1d_islet_replacement`, built from peer-reviewed studies in the OpenIslet corpus). |
| [`ceg_synthetic_benchmark.py`](ceg_synthetic_benchmark.py) & [`benchmark_results.json`](benchmark_results.json) | `1,931`-case synthetic benchmark across five test suites (`core`, `stress`, `residual`, `redteam`, `gauntlet`) and five ablation regimes (`V0`–`V3`). |
| [`ceg_real_text_evaluation.py`](ceg_real_text_evaluation.py) & [`real_text_results.json`](real_text_results.json) | Primary-source evaluation across 12 peer-reviewed PubMed studies (`31` `EvidenceAtom`s, `120` Track A/B items) in idiopathic pulmonary fibrosis, heart failure and thoracic oncology. |
| [`audit/BIBLIOGRAPHY-PRIMARY-LEDGER.json`](audit/BIBLIOGRAPHY-PRIMARY-LEDGER.json) | Primary-source verification ledger (DOIs, PMIDs and standards URLs) for the bibliography. |
| [`t1d/`](t1d/) | Verification, corpus-measurement and ledger-replay scripts and stored outputs behind the type 1 diabetes worked example (§5) and the 1,151-study production corpus (§6.9). |
| [`publication/`](publication/) | Reproducible figure generator (`make_figures.py`), Pandoc + Typst PDF build (`build.py`), record-size measurement (`measure_record_size.py`), PDF geometry audit (`audit_pdf.py`) and vector figures. |
| [`SHA256SUMS`](SHA256SUMS) | SHA-256 checksums for every file in the repository. |

---

## Citation

> `agent-kev` (2026). *Context–Evidence Graph (CEG): In-Context Multi-Study Scientific Synthesis with Large Language Models*. OpenIslet Project. https://openislet.org/ceg · https://github.com/openislet/ceg

---

## Licences

- **Code and scripts (`*.py`, `*.typ`):** [Apache License 2.0](LICENSE)
- **Manuscript, guide, figures and structured data (`*.md`, `*.pdf`, `*.json`, `figures/*`):** [Creative Commons Attribution 4.0 International (`CC BY 4.0`)](LICENSE-DATA) (short verbatim excerpts from cited papers remain the property of their respective authors and publishers)
