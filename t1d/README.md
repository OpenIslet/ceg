# Type 1 Diabetes Worked Example & Snapshot Measurements (`t1d/`)

This folder holds the verification scripts and stored JSON outputs behind Section 5 (Table 1, the 8-study / 10-finding type 1 diabetes worked example) and Section 6.9 (Table 10, measurements of the 1,151-study OpenIslet corpus, 12 November 2026 snapshot) of `../CEG-JOURNAL-MANUSCRIPT-V3.2.md`.

All scripts use only the Python 3 standard library and import `../ceg_reference.py` and `../domain_vignettes.py` unmodified.

A public projection of the settled 1,151-study type 1 diabetes Context–Evidence Graph is available under `CC-BY-4.0` in the OpenIslet repository (`https://github.com/openislet/openislet`; `https://openislet.org/blueprint`), with verbatim quotes replaced by SHA-256 digests (`quote_sha256`) and source-overlapping prose replaced by structured descriptors. The pre-export staging masters and curation ledger read by `--recompute` are pinned by SHA-256 digest in `t1d_corpus_metrics.json` and `production_ledger_replay.json` and are available as a read-only archive on request from `kev@openislet.org`.

## Contents

| File | Purpose |
| :--- | :--- |
| `check_t1d_vignette.py` | Rebuilds the type 1 diabetes worked example (`domain_vignettes.build_t1d_vignette`), checks stored verification outputs, prints Table 1 (`--table`), compares against an OpenIslet snapshot (`--recompute`), and locates quoted sentences in NCBI text (`--quotes`). |
| `t1d_vignette_verification.json` | Stored verification output: context lattice size, transfer-class counts, per-finding transfer classes and unresolved axes, decisions, violation codes, and digests. |
| `t1d_vignette_quote_check.json` | Stored NCBI quote-check output confirming where each of the 10 quoted sentences appears in PubMed Central full text (9) or PubMed abstract (1). |
| `t1d_corpus_metrics.py` | Recomputes the Section 6.9 and Table 10 corpus metrics from the pinned OpenIslet snapshot files. |
| `t1d_corpus_metrics.json` | Stored corpus counts, canonical RFC 8785 byte sizes, and SHA-256 file pins. |
| `replay_production_ledger.py` | Replays the 106-event curation ledger forward and in reverse through `ceg_reference.verify_bijective_patch_replay` and runs tamper controls. |
| `production_ledger_replay.json` | Stored ledger replay and tamper-control output. |
| `SHA256SUMS` | SHA-256 digests of the files in this folder. |

## Commands

Run the worked example, print Table 1, and verify manuscript parity (offline, no external files needed):

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -B ../domain_vignettes.py --domain t1d_islet_replacement
PYTHONDONTWRITEBYTECODE=1 python3 -B check_t1d_vignette.py --table
PYTHONDONTWRITEBYTECODE=1 python3 -B check_t1d_vignette.py --check ../CEG-JOURNAL-MANUSCRIPT-V3.2.md
PYTHONDONTWRITEBYTECODE=1 python3 -B t1d_corpus_metrics.py --check ../CEG-JOURNAL-MANUSCRIPT-V3.2.md
PYTHONDONTWRITEBYTECODE=1 python3 -B replay_production_ledger.py --check ../CEG-JOURNAL-MANUSCRIPT-V3.2.md
```

Recompute against a read-only copy of the pinned OpenIslet snapshot archive (`SNAPSHOT_DIR`):

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -B check_t1d_vignette.py --recompute SNAPSHOT_DIR
PYTHONDONTWRITEBYTECODE=1 python3 -B check_t1d_vignette.py --quotes SNAPSHOT_DIR
PYTHONDONTWRITEBYTECODE=1 python3 -B t1d_corpus_metrics.py --recompute SNAPSHOT_DIR
PYTHONDONTWRITEBYTECODE=1 python3 -B replay_production_ledger.py --recompute SNAPSHOT_DIR
```
