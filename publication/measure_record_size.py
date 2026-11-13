#!/usr/bin/env python3
"""Measure the serialized size of the EvidenceAtoms of the 12-study real-text corpus.

Run from this folder:

    PYTHONDONTWRITEBYTECODE=1 python3 -B measure_record_size.py
    PYTHONDONTWRITEBYTECODE=1 python3 -B measure_record_size.py --check ../CEG-JOURNAL-MANUSCRIPT-V3.2.md

The first form writes record_size.json. The second form recomputes the figures and
verifies that the record-size sentences of section 7.3 of the manuscript state them.

The script imports the package code read-only and writes nothing outside this folder.
Standard library only. Sizes are bytes of RFC 8785 canonical JSON, as produced by
ceg_reference.canonical_json_bytes; no tokenizer is involved.
"""
import sys

sys.dont_write_bytecode = True

import json
import statistics
from collections.abc import Mapping
from pathlib import Path

HERE = Path(__file__).resolve().parent
PKG = HERE.parent
sys.path.insert(0, str(PKG))

import ceg_reference as ref  # noqa: E402
import ceg_real_text_evaluation as real_text  # noqa: E402


def measure() -> dict:
    graph = real_text.build_real_text_corpus()[0]
    atoms = graph.atoms
    atoms = list(atoms.values()) if isinstance(atoms, Mapping) else list(atoms)
    atoms.sort(key=lambda a: a.atom_id)
    sizes = [len(ref.canonical_json_bytes(ref._atom_to_dict(a))) for a in atoms]
    studies = sorted({a.study_id for a in atoms})
    per_study = [sum(1 for a in atoms if a.study_id == s) for s in studies]
    total = sum(sizes)
    return {
        "source": "ceg_real_text_evaluation.build_real_text_corpus",
        "serialization": "RFC 8785 canonical JSON of each atom (ceg_reference.canonical_json_bytes)",
        "unit": "bytes",
        "studies": len(studies),
        "atoms": len(atoms),
        "atoms_per_study_min": min(per_study),
        "atoms_per_study_max": max(per_study),
        "total_bytes": total,
        "median_bytes_per_atom": statistics.median(sizes),
        "min_bytes_per_atom": min(sizes),
        "max_bytes_per_atom": max(sizes),
        "mean_bytes_per_study": round(total / len(studies), 1),
        "bytes_per_atom": {a.atom_id: n for a, n in zip(atoms, sizes)},
    }


NUMBER_WORDS = {1: "one", 2: "two", 3: "three", 4: "four", 5: "five", 6: "six"}


def manuscript_claims(m: dict) -> list:
    per_study_kb = m["total_bytes"] / m["studies"] / 1000
    return [
        f"the {m['atoms']} `EvidenceAtom`s of §6.8 occupy {m['total_bytes']:,} bytes in total",
        f"a median of {m['median_bytes_per_atom']:,} bytes per atom",
        f"(range {m['min_bytes_per_atom']:,} to {m['max_bytes_per_atom']:,})",
        f"about {per_study_kb:.1f} kB per study",
        f"with {NUMBER_WORDS[m['atoms_per_study_min']]} to {NUMBER_WORDS[m['atoms_per_study_max']]} curated outcomes per study",
    ]


def main() -> int:
    m = measure()
    if len(sys.argv) == 3 and sys.argv[1] == "--check":
        text = Path(sys.argv[2]).read_text(encoding="utf-8")
        missing = [c for c in manuscript_claims(m) if text.count(c) != 1]
        if missing:
            for c in missing:
                print(f"RECORD SIZE MISMATCH: manuscript does not state exactly once: {c}")
            return 1
        print(f"RECORD SIZE OK: {len(manuscript_claims(m))} statements in {Path(sys.argv[2]).name} match the measured sizes")
        return 0
    if len(sys.argv) != 1:
        print(__doc__)
        return 2
    out = HERE / "record_size.json"
    out.write_text(json.dumps(m, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote {out.name}: {m['atoms']} atoms, {m['studies']} studies, {m['total_bytes']:,} bytes, "
          f"median {m['median_bytes_per_atom']:,} bytes per atom")
    return 0


if __name__ == "__main__":
    sys.exit(main())
