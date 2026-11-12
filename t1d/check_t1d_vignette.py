#!/usr/bin/env python3
"""Checks on the type 1 diabetes worked example of section 5 (`domain_vignettes.build_t1d_vignette`).

Run from this folder:

    PYTHONDONTWRITEBYTECODE=1 python3 -B check_t1d_vignette.py --check ../CEG-JOURNAL-MANUSCRIPT-V3.1.md
    PYTHONDONTWRITEBYTECODE=1 python3 -B check_t1d_vignette.py --recompute OPENISLET_DIR [--write]
    PYTHONDONTWRITEBYTECODE=1 python3 -B check_t1d_vignette.py --quotes OPENISLET_DIR [--write]
    PYTHONDONTWRITEBYTECODE=1 python3 -B check_t1d_vignette.py --table

--check     needs nothing but this package. It rebuilds the example from ../domain_vignettes.py, confirms that
            it reproduces the stored t1d_vignette_verification.json, and verifies that the manuscript carries
            every row of Table 1 and every stated count, decision, violation code and digest.
--recompute also compares each of the ten findings with its record in the OpenIslet corpus (verbatim quote,
            PubMed identifier, arm identifiers, figure flag, direction, sample size).
--quotes    needs network access to NCBI. It looks for each verbatim quote in the PubMed abstract and, where
            there is one, the PubMed Central full text of its study, and stores the result.
--table     prints Table 1 as Markdown.

The OpenIslet checkout is read, never written. Standard library only. No absolute path is stored.
"""
import sys

sys.dont_write_bytecode = True

import html
import itertools
import json
import re
import unicodedata
from pathlib import Path

HERE = Path(__file__).resolve().parent
PKG = HERE.parent
sys.path.insert(0, str(PKG))
import ceg_reference as ref  # noqa: E402
import domain_vignettes as vignettes  # noqa: E402

STORED = HERE / "t1d_vignette_verification.json"
QUOTES = HERE / "t1d_vignette_quote_check.json"
GRAPH = "reports/2026-11-08-stage1-full-3917-screening/evidence_graph/context_evidence_graph_v19.json"
POLARITY = {
    "favorable_to_intervention": "favorable",
    "unfavorable_or_toxic": "unfavorable",
    "context_dependent_or_mixed": "neutral_or_mixed",
}
WORDS = {1: "one", 2: "two", 3: "three", 4: "four", 5: "five", 6: "six", 7: "seven", 8: "eight", 9: "nine", 10: "ten", 40: "forty"}

# Reader-facing wording for Table 1: (reference number, setting, endpoint, result as recorded).
TABLE = {
    "R1_0340:O2": (46, "Human population genetics, type 2 diabetes", "Type 2 diabetes risk, *SLC30A8* loss-of-function carriers vs non-carriers", "$37\\%$ decrease; odds ratio $0.62$"),
    "R1_0141:O9": (47, "Human iPSC-derived beta-like cells in vitro", "Number of *INS*- and *SLC30A8*-expressing cells, two mutant-genotype groups (four clones) vs grouped controls", "Fewer cells; no number"),
    "R1_0116:O1": (48, "Immunosuppressed clinical islet allograft recipients with type 1 diabetes", "Risk factors for failure to reach insulin independence", "HLA-A\\*24, *SLC30A8* CT or TT genotype, and BMI at or above the group median; no effect size"),
    "R1_0214:O4": (49, "Stem-cell-derived islets grafted in humanized mice", "Guide-RNA enrichment in grafts of mice given human immune cells vs control mice, genome-wide screen", "*B2M*, *HLA-A*, *TAP1/2*, *STAT1*, *JAK1/2*, *CXCL10* enriched; no number"),
    "R1_0214:O12": (49, "Stem-cell-derived islets in vitro, interferon-gamma", "HLA, CXCL10 and inhibitory proteins, *STAT1* knockout vs control", "HLA, CXCL10, HLA-E, PD-L1 and SOCS1 all reduced; no number"),
    "R1_0215:O7": (50, "Human iPSC-derived beta-like cells in vitro, interferon-alpha", "HLA-ABC after *NLRC5* knockdown vs control", "Reduced; no number"),
    "R1_0215:O27": (50, "Human iPSC-derived islet cells in vitro, interferon-alpha", "*BCL2L1* expression, alpha-like vs beta-like cells", "$0.13$ log2 fold change higher in alpha-like cells"),
    "R1_0098:O8": (51, "Human ESC-derived cells, endocrine cells matured in mice, assayed in vitro", "CD55 expression, endocrine cells vs pancreatic endoderm", "Clearly expressed on endocrine cells, lower on endoderm; no number"),
    "R1_0202:O7": (52, "INS-1 rat beta-cell line", "Glucose-stimulated insulin secretion, *CD59* knockout vs control", "Strongly inhibited; no number"),
    "R1_0376:O15": (53, "Stem-cell-derived beta cells in vitro, hypoxia", "*TXNIP* expression, hypoxia vs normoxia", "Upregulated; no number"),
}
AXIS_WORDS = {"model_system": "model", "immune_challenge": "immune", "engraftment_microenvironment": "engraftment", "readout_maturity": "read-out"}


def inadmissible_reasons(lat, source, target) -> dict:
    """Per unresolved axis, why the transfer is inadmissible (same two tests as evaluate_transfer)."""
    out = {}
    for axis, poset in lat.axis_posets.items():
        s_val, t_val = source.axes[axis], target.axes[axis]
        if poset.leq(t_val, s_val):
            continue
        if not lat.has_common_upper_bound(axis, s_val, t_val):
            out[axis] = "has no common upper bound"
        elif axis in lat.bridge_floor and not poset.leq(lat.bridge_floor[axis], s_val):
            out[axis] = "is below the floor"
    return out


def run() -> dict:
    b = vignettes.build_t1d_vignette()
    checks = b.run_all_checks()
    lat = b.graph.lattice
    axes = list(lat.axis_posets)
    sizes = {k: len(p.dominates) for k, p in lat.axis_posets.items()}
    contexts = [ref.ContextSignature(dict(zip(axes, combo))) for combo in itertools.product(*(sorted(lat.axis_posets[a].dominates) for a in axes))]
    pair_classes: dict = {}
    for s in contexts:
        for t in contexts:
            c = lat.evaluate_transfer(s, t)[0]
            pair_classes[c] = pair_classes.get(c, 0) + 1
    target = dict(b.target_context.axes)
    model_axis = {}
    for element in sorted(lat.axis_posets["model_system"].dominates):
        model_axis[element] = lat.evaluate_transfer(ref.ContextSignature(dict(target, model_system=element)), b.target_context)[0]
    findings = {}
    for tup, atom in b.graph.atoms.items():
        cls, unresolved = lat.evaluate_transfer(atom.context, b.target_context)
        findings[tup] = {
            "pmid": atom.provenance.document_id.split(":")[1],
            "context": dict(atom.context.axes),
            "transfer_class": cls,
            "unresolved_axes": list(unresolved),
            "inadmissible_because": inadmissible_reasons(lat, atom.context, b.target_context) if cls == "INADMISSIBLE_TRANSFER" else {},
            "contrast_kind": atom.contrast.contrast_kind,
            "arms": len(atom.contrast.focal_arms) + len(atom.contrast.comparator_arms),
            "quantity_kind": atom.focal_quantity.kind,
            "sample_size": atom.sample_size,
            "requires_figure_digitization": atom.provenance.requires_figure_digitization,
            "primary_pdf_verified": atom.provenance.primary_pdf_verified,
            "rendered_sentence": ref.render_atom_sentence(atom),
        }
    roles = {}
    for rec in b.valid_records:
        roles[rec.record_id] = {"supporting": list(rec.supporting_tuples), "counterevidence": list(rec.counterevidence_tuples), "gate": rec.prospective_gate.gate_id if rec.prospective_gate else None}
    return {
        "reference_release": ref.REFERENCE_VERSION,
        "studies": checks["studies_count"],
        "arm_variables": checks["variables_count"],
        "atoms": checks["atoms_count"],
        "axis_sizes": sizes,
        "contexts": len(contexts),
        "ordered_context_pairs": len(contexts) ** 2,
        "pair_classes": dict(sorted(pair_classes.items())),
        "model_system_class_toward_target": model_axis,
        "target_context": target,
        "patch_replay": checks["patch_replay"],
        "findings": findings,
        "records": {k: dict(v, **roles[k]) for k, v in checks["valid_records_verified"].items()},
        "probes": {k: {"expected_code": v["expected_code_prefix"], "codes": [x.split(":")[0].split(" ")[0] for x in v["violations"]]} for k, v in checks["invalid_probes_rejected"].items()},
        "run_digest": ref.sha256_canonical(checks),
    }


def table_rows(m: dict) -> list:
    rows = []
    if set(TABLE) != set(m["findings"]):
        raise ValueError("Table 1 wording and the vignette findings differ")
    for tup in TABLE:
        f = m["findings"][tup]
        refn, setting, endpoint, result = TABLE[tup]
        kind = f"`{f['contrast_kind']}`" + (f", {f['arms']} arms" if f["contrast_kind"] == "multi_arm_joint" else "")
        endpoint = f"{endpoint} ({kind})"
        result = f"{result} (`{f['quantity_kind']}`" + (f", $n = {f['sample_size']}$)" if f["sample_size"] else ")")
        axes = "all 4 axes" if len(f["unresolved_axes"]) == 4 else ", ".join(AXIS_WORDS[a] for a in f["unresolved_axes"])
        if f["transfer_class"] == "DIRECT_MATCH":
            cls = "`DIRECT_MATCH`"
        elif f["transfer_class"] == "BOUNDED_BRIDGE":
            cls = f"`BOUNDED_BRIDGE`; {axes}"
        else:
            why = "; ".join(f"{AXIS_WORDS[a]} {reason}" for a, reason in sorted(f["inadmissible_because"].items()))
            cls = f"`INADMISSIBLE_TRANSFER`; {axes}; {why}"
        rows.append(f"| `{tup}` | {setting} [{refn}] | {endpoint} | {result} | {cls} |")
    return rows


def statements(m: dict) -> list:
    s = m["axis_sizes"]
    pc = m["pair_classes"]
    pr = m["patch_replay"]
    order = sorted(s.values(), reverse=True)
    out = [
        "$" + " \\times ".join(str(x) for x in order) + f" = {m['contexts']}$",
        f"{m['ordered_context_pairs']:,} ordered context pairs yields {pc['DIRECT_MATCH']:,} `DIRECT_MATCH`, {pc['BOUNDED_BRIDGE']:,} `BOUNDED_BRIDGE` and {pc['INADMISSIBLE_TRANSFER']:,} `INADMISSIBLE_TRANSFER`",
        f"{m['studies']} studies, {m['arm_variables']} arm variables and {m['atoms']} evidence atoms",
        f"SHA-256 prefix `{pr['initial_sha256'][:12]}`",
        f"SHA-256 prefix `{pr['settled_sha256'][:12]}`",
        f"{WORDS[len(m['records'])].capitalize()} synthesis records verify",
        f"all {WORDS[len(m['probes'])]} adversarial probes fail",
        f"Hasse diagram of the {WORDS[s['model_system']]}-element `model_system` axis in the worked type 1 diabetes example",
    ]
    for rid, r in m["records"].items():
        out.append(f"`{rid}` $\\to$ `{r['promotion_decision']}`")
    for p in m["probes"].values():
        out.append(f"`{p['expected_code']}`")
    return out


def norm(s: str) -> str:
    s = html.unescape(re.sub(r"<[^>]+>", "", s))
    s = unicodedata.normalize("NFKC", s).lower()
    for a, b in (("\u2013", "-"), ("\u2212", "-"), ("\u2010", "-"), ("\u2011", "-"), ("\u00d7", "x")):
        s = s.replace(a, b)
    return re.sub(r"[^a-z0-9\u03b1-\u03c9]+", " ", s).strip()


def n_supported(n, summary: str) -> bool:
    if n is None:
        return True
    low = summary.lower()
    if re.search(rf"\b{n}\b", low) or WORDS.get(n, "\0") in low:
        return True
    parts = [int(x) for x in re.findall(r"n = (\d+)", low)]
    return len(parts) == 2 and sum(parts) == n


def corpus_comparison(root: Path) -> dict:
    g = json.load(open(root / GRAPH, encoding="utf-8"))
    outcomes, variables = g["outcome_nodes"], g["variable_nodes"]
    studies = {s["queue_id"]: s for s in g["study_contexts"].values()}
    b = vignettes.build_t1d_vignette()
    out = {}
    for tup, atom in b.graph.atoms.items():
        o = outcomes[tup]
        st = studies[atom.study_id]
        arms = list(atom.contrast.focal_arms) + list(atom.contrast.comparator_arms)
        out[tup] = {
            "quote_identical": atom.provenance.verbatim_quote == o["verbatim_quote"],
            "pmid_identical": atom.provenance.document_id == f"PMID:{st['pmid']}",
            "arms_exist_in_corpus": all(a in variables for a in arms),
            "figure_flag_identical": atom.provenance.requires_figure_digitization == o["requires_figure_digitization"],
            "polarity_matches_direction": POLARITY.get(o["direction"]) == atom.polarity,
            "sample_size_supported": n_supported(atom.sample_size, o["n_summary"]),
            "corpus_n_summary": o["n_summary"],
            "corpus_source_type": (o.get("source_pin") or {}).get("source_type"),
            "corpus_fidelity_tier": st["fidelity_tier"],
            "pmcid": st.get("pmcid"),
        }
    return {"graph_version": g["version"], "findings": out}


def fetch_quotes(root: Path) -> dict:
    import time
    import urllib.parse
    import urllib.request

    comp = corpus_comparison(root)["findings"]
    b = vignettes.build_t1d_vignette()
    base = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?"

    def get(params: dict) -> str:
        with urllib.request.urlopen(base + urllib.parse.urlencode(params), timeout=120) as r:
            return r.read().decode("utf-8", "replace")

    cache: dict = {}
    out = {}
    for tup, atom in b.graph.atoms.items():
        pmid = atom.provenance.document_id.split(":")[1]
        pmcid = comp[tup]["pmcid"]
        if pmid not in cache:
            abstract = get({"db": "pubmed", "id": pmid, "rettype": "abstract", "retmode": "xml"})
            time.sleep(0.4)
            full = get({"db": "pmc", "id": pmcid.replace("PMC", ""), "retmode": "xml"}) if pmcid else ""
            time.sleep(0.4)
            cache[pmid] = (norm(abstract), norm(full) if "<body" in full else "")
        abstract, full = cache[pmid]
        q = norm(atom.provenance.verbatim_quote)
        out[tup] = {
            "pmid": pmid,
            "pmcid": pmcid,
            "found_in_pubmed_abstract": q in abstract,
            "found_in_pmc_full_text": bool(full) and q in full,
            "pmc_full_text_available": bool(full),
        }
    found = sum(v["found_in_pubmed_abstract"] or v["found_in_pmc_full_text"] for v in out.values())
    return {
        "method": "NCBI E-utilities efetch (db=pubmed abstract XML; db=pmc full-text XML). Tags removed, NFKC, lower case, dashes and the multiplication sign folded to ASCII, runs of other characters to one space; the quote must then occur as a substring.",
        "what_this_does_not_show": "Only the quoted sentence is located. Arms, direction, sample size and the absence of a contradicting statement elsewhere in the paper are not checked, and no PDF is read.",
        "quotes_checked": len(out),
        "quotes_found": found,
        "found_in_pmc_full_text": sum(v["found_in_pmc_full_text"] for v in out.values()),
        "found_in_pubmed_abstract_only": sum(v["found_in_pubmed_abstract"] and not v["found_in_pmc_full_text"] for v in out.values()),
        "findings": out,
    }


def figure1_matches(m: dict) -> list:
    src = (PKG / "publication" / "make_figures.py").read_text(encoding="utf-8")
    problems = []
    for element, cls in m["model_system_class_toward_target"].items():
        hit = re.search(rf'"{element}": \([^)]*"({"|".join(("DIRECT_MATCH", "BOUNDED_BRIDGE", "INADMISSIBLE_TRANSFER"))})"\)', src)
        if not hit:
            problems.append(f"Figure 1 has no node {element}")
        elif hit.group(1) != cls:
            problems.append(f"Figure 1 colours {element} as {hit.group(1)}, the lattice gives {cls}")
    return problems


def main(argv: list) -> int:
    if len(argv) >= 2 and argv[1] == "--table":
        print("\n".join(table_rows(json.load(open(STORED, encoding="utf-8")))))
        return 0
    if len(argv) >= 3 and argv[1] == "--check":
        m = json.load(open(STORED, encoding="utf-8"))
        text = Path(argv[2]).read_text(encoding="utf-8")
        problems = []
        live = run()
        stored_core = {k: v for k, v in m.items() if k != "corpus_comparison"}
        if live != stored_core:
            problems.append("the example built from ../domain_vignettes.py does not reproduce the stored verification")
        cc = m.get("corpus_comparison", {}).get("findings", {})
        for tup, c in cc.items():
            bad = [k for k, v in c.items() if isinstance(v, bool) and not v]
            if bad:
                problems.append(f"stored corpus comparison of {tup} failed: {bad}")
        if len(cc) != m["atoms"]:
            problems.append("stored corpus comparison does not cover every finding")
        problems += [f"manuscript lacks Table 1 row: {r[:70]}..." for r in table_rows(m) if r not in text]
        problems += [f"manuscript does not state: {s}" for s in statements(m) if s not in text]
        problems += figure1_matches(m)
        if QUOTES.exists():
            qc = json.load(open(QUOTES, encoding="utf-8"))
            claim = f"all {WORDS[qc['quotes_checked']]} quoted sentences"
            if qc["quotes_found"] != qc["quotes_checked"] or claim not in text:
                problems.append(f"quote check: found {qc['quotes_found']} of {qc['quotes_checked']}; manuscript must state '{claim}'")
            for s in (f"{WORDS[qc['found_in_pmc_full_text']]} in PubMed Central full text", f"{WORDS[qc['found_in_pubmed_abstract_only']]} in a PubMed abstract"):
                if s not in text:
                    problems.append(f"manuscript does not state: {s}")
        else:
            problems.append("t1d_vignette_quote_check.json is missing")
        if problems:
            for p in problems:
                print(f"T1D VIGNETTE MISMATCH: {p}")
            return 1
        print(f"T1D VIGNETTE OK: {len(table_rows(m))} table rows and {len(statements(m))} statements in {Path(argv[2]).name} match; the example reproduces its stored verification; Figure 1 matches the lattice")
        return 0
    if len(argv) >= 3 and argv[1] in ("--recompute", "--quotes"):
        root = Path(argv[2]).expanduser().resolve()
        if argv[1] == "--recompute":
            result = dict(run(), corpus_comparison=corpus_comparison(root))
            target = STORED
            ok = all(v for c in result["corpus_comparison"]["findings"].values() for v in c.values() if isinstance(v, bool))
            label = f"corpus comparison passed: {ok}"
        else:
            result = fetch_quotes(root)
            target = QUOTES
            ok = result["quotes_found"] == result["quotes_checked"]
            label = f"quotes found: {result['quotes_found']} of {result['quotes_checked']}"
        text = json.dumps(result, indent=2, sort_keys=True) + "\n"
        if "--write" in argv:
            target.write_text(text, encoding="utf-8")
            print(f"wrote {target.name}; {label}")
        else:
            same = target.exists() and target.read_text(encoding="utf-8") == text
            print(f"result {'matches' if same else 'DIFFERS FROM'} {target.name}; {label}")
            ok = ok and same
        return 0 if ok else 1
    print(__doc__)
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv))
