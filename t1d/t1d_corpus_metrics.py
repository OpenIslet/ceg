#!/usr/bin/env python3
"""Recompute the production corpus figures quoted in section 6.9 of the manuscript.

The corpus is the OpenIslet type 1 diabetes evidence graph, release v19.0.0-R4S4-SOL61-FOLDIN.
It is not part of this package. This script reads it and nothing else, and it writes only
t1d_corpus_metrics.json next to itself.

Recomputed here from the release data: screening funnel, extraction coverage, tiers, directions,
link relations, outcome sources and flags, outcome reach, cited-tuple and cited-study unions, ledger
event counts, every byte count, and every SHA-256 pin. Read from the release's own records and not
recomputed (the project's validator is not run here): the dossier-record and numeric-token counts of
its validator, the point/missing split of its cited tuples, and the certification ledger status. Run from this folder:

    PYTHONDONTWRITEBYTECODE=1 python3 -B t1d_corpus_metrics.py --recompute OPENISLET_DIR [--write]
    PYTHONDONTWRITEBYTECODE=1 python3 -B t1d_corpus_metrics.py --check ../CEG-JOURNAL-MANUSCRIPT-V3.1.md

--recompute recomputes every figure from the OpenIslet checkout and compares it with the stored
t1d_corpus_metrics.json (--write replaces the stored file). --check needs no OpenIslet checkout: it
verifies that the manuscript states the stored figures. Standard library only. No absolute path is
written to the output.
"""
import sys

sys.dont_write_bytecode = True

import collections
import glob
import hashlib
import json
import os
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import ceg_reference as ref  # noqa: E402  (read-only: canonical RFC 8785 serialization)

STORED = HERE / "t1d_corpus_metrics.json"
REL = "reports/2026-11-08-stage1-full-3917-screening"
EG = f"{REL}/evidence_graph"
S3 = f"{REL}/stage3_conveyor_rounds"
MASTERS = {
    "v1": f"{S3}/stage3_master_1151_settled_gold_extractions.json",
    "v2": f"{S3}/stage3_master_1151_settled_gold_v2_derived.json",
    "v3": f"{S3}/stage3_master_1151_settled_gold_v3_derived.json",
    "v4": f"{S3}/stage3_master_1151_settled_gold_v4_derived.json",
}
LEDGER = f"{S3}/sol61_v1_to_v4_reversible_patch_ledger.json"
GRAPH = f"{EG}/context_evidence_graph_v19.json"
REFRESH = f"{EG}/round4_step4_sol61_v19_foldin/CEG-V19-SECTION6-MEASUREMENT-REFRESH.json"
CLAIMS = f"{EG}/certified_primary_claims_ledger_v19.json"
STAGE1 = f"{REL}/stage1_settled_3922_ledger.json"
WAVE0 = f"{REL}/stage2_empirical_packets/wave0_20_pilot_empirical_packets.json"
DOSSIERS = {
    "IIP-0008": f"{EG}/round4_step4_sol61_v19_foldin/SOL61-R4S4-V19-IIP0008-CONSOLIDATED.json",
    "IIP-0009": f"{EG}/round4_step4_sol61_v19_foldin/SOL61-R4S4-V19-IIP0009-CONSOLIDATED.json",
}
VIEWS = {i: f"{EG}/ceg_single_window_views_v19/CEG-{i}-SINGLE-WINDOW-VIEW-V19.md" for i in ("IIP-0007", "IIP-0008", "IIP-0009")}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load(root: Path, rel: str):
    with open(root / rel, encoding="utf-8") as f:
        return json.load(f)


def pct(n: int, d: int) -> float:
    return round(100.0 * n / d, 1)


def compute(root: Path) -> dict:
    g = load(root, GRAPH)
    refresh = load(root, REFRESH)
    claims = load(root, CLAIMS)
    ledger = load(root, LEDGER)
    stage1 = load(root, STAGE1)
    d8, d9 = load(root, DOSSIERS["IIP-0008"]), load(root, DOSSIERS["IIP-0009"])
    studies, outcomes, edges = g["study_contexts"], g["outcome_nodes"], g["edges"]

    # ---- integrity pins, recomputed
    sizes = {k: (root / v).stat().st_size for k, v in MASTERS.items()}
    master_sha = {k: sha256_file(root / v) for k, v in MASTERS.items()}
    ledger_sha = sha256_file(root / LEDGER)
    view_sha = {k: sha256_file(root / v) for k, v in VIEWS.items()}
    view_bytes = {k: (root / v).stat().st_size for k, v in VIEWS.items()}
    pins_ok = {
        "graph_pins_master_v4": g["source_master_sha256"] == master_sha["v4"],
        "graph_pins_ledger": g["reversible_patch_ledger_sha256"] == ledger_sha,
        "graph_pins_views": all(g["single_window_view_hashes_v19"][k]["sha256"] == view_sha[k] for k in VIEWS),
        "ledger_pins_masters": (
            ledger["original_v1_master_sha256"] == master_sha["v1"]
            and ledger["base_v2_master_sha256"] == master_sha["v2"]
            and ledger["base_v3_master_sha256"] == master_sha["v3"]
            and ledger["derived_v4_master_sha256"] == master_sha["v4"]
        ),
    }

    # ---- raw extraction packets: 23 conveyor-round files plus the Wave-0 pilot file
    round_files = sorted(glob.glob(str(root / S3 / "R*" / "packets_R*.json")))
    packet_bytes = sum(os.path.getsize(p) for p in round_files) + (root / WAVE0).stat().st_size
    packet_studies = 0
    for p in round_files:
        d = json.load(open(p, encoding="utf-8"))
        packet_studies += len(d) if isinstance(d, list) else len(d.get("packets", d.get("studies", d)))
    wave0 = json.load(open(root / WAVE0, encoding="utf-8"))
    packet_studies += len(wave0) if isinstance(wave0, list) else len(wave0.get("packets", wave0.get("studies", wave0)))

    # ---- stage-1 funnel and extraction coverage
    s1 = stage1["summary"]
    index = collections.defaultdict(list)
    for key in ("settled_candidates_3359", "settled_ring1_563"):
        for r in stage1[key]:
            for f in ("pmid", "id_value", "doi", "queue_id"):
                if r.get(f):
                    index[str(r[f]).strip().lower()].append((key, r))
    extracted_from = collections.Counter()
    all_advance = 0
    for s in studies.values():
        hit = []
        for f in ("pmid", "id_value", "doi", "queue_id"):
            if s.get(f) and str(s[f]).strip().lower() in index:
                hit = index[str(s[f]).strip().lower()]
                break
        if any(r["final_decision"] == "ADVANCE" for _, r in hit):
            all_advance += 1
        for key in {k for k, _ in hit}:
            extracted_from[key] += 1

    # ---- structure of the graph
    tiers = collections.Counter(s["fidelity_tier"] for s in studies.values())
    direction = collections.Counter(o["direction"] for o in outcomes.values())
    relations = collections.Counter(e["relation"] for e in edges)
    sources = collections.Counter((o.get("source_pin") or {}).get("source_type") for o in outcomes.values())
    stat_types = collections.Counter(o["stat_type"] for o in outcomes.values())
    fig_dig = sum(1 for o in outcomes.values() if o["requires_figure_digitization"] is True)
    quoted = sum(1 for o in outcomes.values() if (o.get("verbatim_quote") or "").strip())
    pmcid = sum(1 for s in studies.values() if s.get("pmcid"))
    def reach_class(x: dict) -> str:
        link = bool(x.get("cited_by_blueprint_link_edge_ids"))
        dossier = bool(x.get("cited_by_iip0008")) or bool(x.get("cited_by_iip0009"))
        if link and dossier:
            return "claim_and_dossier_cited"
        return "claim_cited_only" if link else "dossier_cited_only" if dossier else "study_arm_background_only"

    reach = dict(collections.Counter(reach_class(o["outcome_reachability"]) for o in outcomes.values()))
    reach["total_cited_by_claim_or_dossier"] = len(outcomes) - reach["study_arm_background_only"]
    pins_ok["reach_matches_release_summary"] = (
        {k: v for k, v in reach.items() if k != "total_cited_by_claim_or_dossier"} == g["outcome_reachability_summary"]["by_status"]
    )
    t8, t9 = set(map(str, d8["all_cited_tuples_v19_verified"])), set(map(str, d9["all_cited_tuples_v19_verified"]))
    st8, st9 = set(map(str, d8["all_cited_studies_v19_verified"])), set(map(str, d9["all_cited_studies_v19_verified"]))
    pins_ok["cited_tuples_match_release_record"] = (
        len(t8 | t9) == refresh["cross_dossier_union_active_tuples"]
        and len(st8 | st9) == refresh["cross_dossier_union_active_studies"]
        and (t8 | t9) <= set(outcomes)
    )
    settled = load(root, MASTERS["v4"])["settled_extractions"]
    canonical_bytes = len(ref.canonical_json_bytes(settled))
    n_out, n_std = len(outcomes), len(studies)
    settled_bytes = sizes["v4"]
    views_total = sum(view_bytes.values())

    event_kinds = collections.Counter(e["json_pointer"].split("/")[3] for e in ledger["events"])
    phases = collections.Counter(e["phase"] for e in ledger["events"])
    vs = refresh["validator_summary"]

    m = {
        "source": {
            "corpus": "OpenIslet type 1 diabetes evidence graph",
            "graph_version": g["version"],
            "graph_schema": g["schema_version"],
            "note": "Figures are recomputed from the release files listed under pins; paths are relative to the OpenIslet checkout.",
        },
        "pins": {
            "graph_sha256": sha256_file(root / GRAPH),
            "master_sha256": master_sha,
            "ledger_sha256": ledger_sha,
            "view_sha256": view_sha,
            "refresh_sha256": sha256_file(root / REFRESH),
            "dossier_sha256": {k: sha256_file(root / v) for k, v in DOSSIERS.items()},
            "claims_ledger_sha256": sha256_file(root / CLAIMS),
        },
        "pins_verified": pins_ok,
        "funnel": {
            "screened_records": s1["total_studies"],
            "screened_distinct_identifier_pairs": refresh["screened_distinct_identifier_pairs"],
            "advanced_at_title_abstract": s1["overall_counts"]["ADVANCE"],
            "parked_as_context": s1["overall_counts"]["PARK_CONTEXT"],
            "excluded_off_topic": s1["overall_counts"]["EXCLUDE_OFF_TOPIC"],
            "extracted_studies": n_std,
            "extracted_all_stage1_advance": all_advance == n_std,
            "extracted_from_candidate_records": extracted_from["settled_candidates_3359"],
            "extracted_from_ring1_records": extracted_from["settled_ring1_563"],
            "advanced_not_extracted": s1["overall_counts"]["ADVANCE"] - n_std,
            "deferred_supporting_catalog_records": s1["candidate_extraction_waves"]["Wave_2B_Supporting_Catalog_P2_P3"],
        },
        "structure": {
            "studies": n_std,
            "study_arms": len(g["variable_nodes"]),
            "outcomes": n_out,
            "edges": len(edges),
            "effector_nodes": len(g["effector_nodes"]),
            "tiers": dict(sorted(tiers.items())),
            "directions": dict(direction.most_common()),
            "edge_relations": dict(relations.most_common()),
            "studies_with_pmcid": pmcid,
        },
        "reachability": {
            "claim_cited_only": reach["claim_cited_only"],
            "claim_and_dossier_cited": reach["claim_and_dossier_cited"],
            "dossier_cited_only": reach["dossier_cited_only"],
            "study_arm_background_only": reach["study_arm_background_only"],
            "cited_by_claim_or_dossier": reach["total_cited_by_claim_or_dossier"],
            "cited_percent": pct(reach["total_cited_by_claim_or_dossier"], n_out),
            "background_percent": pct(reach["study_arm_background_only"], n_out),
        },
        "extraction_quality": {
            "outcomes_with_verbatim_quote": quoted,
            "outcomes_by_source_type": dict(sources.most_common()),
            "abstract_sourced_percent": pct(sources["abstract"], n_out),
            "results_text_percent": pct(sources["results_text"], n_out),
            "requires_figure_digitization": fig_dig,
            "requires_figure_digitization_percent": pct(fig_dig, n_out),
            "qualitative_with_pvalue": stat_types["qualitative_with_pvalue"],
            "qualitative_with_pvalue_percent": pct(stat_types["qualitative_with_pvalue"], n_out),
            "studies_with_pmcid_percent": pct(pmcid, n_std),
        },
        "certification": {
            "certified_outcomes": claims["certified_outcomes_count"],
            "ledger_status": claims["status"],
        },
        "dossiers": {
            "indexed_records": refresh["indexed_dossier_records"],
            "distinct_record_ids": refresh["distinct_canonical_record_ids"],
            "mirrored_records": refresh["mirrored_records"],
            "checked_items_by_dossier": vs["checked_items_by_dossier"],
            "numeric_token_occurrences": vs["checked_numeric_token_occurrences"],
            "numeric_span_occurrences": vs["checked_numeric_span_occurrences"],
            "unvisited_active_records": vs["unvisited_active_records"],
            "active_tuples": {"IIP-0008": len(t8), "IIP-0009": len(t9), "union": len(t8 | t9)},
            "active_studies": {"IIP-0008": len(st8), "IIP-0009": len(st9), "union": len(st8 | st9)},
            "parser_derived_atoms": {"total": refresh["parser_derived_atoms_total"], "point": refresh["parser_derived_atoms_point"], "missing": refresh["parser_derived_atoms_missing"]},
            "verdict_flags": {
                "IIP-0008": {"numeric_promotion_authorized": d8.get("numeric_promotion_authorized")},
                "IIP-0009": {
                    "numeric_model_update_allowed": d9.get("numeric_model_update_allowed"),
                    "payload_promotion_allowed": d9.get("payload_promotion_allowed"),
                    "quantitative_promotion_supported_for_any_item": d9.get("quantitative_promotion_supported_for_any_item"),
                },
            },
        },
        "ledger": {
            "events": len(ledger["events"]),
            "patched_studies": len({e["queue_id"] for e in ledger["events"]}),
            "events_by_phase": dict(sorted(phases.items())),
            "events_by_record_part": dict(event_kinds.most_common()),
            "forward_replay_flag": ledger["forward_replay_v1_to_v4_verified"],
            "reverse_replay_flag": ledger["reverse_replay_v4_to_v1_verified"],
        },
        "bytes": {
            "raw_extraction_packets": packet_bytes,
            "raw_extraction_packet_studies": packet_studies,
            "settled_corpus_file": settled_bytes,
            "settled_corpus_canonical": canonical_bytes,
            "settled_corpus_canonical_per_study": round(canonical_bytes / n_std),
            "settled_corpus_canonical_per_outcome": round(canonical_bytes / n_out),
            "outcomes_per_study": round(n_out / n_std, 1),
            "single_window_views": dict(sorted(view_bytes.items())),
            "single_window_views_total": views_total,
            "indexed_graph": (root / GRAPH).stat().st_size,
        },
    }
    return m


NUMBER_WORDS = {1: "one"}


def n(x) -> str:
    return f"{x:,}"


def kb(x: int) -> str:
    return f"{x / 1000:.1f}"


def million(x: float) -> str:
    return f"{x / 1e6:.1f}"


def statements(m: dict) -> list:
    f, s, r, q, c, d, lg, b = (m[k] for k in ("funnel", "structure", "reachability", "extraction_quality", "certification", "dossiers", "ledger", "bytes"))
    t = s["tiers"]
    dr = s["directions"]
    rel = s["edge_relations"]
    src = q["outcomes_by_source_type"]
    v = b["single_window_views"]
    return [
        f"{n(f['screened_records'])} screened records",
        f"{n(f['screened_distinct_identifier_pairs'])} distinct identifier pairs",
        f"{n(f['advanced_at_title_abstract'])} advanced",
        f"{n(f['extracted_studies'])} studies",
        f"{n(f['advanced_not_extracted'])} advanced records",
        f"{n(s['study_arms'])} study arms",
        f"{n(s['outcomes'])} outcomes",
        f"{n(s['edges'])} study-to-design links",
        f"{n(t['T1_Human_Clinical'])} human clinical",
        f"{n(t['T2_NHP'])} non-human primate",
        f"{n(t['T3_Humanized_Mouse'])} humanized mouse",
        f"{n(t['T4_Human_Islet_or_SC_Islet_InVitro'])} human or stem-cell-derived islet in vitro",
        f"{n(t['T5_Rodent_or_CellLine'])} rodent or cell line",
        f"{n(t['T6_Biophysical_Model_or_Regulatory'])} biophysical model or regulatory",
        f"{n(dr['favorable_to_intervention'])} favorable",
        f"{n(dr['context_dependent_or_mixed'])} context-dependent or mixed",
        f"{n(dr['unfavorable_or_toxic'])} unfavorable or toxic",
        f"{n(dr['null_or_no_difference'])} null",
        f"{n(dr['unknown_or_unresolvable'])} unresolvable",
        f"{n(rel['supports'])} supports",
        f"{n(rel['scoping_qualification'])} scoping qualifications",
        f"{n(rel['novel_upgrade_hypothesis'])} novel upgrade hypotheses",
        f"{n(rel['contradicts'])} contradicts",
        f"{n(r['cited_by_claim_or_dossier'])} of {n(s['outcomes'])} outcomes",
        f"{n(r['study_arm_background_only'])} outcomes ({r['background_percent']}%)",
        f"{n(src['abstract'])} outcomes ({q['abstract_sourced_percent']}%)",
        f"{n(q['requires_figure_digitization'])} outcomes ({q['requires_figure_digitization_percent']}%)",
        f"{n(q['qualitative_with_pvalue'])} outcomes ({q['qualitative_with_pvalue_percent']}%)",
        f"{n(s['studies_with_pmcid'])} of {n(s['studies'])} studies",
        f"{n(c['certified_outcomes'])} of {n(s['outcomes'])} outcomes certified",
        f"{n(d['indexed_records'])} indexed dossier records",
        f"{n(d['distinct_record_ids'])} distinct",
        f"{n(d['active_tuples']['union'])} cited outcome tuples",
        f"{n(d['parser_derived_atoms']['point'])} carry a point-level quantity",
        f"{n(d['parser_derived_atoms']['missing'])} are typed as missing numbers",
        f"{n(lg['events'])} reversible events",
        f"{n(lg['patched_studies'])} studies",
        f"{n(b['settled_corpus_canonical'])} bytes",
        f"{n(b['settled_corpus_file'])} bytes as stored",
        f"{kb(b['settled_corpus_canonical_per_study'])} kB per study",
        f"{kb(b['settled_corpus_canonical_per_outcome'])} kB per outcome",
        f"about {b['outcomes_per_study']} outcomes per study",
        f"{n(b['raw_extraction_packets'])} bytes of raw extraction packets",
        f"{n(v['IIP-0007'])}, {n(v['IIP-0008'])}, and {n(v['IIP-0009'])} bytes",
        f"{n(b['single_window_views_total'])} in total",
        f"{n(b['indexed_graph'])} bytes",
        f"{n(d['numeric_token_occurrences'])} numeric-token occurrences",
        f"across {n(d['active_studies']['union'])} studies",
        f"{million(min(v.values()))} to {million(max(v.values()))} million bytes",
        f"{million(min(v.values()) / 4)} to {million(min(v.values()) / 3)} million tokens",
        f"{million(max(v.values()) / 4)} to {million(max(v.values()) / 3)} million tokens",
        f"{million(b['settled_corpus_canonical'] / 4)} to {million(b['settled_corpus_canonical'] / 3)} million tokens",
        f"{n(f['parked_as_context'])} parked as context",
        f"{n(f['excluded_off_topic'])} excluded as off topic",
        f"{n(f['extracted_from_candidate_records'])} candidate records",
        f"{n(f['extracted_from_ring1_records'])} core-list records",
        f"{n(rel['noninformative'])} noninformative",
        f"results text for {n(src['results_text'])} outcomes",
        f"to a figure caption for {n(src['figure_caption'])}, to a table for {n(src['table_grid'])}, and to methods text for {n(src['methods_text'])}",
        "{} events from the first to the second corpus version, {} from the second to the third, and {} from the third to the fourth".format(
            *(lg["events_by_phase"][k] for k in sorted(lg["events_by_phase"]))
        ),
        "outcome records ({outcomes} events), study-to-design links ({blueprint_links}), arm lists ({arms}), and study-design fields ({study_design})".format(
            **lg["events_by_record_part"]
        ),
        m["pins"]["master_sha256"]["v1"][:12],
        m["pins"]["master_sha256"]["v4"][:12],
        m["pins"]["ledger_sha256"][:12],
    ] + density_statements(m)


def density_statements(m: dict) -> list:
    """Cross-file figures: production density against the 12-study corpus of section 6.8."""
    small = HERE.parent / "publication" / "record_size.json"
    if not small.exists():
        return []
    z = json.load(open(small, encoding="utf-8"))
    ratio = m["bytes"]["settled_corpus_canonical_per_study"] / (z["total_bytes"] / z["studies"])
    return [
        f"about {ratio:.1f} times",
        f"about {z['atoms'] / z['studies']:.1f} outcomes per study",
        f"{kb(round(z['total_bytes'] / z['atoms']))} kB per outcome",
    ]


def main(argv: list) -> int:
    if len(argv) >= 3 and argv[1] == "--check":
        m = json.load(open(STORED, encoding="utf-8"))
        text = Path(argv[2]).read_text(encoding="utf-8")
        missing = [x for x in statements(m) if x not in text]
        if missing:
            for x in missing:
                print(f"T1D CORPUS MISMATCH: manuscript does not state: {x}")
            return 1
        print(f"T1D CORPUS OK: {len(statements(m))} statements in {Path(argv[2]).name} match {STORED.name}")
        return 0
    if len(argv) >= 3 and argv[1] == "--recompute":
        root = Path(argv[2]).expanduser().resolve()
        m = compute(root)
        bad = [k for k, ok in m["pins_verified"].items() if not ok]
        flags = [x for v in m["dossiers"]["verdict_flags"].values() for x in v.values()]
        if any(x is not False for x in flags):
            bad.append("a dossier verdict flag is not False")
        text = json.dumps(m, indent=2, sort_keys=True) + "\n"
        if "--write" in argv:
            STORED.write_text(text, encoding="utf-8")
            print(f"wrote {STORED.name}: {m['structure']['studies']:,} studies, {m['structure']['outcomes']:,} outcomes, integrity pins verified: {not bad}")
        else:
            same = STORED.exists() and STORED.read_text(encoding="utf-8") == text
            print(f"recomputed figures {'match' if same else 'DIFFER FROM'} {STORED.name}; integrity pins verified: {not bad}")
            if not same:
                return 1
        return 1 if bad else 0
    print(__doc__)
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv))
