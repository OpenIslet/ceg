#!/usr/bin/env python3
"""Deterministic conformance and stress suite for Context-Evidence Graphs (CEGs).

The suite is constructive: every case is generated from a sentence or record
template with known ground truth, and each family is replicated with different
numerals and identifiers. Replication tests numeric robustness, not linguistic
diversity, so every stratum reports its number of distinct templates next to
its number of cases. Rates measured here describe the verifier on these
families; they are not estimates of field performance.

Five suites:
  core      F1..F10 adversarial (60 each) and P1..P4 valid (50 each).
  stress    F11..F20 adversarial (60 each) and P5..P8 valid (50 each): harder
            families, including valid prose a strict verifier refuses.
  residual  R1..R4 adversarial (60 each): defects the verifier is NOT expected
            to detect. They document the boundary of surface-prose checking.
  redteam   T5, T11, T14..T18 adversarial (43 sentences, one case each):
            variations an independent reviewer wrote against release 2.0.
  gauntlet  G1..G5 adversarial (38 cases, one per template) and G0 valid
            (10 cases, one per template): 6 domain-agnostic failure classes
            from the V2->V3 audit and parallel gauntlet (k-cycles, cross-clause
            thefts, contrastive subordinators, comma-appositive swaps, Mn/Me
            combining splitters, case-sensitive SI units, label/unit collisions,
            negated inequality polarity, non-contrast multi-arm binding, colon-
            bearing study IDs, and orphaned study records).

Five nested regimes:
  V0   citation presence.
  V1   study-wide bag of numbers.
  V2   prose lint (I1 + I3_tok + I3_clause) that PERMITS numeral-bearing pooled
       citations, falling back to a union-pool check.
  V2s  the same lint, REJECTING numeral-bearing pooled citations.
  V3   full CEG: graph integrity, I2 replay, atom-corpus binding, V2s prose,
       I3_typed with promotion coverage, I4, I5, I6.

Eight exhaustive self-tests (not sampled) accompany the suites: typed-proposition
single-coordinate mutations, ledger tampering, agreement of the transfer
classifier with a literal transcription of Definition 1 over every ordered
pair of contexts, a search for scalar rankings of the diamond axis, the
canonical-rendering round trip, Mode A vs Mode B self-attesting negative
controls plus non-prose surface locks, Layer-1 structure-only vs Layer-2
digest-locked package verification plus colon-safe identifiers, and source-
pinned module loading against poisoned bytecode caches plus package hygiene.

Writes `benchmark_results.json`.
"""
from __future__ import annotations

import copy
import dataclasses
from dataclasses import dataclass
from decimal import Decimal
import hashlib
import itertools
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
from typing import Any, Callable

from ceg_reference import (
    ACTIVE_PROSE_KEYS,
    atom_numeric_pool,
    bind_contrast_arms_conservatively,
    build_endpoint_phrases,
    build_entity_lexicon,
    compute_ceg_section_digests,
    ContextEvidenceGraph,
    ContextLatticeSpec,
    ContextSignature,
    EvidenceAtom,
    extract_mentions,
    IDENTIFIER_MASK_RE,
    ledger_sha256,
    load_json_single_read,
    load_source_pinned_module,
    MultiArmContrast,
    non_prose_surface_sha256,
    normalize_numeric_literal,
    parse_prose_quantity_bound,
    populate_claim_attestation,
    proposition_from_atom,
    ProspectiveFalsifierGate,
    ProvenancePin,
    QuantitySpec,
    record_to_envelope,
    REFERENCE_VERSION,
    render_atom_sentence,
    ReversiblePatchEvent,
    sha256_canonical,
    split_outcome_tuple,
    SynthesisClaimRecord,
    TUPLE_RE,
    TYPED_COORDINATES,
    TypedProposition,
    verify_atoms_bound_to_corpus,
    verify_bijective_patch_replay,
    verify_ceg_package,
    verify_chunk_or_package_hygiene,
    verify_non_prose_surface_lock,
    verify_prose_sentence,
    verify_typed_proposition_homomorphism,
)
from domain_vignettes import build_drug_development_vignette, build_sport_exercise_vignette

BENCHMARK_VERSION = "ceg-synthetic-benchmark-3.2"

ASCII_NUM_RE = re.compile(r"(?<![A-Za-z0-9.])[-+]?\d+(?:\.\d+)?")
MASK_ID_RE = IDENTIFIER_MASK_RE


def _require(condition: bool, message: str) -> None:
    """Explicit invariant check immune to `python3 -O` assert stripping."""
    if not condition:
        raise RuntimeError(message)


@dataclass
class BenchmarkCase:
    """One benchmark item: a synthesis record (and optionally a ledger) with known ground truth."""

    case_id: str
    stratum: str
    suite: str  # core | stress | residual
    is_adversarial: bool
    template: str  # template identifier, unique within the stratum
    description: str
    graph: ContextEvidenceGraph
    record: SynthesisClaimRecord
    initial_corpus: dict[str, Any] | None = None
    settled_corpus: dict[str, Any] | None = None
    patch_events: list[ReversiblePatchEvent] | None = None
    bind_atoms_to_corpus: bool = False


# ---------------------------------------------------------------------------
# Verification regimes
# ---------------------------------------------------------------------------

def evaluate_v0_citation_presence(case: BenchmarkCase) -> bool:
    """V0 accepts iff prose is non-empty and every cited identifier resolves."""
    rec = case.record
    if not rec.prose_sentences or all(not s.strip() for s in rec.prose_sentences):
        return False
    anchors = list(rec.supporting_tuples) + list(rec.counterevidence_tuples)
    if not anchors:
        return False
    for t in anchors:
        if t not in case.graph.atoms:
            return False
    for s in rec.prose_sentences:
        for ct in TUPLE_RE.findall(s):
            if ct not in case.graph.atoms:
                return False
    return True


def evaluate_v1_study_pool(case: BenchmarkCase) -> bool:
    """V1 adds study-wide ASCII numeral containment over the cited studies."""
    if not evaluate_v0_citation_presence(case):
        return False
    rec = case.record
    anchors = set(rec.supporting_tuples) | set(rec.counterevidence_tuples)
    cited_studies = {case.graph.atoms[t].study_id for t in anchors}
    study_pool: set[str] = set()
    for atom in case.graph.atoms.values():
        if atom.study_id in cited_studies:
            study_pool |= atom_numeric_pool(atom)
    for sent in rec.prose_sentences:
        masked = MASK_ID_RE.sub(" ", sent)
        for m in ASCII_NUM_RE.finditer(masked):
            if normalize_numeric_literal(m.group(0)) not in study_pool:
                return False
    return True


def _evaluate_prose(case: BenchmarkCase, policy: str) -> bool:
    if not evaluate_v0_citation_presence(case):
        return False
    rec = case.record
    anchor_set = set(rec.supporting_tuples) | set(rec.counterevidence_tuples)
    for sent in rec.prose_sentences:
        for ct in TUPLE_RE.findall(sent):
            if ct not in anchor_set:
                return False
        if verify_prose_sentence(sent, case.graph.atoms, case.graph.arm_aliases, pooled_numeral_policy=policy):
            return False
    return True


def evaluate_v2_prose_permissive(case: BenchmarkCase) -> bool:
    """V2: I1 + I3_tok + I3_clause, pooled numeral clauses permitted (union-pool check)."""
    return _evaluate_prose(case, "permit")


def evaluate_v2s_prose_strict(case: BenchmarkCase) -> bool:
    """V2s: I1 + I3_tok + I3_clause, pooled numeral clauses rejected."""
    return _evaluate_prose(case, "reject")


def evaluate_v3_full_ceg(case: BenchmarkCase) -> bool:
    """V3: every invariant, with the strict pooled policy."""
    if case.graph.verify_graph_integrity():
        return False
    if case.initial_corpus is not None and case.settled_corpus is not None and case.patch_events is not None:
        try:
            verify_bijective_patch_replay(case.initial_corpus, case.settled_corpus, case.patch_events)
        except ValueError:
            return False
        if case.bind_atoms_to_corpus and verify_atoms_bound_to_corpus(case.graph, case.settled_corpus):
            return False
    return case.graph.verify_synthesis_record(case.record).passed_all_invariants


VERIFIERS: dict[str, Callable[[BenchmarkCase], bool]] = {
    "V0_citation_presence": evaluate_v0_citation_presence,
    "V1_study_pool_numeric": evaluate_v1_study_pool,
    "V2_prose_pooled_permitted": evaluate_v2_prose_permissive,
    "V2s_prose_pooled_rejected": evaluate_v2s_prose_strict,
    "V3_full_ceg": evaluate_v3_full_ceg,
}


# ---------------------------------------------------------------------------
# Parametric study bundle
# ---------------------------------------------------------------------------

def _build_parametric_base_graph(idx: int) -> tuple[ContextEvidenceGraph, ContextSignature, dict[str, ContextSignature]]:
    """A clean four-study bundle over the translational lattice, parameterized by `idx`."""
    base = build_drug_development_vignette()
    lattice = base.graph.lattice
    target_ctx = base.target_context

    ctx_2d = ContextSignature(
        {
            "model_system": "immortalized_2d_line",
            "disease_chronicity": "acute_cytokine_pulse",
            "dosing_exposure": "supraphysiologic_bath_10uM",
            "readout_maturity": "transcript_surrogate",
        }
    )
    ctx_murine = ContextSignature(
        {
            "model_system": "murine_bleomycin_invivo",
            "disease_chronicity": "subacute_bleomycin_day21",
            "dosing_exposure": "pk_matched_unbound_exposure",
            "readout_maturity": "extracellular_matrix_biomarker",
        }
    )
    ctx_pcls = ContextSignature(
        {
            "model_system": "human_pcls_explant",
            "disease_chronicity": "established_chronic_fibrosis",
            "dosing_exposure": "pk_matched_unbound_exposure",
            "readout_maturity": "extracellular_matrix_biomarker",
        }
    )

    sid_clin = f"C{100 + idx}"
    sid_pcls = f"P{100 + idx}"
    sid_mur = f"M{100 + idx}"
    sid_2d = f"L{100 + idx}"

    fvc_treat = f"-{20 + (idx % 15)}.5"
    fvc_plac = f"-{75 + (idx % 20)}.0"
    svc_treat = f"-{22 + (idx % 15)}.0"
    svc_plac = f"-{70 + (idx % 20)}.5"
    ae_treat = f"{14 + (idx % 9)}.2"
    ae_plac = f"{6 + (idx % 5)}.1"
    proc3_treat = f"{24 + (idx % 11)}.4"
    proc3_veh = f"{45 + (idx % 13)}.8"
    viab_treat = f"{94 + (idx % 5)}.2"
    viab_veh = f"{93 + (idx % 5)}.8"
    hyp_treat = f"{130 + idx}.5"
    n_clin = 60 + (idx % 40)
    n_pcls = 12 + (idx % 12)

    studies = {sid_clin: target_ctx, sid_pcls: ctx_pcls, sid_mur: ctx_murine, sid_2d: ctx_2d}
    study_vars = {
        sid_clin: {f"VAR:{sid_clin}:Drug", f"VAR:{sid_clin}:Placebo"},
        sid_pcls: {f"VAR:{sid_pcls}:Drug", f"VAR:{sid_pcls}:Vehicle"},
        sid_mur: {f"VAR:{sid_mur}:Drug", f"VAR:{sid_mur}:ActiveComp", f"VAR:{sid_mur}:Vehicle"},
        sid_2d: {f"VAR:{sid_2d}:DrugBath", f"VAR:{sid_2d}:DMSO"},
    }
    arm_aliases = {
        f"VAR:{sid_clin}:Drug": ("treated arm", "drug arm"),
        f"VAR:{sid_clin}:Placebo": ("placebo arm", "placebo"),
        f"VAR:{sid_pcls}:Drug": ("treated slices", "drug slices"),
        f"VAR:{sid_pcls}:Vehicle": ("vehicle slices", "vehicle"),
    }

    def clin_atom(o: str, endpoint: str, fv: str, cv: str, unit: str, dim: str, stat: str, p: str, pol: str, loc: str) -> EvidenceAtom:
        return EvidenceAtom(
            atom_id=f"ATOM:{sid_clin}:{o}",
            outcome_tuple=f"{sid_clin}:{o}",
            study_id=sid_clin,
            endpoint=endpoint,
            context=target_ctx,
            contrast=MultiArmContrast((f"VAR:{sid_clin}:Drug",), (f"VAR:{sid_clin}:Placebo",), "pairwise"),
            focal_quantity=QuantitySpec(kind="point_level", relation="=", value=fv, unit=unit, unit_dimension=dim),
            comparator_quantity=QuantitySpec(kind="point_level", relation="=", value=cv, unit=unit, unit_dimension=dim),
            statistic_kind=stat,  # type: ignore[arg-type]
            sample_size=n_clin,
            p_value_or_interval=p,
            polarity=pol,  # type: ignore[arg-type]
            provenance=ProvenancePin(f"DOC:{sid_clin}", loc, endpoint, False, True),
        )

    def pcls_atom(o: str, endpoint: str, fv: str, cv: str, unit: str, dim: str, p: str, pol: str) -> EvidenceAtom:
        return EvidenceAtom(
            atom_id=f"ATOM:{sid_pcls}:{o}",
            outcome_tuple=f"{sid_pcls}:{o}",
            study_id=sid_pcls,
            endpoint=endpoint,
            context=ctx_pcls,
            contrast=MultiArmContrast((f"VAR:{sid_pcls}:Drug",), (f"VAR:{sid_pcls}:Vehicle",), "pairwise"),
            focal_quantity=QuantitySpec(kind="point_level", relation="=", value=fv, unit=unit, unit_dimension=dim),
            comparator_quantity=QuantitySpec(kind="point_level", relation="=", value=cv, unit=unit, unit_dimension=dim),
            statistic_kind="mean",
            sample_size=n_pcls,
            p_value_or_interval=p,
            polarity=pol,  # type: ignore[arg-type]
            provenance=ProvenancePin(f"DOC:{sid_pcls}", "Table 1", endpoint, False, True),
        )

    atoms = {
        f"{sid_clin}:O1": clin_atom("O1", "12-week forced vital capacity change", fvc_treat, fvc_plac, "mL", "volume", "mean", "p=0.01", "favorable", "Table 2"),
        f"{sid_clin}:O2": clin_atom("O2", "12-week slow vital capacity secondary change", svc_treat, svc_plac, "mL", "volume", "mean", "p=0.02", "favorable", "Table 3"),
        f"{sid_clin}:O3": clin_atom("O3", "Gastrointestinal adverse event incidence", ae_treat, ae_plac, "%", "percentage", "proportion", "p=0.25", "unfavorable", "Table 4"),
        f"{sid_pcls}:O1": pcls_atom("O1", "Secreted PRO-C3 neo-epitope concentration", proc3_treat, proc3_veh, "ng/mL", "mass_conc", "p=0.003", "favorable"),
        f"{sid_pcls}:O2": pcls_atom("O2", "Slice ATP viability", viab_treat, viab_veh, "%", "percentage", "p=0.70", "neutral_or_mixed"),
        f"{sid_mur}:O1": EvidenceAtom(
            atom_id=f"ATOM:{sid_mur}:O1",
            outcome_tuple=f"{sid_mur}:O1",
            study_id=sid_mur,
            endpoint="Total lung hydroxyproline content",
            context=ctx_murine,
            contrast=MultiArmContrast(
                (f"VAR:{sid_mur}:Drug",),
                (f"VAR:{sid_mur}:ActiveComp", f"VAR:{sid_mur}:Vehicle"),
                "multi_arm_joint",
            ),
            focal_quantity=QuantitySpec(kind="point_level", relation="=", value=hyp_treat, unit="ug/lung", unit_dimension="mass_organ"),
            comparator_quantity=QuantitySpec(kind="interval_bound", lower="180.0", upper="220.0", unit="ug/lung", unit_dimension="mass_organ"),
            statistic_kind="mean",
            sample_size=30,
            p_value_or_interval="p=0.006",
            polarity="favorable",
            provenance=ProvenancePin(f"DOC:{sid_mur}", "Table 2", "Hydroxyproline", False, True),
        ),
        f"{sid_mur}:O2": EvidenceAtom(
            atom_id=f"ATOM:{sid_mur}:O2",
            outcome_tuple=f"{sid_mur}:O2",
            study_id=sid_mur,
            endpoint="Kaplan-Meier survival censor",
            context=ctx_murine,
            contrast=MultiArmContrast((f"VAR:{sid_mur}:Drug",), (f"VAR:{sid_mur}:Vehicle",), "pairwise"),
            focal_quantity=QuantitySpec(kind="censored_observation", relation=">", value="28", unit="days", unit_dimension="duration"),
            comparator_quantity=None,
            statistic_kind="median",
            sample_size=24,
            p_value_or_interval="p=0.03",
            polarity="favorable",
            provenance=ProvenancePin(
                f"DOC:{sid_mur}",
                "Fig 2C",
                "KM survival >28d",
                requires_figure_digitization=(idx % 2 == 0),
                primary_pdf_verified=(idx % 2 == 0),
            ),
        ),
        f"{sid_2d}:O1": EvidenceAtom(
            atom_id=f"ATOM:{sid_2d}:O1",
            outcome_tuple=f"{sid_2d}:O1",
            study_id=sid_2d,
            endpoint="COL1A1 mRNA relative reduction",
            context=ctx_2d,
            contrast=MultiArmContrast((f"VAR:{sid_2d}:DrugBath",), (f"VAR:{sid_2d}:DMSO",), "pairwise"),
            focal_quantity=QuantitySpec(
                kind="relative_change",
                relation="=",
                value=f"{65 + (idx % 20)}.0",
                unit="%",
                unit_dimension="percentage",
                change_direction="decrease",
            ),
            comparator_quantity=None,
            statistic_kind="fold_change",
            sample_size=6,
            p_value_or_interval="p=0.004",
            polarity="favorable",
            provenance=ProvenancePin(f"DOC:{sid_2d}", "Table 1", "COL1A1 mRNA", False, True),
        ),
    }
    raw_arms = {
        t: tuple(a.contrast.focal_arms) + tuple(a.contrast.comparator_arms) for t, a in atoms.items()
    }
    graph = ContextEvidenceGraph(
        lattice=lattice,
        studies=studies,
        study_variables=study_vars,
        arm_aliases=arm_aliases,
        atoms=atoms,
        raw_mentioned_arms_by_tuple=raw_arms,
    )
    ctx_map = {"clin": target_ctx, "pcls": ctx_pcls, "mur": ctx_murine, "2d": ctx_2d}
    return graph, target_ctx, ctx_map


def _rec(
    rid: str,
    target: ContextSignature,
    support: tuple[str, ...],
    sentences: tuple[str, ...],
    *,
    polarity: str = "SCOPING_QUALIFICATION",
    counter: tuple[str, ...] = (),
    props: tuple[TypedProposition, ...] = (),
    gate: ProspectiveFalsifierGate | None = None,
    prom: bool = False,
) -> SynthesisClaimRecord:
    return SynthesisClaimRecord(
        record_id=rid,
        polarity=polarity,  # type: ignore[arg-type]
        target_context=target,
        supporting_tuples=support,
        counterevidence_tuples=counter,
        prose_sentences=sentences,
        typed_propositions=props,
        prospective_gate=gate,
        requests_direct_quantitative_promotion=prom,
    )


def _full_gate(gid: str) -> ProspectiveFalsifierGate:
    return ProspectiveFalsifierGate(
        gate_id=gid,
        unresolved_context_axes=("disease_chronicity", "dosing_exposure", "model_system", "readout_maturity"),
        head_to_head_arms=("Oral Drug", "Placebo"),
        primary_readout_and_unit="FVC (mL) and serum PRO-C3 (ng/mL)",
        quantitative_pass_threshold=">=40 mL FVC preservation",
    )


def _unsigned(v: str) -> str:
    return v.lstrip("-")


# ---------------------------------------------------------------------------
# Core suite: P1..P4 and F1..F10 (the 800 cases of the first release)
# ---------------------------------------------------------------------------

def _core_cases() -> list[BenchmarkCase]:
    cases: list[BenchmarkCase] = []

    for i in range(50):
        g, target_ctx, _ = _build_parametric_base_graph(i)
        sc, sp, sm = f"C{100 + i}", f"P{100 + i}", f"M{100 + i}"
        a_c1, a_c3 = g.atoms[f"{sc}:O1"], g.atoms[f"{sc}:O3"]
        a_p1, a_p2, a_m1 = g.atoms[f"{sp}:O1"], g.atoms[f"{sp}:O2"], g.atoms[f"{sm}:O1"]

        cases.append(BenchmarkCase(
            f"P1-{i:03d}", "P1_valid_direct_promotion", "core", False, "direct",
            "Matched-context clinical outcome with exact typed proposition and separate counter-evidence",
            g,
            _rec(
                f"REC-P1-{i:03d}", target_ctx, (f"{sc}:O1",),
                (
                    f"In {sc}:O1 the treated arm showed {a_c1.focal_quantity.value} mL change whereas the placebo arm showed {a_c1.comparator_quantity.value} mL across {a_c1.sample_size} patients.",
                    f"In {sc}:O3 gastrointestinal events were {a_c3.focal_quantity.value}% in the treated arm versus {a_c3.comparator_quantity.value}% in the placebo arm.",
                ),
                polarity="SUPPORTS", counter=(f"{sc}:O3",),
                props=(proposition_from_atom(a_c1, f"PROP-P1-{i:03d}"),), prom=True,
            ),
        ))
        cases.append(BenchmarkCase(
            f"P2-{i:03d}", "P2_valid_bounded_bridge", "core", False, "bridge",
            "Human PCLS + murine 3-arm bridge held behind a complete prospective falsifier gate",
            g,
            _rec(
                f"REC-P2-{i:03d}", target_ctx, (f"{sp}:O1", f"{sm}:O1"),
                (
                    f"In {sp}:O1 treated slices reduced PRO-C3 from {a_p1.comparator_quantity.value} ng/mL in vehicle slices to {a_p1.focal_quantity.value} ng/mL, while in {sm}:O1 lung hydroxyproline was {a_m1.focal_quantity.value} ug/lung.",
                ),
                polarity="CANDIDATE_UPGRADE",
                props=(proposition_from_atom(a_p1, f"PROP-P2-{i:03d}"),),
                gate=_full_gate(f"GATE-P2-{i:03d}"),
            ),
        ))
        cases.append(BenchmarkCase(
            f"P3-{i:03d}", "P3_valid_multituple_scoping", "core", False, "respectively",
            "Two-tuple 'respectively' sentence with per-tuple quantities and units",
            g,
            _rec(
                f"REC-P3-{i:03d}", target_ctx, (f"{sp}:O1", f"{sp}:O2"),
                (f"{sp}:O1 and {sp}:O2 measured {a_p1.focal_quantity.value} ng/mL and {a_p2.focal_quantity.value}%, respectively.",),
            ),
        ))
        cases.append(BenchmarkCase(
            f"P4-{i:03d}", "P4_valid_counterevidence_boundary", "core", False, "counter",
            "Valid counter-evidence safety scoping record",
            g,
            _rec(
                f"REC-P4-{i:03d}", target_ctx, (),
                (f"In {sc}:O3 adverse gastrointestinal events increased from {a_c3.comparator_quantity.value}% in the placebo arm to {a_c3.focal_quantity.value}% in the treated arm.",),
                polarity="FALSIFIED_GRAVEYARD", counter=(f"{sc}:O3",),
            ),
        ))

    lexical_smugglers = ("tripled", "seventy percent", "forty percent", "⅓", "¾", "five-fold")
    invisible_chars = ("​", "⁠", "﻿", "‍", "‮", "­")
    connectors = ("whereas", "while", "in contrast to", "versus", ";", "and")

    for i in range(60):
        g, target_ctx, ctx_map = _build_parametric_base_graph(200 + i)
        sc, sp, sm, s2d = f"C{300 + i}", f"P{300 + i}", f"M{300 + i}", f"L{300 + i}"
        a_c1, a_c2, a_c3 = g.atoms[f"{sc}:O1"], g.atoms[f"{sc}:O2"], g.atoms[f"{sc}:O3"]
        a_p1, a_p2 = g.atoms[f"{sp}:O1"], g.atoms[f"{sp}:O2"]
        a_m2, a_2d = g.atoms[f"{sm}:O2"], g.atoms[f"{s2d}:O1"]

        cases.append(BenchmarkCase(
            f"F1-{i:03d}", "F1_fabricated_numeric_literal", "core", True, "fabricated",
            "Plausible fabricated numeral absent from the cited study",
            g,
            _rec(f"BAD-F1-{i:03d}", target_ctx, (f"{sp}:O1",), (f"In {sp}:O1 secreted PRO-C3 fell to {51.3 + i:.1f} ng/mL.",)),
        ))
        cases.append(BenchmarkCase(
            f"F2-{i:03d}", "F2_intrastudy_cross_outcome_bleed", "core", True, "bleed",
            "Cites outcome O1 while asserting a numeral of sibling outcome O2 of the same study",
            g,
            _rec(f"BAD-F2-{i:03d}", target_ctx, (f"{sp}:O1",), (f"In {sp}:O1 secreted PRO-C3 concentration averaged {a_p2.focal_quantity.value} ng/mL.",)),
        ))

        if i < 30:
            smug = lexical_smugglers[i % len(lexical_smugglers)]
            f3_sent = f"In {sp}:O1 secreted PRO-C3 changed by {smug} relative to control."
            f3_tmpl = f"lexical:{smug}"
        else:
            inv = invisible_chars[i % len(invisible_chars)]
            v_str = a_p1.focal_quantity.value or "29.4"
            f3_sent = f"In {sp}:O1 secreted PRO-C3 averaged {v_str[:1] + inv + v_str[1:]} ng/mL."
            f3_tmpl = f"invisible:U+{ord(inv):04X}"
        cases.append(BenchmarkCase(
            f"F3-{i:03d}", "F3_lexical_and_unicode_evasion", "core", True, f3_tmpl,
            "Word/multiplier numeral or zero-width/bidi split numeral",
            g,
            _rec(f"BAD-F3-{i:03d}", target_ctx, (f"{sp}:O1",), (f3_sent,)),
        ))

        if i % 4 == 0:
            f4_sent = f"{sp}:O1 and {sp}:O2 measured {a_p2.focal_quantity.value}% and {a_p1.focal_quantity.value} ng/mL, respectively."
            f4_tmpl = "respectively"
        else:
            conn = connectors[i % len(connectors)]
            sep = ";" if conn == ";" else f" {conn}"
            f4_sent = f"In {sp}:O1 PRO-C3 was {a_p2.focal_quantity.value} ng/mL{sep} in {sp}:O2 slice viability was {a_p1.focal_quantity.value}%."
            f4_tmpl = f"connector:{conn}"
        cases.append(BenchmarkCase(
            f"F4-{i:03d}", "F4_multituple_clause_swap", "core", True, f4_tmpl,
            "Two-tuple sentence swapping values across sibling outcome clauses",
            g,
            _rec(f"BAD-F4-{i:03d}", target_ctx, (f"{sp}:O1", f"{sp}:O2"), (f4_sent,)),
        ))

        mode_f5 = i % 3
        if mode_f5 == 0:
            f5_sent = f"In {sp}:O1 treated slices reached {a_p1.comparator_quantity.value} ng/mL whereas vehicle slices had {a_p1.focal_quantity.value} ng/mL."
        elif mode_f5 == 1:
            f5_sent = f"In {sp}:O1 secreted PRO-C3 moved from {a_p1.focal_quantity.value} ng/mL to {a_p1.comparator_quantity.value} ng/mL."
        else:
            f5_sent = f"In {sp}:O1 treatment increased PRO-C3 to {a_p1.focal_quantity.value} ng/mL compared with {a_p1.comparator_quantity.value} ng/mL."
        cases.append(BenchmarkCase(
            f"F5-{i:03d}", "F5_arm_swap_and_polarity_reversal", "core", True, ("arm_swap", "from_to", "verb")[mode_f5],
            "True numbers retained while inverting arm assignment or effect direction",
            g,
            _rec(f"BAD-F5-{i:03d}", target_ctx, (f"{sp}:O1",), (f5_sent,)),
        ))

        misbound = dataclasses.replace(
            proposition_from_atom(a_c1, f"PROP-F6-{i:03d}"),
            claimed_quantity=a_c2.focal_quantity,
            claimed_comparator_quantity=a_c2.comparator_quantity,
        )
        cases.append(BenchmarkCase(
            f"F6-{i:03d}", "F6_pooled_misbinding_in_prose_and_proposition", "core", True, "pooled_both",
            "Pooled citation (O1, O2) binds O2's quantity to O1's endpoint in the prose AND in the typed proposition",
            g,
            _rec(
                f"BAD-F6-{i:03d}", target_ctx, (f"{sc}:O1", f"{sc}:O2"),
                (f"Mean 12-week forced vital capacity change was {a_c2.focal_quantity.value} mL versus {a_c2.comparator_quantity.value} mL ({sc}:O1, {sc}:O2).",),
                polarity="SUPPORTS", props=(misbound,), prom=True,
            ),
        ))

        g_f7 = copy.deepcopy(g)
        if i % 2 == 0:
            bad_prop_f7 = dataclasses.replace(
                proposition_from_atom(a_2d, f"PROP-F7-{i:03d}"),
                claimed_quantity=QuantitySpec(kind="point_level", relation="=", value=a_2d.focal_quantity.value, unit="%", unit_dimension="percentage"),
            )
            rec_f7 = _rec(
                f"BAD-F7-{i:03d}", ctx_map["2d"], (f"{s2d}:O1",),
                (f"In {s2d}:O1 COL1A1 decreased by {a_2d.focal_quantity.value}%.",),
                props=(bad_prop_f7,),
            )
            f7_tmpl = "scale_conflation"
        else:
            old_m1 = g_f7.atoms[f"{sm}:O1"]
            g_f7.atoms[f"{sm}:O1"] = dataclasses.replace(
                old_m1,
                contrast=MultiArmContrast(old_m1.contrast.focal_arms, (f"VAR:{sm}:Vehicle",), "pairwise"),
            )
            rec_f7 = _rec(
                f"BAD-F7-{i:03d}", ctx_map["mur"], (f"{sm}:O1",),
                (f"In {sm}:O1 lung hydroxyproline was {old_m1.focal_quantity.value} ug/lung.",),
            )
            f7_tmpl = "arm_truncation"
        cases.append(BenchmarkCase(
            f"F7-{i:03d}", "F7_scale_conflation_and_multiarm_truncation", "core", True, f7_tmpl,
            "Relative-vs-level scale conflation in a proposition, or a silently dropped comparator arm in the graph",
            g_f7, rec_f7,
        ))

        if i % 2 == 0:
            rec_f8 = _rec(
                f"BAD-F8-{i:03d}", target_ctx, (f"{s2d}:O1",),
                (f"In {s2d}:O1 COL1A1 decreased by {a_2d.focal_quantity.value}%.",),
                polarity="CANDIDATE_UPGRADE", gate=_full_gate("G-INSUFF"),
            )
            f8_tmpl = "inadmissible_upgrade"
        else:
            rec_f8 = _rec(
                f"BAD-F8-{i:03d}", target_ctx, (f"{sp}:O1",),
                (f"In {sp}:O1 treated slices reduced PRO-C3 from {a_p1.comparator_quantity.value} ng/mL to {a_p1.focal_quantity.value} ng/mL.",),
                polarity="SUPPORTS", props=(proposition_from_atom(a_p1, f"PROP-F8-{i:03d}"),), prom=True,
            )
            f8_tmpl = "unbridged_promotion"
        cases.append(BenchmarkCase(
            f"F8-{i:03d}", "F8_context_poset_transfer_overclaim", "core", True, f8_tmpl,
            "Direct promotion or upgrade across unresolved or inadmissible context axes",
            g, rec_f8,
        ))

        cases.append(BenchmarkCase(
            f"F9-{i:03d}", "F9_undigitized_or_unverified_promotion", "core", True,
            "undigitized" if (200 + i) % 2 == 0 else "unverified",
            "Direct promotion of an undigitized figure read-out or an unverified primary source",
            g,
            _rec(
                f"BAD-F9-{i:03d}", ctx_map["mur"], (f"{sm}:O2",),
                (f"In {sm}:O2 survival exceeded 28 days across 24 mice.",),
                polarity="SUPPORTS", props=(proposition_from_atom(a_m2, f"PROP-F9-{i:03d}"),), prom=True,
            ),
        ))

        if i < 30:
            cases.append(BenchmarkCase(
                f"F10-{i:03d}", "F10_support_counter_overlap_and_ledger_tamper", "core", True, "overlap",
                "Outcome listed as both support and counter-evidence (I6)",
                g,
                _rec(
                    f"BAD-F10-{i:03d}", target_ctx, (f"{sc}:O3",),
                    (f"In {sc}:O3 gastrointestinal events were {a_c3.focal_quantity.value}% versus {a_c3.comparator_quantity.value}%.",),
                    counter=(f"{sc}:O3",),
                ),
            ))
        else:
            init_doc = {"studies": [{"study_id": sc, "val": "10.0"}]}
            settled_doc = {"studies": [{"study_id": sc, "val": "99.9"}]}
            forged_ev = ReversiblePatchEvent(
                event_id=f"EV-TAMPER-{i:03d}",
                study_id=sc,
                json_pointer="/studies/0/val",
                old_value="10.0",
                new_value="20.0",
                rationale="Legitimate patch whose settled corpus was silently altered afterwards",
                study_sha256_before=sha256_canonical(init_doc["studies"][0]),
                study_sha256_after=sha256_canonical({"study_id": sc, "val": "20.0"}),
            )
            cases.append(BenchmarkCase(
                f"F10-{i:03d}", "F10_support_counter_overlap_and_ledger_tamper", "core", True, "settled_drift",
                "Silent post-curation corpus drift (I2)",
                g,
                _rec(
                    f"BAD-F10-{i:03d}", target_ctx, (f"{sp}:O1",),
                    (f"In {sp}:O1 treated slices reduced PRO-C3 from {a_p1.comparator_quantity.value} ng/mL to {a_p1.focal_quantity.value} ng/mL.",),
                ),
                initial_corpus=init_doc, settled_corpus=settled_doc, patch_events=[forged_ev],
            ))
    return cases


# ---------------------------------------------------------------------------
# Stress suite: F11..F19 and P5..P8
# ---------------------------------------------------------------------------

def _ledger_tamper_case(i: int) -> BenchmarkCase:
    """F19: ledger or atom-binding defects built on the worked vignette."""
    bundle = build_drug_development_vignette()
    g = bundle.graph
    d0, dm, events = bundle.initial_corpus, bundle.settled_corpus, list(bundle.patch_events)
    benign = _rec(
        f"BAD-F19-{i:03d}", bundle.target_context, ("S104:O3",),
        ("In S104:O3 Grade >=2 gastrointestinal events occurred in 18.8% of the NX-410 40 mg BID arm versus 9.4% on placebo.",),
    )
    mode = i % 5
    bind = False
    if mode == 0:
        k = (i // 5) % len(events)
        ptr = events[k].json_pointer
        st_idx, oc_idx = int(ptr.split("/")[2]), int(ptr.split("/")[4])
        n_out = len(d0["studies"][st_idx]["outcomes"])
        parts = ptr.split("/")
        parts[4] = str(oc_idx - n_out)
        events[k] = dataclasses.replace(events[k], json_pointer="/".join(parts))
        tmpl, desc = "pointer_negative_index", "Ledger pointer rewritten with a negative array index"
    elif mode == 1:
        k = (i // 5) % len(events)
        parts = events[k].json_pointer.split("/")
        if (i // 15) % 2 == 0:
            parts[2] = "0" + parts[2]
        else:
            parts.insert(1, "")
        events[k] = dataclasses.replace(events[k], json_pointer="/".join(parts))
        tmpl, desc = "pointer_noncanonical", "Ledger pointer rewritten with a leading-zero index or an empty leading token"
    elif mode == 2:
        a_idx, b_idx, field_ptr, new_val = (
            (0, 1, "/outcomes/1/requires_figure_digitization", True),
            (2, 3, "/outcomes/0/unit", "L"),
            (4, 2, "/outcomes/0/focal_value", "19.4"),
        )[(i // 5) % 3]
        h_a = sha256_canonical(d0["studies"][a_idx])
        dm = copy.deepcopy(d0)
        tokens = field_ptr.strip("/").split("/")
        node = dm["studies"][b_idx]
        for tok in tokens[:-1]:
            node = node[int(tok)] if isinstance(node, list) else node[tok]
        old_val = node[tokens[-1]]
        node[tokens[-1]] = new_val
        events = [ReversiblePatchEvent(
            event_id="PATCH-MISATTRIBUTED",
            study_id=d0["studies"][a_idx]["study_id"],
            json_pointer=f"/studies/{b_idx}{field_ptr}",
            old_value=old_val, new_value=new_val,
            rationale="Edit to one study recorded under another study's digest guard",
            study_sha256_before=h_a, study_sha256_after=h_a,
        )]
        tmpl, desc = "cross_study_attribution", "Event guarded by one study's digest edits a different study"
    elif mode == 3:
        drifted = f"{19 + (i // 5) % 10}.4"
        g = copy.deepcopy(g)
        a = g.atoms["S103:O1"]
        g.atoms["S103:O1"] = dataclasses.replace(a, focal_quantity=dataclasses.replace(a.focal_quantity, value=drifted))
        benign = _rec(
            f"BAD-F19-{i:03d}", bundle.target_context, ("S103:O1",),
            (f"In S103:O1 treated slices reduced PRO-C3 from 48.6 ng/mL in vehicle slices to {drifted} ng/mL.",),
        )
        bind = True
        tmpl, desc = "atom_drift", "Atom value differs from the settled corpus while the ledger still verifies"
    else:
        st_idx = (i // 5) % len(dm["studies"])
        h = sha256_canonical(dm["studies"][st_idx])
        unit = dm["studies"][st_idx]["outcomes"][0].get("unit", dm["studies"][st_idx]["outcomes"][0].get("endpoint"))
        key = "unit" if "unit" in dm["studies"][st_idx]["outcomes"][0] else "endpoint"
        events = events + [ReversiblePatchEvent(
            event_id="PATCH-NOOP",
            study_id=dm["studies"][st_idx]["study_id"],
            json_pointer=f"/studies/{st_idx}/outcomes/0/{key}",
            old_value=unit, new_value=unit,
            rationale="Padding event that changes nothing",
            study_sha256_before=h, study_sha256_after=h,
        )]
        tmpl, desc = "noop_event", "No-op event appended to the ledger"
    return BenchmarkCase(
        f"F19-{i:03d}", "F19_ledger_and_binding_tamper", "stress", True, tmpl, desc,
        g, benign, initial_corpus=d0, settled_corpus=dm, patch_events=events, bind_atoms_to_corpus=bind,
    )


def _stress_cases() -> list[BenchmarkCase]:
    cases: list[BenchmarkCase] = []

    for i in range(60):
        g, target_ctx, ctx_map = _build_parametric_base_graph(400 + i)
        sc, sp, sm = f"C{500 + i}", f"P{500 + i}", f"M{500 + i}"
        a_c1, a_c2, a_c3 = g.atoms[f"{sc}:O1"], g.atoms[f"{sc}:O2"], g.atoms[f"{sc}:O3"]
        a_p1, a_p2, a_m1 = g.atoms[f"{sp}:O1"], g.atoms[f"{sp}:O2"], g.atoms[f"{sm}:O1"]
        fvc_t, fvc_p = a_c1.focal_quantity.value, a_c1.comparator_quantity.value
        svc_t, svc_p = a_c2.focal_quantity.value, a_c2.comparator_quantity.value
        p1f, p1c = a_p1.focal_quantity.value, a_p1.comparator_quantity.value
        p2f, p2c = a_p2.focal_quantity.value, a_p2.comparator_quantity.value
        both_true = (proposition_from_atom(a_c1, f"PROP-A-{i:03d}"), proposition_from_atom(a_c2, f"PROP-B-{i:03d}"))
        valid_c1 = f"In {sc}:O1 the treated arm showed {fvc_t} mL change whereas the placebo arm showed {fvc_p} mL."

        # F11: the prose misbinds, the typed layer is true (or absent)
        m = i % 3
        if m == 0:
            rec = _rec(f"BAD-F11-{i:03d}", target_ctx, (f"{sc}:O1", f"{sc}:O2"),
                       (f"Mean 12-week forced vital capacity change was {svc_t} mL versus {svc_p} mL ({sc}:O1, {sc}:O2).",),
                       polarity="SUPPORTS", props=both_true, prom=True)
            tmpl = "trailing_pooled_true_propositions"
        elif m == 1:
            rec = _rec(f"BAD-F11-{i:03d}", target_ctx, (f"{sc}:O1", f"{sc}:O2"),
                       (f"Mean 12-week forced vital capacity change was {svc_t} mL versus {svc_p} mL ({sc}:O1, {sc}:O2).",),
                       polarity="SUPPORTS")
            tmpl = "trailing_pooled_no_proposition"
        else:
            rec = _rec(f"BAD-F11-{i:03d}", target_ctx, (f"{sc}:O1", f"{sc}:O2"),
                       (f"Across the spirometric endpoints ({sc}:O1, {sc}:O2), forced vital capacity change was {svc_t} mL versus {svc_p} mL.",),
                       polarity="SUPPORTS", props=both_true, prom=True)
            tmpl = "leading_pooled_true_propositions"
        cases.append(BenchmarkCase(f"F11-{i:03d}", "F11_pooled_prose_misbinding_true_propositions", "stress", True, tmpl,
                                   "Pooled citation misbinds O2's numbers to O1's endpoint while every typed proposition is true or absent", g, rec))

        # F12: valid prose, defective typed layer
        if m == 0:
            rec = _rec(f"BAD-F12-{i:03d}", target_ctx, (f"{sc}:O1",), (valid_c1,), polarity="SUPPORTS",
                       counter=(f"{sc}:O3",), props=(proposition_from_atom(a_c3, f"PROP-F12-{i:03d}"),), prom=True)
            tmpl = "promoted_tuple_uncovered"
        elif m == 1:
            rec = _rec(f"BAD-F12-{i:03d}", target_ctx, (f"{sc}:O1",), (valid_c1,), polarity="SUPPORTS",
                       props=(dataclasses.replace(proposition_from_atom(a_c1, f"PROP-F12-{i:03d}"), claimed_sample_size=None),), prom=True)
            tmpl = "sample_size_omitted"
        else:
            rec = _rec(f"BAD-F12-{i:03d}", target_ctx, (f"{sc}:O1",), (valid_c1,), polarity="SUPPORTS",
                       props=(dataclasses.replace(proposition_from_atom(a_c1, f"PROP-F12-{i:03d}"), claimed_comparator_quantity=a_c2.comparator_quantity),), prom=True)
            tmpl = "comparator_quantity_misbound"
        cases.append(BenchmarkCase(f"F12-{i:03d}", "F12_typed_coverage_and_comparator_gaps", "stress", True, tmpl,
                                   "Promotion whose typed layer omits the promoted tuple, omits the sample size, or misbinds the comparator quantity", g, rec))

        # F13: numeral stranded in a clause that cites nothing
        if m == 0:
            sent = f"In {sp}:O1 PRO-C3 was {p1f} ng/mL versus {p2c} ng/mL in vehicle slices, whereas in {sp}:O2 slice viability was {p2f}%."
            tmpl = "versus_orphan"
        elif m == 1:
            # The first clause is a complete, true contrast; only the uncited middle fragment is false (release 3.2).
            sent = f"In {sp}:O1 PRO-C3 was {p1f} ng/mL in treated slices versus {p1c} ng/mL in vehicle slices; the baseline level was {p2c} ng/mL; in {sp}:O2 slice viability was {p2f}%."
            tmpl = "semicolon_orphan"
        else:
            sent = f"In {sp}:O1 PRO-C3 was {p1f} ng/mL compared with {p2f} ng/mL in vehicle slices, while in {sp}:O2 slice viability was {p2c}%."
            tmpl = "compared_with_orphan"
        cases.append(BenchmarkCase(f"F13-{i:03d}", "F13_orphan_clause_bleed", "stress", True, tmpl,
                                   "Sibling outcome's numeral placed in a clause fragment that carries no citation", g,
                                   _rec(f"BAD-F13-{i:03d}", target_ctx, (f"{sp}:O1", f"{sp}:O2"), (sent,))))

        # F14: 'respectively' misalignment
        if m == 0:
            sent = f"At week 12, {sc}:O1 and {sc}:O2 changed by {svc_t} mL and {fvc_t} mL, respectively."
            support, tmpl = (f"{sc}:O1", f"{sc}:O2"), "leading_shared_numeral"
        elif m == 1:
            sent = f"{sp}:O1 and {sp}:O2 measured {p1f} versus {p2c} and {p2f} versus {p1c}, respectively."
            support, tmpl = (f"{sp}:O1", f"{sp}:O2"), "comparators_swapped_in_pairs"
        else:
            sent = f"{sp}:O1 and {sp}:O2 increased to {p1f} ng/mL and reached {p2f}%, respectively."
            support, tmpl = (f"{sp}:O1", f"{sp}:O2"), "polarity_inside_list"
        cases.append(BenchmarkCase(f"F14-{i:03d}", "F14_respectively_misalignment", "stress", True, tmpl,
                                   "Index-aligned list whose values are shifted, cross-paired, or carry an inverted verb", g,
                                   _rec(f"BAD-F14-{i:03d}", target_ctx, support, (sent,))))

        # F15: unit-scale swaps
        q = i % 4
        if q == 0:
            rec = _rec(f"BAD-F15-{i:03d}", target_ctx, (f"{sc}:O1",),
                       (f"In {sc}:O1 the treated arm showed {fvc_t} L change whereas the placebo arm showed {fvc_p} L.",))
            tmpl = "mL_to_L"
        elif q == 1:
            rec = _rec(f"BAD-F15-{i:03d}", ctx_map["mur"], (f"{sm}:O2",), (f"In {sm}:O2 survival exceeded 28 weeks.",))
            tmpl = "days_to_weeks"
        elif q == 2:
            rec = _rec(f"BAD-F15-{i:03d}", target_ctx, (f"{sp}:O1",),
                       (f"In {sp}:O1 treated slices reduced PRO-C3 from {p1c} mg/mL in vehicle slices to {p1f} mg/mL.",))
            tmpl = "ng_to_mg_per_mL"
        else:
            rec = _rec(f"BAD-F15-{i:03d}", ctx_map["mur"], (f"{sm}:O1",),
                       (f"In {sm}:O1 lung hydroxyproline was {a_m1.focal_quantity.value} mg/lung.",))
            tmpl = "ug_to_mg_per_lung"
        cases.append(BenchmarkCase(f"F15-{i:03d}", "F15_unit_scale_swap", "stress", True, tmpl,
                                   "True numeral carried with a unit of a different scale", g, rec))

        # F16: a numeral of the wrong role presented as the measured value
        if m == 0:
            sent, support, tmpl = f"In {sp}:O2 slice viability was {a_p2.sample_size}% in treated slices.", (f"{sp}:O2",), "sample_size_as_value"
        elif m == 1:
            sent, support, tmpl = f"In {sc}:O1 forced vital capacity change was 12 mL in the treated arm.", (f"{sc}:O1",), "endpoint_name_digit_as_value"
        else:
            sent, support, tmpl = f"In {sp}:O2 slice viability was 0.70% in treated slices.", (f"{sp}:O2",), "p_value_as_value"
        cases.append(BenchmarkCase(f"F16-{i:03d}", "F16_numeral_role_confusion", "stress", True, tmpl,
                                   "Sample size, endpoint-name digit or p-value presented in the outcome's unit", g,
                                   _rec(f"BAD-F16-{i:03d}", target_ctx, support, (sent,))))

        # F17: tokenizer evasion beyond the first release's families
        h = i % 6
        f17 = (
            (f"In {sp}:O1 secreted PRO-C3 fell to ELISA-{51 + i} ng/mL.", "numeral_glued_to_uppercase_token"),
            (f"In {sp}:O1 secreted PRO-C3 fell by fifteen percent relative to vehicle slices.", "teen_number_word"),
            (f"In {sp}:O1 secreted PRO-C3 in treated slices was twice as high compared with vehicle slices.", "multiplicative_adverb"),
            (f"In {sp}:O1 secreted PRO-C3 fell to ⅛ of the vehicle level.", "untabulated_vulgar_fraction"),
            (f"In {sp}:O1 secreted PRO-C3 fell to ¹² ng/mL.", "superscript_digits"),
            (f"In {sp}:O1 secreted PRO-C3 fell to .5 ng/mL.", "leading_dot_decimal"),
        )[h]
        cases.append(BenchmarkCase(f"F17-{i:03d}", "F17_tokenizer_evasion", "stress", True, f17[1],
                                   "Fabricated quantity written in a form a digit regex does not read", g,
                                   _rec(f"BAD-F17-{i:03d}", target_ctx, (f"{sp}:O1",), (f17[0],))))

        # F18: polarity inversion in open-class wording
        f18 = (
            (f"In {sp}:O1 PRO-C3 rose to {p1f} ng/mL in treated slices compared with {p1c} ng/mL in vehicle slices.", "irregular_verb"),
            (f"In {sp}:O1 treated slices showed an increase in PRO-C3 to {p1f} ng/mL compared with {p1c} ng/mL in vehicle slices.", "nominalization"),
            (f"In {sp}:O1 PRO-C3 was not reduced in treated slices ({p1f} ng/mL) compared with vehicle slices ({p1c} ng/mL).", "negated_participle"),
            (f"In {sp}:O1 PRO-C3 was higher in treated slices ({p1f} ng/mL) than in vehicle slices ({p1c} ng/mL).", "comparative_adjective"),
        )[q]
        cases.append(BenchmarkCase(f"F18-{i:03d}", "F18_open_class_polarity_inversion", "stress", True, f18[1],
                                   "True numbers with an inverted effect direction expressed outside the first release's verb list", g,
                                   _rec(f"BAD-F18-{i:03d}", target_ctx, (f"{sp}:O1",), (f18[0],))))

        f20 = (
            f"In {sp}:O1 PRO-C3 increased, rather than decreased, to {p1f} ng/mL in treated slices compared with {p1c} ng/mL in vehicle slices.",
            f"In {sp}:O1 the reduction seen in vehicle slices was absent and PRO-C3 rose to {p1f} ng/mL in treated slices from {p1c} ng/mL.",
            f"In {sp}:O1 the expected decrease did not occur: PRO-C3 increased to {p1f} ng/mL in treated slices versus {p1c} ng/mL in vehicle slices.",
        )[m]
        cases.append(BenchmarkCase(f"F20-{i:03d}", "F20_mixed_polarity_cues", "stress", True,
                                   ("rather_than", "absent_reduction", "negation_after_cue")[m],
                                   "Inverted direction stated alongside a cue of the true direction", g,
                                   _rec(f"BAD-F20-{i:03d}", target_ctx, (f"{sp}:O1",), (f20,))))

        cases.append(_ledger_tamper_case(i))

    for i in range(50):
        g, target_ctx, _ = _build_parametric_base_graph(600 + i)
        sc, sp = f"C{700 + i}", f"P{700 + i}"
        a_c1, a_c2, a_c3 = g.atoms[f"{sc}:O1"], g.atoms[f"{sc}:O2"], g.atoms[f"{sc}:O3"]
        a_p1, a_p2 = g.atoms[f"{sp}:O1"], g.atoms[f"{sp}:O2"]
        fvc_t, fvc_p = a_c1.focal_quantity.value, a_c1.comparator_quantity.value
        svc_t, svc_p = a_c2.focal_quantity.value, a_c2.comparator_quantity.value
        p1f, p1c = a_p1.focal_quantity.value, a_p1.comparator_quantity.value
        p2f, p2c = a_p2.focal_quantity.value, a_p2.comparator_quantity.value

        # P5: TRUE sentence under a pooled citation (the Theorem 2 counterpart of F6/F11)
        if i % 2 == 0:
            sent, tmpl = f"Mean 12-week forced vital capacity change was {fvc_t} mL versus {fvc_p} mL ({sc}:O1, {sc}:O2).", "pooled_true_O1"
        else:
            sent, tmpl = f"Mean 12-week slow vital capacity change was {svc_t} mL versus {svc_p} mL ({sc}:O1, {sc}:O2).", "pooled_true_O2"
        cases.append(BenchmarkCase(f"P5-{i:03d}", "P5_valid_pooled_citation", "stress", False, tmpl,
                                   "True quantitative sentence whose citation pools two same-unit outcomes", g,
                                   _rec(f"REC-P5-{i:03d}", target_ctx, (f"{sc}:O1", f"{sc}:O2"), (sent,))))

        # P6: valid natural syntax the hardened verifier is meant to accept
        k = i % 7
        p6 = (
            ((f"{sp}:O1",), f"In {sp}:O1, compared with vehicle slices, treated slices lowered PRO-C3 to {p1f} ng/mL from {p1c} ng/mL.", "comparator_fronted_to_from"),
            ((f"{sc}:O1",), f"In {sc}:O1 the treated arm showed –{_unsigned(fvc_t)} mL change whereas the placebo arm showed –{_unsigned(fvc_p)} mL.", "en_dash_minus"),
            ((f"{sp}:O1",), f"In {sp}:O1 treated slices reduced PRO-C3 from {p1c} ng/mL in vehicle slices to {p1f} ng/mL without loss of viability.", "trailing_benign_negation"),
            ((f"{sp}:O1", f"{sp}:O2"), f"{sp}:O1 and {sp}:O2 measured {p1f} versus {p1c} ng/mL and {p2f} versus {p2c}%, respectively.", "respectively_pairs"),
            ((f"{sc}:O1", f"{sc}:O2"), f"At week 12, {sc}:O1 and {sc}:O2 changed by {fvc_t} mL and {svc_t} mL, respectively.", "respectively_leading_context_numeral"),
            ((f"{sc}:O1",), f"In the double-blind {sc}:O1 analysis the treated arm showed {fvc_t} mL change whereas the placebo arm showed {fvc_p} mL.", "trial_design_idiom"),
            ((f"{sp}:O1", f"{sp}:O2"), f"PRO-C3 was {p1f} ng/mL versus {p1c} ng/mL in vehicle slices ({sp}:O1), whereas slice viability was {p2f}% versus {p2c}% ({sp}:O2).", "citation_last_with_versus"),
        )[k]
        cases.append(BenchmarkCase(f"P6-{i:03d}", "P6_valid_natural_syntax_supported", "stress", False, p6[2],
                                   "Valid sentence in a natural form the first release rejected", g,
                                   _rec(f"REC-P6-{i:03d}", target_ctx, p6[0], (p6[1],))))

        # P7: valid natural syntax that remains outside the synthesis language
        diff = str(Decimal(p1c) - Decimal(p1f))
        j = i % 5
        p7 = (
            ((f"{sc}:O1",), f"In {sc}:O1 forced vital capacity declined by {_unsigned(fvc_t)} mL in the treated arm versus {_unsigned(fvc_p)} mL in the placebo arm.", "magnitude_of_negative_change"),
            ((f"{sp}:O1",), f"In {sp}:O1 treated slices reduced PRO-C3 by {diff} ng/mL relative to vehicle slices.", "derived_difference"),
            ((f"{sc}:O1",), f"In the Phase 2 study {sc}:O1 the treated arm showed {fvc_t} mL change whereas the placebo arm showed {fvc_p} mL.", "contextual_numeral_not_in_atom"),
            ((f"{sp}:O1",), f"In {sp}:O1 treated slices reduced PRO-C3 to {p1f} ng/mL without reducing viability, versus {p1c} ng/mL in vehicle slices.", "negation_scoped_to_other_noun"),
            ((f"{sp}:O1",), f"In {sp}:O1 PRO-C3 was {p1f} ± 3.1 ng/mL in treated slices versus {p1c} ± 4.2 ng/mL in vehicle slices.", "dispersion_not_in_atom"),
        )[j]
        cases.append(BenchmarkCase(f"P7-{i:03d}", "P7_valid_natural_syntax_unsupported", "stress", False, p7[2],
                                   "Valid sentence the verifier refuses: derived, contextual or magnitude-phrased numerals, or ambiguous negation scope", g,
                                   _rec(f"REC-P7-{i:03d}", target_ctx, p7[0], (p7[1],))))

        # P8: canonical single-tuple rendering of an atom
        tup = sorted(g.atoms)[i % len(g.atoms)]
        atom = g.atoms[tup]
        cases.append(BenchmarkCase(f"P8-{i:03d}", "P8_valid_canonical_rendering", "stress", False, f"render:{atom.focal_quantity.kind}:{tup.rsplit(':', 1)[1]}:{tup[0]}",
                                   "Canonical rendering of an evidence atom", g,
                                   _rec(f"REC-P8-{i:03d}", atom.context, (tup,), (render_atom_sentence(atom),))))
    return cases


# ---------------------------------------------------------------------------
# Residual suite: defects outside the reach of surface-prose checking
# ---------------------------------------------------------------------------

def _residual_cases() -> list[BenchmarkCase]:
    cases: list[BenchmarkCase] = []
    for i in range(60):
        g, target_ctx, _ = _build_parametric_base_graph(800 + i)
        sc, sp = f"C{900 + i}", f"P{900 + i}"
        a_c2, a_p1, a_p2 = g.atoms[f"{sc}:O2"], g.atoms[f"{sp}:O1"], g.atoms[f"{sp}:O2"]
        svc_t, svc_p = a_c2.focal_quantity.value, a_c2.comparator_quantity.value
        p1f, p1c = a_p1.focal_quantity.value, a_p1.comparator_quantity.value
        m = i % 3

        r1 = (
            ((f"{sp}:O1",), f"In {sp}:O1 slice viability was {p1f} ng/mL versus {p1c} ng/mL."),
            ((f"{sc}:O1", f"{sc}:O2"), f"For {sc}:O1 and {sc}:O2, mean forced vital capacity change was {svc_t} mL versus {svc_p} mL."),
            ((f"{sc}:O2",), f"In {sc}:O2 forced vital capacity change was {svc_t} mL versus {svc_p} mL."),
        )[m]
        cases.append(BenchmarkCase(f"R1-{i:03d}", "R1_endpoint_noun_misnaming", "residual", True,
                                   ("single_tuple_wrong_noun", "and_split_binds_to_second_tuple", "sibling_noun")[m],
                                   "Right tuple and numbers, wrong endpoint name in the prose", g,
                                   _rec(f"BAD-R1-{i:03d}", target_ctx, r1[0], (r1[1],))))

        r2 = (
            ((f"{sp}:O2",), f"In {sp}:O2 slice viability was {a_p2.sample_size} in treated slices."),
            ((f"{sc}:O1",), f"In {sc}:O1 the change at the primary read-out was 12 in the treated arm."),
            ((f"{sp}:O2",), f"In {sp}:O2 slice viability was 0.70 in treated slices."),
        )[m]
        cases.append(BenchmarkCase(f"R2-{i:03d}", "R2_unitless_role_confusion", "residual", True,
                                   ("sample_size", "endpoint_name_digit", "p_value")[m],
                                   "Sample size, name digit or p-value presented as the measured value with no unit", g,
                                   _rec(f"BAD-R2-{i:03d}", target_ctx, r2[0], (r2[1],))))

        r3 = (
            f"{sp}:O1 shows that treatment abolished collagen formation in human lung tissue.",
            f"In {sp}:O1 treatment worsened PRO-C3 ({p1f} ng/mL in treated slices versus {p1c} ng/mL in vehicle slices).",
            f"{sp}:O1 establishes that the compound is curative in fibrotic lung.",
        )[m]
        cases.append(BenchmarkCase(f"R3-{i:03d}", "R3_qualitative_overclaim", "residual", True,
                                   ("abolished", "worsened_with_true_numbers", "curative")[m],
                                   "Qualitative overstatement or inversion with no numeral or no listed polarity cue", g,
                                   _rec(f"BAD-R3-{i:03d}", target_ctx, (f"{sp}:O1",), (r3,))))

        r4 = (
            f"In {sp}:O1, far from being lowered, PRO-C3 reached {p1f} ng/mL in treated slices versus {p1c} ng/mL in vehicle slices.",
            f"In {sp}:O1, contrary to the expected reduction, PRO-C3 reached {p1f} ng/mL in treated slices versus {p1c} ng/mL in vehicle slices.",
            f"In {sp}:O1 the anticipated decrease was illusory: PRO-C3 reached {p1f} ng/mL in treated slices versus {p1c} ng/mL in vehicle slices.",
        )[m]
        cases.append(BenchmarkCase(f"R4-{i:03d}", "R4_implicit_negation_of_true_direction", "residual", True,
                                   ("far_from", "contrary_to", "illusory")[m],
                                   "The true direction is named and then denied without a listed negation cue", g,
                                   _rec(f"BAD-R4-{i:03d}", target_ctx, (f"{sp}:O1",), (r4,))))
    return cases


# ---------------------------------------------------------------------------
# Red-team suite: variations written by an independent adversarial reviewer
# ---------------------------------------------------------------------------

def _redteam_cases() -> list[BenchmarkCase]:
    """Variations of the stress and core mechanisms written against release 2.0 of the verifier.

    An independent reviewer with access to the code edited the suite's own
    templates minimally until the verifier accepted them. Each sentence is false
    under the atoms of the parametric graph and appears once: there is no
    replication here. Release 2.0 accepted all of them (see
    audit/compare_releases.py); the current release was revised in response, so
    its results on this suite are in-sample.
    """
    g, target_ctx, ctx_map = _build_parametric_base_graph(400)
    sc, sp, sm = "C500", "P500", "M500"
    a_c1, a_c2, a_c3 = g.atoms[f"{sc}:O1"], g.atoms[f"{sc}:O2"], g.atoms[f"{sc}:O3"]
    a_p1, a_p2, a_m1 = g.atoms[f"{sp}:O1"], g.atoms[f"{sp}:O2"], g.atoms[f"{sm}:O1"]
    fvc_t, fvc_p = a_c1.focal_quantity.value, a_c1.comparator_quantity.value
    svc_t, svc_p = a_c2.focal_quantity.value, a_c2.comparator_quantity.value
    p1f, p1c = a_p1.focal_quantity.value, a_p1.comparator_quantity.value
    p2f, p2c = a_p2.focal_quantity.value, a_p2.comparator_quantity.value
    n_p = a_p2.sample_size
    n_words = {12: "twelve", 13: "thirteen", 14: "fourteen", 15: "fifteen", 16: "sixteen", 17: "seventeen", 18: "eighteen",
               19: "nineteen", 20: "twenty", 21: "twenty-one", 22: "twenty-two", 23: "twenty-three"}
    n_c = a_c3.sample_size
    n_c_words = "sixty" if n_c == 60 else "sixty-" + ["", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"][n_c % 10]
    both = (proposition_from_atom(a_c1, "PROP-T-A"), proposition_from_atom(a_c2, "PROP-T-B"))
    pooled = {"polarity": "SUPPORTS", "props": both, "prom": True}
    mur = {"target": ctx_map["mur"]}

    rows: list[tuple[str, str, str, dict[str, Any]]] = [
        # T11: pooled-citation misbinding in forms that the scoping resolves to one outcome
        ("T11", "square_brackets", f"Mean 12-week forced vital capacity change was {svc_t} mL versus {svc_p} mL [{sc}:O2, {sc}:O1].", pooled),
        ("T11", "separate_parentheses", f"Mean 12-week forced vital capacity change was {svc_t} mL versus {svc_p} mL ({sc}:O2) and ({sc}:O1).", pooled),
        ("T11", "see_citation", f"Mean 12-week forced vital capacity change was {svc_t} mL versus {svc_p} mL; see {sc}:O2 and {sc}:O1.", pooled),
        # T14: index-aligned lists
        ("T14", "pairs_internally_reversed", f"{sp}:O1 and {sp}:O2 measured {p1c} versus {p1f} ng/mL and {p2c} versus {p2f}%, respectively.", {}),
        ("T14", "arms_named_values_reversed", f"In treated slices versus vehicle slices, {sp}:O1 and {sp}:O2 measured {p1c} ng/mL versus {p1f} ng/mL and {p2c}% versus {p2f}%, respectively.", {}),
        ("T14", "comparator_values_given_to_treated", f"In treated slices {sp}:O1 and {sp}:O2 measured {p1c} ng/mL and {p2c}%, respectively.", {}),
        ("T14", "inverted_verb_after_list", f"{sp}:O1 and {sp}:O2 reached {p1f} ng/mL and {p2f}%, respectively, with PRO-C3 increased relative to vehicle slices.", {}),
        ("T14", "from_to_reversed_in_list", f"{sp}:O1 and {sp}:O2 moved from {p1f} ng/mL to {p1c} ng/mL and from {p2f}% to {p2c}%, respectively.", {}),
        # T15: units of another scale
        ("T15", "mcg_per_mL", f"In {sp}:O1 treated slices reduced PRO-C3 from {p1c} mcg/mL in vehicle slices to {p1f} mcg/mL.", {}),
        ("T15", "ng_per_uL", f"In {sp}:O1 treated slices reduced PRO-C3 from {p1c} ng/uL in vehicle slices to {p1f} ng/uL.", {}),
        ("T15", "thin_space_before_unit", f"In {sp}:O1 treated slices reduced PRO-C3 from {p1c}\u2009mg/mL in vehicle slices to {p1f}\u2009mg/mL.", {}),
        ("T15", "unit_in_parentheses", f"In {sp}:O1 treated slices reduced PRO-C3 (mg/mL) from {p1c} in vehicle slices to {p1f}.", {}),
        ("T15", "litres", f"In {sc}:O1 the treated arm showed {fvc_t} litres change whereas the placebo arm showed {fvc_p} litres.", {}),
        ("T15", "lower_case_l", f"In {sc}:O1 the treated arm showed {fvc_t} l change whereas the placebo arm showed {fvc_p} l.", {}),
        ("T15", "months_abbreviated", f"In {sm}:O2 survival exceeded 28 mo.", mur),
        ("T15", "kg_per_lung", f"In {sm}:O1 lung hydroxyproline was {a_m1.focal_quantity.value} kg/lung.", mur),
        # T16: a numeral of another role given as the value
        ("T16", "thin_space_percent", f"In {sp}:O2 slice viability was {n_p}\u2009% in treated slices.", {}),
        ("T16", "per_cent", f"In {sp}:O2 slice viability was {n_p} per cent in treated slices.", {}),
        ("T16", "unit_in_parentheses", f"In {sp}:O2 slice viability (%) was {n_p} in treated slices.", {}),
        ("T16", "sample_size_as_word", f"In {sp}:O1 secreted PRO-C3 was {n_words[a_p1.sample_size]} ng/mL in treated slices.", {}),
        ("T16", "sample_size_as_compound_word", f"In {sc}:O3 gastrointestinal events occurred in {n_c_words} percent of the treated arm.", {}),
        ("T16", "p_value_with_other_unit", f"In {sc}:O1 forced vital capacity change was 0.01 L in the treated arm.", {}),
        ("T16", "p_value_as_fold", f"In {sp}:O2 slice viability changed 0.70-fold in treated slices.", {}),
        # T17: quantities the tokenizer did not read
        ("T17", "fused_fold_word", f"In {sp}:O1 secreted PRO-C3 fell twelvefold relative to vehicle slices.", {}),
        ("T17", "ordinal_fraction", f"In {sp}:O1 secreted PRO-C3 fell to a tenth of the vehicle level.", {}),
        ("T17", "multiplier_prefix", f"In {sp}:O1 secreted PRO-C3 fell x10 relative to vehicle slices.", {}),
        ("T17", "glued_integer_no_unit", f"In {sp}:O1 secreted PRO-C3 fell to ELISA51 in treated slices.", {}),
        ("T17", "exponent_suffix", f"In {sp}:O1 secreted PRO-C3 was {p1f}e3 in treated slices.", {}),
        ("T17", "k_suffix", f"In {sp}:O1 secreted PRO-C3 was {p1f}k ng/mL in treated slices.", {}),
        ("T17", "non_breaking_hyphen_minus", f"In {sp}:O1 the change in secreted PRO-C3 was \u2011{p1f} ng/mL in treated slices.", {}),
        ("T17", "million", f"In {sp}:O1 secreted PRO-C3 fell to {p1f} million ng/mL.", {}),
        # T18: polarity
        ("T18", "comparative_arms_exchanged", f"In {sp}:O1 PRO-C3 was lower in vehicle slices ({p1c} ng/mL) than in treated slices ({p1f} ng/mL).", {}),
        ("T18", "comparator_as_subject", f"In {sp}:O1 vehicle slices showed lower PRO-C3 ({p1c} ng/mL) than treated slices ({p1f} ng/mL).", {}),
        ("T18", "negation_three_words_away", f"In {sp}:O1 treated slices did not show any measurable reduction in PRO-C3 ({p1f} ng/mL) compared with vehicle slices ({p1c} ng/mL).", {}),
        ("T18", "negation_with_comma", f"In {sp}:O1 PRO-C3 was not, however, reduced in treated slices ({p1f} ng/mL) compared with vehicle slices ({p1c} ng/mL).", {}),
        ("T18", "cue_reassigned_to_comparator", f"In {sp}:O1 the reduction in PRO-C3 was seen in vehicle slices ({p1c} ng/mL), not in treated slices ({p1f} ng/mL).", {}),
        ("T18", "non_prefix", f"In {sp}:O1 PRO-C3 was non-reduced in treated slices ({p1f} ng/mL) compared with vehicle slices ({p1c} ng/mL).", {}),
        # T5: arm and order
        ("T5", "canonical_rendering_values_exchanged", f"In {sp}:O1 the endpoint Secreted PRO-C3 neo-epitope concentration was {p1c} ng/mL versus {p1f} ng/mL (n = {a_p1.sample_size}).", {}),
        ("T5", "comparator_value_given_to_focal_arm", f"In {sp}:O1 PRO-C3 was {p1c} ng/mL in treated slices.", {}),
        ("T5", "unregistered_arm_paraphrase", f"In {sp}:O1 PRO-C3 was {p1c} ng/mL in drug-exposed tissue versus {p1f} ng/mL in vehicle slices.", {}),
        ("T5", "from_to_swap_with_intervening_decimal", f"In {sp}:O1 treated slices reduced PRO-C3 from {p1f} ng/mL (p=0.003) to {p1c} ng/mL.", {}),
        ("T5", "from_to_swap_decimal_commas", f"In {sp}:O1 treated slices reduced PRO-C3 from {p1f.replace('.', ',')} ng/mL to {p1c.replace('.', ',')} ng/mL.", {}),
        ("T5", "unrelated_from_to", f"In {sp}:O1, from {a_p1.sample_size} donors (p from 0.003 to 0.003), PRO-C3 was {p1c} ng/mL in treated slices versus {p1f} ng/mL in vehicle slices.", {}),
    ]
    names = {
        "T5": "T5_arm_and_order_variations",
        "T11": "T11_pooled_citation_forms",
        "T14": "T14_list_variations",
        "T15": "T15_unit_variations",
        "T16": "T16_role_variations",
        "T17": "T17_tokenizer_variations",
        "T18": "T18_polarity_variations",
    }
    cases: list[BenchmarkCase] = []
    counters: dict[str, int] = {}
    for code, tmpl, sentence, kw in rows:
        i = counters.get(code, 0)
        counters[code] = i + 1
        cited = tuple(dict.fromkeys(TUPLE_RE.findall(sentence)))
        rec = _rec(
            f"BAD-{code}-{i:03d}", kw.get("target", target_ctx), cited, (sentence,),
            polarity=kw.get("polarity", "SCOPING_QUALIFICATION"), props=kw.get("props", ()), prom=kw.get("prom", False),
        )
        cases.append(BenchmarkCase(f"{code}-{i:03d}", names[code], "redteam", True, tmpl,
                                   "Independent reviewer's variation, false under the cited atoms", g, rec))
    return cases


def _build_gauntlet_graph() -> tuple[ContextEvidenceGraph, ContextEvidenceGraph, ContextSignature, dict[str, ContextSignature]]:
    """Extended parametric graphs: `g_base` (standard IDs) + `g_colon` (colon-bearing IDs)."""
    g, target_ctx, ctx_map = _build_parametric_base_graph(420)
    studies = dict(g.studies)
    study_vars = {k: set(v) for k, v in g.study_variables.items()}
    arm_aliases = dict(g.arm_aliases)
    atoms = dict(g.atoms)
    raw_arms = dict(g.raw_mentioned_arms_by_tuple)

    sp, sm = "P520", "M520"
    p_arms = (f"VAR:{sp}:Drug", f"VAR:{sp}:Vehicle")
    arm_aliases[f"VAR:{sm}:Drug"] = ("drug group", "treated mice")
    arm_aliases[f"VAR:{sm}:ActiveComp"] = ("active comparator", "pirfenidone comparator")
    arm_aliases[f"VAR:{sm}:Vehicle"] = ("vehicle group", "vehicle mice")

    extra_pcls = [
        ("O3", "Unbound bath drug concentration", QuantitySpec("point_level", "=", "10", unit="uM", unit_dimension="molarity"), None, "mean"),
        ("O4", "Buffer glucose concentration", QuantitySpec("point_level", "=", "5", unit="mM", unit_dimension="molarity"), None, "mean"),
        ("O5", "Receptor dissociation constant", QuantitySpec("point_level", "=", "25", unit="nM", unit_dimension="molarity"), None, "mean"),
        ("O6", "Potency threshold", QuantitySpec("point_level", "=", "50", unit="pM", unit_dimension="molarity"), None, "mean"),
        ("O7", "Fibrotic septa thickness", QuantitySpec("point_level", "=", "10", unit="um", unit_dimension="length"), None, "mean"),
        ("O8", "Cytotoxic LDH release", QuantitySpec("censored_observation", "<", "5", unit="%", unit_dimension="percent"), None, "proportion"),
        ("O9", "Target engagement floor", QuantitySpec("censored_observation", ">=", "5", unit="%", unit_dimension="percent"), None, "proportion"),
        ("O10", "Off-target kinase inhibition cap", QuantitySpec("censored_observation", "<=", "10", unit="%", unit_dimension="percent"), None, "proportion"),
        (
            "O11",
            "Target residence half-life",
            QuantitySpec("point_level", "=", "24", unit="h", unit_dimension="time"),
            QuantitySpec("point_level", "=", "8", unit="h", unit_dimension="time"),
            "mean",
        ),
    ]
    for oid, ep, fq, cq, stat in extra_pcls:
        tup = f"{sp}:{oid}"
        contrast = (
            MultiArmContrast((p_arms[0],), (p_arms[1],), "pairwise")
            if cq is not None
            else MultiArmContrast((p_arms[0],), (), "single_arm")
        )
        atoms[tup] = EvidenceAtom(
            atom_id=f"ATOM:{tup}",
            outcome_tuple=tup,
            study_id=sp,
            endpoint=ep,
            context=ctx_map["pcls"],
            contrast=contrast,
            focal_quantity=fq,
            comparator_quantity=cq,
            statistic_kind=stat,  # type: ignore[arg-type]
            sample_size=16,
            p_value_or_interval=None,
            polarity="favorable",
            provenance=ProvenancePin(f"SYNTHETIC-DOC:{sp}", "Table 2", ep, False, True),
        )
        raw_arms[tup] = p_arms if cq is not None else (p_arms[0],)

    g_base = ContextEvidenceGraph(
        lattice=g.lattice,
        studies=dict(studies),
        study_variables={k: set(v) for k, v in study_vars.items()},
        arm_aliases=dict(arm_aliases),
        atoms=dict(atoms),
        raw_mentioned_arms_by_tuple=dict(raw_arms),
    )

    # Colon-containing document/study IDs (`PMID:...`, `DOI:10..../...`) isolated in `g_colon`
    for col_sid, fv, cv, ep in (
        ("PMID:24836310", "-114.7", "-239.9", "Annual rate of FVC decline"),
        ("DOI:10.1056/NEJMoa1402582", "-113.6", "-207.3", "Replicate annual rate of FVC decline"),
    ):
        studies[col_sid] = target_ctx
        c_arms = (f"VAR:{col_sid}:Active", f"VAR:{col_sid}:Placebo")
        study_vars[col_sid] = set(c_arms)
        arm_aliases[c_arms[0]] = ("active group",)
        arm_aliases[c_arms[1]] = ("placebo group",)
        tup = f"{col_sid}:O1"
        atoms[tup] = EvidenceAtom(
            atom_id=f"ATOM:{tup}",
            outcome_tuple=tup,
            study_id=col_sid,
            endpoint=ep,
            context=target_ctx,
            contrast=MultiArmContrast((c_arms[0],), (c_arms[1],), "pairwise"),
            focal_quantity=QuantitySpec("point_level", "=", fv, unit="mL/year", unit_dimension="flow_rate"),
            comparator_quantity=QuantitySpec("point_level", "=", cv, unit="mL/year", unit_dimension="flow_rate"),
            statistic_kind="rate",
            sample_size=515,
            p_value_or_interval="p<0.001",
            polarity="favorable",
            provenance=ProvenancePin(col_sid, "Results", ep, False, True),
        )
        raw_arms[tup] = c_arms

    g_colon = ContextEvidenceGraph(
        lattice=g.lattice,
        studies=studies,
        study_variables=study_vars,
        arm_aliases=arm_aliases,
        atoms=atoms,
        raw_mentioned_arms_by_tuple=raw_arms,
    )
    return g_base, g_colon, target_ctx, ctx_map


def _gauntlet_cases() -> list[BenchmarkCase]:
    """Domain-agnostic R3-S11 / R3-S12 gauntlet probes (38 adversarial + 10 valid, 1 case per template)."""
    g, g_colon, target_ctx, ctx_map = _build_gauntlet_graph()
    sc, sp, sm = "C520", "P520", "M520"
    a_c1, a_c2 = g.atoms[f"{sc}:O1"], g.atoms[f"{sc}:O2"]
    a_p1, a_p2 = g.atoms[f"{sp}:O1"], g.atoms[f"{sp}:O2"]
    a_m1, a_m2 = g.atoms[f"{sm}:O1"], g.atoms[f"{sm}:O2"]
    fvc_t, fvc_p = a_c1.focal_quantity.value, a_c1.comparator_quantity.value
    svc_t, svc_p = a_c2.focal_quantity.value, a_c2.comparator_quantity.value
    p1f, p1c = a_p1.focal_quantity.value, a_p1.comparator_quantity.value
    p2f, p2c = a_p2.focal_quantity.value, a_p2.comparator_quantity.value
    n_p = a_p2.sample_size
    hyp_v = a_m1.focal_quantity.value
    surv_v = a_m2.focal_quantity.value
    mur = {"target": ctx_map["mur"]}

    adv_rows: list[tuple[str, str, str, dict[str, Any]]] = [
        # G1: general per-clause proposition grounding (k-cycles, 1-way & duplicate thefts, contrast subordinators, comma-appositives)
        (
            "G1",
            "three_tuple_k_cycle",
            f"In {sc}:O1 change was {svc_t} mL, whereas in {sc}:O2 change was {p1f} mL, while in {sp}:O1 PRO-C3 was {fvc_t} ng/mL.",
            {},
        ),
        (
            "G1",
            "four_tuple_k_cycle",
            f"In {sc}:O1 change was {svc_t} mL; in {sc}:O2 change was {p1f} mL; in {sp}:O1 PRO-C3 was {p2f} ng/mL; in {sp}:O2 viability was {fvc_t}%.",
            {},
        ),
        (
            "G1",
            "one_way_cross_clause_theft",
            f"In {sc}:O1 forced vital capacity change was {fvc_t} mL versus {fvc_p} mL, whereas in {sc}:O2 slow vital capacity change was {fvc_t} mL versus {svc_p} mL.",
            {},
        ),
        (
            "G1",
            "duplicate_cross_clause_value_theft",
            f"In {sp}:O1 PRO-C3 was {p1f} ng/mL in treated slices; in {sp}:O2 viability was {p1f}% in treated slices.",
            {},
        ),
        (
            "G1",
            "although_without_comma_swap",
            f"In {sc}:O1 forced vital capacity change was {svc_t} mL although in {sc}:O2 slow vital capacity change was {fvc_t} mL.",
            {},
        ),
        (
            "G1",
            "even_though_without_comma_theft",
            f"In {sc}:O1 forced vital capacity change was {fvc_t} mL even though in {sc}:O2 slow vital capacity change was {fvc_t} mL.",
            {},
        ),
        (
            "G1",
            "though_without_comma_swap",
            f"In {sc}:O1 forced vital capacity change was {svc_t} mL though in {sc}:O2 slow vital capacity change was {fvc_t} mL.",
            {},
        ),
        (
            "G1",
            "comma_appositive_comparative_arm_swap",
            f"In {sp}:O11 vehicle slices had longer target residence half-life, 24 h, than treated slices, 8 h.",
            {},
        ),
        # G2: Unicode combining/enclosing marks (Mn/Me), non-ASCII Zs/VS digit splitters & homoglyphs (UTS #39)
        (
            "G2",
            "combining_acute_u0301_digit_split",
            f"In {sp}:O1 treated slices reduced PRO-C3 to {p1f[0]}\u0301{p1f[1:]} ng/mL versus {p1c} ng/mL in vehicle slices.",
            {},
        ),
        (
            "G2",
            "enclosing_circle_u20dd_digit_split",
            f"In {sp}:O1 treated slices reduced PRO-C3 to {p1f[0]}\u20dd{p1f[1:]} ng/mL versus {p1c} ng/mL in vehicle slices.",
            {},
        ),
        (
            "G2",
            "combining_mark_on_ascii_identifier",
            f"In {sp}:O1 treated\u0301 slices reduced PRO-C3 to {p1f} ng/mL versus {p1c} ng/mL in vehicle slices.",
            {},
        ),
        (
            "G2",
            "nbsp_u00a0_digit_splitter",
            f"In {sp}:O1 treated slices reduced PRO-C3 to {p1f[0]}\u00a0{p1f[1:]} ng/mL versus {p1c} ng/mL in vehicle slices.",
            {},
        ),
        (
            "G2",
            "figure_space_u2007_digit_splitter",
            f"In {sp}:O1 treated slices reduced PRO-C3 to {p1f[0]}\u2007{p1f[1:]} ng/mL versus {p1c} ng/mL in vehicle slices.",
            {},
        ),
        (
            "G2",
            "variation_selector_ufe0f_digit_splitter",
            f"In {sp}:O1 treated slices reduced PRO-C3 to {p1f[0]}\ufe0f{p1f[1:]} ng/mL versus {p1c} ng/mL in vehicle slices.",
            {},
        ),
        (
            "G2",
            "cyrillic_mixed_script_homoglyph",
            f"In {sp}:O1 tre\u0430ted slices reduced PRO-C3 to {p1f} ng/mL versus {p1c} ng/mL in vehicle slices.",
            {},
        ),
        (
            "G2",
            "mathematical_bold_digit_homoglyph",
            f"In {sp}:O11 target residence half-life was \U0001d7d04 h in treated slices versus 8 h in vehicle slices.",
            {},
        ),
        # G3: Case-sensitive SI prefixes (uM/mM/nM/pM vs um/mm/nm/pm) & label/reagent unit collisions
        (
            "G3",
            "si_case_uM_vs_um",
            f"In {sp}:O3 unbound bath drug concentration (um) was 10 in treated slices.",
            {},
        ),
        (
            "G3",
            "si_case_mM_vs_mm",
            f"In {sp}:O4 buffer glucose concentration (mm) was 5 in treated slices.",
            {},
        ),
        (
            "G3",
            "si_case_nM_vs_nm",
            f"In {sp}:O5 receptor dissociation constant was 25 (nm) in treated slices.",
            {},
        ),
        (
            "G3",
            "si_case_pM_vs_pm",
            f"In {sp}:O6 potency threshold was 50 pm in treated slices.",
            {},
        ),
        (
            "G3",
            "panel_d_label_unit_collision",
            f"In {sm}:O2 survival exceeded {surv_v} weeks (panel d).",
            mur,
        ),
        (
            "G3",
            "panel_h_label_unit_collision",
            f"In {sp}:O11 target residence half-life was 24 min versus 8 min (panel h).",
            {},
        ),
        (
            "G3",
            "table_D_and_panel_d_label_collision",
            f"In {sm}:O2 survival exceeded {surv_v} h (Table D, panel d).",
            mur,
        ),
        (
            "G3",
            "reagent_actinomycin_D_unit_spoof",
            f"In {sm}:O2 survival with actinomycin D and vitamin D exceeded {surv_v} h.",
            mur,
        ),
        # G4: Symbolic & verbal negated inequalities, residual/stacked negations, and % boundary
        (
            "G4",
            "symbolic_not_lt_on_true_lt",
            f"In {sp}:O8 cytotoxic LDH release was not < 5% in treated slices.",
            {},
        ),
        (
            "G4",
            "symbolic_not_le_on_point",
            f"In {sp}:O1 PRO-C3 was not <= {p1f} ng/mL in treated slices.",
            {},
        ),
        (
            "G4",
            "symbolic_never_gt_on_gt_censored",
            f"In {sm}:O2 survival was never > {surv_v} days.",
            mur,
        ),
        (
            "G4",
            "verbal_not_fewer_than_on_lt",
            f"In {sp}:O8 cytotoxic LDH release was not fewer than 5% in treated slices.",
            {},
        ),
        (
            "G4",
            "verbal_never_less_than_on_lt",
            f"In {sp}:O8 cytotoxic LDH release was never less than 5% in treated slices.",
            {},
        ),
        (
            "G4",
            "residual_bare_not_point_value",
            f"In {sp}:O11 target residence half-life was not 24 h in treated slices.",
            {},
        ),
        (
            "G4",
            "stacked_double_negation_bound",
            f"In {sm}:O2 survival was not never < {surv_v} days.",
            mur,
        ),
        (
            "G4",
            "percent_before_period_role_swap",
            f"In {sp}:O2 slice viability in treated slices reached {n_p}%.",
            {},
        ),
        # G5: Conservative multi-arm & non-contrast arm binding
        (
            "G5",
            "symmetric_and_two_arms",
            f"In {sp}:O1 treated slices and vehicle slices measured {p1f} ng/mL.",
            {},
        ),
        (
            "G5",
            "parenthetical_arm_sidenote",
            f"In {sp}:O1 treated slices (see also vehicle slices) measured {p1c} ng/mL.",
            {},
        ),
        (
            "G5",
            "decoy_parenthetical_versus",
            f"In {sp}:O1 treated slices and vehicle slices (a post-hoc versus baseline note) measured {p1f} ng/mL.",
            {},
        ),
        (
            "G5",
            "three_arm_pairwise_clause",
            f"In {sm}:O2 the drug group versus the active comparator and vehicle group exceeded {surv_v} days.",
            mur,
        ),
        (
            "G5",
            "comparator_subject_single_value_comma",
            f"In {sp}:O1 vehicle slices showed secreted PRO-C3, {p1f} ng/mL, relative to treated slices.",
            {},
        ),
        (
            "G5",
            "clause_reset_absent_reduction",
            f"In {sp}:O1 vehicle slices were unchanged, and treated slices showed no reduction in PRO-C3 at {p1f} ng/mL versus {p1c} ng/mL.",
            {},
        ),
    ]

    val_rows: list[tuple[str, str, dict[str, Any]]] = [
        (
            "three_clause_valid_grounding",
            f"In {sc}:O1 change was {fvc_t} mL versus {fvc_p} mL, whereas in {sc}:O2 change was {svc_t} mL versus {svc_p} mL, while in {sp}:O1 PRO-C3 was {p1f} ng/mL versus {p1c} ng/mL.",
            {},
        ),
        (
            "although_valid_contrast",
            f"In {sc}:O1 forced vital capacity change was {fvc_t} mL although in {sc}:O2 slow vital capacity change was {svc_t} mL.",
            {},
        ),
        (
            "percent_before_period_and_paren",
            f"In {sp}:O2 slice viability in treated slices was {p2f}% (versus {p2c}% in vehicle slices), whereas in {sp}:O9 target engagement floor was >= 5%.",
            {},
        ),
        (
            "spaced_d_and_h_valid_units",
            f"In {sm}:O2 survival exceeded {surv_v} d, whereas in {sp}:O11 target residence half-life was 24 h versus 8 h.",
            mur,
        ),
        (
            "case_sensitive_uM_valid",
            f"In {sp}:O3 unbound bath drug concentration was 10 uM.",
            {},
        ),
        (
            "case_sensitive_um_valid",
            f"In {sp}:O7 fibrotic septa thickness was 10 um.",
            {},
        ),
        (
            "symbolic_not_lt_valid_ge",
            f"In {sp}:O9 target engagement floor was not < 5%.",
            {},
        ),
        (
            "symbolic_never_gt_valid_le",
            f"In {sp}:O10 off-target kinase inhibition cap was never > 10%.",
            {},
        ),
        (
            "verbal_not_fewer_than_valid_ge",
            f"In {sp}:O9 target engagement floor was not fewer than 5%.",
            {},
        ),
        (
            "colon_pmid_and_doi_valid_tuples",
            "In PMID:24836310:O1 annual rate of FVC decline was -114.7 mL/year versus -239.9 mL/year, whereas in DOI:10.1056/NEJMoa1402582:O1 replicate annual rate of FVC decline was -113.6 mL/year versus -207.3 mL/year.",
            {"use_colon": True},
        ),
    ]

    names = {
        "G1": "G1_per_clause_k_cycle_and_theft",
        "G2": "G2_unicode_marks_and_homoglyphs",
        "G3": "G3_si_case_and_label_unit_collisions",
        "G4": "G4_negated_bounds_and_percent_boundary",
        "G5": "G5_conservative_multi_arm_binding",
    }
    cases: list[BenchmarkCase] = []
    counters: dict[str, int] = {}
    for code, tmpl, sentence, kw in adv_rows:
        idx = counters.get(code, 0)
        counters[code] = idx + 1
        cited = tuple(dict.fromkeys(TUPLE_RE.findall(sentence)))
        rec = _rec(f"BAD-{code}-{idx:03d}", kw.get("target", target_ctx), cited, (sentence,))
        cases.append(
            BenchmarkCase(
                f"{code}-{idx:03d}",
                names[code],
                "gauntlet",
                True,
                tmpl,
                "Domain-agnostic R3-S11/R3-S12 gauntlet negative control",
                g_colon if kw.get("use_colon") else g,
                rec,
            )
        )
    for idx, (tmpl, sentence, kw) in enumerate(val_rows):
        cited = tuple(dict.fromkeys(TUPLE_RE.findall(sentence)))
        rec = _rec(f"REC-G0-{idx:03d}", kw.get("target", target_ctx), cited, (sentence,))
        cases.append(
            BenchmarkCase(
                f"G0-{idx:03d}",
                "G0_valid_gauntlet_controls",
                "gauntlet",
                False,
                tmpl,
                "Domain-agnostic R3-S11/R3-S12 gauntlet positive control",
                g_colon if kw.get("use_colon") else g,
                rec,
            )
        )
    return cases


def generate_benchmark_cases(*, include_gauntlet: bool = True) -> list[BenchmarkCase]:
    """All cases, in the fixed order core, stress, residual, red-team, gauntlet."""
    base = _core_cases() + _stress_cases() + _residual_cases() + _redteam_cases()
    return base + _gauntlet_cases() if include_gauntlet else base


# ---------------------------------------------------------------------------
# Exhaustive self-tests
# ---------------------------------------------------------------------------

def _all_contexts(lattice: ContextLatticeSpec) -> list[ContextSignature]:
    axes = sorted(lattice.axis_posets)
    elems = [sorted(lattice.axis_posets[a].dominates) for a in axes]
    return [ContextSignature(dict(zip(axes, combo))) for combo in itertools.product(*elems)]


def _definition_tau(lattice: ContextLatticeSpec, s: ContextSignature, t: ContextSignature) -> str:
    """Literal transcription of Definition 1 from down-sets, independent of evaluate_transfer."""
    down = {a: p.dominates for a, p in lattice.axis_posets.items()}
    unresolved = [a for a in down if t.axes[a] not in down[a][s.axes[a]]]
    if not unresolved:
        return "DIRECT_MATCH"
    for a in unresolved:
        has_cub = any(s.axes[a] in d and t.axes[a] in d for d in down[a].values())
        below_floor = a in lattice.bridge_floor and lattice.bridge_floor[a] not in down[a][s.axes[a]]
        if not has_cub or below_floor:
            return "INADMISSIBLE_TRANSFER"
    return "BOUNDED_BRIDGE"


def selftest_transfer_definition() -> dict[str, Any]:
    out: dict[str, Any] = {}
    for name, bundle in (("drug_development", build_drug_development_vignette()), ("sport_science", build_sport_exercise_vignette())):
        lat = bundle.graph.lattice
        lat.validate()
        ctxs = _all_contexts(lat)
        counts = {"DIRECT_MATCH": 0, "BOUNDED_BRIDGE": 0, "INADMISSIBLE_TRANSFER": 0}
        disagreements = direct_iff_violations = 0
        for s in ctxs:
            for t in ctxs:
                cls, unres = lat.evaluate_transfer(s, t)
                counts[cls] += 1
                if cls != _definition_tau(lat, s, t):
                    disagreements += 1
                target_below_source = all(lat.axis_posets[a].leq(t.axes[a], s.axes[a]) for a in lat.axis_posets)
                if (cls == "DIRECT_MATCH") != target_below_source or (cls == "DIRECT_MATCH") != (unres == ()):
                    direct_iff_violations += 1
        out[name] = {
            "contexts": len(ctxs),
            "ordered_pairs": len(ctxs) ** 2,
            "class_counts": counts,
            "disagreements_with_definition": disagreements,
            "direct_match_iff_target_below_source_violations": direct_iff_violations,
        }
    return out


def selftest_scalar_ranking_search() -> dict[str, Any]:
    """Theorem 1(2) on the diamond axis: no rank assignment reproduces the order.

    Every map r from the axis elements to {0..n-1} is tried (this covers every
    weak order). A threshold rule 'transfer from x to y iff r(y) <= r(x)' is
    faithful iff it licenses exactly the pairs y <= x of the axis.
    """
    poset = build_drug_development_vignette().graph.lattice.axis_posets["model_system"]
    elems = sorted(poset.dominates)
    n = len(elems)
    true_pairs = {(x, y) for x in elems for y in elems if poset.leq(y, x)}
    incomparable = sum(1 for x in elems for y in elems if x < y and not poset.leq(x, y) and not poset.leq(y, x))
    faithful = 0
    best_errors = None
    for ranks in itertools.product(range(n), repeat=n):
        r = dict(zip(elems, ranks))
        licensed = {(x, y) for x in elems for y in elems if r[y] <= r[x]}
        errors = len(licensed ^ true_pairs)
        if errors == 0:
            faithful += 1
        if best_errors is None or errors < best_errors:
            best_errors = errors
    return {
        "axis": "model_system",
        "elements": n,
        "incomparable_unordered_pairs": incomparable,
        "rank_assignments_tried": n ** n,
        "faithful_rank_assignments": faithful,
        "minimum_misclassified_ordered_pairs": best_errors,
    }


def _mutate_quantity(q: QuantitySpec | None, fallback: QuantitySpec) -> QuantitySpec | None:
    if q is None:
        return fallback
    if q.kind == "interval_bound":
        return dataclasses.replace(q, upper=str(Decimal(q.upper or "0") + 1))
    if q.kind == "missing_numeric":
        return dataclasses.replace(q, missing_reason=(q.missing_reason or "") + " (altered)")
    return dataclasses.replace(q, value=str(Decimal(q.value or "0") + 1))


def selftest_typed_mutations() -> dict[str, Any]:
    """Theorem 3: the true proposition is accepted; each single-coordinate change is rejected."""
    atoms: list[tuple[dict[str, EvidenceAtom], EvidenceAtom]] = []
    for bundle in (build_drug_development_vignette(), build_sport_exercise_vignette()):
        atoms.extend((bundle.graph.atoms, a) for a in bundle.graph.atoms.values())
    g, _, _ = _build_parametric_base_graph(0)
    atoms.extend((g.atoms, a) for a in g.atoms.values())

    true_accepted = 0
    rejected = {c: 0 for c in TYPED_COORDINATES}
    for atom_map, atom in atoms:
        p = proposition_from_atom(atom, "PROP:SELFTEST")
        if not verify_typed_proposition_homomorphism(p, atom_map):
            true_accepted += 1
        other_stat = "median" if atom.statistic_kind != "median" else "mean"
        mutants = {
            "endpoint": dataclasses.replace(p, claimed_endpoint=p.claimed_endpoint + " (per-protocol)"),
            "focal_arms": dataclasses.replace(p, claimed_focal_arms=tuple(p.claimed_comparator_arms) or tuple(p.claimed_focal_arms) + ("VAR:EXTRA",)),
            "comparator_arms": dataclasses.replace(p, claimed_comparator_arms=tuple(p.claimed_comparator_arms)[1:] if p.claimed_comparator_arms else ("VAR:EXTRA",)),
            "focal_quantity": dataclasses.replace(p, claimed_quantity=_mutate_quantity(p.claimed_quantity, p.claimed_quantity)),
            "comparator_quantity": dataclasses.replace(p, claimed_comparator_quantity=_mutate_quantity(p.claimed_comparator_quantity, p.claimed_quantity)),
            "statistic": dataclasses.replace(p, claimed_statistic=other_stat),
            "sample_size": dataclasses.replace(p, claimed_sample_size=(p.claimed_sample_size or 0) + 1),
        }
        for coord, mutant in mutants.items():
            if verify_typed_proposition_homomorphism(mutant, atom_map):
                rejected[coord] += 1
    permuted_tried = permuted_accepted = loose_n_tried = loose_n_rejected = 0
    for atom_map, atom in atoms:
        p = proposition_from_atom(atom, "PROP:SELFTEST")
        if len(p.claimed_comparator_arms) >= 2:
            permuted_tried += 1
            reordered = dataclasses.replace(p, claimed_comparator_arms=tuple(reversed(p.claimed_comparator_arms)))
            permuted_accepted += int(not verify_typed_proposition_homomorphism(reordered, atom_map))
        if p.claimed_sample_size is not None:
            loose_n_tried += 1
            as_float = dataclasses.replace(p, claimed_sample_size=float(p.claimed_sample_size))  # type: ignore[arg-type]
            loose_n_rejected += int(bool(verify_typed_proposition_homomorphism(as_float, atom_map)))
    return {
        "atoms": len(atoms),
        "true_propositions_accepted": true_accepted,
        "single_coordinate_mutants": len(atoms) * len(TYPED_COORDINATES),
        "mutants_rejected": sum(rejected.values()),
        "mutants_rejected_by_coordinate": rejected,
        "reordered_arm_sets_tried": permuted_tried,
        "reordered_arm_sets_accepted": permuted_accepted,
        "float_sample_sizes_tried": loose_n_tried,
        "float_sample_sizes_rejected": loose_n_rejected,
    }


# The rule that names each adversarial stratum's defect (release 3.2). A rejected case is "on
# target" when at least one of its violation codes is listed here. An entry keyed by
# (stratum, template) refines its stratum; a key ending in ":" matches a template prefix.
EXPECTED_CODES_BY_STRATUM: dict[str, tuple[str, ...]] = {
    "F1_fabricated_numeric_literal": ("I3_TOK_UNGROUNDED_VALUE",),
    "F2_intrastudy_cross_outcome_bleed": ("I3_TOK_UNGROUNDED_VALUE",),
    "F4_multituple_clause_swap": ("I3_CLAUSE_CROSS_OUTCOME_SWAP", "I3_CLAUSE_RESPECTIVELY_SWAP"),
    "F6_pooled_misbinding_in_prose_and_proposition": ("I3_CLAUSE_AMBIGUOUS_POOLED_CITATION",),
    "F9_undigitized_or_unverified_promotion": ("I5_ILLEGAL_QUANTITATIVE_PROMOTION_WITHOUT_PRIMARY_OR_DIGITIZED_SOURCE",),
    "F11_pooled_prose_misbinding_true_propositions": ("I3_CLAUSE_AMBIGUOUS_POOLED_CITATION",),
    "F13_orphan_clause_bleed": ("I3_CLAUSE_CROSS_OUTCOME_SWAP",),
    "F15_unit_scale_swap": ("I3_CLAUSE_UNIT_MISMATCH",),
    "F16_numeral_role_confusion": ("I3_TOK_ROLE_CONFUSION",),
    "F17_tokenizer_evasion": ("I3_TOK_UNGROUNDED_VALUE", "I3_TOK_NONCANONICAL_NUMERAL"),
    "F18_open_class_polarity_inversion": ("I3_CLAUSE_DIRECTION_REVERSAL", "I3_CLAUSE_NEGATED_DIRECTION"),
    "F20_mixed_polarity_cues": ("I3_CLAUSE_DIRECTION_REVERSAL", "I3_CLAUSE_NEGATED_DIRECTION"),
    "T5_arm_and_order_variations": ("I3_CLAUSE_ARM_VALUE_SWAP", "I3_CLAUSE_FROM_TO_SWAP", "I3_CLAUSE_UNLABELED_VALUE_ORDER"),
    "T11_pooled_citation_forms": ("I3_CLAUSE_AMBIGUOUS_POOLED_CITATION",),
    "T14_list_variations": (
        "I3_CLAUSE_ARM_VALUE_SWAP",
        "I3_CLAUSE_DIRECTION_REVERSAL",
        "I3_CLAUSE_FROM_TO_SWAP",
        "I3_CLAUSE_UNLABELED_VALUE_ORDER",
        "I3_CLAUSE_RESPECTIVELY_SWAP",
    ),
    "T15_unit_variations": ("I3_CLAUSE_UNIT_MISMATCH", "I3_CLAUSE_UNREAD_UNIT"),
    "T16_role_variations": ("I3_TOK_ROLE_CONFUSION", "I3_CLAUSE_UNIT_MISMATCH"),
    "T17_tokenizer_variations": ("I3_TOK_UNGROUNDED_VALUE", "I3_TOK_NONCANONICAL_NUMERAL", "I3_CLAUSE_UNREAD_UNIT"),
    "T18_polarity_variations": ("I3_CLAUSE_DIRECTION_REVERSAL", "I3_CLAUSE_NEGATED_DIRECTION"),
    "G1_per_clause_k_cycle_and_theft": ("I3_CLAUSE_CROSS_OUTCOME_SWAP", "I3_CLAUSE_ARM_VALUE_SWAP"),
    "G2_unicode_marks_and_homoglyphs": ("I3_TOK_FORBIDDEN_UNICODE", "I3_TOK_CONFUSABLE_HOMOGLYPH"),
    "G3_si_case_and_label_unit_collisions": ("I3_CLAUSE_UNIT_MISMATCH",),
    "G4_negated_bounds_and_percent_boundary": ("I3_CLAUSE_INEQUALITY_MISMATCH",),
    # G5 holds conservative refusals: the verifier declines to bind two arms without an explicit
    # contrast, so the corrected sentence in the same frame is refused as well.
    "G5_conservative_multi_arm_binding": (
        "I3_CLAUSE_NON_CONTRAST_MULTI_ARM",
        "I3_CLAUSE_MULTI_ARM_AMBIGUITY",
        "I3_CLAUSE_ARM_VALUE_SWAP",
        "I3_CLAUSE_NEGATED_DIRECTION",
    ),
}
EXPECTED_CODES_BY_TEMPLATE: dict[tuple[str, str], tuple[str, ...]] = {
    ("F3_lexical_and_unicode_evasion", "lexical:"): ("I3_TOK_UNGROUNDED_VALUE",),
    ("F3_lexical_and_unicode_evasion", "invisible:"): ("I3_TOK_FORBIDDEN_UNICODE",),
    ("F5_arm_swap_and_polarity_reversal", "arm_swap"): ("I3_CLAUSE_ARM_VALUE_SWAP",),
    ("F5_arm_swap_and_polarity_reversal", "from_to"): ("I3_CLAUSE_FROM_TO_SWAP",),
    ("F5_arm_swap_and_polarity_reversal", "verb"): ("I3_CLAUSE_DIRECTION_REVERSAL",),
    ("F7_scale_conflation_and_multiarm_truncation", "scale_conflation"): ("I3_TYPED_QUANTITY_MISMATCH",),
    ("F7_scale_conflation_and_multiarm_truncation", "arm_truncation"): ("I1_ATOM_VALIDATION_ERROR",),
    ("F8_context_poset_transfer_overclaim", "inadmissible_upgrade"): ("I4_INADMISSIBLE_CONTEXT_TRANSFER",),
    ("F8_context_poset_transfer_overclaim", "unbridged_promotion"): ("I4_ILLEGAL_QUANTITATIVE_PROMOTION_ACROSS_CONTEXT_GAP",),
    ("F10_support_counter_overlap_and_ledger_tamper", "overlap"): ("I6_SUPPORT_COUNTEREVIDENCE_OVERLAP",),
    ("F10_support_counter_overlap_and_ledger_tamper", "settled_drift"): ("I2_FORWARD_SETTLED_DIGEST_MISMATCH",),
    ("F12_typed_coverage_and_comparator_gaps", "promoted_tuple_uncovered"): ("I3_TYPED_PROMOTION_COVERAGE_GAP",),
    ("F12_typed_coverage_and_comparator_gaps", "sample_size_omitted"): ("I3_TYPED_SAMPLE_SIZE_MISMATCH",),
    ("F12_typed_coverage_and_comparator_gaps", "comparator_quantity_misbound"): ("I3_TYPED_COMPARATOR_QUANTITY_MISMATCH",),
    ("F14_respectively_misalignment", "leading_shared_numeral"): ("I3_CLAUSE_RESPECTIVELY_SWAP",),
    ("F14_respectively_misalignment", "comparators_swapped_in_pairs"): ("I3_CLAUSE_RESPECTIVELY_SWAP", "I3_CLAUSE_UNLABELED_VALUE_ORDER"),
    ("F14_respectively_misalignment", "polarity_inside_list"): ("I3_CLAUSE_DIRECTION_REVERSAL",),
    ("F19_ledger_and_binding_tamper", "pointer_negative_index"): ("I2_INVALID_ARRAY_INDEX",),
    ("F19_ledger_and_binding_tamper", "pointer_noncanonical"): ("I2_POINTER_OUTSIDE_DECLARED_STUDY",),
    ("F19_ledger_and_binding_tamper", "cross_study_attribution"): ("I2_POINTER_OUTSIDE_DECLARED_STUDY",),
    ("F19_ledger_and_binding_tamper", "atom_drift"): ("I1_ATOM_CORPUS_FIELD_MISMATCH",),
    ("F19_ledger_and_binding_tamper", "noop_event"): ("I2_NOOP_EVENT",),
    ("G4_negated_bounds_and_percent_boundary", "percent_before_period_role_swap"): ("I3_TOK_ROLE_CONFUSION",),
}
# Frozen red-team sentences that are refused before their named defect is evaluated. They are
# kept unchanged, because they were written against release 2.0, and are reported, not counted
# as evidence for the unit rules.
KNOWN_OFF_TARGET_TEMPLATES: dict[tuple[str, str], str] = {
    ("T15_unit_variations", "thin_space_before_unit"): "refused for the thin space U+2009 (I3_TOK_FORBIDDEN_UNICODE); the wrong unit is not evaluated",
    ("T16_role_variations", "thin_space_percent"): "refused for the thin space U+2009 (I3_TOK_FORBIDDEN_UNICODE); the numeral's role is not evaluated",
}


def _violation_code(violation: str) -> str:
    return violation.split(":")[0].split(" ")[0]


def _expected_codes(case: BenchmarkCase) -> tuple[str, ...] | None:
    for (stratum, template), codes in EXPECTED_CODES_BY_TEMPLATE.items():
        if case.stratum == stratum and (case.template == template or (template.endswith(":") and case.template.startswith(template))):
            return codes
    return EXPECTED_CODES_BY_STRATUM.get(case.stratum)


def _all_v3_codes(case: BenchmarkCase) -> set[str]:
    """Every violation code the full verifier raises on a case: replay, binding, graph integrity and the record."""
    codes: set[str] = set()
    if case.initial_corpus is not None and case.settled_corpus is not None and case.patch_events is not None:
        try:
            verify_bijective_patch_replay(case.initial_corpus, case.settled_corpus, case.patch_events)
        except ValueError as exc:
            codes.add(_violation_code(str(exc)))
        if case.bind_atoms_to_corpus:
            codes |= {_violation_code(v) for v in verify_atoms_bound_to_corpus(case.graph, case.settled_corpus)}
    codes |= {_violation_code(v) for v in case.graph.verify_graph_integrity()}
    codes |= {_violation_code(v) for v in case.graph.verify_synthesis_record(case.record).violations}
    return codes


def selftest_expected_violation_codes() -> dict[str, Any]:
    """Every rejected adversarial case outside the residual suite must be rejected by the rule that names its defect."""
    cases = [c for c in generate_benchmark_cases(include_gauntlet=True) if c.is_adversarial and c.suite != "residual"]
    on_target = with_additional_codes = 0
    off_target: dict[str, dict[str, Any]] = {}
    for c in cases:
        expected = _expected_codes(c)
        _require(expected is not None, f"no expected violation code declared for {c.stratum}/{c.template}")
        codes = _all_v3_codes(c)
        _require(bool(codes), f"adversarial case {c.case_id} raised no violation")
        if codes & set(expected or ()):
            on_target += 1
            with_additional_codes += int(bool(codes - set(expected or ())))
        else:
            row = off_target.setdefault(f"{c.stratum}/{c.template}", {"cases": 0, "codes": set()})
            row["cases"] += 1
            row["codes"] |= codes
    known = {f"{s}/{t}" for s, t in KNOWN_OFF_TARGET_TEMPLATES}
    _require(set(off_target) == known, f"off-target rejections differ from the declared list: {sorted(set(off_target) ^ known)}")
    return {
        "adversarial_cases_checked": len(cases),
        "rejected_by_the_rule_naming_the_defect": on_target,
        "of_which_also_raising_another_code": with_additional_codes,
        "rejected_before_the_named_defect_is_evaluated": sum(r["cases"] for r in off_target.values()),
        "off_target_templates": {
            k: {"cases": r["cases"], "codes": sorted(r["codes"]), "note": KNOWN_OFF_TARGET_TEMPLATES[tuple(k.split("/", 1))]}  # type: ignore[index]
            for k, r in sorted(off_target.items())
        },
    }


_NUMERAL_HEAD_RE = re.compile(r"[-\u2212\u2013]?[0-9]+(?:[.,][0-9]+)*")


def selftest_tamper_and_reattest() -> dict[str, Any]:
    """Start from accepted valid records, change one thing, attach a fresh valid attestation, and require refusal.

    The dual-mode test attaches an attestation to records that are already false. This test
    goes the other way: each record starts valid and accepted, a single semantic change is made,
    and the self-attesting adversary recomputes the attestation. Every tampered record must be
    refused on a code of the expected family, with no stale-attestation error to lean on.
    """
    cases = generate_benchmark_cases(include_gauntlet=True)
    valid_v3 = [c for c in cases if not c.is_adversarial and c.initial_corpus is None and evaluate_v3_full_ceg(c)]
    tried: dict[str, int] = {}
    refused_on_expected: dict[str, int] = {}

    def run(kind: str, c: BenchmarkCase, rec: SynthesisClaimRecord, expected_prefix: str) -> None:
        rec_b = populate_claim_attestation(rec, c.graph.atoms, c.graph.arm_aliases)
        rep = c.graph.verify_synthesis_record(rec_b, check_attestation_digest=True)
        tried[kind] = tried.get(kind, 0) + 1
        if (
            not rep.passed_all_invariants
            and not any("I6_STALE_CLAIM_ATTESTATION" in v for v in rep.violations)
            and any(v.startswith(expected_prefix) for v in rep.violations)
        ):
            refused_on_expected[kind] = refused_on_expected.get(kind, 0) + 1

    for c in valid_v3:
        rec = c.record
        anchors = tuple(rec.supporting_tuples) + tuple(rec.counterevidence_tuples)
        lexicon = build_entity_lexicon(c.graph.atoms, c.graph.arm_aliases) | build_endpoint_phrases(c.graph.atoms)
        values: set[str] = set()
        for t in anchors:
            atom = c.graph.atoms[t]
            for q in (atom.focal_quantity, atom.comparator_quantity):
                if q is not None:
                    values |= {normalize_numeric_literal(x) for x in (q.value, q.lower, q.upper) if x is not None}

        # 1. One value in the prose replaced by a number no cited atom holds.
        done = False
        for idx, sent in enumerate(rec.prose_sentences):
            for m in extract_mentions(sent, lexicon):
                head = _NUMERAL_HEAD_RE.match(sent[m.start:m.end])
                if m.value in values and head:
                    forged = sent[:m.start] + head.group(0) + "7" + sent[m.start + head.end():]
                    prose = tuple(forged if j == idx else s for j, s in enumerate(rec.prose_sentences))
                    run("prose_value_fabricated", c, dataclasses.replace(rec, prose_sentences=prose), "I3_")
                    done = True
                    break
            if done:
                break
        _require(done, f"no value mention to tamper in {c.case_id}")

        # 2. An anchor redirected to a tuple that does not exist.
        if rec.supporting_tuples:
            forged_t = (rec.supporting_tuples[0] + "9",) + tuple(rec.supporting_tuples[1:])
            run("anchor_redirected_to_unknown_tuple", c, dataclasses.replace(rec, supporting_tuples=forged_t), "I1_")
        else:
            forged_t = (rec.counterevidence_tuples[0] + "9",) + tuple(rec.counterevidence_tuples[1:])
            run("anchor_redirected_to_unknown_tuple", c, dataclasses.replace(rec, counterevidence_tuples=forged_t), "I1_")

        # 3. One typed proposition given a different quantity.
        if rec.typed_propositions:
            first = rec.typed_propositions[0]
            atom0 = c.graph.atoms[first.outcome_tuple]
            mutated = dataclasses.replace(first, claimed_quantity=_mutate_quantity(first.claimed_quantity, atom0.focal_quantity))
            run(
                "typed_quantity_mutated",
                c,
                dataclasses.replace(rec, typed_propositions=(mutated,) + tuple(rec.typed_propositions[1:])),
                "I3_TYPED_",
            )

        # 4. A supporting tuple also listed as counterevidence.
        if rec.supporting_tuples and rec.supporting_tuples[0] not in rec.counterevidence_tuples:
            run(
                "support_also_listed_as_counterevidence",
                c,
                dataclasses.replace(rec, counterevidence_tuples=tuple(rec.counterevidence_tuples) + (rec.supporting_tuples[0],)),
                "I6_SUPPORT_COUNTEREVIDENCE_OVERLAP",
            )

        # 5. The target context moved outside the declared lattice.
        axes = dict(rec.target_context.axes)
        first_axis = sorted(axes)[0]
        axes[first_axis] = "UNDECLARED_LEVEL"
        run("target_context_outside_lattice", c, dataclasses.replace(rec, target_context=ContextSignature(axes)), "I4_TARGET_CONTEXT_INVALID")

        # 6. Promotion requested by a record that carries no typed proposition.
        if not rec.typed_propositions and rec.requests_direct_quantitative_promotion is False:
            run(
                "promotion_requested_without_typed_layer",
                c,
                dataclasses.replace(rec, requests_direct_quantitative_promotion=True),
                "I3_TYPED_MISSING_FOR_QUANTITATIVE_PROMOTION",
            )

    _require(tried == refused_on_expected, f"a re-attested tamper was accepted or refused on another code: {tried} vs {refused_on_expected}")
    return {
        "valid_records_used": len(valid_v3),
        "tampered_records_tried": sum(tried.values()),
        "refused_on_expected_code_family": sum(refused_on_expected.values()),
        "tried_by_kind": dict(sorted(tried.items())),
        "refused_by_kind": dict(sorted(refused_on_expected.items())),
    }


def _leaf_pointers(node: Any, prefix: str = "") -> list[str]:
    if isinstance(node, dict):
        return [p for k, v in node.items() for p in _leaf_pointers(v, f"{prefix}/{k.replace('~', '~0').replace('/', '~1')}")]
    if isinstance(node, list):
        return [p for i, v in enumerate(node) for p in _leaf_pointers(v, f"{prefix}/{i}")]
    return [prefix]


def _replay_accepts(
    d0: dict[str, Any], dm: dict[str, Any], events: list[ReversiblePatchEvent], *, check_step_digests: bool = True
) -> bool:
    try:
        verify_bijective_patch_replay(d0, dm, events, check_step_digests=check_step_digests)
        return True
    except ValueError:
        return False


def _state_before_event(d0: dict[str, Any], events: list[ReversiblePatchEvent], k: int) -> dict[str, Any]:
    """The corpus as it stands just before event k, obtained by applying events 0..k-1 to a copy of D_0."""
    work = copy.deepcopy(d0)
    for ev in events[:k]:
        node: Any = work
        tokens = [t.replace("~1", "/").replace("~0", "~") for t in ev.json_pointer[1:].split("/")]
        for tok in tokens[:-1]:
            node = node[int(tok)] if isinstance(node, list) else node[tok]
        last: Any = int(tokens[-1]) if isinstance(node, list) else tokens[-1]
        node[last] = copy.deepcopy(ev.new_value)
    return work


def selftest_ledger_tampering() -> dict[str, Any]:
    """Proposition 1: every listed tamper is rejected; commuting transpositions are accepted."""
    out: dict[str, Any] = {}
    for name, bundle in (("drug_development", build_drug_development_vignette()), ("sport_science", build_sport_exercise_vignette())):
        d0, dm, events = bundle.initial_corpus, bundle.settled_corpus, list(bundle.patch_events)
        _require(_replay_accepts(d0, dm, events), f"Clean vignette ledger replay failed for {name}")
        tried: dict[str, int] = {}
        accepted: dict[str, int] = {}

        def attempt(kind: str, d0_: dict[str, Any], dm_: dict[str, Any], evs: list[ReversiblePatchEvent]) -> None:
            tried[kind] = tried.get(kind, 0) + 1
            if _replay_accepts(d0_, dm_, evs):
                accepted[kind] = accepted.get(kind, 0) + 1

        for k in range(len(events)):
            attempt("event_deleted", d0, dm, events[:k] + events[k + 1:])
            ev = events[k]
            parts = ev.json_pointer.split("/")
            for label, ptr in (
                ("pointer_leading_zero", "/".join(parts[:2] + ["0" + parts[2]] + parts[3:])),
                ("pointer_empty_leading_token", "/" + ev.json_pointer),
                ("pointer_negative_index", "/".join(parts[:4] + [str(int(parts[4]) - len(d0["studies"][int(parts[2])]["outcomes"]))] + parts[5:])),
                ("pointer_redirected_to_sibling_field", "/".join(parts[:-1] + ["outcome_id"])),
            ):
                attempt(label, d0, dm, events[:k] + [dataclasses.replace(ev, json_pointer=ptr)] + events[k + 1:])
            attempt("old_value_altered", d0, dm, events[:k] + [dataclasses.replace(ev, old_value="ALTERED")] + events[k + 1:])
            attempt("new_value_altered", d0, dm, events[:k] + [dataclasses.replace(ev, new_value="ALTERED")] + events[k + 1:])
            attempt("pre_digest_altered", d0, dm, events[:k] + [dataclasses.replace(ev, study_sha256_before="0" * 64)] + events[k + 1:])
            attempt("post_digest_altered", d0, dm, events[:k] + [dataclasses.replace(ev, study_sha256_after="0" * 64)] + events[k + 1:])
            attempt("event_duplicated", d0, dm, events[:k + 1] + [ev] + events[k + 1:])
            # The same duplicate under a fresh identifier: the duplicate-identifier check cannot catch it, replay must.
            renamed_copy = dataclasses.replace(ev, event_id=ev.event_id + "-AGAIN")
            attempt("event_duplicated_with_fresh_id", d0, dm, events[:k + 1] + [renamed_copy] + events[k + 1:])
            # Proposition 1(iii)(d): the event redirected to a sibling member of the same study that
            # holds the same old value, with step digests checked and with them switched off.
            state = _state_before_event(d0, events, k)
            study_prefix = "/".join(parts[:3]) + "/"
            for sibling in _leaf_pointers(state):
                if not sibling.startswith(study_prefix + "outcomes/") or sibling == ev.json_pointer:
                    continue
                node_s: Any = state
                tokens_s = [t.replace("~1", "/").replace("~0", "~") for t in sibling[1:].split("/")]
                for tok in tokens_s:
                    node_s = node_s[int(tok)] if isinstance(node_s, list) else node_s[tok]
                if type(node_s) is not type(ev.old_value) or node_s != ev.old_value:
                    continue
                redirected = events[:k] + [dataclasses.replace(ev, json_pointer=sibling)] + events[k + 1:]
                attempt("pointer_redirected_same_old_value", d0, dm, redirected)
                tried["pointer_redirected_same_old_value_no_step_digests"] = tried.get("pointer_redirected_same_old_value_no_step_digests", 0) + 1
                if _replay_accepts(d0, dm, redirected, check_step_digests=False):
                    accepted["pointer_redirected_same_old_value_no_step_digests"] = accepted.get("pointer_redirected_same_old_value_no_step_digests", 0) + 1
            other = next(s for s in d0["studies"] if s["study_id"] != ev.study_id)
            attempt("study_attribution_altered", d0, dm, events[:k] + [dataclasses.replace(ev, study_id=other["study_id"])] + events[k + 1:])

        commuting_tried = commuting_accepted = 0
        for k in range(len(events) - 1):
            swapped = events[:k] + [events[k + 1], events[k]] + events[k + 2:]
            if events[k].study_id == events[k + 1].study_id:
                attempt("same_study_transposition", d0, dm, swapped)
            else:
                commuting_tried += 1
                commuting_accepted += int(_replay_accepts(d0, dm, swapped))

        for ptr in _leaf_pointers(dm):
            tampered = copy.deepcopy(dm)
            node: Any = tampered
            tokens = [t.replace("~1", "/").replace("~0", "~") for t in ptr[1:].split("/")]
            for tok in tokens[:-1]:
                node = node[int(tok)] if isinstance(node, list) else node[tok]
            last: Any = int(tokens[-1]) if isinstance(node, list) else tokens[-1]
            node[last] = "TAMPERED" if node[last] != "TAMPERED" else "TAMPERED2"
            attempt("settled_leaf_altered", d0, tampered, events)
        for ptr in _leaf_pointers(d0):
            tampered = copy.deepcopy(d0)
            node = tampered
            tokens = [t.replace("~1", "/").replace("~0", "~") for t in ptr[1:].split("/")]
            for tok in tokens[:-1]:
                node = node[int(tok)] if isinstance(node, list) else node[tok]
            last = int(tokens[-1]) if isinstance(node, list) else tokens[-1]
            node[last] = "TAMPERED" if node[last] != "TAMPERED" else "TAMPERED2"
            attempt("initial_leaf_altered", tampered, dm, events)

        # Changes that replay accepts by design. The ledger digest must differ for each.
        base_digest = ledger_sha256(events)
        unbound: list[list[ReversiblePatchEvent]] = []
        for k, ev in enumerate(events):
            unbound.append(events[:k] + [dataclasses.replace(ev, rationale=ev.rationale + " (reworded)")] + events[k + 1:])
            unbound.append(events[:k] + [dataclasses.replace(ev, event_id=ev.event_id + "-RENAMED")] + events[k + 1:])
        last = events[-1]
        st_idx = int(last.json_pointer.split("/")[2])
        h_m = sha256_canonical(dm["studies"][st_idx])
        detour = copy.deepcopy(dm["studies"][st_idx])
        detour["outcomes"][0]["outcome_id"] = "O1-DETOUR"
        original_id = dm["studies"][st_idx]["outcomes"][0]["outcome_id"]
        unbound.append(events + [
            ReversiblePatchEvent("PAD-1", last.study_id, f"/studies/{st_idx}/outcomes/0/outcome_id", original_id, "O1-DETOUR", "edit", h_m, sha256_canonical(detour)),
            ReversiblePatchEvent("PAD-2", last.study_id, f"/studies/{st_idx}/outcomes/0/outcome_id", "O1-DETOUR", original_id, "revert", sha256_canonical(detour), h_m),
        ])
        unbound_accepted = sum(int(_replay_accepts(d0, dm, evs)) for evs in unbound)
        unbound_digest_changed = sum(int(ledger_sha256(evs) != base_digest) for evs in unbound)

        out[name] = {
            "events": len(events),
            "replay_unbound_changes_tried": len(unbound),
            "replay_unbound_changes_accepted_by_replay": unbound_accepted,
            "replay_unbound_changes_detected_by_ledger_digest": unbound_digest_changed,
            "tampers_tried": sum(tried.values()),
            "tampers_accepted": sum(accepted.values()),
            "tampers_tried_by_kind": dict(sorted(tried.items())),
            "tampers_accepted_by_kind": dict(sorted(accepted.items())),
            "commuting_transpositions_tried": commuting_tried,
            "commuting_transpositions_accepted": commuting_accepted,
        }

    # A corpus whose studies share one Python container: an edit guarded by the first study must not
    # reach the second. This is one constructed corpus, independent of the vignettes, so it is run once.
    shared = [{"outcome_id": "O1", "focal_value": "1.0"}]
    aliased = {"studies": [{"study_id": "A", "outcomes": shared}, {"study_id": "B", "outcomes": shared}]}
    aliased_after = copy.deepcopy(aliased)
    h_before = sha256_canonical(aliased_after["studies"][0])
    aliased_after["studies"][0]["outcomes"][0]["focal_value"] = "2.0"
    alias_accepted = _replay_accepts(aliased, aliased_after, [ReversiblePatchEvent(
        "E-ALIAS", "A", "/studies/0/outcomes/0/focal_value", "1.0", "2.0", "edit through a shared container",
        h_before, sha256_canonical(aliased_after["studies"][0]),
    )])
    out["shared_container_probe"] = {"tampers_tried": 1, "tampers_accepted": int(alias_accepted)}
    return out


def _hostile_name_graph() -> ContextEvidenceGraph:
    """Valid atoms whose endpoint names and units contain direction words, digits and superscripts."""
    bundle = build_drug_development_vignette()
    ctx = bundle.target_context
    arms = ("VAR:H1:Active", "VAR:H1:Control")
    specs = [
        ("O1", "Proportion with reduced exacerbations", "62.0", "31.0", "%", "proportion"),
        ("O2", "Lower-limb strength", "120.0", "100.0", "N", "mean"),
        ("O3", "Peak VO\u2082 uptake", "41.5", "38.0", "mL/kg/min", "mean"),
        ("O4", "Filtration rate per 1.73 m\u00b2", "74.0", "68.5", "mL/min", "mean"),
        ("O5", "FEV1 percent predicted", "62.0", "55.0", "% predicted", "mean"),
        ("O6", "Platelet count increase over baseline", "250", "240", "x10^9/L", "mean"),
        ("O7", "Higher-dose 3x ULN elevation incidence", "4.0", "9.0", "%", "proportion"),
    ]
    atoms = {
        f"H1:{o}": EvidenceAtom(
            atom_id=f"ATOM:H1:{o}", outcome_tuple=f"H1:{o}", study_id="H1", endpoint=ep, context=ctx,
            contrast=MultiArmContrast((arms[0],), (arms[1],), "pairwise"),
            focal_quantity=QuantitySpec(kind="point_level", relation="=", value=fv, unit=unit, unit_dimension="unspecified"),
            comparator_quantity=QuantitySpec(kind="point_level", relation="=", value=cv, unit=unit, unit_dimension="unspecified"),
            statistic_kind=stat,  # type: ignore[arg-type]
            sample_size=40, p_value_or_interval="p=0.04", polarity="unresolved",
            provenance=ProvenancePin("SYNTHETIC-DOC:H1", "Table 1", ep, False, True),
        )
        for o, ep, fv, cv, unit, stat in specs
    }
    return ContextEvidenceGraph(
        lattice=bundle.graph.lattice, studies={"H1": ctx}, study_variables={"H1": set(arms)},
        arm_aliases={arms[0]: ("active arm",), arms[1]: ("control arm",)}, atoms=atoms,
        raw_mentioned_arms_by_tuple={t: arms for t in atoms},
    )


def selftest_canonical_rendering() -> dict[str, Any]:
    """Every atom's canonical sentence passes the strict prose lint."""
    hostile = _hostile_name_graph()
    _require(hostile.verify_graph_integrity() == [], "Hostile-name graph integrity failed")
    hostile_accepted = sum(
        int(not verify_prose_sentence(render_atom_sentence(a), hostile.atoms, hostile.arm_aliases, pooled_numeral_policy="reject"))
        for a in hostile.atoms.values()
    )
    total = accepted = 0
    graphs = [build_drug_development_vignette().graph, build_sport_exercise_vignette().graph]
    graphs += [_build_parametric_base_graph(i)[0] for i in range(0, 1000, 37)]
    for g in graphs:
        for atom in g.atoms.values():
            total += 1
            if not verify_prose_sentence(render_atom_sentence(atom), g.atoms, g.arm_aliases, pooled_numeral_policy="reject"):
                accepted += 1
    return {
        "atoms_rendered": total,
        "renderings_accepted_by_strict_lint": accepted,
        "hostile_name_atoms_rendered": len(hostile.atoms),
        "hostile_name_renderings_accepted": hostile_accepted,
    }


def selftest_dual_mode_and_non_prose_lock() -> dict[str, Any]:
    """Mode A (raw tamper) vs Mode B (self-attesting adversary) and 100% non-prose surface locks."""
    cases = generate_benchmark_cases(include_gauntlet=True)
    record_adv = [
        c for c in cases
        if c.is_adversarial
        and c.suite != "residual"
        and not c.graph.verify_graph_integrity()
        and c.initial_corpus is None
    ]
    mode_a_rejected = mode_a_with_stale_and_semantic = 0
    mode_b_rejected = mode_b_semantic_only = mode_b_accepted = mode_b_stale_errors = 0

    for c in record_adv:
        rec_a = dataclasses.replace(c.record, attestation_sha256="0" * 64)
        rep_a = c.graph.verify_synthesis_record(rec_a, check_attestation_digest=True)
        if not rep_a.passed_all_invariants:
            mode_a_rejected += 1
        has_stale_a = any("I6_STALE_CLAIM_ATTESTATION" in v for v in rep_a.violations)
        has_sem_a = any("I6_STALE_CLAIM_ATTESTATION" not in v for v in rep_a.violations)
        if has_stale_a and has_sem_a:
            mode_a_with_stale_and_semantic += 1

        rec_b = populate_claim_attestation(c.record, c.graph.atoms, c.graph.arm_aliases)
        rep_b = c.graph.verify_synthesis_record(rec_b, check_attestation_digest=True)
        if rep_b.passed_all_invariants:
            mode_b_accepted += 1
        else:
            mode_b_rejected += 1
        has_stale_b = any("I6_STALE_CLAIM_ATTESTATION" in v for v in rep_b.violations)
        if has_stale_b:
            mode_b_stale_errors += 1
        elif rep_b.violations:
            mode_b_semantic_only += 1

    _require(mode_a_rejected == len(record_adv), f"Mode A accepted adversarial cases: {len(record_adv) - mode_a_rejected}")
    _require(mode_a_with_stale_and_semantic == len(record_adv), "Mode A short-circuited semantic checks on stale attestation")
    _require(mode_b_accepted == 0, f"Mode B self-attesting adversary accepted {mode_b_accepted} cases")
    _require(mode_b_stale_errors == 0, f"Mode B produced {mode_b_stale_errors} stale attestation errors")
    _require(mode_b_semantic_only == len(record_adv), "Mode B failed to produce semantic error codes on every case")

    valid_v3 = [c for c in cases if not c.is_adversarial and evaluate_v3_full_ceg(c)]
    valid_attested_accepted = 0
    for c in valid_v3:
        rec_v = populate_claim_attestation(c.record, c.graph.atoms, c.graph.arm_aliases)
        if c.graph.verify_synthesis_record(rec_v, check_attestation_digest=True).passed_all_invariants:
            valid_attested_accepted += 1
    _require(valid_attested_accepted == len(valid_v3), "Attested valid V3 records failed verification")

    non_prose_tampers_tried = non_prose_tampers_rejected = prose_key_exempt_verified = 0
    for bundle in (build_drug_development_vignette(), build_sport_exercise_vignette()):
        for rec in bundle.valid_records:
            rec_att = populate_claim_attestation(rec, bundle.graph.atoms, bundle.graph.arm_aliases)
            env = record_to_envelope(
                rec_att,
                extra_metadata={
                    "audit_note": "verified",
                    "reviewer_tag": "R1",
                    "summary_banner": "approved",
                },
            )
            expected_lock = non_prose_surface_sha256(env)
            _require(verify_non_prose_surface_lock(env, expected_lock) == [], "Clean non-prose surface lock failed")

            prose_mut = copy.deepcopy(env)
            prose_mut["prose_sentences"] = ["Mutated prose checked by active prose scanner."]
            if verify_non_prose_surface_lock(prose_mut, expected_lock) == []:
                prose_key_exempt_verified += 1

            for key in sorted(env):
                if key in ACTIVE_PROSE_KEYS:
                    continue
                mut = copy.deepcopy(env)
                mut[key] = "SMUGGLED_UNVERIFIED_CLAIM" if mut[key] != "SMUGGLED_UNVERIFIED_CLAIM" else "SMUGGLED_ALT"
                non_prose_tampers_tried += 1
                errs = verify_non_prose_surface_lock(mut, expected_lock)
                if any("I6_NON_PROSE_METADATA_TAMPER" in e for e in errs):
                    non_prose_tampers_rejected += 1

            for rogue_key in ("smuggled_claim", "unverified_footnote", "shadow_conclusion"):
                mut = copy.deepcopy(env)
                mut[rogue_key] = "Drug X cured 100% of patients without adverse events."
                non_prose_tampers_tried += 1
                errs = verify_non_prose_surface_lock(mut, expected_lock)
                if any("I6_NON_PROSE_METADATA_TAMPER" in e for e in errs):
                    non_prose_tampers_rejected += 1

    _require(non_prose_tampers_rejected == non_prose_tampers_tried, "Non-prose surface lock missed a metadata tamper")
    return {
        "record_adversarial_cases_tested": len(record_adv),
        "mode_a_rejected": mode_a_rejected,
        "mode_a_stale_plus_semantic_verified": mode_a_with_stale_and_semantic,
        "mode_b_self_attesting_rejected": mode_b_rejected,
        "mode_b_self_attesting_accepted": mode_b_accepted,
        "mode_b_semantic_only_verified": mode_b_semantic_only,
        "valid_v3_attested_records_tested": len(valid_v3),
        "valid_v3_attested_records_accepted": valid_attested_accepted,
        "non_prose_metadata_tampers_tried": non_prose_tampers_tried,
        "non_prose_metadata_tampers_rejected": non_prose_tampers_rejected,
        "active_prose_exemptions_verified": prose_key_exempt_verified,
    }


def selftest_structure_only_and_colon_ids() -> dict[str, Any]:
    """Layer-1 structure-only (`check_top_digests=False`) vs Layer-2 digest-locked verification and colon-safe IDs."""
    bundle_summaries: dict[str, Any] = {}
    for name, bundle in (
        ("drug_development", build_drug_development_vignette()),
        ("sport_science", build_sport_exercise_vignette()),
    ):
        att_records = [
            populate_claim_attestation(r, bundle.graph.atoms, bundle.graph.arm_aliases)
            for r in bundle.valid_records
        ]
        envelopes = [record_to_envelope(r, extra_metadata={"vignette": name}) for r in att_records]
        sec_digests = compute_ceg_section_digests(
            bundle.graph,
            bundle.initial_corpus,
            bundle.settled_corpus,
            bundle.patch_events,
            att_records,
            record_envelopes=envelopes,
        )
        res_l1 = verify_ceg_package(
            bundle.graph,
            bundle.initial_corpus,
            bundle.settled_corpus,
            bundle.patch_events,
            att_records,
            section_digests=sec_digests,
            record_envelopes=envelopes,
            check_top_digests=False,
        )
        res_l2 = verify_ceg_package(
            bundle.graph,
            bundle.initial_corpus,
            bundle.settled_corpus,
            bundle.patch_events,
            att_records,
            section_digests=sec_digests,
            record_envelopes=envelopes,
            check_top_digests=True,
        )
        _require(res_l1["passed"] and res_l2["passed"], f"Clean package verification failed on {name}")

        zero_step_events = [
            dataclasses.replace(ev, study_sha256_before="0" * 64, study_sha256_after="0" * 64)
            for ev in bundle.patch_events
        ]
        res_zero_l1 = verify_ceg_package(
            bundle.graph,
            bundle.initial_corpus,
            bundle.settled_corpus,
            zero_step_events,
            [dataclasses.replace(r, attestation_sha256="0" * 64) for r in att_records],
            section_digests={k: "0" * 64 for k in sec_digests},
            record_envelopes=envelopes,
            check_top_digests=False,
        )
        _require(res_zero_l1["passed"], f"Structure-only Layer 1 relied on precomputed digests on {name}")

        structural_tried = structural_rejected_l1 = 0

        def check_structural_tamper(
            g_: ContextEvidenceGraph,
            d0_: dict[str, Any],
            dm_: dict[str, Any],
            evs_: list[ReversiblePatchEvent],
            recs_: list[SynthesisClaimRecord],
            envs_: list[dict[str, Any]] | None = None,
        ) -> None:
            nonlocal structural_tried, structural_rejected_l1
            structural_tried += 1
            rep = verify_ceg_package(
                g_, d0_, dm_, evs_, recs_, record_envelopes=envs_, check_top_digests=False
            )
            if not rep["passed"]:
                structural_rejected_l1 += 1

        # 1. Catalogue outcome dropped from settled_corpus
        dm_drop_oc = copy.deepcopy(bundle.settled_corpus)
        dm_drop_oc["studies"][0]["outcomes"] = dm_drop_oc["studies"][0]["outcomes"][1:]
        check_structural_tamper(bundle.graph, bundle.initial_corpus, dm_drop_oc, list(bundle.patch_events), att_records)

        # 2. Orphan study in graph.studies absent from settled_corpus
        g_orphan = copy.deepcopy(bundle.graph)
        g_orphan.studies["ORPHAN_STUDY"] = bundle.target_context
        g_orphan.study_variables["ORPHAN_STUDY"] = set()
        check_structural_tamper(g_orphan, bundle.initial_corpus, bundle.settled_corpus, list(bundle.patch_events), att_records)

        # 3. Atom value drift relative to settled_corpus
        g_val = copy.deepcopy(bundle.graph)
        first_tup = sorted(g_val.atoms)[0]
        first_atom = g_val.atoms[first_tup]
        g_val.atoms[first_tup] = dataclasses.replace(
            first_atom,
            focal_quantity=dataclasses.replace(first_atom.focal_quantity, value="9999.9"),
        )
        check_structural_tamper(g_val, bundle.initial_corpus, bundle.settled_corpus, list(bundle.patch_events), att_records)

        # 4. Dangling study context on an atom
        g_dang = copy.deepcopy(bundle.graph)
        g_dang.studies.pop(first_atom.study_id, None)
        check_structural_tamper(g_dang, bundle.initial_corpus, bundle.settled_corpus, list(bundle.patch_events), att_records)

        # 5. Dangling anchor tuple in a synthesis record
        bad_anchor_rec = dataclasses.replace(att_records[0], supporting_tuples=("MISSING_STUDY:O99",))
        check_structural_tamper(bundle.graph, bundle.initial_corpus, bundle.settled_corpus, list(bundle.patch_events), [bad_anchor_rec])

        # 6. Ledger value transition tamper (with zeroed step digests)
        bad_ev = dataclasses.replace(zero_step_events[0], new_value="TAMPERED_VALUE")
        check_structural_tamper(bundle.graph, bundle.initial_corpus, bundle.settled_corpus, [bad_ev] + zero_step_events[1:], att_records)

        # 7. All semantic invalid probes from the vignette
        for _, bad_rec, _ in bundle.invalid_probe_records:
            check_structural_tamper(bundle.graph, bundle.initial_corpus, bundle.settled_corpus, list(bundle.patch_events), [bad_rec])

        # 8. Orphan corpus study with zero outcomes absent from graph.studies
        d0_orphan_st = copy.deepcopy(bundle.initial_corpus)
        dm_orphan_st = copy.deepcopy(bundle.settled_corpus)
        for doc_st in (d0_orphan_st, dm_orphan_st):
            doc_st["studies"].append({"study_id": "ORPHAN_CORPUS_STUDY", "outcomes": []})
        check_structural_tamper(bundle.graph, d0_orphan_st, dm_orphan_st, list(bundle.patch_events), att_records)

        # 9. Non-boolean provenance flag ("false" string) in atom and corpus
        g_str_bool = copy.deepcopy(bundle.graph)
        sup_tup = att_records[0].supporting_tuples[0]
        sup_atom = g_str_bool.atoms[sup_tup]
        g_str_bool.atoms[sup_tup] = dataclasses.replace(
            sup_atom,
            provenance=dataclasses.replace(sup_atom.provenance, primary_pdf_verified="false"),  # type: ignore[arg-type]
        )
        d0_str_bool = copy.deepcopy(bundle.initial_corpus)
        dm_str_bool = copy.deepcopy(bundle.settled_corpus)
        sup_sid, sup_oid = split_outcome_tuple(sup_tup)
        for doc_b in (d0_str_bool, dm_str_bool):
            for st_b in doc_b["studies"]:
                if st_b["study_id"] == sup_sid:
                    for oc_b in st_b["outcomes"]:
                        if oc_b["outcome_id"] == sup_oid:
                            oc_b["primary_pdf_verified"] = "false"
        rec_str_bool = populate_claim_attestation(att_records[0], g_str_bool.atoms, g_str_bool.arm_aliases)
        check_structural_tamper(g_str_bool, d0_str_bool, dm_str_bool, list(bundle.patch_events), [rec_str_bool] + att_records[1:])

        # 10. Supplied record_envelopes decoupled from verified records (weakened gate threshold or renamed ID + Mode B re-attestation)
        gate_indices = [i for i, r in enumerate(att_records) if r.prospective_gate is not None]
        stale_env_recs = list(att_records)
        if gate_indices:
            gate_idx = gate_indices[0]
            gate_rec = att_records[gate_idx]
            assert_gate = gate_rec.prospective_gate
            if assert_gate is None:
                raise RuntimeError("Expected prospective gate on bridge record")
            weakened_gate = dataclasses.replace(assert_gate, quantitative_pass_threshold=">=0% unverified threshold")
            stale_env_recs[gate_idx] = populate_claim_attestation(
                dataclasses.replace(gate_rec, prospective_gate=weakened_gate, attestation_sha256=None),
                bundle.graph.atoms,
                bundle.graph.arm_aliases,
            )
        else:
            stale_env_recs[0] = populate_claim_attestation(
                dataclasses.replace(att_records[0], record_id="RENAMED_RECORD", attestation_sha256=None),
                bundle.graph.atoms,
                bundle.graph.arm_aliases,
            )
        check_structural_tamper(
            bundle.graph, bundle.initial_corpus, bundle.settled_corpus, list(bundle.patch_events), stale_env_recs, envelopes
        )
        res_stale_l2 = verify_ceg_package(
            bundle.graph,
            bundle.initial_corpus,
            bundle.settled_corpus,
            bundle.patch_events,
            stale_env_recs,
            section_digests=sec_digests,
            record_envelopes=envelopes,
            check_top_digests=True,
        )
        _require(not res_stale_l2["passed"], f"Layer 2 accepted stale record_envelopes on {name}")

        # 11. Supplied record_envelopes count mismatch against verified records
        check_structural_tamper(
            bundle.graph,
            bundle.initial_corpus,
            bundle.settled_corpus,
            list(bundle.patch_events),
            att_records + [att_records[0]],
            envelopes,
        )

        _require(
            structural_rejected_l1 == structural_tried,
            f"Structure-only Layer 1 missed {structural_tried - structural_rejected_l1} structural/semantic tampers on {name}",
        )

        # Non-structural metadata tampers: accepted by Layer 1, rejected by Layer 2
        ev0 = bundle.patch_events[0]
        last = bundle.patch_events[-1]
        st_idx = int(last.json_pointer.split("/")[2])
        h_m = sha256_canonical(bundle.settled_corpus["studies"][st_idx])
        detour = copy.deepcopy(bundle.settled_corpus["studies"][st_idx])
        orig_oid = detour["outcomes"][0]["outcome_id"]
        detour["outcomes"][0]["outcome_id"] = "O1-DETOUR"
        detour_evs = list(bundle.patch_events) + [
            ReversiblePatchEvent("PAD-1", last.study_id, f"/studies/{st_idx}/outcomes/0/outcome_id", orig_oid, "O1-DETOUR", "edit", h_m, sha256_canonical(detour)),
            ReversiblePatchEvent("PAD-2", last.study_id, f"/studies/{st_idx}/outcomes/0/outcome_id", "O1-DETOUR", orig_oid, "revert", sha256_canonical(detour), h_m),
        ]
        rogue_envelopes = copy.deepcopy(envelopes)
        rogue_envelopes[0]["unverified_banner"] = "Claim promoted without prospective gate."

        layer2_only_cases = [
            (list(bundle.patch_events[:0]) + [dataclasses.replace(ev0, rationale=ev0.rationale + " (edited)")] + list(bundle.patch_events[1:]), att_records, envelopes),
            ([dataclasses.replace(ev0, event_id=ev0.event_id + "-ALT")] + list(bundle.patch_events[1:]), att_records, envelopes),
            (detour_evs, att_records, envelopes),
            (list(bundle.patch_events), [dataclasses.replace(att_records[0], attestation_sha256="f" * 64)] + att_records[1:], envelopes),
            (list(bundle.patch_events), att_records, rogue_envelopes),
        ]
        l1_accepted = l2_rejected = 0
        for evs_, recs_, envs_ in layer2_only_cases:
            r1 = verify_ceg_package(
                bundle.graph, bundle.initial_corpus, bundle.settled_corpus, evs_, recs_,
                section_digests=sec_digests, record_envelopes=envs_, check_top_digests=False,
            )
            r2 = verify_ceg_package(
                bundle.graph, bundle.initial_corpus, bundle.settled_corpus, evs_, recs_,
                section_digests=sec_digests, record_envelopes=envs_, check_top_digests=True,
            )
            l1_accepted += int(r1["passed"])
            l2_rejected += int(not r2["passed"])
        _require(l1_accepted == len(layer2_only_cases) and l2_rejected == len(layer2_only_cases), f"Layer 1 / Layer 2 separation failed on {name}")

        bundle_summaries[name] = {
            "structural_tampers_tried": structural_tried,
            "structural_tampers_rejected_without_digests": structural_rejected_l1,
            "layer2_metadata_tampers_tried": len(layer2_only_cases),
            "layer2_metadata_tampers_accepted_by_layer1": l1_accepted,
            "layer2_metadata_tampers_rejected_by_layer2": l2_rejected,
        }

    # Colon-safe outcome tuple splitting & TUPLE_RE extraction
    colon_vectors = [
        ("PMID:24836310:O1", ("PMID:24836310", "O1")),
        ("DOI:10.1056/NEJMoa1402582:O1", ("DOI:10.1056/NEJMoa1402582", "O1")),
        ("NCT01335464:SUBGROUP_A:O2", ("NCT01335464:SUBGROUP_A", "O2")),
        ("STUDY_ALPHA:O10b", ("STUDY_ALPHA", "O10b")),
    ]
    colon_ok = 0
    for raw_tup, expected_pair in colon_vectors:
        got_pair = split_outcome_tuple(raw_tup)
        extracted = TUPLE_RE.findall(f"In {raw_tup}, the primary endpoint was met ({raw_tup}).")
        if got_pair == expected_pair and extracted == [raw_tup, raw_tup]:
            colon_ok += 1
    _require(colon_ok == len(colon_vectors), "Colon-safe outcome tuple parsing failed")

    # Conservative contrast-arm binding & negated inequality bound parsing
    aliases = {
        "VAR:S:ArmA": ("nintedanib", "arm a"),
        "VAR:S:ArmB": ("placebo", "arm b"),
        "VAR:S:ArmC": ("pirfenidone", "arm c"),
    }
    arm_vectors = [
        ("Nintedanib lowered decline versus placebo.", ("VAR:S:ArmA", "VAR:S:ArmB")),
        ("Compared with placebo, nintedanib slowed decline.", ("VAR:S:ArmA", "VAR:S:ArmB")),
        ("Nintedanib and placebo cohorts were enrolled.", (None, None)),
        ("Nintedanib (placebo cohort archived) was assayed.", (None, None)),
        ("Nintedanib versus placebo versus pirfenidone was compared.", (None, None)),
    ]
    arm_ok = sum(int(bind_contrast_arms_conservatively(txt, aliases) == exp) for txt, exp in arm_vectors)
    _require(arm_ok == len(arm_vectors), "Conservative multi-arm binding check failed")

    bound_vectors = [
        ("not < 5%", ("censored_observation", ">=", "5", "%")),
        ("not <= 5%", ("censored_observation", ">", "5", "%")),
        ("never > 10%", ("censored_observation", "<=", "10", "%")),
        ("never >= 10%", ("censored_observation", "<", "10", "%")),
        ("not fewer than 5 days", ("censored_observation", ">=", "5", "days")),
        ("never less than 7 d", ("censored_observation", ">=", "7", "days")),
        ("not 5%", ("missing_numeric", None, None, None)),
        ("not never < 5%", ("missing_numeric", None, None, None)),
    ]
    bound_ok = 0
    for frag, exp_q in bound_vectors:
        q = parse_prose_quantity_bound(frag)
        if (q.kind, q.relation, q.value, q.unit) == exp_q:
            bound_ok += 1
    _require(bound_ok == len(bound_vectors), f"Quantity bound parser check failed: {bound_ok}/{len(bound_vectors)}")

    return {
        "bundles": bundle_summaries,
        "colon_tuple_vectors_tested": len(colon_vectors),
        "colon_tuple_vectors_passed": colon_ok,
        "arm_binding_vectors_tested": len(arm_vectors),
        "arm_binding_vectors_passed": arm_ok,
        "negated_bound_vectors_tested": len(bound_vectors),
        "negated_bound_vectors_passed": bound_ok,
    }


def selftest_source_pinned_pyc_and_hygiene() -> dict[str, Any]:
    """Source-pinned module loader vs poisoned PYTHONPYCACHEPREFIX `.pyc` and package hygiene checks."""
    ceg_ref_dir = str(Path(__file__).resolve().parent)
    with tempfile.TemporaryDirectory() as tmp:
        tmp_root = Path(tmp)
        mod_dir = tmp_root / "pkg"
        cache_dir = tmp_root / "pycache_poison"
        outside_dir = tmp_root / "outside"
        mod_dir.mkdir()
        cache_dir.mkdir()
        outside_dir.mkdir()

        # Both strings have identical byte length (34 bytes) so mtime+size `.pyc` header matches!
        poison_src = b"SENTINEL = 'POISONED_BYTECODE_XX'\n"
        authentic_src = b"SENTINEL = 'AUTHENTIC_SOURCE_V3!'\n"
        _require(len(poison_src) == len(authentic_src), "Sentinel byte lengths must match for stealth `.pyc` test")

        target_py = mod_dir / "pinned_demo.py"
        target_py.write_bytes(poison_src)
        fixed_ns = 1_700_000_000_000_000_000
        os.utime(target_py, ns=(fixed_ns, fixed_ns))

        base_env = {k: v for k, v in os.environ.items() if k != "PYTHONDONTWRITEBYTECODE"}
        base_env["PYTHONPYCACHEPREFIX"] = str(cache_dir)
        base_env["PYTHONPATH"] = os.pathsep.join([str(mod_dir), ceg_ref_dir])

        # Step 1: compile poisoned `.pyc` into PYTHONPYCACHEPREFIX via standard import without `-B`
        warm = subprocess.run(
            [sys.executable, "-c", "import pinned_demo; print(pinned_demo.SENTINEL)"],
            env=base_env,
            capture_output=True,
            text=True,
            check=True,
        )
        _require(warm.stdout.strip() == "POISONED_BYTECODE_XX", "Failed to warm PYTHONPYCACHEPREFIX cache")

        # Step 2: overwrite `.py` with authentic bytes of identical size & mtime so standard import stays fooled
        target_py.write_bytes(authentic_src)
        os.utime(target_py, ns=(fixed_ns, fixed_ns))
        authentic_sha = hashlib.sha256(authentic_src).hexdigest()

        probe_code = (
            "import json, sys\n"
            "import pinned_demo\n"
            "from ceg_reference import load_source_pinned_module\n"
            "mod = load_source_pinned_module('pinned_clean', sys.argv[1], expected_sha256=sys.argv[2], root_dir=sys.argv[3])\n"
            "print(json.dumps({'standard_import': pinned_demo.SENTINEL, 'source_pinned': mod.SENTINEL, 'cached_is_none': mod.__cached__ is None}))\n"
        )
        probe = subprocess.run(
            [sys.executable, "-c", probe_code, "pinned_demo.py", authentic_sha, str(mod_dir)],
            env=base_env,
            capture_output=True,
            text=True,
            check=True,
        )
        parsed_probe = json.loads(probe.stdout.strip())
        _require(parsed_probe["standard_import"] == "POISONED_BYTECODE_XX", "Stealth `.pyc` poison control did not trigger on standard import")
        _require(parsed_probe["source_pinned"] == "AUTHENTIC_SOURCE_V3!", "load_source_pinned_module failed to bypass poisoned `.pyc`")
        _require(parsed_probe["cached_is_none"] is True, "load_source_pinned_module left non-None __cached__")

        # Step 3: SHA-256 mismatch and symlink escape must be rejected by load_source_pinned_module
        loader_rejections = 0
        try:
            load_source_pinned_module("bad_sha", "pinned_demo.py", expected_sha256="0" * 64, root_dir=mod_dir)
        except ValueError:
            loader_rejections += 1
        outside_py = outside_dir / "escape.py"
        outside_py.write_text("X = 1\n", encoding="utf-8")
        sym_py = mod_dir / "sym_escape.py"
        sym_py.symlink_to(outside_py)
        try:
            load_source_pinned_module("bad_sym", "sym_escape.py", root_dir=mod_dir)
        except ValueError:
            loader_rejections += 1
        _require(loader_rejections == 2, "load_source_pinned_module missed SHA-256 or symlink rejection")

        # Step 4: Single-read JSON & chunk/package hygiene checks
        valid_json = mod_dir / "valid.json"
        valid_bytes = b'{"status": "ok", "count": 3}\n'
        valid_json.write_bytes(valid_bytes)
        valid_sha = hashlib.sha256(valid_bytes).hexdigest()
        loaded_obj, got_sha, got_len = load_json_single_read("valid.json", expected_sha256=valid_sha, root_dir=mod_dir)
        _require(loaded_obj == {"status": "ok", "count": 3} and got_sha == valid_sha and got_len == len(valid_bytes), "load_json_single_read failed on valid JSON")

        clean_hygiene = verify_chunk_or_package_hygiene(
            mod_dir,
            [{"path": "valid.json", "utf8_bytes": len(valid_bytes), "sha256": valid_sha, "require_json": True}],
            byte_budget=1024,
            token_budget=64,
        )
        _require(clean_hygiene == [], f"Clean hygiene manifest rejected: {clean_hygiene}")

        nonfinite_rejected = 0
        for idx, bad_payload in enumerate((b'{"x": NaN}', b'{"x": Infinity}', b'{"x": -Infinity}')):
            p = mod_dir / f"nonfinite_{idx}.json"
            p.write_bytes(bad_payload)
            try:
                load_json_single_read(p.name, root_dir=mod_dir)
            except ValueError:
                nonfinite_rejected += 1
        _require(nonfinite_rejected == 3, "load_json_single_read accepted NaN/Infinity")

        bad_utf8 = mod_dir / "bad_utf8.json"
        bad_utf8.write_bytes(b"\xff\xfe\xfd")
        outside_json = outside_dir / "secret.json"
        outside_json.write_bytes(b'{"secret": 1}')
        sym_json = mod_dir / "sym_secret.json"
        sym_json.symlink_to(outside_json)

        hygiene_adversarial_runs = [
            verify_chunk_or_package_hygiene(mod_dir, [], byte_budget=0),
            verify_chunk_or_package_hygiene(mod_dir, [], byte_budget=-10),
            verify_chunk_or_package_hygiene(mod_dir, [], byte_budget=1024, token_budget=0),
            verify_chunk_or_package_hygiene(mod_dir, [], byte_budget=10.5),  # type: ignore[arg-type]
            verify_chunk_or_package_hygiene(mod_dir, [{"path": "valid.json", "utf8_bytes": len(valid_bytes) + 1}], byte_budget=1024),
            verify_chunk_or_package_hygiene(mod_dir, [{"path": "valid.json", "sha256": "0" * 64}], byte_budget=1024),
            verify_chunk_or_package_hygiene(mod_dir, [{"path": "valid.json"}], byte_budget=8),
            verify_chunk_or_package_hygiene(mod_dir, [{"path": "valid.json"}], byte_budget=1024, token_budget=1),
            verify_chunk_or_package_hygiene(mod_dir, [{"path": "bad_utf8.json"}], byte_budget=1024),
            verify_chunk_or_package_hygiene(mod_dir, [{"path": "nonfinite_0.json"}], byte_budget=1024),
            verify_chunk_or_package_hygiene(mod_dir, [{"path": "../outside/secret.json"}], byte_budget=1024),
            verify_chunk_or_package_hygiene(mod_dir, [{"path": str(outside_json)}], byte_budget=1024),
            verify_chunk_or_package_hygiene(mod_dir, [{"path": "sym_secret.json"}], byte_budget=1024),
        ]
        hygiene_rejected = sum(int(bool(errs)) for errs in hygiene_adversarial_runs)
        _require(hygiene_rejected == len(hygiene_adversarial_runs), "Hygiene verifier missed an adversarial manifest case")

    return {
        "poisoned_pyc_bypassed_without_minus_B": True,
        "module_cached_attr_is_none": True,
        "loader_tamper_probes_tried": 2,
        "loader_tamper_probes_rejected": loader_rejections,
        "nonfinite_json_constants_tried": 3,
        "nonfinite_json_constants_rejected": nonfinite_rejected,
        "hygiene_adversarial_cases_tried": len(hygiene_adversarial_runs),
        "hygiene_adversarial_cases_rejected": hygiene_rejected,
    }


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def _summarize(cases: list[BenchmarkCase], outcomes: dict[str, dict[str, bool]]) -> dict[str, dict[str, Any]]:
    summary: dict[str, dict[str, Any]] = {}
    adv = [c for c in cases if c.is_adversarial]
    val = [c for c in cases if not c.is_adversarial]
    for v_name in VERIFIERS:
        tp = sum(1 for c in adv if not outcomes[v_name][c.case_id])
        tn = sum(1 for c in val if outcomes[v_name][c.case_id])
        entry: dict[str, Any] = {
            "adversarial_total": len(adv),
            "adversarial_rejected": tp,
            "adversarial_accepted": len(adv) - tp,
            "valid_total": len(val),
            "valid_accepted": tn,
            "valid_rejected": len(val) - tn,
        }
        if adv:
            entry["adversarial_rejection_rate_pct"] = round(100.0 * tp / len(adv), 2)
        if val:
            entry["valid_acceptance_rate_pct"] = round(100.0 * tn / len(val), 2)
        summary[v_name] = entry
    return summary


def run_benchmark(write_json: bool = True) -> dict[str, Any]:
    vignettes = {
        "drug_development": build_drug_development_vignette().run_all_checks(),
        "sport_science": build_sport_exercise_vignette().run_all_checks(),
    }
    cases = generate_benchmark_cases(include_gauntlet=True)
    _require(len({c.case_id for c in cases}) == len(cases), "case identifiers must be unique")

    outcomes: dict[str, dict[str, bool]] = {v: {} for v in VERIFIERS}
    for v_name, v_fn in VERIFIERS.items():
        for c in cases:
            outcomes[v_name][c.case_id] = v_fn(c)

    per_stratum: dict[str, dict[str, Any]] = {}
    for c in cases:
        st = per_stratum.setdefault(
            c.stratum,
            {"suite": c.suite, "is_adversarial": c.is_adversarial, "total_cases": 0, "templates": {}, "by_verifier": {}},
        )
        st["total_cases"] += 1
        st["templates"].setdefault(c.template, 0)
        st["templates"][c.template] += 1
        for v_name in VERIFIERS:
            v_st = st["by_verifier"].setdefault(v_name, {"accepted": 0, "rejected": 0, "accepted_templates": set(), "rejected_templates": set()})
            if outcomes[v_name][c.case_id]:
                v_st["accepted"] += 1
                v_st["accepted_templates"].add(c.template)
            else:
                v_st["rejected"] += 1
                v_st["rejected_templates"].add(c.template)
    for st in per_stratum.values():
        st["distinct_templates"] = len(st["templates"])
        st["templates"] = dict(sorted(st["templates"].items()))
        for v_st in st["by_verifier"].values():
            v_st["rejection_rate_pct"] = round(100.0 * v_st["rejected"] / st["total_cases"], 2)
            v_st["accepted_templates"] = sorted(v_st["accepted_templates"])
            v_st["rejected_templates"] = sorted(v_st["rejected_templates"])

    suites = ("core", "stress", "residual", "redteam", "gauntlet")
    overall = {s: _summarize([c for c in cases if c.suite == s], outcomes) for s in suites}
    suite_counts = {
        s: {
            "cases": sum(1 for c in cases if c.suite == s),
            "adversarial_cases": sum(1 for c in cases if c.suite == s and c.is_adversarial),
            "valid_cases": sum(1 for c in cases if c.suite == s and not c.is_adversarial),
            "distinct_templates": sum(v["distinct_templates"] for v in per_stratum.values() if v["suite"] == s),
        }
        for s in suites
    }

    all_sentences = [s for c in cases for s in c.record.prose_sentences]
    results = {
        "benchmark_version": BENCHMARK_VERSION,
        "reference_version": REFERENCE_VERSION,
        "total_cases": len(cases),
        "adversarial_cases": sum(1 for c in cases if c.is_adversarial),
        "valid_cases": sum(1 for c in cases if not c.is_adversarial),
        "distinct_templates": sum(v["distinct_templates"] for v in per_stratum.values()),
        "suites": suite_counts,
        # Cases are sentence or record frames re-instantiated with other identifiers and numerals.
        "prose_sentence_frames": {
            "prose_sentences": len(all_sentences),
            "distinct_sentences": len(set(all_sentences)),
            "distinct_after_replacing_every_numeral": len({re.sub(r"[0-9]+(?:\.[0-9]+)?", "#", s) for s in all_sentences}),
        },
        "worked_vignette_verification": vignettes,
        "overall_summary": overall,
        "per_stratum_breakdown": per_stratum,
        "self_tests": {
            "transfer_classifier_vs_definition": selftest_transfer_definition(),
            "scalar_ranking_search": selftest_scalar_ranking_search(),
            "typed_single_coordinate_mutations": selftest_typed_mutations(),
            "ledger_tampering": selftest_ledger_tampering(),
            "canonical_rendering_round_trip": selftest_canonical_rendering(),
            "dual_mode_and_non_prose_surface_lock": selftest_dual_mode_and_non_prose_lock(),
            "expected_violation_codes": selftest_expected_violation_codes(),
            "tamper_and_reattest": selftest_tamper_and_reattest(),
            "structure_only_and_colon_ids": selftest_structure_only_and_colon_ids(),
            "source_pinned_pyc_and_hygiene": selftest_source_pinned_pyc_and_hygiene(),
        },
    }

    if write_json:
        out_path = Path(__file__).resolve().parent / "benchmark_results.json"
        out_path.write_text(json.dumps(results, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return results


if __name__ == "__main__":
    res = run_benchmark(write_json=True)
    for suite, summary in res["overall_summary"].items():
        print(suite, json.dumps({k: (v["adversarial_rejected"], v["adversarial_total"], v["valid_accepted"], v["valid_total"]) for k, v in summary.items()}))
