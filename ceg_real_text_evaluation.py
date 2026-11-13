#!/usr/bin/env python3
"""Real-text evaluation of Context-Evidence Graphs (CEGs) on the abstracts of 12 PubMed studies.

Evaluates the five CEG verification regimes (V0, V1, V2, V2s, V3) on a curated,
zero-network-at-runtime corpus of 12 peer-reviewed biomedical studies indexed in
PubMed across three clinical/translational domains:
  - Idiopathic pulmonary fibrosis (IPF): INPULSIS-1/2 (PMID:24836310), ASCEND
    (PMID:24836312), BI 1015550 Phase 2 (PMID:35569036), and primary human lung
    fibroblast MRTF pharmacology (PMID:36585256).
  - Heart failure (HF): PARADIGM-HF (PMID:25176015), PARAGON-HF (PMID:31475794),
    VICTORIA (PMID:32222134), and post-myocardial-infarction rat remodeling
    (PMID:25362207).
  - Thoracic oncology (NSCLC): KEYNOTE-024 (PMID:27718847), AURA3 (PMID:27959700),
    ADAURA (PMID:32955177), and Ba/F3 + patient-derived EGFR exon 20 insertion
    pharmacology (PMID:31108249).

Two complementary tracks are executed:
  - Track A (Raw Verbatim Published Sentences, n = 36; 3 unmodified abstract
    sentences per study): measures how often unconstrained human clinical prose
    passes versus triggers specific surface-contract refusals (inline confidence
    intervals or SDs, derived differences/hazard ratios, raw numerator counts,
    publisher thin spaces, or multi-outcome pooling) when anchored to its true
    outcome tuple(s).
  - Track B (Grounded Synthesis & Realistic Defect Perturbations, n = 84; 24
    valid synthesis controls and 60 adversarial synthesis defects across 6
    clinical/translational failure families D1-D6): measures verifier precision
    and recall when CEG is used as intended as a synthesis-output contract.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, replace
import json
from pathlib import Path
import re
from typing import Any, Mapping, Sequence

from ceg_reference import (
    ContextAxisPoset,
    ContextEvidenceGraph,
    ContextLatticeSpec,
    ContextSignature,
    EvidenceAtom,
    IDENTIFIER_MASK_RE,
    MultiArmContrast,
    PooledPolicy,
    ProspectiveFalsifierGate,
    ProvenancePin,
    QuantitySpec,
    REFERENCE_VERSION,
    ReversiblePatchEvent,
    SynthesisClaimRecord,
    TUPLE_RE,
    TypedProposition,
    atom_numeric_pool,
    ledger_sha256,
    normalize_numeric_literal,
    populate_claim_attestation,
    proposition_from_atom,
    resolve_json_pointer,
    sha256_canonical,
    settled_corpus_from_graph,
    verify_atoms_bound_to_corpus,
    verify_bijective_patch_replay,
    verify_prose_sentence,
)

REAL_TEXT_EVAL_VERSION = "3.2"
ASCII_NUM_RE = re.compile(r"(?<![A-Za-z0-9.])[-+]?\d+(?:\.\d+)?")
MASK_ID_RE = IDENTIFIER_MASK_RE


@dataclass(frozen=True)
class VerbatimSentenceCase:
    """One unmodified sentence from a published PubMed abstract."""

    sentence_id: str
    domain: str
    study_id: str
    pmid: str
    verbatim_sentence: str
    true_tuples: tuple[str, ...]
    primary_phenomenon: str


@dataclass(frozen=True)
class GroundedSynthesisCase:
    """One synthesis-contract claim record grounded in the 12-study corpus."""

    case_id: str
    domain: str
    family: str
    is_adversarial: bool
    description: str
    record: SynthesisClaimRecord
    tamper_kind: str | None = None
    # For an adversarial record: the same record with its false sentence corrected. The harness
    # requires the repaired twin to be accepted, so the rejection is attributable to the defect.
    repaired_prose: tuple[str, ...] | None = None


# ---------------------------------------------------------------------------
# 1. Context poset and 12-study corpus curated from PubMed abstracts
# ---------------------------------------------------------------------------

def build_clinical_translational_lattice() -> ContextLatticeSpec:
    """Six-axis product poset spanning preclinical assays, subgroups, interventions and Phase 3 ITT.

    Release 3.2 removes three over-wide orderings of release 3.1. A clinical trial no
    longer dominates a preclinical tier, the overall population no longer dominates
    a subgroup or a preclinical model, and an intervention axis keeps the results of
    two different drugs apart. Each of these three axes has a formal top element
    (`any_*`) that no study or target uses: it only supplies the common upper bound
    that makes such a transfer a bounded bridge rather than an inadmissible one.
    """
    model_axis = ContextAxisPoset(
        name="evidence_tier",
        dominates={
            "engineered_cell_line": frozenset({"engineered_cell_line"}),
            # `translational_model` is the bridge-floor marker: a tier that dominates it is at or above the floor.
            "translational_model": frozenset({"translational_model"}),
            "primary_human_cells": frozenset({"primary_human_cells", "translational_model"}),
            "in_vivo_rodent": frozenset({"in_vivo_rodent", "translational_model"}),
            "phase2_rct": frozenset({"phase2_rct", "translational_model"}),
            "phase3_rct": frozenset({"phase3_rct", "phase2_rct", "translational_model"}),
            "any_evidence_tier": frozenset(
                {
                    "any_evidence_tier",
                    "phase3_rct",
                    "phase2_rct",
                    "primary_human_cells",
                    "in_vivo_rodent",
                    "translational_model",
                    "engineered_cell_line",
                }
            ),
        },
    )
    interventions = (
        "nintedanib",
        "pirfenidone",
        "bi_1015550",
        "sacubitril_valsartan",
        "vericiguat",
        "pembrolizumab",
        "osimertinib",
    )
    intervention_axis = ContextAxisPoset(
        name="intervention",
        dominates={
            **{name: frozenset({name}) for name in interventions},
            "any_intervention": frozenset({"any_intervention", *interventions}),
        },
    )
    indication_axis = ContextAxisPoset(
        name="indication",
        dominates={
            "ipf": frozenset({"ipf"}),
            "hfref": frozenset({"hfref"}),
            "hfpef": frozenset({"hfpef"}),
            "nsclc_pd_l1_high_1l": frozenset({"nsclc_pd_l1_high_1l"}),
            "nsclc_egfr_t790m_2l": frozenset({"nsclc_egfr_t790m_2l"}),
            "nsclc_egfr_adjuvant": frozenset({"nsclc_egfr_adjuvant"}),
            "nsclc_egfr_exon20ins": frozenset({"nsclc_egfr_exon20ins"}),
        },
    )
    stratum_axis = ContextAxisPoset(
        name="population_stratum",
        dominates={
            "preclinical_model": frozenset({"preclinical_model"}),
            "antifibrotic_naive_subgroup": frozenset({"antifibrotic_naive_subgroup"}),
            "background_antifibrotic_subgroup": frozenset({"background_antifibrotic_subgroup"}),
            "cns_metastases_subgroup": frozenset({"cns_metastases_subgroup"}),
            "stage_ii_iiia_subgroup": frozenset({"stage_ii_iiia_subgroup"}),
            "overall_itt": frozenset({"overall_itt"}),
            "any_population": frozenset(
                {
                    "any_population",
                    "overall_itt",
                    "antifibrotic_naive_subgroup",
                    "background_antifibrotic_subgroup",
                    "cns_metastases_subgroup",
                    "stage_ii_iiia_subgroup",
                    "preclinical_model",
                }
            ),
        },
    )
    horizon_axis = ContextAxisPoset(
        name="followup_horizon",
        dominates={
            "acute_assay": frozenset({"acute_assay"}),
            "short_term_4_to_12w": frozenset({"short_term_4_to_12w", "acute_assay"}),
            "chronic_6m_plus": frozenset({"chronic_6m_plus", "short_term_4_to_12w", "acute_assay"}),
        },
    )
    readout_axis = ContextAxisPoset(
        name="readout_level",
        dominates={
            "molecular_potency": frozenset({"molecular_potency"}),
            "organ_remodeling_surrogate": frozenset({"organ_remodeling_surrogate", "molecular_potency"}),
            "clinical_efficacy_or_safety": frozenset(
                {"clinical_efficacy_or_safety", "organ_remodeling_surrogate", "molecular_potency"}
            ),
        },
    )
    return ContextLatticeSpec(
        axis_posets={
            "evidence_tier": model_axis,
            "indication": indication_axis,
            "intervention": intervention_axis,
            "population_stratum": stratum_axis,
            "followup_horizon": horizon_axis,
            "readout_level": readout_axis,
        },
        bridge_floor={
            "evidence_tier": "translational_model",
        },
    )


STUDY_METADATA: dict[str, dict[str, str]] = {
    "R101": {
        "pmid": "24836310",
        "doi": "10.1056/NEJMoa1402584",
        "trial": "INPULSIS-1 and INPULSIS-2",
        "domain": "IPF",
        "citation": "Richeldi et al., N Engl J Med 2014;370(22):2071-2082",
    },
    "R102": {
        "pmid": "24836312",
        "doi": "10.1056/NEJMoa1402582",
        "trial": "ASCEND",
        "domain": "IPF",
        "citation": "King et al., N Engl J Med 2014;370(22):2083-2092",
    },
    "R103": {
        "pmid": "35569036",
        "doi": "10.1056/NEJMoa2201737",
        "trial": "1305-0013 (BI 1015550 Phase 2)",
        "domain": "IPF",
        "citation": "Richeldi et al., N Engl J Med 2022;386(23):2178-2187",
    },
    "R104": {
        "pmid": "36585256",
        "doi": "10.1183/13993003.00604-2022",
        "trial": "Pirfenidone MRTF Fibroblast Study",
        "domain": "IPF",
        "citation": "Ma et al., Eur Respir J 2023;61(4):2200604",
    },
    "R201": {
        "pmid": "25176015",
        "doi": "10.1056/NEJMoa1409077",
        "trial": "PARADIGM-HF",
        "domain": "HF",
        "citation": "McMurray et al., N Engl J Med 2014;371(11):993-1004",
    },
    "R202": {
        "pmid": "31475794",
        "doi": "10.1056/NEJMoa1908655",
        "trial": "PARAGON-HF",
        "domain": "HF",
        "citation": "Solomon et al., N Engl J Med 2019;381(17):1609-1620",
    },
    "R203": {
        "pmid": "32222134",
        "doi": "10.1056/NEJMoa1915928",
        "trial": "VICTORIA",
        "domain": "HF",
        "citation": "Armstrong et al., N Engl J Med 2020;382(20):1883-1893",
    },
    "R204": {
        "pmid": "25362207",
        "doi": "10.1161/CIRCHEARTFAILURE.114.001785",
        "trial": "Post-MI Rat LCZ696 Remodeling",
        "domain": "HF",
        "citation": "von Lueder et al., Circ Heart Fail 2015;8(1):71-78",
    },
    "R301": {
        "pmid": "27718847",
        "doi": "10.1056/NEJMoa1606774",
        "trial": "KEYNOTE-024",
        "domain": "ONC",
        "citation": "Reck et al., N Engl J Med 2016;375(19):1823-1833",
    },
    "R302": {
        "pmid": "27959700",
        "doi": "10.1056/NEJMoa1612674",
        "trial": "AURA3",
        "domain": "ONC",
        "citation": "Mok et al., N Engl J Med 2017;376(7):629-640",
    },
    "R303": {
        "pmid": "32955177",
        "doi": "10.1056/NEJMoa2027071",
        "trial": "ADAURA",
        "domain": "ONC",
        "citation": "Wu et al., N Engl J Med 2020;383(18):1711-1723",
    },
    "R304": {
        "pmid": "31108249",
        "doi": "10.1016/j.jtho.2019.05.006",
        "trial": "Preclinical EGFR Exon 20 Insertion Osimertinib",
        "domain": "ONC",
        "citation": "Lee et al., J Thorac Oncol 2019;14(9):1556-1566",
    },
}


def _ctx(
    tier: str,
    indication: str,
    stratum: str,
    horizon: str,
    readout: str,
    intervention: str,
) -> ContextSignature:
    return ContextSignature(
        {
            "evidence_tier": tier,
            "indication": indication,
            "intervention": intervention,
            "population_stratum": stratum,
            "followup_horizon": horizon,
            "readout_level": readout,
        }
    )


def _make_patch_ledger(
    initial_corpus: dict[str, Any],
    edits: Sequence[tuple[str, str, str, Any, str]],
) -> tuple[dict[str, Any], list[ReversiblePatchEvent]]:
    """Apply study-scoped edits and emit RFC 8785 / RFC 6901 ReversiblePatchEvents."""
    work = copy.deepcopy(initial_corpus)
    events: list[ReversiblePatchEvent] = []
    for event_id, study_id, pointer, new_val, rationale in edits:
        study_obj = next(s for s in work["studies"] if s["study_id"] == study_id)
        before_sha = sha256_canonical(study_obj)
        parent, key = resolve_json_pointer(work, pointer)
        old_val = copy.deepcopy(parent[key])
        parent[key] = copy.deepcopy(new_val)
        after_sha = sha256_canonical(study_obj)
        events.append(
            ReversiblePatchEvent(
                event_id=event_id,
                study_id=study_id,
                json_pointer=pointer,
                old_value=old_val,
                new_value=new_val,
                study_sha256_before=before_sha,
                study_sha256_after=after_sha,
                rationale=rationale,
            )
        )
    return work, events


def build_real_text_corpus() -> tuple[
    ContextEvidenceGraph,
    dict[str, Any],
    dict[str, Any],
    list[ReversiblePatchEvent],
]:
    """Construct the 12-study, 31-atom CEG curated from PubMed abstracts, and its curation ledger."""
    lattice = build_clinical_translational_lattice()

    studies: dict[str, ContextSignature] = {
        "R101": _ctx("phase3_rct", "ipf", "overall_itt", "chronic_6m_plus", "clinical_efficacy_or_safety", "nintedanib"),
        "R102": _ctx("phase3_rct", "ipf", "overall_itt", "chronic_6m_plus", "clinical_efficacy_or_safety", "pirfenidone"),
        "R103": _ctx("phase2_rct", "ipf", "overall_itt", "short_term_4_to_12w", "clinical_efficacy_or_safety", "bi_1015550"),
        "R104": _ctx("primary_human_cells", "ipf", "preclinical_model", "acute_assay", "molecular_potency", "pirfenidone"),
        "R201": _ctx("phase3_rct", "hfref", "overall_itt", "chronic_6m_plus", "clinical_efficacy_or_safety", "sacubitril_valsartan"),
        "R202": _ctx("phase3_rct", "hfpef", "overall_itt", "chronic_6m_plus", "clinical_efficacy_or_safety", "sacubitril_valsartan"),
        "R203": _ctx("phase3_rct", "hfref", "overall_itt", "chronic_6m_plus", "clinical_efficacy_or_safety", "vericiguat"),
        "R204": _ctx("in_vivo_rodent", "hfref", "preclinical_model", "short_term_4_to_12w", "organ_remodeling_surrogate", "sacubitril_valsartan"),
        "R301": _ctx("phase3_rct", "nsclc_pd_l1_high_1l", "overall_itt", "chronic_6m_plus", "clinical_efficacy_or_safety", "pembrolizumab"),
        "R302": _ctx("phase3_rct", "nsclc_egfr_t790m_2l", "overall_itt", "chronic_6m_plus", "clinical_efficacy_or_safety", "osimertinib"),
        "R303": _ctx("phase3_rct", "nsclc_egfr_adjuvant", "overall_itt", "chronic_6m_plus", "clinical_efficacy_or_safety", "osimertinib"),
        "R304": _ctx("engineered_cell_line", "nsclc_egfr_exon20ins", "preclinical_model", "acute_assay", "molecular_potency", "osimertinib"),
    }

    study_variables: dict[str, set[str]] = {
        "R101": {"VAR:R101:Nintedanib_150mg_BID", "VAR:R101:Placebo"},
        "R102": {"VAR:R102:Pirfenidone_2403mg", "VAR:R102:Placebo"},
        "R103": {"VAR:R103:BI1015550_18mg_BID", "VAR:R103:Placebo"},
        "R104": {"VAR:R104:Pirfenidone_InVitro"},
        "R201": {"VAR:R201:LCZ696_200mg_BID", "VAR:R201:Enalapril_10mg_BID"},
        "R202": {"VAR:R202:Sacubitril_Valsartan", "VAR:R202:Valsartan"},
        "R203": {"VAR:R203:Vericiguat_10mg", "VAR:R203:Placebo"},
        "R204": {"VAR:R204:MI_ARNi_LCZ696", "VAR:R204:MI_Vehicle"},
        "R301": {"VAR:R301:Pembrolizumab_200mg", "VAR:R301:Platinum_Chemotherapy"},
        "R302": {"VAR:R302:Osimertinib_80mg", "VAR:R302:Platinum_Pemetrexed"},
        "R303": {"VAR:R303:Adjuvant_Osimertinib_80mg", "VAR:R303:Placebo"},
        "R304": {"VAR:R304:Osimertinib_InVitro"},
    }

    arm_aliases: dict[str, tuple[str, ...]] = {
        "VAR:R101:Nintedanib_150mg_BID": ("nintedanib group", "nintedanib"),
        "VAR:R101:Placebo": ("placebo group", "placebo"),
        "VAR:R102:Pirfenidone_2403mg": ("pirfenidone group", "pirfenidone"),
        "VAR:R102:Placebo": ("placebo group", "placebo"),
        "VAR:R103:BI1015550_18mg_BID": ("bi 1015550 group", "bi 1015550"),
        "VAR:R103:Placebo": ("placebo group", "placebo"),
        "VAR:R104:Pirfenidone_InVitro": ("pirfenidone", "pfd"),
        "VAR:R201:LCZ696_200mg_BID": ("lcz696 group", "lcz696"),
        "VAR:R201:Enalapril_10mg_BID": ("enalapril group", "enalapril"),
        "VAR:R202:Sacubitril_Valsartan": ("sacubitril-valsartan group", "sacubitril-valsartan"),
        "VAR:R202:Valsartan": ("valsartan group", "valsartan"),
        "VAR:R203:Vericiguat_10mg": ("vericiguat group", "vericiguat"),
        "VAR:R203:Placebo": ("placebo group", "placebo"),
        "VAR:R204:MI_ARNi_LCZ696": ("mi-arni", "lcz696"),
        "VAR:R204:MI_Vehicle": ("mi-vehicle", "vehicle"),
        "VAR:R301:Pembrolizumab_200mg": ("pembrolizumab group", "pembrolizumab"),
        "VAR:R301:Platinum_Chemotherapy": ("chemotherapy group", "platinum-based chemotherapy", "chemotherapy"),
        "VAR:R302:Osimertinib_80mg": ("osimertinib group", "osimertinib"),
        "VAR:R302:Platinum_Pemetrexed": ("platinum therapy plus pemetrexed", "platinum-pemetrexed"),
        "VAR:R303:Adjuvant_Osimertinib_80mg": ("osimertinib group", "osimertinib"),
        "VAR:R303:Placebo": ("placebo group", "placebo"),
        "VAR:R304:Osimertinib_InVitro": ("osimertinib",),
    }

    c_r103_naive = _ctx("phase2_rct", "ipf", "antifibrotic_naive_subgroup", "short_term_4_to_12w", "clinical_efficacy_or_safety", "bi_1015550")
    c_r103_bg = _ctx("phase2_rct", "ipf", "background_antifibrotic_subgroup", "short_term_4_to_12w", "clinical_efficacy_or_safety", "bi_1015550")
    c_r302_cns = _ctx("phase3_rct", "nsclc_egfr_t790m_2l", "cns_metastases_subgroup", "chronic_6m_plus", "clinical_efficacy_or_safety", "osimertinib")
    c_r303_st2 = _ctx("phase3_rct", "nsclc_egfr_adjuvant", "stage_ii_iiia_subgroup", "chronic_6m_plus", "clinical_efficacy_or_safety", "osimertinib")

    atoms_list: list[EvidenceAtom] = [
        # --- R101: INPULSIS-1 & INPULSIS-2 (PMID:24836310) ---
        EvidenceAtom(
            atom_id="ATOM:R101:O1",
            outcome_tuple="R101:O1",
            study_id="R101",
            endpoint="INPULSIS-1 adjusted annual rate of FVC change",
            context=studies["R101"],
            contrast=MultiArmContrast(("VAR:R101:Nintedanib_150mg_BID",), ("VAR:R101:Placebo",), "pairwise"),
            focal_quantity=QuantitySpec("point_level", "=", "-114.7", unit="mL/year", unit_dimension="flow_decline"),
            comparator_quantity=QuantitySpec("point_level", "=", "-239.9", unit="mL/year", unit_dimension="flow_decline"),
            statistic_kind="mean",
            sample_size=513,
            p_value_or_interval="P<0.001",
            polarity="favorable",
            provenance=ProvenancePin("PMID:24836310", "Abstract Results; sample size 513 (309 + 204 treated): ClinicalTrials.gov NCT01335464", "-114.7 ml with nintedanib versus -239.9 ml with placebo", False, True),
        ),
        EvidenceAtom(
            atom_id="ATOM:R101:O2",
            outcome_tuple="R101:O2",
            study_id="R101",
            endpoint="INPULSIS-2 adjusted annual rate of FVC change",
            context=studies["R101"],
            contrast=MultiArmContrast(("VAR:R101:Nintedanib_150mg_BID",), ("VAR:R101:Placebo",), "pairwise"),
            focal_quantity=QuantitySpec("point_level", "=", "-113.6", unit="mL/year", unit_dimension="flow_decline"),
            comparator_quantity=QuantitySpec("point_level", "=", "-207.3", unit="mL/year", unit_dimension="flow_decline"),
            statistic_kind="mean",
            sample_size=548,
            p_value_or_interval="P<0.001",
            polarity="favorable",
            provenance=ProvenancePin("PMID:24836310", "Abstract Results; sample size 548 (329 + 219 treated): ClinicalTrials.gov NCT01335477", "-113.6 ml with nintedanib versus -207.3 ml with placebo", False, True),
        ),
        EvidenceAtom(
            atom_id="ATOM:R101:O3",
            outcome_tuple="R101:O3",
            study_id="R101",
            endpoint="INPULSIS-1 diarrhea adverse event rate",
            context=studies["R101"],
            contrast=MultiArmContrast(("VAR:R101:Nintedanib_150mg_BID",), ("VAR:R101:Placebo",), "pairwise"),
            focal_quantity=QuantitySpec("point_level", "=", "61.5", unit="%", unit_dimension="proportion"),
            comparator_quantity=QuantitySpec("point_level", "=", "18.6", unit="%", unit_dimension="proportion"),
            statistic_kind="proportion",
            sample_size=513,
            p_value_or_interval=None,
            polarity="unfavorable",
            provenance=ProvenancePin("PMID:24836310", "Abstract Results; sample size 513 (309 + 204 treated): ClinicalTrials.gov NCT01335464", "rates of 61.5% and 18.6% in the nintedanib and placebo groups", False, True),
        ),
        EvidenceAtom(
            atom_id="ATOM:R101:O4",
            outcome_tuple="R101:O4",
            study_id="R101",
            endpoint="Diarrhea-associated medication discontinuation rate",
            context=studies["R101"],
            contrast=MultiArmContrast(("VAR:R101:Nintedanib_150mg_BID",), (), "single_arm"),
            focal_quantity=QuantitySpec("censored_observation", "<", "5", unit="%", unit_dimension="proportion"),
            comparator_quantity=None,
            statistic_kind="proportion",
            sample_size=638,
            p_value_or_interval=None,
            polarity="neutral_or_mixed",
            provenance=ProvenancePin("PMID:24836310", "Abstract Conclusions; sample size 638 (309 + 329 nintedanib-treated): ClinicalTrials.gov NCT01335464 and NCT01335477", "led to discontinuation of the study medication in less than 5% of patients", False, True),
        ),
        # --- R102: ASCEND (PMID:24836312) ---
        EvidenceAtom(
            atom_id="ATOM:R102:O1",
            outcome_tuple="R102:O1",
            study_id="R102",
            endpoint="52-week >=10 percentage-point FVC decline or death relative reduction",
            context=studies["R102"],
            contrast=MultiArmContrast(("VAR:R102:Pirfenidone_2403mg",), ("VAR:R102:Placebo",), "pairwise"),
            focal_quantity=QuantitySpec("relative_change", "=", "47.9", unit="%", change_direction="decrease"),
            comparator_quantity=None,
            statistic_kind="proportion",
            sample_size=555,
            p_value_or_interval="P<0.001",
            polarity="favorable",
            provenance=ProvenancePin("PMID:24836312", "Abstract Results; sample size 555: Abstract Methods", "relative reduction of 47.9% in the proportion of patients", False, True),
        ),
        EvidenceAtom(
            atom_id="ATOM:R102:O2",
            outcome_tuple="R102:O2",
            study_id="R102",
            endpoint="52-week zero-FVC-decline patient proportion relative increase",
            context=studies["R102"],
            contrast=MultiArmContrast(("VAR:R102:Pirfenidone_2403mg",), ("VAR:R102:Placebo",), "pairwise"),
            focal_quantity=QuantitySpec("relative_change", "=", "132.5", unit="%", change_direction="increase"),
            comparator_quantity=None,
            statistic_kind="proportion",
            sample_size=555,
            p_value_or_interval="P<0.001",
            polarity="favorable",
            provenance=ProvenancePin("PMID:24836312", "Abstract Results; sample size 555: Abstract Methods", "relative increase of 132.5% in the proportion of patients with no decline in FVC", False, True),
        ),
        # --- R103: BI 1015550 Phase 2 (PMID:35569036) ---
        EvidenceAtom(
            atom_id="ATOM:R103:O1",
            outcome_tuple="R103:O1",
            study_id="R103",
            endpoint="12-week median FVC change without background antifibrotics",
            context=c_r103_naive,
            contrast=MultiArmContrast(("VAR:R103:BI1015550_18mg_BID",), ("VAR:R103:Placebo",), "pairwise"),
            focal_quantity=QuantitySpec("point_level", "=", "5.7", unit="mL", unit_dimension="volume"),
            comparator_quantity=QuantitySpec("point_level", "=", "-81.7", unit="mL", unit_dimension="volume"),
            statistic_kind="median",
            sample_size=73,
            p_value_or_interval="posterior_prob=0.998",
            polarity="favorable",
            provenance=ProvenancePin("PMID:35569036", "Abstract Results; sample size 73 (48 + 25, no background antifibrotic): ClinicalTrials.gov NCT04419506", "the median change in the FVC was 5.7 ml (95% credible interval, -39.1 to 50.5) in the BI 1015550 group and -81.7 ml (95% credible interval, -133.5 to -44.8) in the placebo group", False, True),
        ),
        EvidenceAtom(
            atom_id="ATOM:R103:O2",
            outcome_tuple="R103:O2",
            study_id="R103",
            endpoint="12-week median FVC change with background antifibrotics",
            context=c_r103_bg,
            contrast=MultiArmContrast(("VAR:R103:BI1015550_18mg_BID",), ("VAR:R103:Placebo",), "pairwise"),
            focal_quantity=QuantitySpec("point_level", "=", "2.7", unit="mL", unit_dimension="volume"),
            comparator_quantity=QuantitySpec("point_level", "=", "-59.2", unit="mL", unit_dimension="volume"),
            statistic_kind="median",
            sample_size=74,
            p_value_or_interval="posterior_prob=0.986",
            polarity="favorable",
            provenance=ProvenancePin("PMID:35569036", "Abstract Results; sample size 74 (49 + 25, background antifibrotic): ClinicalTrials.gov NCT04419506", "the median change in the FVC was 2.7 ml (95% credible interval, -32.8 to 38.2) in the BI 1015550 group and -59.2 ml (95% credible interval, -111.8 to -17.9) in the placebo group", False, True),
        ),
        # --- R104: Preclinical Pirfenidone MRTF (PMID:36585256) ---
        EvidenceAtom(
            atom_id="ATOM:R104:O1",
            outcome_tuple="R104:O1",
            study_id="R104",
            endpoint="MRTF activation half-maximal inhibitory concentration",
            context=studies["R104"],
            contrast=MultiArmContrast(("VAR:R104:Pirfenidone_InVitro",), (), "single_arm"),
            focal_quantity=QuantitySpec("interval_bound", None, lower="50", upper="150", unit="uM", unit_dimension="molarity"),
            comparator_quantity=None,
            statistic_kind="mean",
            sample_size=None,
            p_value_or_interval=None,
            polarity="favorable",
            provenance=ProvenancePin("PMID:36585256", "Abstract Results", "half-maximal inhibitory concentration 50-150", False, False),
        ),
        EvidenceAtom(
            atom_id="ATOM:R104:O2",
            outcome_tuple="R104:O2",
            study_id="R104",
            endpoint="MRTF maximal fibroblast inhibition",
            context=studies["R104"],
            contrast=MultiArmContrast(("VAR:R104:Pirfenidone_InVitro",), (), "single_arm"),
            focal_quantity=QuantitySpec("censored_observation", ">", "90", unit="%", unit_dimension="proportion"),
            comparator_quantity=None,
            statistic_kind="proportion",
            sample_size=None,
            p_value_or_interval=None,
            polarity="favorable",
            provenance=ProvenancePin("PMID:36585256", "Abstract Results", "maximal inhibition >90%", False, False),
        ),
        # --- R201: PARADIGM-HF (PMID:25176015) ---
        EvidenceAtom(
            atom_id="ATOM:R201:O1",
            outcome_tuple="R201:O1",
            study_id="R201",
            endpoint="Primary composite cardiovascular death or heart-failure hospitalization incidence",
            context=studies["R201"],
            contrast=MultiArmContrast(("VAR:R201:LCZ696_200mg_BID",), ("VAR:R201:Enalapril_10mg_BID",), "pairwise"),
            focal_quantity=QuantitySpec("point_level", "=", "21.8", unit="%", unit_dimension="proportion"),
            comparator_quantity=QuantitySpec("point_level", "=", "26.5", unit="%", unit_dimension="proportion"),
            statistic_kind="proportion",
            sample_size=8399,
            p_value_or_interval="P<0.001",
            polarity="favorable",
            provenance=ProvenancePin("PMID:25176015", "Abstract Results; sample size 8399 (4187 + 4212, full analysis set): ClinicalTrials.gov NCT01035255", "914 patients (21.8%) in the LCZ696 group and 1117 patients (26.5%) in the enalapril group", False, True),
        ),
        EvidenceAtom(
            atom_id="ATOM:R201:O2",
            outcome_tuple="R201:O2",
            study_id="R201",
            endpoint="All-cause mortality incidence",
            context=studies["R201"],
            contrast=MultiArmContrast(("VAR:R201:LCZ696_200mg_BID",), ("VAR:R201:Enalapril_10mg_BID",), "pairwise"),
            focal_quantity=QuantitySpec("point_level", "=", "17.0", unit="%", unit_dimension="proportion"),
            comparator_quantity=QuantitySpec("point_level", "=", "19.8", unit="%", unit_dimension="proportion"),
            statistic_kind="proportion",
            sample_size=8399,
            p_value_or_interval="P<0.001",
            polarity="favorable",
            provenance=ProvenancePin("PMID:25176015", "Abstract Results; sample size 8399 (4187 + 4212, full analysis set): ClinicalTrials.gov NCT01035255", "711 patients (17.0%) receiving LCZ696 and 835 patients (19.8%) receiving enalapril died", False, True),
        ),
        EvidenceAtom(
            atom_id="ATOM:R201:O3",
            outcome_tuple="R201:O3",
            study_id="R201",
            endpoint="Cardiovascular death incidence",
            context=studies["R201"],
            contrast=MultiArmContrast(("VAR:R201:LCZ696_200mg_BID",), ("VAR:R201:Enalapril_10mg_BID",), "pairwise"),
            focal_quantity=QuantitySpec("point_level", "=", "13.3", unit="%", unit_dimension="proportion"),
            comparator_quantity=QuantitySpec("point_level", "=", "16.5", unit="%", unit_dimension="proportion"),
            statistic_kind="proportion",
            sample_size=8399,
            p_value_or_interval="P<0.001",
            polarity="favorable",
            provenance=ProvenancePin("PMID:25176015", "Abstract Results; sample size 8399 (4187 + 4212, full analysis set): ClinicalTrials.gov NCT01035255", "558 (13.3%) and 693 (16.5%), respectively, died from cardiovascular causes", False, True),
        ),
        # --- R202: PARAGON-HF (PMID:31475794) ---
        EvidenceAtom(
            atom_id="ATOM:R202:O1",
            outcome_tuple="R202:O1",
            study_id="R202",
            endpoint="HFpEF cardiovascular death incidence",
            context=studies["R202"],
            contrast=MultiArmContrast(("VAR:R202:Sacubitril_Valsartan",), ("VAR:R202:Valsartan",), "pairwise"),
            focal_quantity=QuantitySpec("point_level", "=", "8.5", unit="%", unit_dimension="proportion"),
            comparator_quantity=QuantitySpec("point_level", "=", "8.9", unit="%", unit_dimension="proportion"),
            statistic_kind="proportion",
            sample_size=4796,
            p_value_or_interval=None,
            polarity="neutral_or_mixed",
            provenance=ProvenancePin("PMID:31475794", "Abstract Results; sample size 4796 (2407 + 2389, full analysis set): ClinicalTrials.gov NCT01920711", "incidence of death from cardiovascular causes was 8.5% in the sacubitril-valsartan group and 8.9% in the valsartan group", False, True),
        ),
        EvidenceAtom(
            atom_id="ATOM:R202:O2",
            outcome_tuple="R202:O2",
            study_id="R202",
            endpoint="8-month NYHA class improvement rate",
            context=studies["R202"],
            contrast=MultiArmContrast(("VAR:R202:Sacubitril_Valsartan",), ("VAR:R202:Valsartan",), "pairwise"),
            focal_quantity=QuantitySpec("point_level", "=", "15.0", unit="%", unit_dimension="proportion"),
            comparator_quantity=QuantitySpec("point_level", "=", "12.6", unit="%", unit_dimension="proportion"),
            statistic_kind="proportion",
            sample_size=4618,
            p_value_or_interval=None,
            polarity="favorable",
            provenance=ProvenancePin("PMID:31475794", "Abstract Results; sample size 4618 (2316 + 2302 assessed for NYHA class at 8 months): ClinicalTrials.gov NCT01920711", "NYHA class improved in 15.0% of the patients in the sacubitril-valsartan group and in 12.6% of those in the valsartan group", False, True),
        ),
        EvidenceAtom(
            atom_id="ATOM:R202:O3",
            outcome_tuple="R202:O3",
            study_id="R202",
            endpoint="Worsening renal function rate",
            context=studies["R202"],
            contrast=MultiArmContrast(("VAR:R202:Sacubitril_Valsartan",), ("VAR:R202:Valsartan",), "pairwise"),
            focal_quantity=QuantitySpec("point_level", "=", "1.4", unit="%", unit_dimension="proportion"),
            comparator_quantity=QuantitySpec("point_level", "=", "2.7", unit="%", unit_dimension="proportion"),
            statistic_kind="proportion",
            sample_size=4796,
            p_value_or_interval=None,
            polarity="favorable",
            provenance=ProvenancePin("PMID:31475794", "Abstract Results; sample size 4796 (2407 + 2389, full analysis set): ClinicalTrials.gov NCT01920711", "renal function worsened in 1.4% and 2.7%, respectively", False, True),
        ),
        # --- R203: VICTORIA (PMID:32222134) ---
        EvidenceAtom(
            atom_id="ATOM:R203:O1",
            outcome_tuple="R203:O1",
            study_id="R203",
            endpoint="Primary composite cardiovascular death or first heart-failure hospitalization incidence",
            context=studies["R203"],
            contrast=MultiArmContrast(("VAR:R203:Vericiguat_10mg",), ("VAR:R203:Placebo",), "pairwise"),
            focal_quantity=QuantitySpec("point_level", "=", "35.5", unit="%", unit_dimension="proportion"),
            comparator_quantity=QuantitySpec("point_level", "=", "38.5", unit="%", unit_dimension="proportion"),
            statistic_kind="proportion",
            sample_size=5050,
            p_value_or_interval="P=0.02",
            polarity="favorable",
            provenance=ProvenancePin("PMID:32222134", "Abstract Results; sample size 5050: Abstract Methods", "897 of 2526 patients (35.5%) in the vericiguat group and in 972 of 2524 patients (38.5%) in the placebo group", False, True),
        ),
        EvidenceAtom(
            atom_id="ATOM:R203:O2",
            outcome_tuple="R203:O2",
            study_id="R203",
            endpoint="Heart-failure hospitalization incidence",
            context=studies["R203"],
            contrast=MultiArmContrast(("VAR:R203:Vericiguat_10mg",), ("VAR:R203:Placebo",), "pairwise"),
            focal_quantity=QuantitySpec("point_level", "=", "27.4", unit="%", unit_dimension="proportion"),
            comparator_quantity=QuantitySpec("point_level", "=", "29.6", unit="%", unit_dimension="proportion"),
            statistic_kind="proportion",
            sample_size=5050,
            p_value_or_interval=None,
            polarity="favorable",
            provenance=ProvenancePin("PMID:32222134", "Abstract Results; sample size 5050: Abstract Methods", "691 patients (27.4%) in the vericiguat group and 747 patients (29.6%) in the placebo group", False, True),
        ),
        EvidenceAtom(
            atom_id="ATOM:R203:O3",
            outcome_tuple="R203:O3",
            study_id="R203",
            endpoint="Symptomatic hypotension adverse event incidence",
            context=studies["R203"],
            contrast=MultiArmContrast(("VAR:R203:Vericiguat_10mg",), ("VAR:R203:Placebo",), "pairwise"),
            focal_quantity=QuantitySpec("point_level", "=", "9.1", unit="%", unit_dimension="proportion"),
            comparator_quantity=QuantitySpec("point_level", "=", "7.9", unit="%", unit_dimension="proportion"),
            statistic_kind="proportion",
            sample_size=5034,
            p_value_or_interval="P=0.12",
            polarity="unfavorable",
            provenance=ProvenancePin("PMID:32222134", "Abstract Results; sample size 5034 (2519 + 2515, as treated): ClinicalTrials.gov NCT02861534", "Symptomatic hypotension occurred in 9.1% of the patients in the vericiguat group and in 7.9% of the patients in the placebo group", False, True),
        ),
        # --- R204: Preclinical Rat Post-MI LCZ696 (PMID:25362207) ---
        EvidenceAtom(
            atom_id="ATOM:R204:O1",
            outcome_tuple="R204:O1",
            study_id="R204",
            endpoint="Left ventricular end-diastolic diameter",
            context=studies["R204"],
            contrast=MultiArmContrast(("VAR:R204:MI_ARNi_LCZ696",), ("VAR:R204:MI_Vehicle",), "pairwise"),
            focal_quantity=QuantitySpec("point_level", "=", "9.7", unit="mm", unit_dimension="length"),
            comparator_quantity=QuantitySpec("point_level", "=", "10.5", unit="mm", unit_dimension="length"),
            statistic_kind="mean",
            sample_size=17,
            p_value_or_interval="P<0.05",
            polarity="favorable",
            provenance=ProvenancePin("PMID:25362207", "Abstract Results", "lower LV end-diastolic diameter (by echocardiography; 9.7±0.2 versus 10.5±0.3 mm)", False, False),
        ),
        EvidenceAtom(
            atom_id="ATOM:R204:O2",
            outcome_tuple="R204:O2",
            study_id="R204",
            endpoint="Left ventricular ejection fraction",
            context=studies["R204"],
            contrast=MultiArmContrast(("VAR:R204:MI_ARNi_LCZ696",), ("VAR:R204:MI_Vehicle",), "pairwise"),
            focal_quantity=QuantitySpec("point_level", "=", "60", unit="%", unit_dimension="proportion"),
            comparator_quantity=QuantitySpec("point_level", "=", "47", unit="%", unit_dimension="proportion"),
            statistic_kind="mean",
            sample_size=17,
            p_value_or_interval="P<0.05",
            polarity="favorable",
            provenance=ProvenancePin("PMID:25362207", "Abstract Results", "higher LV ejection fraction (60±2 versus 47±5%)", False, False),
        ),
        # --- R301: KEYNOTE-024 (PMID:27718847) ---
        EvidenceAtom(
            atom_id="ATOM:R301:O1",
            outcome_tuple="R301:O1",
            study_id="R301",
            endpoint="Median progression-free survival",
            context=studies["R301"],
            contrast=MultiArmContrast(("VAR:R301:Pembrolizumab_200mg",), ("VAR:R301:Platinum_Chemotherapy",), "pairwise"),
            focal_quantity=QuantitySpec("point_level", "=", "10.3", unit="months", unit_dimension="duration"),
            comparator_quantity=QuantitySpec("point_level", "=", "6.0", unit="months", unit_dimension="duration"),
            statistic_kind="median",
            sample_size=305,
            p_value_or_interval="P<0.001",
            polarity="favorable",
            provenance=ProvenancePin("PMID:27718847", "Abstract Results; sample size 305: Abstract Methods", "Median progression-free survival was 10.3 months (95% confidence interval [CI], 6.7 to not reached) in the pembrolizumab group versus 6.0 months (95% CI, 4.2 to 6.2) in the chemotherapy group", False, True),
        ),
        EvidenceAtom(
            atom_id="ATOM:R301:O2",
            outcome_tuple="R301:O2",
            study_id="R301",
            endpoint="Estimated 6-month overall survival rate",
            context=studies["R301"],
            contrast=MultiArmContrast(("VAR:R301:Pembrolizumab_200mg",), ("VAR:R301:Platinum_Chemotherapy",), "pairwise"),
            focal_quantity=QuantitySpec("point_level", "=", "80.2", unit="%", unit_dimension="proportion"),
            comparator_quantity=QuantitySpec("point_level", "=", "72.4", unit="%", unit_dimension="proportion"),
            statistic_kind="proportion",
            sample_size=305,
            p_value_or_interval="P=0.005",
            polarity="favorable",
            provenance=ProvenancePin("PMID:27718847", "Abstract Results; sample size 305: Abstract Methods", "80.2% in the pembrolizumab group versus 72.4% in the chemotherapy group", False, True),
        ),
        EvidenceAtom(
            atom_id="ATOM:R301:O3",
            outcome_tuple="R301:O3",
            study_id="R301",
            endpoint="Grade 3 to 5 treatment-related adverse event incidence",
            context=studies["R301"],
            contrast=MultiArmContrast(("VAR:R301:Pembrolizumab_200mg",), ("VAR:R301:Platinum_Chemotherapy",), "pairwise"),
            focal_quantity=QuantitySpec("point_level", "=", "26.6", unit="%", unit_dimension="proportion"),
            comparator_quantity=QuantitySpec("point_level", "=", "53.3", unit="%", unit_dimension="proportion"),
            statistic_kind="proportion",
            sample_size=304,
            p_value_or_interval=None,
            polarity="favorable",
            provenance=ProvenancePin("PMID:27718847", "Abstract Results; sample size 304 (154 + 150, as treated): ClinicalTrials.gov NCT02142738", "grade 3, 4, or 5 treatment-related adverse events (26.6% vs. 53.3%)", False, True),
        ),
        # --- R302: AURA3 (PMID:27959700) ---
        EvidenceAtom(
            atom_id="ATOM:R302:O1",
            outcome_tuple="R302:O1",
            study_id="R302",
            endpoint="Median progression-free survival",
            context=studies["R302"],
            contrast=MultiArmContrast(("VAR:R302:Osimertinib_80mg",), ("VAR:R302:Platinum_Pemetrexed",), "pairwise"),
            focal_quantity=QuantitySpec("point_level", "=", "10.1", unit="months", unit_dimension="duration"),
            comparator_quantity=QuantitySpec("point_level", "=", "4.4", unit="months", unit_dimension="duration"),
            statistic_kind="median",
            sample_size=419,
            p_value_or_interval="P<0.001",
            polarity="favorable",
            provenance=ProvenancePin("PMID:27959700", "Abstract Results; sample size 419: Abstract Methods", "10.1 months vs. 4.4 months", False, True),
        ),
        EvidenceAtom(
            atom_id="ATOM:R302:O2",
            outcome_tuple="R302:O2",
            study_id="R302",
            endpoint="Central nervous system metastases subgroup median progression-free survival",
            context=c_r302_cns,
            contrast=MultiArmContrast(("VAR:R302:Osimertinib_80mg",), ("VAR:R302:Platinum_Pemetrexed",), "pairwise"),
            focal_quantity=QuantitySpec("point_level", "=", "8.5", unit="months", unit_dimension="duration"),
            comparator_quantity=QuantitySpec("point_level", "=", "4.2", unit="months", unit_dimension="duration"),
            statistic_kind="median",
            sample_size=144,
            p_value_or_interval=None,
            polarity="favorable",
            provenance=ProvenancePin("PMID:27959700", "Abstract Results; sample size 144: Abstract Results", "Among 144 patients with metastases to the central nervous system (CNS), the median duration of progression-free survival was longer among patients receiving osimertinib than among those receiving platinum therapy plus pemetrexed (8.5 months vs. 4.2 months; hazard ratio, 0.32; 95% CI, 0.21 to 0.49)", False, True),
        ),
        EvidenceAtom(
            atom_id="ATOM:R302:O3",
            outcome_tuple="R302:O3",
            study_id="R302",
            endpoint="Objective response rate",
            context=studies["R302"],
            contrast=MultiArmContrast(("VAR:R302:Osimertinib_80mg",), ("VAR:R302:Platinum_Pemetrexed",), "pairwise"),
            focal_quantity=QuantitySpec("point_level", "=", "71", unit="%", unit_dimension="proportion"),
            comparator_quantity=QuantitySpec("point_level", "=", "31", unit="%", unit_dimension="proportion"),
            statistic_kind="proportion",
            sample_size=419,
            p_value_or_interval="P<0.001",
            polarity="favorable",
            provenance=ProvenancePin("PMID:27959700", "Abstract Results; sample size 419: Abstract Methods", "objective response rate was significantly better with osimertinib (71%; 95% CI, 65 to 76) than with platinum therapy plus pemetrexed (31%; 95% CI, 24 to 40)", False, True),
        ),
        EvidenceAtom(
            atom_id="ATOM:R302:O4",
            outcome_tuple="R302:O4",
            study_id="R302",
            endpoint="adverse events of grade 3 or higher",
            context=studies["R302"],
            contrast=MultiArmContrast(("VAR:R302:Osimertinib_80mg",), ("VAR:R302:Platinum_Pemetrexed",), "pairwise"),
            focal_quantity=QuantitySpec("point_level", "=", "23", unit="%", unit_dimension="proportion"),
            comparator_quantity=QuantitySpec("point_level", "=", "47", unit="%", unit_dimension="proportion"),
            statistic_kind="proportion",
            sample_size=415,
            p_value_or_interval=None,
            polarity="favorable",
            provenance=ProvenancePin("PMID:27959700", "Abstract Results; sample size 415 (279 + 136, as treated): ClinicalTrials.gov NCT02151981", "adverse events of grade 3 or higher was lower with osimertinib (23%) than with platinum therapy plus pemetrexed (47%)", False, True),
        ),
        # --- R303: ADAURA (PMID:32955177) ---
        EvidenceAtom(
            atom_id="ATOM:R303:O1",
            outcome_tuple="R303:O1",
            study_id="R303",
            endpoint="Stage II to IIIA 24-month disease-free survival rate",
            context=c_r303_st2,
            contrast=MultiArmContrast(("VAR:R303:Adjuvant_Osimertinib_80mg",), ("VAR:R303:Placebo",), "pairwise"),
            focal_quantity=QuantitySpec("point_level", "=", "90", unit="%", unit_dimension="proportion"),
            comparator_quantity=QuantitySpec("point_level", "=", "44", unit="%", unit_dimension="proportion"),
            statistic_kind="proportion",
            sample_size=470,
            p_value_or_interval="P<0.001",
            polarity="favorable",
            provenance=ProvenancePin("PMID:32955177", "Abstract Results; sample size 470 (233 + 237, stage II to IIIA): ClinicalTrials.gov NCT02511106", "90% of the patients with stage II to IIIA disease in the osimertinib group (95% confidence interval [CI], 84 to 93) and 44% of those in the placebo group (95% CI, 37 to 51)", False, True),
        ),
        EvidenceAtom(
            atom_id="ATOM:R303:O2",
            outcome_tuple="R303:O2",
            study_id="R303",
            endpoint="Overall stage IB to IIIA 24-month disease-free survival rate",
            context=studies["R303"],
            contrast=MultiArmContrast(("VAR:R303:Adjuvant_Osimertinib_80mg",), ("VAR:R303:Placebo",), "pairwise"),
            focal_quantity=QuantitySpec("point_level", "=", "89", unit="%", unit_dimension="proportion"),
            comparator_quantity=QuantitySpec("point_level", "=", "52", unit="%", unit_dimension="proportion"),
            statistic_kind="proportion",
            sample_size=682,
            p_value_or_interval="P<0.001",
            polarity="favorable",
            provenance=ProvenancePin("PMID:32955177", "Abstract Results; sample size 682: Abstract Results", "In the overall population, 89% of the patients in the osimertinib group (95% CI, 85 to 92) and 52% of those in the placebo group (95% CI, 46 to 58)", False, True),
        ),
        # --- R304: Preclinical EGFR Exon 20 Insertion Osimertinib (PMID:31108249) ---
        EvidenceAtom(
            atom_id="ATOM:R304:O1",
            outcome_tuple="R304:O1",
            study_id="R304",
            endpoint="EGFR exon 20 insertion-mutant IC50 potency",
            context=studies["R304"],
            contrast=MultiArmContrast(("VAR:R304:Osimertinib_InVitro",), (), "single_arm"),
            focal_quantity=QuantitySpec("interval_bound", None, lower="14.7", upper="62.7", unit="nM", unit_dimension="molarity"),
            comparator_quantity=None,
            statistic_kind="mean",
            sample_size=None,
            p_value_or_interval=None,
            polarity="favorable",
            provenance=ProvenancePin("PMID:31108249", "Abstract Results", "osimertinib was highly potent against EGFR exon 20 insertion-mutant cells (IC50, 14.7-62.7 nM)", False, False),
        ),
    ]

    atoms = {a.outcome_tuple: a for a in atoms_list}
    raw_mentioned_arms_by_tuple = {
        tup: tuple(list(a.contrast.focal_arms) + list(a.contrast.comparator_arms))
        for tup, a in atoms.items()
    }
    graph = ContextEvidenceGraph(
        lattice=lattice,
        studies=studies,
        study_variables=study_variables,
        arm_aliases=arm_aliases,
        atoms=atoms,
        raw_mentioned_arms_by_tuple=raw_mentioned_arms_by_tuple,
    )

    # The settled corpus is the complete projection of the graph (every atom field, nulls explicit);
    # the initial corpus then receives 4 realistic pre-audit extraction errors.
    settled_target = settled_corpus_from_graph(
        graph,
        "CEG-PUBMED-REAL-TEXT-12",
        {sid: {"pmid": STUDY_METADATA[sid]["pmid"]} for sid in studies},
    )
    if [st["study_id"] for st in settled_target["studies"]][3:10:3] != ["R104", "R203", "R302"]:
        raise RuntimeError("Real-text study order changed; the ledger pointers below assume it")
    initial_corpus = copy.deepcopy(settled_target)
    # Introduce 4 realistic initial extraction defects into initial_corpus, then repair via ReversiblePatchEvents
    initial_corpus["studies"][3]["outcomes"][0]["unit"] = "um"         # R104:O1 SI case fold defect (um instead of uM)
    initial_corpus["studies"][4]["outcomes"][0]["focal_value"] = "26.5"  # R201:O1 arm-swapped initial extraction
    initial_corpus["studies"][4]["outcomes"][0]["comparator_value"] = "21.8"
    initial_corpus["studies"][9]["outcomes"][1]["sample_size"] = 419   # R302:O2 ITT N=419 instead of CNS subgroup N=144

    settled_corpus, patch_events = _make_patch_ledger(
        initial_corpus,
        [
            (
                "PATCH-REAL-01",
                "R104",
                "/studies/3/outcomes/0/unit",
                "uM",
                "Restore micromolar (uM) unit on PMID:36585256 IC50 range; initial OCR case-folded uM to micrometre (um).",
            ),
            (
                "PATCH-REAL-02",
                "R201",
                "/studies/4/outcomes/0/focal_value",
                "21.8",
                "Restore LCZ696 primary composite rate 21.8% in PARADIGM-HF (PMID:25176015).",
            ),
            (
                "PATCH-REAL-03",
                "R201",
                "/studies/4/outcomes/0/comparator_value",
                "26.5",
                "Restore enalapril comparator rate 26.5% in PARADIGM-HF (PMID:25176015).",
            ),
            (
                "PATCH-REAL-04",
                "R302",
                "/studies/9/outcomes/1/sample_size",
                144,
                "Bind AURA3 CNS-metastases subgroup sample size n=144 rather than overall ITT n=419 (PMID:27959700).",
            ),
        ],
    )
    if settled_corpus != settled_target:
        raise RuntimeError("Real-text settled_corpus does not match settled_target")
    return graph, initial_corpus, settled_corpus, patch_events


# ---------------------------------------------------------------------------
# 2. Track A: 36 unmodified verbatim sentences from the 12 PubMed abstracts
# ---------------------------------------------------------------------------

def build_track_a_verbatim_cases() -> list[VerbatimSentenceCase]:
    """Three unmodified empirical sentences per PubMed abstract (n = 36 total)."""
    return [
        # IPF: R101 (PMID:24836310)
        VerbatimSentenceCase(
            "TA-R101-1",
            "IPF",
            "R101",
            "24836310",
            "A total of 1066 patients were randomly assigned in a 3:2 ratio to receive nintedanib or placebo.",
            ("R101:O1",),
            "unmodeled_ratio_or_design_numeral",
        ),
        VerbatimSentenceCase(
            "TA-R101-2",
            "IPF",
            "R101",
            "24836310",
            "The adjusted annual rate of change in FVC was -114.7 ml with nintedanib versus -239.9 ml with placebo (difference, 125.3 ml; 95% confidence interval [CI], 77.7 to 172.8; P<0.001) in INPULSIS-1 and -113.6 ml with nintedanib versus -207.3 ml with placebo (difference, 93.7 ml; 95% CI, 44.8 to 142.7; P<0.001) in INPULSIS-2.",
            ("R101:O1", "R101:O2"),
            "inline_ci_and_derived_difference",
        ),
        VerbatimSentenceCase(
            "TA-R101-3",
            "IPF",
            "R101",
            "24836310",
            "In patients with idiopathic pulmonary fibrosis, nintedanib reduced the decline in FVC, which is consistent with a slowing of disease progression; nintedanib was frequently associated with diarrhea, which led to discontinuation of the study medication in less than 5% of patients.",
            ("R101:O4",),
            "single_bound_clinical_prose",
        ),
        # IPF: R102 (PMID:24836312)
        VerbatimSentenceCase(
            "TA-R102-1",
            "IPF",
            "R102",
            "24836312",
            "In this phase 3 study, we randomly assigned 555 patients with idiopathic pulmonary fibrosis to receive either oral pirfenidone (2403 mg per day) or placebo for 52 weeks.",
            ("R102:O1",),
            "unmodeled_dose_or_design_numeral",
        ),
        VerbatimSentenceCase(
            "TA-R102-2",
            "IPF",
            "R102",
            "24836312",
            "In the pirfenidone group, as compared with the placebo group, there was a relative reduction of 47.9% in the proportion of patients who had an absolute decline of 10 percentage points or more in the percentage of the predicted FVC or who died; there was also a relative increase of 132.5% in the proportion of patients with no decline in FVC (P<0.001).",
            ("R102:O1", "R102:O2"),
            "pooled_multi_outcome_clause",
        ),
        VerbatimSentenceCase(
            "TA-R102-3",
            "IPF",
            "R102",
            "24836312",
            "Pirfenidone reduced the decline in the 6-minute walk distance (P=0.04) and improved progression-free survival (P<0.001).",
            ("R102:O1",),
            "unmodeled_secondary_pvalue",
        ),
        # IPF: R103 (PMID:35569036)
        VerbatimSentenceCase(
            "TA-R103-1",
            "IPF",
            "R103",
            "35569036",
            "Among patients without background antifibrotic use, the median change in the FVC was 5.7 ml (95% credible interval, -39.1 to 50.5) in the BI 1015550 group and -81.7 ml (95% credible interval, -133.5 to -44.8) in the placebo group (median difference, 88.4 ml; 95% credible interval, 29.5 to 154.2; probability that BI 1015550 was superior to placebo, 0.998).",
            ("R103:O1",),
            "inline_ci_and_derived_difference",
        ),
        VerbatimSentenceCase(
            "TA-R103-2",
            "IPF",
            "R103",
            "35569036",
            "Among patients with background antifibrotic use, the median change in the FVC was 2.7 ml (95% credible interval, -32.8 to 38.2) in the BI 1015550 group and -59.2 ml (95% credible interval, -111.8 to -17.9) in the placebo group (median difference, 62.4 ml; 95% credible interval, 6.3 to 125.5; probability that BI 1015550 was superior to placebo, 0.986).",
            ("R103:O2",),
            "inline_ci_and_derived_difference",
        ),
        VerbatimSentenceCase(
            "TA-R103-3",
            "IPF",
            "R103",
            "35569036",
            "A total of 147 patients were randomly assigned to receive BI 1015550 or placebo.",
            ("R103:O1",),
            "sample_size_only_sentence",
        ),
        # IPF: R104 (PMID:36585256)
        VerbatimSentenceCase(
            "TA-R104-1",
            "IPF",
            "R104",
            "36585256",
            "Furthermore, we showed that PFD inhibits MRTF activation in primary human lung fibroblasts at clinically achievable concentrations (half-maximal inhibitory concentration 50-150\u2005µM, maximal inhibition >90%, maximal concentration of PFD in patients <100\u2005µM).",
            ("R104:O1", "R104:O2"),
            "pooled_multi_outcome_and_pk_bound",
        ),
        VerbatimSentenceCase(
            "TA-R104-2",
            "IPF",
            "R104",
            "36585256",
            "Mechanistically, PFD appears to exert its inhibitory effects by promoting the interaction between MRTF and actin indirectly.",
            ("R104:O1",),
            "qualitative_mechanistic_sentence",
        ),
        VerbatimSentenceCase(
            "TA-R104-3",
            "IPF",
            "R104",
            "36585256",
            "Finally, PFD-treated IPF lungs exhibit significantly less MRTF activation in fibroblast foci areas than naïve IPF lungs.",
            ("R104:O1",),
            "qualitative_mechanistic_sentence",
        ),
        # HF: R201 (PMID:25176015)
        VerbatimSentenceCase(
            "TA-R201-1",
            "HF",
            "R201",
            "25176015",
            "At the time of study closure, the primary outcome had occurred in 914 patients (21.8%) in the LCZ696 group and 1117 patients (26.5%) in the enalapril group (hazard ratio in the LCZ696 group, 0.80; 95% confidence interval [CI], 0.73 to 0.87; P<0.001).",
            ("R201:O1",),
            "inline_event_counts_hr_and_ci",
        ),
        VerbatimSentenceCase(
            "TA-R201-2",
            "HF",
            "R201",
            "25176015",
            "A total of 711 patients (17.0%) receiving LCZ696 and 835 patients (19.8%) receiving enalapril died (hazard ratio for death from any cause, 0.84; 95% CI, 0.76 to 0.93; P<0.001); of these patients, 558 (13.3%) and 693 (16.5%), respectively, died from cardiovascular causes (hazard ratio, 0.80; 95% CI, 0.71 to 0.89; P<0.001).",
            ("R201:O2", "R201:O3"),
            "inline_event_counts_hr_and_ci",
        ),
        VerbatimSentenceCase(
            "TA-R201-3",
            "HF",
            "R201",
            "25176015",
            "As compared with enalapril, LCZ696 also reduced the risk of hospitalization for heart failure by 21% (P<0.001) and decreased the symptoms and physical limitations of heart failure (P=0.001).",
            ("R201:O1",),
            "derived_relative_risk_reduction",
        ),
        # HF: R202 (PMID:31475794)
        VerbatimSentenceCase(
            "TA-R202-1",
            "HF",
            "R202",
            "31475794",
            "There were 894 primary events in 526 patients in the sacubitril-valsartan group and 1009 primary events in 557 patients in the valsartan group (rate ratio, 0.87; 95% confidence interval [CI], 0.75 to 1.01; P\u2009=\u20090.06).",
            ("R202:O1",),
            "inline_event_counts_hr_and_ci",
        ),
        VerbatimSentenceCase(
            "TA-R202-2",
            "HF",
            "R202",
            "31475794",
            "The incidence of death from cardiovascular causes was 8.5% in the sacubitril-valsartan group and 8.9% in the valsartan group (hazard ratio, 0.95; 95% CI, 0.79 to 1.16); there were 690 and 797 total hospitalizations for heart failure, respectively (rate ratio, 0.85; 95% CI, 0.72 to 1.00).",
            ("R202:O1",),
            "inline_event_counts_hr_and_ci",
        ),
        VerbatimSentenceCase(
            "TA-R202-3",
            "HF",
            "R202",
            "31475794",
            "NYHA class improved in 15.0% of the patients in the sacubitril-valsartan group and in 12.6% of those in the valsartan group (odds ratio, 1.45; 95% CI, 1.13 to 1.86); renal function worsened in 1.4% and 2.7%, respectively (hazard ratio, 0.50; 95% CI, 0.33 to 0.77).",
            ("R202:O2", "R202:O3"),
            "inline_or_hr_and_ci",
        ),
        # HF: R203 (PMID:32222134)
        VerbatimSentenceCase(
            "TA-R203-1",
            "HF",
            "R203",
            "32222134",
            "Over a median of 10.8 months, a primary-outcome event occurred in 897 of 2526 patients (35.5%) in the vericiguat group and in 972 of 2524 patients (38.5%) in the placebo group (hazard ratio, 0.90; 95% confidence interval [CI], 0.82 to 0.98; P\u2009=\u20090.02).",
            ("R203:O1",),
            "inline_event_counts_hr_and_ci",
        ),
        VerbatimSentenceCase(
            "TA-R203-2",
            "HF",
            "R203",
            "32222134",
            "A total of 691 patients (27.4%) in the vericiguat group and 747 patients (29.6%) in the placebo group were hospitalized for heart failure (hazard ratio, 0.90; 95% CI, 0.81 to 1.00).",
            ("R203:O2",),
            "inline_event_counts_hr_and_ci",
        ),
        VerbatimSentenceCase(
            "TA-R203-3",
            "HF",
            "R203",
            "32222134",
            "Symptomatic hypotension occurred in 9.1% of the patients in the vericiguat group and in 7.9% of the patients in the placebo group (P\u2009=\u20090.12), and syncope occurred in 4.0% of the patients in the vericiguat group and in 3.5% of the patients in the placebo group (P\u2009=\u20090.30).",
            ("R203:O3",),
            "pooled_multi_outcome_clause",
        ),
        # HF: R204 (PMID:25362207)
        VerbatimSentenceCase(
            "TA-R204-1",
            "HF",
            "R204",
            "25362207",
            "One week after MI, adult male Sprague-Dawley rats were randomized to treatment for 4 weeks with LCZ696 (68 mg/kg body weight perorally; MI-ARNi, n=11) or vehicle (MI-vehicle, n=6).",
            ("R204:O1",),
            "unmodeled_dose_and_arm_denominators",
        ),
        VerbatimSentenceCase(
            "TA-R204-2",
            "HF",
            "R204",
            "25362207",
            "Five weeks after MI, MI-ARNi versus MI-vehicle demonstrated lower LV end-diastolic diameter (by echocardiography; 9.7±0.2 versus 10.5±0.3 mm), higher LV ejection fraction (60±2 versus 47±5%), diastolic wall strain (0.23±0.02 versus 0.13±0.02), and circular strain (-9.8±0.5 versus -7.3±0.5%; all P<0.05).",
            ("R204:O1", "R204:O2"),
            "inline_sem_dispersion_and_multi_outcome",
        ),
        VerbatimSentenceCase(
            "TA-R204-3",
            "HF",
            "R204",
            "25362207",
            "LCZ696 attenuated cardiac remodeling and dysfunction after MI.",
            ("R204:O1",),
            "qualitative_mechanistic_sentence",
        ),
        # ONC: R301 (PMID:27718847)
        VerbatimSentenceCase(
            "TA-R301-1",
            "ONC",
            "R301",
            "27718847",
            "Median progression-free survival was 10.3 months (95% confidence interval [CI], 6.7 to not reached) in the pembrolizumab group versus 6.0 months (95% CI, 4.2 to 6.2) in the chemotherapy group (hazard ratio for disease progression or death, 0.50; 95% CI, 0.37 to 0.68; P<0.001).",
            ("R301:O1",),
            "inline_ci_and_hazard_ratio",
        ),
        VerbatimSentenceCase(
            "TA-R301-2",
            "ONC",
            "R301",
            "27718847",
            "The estimated rate of overall survival at 6 months was 80.2% in the pembrolizumab group versus 72.4% in the chemotherapy group (hazard ratio for death, 0.60; 95% CI, 0.41 to 0.89; P=0.005).",
            ("R301:O2",),
            "inline_ci_and_hazard_ratio",
        ),
        VerbatimSentenceCase(
            "TA-R301-3",
            "ONC",
            "R301",
            "27718847",
            "The response rate was higher in the pembrolizumab group than in the chemotherapy group (44.8% vs. 27.8%), the median duration of response was longer (not reached [range, 1.9+ to 14.5+ months] vs. 6.3 months [range, 2.1+ to 12.6+]), and treatment-related adverse events of any grade were less frequent (occurring in 73.4% vs. 90.0% of patients), as were grade 3, 4, or 5 treatment-related adverse events (26.6% vs. 53.3%).",
            ("R301:O3",),
            "pooled_multi_outcome_clause",
        ),
        # ONC: R302 (PMID:27959700)
        VerbatimSentenceCase(
            "TA-R302-1",
            "ONC",
            "R302",
            "27959700",
            "The median duration of progression-free survival was significantly longer with osimertinib than with platinum therapy plus pemetrexed (10.1 months vs. 4.4 months; hazard ratio; 0.30; 95% confidence interval [CI], 0.23 to 0.41; P<0.001).",
            ("R302:O1",),
            "inline_ci_and_hazard_ratio",
        ),
        VerbatimSentenceCase(
            "TA-R302-2",
            "ONC",
            "R302",
            "27959700",
            "Among 144 patients with metastases to the central nervous system (CNS), the median duration of progression-free survival was longer among patients receiving osimertinib than among those receiving platinum therapy plus pemetrexed (8.5 months vs. 4.2 months; hazard ratio, 0.32; 95% CI, 0.21 to 0.49).",
            ("R302:O2",),
            "inline_ci_and_hazard_ratio",
        ),
        VerbatimSentenceCase(
            "TA-R302-3",
            "ONC",
            "R302",
            "27959700",
            "The proportion of patients with adverse events of grade 3 or higher was lower with osimertinib (23%) than with platinum therapy plus pemetrexed (47%).",
            ("R302:O4",),
            "single_contrast_clinical_prose",
        ),
        # ONC: R303 (PMID:32955177)
        VerbatimSentenceCase(
            "TA-R303-1",
            "ONC",
            "R303",
            "32955177",
            "A total of 682 patients underwent randomization (339 to the osimertinib group and 343 to the placebo group).",
            ("R303:O2",),
            "unmodeled_arm_denominators",
        ),
        VerbatimSentenceCase(
            "TA-R303-2",
            "ONC",
            "R303",
            "32955177",
            "At 24 months, 90% of the patients with stage II to IIIA disease in the osimertinib group (95% confidence interval [CI], 84 to 93) and 44% of those in the placebo group (95% CI, 37 to 51) were alive and disease-free (overall hazard ratio for disease recurrence or death, 0.17; 99.06% CI, 0.11 to 0.26; P<0.001).",
            ("R303:O1",),
            "inline_ci_and_hazard_ratio",
        ),
        VerbatimSentenceCase(
            "TA-R303-3",
            "ONC",
            "R303",
            "32955177",
            "In the overall population, 89% of the patients in the osimertinib group (95% CI, 85 to 92) and 52% of those in the placebo group (95% CI, 46 to 58) were alive and disease-free at 24 months (overall hazard ratio for disease recurrence or death, 0.20; 99.12% CI, 0.14 to 0.30; P<0.001).",
            ("R303:O2",),
            "inline_ci_and_hazard_ratio",
        ),
        # ONC: R304 (PMID:31108249)
        VerbatimSentenceCase(
            "TA-R304-1",
            "ONC",
            "R304",
            "31108249",
            "EGFR exon 20 insertion-mutant NSCLC cells, excluding EGFR A763insFQEA, were resistant to first-generation EGFR TKIs (concentration that inhibits 50% [IC50], 1.1 ± 0.067 to 5.4 ± 0.115 μM).",
            ("R304:O1",),
            "inline_sem_dispersion_and_other_drug",
        ),
        VerbatimSentenceCase(
            "TA-R304-2",
            "ONC",
            "R304",
            "31108249",
            "Third-generation EGFR TKI osimertinib was highly potent against EGFR exon 20 insertion-mutant cells (IC50, 14.7-62.7 nM), including EGFR H773insH, and spared wild-type EGFR cells.",
            ("R304:O1",),
            "glued_gene_mutation_identifier",
        ),
        VerbatimSentenceCase(
            "TA-R304-3",
            "ONC",
            "R304",
            "31108249",
            "Osimertinib is active against EGFR exon 20 insertion-mutant NSCLC and flexibly binds within drug-binding pockets in preclinical models.",
            ("R304:O1",),
            "contextual_exon_numeral_matched_via_endpoint",
        ),
    ]


# ---------------------------------------------------------------------------
# 3. Track B: 84 grounded synthesis records (24 valid + 60 realistic defects)
# ---------------------------------------------------------------------------

def build_track_b_cases(graph: ContextEvidenceGraph) -> list[GroundedSynthesisCase]:
    """Construct 24 valid synthesis controls and 60 realistic synthesis perturbations."""
    atoms = graph.atoms
    studies = graph.studies

    def prop(tup: str, pid: str) -> TypedProposition:
        return proposition_from_atom(atoms[tup], pid)

    cases: list[GroundedSynthesisCase] = []

    # ---- B_VALID (24 cases across IPF, HF, ONC) ----
    valid_specs: list[tuple[str, str, str, str, SynthesisClaimRecord]] = [
        # B_P1: Direct clinical promotions (9 cases)
        (
            "TB-VAL-01",
            "IPF",
            "B_P1_direct_clinical_promotion",
            "INPULSIS-1 & INPULSIS-2 replicate FVC rate promotion",
            SynthesisClaimRecord(
                "REC-VAL-01",
                "SUPPORTS",
                studies["R101"],
                ("R101:O1", "R101:O2"),
                (),
                (
                    "In R101:O1 nintedanib showed an adjusted annual FVC change of -114.7 mL/year versus -239.9 mL/year with placebo across 513 patients, while in R101:O2 the rate was -113.6 mL/year versus -207.3 mL/year.",
                ),
                (prop("R101:O1", "P-VAL-01A"), prop("R101:O2", "P-VAL-01B")),
                requests_direct_quantitative_promotion=True,
            ),
        ),
        (
            "TB-VAL-02",
            "IPF",
            "B_P1_direct_clinical_promotion",
            "ASCEND pirfenidone relative FVC decline reduction promotion",
            SynthesisClaimRecord(
                "REC-VAL-02",
                "SUPPORTS",
                studies["R102"],
                ("R102:O1", "R102:O2"),
                (),
                (
                    "In R102:O1 pirfenidone achieved a 47.9% relative reduction in the proportion of patients with a decline of 10 percentage points or more in predicted FVC or death across 555 patients, and in R102:O2 pirfenidone produced a 132.5% relative increase in the proportion of patients with no FVC decline.",
                ),
                (prop("R102:O1", "P-VAL-02A"), prop("R102:O2", "P-VAL-02B")),
                requests_direct_quantitative_promotion=True,
            ),
        ),
        (
            "TB-VAL-03",
            "IPF",
            "B_P1_direct_clinical_promotion",
            "BI 1015550 Phase 2 subgroup-specific promotion within matching subgroup target",
            SynthesisClaimRecord(
                "REC-VAL-03",
                "SUPPORTS",
                atoms["R103:O1"].context,
                ("R103:O1",),
                (),
                (
                    "In R103:O1 BI 1015550 yielded a 12-week median FVC change of 5.7 mL versus -81.7 mL in the placebo group across 73 randomized patients.",
                ),
                (prop("R103:O1", "P-VAL-03"),),
                requests_direct_quantitative_promotion=True,
            ),
        ),
        (
            "TB-VAL-04",
            "HF",
            "B_P1_direct_clinical_promotion",
            "PARADIGM-HF primary composite and mortality promotion",
            SynthesisClaimRecord(
                "REC-VAL-04",
                "SUPPORTS",
                studies["R201"],
                ("R201:O1", "R201:O2"),
                (),
                (
                    "In R201:O1 the primary composite occurred in 21.8% of the LCZ696 group versus 26.5% of the enalapril group across 8399 patients; in R201:O2 all-cause mortality was 17.0% versus 19.8%.",
                ),
                (prop("R201:O1", "P-VAL-04A"), prop("R201:O2", "P-VAL-04B")),
                requests_direct_quantitative_promotion=True,
            ),
        ),
        (
            "TB-VAL-05",
            "HF",
            "B_P1_direct_clinical_promotion",
            "PARAGON-HF NYHA improvement and renal preservation promotion",
            SynthesisClaimRecord(
                "REC-VAL-05",
                "SUPPORTS",
                studies["R202"],
                ("R202:O2", "R202:O3"),
                (),
                (
                    "In R202:O2 NYHA class improved in 15.0% of the sacubitril-valsartan group versus 12.6% of the valsartan group, while in R202:O3 worsening renal function was lower at 1.4% versus 2.7%.",
                ),
                (prop("R202:O2", "P-VAL-05A"), prop("R202:O3", "P-VAL-05B")),
                requests_direct_quantitative_promotion=True,
            ),
        ),
        (
            "TB-VAL-06",
            "HF",
            "B_P1_direct_clinical_promotion",
            "VICTORIA vericiguat primary composite and HF hospitalization promotion",
            SynthesisClaimRecord(
                "REC-VAL-06",
                "SUPPORTS",
                studies["R203"],
                ("R203:O1", "R203:O2"),
                (),
                (
                    "In R203:O1 primary-outcome events were lower in the vericiguat group at 35.5% versus 38.5% in the placebo group across 5050 patients, and in R203:O2 heart-failure hospitalization was 27.4% versus 29.6%.",
                ),
                (prop("R203:O1", "P-VAL-06A"), prop("R203:O2", "P-VAL-06B")),
                requests_direct_quantitative_promotion=True,
            ),
        ),
        (
            "TB-VAL-07",
            "ONC",
            "B_P1_direct_clinical_promotion",
            "KEYNOTE-024 pembrolizumab PFS and 6-month OS promotion",
            SynthesisClaimRecord(
                "REC-VAL-07",
                "SUPPORTS",
                studies["R301"],
                ("R301:O1", "R301:O2"),
                (),
                (
                    "In R301:O1 median progression-free survival was longer in the pembrolizumab group at 10.3 months versus 6.0 months in the chemotherapy group across 305 patients, while in R301:O2 6-month overall survival was 80.2% versus 72.4%.",
                ),
                (prop("R301:O1", "P-VAL-07A"), prop("R301:O2", "P-VAL-07B")),
                requests_direct_quantitative_promotion=True,
            ),
        ),
        (
            "TB-VAL-08",
            "ONC",
            "B_P1_direct_clinical_promotion",
            "AURA3 osimertinib overall PFS and ORR promotion",
            SynthesisClaimRecord(
                "REC-VAL-08",
                "SUPPORTS",
                studies["R302"],
                ("R302:O1", "R302:O3"),
                (),
                (
                    "In R302:O1 median progression-free survival was longer with osimertinib at 10.1 months versus 4.4 months with platinum therapy plus pemetrexed across 419 patients; in R302:O3 the objective response rate was 71% versus 31%.",
                ),
                (prop("R302:O1", "P-VAL-08A"), prop("R302:O3", "P-VAL-08B")),
                requests_direct_quantitative_promotion=True,
            ),
        ),
        (
            "TB-VAL-09",
            "ONC",
            "B_P1_direct_clinical_promotion",
            "ADAURA overall stage IB-IIIA 24-month DFS promotion",
            SynthesisClaimRecord(
                "REC-VAL-09",
                "SUPPORTS",
                studies["R303"],
                ("R303:O2",),
                (),
                (
                    "In R303:O2 24-month disease-free survival was higher in the osimertinib group at 89% versus 52% in the placebo group across 682 randomized patients.",
                ),
                (prop("R303:O2", "P-VAL-09"),),
                requests_direct_quantitative_promotion=True,
            ),
        ),
        # B_P2: Bounded translational & subgroup bridges with ProspectiveFalsifierGate (6 cases)
        (
            "TB-VAL-10",
            "IPF",
            "B_P2_bounded_translational_bridge",
            "Pirfenidone fibroblast MRTF IC50 bridged to Phase 3 IPF with falsifier gate",
            SynthesisClaimRecord(
                "REC-VAL-10",
                "CANDIDATE_UPGRADE",
                studies["R102"],
                ("R104:O1", "R104:O2"),
                (),
                (
                    "In R104:O1 pirfenidone inhibited fibroblast MRTF activation with a half-maximal inhibitory concentration of 50-150 uM, while in R104:O2 maximal inhibition was > 90%.",
                ),
                (prop("R104:O1", "P-VAL-10A"), prop("R104:O2", "P-VAL-10B")),
                prospective_gate=ProspectiveFalsifierGate(
                    "GATE-IPF-MRTF-01",
                    ("evidence_tier", "population_stratum", "followup_horizon", "readout_level"),
                    ("pirfenidone_2403mg", "placebo"),
                    "52-week FVC decline and lung biopsy MRTF nuclear translocation (%)",
                    "FVC decline reduction >=30% with paired MRTF inhibition",
                ),
            ),
        ),
        (
            "TB-VAL-11",
            "IPF",
            "B_P2_bounded_translational_bridge",
            "Phase 2 12-week BI 1015550 subgroups bridged to 52-week Phase 3 ITT IPF",
            SynthesisClaimRecord(
                "REC-VAL-11",
                "CANDIDATE_UPGRADE",
                _ctx("phase3_rct", "ipf", "overall_itt", "chronic_6m_plus", "clinical_efficacy_or_safety", "bi_1015550"),
                ("R103:O1", "R103:O2"),
                (),
                (
                    "In R103:O1 12-week median FVC change was 5.7 mL in the BI 1015550 group versus -81.7 mL on placebo, whereas in R103:O2 it was 2.7 mL versus -59.2 mL.",
                ),
                (prop("R103:O1", "P-VAL-11A"), prop("R103:O2", "P-VAL-11B")),
                prospective_gate=ProspectiveFalsifierGate(
                    "GATE-IPF-PDE4B-02",
                    ("evidence_tier", "population_stratum", "followup_horizon"),
                    ("bi_1015550_18mg_bid", "placebo"),
                    "52-week adjusted annual FVC change (mL/year)",
                    "Superiority over placebo at 52 weeks across pooled ITT strata",
                ),
            ),
        ),
        (
            "TB-VAL-12",
            "HF",
            "B_P2_bounded_translational_bridge",
            "Rat post-MI LCZ696 remodeling bridged to clinical HFrEF with falsifier gate",
            SynthesisClaimRecord(
                "REC-VAL-12",
                "CANDIDATE_UPGRADE",
                studies["R201"],
                ("R204:O1", "R204:O2"),
                (),
                (
                    "In R204:O1 left ventricular end-diastolic diameter was lower in MI-ARNi at 9.7 mm versus 10.5 mm in MI-vehicle across 17 rats, while in R204:O2 left ventricular ejection fraction was higher in MI-ARNi at 60% versus 47% in MI-vehicle.",
                ),
                (prop("R204:O1", "P-VAL-12A"), prop("R204:O2", "P-VAL-12B")),
                prospective_gate=ProspectiveFalsifierGate(
                    "GATE-HF-ARNI-03",
                    ("evidence_tier", "population_stratum", "followup_horizon", "readout_level"),
                    ("lcz696_200mg_bid", "enalapril_10mg_bid"),
                    "Cardiovascular death or heart-failure hospitalization (%)",
                    "Hazard ratio <0.85 at p<0.01 in chronic HFrEF",
                ),
            ),
        ),
        (
            "TB-VAL-13",
            "ONC",
            "B_P2_bounded_translational_bridge",
            "AURA3 CNS-metastases subgroup bridged to overall T790M+ ITT target with gate",
            SynthesisClaimRecord(
                "REC-VAL-13",
                "CANDIDATE_UPGRADE",
                studies["R302"],
                ("R302:O2",),
                (),
                (
                    "In R302:O2 median progression-free survival was longer with osimertinib at 8.5 months versus 4.2 months with platinum therapy plus pemetrexed across 144 patients.",
                ),
                (prop("R302:O2", "P-VAL-13"),),
                prospective_gate=ProspectiveFalsifierGate(
                    "GATE-ONC-CNS-04",
                    ("population_stratum",),
                    ("osimertinib_80mg", "platinum_pemetrexed"),
                    "ITT progression-free survival (months)",
                    "Hazard ratio <0.50 across unselected T790M-positive patients",
                ),
            ),
        ),
        (
            "TB-VAL-14",
            "ONC",
            "B_P2_bounded_translational_bridge",
            "ADAURA Stage II-IIIA subgroup bridged to overall Stage IB-IIIA target with gate",
            SynthesisClaimRecord(
                "REC-VAL-14",
                "CANDIDATE_UPGRADE",
                studies["R303"],
                ("R303:O1",),
                (),
                (
                    "In R303:O1 24-month disease-free survival reached 90% in the osimertinib group versus 44% in the placebo group across 470 randomized patients.",
                ),
                (prop("R303:O1", "P-VAL-14"),),
                prospective_gate=ProspectiveFalsifierGate(
                    "GATE-ONC-ADAURA-05",
                    ("population_stratum",),
                    ("adjuvant_osimertinib_80mg", "placebo"),
                    "Stage IB-IIIA 24-month disease-free survival (%)",
                    "Statistically significant DFS benefit including stage IB",
                ),
            ),
        ),
        (
            "TB-VAL-15",
            "ONC",
            "B_P2_bounded_translational_bridge",
            "Preclinical EGFR exon20ins IC50 scoped qualitatively within exon20ins cellular context",
            SynthesisClaimRecord(
                "REC-VAL-15",
                "SUPPORTS",
                studies["R304"],
                ("R304:O1",),
                (),
                (
                    "In R304:O1 osimertinib inhibited EGFR exon 20 insertion-mutant cells with an IC50 of 14.7-62.7 nM.",
                ),
                (prop("R304:O1", "P-VAL-15"),),
                requests_direct_quantitative_promotion=False,
            ),
        ),
        # B_P3: Safety & counterevidence scoping qualifications (5 cases)
        (
            "TB-VAL-16",
            "IPF",
            "B_P3_safety_counterevidence_scoping",
            "INPULSIS-1 diarrhea rate and <5% discontinuation counterevidence scoping",
            SynthesisClaimRecord(
                "REC-VAL-16",
                "SCOPING_QUALIFICATION",
                studies["R101"],
                ("R101:O1",),
                ("R101:O3", "R101:O4"),
                (
                    "In R101:O1 nintedanib slowed annual FVC change to -114.7 mL/year versus -239.9 mL/year on placebo, although in R101:O3 diarrhea increased from 18.6% on placebo to 61.5% on nintedanib.",
                    "In R101:O4 diarrhea-associated medication discontinuation remained < 5%.",
                ),
                (),
            ),
        ),
        (
            "TB-VAL-17",
            "HF",
            "B_P3_safety_counterevidence_scoping",
            "PARAGON-HF neutral HFpEF cardiovascular mortality counterevidence record",
            SynthesisClaimRecord(
                "REC-VAL-17",
                "CONTRADICTS",
                studies["R202"],
                (),
                ("R202:O1",),
                (
                    "In R202:O1 cardiovascular death incidence was 8.5% in the sacubitril-valsartan group versus 8.9% in the valsartan group across 4796 patients.",
                ),
                (),
            ),
        ),
        (
            "TB-VAL-18",
            "HF",
            "B_P3_safety_counterevidence_scoping",
            "VICTORIA symptomatic hypotension safety qualification",
            SynthesisClaimRecord(
                "REC-VAL-18",
                "SCOPING_QUALIFICATION",
                studies["R203"],
                ("R203:O1",),
                ("R203:O3",),
                (
                    "In R203:O1 vericiguat reduced primary composite events to 35.5% versus 38.5% on placebo, whereas in R203:O3 symptomatic hypotension increased to 9.1% in the vericiguat group versus 7.9% in the placebo group.",
                ),
                (),
            ),
        ),
        (
            "TB-VAL-19",
            "ONC",
            "B_P3_safety_counterevidence_scoping",
            "KEYNOTE-024 lower Grade 3-5 treatment-related adverse events with pembrolizumab",
            SynthesisClaimRecord(
                "REC-VAL-19",
                "SCOPING_QUALIFICATION",
                studies["R301"],
                ("R301:O3",),
                (),
                (
                    "In R301:O3 grade 3 to 5 treatment-related adverse events decreased from 53.3% in the chemotherapy group to 26.6% in the pembrolizumab group across 304 patients.",
                ),
                (),
            ),
        ),
        (
            "TB-VAL-20",
            "ONC",
            "B_P3_safety_counterevidence_scoping",
            "AURA3 verbatim Grade >=3 AE sentence as a valid synthesis scoping record",
            SynthesisClaimRecord(
                "REC-VAL-20",
                "SCOPING_QUALIFICATION",
                studies["R302"],
                ("R302:O4",),
                (),
                (
                    "In R302:O4 the proportion of patients with adverse events of grade 3 or higher was lower with osimertinib (23%) than with platinum therapy plus pemetrexed (47%).",
                ),
                (),
            ),
        ),
        # B_P4: Natural comparative, appositive, inverted negated bound, and SI-sensitive valid controls (4 cases)
        (
            "TB-VAL-21",
            "ONC",
            "B_P4_natural_comparative_and_censored_syntax",
            "Valid comma-appositive comparative phrasing on AURA3 PFS",
            SynthesisClaimRecord(
                "REC-VAL-21",
                "SUPPORTS",
                studies["R302"],
                ("R302:O1",),
                (),
                (
                    "In R302:O1 osimertinib had a longer median progression-free survival, 10.1 months, than platinum therapy plus pemetrexed, 4.4 months.",
                ),
                (prop("R302:O1", "P-VAL-21"),),
                requests_direct_quantitative_promotion=True,
            ),
        ),
        (
            "TB-VAL-22",
            "IPF",
            "B_P4_natural_comparative_and_censored_syntax",
            "Valid inverted negated bound ('not <= 90%' -> '> 90%') on R104:O2",
            SynthesisClaimRecord(
                "REC-VAL-22",
                "SUPPORTS",
                studies["R104"],
                ("R104:O2",),
                (),
                (
                    "In R104:O2 pirfenidone maximal fibroblast inhibition was not <= 90%.",
                ),
                (prop("R104:O2", "P-VAL-22"),),
                requests_direct_quantitative_promotion=False,  # the atom is a preclinical outcome that was not checked against its primary source
            ),
        ),
        (
            "TB-VAL-23",
            "HF",
            "B_P4_natural_comparative_and_censored_syntax",
            "Valid index-aligned 'respectively' across PARADIGM-HF all-cause and CV mortality",
            SynthesisClaimRecord(
                "REC-VAL-23",
                "SUPPORTS",
                studies["R201"],
                ("R201:O2", "R201:O3"),
                (),
                (
                    "R201:O2 and R201:O3 recorded 17.0% versus 19.8% and 13.3% versus 16.5%, respectively.",
                ),
                (prop("R201:O2", "P-VAL-23A"), prop("R201:O3", "P-VAL-23B")),
                requests_direct_quantitative_promotion=True,
            ),
        ),
        (
            "TB-VAL-24",
            "HF",
            "B_P4_natural_comparative_and_censored_syntax",
            "Valid 'although' contrast across two distinct HFrEF trials without pooling (qualitative: two different drugs)",
            SynthesisClaimRecord(
                "REC-VAL-24",
                "SUPPORTS",
                studies["R203"],
                ("R201:O1", "R203:O1"),
                (),
                (
                    "In R201:O1 primary events were 21.8% in the LCZ696 group versus 26.5% in the enalapril group, although in R203:O1 primary events were 35.5% in the vericiguat group versus 38.5% in the placebo group.",
                ),
                (prop("R201:O1", "P-VAL-24A"), prop("R203:O1", "P-VAL-24B")),
                requests_direct_quantitative_promotion=False,  # the two trials tested different drugs, so one of them is a bounded bridge on the intervention axis
            ),
        ),
    ]
    for cid, dom, fam, desc, rec in valid_specs:
        cases.append(GroundedSynthesisCase(cid, dom, fam, False, desc, rec))

    # ---- B_DEFECTS (60 realistic clinical/translational synthesis defects across D1-D6) ----

    # D1: Cross-trial endpoint pooling across incompatible designs/indications (10 cases)
    d1_specs: list[tuple[str, str, str, SynthesisClaimRecord]] = [
        (
            "TB-D1-01",
            "HF",
            "Pooled HFrEF PARADIGM-HF and HFpEF PARAGON-HF mortality in one clause + inadmissible HFpEF->HFrEF transfer",
            SynthesisClaimRecord(
                "REC-D1-01",
                "SUPPORTS",
                studies["R201"],
                ("R201:O3", "R202:O1"),
                (),
                ("Cardiovascular death incidence was 13.3% versus 16.5% [R201:O3, R202:O1].",),
                (prop("R201:O3", "P-D1-01A"), prop("R202:O1", "P-D1-01B")),
                requests_direct_quantitative_promotion=True,
            ),
        ),
        (
            "TB-D1-02",
            "HF",
            "Promoting HFpEF PARAGON-HF outcome into HFrEF target context (inadmissible indication transfer)",
            SynthesisClaimRecord(
                "REC-D1-02",
                "SUPPORTS",
                studies["R201"],
                ("R202:O2",),
                (),
                ("In R202:O2 NYHA class improved in 15.0% of the sacubitril-valsartan group versus 12.6% of the valsartan group.",),
                (prop("R202:O2", "P-D1-02"),),
                requests_direct_quantitative_promotion=True,
            ),
        ),
        (
            "TB-D1-03",
            "ONC",
            "Pooled 1L PD-L1>=50% KEYNOTE-024 and 2L EGFR-T790M+ AURA3 PFS in one clause",
            SynthesisClaimRecord(
                "REC-D1-03",
                "SUPPORTS",
                studies["R301"],
                ("R301:O1", "R302:O1"),
                (),
                ("Median progression-free survival was 10.3 months versus 6.0 months (R301:O1, R302:O1).",),
                (prop("R301:O1", "P-D1-03A"), prop("R302:O1", "P-D1-03B")),
                requests_direct_quantitative_promotion=True,
            ),
        ),
        (
            "TB-D1-04",
            "ONC",
            "Promoting 2L T790M+ AURA3 PFS into adjuvant resected NSCLC target context",
            SynthesisClaimRecord(
                "REC-D1-04",
                "SUPPORTS",
                studies["R303"],
                ("R302:O1",),
                (),
                ("In R302:O1 median progression-free survival was 10.1 months with osimertinib versus 4.4 months with platinum therapy plus pemetrexed.",),
                (prop("R302:O1", "P-D1-04"),),
                requests_direct_quantitative_promotion=True,
            ),
        ),
        (
            "TB-D1-05",
            "IPF",
            "Pooled 52-week INPULSIS-1 and 12-week Phase 2 BI 1015550 FVC endpoints via trailing 'see' citations",
            SynthesisClaimRecord(
                "REC-D1-05",
                "SUPPORTS",
                studies["R101"],
                ("R101:O1", "R103:O1"),
                (),
                ("Forced vital capacity change was -114.7 mL/year versus -239.9 mL/year; see R101:O1 and R103:O1.",),
                (prop("R101:O1", "P-D1-05A"), prop("R103:O1", "P-D1-05B")),
                requests_direct_quantitative_promotion=True,
            ),
        ),
        (
            "TB-D1-06",
            "IPF",
            "Pooled INPULSIS-1 and INPULSIS-2 FVC rates under separate parenthetical citations",
            SynthesisClaimRecord(
                "REC-D1-06",
                "SUPPORTS",
                studies["R101"],
                ("R101:O1", "R101:O2"),
                (),
                ("Adjusted annual FVC change was -114.7 mL/year versus -239.9 mL/year (R101:O1) and (R101:O2).",),
                (prop("R101:O1", "P-D1-06A"), prop("R101:O2", "P-D1-06B")),
                requests_direct_quantitative_promotion=True,
            ),
        ),
        (
            "TB-D1-07",
            "HF",
            "Pooled PARADIGM-HF and VICTORIA primary composites with leading 'For X and Y' header",
            SynthesisClaimRecord(
                "REC-D1-07",
                "SUPPORTS",
                studies["R201"],
                ("R201:O1", "R203:O1"),
                (),
                ("For R201:O1 and R203:O1, primary composite events occurred in 21.8% versus 26.5%.",),
                (prop("R201:O1", "P-D1-07A"), prop("R203:O1", "P-D1-07B")),
                requests_direct_quantitative_promotion=True,
            ),
        ),
        (
            "TB-D1-08",
            "ONC",
            "Promoting adjuvant ADAURA DFS into 1L metastatic PD-L1-high NSCLC target context",
            SynthesisClaimRecord(
                "REC-D1-08",
                "SUPPORTS",
                studies["R301"],
                ("R303:O2",),
                (),
                ("In R303:O2 24-month disease-free survival was 89% in the osimertinib group versus 52% in the placebo group.",),
                (prop("R303:O2", "P-D1-08"),),
                requests_direct_quantitative_promotion=True,
            ),
        ),
        (
            "TB-D1-09",
            "IPF",
            "Synthetic midpoint pooling (-114.15 mL/year) across INPULSIS-1 and INPULSIS-2",
            SynthesisClaimRecord(
                "REC-D1-09",
                "SUPPORTS",
                studies["R101"],
                ("R101:O1", "R101:O2"),
                (),
                ("In R101:O1 pooled annual FVC change was -114.15 mL/year with nintedanib, while in R101:O2 placebo was -207.3 mL/year.",),
                (prop("R101:O1", "P-D1-09A"), prop("R101:O2", "P-D1-09B")),
                requests_direct_quantitative_promotion=True,
            ),
        ),
        (
            "TB-D1-10",
            "HF",
            "Cross-indication IPF outcome cited inside HFrEF synthesis promotion",
            SynthesisClaimRecord(
                "REC-D1-10",
                "SUPPORTS",
                studies["R203"],
                ("R102:O1",),
                (),
                ("In R102:O1 pirfenidone achieved a 47.9% relative reduction across 555 patients.",),
                (prop("R102:O1", "P-D1-10"),),
                requests_direct_quantitative_promotion=True,
            ),
        ),
    ]
    for cid, dom, desc, rec in d1_specs:
        cases.append(GroundedSynthesisCase(cid, dom, "D1_cross_trial_incompatible_pooling", True, desc, rec))

    # D2: Active/placebo arm swaps, non-contrast multi-arm bindings & comma-appositive swaps (10 cases)
    d2_specs: list[tuple[Any, ...]] = [
        (
            "TB-D2-01",
            "IPF",
            "INPULSIS-1 nintedanib and placebo annual FVC rates swapped",
            SynthesisClaimRecord(
                "REC-D2-01",
                "SUPPORTS",
                studies["R101"],
                ("R101:O1",),
                (),
                ("In R101:O1 nintedanib showed an annual FVC change of -239.9 mL/year versus -114.7 mL/year with placebo.",),
                (prop("R101:O1", "P-D2-01"),),
            ),
        ),
        (
            "TB-D2-02",
            "IPF",
            "INPULSIS-1 diarrhea rates inverted via from/to trajectory swap",
            SynthesisClaimRecord(
                "REC-D2-02",
                "SCOPING_QUALIFICATION",
                studies["R101"],
                (),
                ("R101:O3",),
                ("In R101:O3 diarrhea fell from 61.5% in the placebo group to 18.6% in the nintedanib group.",),
                (),
            ),
        ),
        (
            "TB-D2-03",
            "HF",
            "PARADIGM-HF LCZ696 and enalapril primary composite rates swapped",
            SynthesisClaimRecord(
                "REC-D2-03",
                "SUPPORTS",
                studies["R201"],
                ("R201:O1",),
                (),
                ("In R201:O1 primary events occurred in 26.5% of the LCZ696 group versus 21.8% of the enalapril group.",),
                (prop("R201:O1", "P-D2-03"),),
            ),
        ),
        (
            "TB-D2-04",
            "HF",
            "VICTORIA comma-appositive single-value swap attributing placebo rate (38.5%) to vericiguat",
            SynthesisClaimRecord(
                "REC-D2-04",
                "SUPPORTS",
                studies["R203"],
                ("R203:O1",),
                (),
                ("In R203:O1 the vericiguat group had a lower primary composite rate, 38.5%, than the placebo group.",),
                (prop("R203:O1", "P-D2-04"),),
            ),
        ),
        (
            "TB-D2-05",
            "ONC",
            "KEYNOTE-024 comma-appositive two-value PFS swap ('longer ... 6.0 months ... than ... 10.3 months')",
            SynthesisClaimRecord(
                "REC-D2-05",
                "SUPPORTS",
                studies["R301"],
                ("R301:O1",),
                (),
                ("In R301:O1 pembrolizumab had a longer median progression-free survival, 6.0 months, than chemotherapy, 10.3 months.",),
                (prop("R301:O1", "P-D2-05"),),
            ),
        ),
        (
            "TB-D2-06",
            "ONC",
            "AURA3 progression-free survival values exchanged between osimertinib and platinum-pemetrexed",
            SynthesisClaimRecord(
                "REC-D2-06",
                "SUPPORTS",
                studies["R302"],
                ("R302:O1",),
                (),
                ("In R302:O1 median progression-free survival was 4.4 months with osimertinib versus 10.1 months with platinum therapy plus pemetrexed.",),
                (prop("R302:O1", "P-D2-06"),),
            ),
            ("In R302:O1 median progression-free survival was 10.1 months with osimertinib versus 4.4 months with platinum therapy plus pemetrexed.",),
        ),
        (
            "TB-D2-07",
            "ONC",
            "ADAURA parenthetical arm side-note swap ('osimertinib group (placebo group also evaluated) had 52%')",
            SynthesisClaimRecord(
                "REC-D2-07",
                "SUPPORTS",
                studies["R303"],
                ("R303:O2",),
                (),
                ("In R303:O2 the osimertinib group (placebo group also evaluated) had 24-month disease-free survival of 52%.",),
                (prop("R303:O2", "P-D2-07"),),
            ),
        ),
        (
            "TB-D2-08",
            "HF",
            "PARAGON-HF polarity reversal claiming worsening renal function increased with sacubitril-valsartan",
            SynthesisClaimRecord(
                "REC-D2-08",
                "SUPPORTS",
                studies["R202"],
                ("R202:O3",),
                (),
                ("In R202:O3 worsening renal function was higher in the sacubitril-valsartan group (1.4%) than in the valsartan group (2.7%).",),
                (prop("R202:O3", "P-D2-08"),),
            ),
        ),
        (
            "TB-D2-09",
            "ONC",
            "KEYNOTE-024 post-hoc focal denial ('reduction ... seen in chemotherapy, not in pembrolizumab')",
            SynthesisClaimRecord(
                "REC-D2-09",
                "SCOPING_QUALIFICATION",
                studies["R301"],
                ("R301:O3",),
                (),
                ("In R301:O3 the reduction in grade 3 to 5 adverse events was seen in the chemotherapy group (53.3%), not in the pembrolizumab group (26.6%).",),
                (),
            ),
        ),
        (
            "TB-D2-10",
            "IPF",
            "TypedProposition focal/comparator arm swap on BI 1015550 R103:O1 despite valid prose",
            SynthesisClaimRecord(
                "REC-D2-10",
                "SUPPORTS",
                atoms["R103:O1"].context,
                ("R103:O1",),
                (),
                ("In R103:O1 BI 1015550 yielded a 12-week median FVC change of 5.7 mL versus -81.7 mL in the placebo group.",),
                (
                    TypedProposition(
                        "P-D2-10",
                        "R103:O1",
                        atoms["R103:O1"].endpoint,
                        ("VAR:R103:Placebo",),
                        ("VAR:R103:BI1015550_18mg_BID",),
                        atoms["R103:O1"].focal_quantity,
                        atoms["R103:O1"].statistic_kind,
                        atoms["R103:O1"].sample_size,
                        atoms["R103:O1"].comparator_quantity,
                    ),
                ),
                requests_direct_quantitative_promotion=True,
            ),
        ),
    ]
    for spec in d2_specs:
        cid, dom, desc, rec = spec[:4]
        cases.append(
            GroundedSynthesisCase(
                cid, dom, "D2_active_placebo_arm_and_appositive_swap", True, desc, rec, None, spec[4] if len(spec) > 4 else None
            )
        )

    # D3: Subgroup-to-overall promotion & cross-stratum misattribution (10 cases)
    d3_specs: list[tuple[str, str, str, SynthesisClaimRecord]] = [
        (
            "TB-D3-01",
            "IPF",
            "Direct ITT promotion of BI 1015550 antifibrotic-naive subgroup R103:O1 into overall ITT Phase 2 context",
            SynthesisClaimRecord(
                "REC-D3-01",
                "SUPPORTS",
                studies["R103"],
                ("R103:O1",),
                (),
                ("In R103:O1 median FVC change was 5.7 mL in the BI 1015550 group versus -81.7 mL in the placebo group.",),
                (prop("R103:O1", "P-D3-01"),),
                requests_direct_quantitative_promotion=True,
            ),
        ),
        (
            "TB-D3-02",
            "IPF",
            "Direct ITT promotion of BI 1015550 background-antifibrotic subgroup R103:O2 into overall ITT Phase 2 context",
            SynthesisClaimRecord(
                "REC-D3-02",
                "SUPPORTS",
                studies["R103"],
                ("R103:O2",),
                (),
                ("In R103:O2 median FVC change was 2.7 mL in the BI 1015550 group versus -59.2 mL in the placebo group.",),
                (prop("R103:O2", "P-D3-02"),),
                requests_direct_quantitative_promotion=True,
            ),
        ),
        (
            "TB-D3-03",
            "IPF",
            "Antifibrotic-naive subgroup gain (5.7 mL) misattributed to background-antifibrotic atom R103:O2",
            SynthesisClaimRecord(
                "REC-D3-03",
                "SUPPORTS",
                atoms["R103:O2"].context,
                ("R103:O2",),
                (),
                ("In R103:O2 BI 1015550 achieved a median FVC change of 5.7 mL versus -59.2 mL on placebo.",),
                (prop("R103:O2", "P-D3-03"),),
                requests_direct_quantitative_promotion=True,
            ),
        ),
        (
            "TB-D3-04",
            "ONC",
            "Direct promotion of AURA3 CNS-metastases subgroup R302:O2 to overall T790M+ ITT context without gate",
            SynthesisClaimRecord(
                "REC-D3-04",
                "SUPPORTS",
                studies["R302"],
                ("R302:O2",),
                (),
                ("In R302:O2 median progression-free survival was 8.5 months with osimertinib versus 4.2 months with platinum therapy plus pemetrexed.",),
                (prop("R302:O2", "P-D3-04"),),
                requests_direct_quantitative_promotion=True,
            ),
        ),
        (
            "TB-D3-05",
            "ONC",
            "AURA3 overall ITT PFS (10.1 months) misattributed to CNS-metastases subgroup atom R302:O2",
            SynthesisClaimRecord(
                "REC-D3-05",
                "SUPPORTS",
                atoms["R302:O2"].context,
                ("R302:O2",),
                (),
                ("In R302:O2 CNS-metastases progression-free survival was 10.1 months with osimertinib versus 4.2 months with platinum therapy plus pemetrexed.",),
                (prop("R302:O2", "P-D3-05"),),
                requests_direct_quantitative_promotion=True,
            ),
        ),
        (
            "TB-D3-06",
            "ONC",
            "AURA3 CNS subgroup TypedProposition claiming overall ITT sample size n=419 instead of n=144",
            SynthesisClaimRecord(
                "REC-D3-06",
                "SUPPORTS",
                atoms["R302:O2"].context,
                ("R302:O2",),
                (),
                ("In R302:O2 median progression-free survival was 8.5 months with osimertinib versus 4.2 months with platinum therapy plus pemetrexed.",),
                (
                    TypedProposition(
                        "P-D3-06",
                        "R302:O2",
                        atoms["R302:O2"].endpoint,
                        atoms["R302:O2"].contrast.focal_arms,
                        atoms["R302:O2"].contrast.comparator_arms,
                        atoms["R302:O2"].focal_quantity,
                        atoms["R302:O2"].statistic_kind,
                        419,
                        atoms["R302:O2"].comparator_quantity,
                    ),
                ),
                requests_direct_quantitative_promotion=True,
            ),
        ),
        (
            "TB-D3-07",
            "ONC",
            "Direct promotion of ADAURA Stage II-IIIA subgroup R303:O1 (90% vs 44%) into overall Stage IB-IIIA ITT context",
            SynthesisClaimRecord(
                "REC-D3-07",
                "SUPPORTS",
                studies["R303"],
                ("R303:O1",),
                (),
                ("In R303:O1 24-month disease-free survival was 90% in the osimertinib group versus 44% in the placebo group.",),
                (prop("R303:O1", "P-D3-07"),),
                requests_direct_quantitative_promotion=True,
            ),
        ),
        (
            "TB-D3-08",
            "ONC",
            "ADAURA Stage II-IIIA primary rate (90% vs 44%) misattributed to overall Stage IB-IIIA atom R303:O2",
            SynthesisClaimRecord(
                "REC-D3-08",
                "SUPPORTS",
                studies["R303"],
                ("R303:O2",),
                (),
                ("In R303:O2 overall stage IB to IIIA 24-month disease-free survival was 90% in the osimertinib group versus 44% in the placebo group.",),
                (prop("R303:O2", "P-D3-08"),),
                requests_direct_quantitative_promotion=True,
            ),
        ),
        (
            "TB-D3-09",
            "ONC",
            "CANDIDATE_UPGRADE from ADAURA Stage II-IIIA subgroup to overall ITT omitting ProspectiveFalsifierGate",
            SynthesisClaimRecord(
                "REC-D3-09",
                "CANDIDATE_UPGRADE",
                studies["R303"],
                ("R303:O1",),
                (),
                ("In R303:O1 24-month disease-free survival was 90% in the osimertinib group versus 44% in the placebo group.",),
                (prop("R303:O1", "P-D3-09"),),
                prospective_gate=None,
            ),
        ),
        (
            "TB-D3-10",
            "IPF",
            "CANDIDATE_UPGRADE from BI 1015550 subgroup to Phase 3 ITT with incomplete ProspectiveFalsifierGate axes",
            SynthesisClaimRecord(
                "REC-D3-10",
                "CANDIDATE_UPGRADE",
                studies["R101"],
                ("R103:O1",),
                (),
                ("In R103:O1 median FVC change was 5.7 mL in the BI 1015550 group versus -81.7 mL in the placebo group.",),
                (prop("R103:O1", "P-D3-10"),),
                prospective_gate=ProspectiveFalsifierGate(
                    "GATE-INCOMPLETE-D3-10",
                    ("evidence_tier",),  # omits population_stratum and followup_horizon
                    ("bi_1015550", "placebo"),
                    "52-week FVC change (mL/year)",
                    "Superiority at p<0.05",
                ),
            ),
        ),
    ]
    for cid, dom, desc, rec in d3_specs:
        cases.append(GroundedSynthesisCase(cid, dom, "D3_subgroup_to_overall_promotion", True, desc, rec))

    # D4: Rounding, unit drift, and SI prefix case collisions (10 cases)
    d4_specs: list[tuple[str, str, str, SynthesisClaimRecord]] = [
        (
            "TB-D4-01",
            "IPF",
            "Integer rounding of INPULSIS-1 FVC slope (-114.7 -> -115 mL/year and -239.9 -> -240 mL/year)",
            SynthesisClaimRecord(
                "REC-D4-01",
                "SUPPORTS",
                studies["R101"],
                ("R101:O1",),
                (),
                ("In R101:O1 adjusted annual FVC change was -115 mL/year with nintedanib versus -240 mL/year with placebo.",),
                (prop("R101:O1", "P-D4-01"),),
            ),
        ),
        (
            "TB-D4-02",
            "HF",
            "Integer rounding of PARADIGM-HF primary composite (21.8% -> 22% and 26.5% -> 27%)",
            SynthesisClaimRecord(
                "REC-D4-02",
                "SUPPORTS",
                studies["R201"],
                ("R201:O1",),
                (),
                ("In R201:O1 primary composite events occurred in 22% of the LCZ696 group versus 27% of the enalapril group.",),
                (prop("R201:O1", "P-D4-02"),),
            ),
        ),
        (
            "TB-D4-03",
            "ONC",
            "Rounding of KEYNOTE-024 median PFS (10.3 -> 10 months)",
            SynthesisClaimRecord(
                "REC-D4-03",
                "SUPPORTS",
                studies["R301"],
                ("R301:O1",),
                (),
                ("In R301:O1 median progression-free survival was 10 months in the pembrolizumab group versus 6.0 months in the chemotherapy group.",),
                (prop("R301:O1", "P-D4-03"),),
            ),
        ),
        (
            "TB-D4-04",
            "IPF",
            "SI case collision: pirfenidone MRTF IC50 stated in micrometres ('50-150 um') instead of micromolar ('uM')",
            SynthesisClaimRecord(
                "REC-D4-04",
                "SUPPORTS",
                studies["R104"],
                ("R104:O1",),
                (),
                ("In R104:O1 pirfenidone inhibited fibroblast MRTF activation at 50 to 150 um.",),
                (prop("R104:O1", "P-D4-04"),),
            ),
        ),
        (
            "TB-D4-05",
            "ONC",
            "SI case collision: osimertinib EGFR exon20ins IC50 stated in nanometres ('14.7 to 62.7 nm') instead of nanomolar ('nM')",
            SynthesisClaimRecord(
                "REC-D4-05",
                "SUPPORTS",
                studies["R304"],
                ("R304:O1",),
                (),
                ("In R304:O1 osimertinib inhibited EGFR exon 20 insertion-mutant cells at 14.7 to 62.7 nm.",),
                (prop("R304:O1", "P-D4-05"),),
            ),
        ),
        (
            "TB-D4-06",
            "HF",
            "SI case collision: rat LV end-diastolic diameter stated in millimolar ('9.7 mM vs 10.5 mM') instead of millimetres ('mm')",
            SynthesisClaimRecord(
                "REC-D4-06",
                "SUPPORTS",
                studies["R204"],
                ("R204:O1",),
                (),
                ("In R204:O1 MI-ARNi lowered left ventricular end-diastolic diameter to 9.7 mM versus 10.5 mM in MI-vehicle.",),
                (prop("R204:O1", "P-D4-06"),),
            ),
        ),
        (
            "TB-D4-07",
            "IPF",
            "Unit rate-to-volume drift: INPULSIS-1 mL/year slope stated as litres ('-114.7 L vs -239.9 L')",
            SynthesisClaimRecord(
                "REC-D4-07",
                "SUPPORTS",
                studies["R101"],
                ("R101:O1",),
                (),
                ("In R101:O1 adjusted annual FVC change was -114.7 L with nintedanib versus -239.9 L with placebo.",),
                (prop("R101:O1", "P-D4-07"),),
            ),
        ),
        (
            "TB-D4-08",
            "ONC",
            "Time-unit drift: KEYNOTE-024 median PFS stated in weeks ('10.3 weeks vs 6.0 weeks') instead of months",
            SynthesisClaimRecord(
                "REC-D4-08",
                "SUPPORTS",
                studies["R301"],
                ("R301:O1",),
                (),
                ("In R301:O1 median progression-free survival was 10.3 weeks in the pembrolizumab group versus 6.0 weeks in the chemotherapy group.",),
                (prop("R301:O1", "P-D4-08"),),
            ),
        ),
        (
            "TB-D4-09",
            "IPF",
            "Parenthesized unit header drift: BI 1015550 FVC change (L) applied to mL numerals 5.7 and -81.7",
            SynthesisClaimRecord(
                "REC-D4-09",
                "SUPPORTS",
                atoms["R103:O1"].context,
                ("R103:O1",),
                (),
                ("In R103:O1 median FVC change (L) was 5.7 in the BI 1015550 group versus -81.7 in the placebo group.",),
                (prop("R103:O1", "P-D4-09"),),
            ),
        ),
        (
            "TB-D4-10",
            "HF",
            "Sample-size role confusion: rat n=17 stated as 17% ejection fraction via parenthesized (%) header",
            SynthesisClaimRecord(
                "REC-D4-10",
                "SUPPORTS",
                studies["R204"],
                ("R204:O2",),
                (),
                ("In R204:O2 left ventricular ejection fraction (%) was 17 in MI-ARNi.",),
                (prop("R204:O2", "P-D4-10"),),
            ),
        ),
    ]
    for cid, dom, desc, rec in d4_specs:
        cases.append(GroundedSynthesisCase(cid, dom, "D4_rounding_unit_and_si_case_drift", True, desc, rec))

    # D5: Preclinical-to-clinical context conflation (10 cases)
    d5_specs: list[tuple[str, str, str, SynthesisClaimRecord, str | None]] = [
        (
            "TB-D5-01",
            "IPF",
            "Direct quantitative promotion of fibroblast MRTF IC50 (R104:O1) into Phase 3 IPF clinical target",
            SynthesisClaimRecord(
                "REC-D5-01",
                "SUPPORTS",
                studies["R102"],
                ("R104:O1",),
                (),
                ("In R104:O1 pirfenidone inhibited MRTF activation at 50-150 uM.",),
                (prop("R104:O1", "P-D5-01"),),
                requests_direct_quantitative_promotion=True,
            ),
            None,
        ),
        (
            "TB-D5-02",
            "IPF",
            "Direct quantitative promotion of fibroblast >90% MRTF inhibition (R104:O2) into Phase 3 IPF clinical target",
            SynthesisClaimRecord(
                "REC-D5-02",
                "SUPPORTS",
                studies["R102"],
                ("R104:O2",),
                (),
                ("In R104:O2 pirfenidone maximal fibroblast inhibition was > 90%.",),
                (prop("R104:O2", "P-D5-02"),),
                requests_direct_quantitative_promotion=True,
            ),
            None,
        ),
        (
            "TB-D5-03",
            "HF",
            "Direct quantitative promotion of rat post-MI LV diameter (R204:O1) into Phase 3 HFrEF clinical target",
            SynthesisClaimRecord(
                "REC-D5-03",
                "SUPPORTS",
                studies["R201"],
                ("R204:O1",),
                (),
                ("In R204:O1 MI-ARNi lowered left ventricular end-diastolic diameter to 9.7 mm versus 10.5 mm in MI-vehicle.",),
                (prop("R204:O1", "P-D5-03"),),
                requests_direct_quantitative_promotion=True,
            ),
            None,
        ),
        (
            "TB-D5-04",
            "HF",
            "Direct quantitative promotion of rat post-MI LVEF 60% vs 47% (R204:O2) into Phase 3 HFrEF clinical target",
            SynthesisClaimRecord(
                "REC-D5-04",
                "SUPPORTS",
                studies["R201"],
                ("R204:O2",),
                (),
                ("In R204:O2 left ventricular ejection fraction was higher in MI-ARNi at 60% versus 47% in MI-vehicle.",),
                (prop("R204:O2", "P-D5-04"),),
                requests_direct_quantitative_promotion=True,
            ),
            None,
        ),
        (
            "TB-D5-05",
            "ONC",
            "Inadmissible upgrade of Ba/F3 engineered cell line IC50 (R304:O1, below primary_human_cells floor) to Phase 3 clinical target",
            SynthesisClaimRecord(
                "REC-D5-05",
                "CANDIDATE_UPGRADE",
                _ctx("phase3_rct", "nsclc_egfr_exon20ins", "overall_itt", "chronic_6m_plus", "clinical_efficacy_or_safety", "osimertinib"),
                ("R304:O1",),
                (),
                ("In R304:O1 osimertinib inhibited EGFR exon 20 insertion-mutant cells with an IC50 of 14.7-62.7 nM.",),
                (prop("R304:O1", "P-D5-05"),),
                prospective_gate=ProspectiveFalsifierGate(
                    "GATE-EXON20-05",
                    ("evidence_tier", "population_stratum", "followup_horizon", "readout_level"),
                    ("osimertinib_80mg", "platinum_chemotherapy"),
                    "Progression-free survival (months)",
                    "Median PFS >7 months",
                ),
            ),
            None,
        ),
        (
            "TB-D5-06",
            "ONC",
            "Conflating EGFR exon20ins preclinical IC50 (R304:O1) with EGFR T790M Phase 3 clinical context (R302)",
            SynthesisClaimRecord(
                "REC-D5-06",
                "SUPPORTS",
                studies["R302"],
                ("R304:O1",),
                (),
                ("In R304:O1 osimertinib inhibited EGFR exon 20 insertion-mutant cells at 14.7-62.7 nM.",),
                (prop("R304:O1", "P-D5-06"),),
            ),
            None,
        ),
        (
            "TB-D5-07",
            "HF",
            "Rat post-MI LCZ696 remodeling bridged into HFpEF (PARAGON-HF) clinical context (inadmissible indication mismatch)",
            SynthesisClaimRecord(
                "REC-D5-07",
                "CANDIDATE_UPGRADE",
                studies["R202"],
                ("R204:O2",),
                (),
                ("In R204:O2 left ventricular ejection fraction rose from 47% in MI-vehicle to 60% in MI-ARNi.",),
                (prop("R204:O2", "P-D5-07"),),
                prospective_gate=ProspectiveFalsifierGate(
                    "GATE-HFPEF-07",
                    ("evidence_tier", "indication", "population_stratum", "followup_horizon", "readout_level"),
                    ("sacubitril_valsartan", "valsartan"),
                    "Cardiovascular death (%)",
                    "HR <0.85",
                ),
            ),
            None,
        ),
        (
            "TB-D5-08",
            "IPF",
            "CANDIDATE_UPGRADE from fibroblast MRTF inhibition (R104:O2) to Phase 3 IPF omitting ProspectiveFalsifierGate",
            SynthesisClaimRecord(
                "REC-D5-08",
                "CANDIDATE_UPGRADE",
                studies["R102"],
                ("R104:O2",),
                (),
                ("In R104:O2 pirfenidone maximal fibroblast inhibition exceeded 90%.",),
                (prop("R104:O2", "P-D5-08"),),
                prospective_gate=None,
            ),
            None,
        ),
        (
            "TB-D5-09",
            "IPF",
            "Curation ledger tamper: unrecorded edit reverting R104:O1 settled corpus unit from 'uM' back to 'um'",
            SynthesisClaimRecord(
                "REC-D5-09",
                "SUPPORTS",
                studies["R104"],
                ("R104:O1",),
                (),
                ("In R104:O1 pirfenidone inhibited fibroblast MRTF activation with a half-maximal inhibitory concentration of 50-150 uM.",),
                (prop("R104:O1", "P-D5-09"),),
                requests_direct_quantitative_promotion=False,
            ),
            "tamper_settled_unit_um",
        ),
        (
            "TB-D5-10",
            "ONC",
            "Atom-to-corpus drift: the same unrecorded edit (ADAURA R303:O1 focal value 90 to 92) written into both D_0 and D_m; replay passes, binding refuses",
            SynthesisClaimRecord(
                "REC-D5-10",
                "SUPPORTS",
                atoms["R303:O1"].context,
                ("R303:O1",),
                (),
                ("In R303:O1 24-month disease-free survival was 90% in the osimertinib group versus 44% in the placebo group.",),
                (prop("R303:O1", "P-D5-10"),),
                requests_direct_quantitative_promotion=True,
            ),
            "tamper_atom_corpus_drift",
        ),
    ]
    for cid, dom, desc, rec, t_kind in d5_specs:
        cases.append(GroundedSynthesisCase(cid, dom, "D5_preclinical_to_clinical_conflation", True, desc, rec, t_kind))

    # D6: Cross-clause numeric theft (>2-cycles & 'although'), negated inequality bounds, and Unicode Mn/Me evasion (10 cases)
    d6_specs: list[tuple[Any, ...]] = [
        (
            "TB-D6-01",
            "HF",
            "3-cycle cross-clause numeric rotation across R201:O1 -> R201:O2 -> R201:O3 -> R201:O1",
            SynthesisClaimRecord(
                "REC-D6-01",
                "SUPPORTS",
                studies["R201"],
                ("R201:O1", "R201:O2", "R201:O3"),
                (),
                (
                    "In R201:O1 primary composite incidence was 17.0% versus 19.8%; in R201:O2 all-cause mortality was 13.3% versus 16.5%; in R201:O3 cardiovascular death was 21.8% versus 26.5%.",
                ),
                (prop("R201:O1", "P-D6-01A"), prop("R201:O2", "P-D6-01B"), prop("R201:O3", "P-D6-01C")),
                requests_direct_quantitative_promotion=True,
            ),
        ),
        (
            "TB-D6-02",
            "HF",
            "One-way cross-clause theft copying PARADIGM-HF 21.8% into VICTORIA R203:O1 while R201:O1 keeps 21.8%",
            SynthesisClaimRecord(
                "REC-D6-02",
                "SUPPORTS",
                studies["R203"],
                ("R201:O1", "R203:O1"),
                (),
                (
                    "In R201:O1 primary events were 21.8% in the LCZ696 group versus 26.5% in the enalapril group; in R203:O1 primary events were 21.8% in the vericiguat group versus 38.5% in the placebo group.",
                ),
                (prop("R201:O1", "P-D6-02A"), prop("R203:O1", "P-D6-02B")),
            ),
        ),
        (
            "TB-D6-03",
            "ONC",
            "Mid-sentence 'although' subordinator cross-clause theft stealing KEYNOTE-024 10.3 months into AURA3 R302:O2",
            SynthesisClaimRecord(
                "REC-D6-03",
                "SUPPORTS",
                studies["R301"],
                ("R301:O1", "R302:O2"),
                (),
                (
                    "In R302:O2 CNS progression-free survival was 10.3 months with osimertinib, although in R301:O1 median progression-free survival was 10.3 months in the pembrolizumab group versus 6.0 months in the chemotherapy group.",
                ),
                (prop("R301:O1", "P-D6-03A"), prop("R302:O2", "P-D6-03B")),
            ),
        ),
        (
            "TB-D6-04",
            "IPF",
            "Mid-sentence 'even though' subordinator theft stealing INPULSIS-1 61.5% diarrhea rate into ASCEND R102:O1",
            SynthesisClaimRecord(
                "REC-D6-04",
                "SCOPING_QUALIFICATION",
                studies["R101"],
                ("R102:O1",),
                ("R101:O3",),
                (
                    "In R102:O1 pirfenidone achieved a 61.5% relative reduction, even though in R101:O3 diarrhea occurred in 61.5% of the nintedanib group versus 18.6% of the placebo group.",
                ),
                (),
            ),
        ),
        (
            "TB-D6-05",
            "IPF",
            "Symbolic negated inequality ('not < 5%' -> '>= 5%') contradicting INPULSIS-1 '< 5%' discontinuation bound (R101:O4)",
            SynthesisClaimRecord(
                "REC-D6-05",
                "SCOPING_QUALIFICATION",
                studies["R101"],
                (),
                ("R101:O4",),
                ("In R101:O4 diarrhea-associated medication discontinuation was not < 5%.",),
                (),
            ),
        ),
        (
            "TB-D6-06",
            "IPF",
            "Verbal 'never less than 5%' (-> '>= 5%') contradicting INPULSIS-1 '< 5%' discontinuation bound (R101:O4)",
            SynthesisClaimRecord(
                "REC-D6-06",
                "SCOPING_QUALIFICATION",
                studies["R101"],
                (),
                ("R101:O4",),
                ("In R101:O4 diarrhea-associated medication discontinuation was never less than 5%.",),
                (),
            ),
        ),
        (
            "TB-D6-07",
            "IPF",
            "Symbolic 'never > 90%' (-> '<= 90%') contradicting pirfenidone '> 90%' maximal MRTF inhibition (R104:O2)",
            SynthesisClaimRecord(
                "REC-D6-07",
                "SUPPORTS",
                studies["R104"],
                ("R104:O2",),
                (),
                ("In R104:O2 pirfenidone maximal fibroblast inhibition was never > 90%.",),
                (prop("R104:O2", "P-D6-07"),),
            ),
        ),
        (
            "TB-D6-08",
            "ONC",
            "Unicode enclosing mark (U+20DD, category Me) splitting '4\u20dd4%' to evade numeric extraction on ADAURA R303:O1",
            SynthesisClaimRecord(
                "REC-D6-08",
                "SUPPORTS",
                atoms["R303:O1"].context,
                ("R303:O1",),
                (),
                ("In R303:O1 24-month disease-free survival was 90% in the osimertinib group versus 4\u20dd4% in the placebo group.",),
                (prop("R303:O1", "P-D6-08"),),
            ),
        ),
        (
            "TB-D6-09",
            "ONC",
            "Unicode nonspacing mark (U+0301, category Mn) between digits ('10\u0301.1 months') on AURA3 R302:O1",
            SynthesisClaimRecord(
                "REC-D6-09",
                "SUPPORTS",
                studies["R302"],
                ("R302:O1",),
                (),
                ("In R302:O1 median progression-free survival was 10\u0301.1 months with osimertinib versus 4.4 months with platinum therapy plus pemetrexed.",),
                (prop("R302:O1", "P-D6-09"),),
            ),
        ),
        (
            "TB-D6-10",
            "HF",
            "False '95% relative reduction' written with a fullwidth digit (U+FF15) to slip past numeral reading in R203:O1",
            SynthesisClaimRecord(
                "REC-D6-10",
                "SUPPORTS",
                studies["R203"],
                ("R203:O1",),
                (),
                ("In R203:O1 primary-outcome events occurred in 35.5% of the vericiguat group versus 38.5% of the placebo group, a 9\uff15% relative reduction.",),
                (prop("R203:O1", "P-D6-10"),),
            ),
            ("In R203:O1 primary-outcome events occurred in 35.5% of the vericiguat group versus 38.5% of the placebo group.",),
        ),
    ]
    for spec in d6_specs:
        cid, dom, desc, rec = spec[:4]
        cases.append(
            GroundedSynthesisCase(
                cid, dom, "D6_cross_clause_theft_and_negated_bounds", True, desc, rec, None, spec[4] if len(spec) > 4 else None
            )
        )

    return cases


# ---------------------------------------------------------------------------
# 4. Five-regime evaluation (V0, V1, V2, V2s, V3) on Track A and Track B
# ---------------------------------------------------------------------------

def _v0_sentence(sentence: str, atoms: Mapping[str, EvidenceAtom]) -> list[str]:
    cited = TUPLE_RE.findall(sentence)
    if not cited:
        return ["V0_NO_CITATION"]
    bad = [t for t in cited if t not in atoms]
    return [f"V0_UNKNOWN:{bad}"] if bad else []


def _v1_sentence(sentence: str, atoms: Mapping[str, EvidenceAtom]) -> list[str]:
    errs = _v0_sentence(sentence, atoms)
    if errs:
        return errs
    cited = TUPLE_RE.findall(sentence)
    cited_studies = {t.split(":")[0] for t in cited}
    pool: set[str] = set()
    for a in atoms.values():
        if a.study_id in cited_studies:
            pool |= atom_numeric_pool(a)
    masked = MASK_ID_RE.sub(lambda m: " " * len(m.group(0)), sentence)
    ungrounded = [
        normalize_numeric_literal(m.group(0))
        for m in ASCII_NUM_RE.finditer(masked)
        if normalize_numeric_literal(m.group(0)) not in pool
    ]
    return [f"V1_UNGROUNDED:{ungrounded}"] if ungrounded else []


def _classify_track_a_refusal(violations: Sequence[str]) -> str:
    if not violations:
        return "accepted"
    head = violations[0].split(":")[0].split(" ")[0]
    return head


def evaluate_track_a(
    graph: ContextEvidenceGraph,
    cases: Sequence[VerbatimSentenceCase],
) -> dict[str, Any]:
    """Evaluate all 36 unmodified PubMed sentences both unanchored and source-anchored."""
    verifiers = (
        "V0_citation_presence",
        "V1_study_pool_numeric",
        "V2_prose_pooled_permitted",
        "V2s_prose_pooled_rejected",
        "V3_full_ceg",
    )
    unanchored_accepted = {v: 0 for v in verifiers}
    anchored_accepted = {v: 0 for v in verifiers}
    anchored_by_domain: dict[str, dict[str, int]] = {
        dom: {"n": 0, "V0_accepted": 0, "V1_accepted": 0, "V2_accepted": 0, "V2s_accepted": 0, "V3_accepted": 0}
        for dom in ("IPF", "HF", "ONC")
    }
    v3_refusal_codes: dict[str, int] = {}
    phenomenon_counts: dict[str, dict[str, int]] = {}
    per_sentence: list[dict[str, Any]] = []

    for c in cases:
        # 1. Raw unanchored sentence (exact published sentence with no [Rxxx:Oy] anchor)
        u_v0 = not _v0_sentence(c.verbatim_sentence, graph.atoms)
        u_v1 = not _v1_sentence(c.verbatim_sentence, graph.atoms)
        u_v2 = not verify_prose_sentence(c.verbatim_sentence, graph.atoms, graph.arm_aliases, pooled_numeral_policy="permit")
        u_v2s = not verify_prose_sentence(c.verbatim_sentence, graph.atoms, graph.arm_aliases, pooled_numeral_policy="reject")
        u_v3 = u_v2s
        for k, ok in zip(verifiers, (u_v0, u_v1, u_v2, u_v2s, u_v3)):
            unanchored_accepted[k] += int(ok)

        # 2. Source-anchored verbatim sentence (appending true tuple anchor tag only)
        anchor_tag = "[" + ", ".join(c.true_tuples) + "]"
        anchored_sent = f"{c.verbatim_sentence} {anchor_tag}"
        a_v0_errs = _v0_sentence(anchored_sent, graph.atoms)
        a_v1_errs = _v1_sentence(anchored_sent, graph.atoms)
        a_v2_errs = verify_prose_sentence(anchored_sent, graph.atoms, graph.arm_aliases, pooled_numeral_policy="permit")
        a_v2s_errs = verify_prose_sentence(anchored_sent, graph.atoms, graph.arm_aliases, pooled_numeral_policy="reject")
        rec = SynthesisClaimRecord(
            record_id=f"REC-{c.sentence_id}",
            polarity="SCOPING_QUALIFICATION",
            target_context=graph.studies[c.study_id],
            supporting_tuples=c.true_tuples,
            counterevidence_tuples=(),
            prose_sentences=(anchored_sent,),
            typed_propositions=(),
        )
        rep_v3 = graph.verify_synthesis_record(rec, pooled_numeral_policy="reject")

        verdicts = (
            len(a_v0_errs) == 0,
            len(a_v1_errs) == 0,
            len(a_v2_errs) == 0,
            len(a_v2s_errs) == 0,
            rep_v3.passed_all_invariants,
        )
        for k, ok in zip(verifiers, verdicts):
            anchored_accepted[k] += int(ok)

        dom_row = anchored_by_domain[c.domain]
        dom_row["n"] += 1
        dom_row["V0_accepted"] += int(verdicts[0])
        dom_row["V1_accepted"] += int(verdicts[1])
        dom_row["V2_accepted"] += int(verdicts[2])
        dom_row["V2s_accepted"] += int(verdicts[3])
        dom_row["V3_accepted"] += int(verdicts[4])

        primary_code = _classify_track_a_refusal(rep_v3.violations)
        v3_refusal_codes[primary_code] = v3_refusal_codes.get(primary_code, 0) + 1
        ph = phenomenon_counts.setdefault(c.primary_phenomenon, {"total": 0, "v3_accepted": 0, "v3_refused": 0})
        ph["total"] += 1
        ph["v3_accepted" if verdicts[4] else "v3_refused"] += 1

        per_sentence.append(
            {
                "sentence_id": c.sentence_id,
                "domain": c.domain,
                "study_id": c.study_id,
                "pmid": c.pmid,
                "true_tuples": list(c.true_tuples),
                "primary_phenomenon": c.primary_phenomenon,
                "anchored_verdicts": dict(zip(verifiers, verdicts)),
                "v3_primary_code": primary_code,
                "v3_violations": list(rep_v3.violations),
            }
        )

    total = len(cases)
    return {
        "total_verbatim_sentences": total,
        "unanchored_summary": {
            v: {
                "accepted": unanchored_accepted[v],
                "refused": total - unanchored_accepted[v],
                "accepted_pct": round(100.0 * unanchored_accepted[v] / total, 2),
            }
            for v in verifiers
        },
        "source_anchored_summary": {
            v: {
                "accepted": anchored_accepted[v],
                "refused": total - anchored_accepted[v],
                "accepted_pct": round(100.0 * anchored_accepted[v] / total, 2),
                "refused_pct": round(100.0 * (total - anchored_accepted[v]) / total, 2),
            }
            for v in verifiers
        },
        "source_anchored_by_domain": anchored_by_domain,
        "v3_primary_outcome_codes": dict(sorted(v3_refusal_codes.items())),
        "phenomenon_breakdown": dict(sorted(phenomenon_counts.items())),
        "per_sentence": per_sentence,
    }


def _eval_track_b_case(
    c: GroundedSynthesisCase,
    graph: ContextEvidenceGraph,
    initial_corpus: Mapping[str, Any],
    settled_corpus: Mapping[str, Any],
    patch_events: Sequence[ReversiblePatchEvent],
    expected_ledger_digest: str,
) -> tuple[dict[str, bool], list[str]]:
    """Verdict of each regime on one case, and every reason the full verifier gave for refusing it."""
    rec = c.record
    anchors = set(rec.supporting_tuples) | set(rec.counterevidence_tuples)
    # V0
    v0_ok = bool(anchors) and all(t in graph.atoms for t in anchors)
    if v0_ok:
        for s in rec.prose_sentences:
            cited = TUPLE_RE.findall(s)
            if not cited or any(t not in anchors or t not in graph.atoms for t in cited):
                v0_ok = False
                break
    # V1
    v1_ok = v0_ok
    if v1_ok:
        for s in rec.prose_sentences:
            if _v1_sentence(s, graph.atoms):
                v1_ok = False
                break
    # V2 / V2s
    def prose_regime(policy: PooledPolicy) -> bool:
        if not v0_ok:
            return False
        for s in rec.prose_sentences:
            if verify_prose_sentence(s, graph.atoms, graph.arm_aliases, pooled_numeral_policy=policy):
                return False
        return True

    v2_ok = prose_regime("permit")
    v2s_ok = prose_regime("reject")

    # V3 full CEG: graph integrity, I2 replay, atom-to-corpus binding and the record, all evaluated.
    loc_initial = copy.deepcopy(dict(initial_corpus))
    loc_settled = copy.deepcopy(dict(settled_corpus))
    if c.tamper_kind == "tamper_settled_unit_um":
        # One-sided edit of the settled corpus only: replay no longer reaches it.
        loc_settled["studies"][3]["outcomes"][0]["unit"] = "um"
    elif c.tamper_kind == "tamper_atom_corpus_drift":
        # The same edit on both sides, in a study that no ledger event touches (R303): replay
        # passes with every step digest intact, and only the atom-to-corpus binding can see it.
        for doc in (loc_initial, loc_settled):
            if doc["studies"][10]["study_id"] != "R303" or doc["studies"][10]["outcomes"][0]["focal_value"] != "90":
                raise RuntimeError("tamper_atom_corpus_drift expects R303:O1 with focal value 90 at /studies/10/outcomes/0")
            doc["studies"][10]["outcomes"][0]["focal_value"] = "92"
    elif c.tamper_kind is not None:
        raise RuntimeError(f"unknown tamper_kind {c.tamper_kind!r}")

    reasons: list[str] = list(graph.verify_graph_integrity())
    try:
        receipt = verify_bijective_patch_replay(loc_initial, loc_settled, patch_events)
        if receipt["ledger_sha256"] != expected_ledger_digest:
            reasons.append("LEDGER_DIGEST_MISMATCH")
    except ValueError as exc:
        reasons.append(str(exc))
    reasons.extend(verify_atoms_bound_to_corpus(graph, loc_settled))
    rep = graph.verify_synthesis_record(rec, pooled_numeral_policy="reject")
    reasons.extend(rep.violations)

    return (
        {
            "V0_citation_presence": v0_ok,
            "V1_study_pool_numeric": v1_ok,
            "V2_prose_pooled_permitted": v2_ok,
            "V2s_prose_pooled_rejected": v2s_ok,
            "V3_full_ceg": not reasons,
        },
        reasons,
    )


def evaluate_track_b(
    graph: ContextEvidenceGraph,
    initial_corpus: Mapping[str, Any],
    settled_corpus: Mapping[str, Any],
    patch_events: Sequence[ReversiblePatchEvent],
    cases: Sequence[GroundedSynthesisCase],
) -> dict[str, Any]:
    verifiers = (
        "V0_citation_presence",
        "V1_study_pool_numeric",
        "V2_prose_pooled_permitted",
        "V2s_prose_pooled_rejected",
        "V3_full_ceg",
    )
    expected_digest = ledger_sha256(patch_events)
    evaluated = {
        c.case_id: _eval_track_b_case(c, graph, initial_corpus, settled_corpus, patch_events, expected_digest)
        for c in cases
    }
    verdicts_by_case = {cid: verdicts for cid, (verdicts, _reasons) in evaluated.items()}
    reasons_by_case = {cid: reasons for cid, (_verdicts, reasons) in evaluated.items()}

    def _code(reason: str) -> str:
        return reason.split(":")[0].split(" ")[0]

    # A repaired twin (the same record with its false sentence corrected) must be accepted.
    repaired_twins: dict[str, bool] = {}
    for c in cases:
        if c.repaired_prose is None:
            continue
        twin = GroundedSynthesisCase(
            c.case_id, c.domain, c.family, False, c.description,
            replace(c.record, prose_sentences=c.repaired_prose), c.tamper_kind,
        )
        twin_verdicts, twin_reasons = _eval_track_b_case(
            twin, graph, initial_corpus, settled_corpus, patch_events, expected_digest
        )
        if not twin_verdicts["V3_full_ceg"]:
            raise RuntimeError(f"repaired twin of {c.case_id} is refused: {twin_reasons}")
        repaired_twins[c.case_id] = True

    adv_cases = [c for c in cases if c.is_adversarial]
    val_cases = [c for c in cases if not c.is_adversarial]

    overall: dict[str, Any] = {}
    for v in verifiers:
        tp = sum(1 for c in adv_cases if not verdicts_by_case[c.case_id][v])
        tn = sum(1 for c in val_cases if verdicts_by_case[c.case_id][v])
        overall[v] = {
            "adversarial_total": len(adv_cases),
            "adversarial_rejected": tp,
            "adversarial_accepted": len(adv_cases) - tp,
            "adversarial_rejection_rate_pct": round(100.0 * tp / len(adv_cases), 2),
            "valid_total": len(val_cases),
            "valid_accepted": tn,
            "valid_rejected": len(val_cases) - tn,
            "valid_acceptance_rate_pct": round(100.0 * tn / len(val_cases), 2),
        }

    families = sorted({c.family for c in cases})
    by_family: dict[str, Any] = {}
    for fam in families:
        fcases = [c for c in cases if c.family == fam]
        is_adv = fcases[0].is_adversarial
        by_v: dict[str, Any] = {}
        for v in verifiers:
            rej = sum(1 for c in fcases if not verdicts_by_case[c.case_id][v])
            acc = len(fcases) - rej
            by_v[v] = {
                "accepted": acc,
                "rejected": rej,
                "rejection_rate_pct": round(100.0 * rej / len(fcases), 2),
                "acceptance_rate_pct": round(100.0 * acc / len(fcases), 2),
            }
        by_family[fam] = {
            "n": len(fcases),
            "is_adversarial": is_adv,
            "by_verifier": by_v,
        }

    by_domain: dict[str, Any] = {}
    for dom in ("IPF", "HF", "ONC"):
        d_adv = [c for c in adv_cases if c.domain == dom]
        d_val = [c for c in val_cases if c.domain == dom]
        by_domain[dom] = {
            "valid_total": len(d_val),
            "adversarial_total": len(d_adv),
            "v3_valid_accepted": sum(1 for c in d_val if verdicts_by_case[c.case_id]["V3_full_ceg"]),
            "v3_adversarial_rejected": sum(1 for c in d_adv if not verdicts_by_case[c.case_id]["V3_full_ceg"]),
        }

    # Mode B (self-attesting adversary): attach a freshly computed, valid attestation to each
    # adversarial record and require that the record is still refused on its content. The two
    # corpus tampers are excluded: their defect is in the corpus, their record is valid, and an
    # attestation has nothing to say about them. They are reported separately below.
    record_level_adv = [c for c in adv_cases if c.tamper_kind is None]
    corpus_tamper_adv = [c for c in adv_cases if c.tamper_kind is not None]
    mode_b_semantic_rejected = 0
    for c in record_level_adv:
        re_attested = populate_claim_attestation(c.record, graph.atoms, graph.arm_aliases)
        rep = graph.verify_synthesis_record(re_attested, pooled_numeral_policy="reject")
        non_stale_violations = [v for v in rep.violations if not v.startswith("I6_STALE_CLAIM_ATTESTATION")]
        if not rep.passed_all_invariants and non_stale_violations:
            mode_b_semantic_rejected += 1
    corpus_tampers: dict[str, Any] = {}
    for c in corpus_tamper_adv:
        record_alone = graph.verify_synthesis_record(c.record, pooled_numeral_policy="reject")
        corpus_tampers[c.case_id] = {
            "tamper_kind": c.tamper_kind,
            "record_alone_passes": record_alone.passed_all_invariants,
            "rejected_by_full_verifier": not verdicts_by_case[c.case_id]["V3_full_ceg"],
            "codes": sorted({_code(r) for r in reasons_by_case[c.case_id]}),
        }

    # Records whose only defect is what the record declares about itself: with evidence and prose
    # unchanged, the record is accepted once its promotion request alone, or its polarity label
    # alone, is changed. For these records the verifier checks a declaration, not a sentence.
    promotion_request_only: list[str] = []
    polarity_label_only: list[str] = []
    for c in record_level_adv:
        flipped = replace(c.record, requests_direct_quantitative_promotion=not c.record.requests_direct_quantitative_promotion)
        if graph.verify_synthesis_record(flipped, pooled_numeral_policy="reject").passed_all_invariants:
            promotion_request_only.append(c.case_id)
        for polarity in ("SUPPORTS", "CONTRADICTS", "SCOPING_QUALIFICATION", "CANDIDATE_UPGRADE", "FALSIFIED_GRAVEYARD"):
            if polarity == c.record.polarity:
                continue
            if graph.verify_synthesis_record(replace(c.record, polarity=polarity), pooled_numeral_policy="reject").passed_all_invariants:
                polarity_label_only.append(c.case_id)
                break

    mode_b_valid_accepted = 0
    for c in val_cases:
        attested_val = populate_claim_attestation(c.record, graph.atoms, graph.arm_aliases)
        if graph.verify_synthesis_record(attested_val, pooled_numeral_policy="reject").passed_all_invariants:
            mode_b_valid_accepted += 1

    return {
        "total_cases": len(cases),
        "valid_cases": len(val_cases),
        "adversarial_cases": len(adv_cases),
        "overall_summary": overall,
        "mode_b_self_attesting_summary": {
            "adversarial_total": len(record_level_adv),
            "semantic_rejected": mode_b_semantic_rejected,
            "semantic_rejection_rate_pct": round(100.0 * mode_b_semantic_rejected / len(record_level_adv), 2),
            "valid_total": len(val_cases),
            "valid_accepted": mode_b_valid_accepted,
            "valid_acceptance_rate_pct": round(100.0 * mode_b_valid_accepted / len(val_cases), 2),
        },
        "corpus_tampers": corpus_tampers,
        "repaired_twins_accepted": repaired_twins,
        "declaration_only_defects": {
            "record_level_adversarial_total": len(record_level_adv),
            "accepted_after_promotion_request_change_only": promotion_request_only,
            "accepted_after_polarity_label_change_only": polarity_label_only,
            "either": len(set(promotion_request_only) | set(polarity_label_only)),
        },
        "valid_promotions": sum(
            1 for c in val_cases if c.record.requests_direct_quantitative_promotion is True and verdicts_by_case[c.case_id]["V3_full_ceg"]
        ),
        "by_family": by_family,
        "by_domain": by_domain,
        "per_case": [
            {
                "case_id": c.case_id,
                "family": c.family,
                "is_adversarial": c.is_adversarial,
                "tamper_kind": c.tamper_kind,
                "verdicts": verdicts_by_case[c.case_id],
                "v3_codes": sorted({_code(r) for r in reasons_by_case[c.case_id]}),
            }
            for c in cases
        ],
    }


def run_real_text_evaluation() -> dict[str, Any]:
    graph, initial_corpus, settled_corpus, patch_events = build_real_text_corpus()
    integrity_errors = graph.verify_graph_integrity()
    if integrity_errors:
        raise RuntimeError(f"Real-text graph integrity failed: {integrity_errors}")
    replay_receipt = verify_bijective_patch_replay(initial_corpus, settled_corpus, patch_events)
    binding_errors = verify_atoms_bound_to_corpus(graph, settled_corpus)
    if binding_errors:
        raise RuntimeError(f"Real-text corpus binding failed: {binding_errors}")

    # Every value an atom states must be readable in its own verbatim quote.
    quotes_missing_a_value = []
    for tup, atom in sorted(graph.atoms.items()):
        literals = [
            lit
            for q in (atom.focal_quantity, atom.comparator_quantity)
            if q is not None
            for lit in (q.value, q.lower, q.upper)
            if lit is not None
        ]
        if any(lit not in atom.provenance.verbatim_quote and lit.lstrip("-") not in atom.provenance.verbatim_quote for lit in literals):
            quotes_missing_a_value.append(tup)
    if quotes_missing_a_value:
        raise RuntimeError(f"Verbatim quote does not contain every atom value: {quotes_missing_a_value}")

    track_a_cases = build_track_a_verbatim_cases()
    track_b_cases = build_track_b_cases(graph)

    track_a_res = evaluate_track_a(graph, track_a_cases)
    track_b_res = evaluate_track_b(graph, initial_corpus, settled_corpus, patch_events, track_b_cases)

    return {
        "evaluation_version": REAL_TEXT_EVAL_VERSION,
        "reference_version": REFERENCE_VERSION,
        "corpus_summary": {
            "studies": len(graph.studies),
            "domains": {"IPF": 4, "HF": 4, "ONC": 4},
            "clinical_rcts": 9,
            "preclinical_studies": 3,
            "atoms": len(graph.atoms),
            "quantity_kinds": {
                k: sum(1 for a in graph.atoms.values() if a.focal_quantity.kind == k)
                for k in ("point_level", "relative_change", "interval_bound", "censored_observation")
            },
            "atoms_flagged_primary_source_verified": sum(
                1 for a in graph.atoms.values() if a.provenance.primary_pdf_verified
            ),
            "quotes_containing_every_atom_value": len(graph.atoms) - len(quotes_missing_a_value),
            "graph_integrity_errors": len(integrity_errors),
            "atom_corpus_binding_errors": len(binding_errors),
            "curation_ledger": replay_receipt,
            "study_metadata": STUDY_METADATA,
        },
        "track_a_verbatim_prose": track_a_res,
        "track_b_grounded_synthesis": track_b_res,
    }


def main() -> None:
    out = run_real_text_evaluation()
    out_path = Path(__file__).resolve().parent / "real_text_results.json"
    out_path.write_text(json.dumps(out, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    ta = out["track_a_verbatim_prose"]["source_anchored_summary"]["V3_full_ceg"]
    tb = out["track_b_grounded_synthesis"]["overall_summary"]["V3_full_ceg"]
    print(
        f"Real-text evaluation v{out['evaluation_version']} (ceg_reference v{out['reference_version']}): "
        f"studies={out['corpus_summary']['studies']}, atoms={out['corpus_summary']['atoms']}"
    )
    print(
        f"  Track A (source-anchored verbatim sentences, n={out['track_a_verbatim_prose']['total_verbatim_sentences']}): "
        f"V3 accepted={ta['accepted']} ({ta['accepted_pct']}%), refused={ta['refused']} ({ta['refused_pct']}%)"
    )
    print(
        f"  Track B (grounded synthesis, n={out['track_b_grounded_synthesis']['total_cases']}): "
        f"V3 valid accepted={tb['valid_accepted']}/{tb['valid_total']} ({tb['valid_acceptance_rate_pct']}%), "
        f"adversarial rejected={tb['adversarial_rejected']}/{tb['adversarial_total']} ({tb['adversarial_rejection_rate_pct']}%)"
    )


if __name__ == "__main__":
    main()
