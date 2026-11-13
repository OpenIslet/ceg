#!/usr/bin/env python3
"""Worked domain vignettes for Context-Evidence Graph (CEG) evaluation.

Provides two synthetic, deterministic worked case studies and one built from real published studies:
  1. `drug_development` (default): Translational antifibrotic small-molecule
     development (`NX-410`) across immortalized 2D cell lines, bleomycin
     murine models, explanted human precision-cut lung slices (PCLS), and
     Phase Ib/IIa clinical cohorts.
  2. `sport_science` (swap-pack): Ergogenic ketogenic/carbohydrate periodization
     across C2C12 myotube metabolic assays, recreational crossover trials, and
     elite heat-acclimated endurance cohorts.
  3. `t1d_islet_replacement`: eight peer-reviewed type 1 diabetes studies (ten
     findings) taken from the OpenIslet corpus, used as the worked example of
     Section 5 of the manuscript. The synthetic benchmark does not use it.

The two synthetic vignettes instantiate:
  - A non-linear product context poset (C, <=_C) with incomparable translational
    branches (demonstrating why scalar evidence pyramids lose partial-order
    information).
  - A bijective curation patch ledger (D_0 -> D_k -> D_0) verified under I2.
  - Multi-arm outcomes, relative vs. absolute quantity types, undigitized figure
    outcomes, counterevidence outcomes, and canonical synthesis records spanning
    PROMOTE_QUANTITATIVE_PARAMETER, HOLD_FOR_PROSPECTIVE_FALSIFIER,
    QUALITATIVE_SCOPING_ONLY, and REJECT_INADMISSIBLE_OR_INVALID.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any

from ceg_reference import (
    CEGVerificationReport,
    ContextAxisPoset,
    ContextEvidenceGraph,
    ContextLatticeSpec,
    ContextSignature,
    EvidenceAtom,
    MultiArmContrast,
    ProspectiveFalsifierGate,
    ProvenancePin,
    QuantitySpec,
    ReversiblePatchEvent,
    SynthesisClaimRecord,
    TypedProposition,
    corpus_outcome_from_atom,
    populate_claim_attestation,
    proposition_from_atom,
    render_atom_sentence,
    settled_corpus_from_graph,
    sha256_canonical,
    verify_atoms_bound_to_corpus,
    verify_bijective_patch_replay,
)


@dataclass
class DomainVignetteBundle:
    """Self-contained worked case study bundle."""

    domain_name: str
    target_context: ContextSignature
    graph: ContextEvidenceGraph
    initial_corpus: dict[str, Any]
    settled_corpus: dict[str, Any]
    patch_events: list[ReversiblePatchEvent]
    valid_records: list[SynthesisClaimRecord]
    invalid_probe_records: list[tuple[str, SynthesisClaimRecord, str]]

    def run_all_checks(self) -> dict[str, Any]:
        graph_errs = self.graph.verify_graph_integrity()
        if graph_errs:
            raise AssertionError(f"VIGNETTE_GRAPH_INTEGRITY_FAILED ({self.domain_name}): {graph_errs}")

        replay_res = verify_bijective_patch_replay(
            self.initial_corpus,
            self.settled_corpus,
            self.patch_events,
        )
        binding_errs = verify_atoms_bound_to_corpus(self.graph, self.settled_corpus)
        if binding_errs:
            raise AssertionError(f"VIGNETTE_ATOM_CORPUS_BINDING_FAILED ({self.domain_name}): {binding_errs}")

        valid_reports: dict[str, dict[str, Any]] = {}
        for rec in self.valid_records:
            rep: CEGVerificationReport = self.graph.verify_synthesis_record(rec)
            if not rep.passed_all_invariants:
                raise AssertionError(
                    f"EXPECTED_VALID_RECORD_FAILED ({self.domain_name}:{rec.record_id}): {rep.violations}"
                )
            valid_reports[rec.record_id] = {
                "polarity": rec.polarity,
                "transfer_classes": rep.transfer_classes,
                "unresolved_axes": list(rep.unresolved_axes_union),
                "promotion_decision": rep.promotion_decision,
            }

        rejected_probes: dict[str, dict[str, Any]] = {}
        for label, bad_rec, expected_code_prefix in self.invalid_probe_records:
            rep_bad = self.graph.verify_synthesis_record(bad_rec)
            if rep_bad.passed_all_invariants:
                raise AssertionError(
                    f"EXPECTED_INVALID_PROBE_ACCEPTED ({self.domain_name}:{label})"
                )
            if not any(expected_code_prefix in v for v in rep_bad.violations):
                raise AssertionError(
                    f"UNEXPECTED_VIOLATION_CODE ({self.domain_name}:{label}): "
                    f"expected {expected_code_prefix!r} in {rep_bad.violations}"
                )
            rejected_probes[label] = {
                "expected_code_prefix": expected_code_prefix,
                "violations": rep_bad.violations,
            }

        return {
            "domain_name": self.domain_name,
            "studies_count": len(self.graph.studies),
            "variables_count": sum(len(v) for v in self.graph.study_variables.values()),
            "atoms_count": len(self.graph.atoms),
            "patch_replay": replay_res,
            "atoms_bound_to_settled_corpus": True,
            "valid_records_verified": valid_reports,
            "invalid_probes_rejected": rejected_probes,
        }


def _make_patch_event(
    work_doc: dict[str, Any],
    event_id: str,
    study_idx: int,
    pointer_suffix: str,
    new_value: Any,
    rationale: str,
) -> ReversiblePatchEvent:
    study_obj = work_doc["studies"][study_idx]
    study_id = study_obj["study_id"]
    before_sha = sha256_canonical(study_obj)
    full_ptr = f"/studies/{study_idx}{pointer_suffix}"

    # Navigate relative pointer on study_obj
    segs = pointer_suffix.lstrip("/").split("/")
    cur: Any = study_obj
    for seg in segs[:-1]:
        cur = cur[int(seg)] if isinstance(cur, list) else cur[seg]
    last = segs[-1]
    if isinstance(cur, list):
        old_val = copy.deepcopy(cur[int(last)])
        cur[int(last)] = copy.deepcopy(new_value)
    else:
        old_val = copy.deepcopy(cur[last])
        cur[last] = copy.deepcopy(new_value)

    after_sha = sha256_canonical(study_obj)
    return ReversiblePatchEvent(
        event_id=event_id,
        study_id=study_id,
        json_pointer=full_ptr,
        old_value=old_val,
        new_value=copy.deepcopy(new_value),
        rationale=rationale,
        study_sha256_before=before_sha,
        study_sha256_after=after_sha,
    )


def _corpora_and_ledger(
    graph: ContextEvidenceGraph,
    corpus_id: str,
    repairs: list[tuple[str, str, str, Any, str]],
    study_metadata: dict[str, dict[str, Any]] | None = None,
) -> tuple[dict[str, Any], dict[str, Any], list[ReversiblePatchEvent]]:
    """Complete corpora and ledger for a vignette.

    The settled corpus D_m is the complete projection of the graph: every corpus
    outcome record carries every atom field. Each repair is
    `(event_id, study_id, pointer_suffix, raw_value, rationale)`: the raw
    extraction D_0 is D_m with `raw_value` written at that pointer, and the
    ledger event restores the settled value.
    """
    settled = settled_corpus_from_graph(graph, corpus_id, study_metadata)
    position = {st["study_id"]: i for i, st in enumerate(settled["studies"])}

    def _locate(doc: dict[str, Any], study_id: str, suffix: str) -> tuple[Any, Any]:
        cur: Any = doc["studies"][position[study_id]]
        segs = suffix.lstrip("/").split("/")
        for seg in segs[:-1]:
            cur = cur[int(seg)] if isinstance(cur, list) else cur[seg]
        last: Any = int(segs[-1]) if isinstance(cur, list) else segs[-1]
        return cur, last

    initial = copy.deepcopy(settled)
    for _event_id, study_id, suffix, raw_value, _rationale in repairs:
        parent, key = _locate(initial, study_id, suffix)
        parent[key] = copy.deepcopy(raw_value)

    work_doc = copy.deepcopy(initial)
    events: list[ReversiblePatchEvent] = []
    for event_id, study_id, suffix, _raw_value, rationale in repairs:
        parent, key = _locate(settled, study_id, suffix)
        events.append(
            _make_patch_event(work_doc, event_id, position[study_id], suffix, copy.deepcopy(parent[key]), rationale)
        )
    if work_doc != settled:
        raise RuntimeError(f"VIGNETTE_PATCHES_DO_NOT_REACH_SETTLED_CORPUS ({corpus_id})")
    return initial, settled, events


def build_drug_development_vignette() -> DomainVignetteBundle:
    """Build the translational antifibrotic (`NX-410`) drug-development case study."""
    # 1. Define 4-axis product context poset with a non-linear diamond on `model_system`:
    #    immortalized_2d_line < primary_mammalian_floor < {murine_bleomycin_invivo || human_pcls_explant} < human_clinical_cohort
    #    plus an incompatible off-branch `dermal_keratinocyte_scratch` with no common upper bound.
    model_poset = ContextAxisPoset(
        name="model_system",
        dominates={
            "immortalized_2d_line": frozenset({"immortalized_2d_line"}),
            "primary_mammalian_floor": frozenset({"immortalized_2d_line", "primary_mammalian_floor"}),
            "murine_bleomycin_invivo": frozenset(
                {"immortalized_2d_line", "primary_mammalian_floor", "murine_bleomycin_invivo"}
            ),
            "human_pcls_explant": frozenset(
                {"immortalized_2d_line", "primary_mammalian_floor", "human_pcls_explant"}
            ),
            "human_clinical_cohort": frozenset(
                {
                    "immortalized_2d_line",
                    "primary_mammalian_floor",
                    "murine_bleomycin_invivo",
                    "human_pcls_explant",
                    "human_clinical_cohort",
                }
            ),
            "dermal_keratinocyte_scratch": frozenset({"dermal_keratinocyte_scratch"}),
        },
    )
    chronicity_poset = ContextAxisPoset(
        name="disease_chronicity",
        dominates={
            "acute_cytokine_pulse": frozenset({"acute_cytokine_pulse"}),
            "subacute_bleomycin_day21": frozenset({"acute_cytokine_pulse", "subacute_bleomycin_day21"}),
            "established_chronic_fibrosis": frozenset(
                {"acute_cytokine_pulse", "subacute_bleomycin_day21", "established_chronic_fibrosis"}
            ),
        },
    )
    exposure_poset = ContextAxisPoset(
        name="dosing_exposure",
        dominates={
            "supraphysiologic_bath_10uM": frozenset({"supraphysiologic_bath_10uM"}),
            "pk_matched_unbound_exposure": frozenset(
                {"supraphysiologic_bath_10uM", "pk_matched_unbound_exposure"}
            ),
            "oral_steady_state_bid": frozenset(
                {"supraphysiologic_bath_10uM", "pk_matched_unbound_exposure", "oral_steady_state_bid"}
            ),
        },
    )
    readout_poset = ContextAxisPoset(
        name="readout_maturity",
        dominates={
            "transcript_surrogate": frozenset({"transcript_surrogate"}),
            "extracellular_matrix_biomarker": frozenset(
                {"transcript_surrogate", "extracellular_matrix_biomarker"}
            ),
            "spirometric_clinical_endpoint": frozenset(
                {"transcript_surrogate", "extracellular_matrix_biomarker", "spirometric_clinical_endpoint"}
            ),
        },
    )

    lattice = ContextLatticeSpec(
        axis_posets={
            "model_system": model_poset,
            "disease_chronicity": chronicity_poset,
            "dosing_exposure": exposure_poset,
            "readout_maturity": readout_poset,
        },
        bridge_floor={
            "model_system": "primary_mammalian_floor",
            "disease_chronicity": "subacute_bleomycin_day21",
            "dosing_exposure": "pk_matched_unbound_exposure",
        },
    )

    target_ctx = ContextSignature(
        {
            "model_system": "human_clinical_cohort",
            "disease_chronicity": "established_chronic_fibrosis",
            "dosing_exposure": "oral_steady_state_bid",
            "readout_maturity": "spirometric_clinical_endpoint",
        }
    )


    # 3. Build study contexts, variable sets, and canonical EvidenceAtoms
    ctx_s101 = ContextSignature(
        {
            "model_system": "immortalized_2d_line",
            "disease_chronicity": "acute_cytokine_pulse",
            "dosing_exposure": "supraphysiologic_bath_10uM",
            "readout_maturity": "transcript_surrogate",
        }
    )
    ctx_s102 = ContextSignature(
        {
            "model_system": "murine_bleomycin_invivo",
            "disease_chronicity": "subacute_bleomycin_day21",
            "dosing_exposure": "pk_matched_unbound_exposure",
            "readout_maturity": "extracellular_matrix_biomarker",
        }
    )
    ctx_s103 = ContextSignature(
        {
            "model_system": "human_pcls_explant",
            "disease_chronicity": "established_chronic_fibrosis",
            "dosing_exposure": "pk_matched_unbound_exposure",
            "readout_maturity": "extracellular_matrix_biomarker",
        }
    )
    ctx_s104 = target_ctx
    ctx_s105 = ContextSignature(
        {
            "model_system": "human_clinical_cohort",
            "disease_chronicity": "established_chronic_fibrosis",
            "dosing_exposure": "oral_steady_state_bid",
            "readout_maturity": "extracellular_matrix_biomarker",
        }
    )

    studies = {
        "S101": ctx_s101,
        "S102": ctx_s102,
        "S103": ctx_s103,
        "S104": ctx_s104,
        "S105": ctx_s105,
    }

    study_variables = {
        "S101": {"VAR:S101:NX410_10uM", "VAR:S101:DMSO_Control"},
        "S102": {"VAR:S102:NX410_30mgkg", "VAR:S102:Pirfenidone_100mgkg", "VAR:S102:Vehicle"},
        "S103": {"VAR:S103:NX410_180nM_Unbound", "VAR:S103:Vehicle_Slice"},
        "S104": {"VAR:S104:NX410_40mg_BID", "VAR:S104:Placebo_BID"},
        "S105": {"VAR:S105:NX410_120mg_BID", "VAR:S105:Placebo_BID"},
    }

    arm_aliases = {
        "VAR:S101:NX410_10uM": ("nx-410 10 um", "nx-410", "10 um bath"),
        "VAR:S101:DMSO_Control": ("dmso control", "dmso"),
        "VAR:S102:NX410_30mgkg": ("nx-410", "nx-410 30 mg/kg"),
        "VAR:S102:Pirfenidone_100mgkg": ("pirfenidone",),
        "VAR:S102:Vehicle": ("vehicle",),
        "VAR:S103:NX410_180nM_Unbound": ("nx-410", "treated slices"),
        "VAR:S103:Vehicle_Slice": ("vehicle slices", "vehicle"),
        "VAR:S104:NX410_40mg_BID": ("nx-410 40 mg bid", "nx-410 arm", "nx-410", "treated cohort"),
        "VAR:S104:Placebo_BID": ("placebo", "placebo arm"),
        "VAR:S105:NX410_120mg_BID": ("nx-410 120 mg bid", "nx-410", "120 mg bid"),
        "VAR:S105:Placebo_BID": ("placebo",),
    }

    atoms = {
        "S101:O1": EvidenceAtom(
            atom_id="ATOM:S101:O1",
            outcome_tuple="S101:O1",
            study_id="S101",
            endpoint="COL1A1 transcript relative expression",
            context=ctx_s101,
            contrast=MultiArmContrast(
                focal_arms=("VAR:S101:NX410_10uM",),
                comparator_arms=("VAR:S101:DMSO_Control",),
                contrast_kind="pairwise",
            ),
            focal_quantity=QuantitySpec(
                kind="relative_change",
                relation="=",
                value="72.0",
                unit="%",
                unit_dimension="dimensionless_percentage",
                change_direction="decrease",
            ),
            comparator_quantity=None,
            statistic_kind="fold_change",
            sample_size=6,
            p_value_or_interval="p=0.004",
            polarity="favorable",
            provenance=ProvenancePin(
                document_id="SYNTHETIC-DOI:S101",
                section_pointer="Results p.4, Table 1",
                verbatim_quote="10 uM NX-410 reduced TGF-beta1-stimulated COL1A1 mRNA by 72.0% vs DMSO (n=6, p=0.004).",
                requires_figure_digitization=False,
                primary_pdf_verified=True,
            ),
        ),
        "S102:O1": EvidenceAtom(
            atom_id="ATOM:S102:O1",
            outcome_tuple="S102:O1",
            study_id="S102",
            endpoint="Total lung hydroxyproline content",
            context=ctx_s102,
            contrast=MultiArmContrast(
                focal_arms=("VAR:S102:NX410_30mgkg",),
                comparator_arms=("VAR:S102:Pirfenidone_100mgkg", "VAR:S102:Vehicle"),
                contrast_kind="multi_arm_joint",
            ),
            focal_quantity=QuantitySpec(
                kind="point_level",
                relation="=",
                value="142.5",
                unit="ug/lung",
                unit_dimension="mass_per_organ",
            ),
            comparator_quantity=QuantitySpec(
                kind="interval_bound",
                lower="176.0",
                upper="214.0",
                unit="ug/lung",
                unit_dimension="mass_per_organ",
            ),
            statistic_kind="mean",
            sample_size=36,
            p_value_or_interval="p=0.008",
            polarity="favorable",
            provenance=ProvenancePin(
                document_id="SYNTHETIC-DOI:S102",
                section_pointer="Results p.6, Table 2",
                verbatim_quote="Day-21 lung hydroxyproline averaged 142.5 ug/lung with NX-410 vs 176.0 ug/lung pirfenidone and 214.0 ug/lung vehicle (N=36, p=0.008).",
                requires_figure_digitization=False,
                primary_pdf_verified=True,
            ),
        ),
        "S102:O2": EvidenceAtom(
            atom_id="ATOM:S102:O2",
            outcome_tuple="S102:O2",
            study_id="S102",
            endpoint="Day-28 survival censor",
            context=ctx_s102,
            contrast=MultiArmContrast(
                focal_arms=("VAR:S102:NX410_30mgkg",),
                comparator_arms=("VAR:S102:Vehicle",),
                contrast_kind="pairwise",
            ),
            focal_quantity=QuantitySpec(
                kind="censored_observation",
                relation=">",
                value="28",
                unit="days",
                unit_dimension="time_duration",
            ),
            comparator_quantity=None,
            statistic_kind="median",
            sample_size=24,
            p_value_or_interval="p=0.03",
            polarity="favorable",
            provenance=ProvenancePin(
                document_id="SYNTHETIC-DOI:S102",
                section_pointer="Results p.7, Fig. 2C",
                verbatim_quote="Median survival exceeded 28 days in NX-410 mice on Kaplan-Meier inspection (Fig. 2C, n=24, p=0.03).",
                requires_figure_digitization=True,
                primary_pdf_verified=True,
            ),
        ),
        "S103:O1": EvidenceAtom(
            atom_id="ATOM:S103:O1",
            outcome_tuple="S103:O1",
            study_id="S103",
            endpoint="Secreted PRO-C3 neo-epitope concentration",
            context=ctx_s103,
            contrast=MultiArmContrast(
                focal_arms=("VAR:S103:NX410_180nM_Unbound",),
                comparator_arms=("VAR:S103:Vehicle_Slice",),
                contrast_kind="pairwise",
            ),
            focal_quantity=QuantitySpec(
                kind="point_level",
                relation="=",
                value="29.4",
                unit="ng/mL",
                unit_dimension="mass_concentration",
            ),
            comparator_quantity=QuantitySpec(
                kind="point_level",
                relation="=",
                value="48.6",
                unit="ng/mL",
                unit_dimension="mass_concentration",
            ),
            statistic_kind="mean",
            sample_size=18,
            p_value_or_interval="p=0.002",
            polarity="favorable",
            provenance=ProvenancePin(
                document_id="SYNTHETIC-DOI:S103",
                section_pointer="Results p.5, Table 1",
                verbatim_quote="In explanted IPF precision-cut lung slices (n=18 donors), NX-410 lowered secreted PRO-C3 from 48.6 ng/mL in vehicle slices to 29.4 ng/mL (p=0.002).",
                requires_figure_digitization=False,
                primary_pdf_verified=True,
            ),
        ),
        "S103:O2": EvidenceAtom(
            atom_id="ATOM:S103:O2",
            outcome_tuple="S103:O2",
            study_id="S103",
            endpoint="Tissue slice ATP viability",
            context=ctx_s103,
            contrast=MultiArmContrast(
                focal_arms=("VAR:S103:NX410_180nM_Unbound",),
                comparator_arms=("VAR:S103:Vehicle_Slice",),
                contrast_kind="pairwise",
            ),
            focal_quantity=QuantitySpec(
                kind="point_level",
                relation="=",
                value="96.2",
                unit="%",
                unit_dimension="dimensionless_percentage",
            ),
            comparator_quantity=QuantitySpec(
                kind="point_level",
                relation="=",
                value="95.8",
                unit="%",
                unit_dimension="dimensionless_percentage",
            ),
            statistic_kind="mean",
            sample_size=18,
            p_value_or_interval="p=0.74",
            polarity="neutral_or_mixed",
            provenance=ProvenancePin(
                document_id="SYNTHETIC-DOI:S103",
                section_pointer="Results p.5, Table 1",
                verbatim_quote="ATP slice viability remained 96.2% in treated slices vs 95.8% in vehicle slices (n=18, p=0.74).",
                requires_figure_digitization=False,
                primary_pdf_verified=True,
            ),
        ),
        "S104:O1": EvidenceAtom(
            atom_id="ATOM:S104:O1",
            outcome_tuple="S104:O1",
            study_id="S104",
            endpoint="12-week forced vital capacity change",
            context=ctx_s104,
            contrast=MultiArmContrast(
                focal_arms=("VAR:S104:NX410_40mg_BID",),
                comparator_arms=("VAR:S104:Placebo_BID",),
                contrast_kind="pairwise",
            ),
            focal_quantity=QuantitySpec(
                kind="point_level",
                relation="=",
                value="-28.5",
                unit="mL",
                unit_dimension="volume",
            ),
            comparator_quantity=QuantitySpec(
                kind="point_level",
                relation="=",
                value="-84.0",
                unit="mL",
                unit_dimension="volume",
            ),
            statistic_kind="mean",
            sample_size=64,
            p_value_or_interval="p=0.014",
            polarity="favorable",
            provenance=ProvenancePin(
                document_id="SYNTHETIC-REGISTRY:S104",
                section_pointer="Primary Efficacy Table 2",
                verbatim_quote="Mean 12-week FVC change was -28.5 mL in the NX-410 40 mg BID arm versus -84.0 mL in the placebo arm (N=64, p=0.014).",
                requires_figure_digitization=False,
                primary_pdf_verified=True,
            ),
        ),
        "S104:O2": EvidenceAtom(
            atom_id="ATOM:S104:O2",
            outcome_tuple="S104:O2",
            study_id="S104",
            endpoint="12-week supine vital capacity secondary readout",
            context=ctx_s104,
            contrast=MultiArmContrast(
                focal_arms=("VAR:S104:NX410_40mg_BID",),
                comparator_arms=("VAR:S104:Placebo_BID",),
                contrast_kind="pairwise",
            ),
            focal_quantity=QuantitySpec(
                kind="point_level",
                relation="=",
                value="-31.0",
                unit="mL",
                unit_dimension="volume",
            ),
            comparator_quantity=QuantitySpec(
                kind="point_level",
                relation="=",
                value="-79.5",
                unit="mL",
                unit_dimension="volume",
            ),
            statistic_kind="mean",
            sample_size=64,
            p_value_or_interval="p=0.021",
            polarity="favorable",
            provenance=ProvenancePin(
                document_id="SYNTHETIC-REGISTRY:S104",
                section_pointer="Secondary Efficacy Table 3",
                verbatim_quote="Mean 12-week supine vital capacity change was -31.0 mL in the NX-410 40 mg BID arm versus -79.5 mL in the placebo arm (N=64, p=0.021).",
                requires_figure_digitization=False,
                primary_pdf_verified=True,
            ),
        ),
        "S104:O3": EvidenceAtom(
            atom_id="ATOM:S104:O3",
            outcome_tuple="S104:O3",
            study_id="S104",
            endpoint="Grade >=2 gastrointestinal adverse event incidence",
            context=ctx_s104,
            contrast=MultiArmContrast(
                focal_arms=("VAR:S104:NX410_40mg_BID",),
                comparator_arms=("VAR:S104:Placebo_BID",),
                contrast_kind="pairwise",
            ),
            focal_quantity=QuantitySpec(
                kind="point_level",
                relation="=",
                value="18.8",
                unit="%",
                unit_dimension="dimensionless_percentage",
            ),
            comparator_quantity=QuantitySpec(
                kind="point_level",
                relation="=",
                value="9.4",
                unit="%",
                unit_dimension="dimensionless_percentage",
            ),
            statistic_kind="proportion",
            sample_size=64,
            p_value_or_interval="p=0.28",
            polarity="unfavorable",
            provenance=ProvenancePin(
                document_id="SYNTHETIC-REGISTRY:S104",
                section_pointer="Safety Table 4",
                verbatim_quote="Grade >=2 gastrointestinal adverse events occurred in 18.8% of NX-410 40 mg BID participants vs 9.4% on placebo (N=64, p=0.28).",
                requires_figure_digitization=False,
                primary_pdf_verified=True,
            ),
        ),
        "S105:O1": EvidenceAtom(
            atom_id="ATOM:S105:O1",
            outcome_tuple="S105:O1",
            study_id="S105",
            endpoint="ALT >3x ULN transaminase elevation incidence",
            context=ctx_s105,
            contrast=MultiArmContrast(
                focal_arms=("VAR:S105:NX410_120mg_BID",),
                comparator_arms=("VAR:S105:Placebo_BID",),
                contrast_kind="pairwise",
            ),
            focal_quantity=QuantitySpec(
                kind="point_level",
                relation="=",
                value="25.0",
                unit="%",
                unit_dimension="dimensionless_percentage",
            ),
            comparator_quantity=QuantitySpec(
                kind="point_level",
                relation="=",
                value="0",
                unit="%",
                unit_dimension="dimensionless_percentage",
            ),
            statistic_kind="proportion",
            sample_size=36,
            p_value_or_interval="p=0.04",
            polarity="unfavorable",
            provenance=ProvenancePin(
                document_id="SYNTHETIC-REGISTRY:S105",
                section_pointer="Dose-Escalation Safety Table 2",
                verbatim_quote="At 120 mg BID, reversible ALT >3x ULN elevations occurred in 25.0% vs 0% placebo (N=36, p=0.04).",
                requires_figure_digitization=False,
                primary_pdf_verified=True,
            ),
        ),
    }

    raw_mentioned_arms = {
        "S101:O1": ("VAR:S101:NX410_10uM", "VAR:S101:DMSO_Control"),
        "S102:O1": ("VAR:S102:NX410_30mgkg", "VAR:S102:Pirfenidone_100mgkg", "VAR:S102:Vehicle"),
        "S102:O2": ("VAR:S102:NX410_30mgkg", "VAR:S102:Vehicle"),
        "S103:O1": ("VAR:S103:NX410_180nM_Unbound", "VAR:S103:Vehicle_Slice"),
        "S103:O2": ("VAR:S103:NX410_180nM_Unbound", "VAR:S103:Vehicle_Slice"),
        "S104:O1": ("VAR:S104:NX410_40mg_BID", "VAR:S104:Placebo_BID"),
        "S104:O2": ("VAR:S104:NX410_40mg_BID", "VAR:S104:Placebo_BID"),
        "S104:O3": ("VAR:S104:NX410_40mg_BID", "VAR:S104:Placebo_BID"),
        "S105:O1": ("VAR:S105:NX410_120mg_BID", "VAR:S105:Placebo_BID"),
    }

    graph = ContextEvidenceGraph(
        lattice=lattice,
        studies=studies,
        study_variables=study_variables,
        arm_aliases=arm_aliases,
        atoms=atoms,
        raw_mentioned_arms_by_tuple=raw_mentioned_arms,
    )

    # 2. Corpora and ledger. The settled corpus D_3 is the complete projection of the graph;
    #    the raw extraction D_0 differs from it by three extraction defects, each repaired by one event.
    initial_corpus, settled_corpus, patch_events = _corpora_and_ledger(
        graph,
        "NX410_TRANSLATIONAL_EVIDENCE_CORPUS",
        [
            # Raw extraction mis-typed the relative % reduction as a level.
            ("PATCH-001", "S101", "/outcomes/0/quantity_kind", "point_level", "Reclassify 72.0% COL1A1 reduction from point_level to relative_change."),
            # Raw extraction omitted the active comparator arm.
            ("PATCH-002", "S102", "/outcomes/0/comparator_arms", ["VAR:S102:Vehicle"], "Restore active comparator arm Pirfenidone_100mgkg in 3-arm murine contrast."),
            # Raw extraction did not flag the Kaplan-Meier readout as undigitized.
            ("PATCH-003", "S102", "/outcomes/1/requires_figure_digitization", False, "Flag Kaplan-Meier curve readout S102:O2 (Fig. 2C) as requiring figure digitization."),
        ],
        study_metadata={
            "S101": {"title": "High-concentration NX-410 bath suppresses TGF-beta1-induced COL1A1 mRNA in A549-fibroblast co-culture"},
            "S102": {"title": "Three-arm therapeutic evaluation of NX-410, pirfenidone, and vehicle in murine bleomycin lung fibrosis"},
            "S103": {"title": "Explanted human IPF precision-cut lung slices treated at clinically matched unbound NX-410 concentrations"},
            "S104": {"title": "Randomized 12-week Phase IIa trial of oral NX-410 40 mg BID versus placebo in idiopathic pulmonary fibrosis"},
            "S105": {"title": "Phase Ib dose-escalation safety cohort of high-dose oral NX-410 120 mg BID"},
        },
    )

    # 4. Canonical valid synthesis records spanning all valid epistemic roles
    valid_records = [
        SynthesisClaimRecord(
            record_id="SYN-CLIN-01",
            polarity="SUPPORTS",
            target_context=target_ctx,
            supporting_tuples=("S104:O1",),
            counterevidence_tuples=("S104:O3",),
            prose_sentences=(
                "In S104:O1 the NX-410 40 mg BID arm showed -28.5 mL forced vital capacity change whereas the placebo arm declined -84.0 mL across 64 patients.",
                "In S104:O3 Grade >=2 gastrointestinal events occurred in 18.8% of the NX-410 40 mg BID arm versus 9.4% on placebo.",
            ),
            typed_propositions=(
                TypedProposition(
                    proposition_id="PROP:SYN-CLIN-01:A",
                    outcome_tuple="S104:O1",
                    claimed_endpoint="12-week forced vital capacity change",
                    claimed_focal_arms=("VAR:S104:NX410_40mg_BID",),
                    claimed_comparator_arms=("VAR:S104:Placebo_BID",),
                    claimed_quantity=QuantitySpec(
                        kind="point_level",
                        relation="=",
                        value="-28.5",
                        unit="mL",
                        unit_dimension="volume",
                    ),
                    claimed_statistic="mean",
                    claimed_sample_size=64,
                    claimed_comparator_quantity=QuantitySpec(
                        kind="point_level",
                        relation="=",
                        value="-84.0",
                        unit="mL",
                        unit_dimension="volume",
                    ),
                ),
            ),
            requests_direct_quantitative_promotion=True,
        ),
        SynthesisClaimRecord(
            record_id="SYN-BRIDGE-02",
            polarity="CANDIDATE_UPGRADE",
            target_context=target_ctx,
            supporting_tuples=("S102:O1", "S103:O1", "S103:O2"),
            counterevidence_tuples=(),
            prose_sentences=(
                "In S103:O1 treated slices reduced PRO-C3 from 48.6 ng/mL in vehicle slices to 29.4 ng/mL, while in S103:O2 slice ATP viability remained 96.2% versus 95.8%.",
                "In S102:O1 murine lung hydroxyproline averaged 142.5 ug/lung with NX-410 compared with 176.0 to 214.0 ug/lung across active pirfenidone and vehicle controls.",
            ),
            typed_propositions=(
                TypedProposition(
                    proposition_id="PROP:SYN-BRIDGE-02:A",
                    outcome_tuple="S103:O1",
                    claimed_endpoint="Secreted PRO-C3 neo-epitope concentration",
                    claimed_focal_arms=("VAR:S103:NX410_180nM_Unbound",),
                    claimed_comparator_arms=("VAR:S103:Vehicle_Slice",),
                    claimed_quantity=QuantitySpec(
                        kind="point_level",
                        relation="=",
                        value="29.4",
                        unit="ng/mL",
                        unit_dimension="mass_concentration",
                    ),
                    claimed_statistic="mean",
                    claimed_sample_size=18,
                    claimed_comparator_quantity=QuantitySpec(
                        kind="point_level",
                        relation="=",
                        value="48.6",
                        unit="ng/mL",
                        unit_dimension="mass_concentration",
                    ),
                ),
            ),
            prospective_gate=ProspectiveFalsifierGate(
                gate_id="GATE-BIOMARKER-PK-01",
                unresolved_context_axes=(
                    "disease_chronicity",
                    "dosing_exposure",
                    "model_system",
                    "readout_maturity",
                ),
                head_to_head_arms=("NX-410 40 mg BID oral", "Placebo BID oral"),
                primary_readout_and_unit=" Paired serum PRO-C3 (ng/mL) and 24-week FVC slope (mL)",
                quantitative_pass_threshold=">=25% serum PRO-C3 reduction at week 12 concordant with >=40 mL FVC preservation",
            ),
            requests_direct_quantitative_promotion=False,
        ),
        SynthesisClaimRecord(
            record_id="SYN-SCOPE-03",
            polarity="FALSIFIED_GRAVEYARD",
            target_context=target_ctx,
            supporting_tuples=(),
            counterevidence_tuples=("S105:O1",),
            prose_sentences=(
                "Escalation to 120 mg BID is excluded because S105:O1 observed 25.0% transaminase elevations versus 0% on placebo across 36 participants.",
            ),
            typed_propositions=(),
            requests_direct_quantitative_promotion=False,
        ),
    ]

    # 5. Canonical invalid probe records illustrating each invariant tier
    invalid_probes: list[tuple[str, SynthesisClaimRecord, str]] = [
        (
            "PROBE_INADMISSIBLE_2D_OVERCLAIM",
            SynthesisClaimRecord(
                record_id="BAD-01",
                polarity="CANDIDATE_UPGRADE",
                target_context=target_ctx,
                supporting_tuples=("S101:O1",),
                counterevidence_tuples=(),
                prose_sentences=("In S101:O1 10 uM bath reduced COL1A1 transcript by 72.0%.",),
                typed_propositions=(),
                prospective_gate=ProspectiveFalsifierGate(
                    gate_id="G-DUMMY",
                    unresolved_context_axes=(
                        "disease_chronicity",
                        "dosing_exposure",
                        "model_system",
                        "readout_maturity",
                    ),
                    head_to_head_arms=("A", "B"),
                    primary_readout_and_unit="FVC (mL)",
                    quantitative_pass_threshold=">0",
                ),
            ),
            "I4_INADMISSIBLE_CONTEXT_TRANSFER",
        ),
        (
            "PROBE_CLAUSE_CROSS_OUTCOME_SWAP",
            SynthesisClaimRecord(
                record_id="BAD-02",
                polarity="SCOPING_QUALIFICATION",
                target_context=target_ctx,
                supporting_tuples=("S103:O1", "S103:O2"),
                counterevidence_tuples=(),
                prose_sentences=(
                    "In S103:O1 PRO-C3 was 96.2 ng/mL whereas in S103:O2 slice viability was 29.4%.",
                ),
                typed_propositions=(),
            ),
            "I3_CLAUSE_CROSS_OUTCOME_SWAP",
        ),
        (
            "PROBE_POOLED_CLAUSE_ENDPOINT_MISBINDING_CAUGHT_BY_I3_TYPED",
            SynthesisClaimRecord(
                record_id="BAD-03",
                polarity="SUPPORTS",
                target_context=target_ctx,
                supporting_tuples=("S104:O1", "S104:O2"),
                counterevidence_tuples=(),
                # Note: trailing pooled citation "(S104:O1, S104:O2)" with identical mL units passes union/clause token pool,
                # demonstrating Theorem 2 (surface-prose pool collision) and closed by Theorem 3 (I3_typed)!
                prose_sentences=(
                    "Mean 12-week forced vital capacity change was -31.0 mL versus -79.5 mL on placebo (S104:O1, S104:O2).",
                ),
                typed_propositions=(
                    TypedProposition(
                        proposition_id="PROP:BAD-03",
                        outcome_tuple="S104:O1",
                        claimed_endpoint="12-week forced vital capacity change",
                        claimed_focal_arms=("VAR:S104:NX410_40mg_BID",),
                        claimed_comparator_arms=("VAR:S104:Placebo_BID",),
                        claimed_quantity=QuantitySpec(
                            kind="point_level",
                            relation="=",
                            value="-31.0",
                            unit="mL",
                            unit_dimension="volume",
                        ),
                        claimed_statistic="mean",
                        claimed_sample_size=64,
                        claimed_comparator_quantity=QuantitySpec(
                            kind="point_level",
                            relation="=",
                            value="-79.5",
                            unit="mL",
                            unit_dimension="volume",
                        ),
                    ),
                ),
                requests_direct_quantitative_promotion=True,
            ),
            "I3_TYPED_QUANTITY_MISMATCH",
        ),
        (
            "PROBE_UNDIGITIZED_FIGURE_PROMOTION",
            SynthesisClaimRecord(
                record_id="BAD-04",
                polarity="SUPPORTS",
                target_context=ctx_s102,
                supporting_tuples=("S102:O2",),
                counterevidence_tuples=(),
                prose_sentences=("In S102:O2 median survival exceeded 28 days.",),
                typed_propositions=(
                    TypedProposition(
                        proposition_id="PROP:BAD-04",
                        outcome_tuple="S102:O2",
                        claimed_endpoint="Day-28 survival censor",
                        claimed_focal_arms=("VAR:S102:NX410_30mgkg",),
                        claimed_comparator_arms=("VAR:S102:Vehicle",),
                        claimed_quantity=QuantitySpec(
                            kind="censored_observation",
                            relation=">",
                            value="28",
                            unit="days",
                            unit_dimension="time_duration",
                        ),
                        claimed_statistic="median",
                        claimed_sample_size=24,
                    ),
                ),
                requests_direct_quantitative_promotion=True,
            ),
            "I5_ILLEGAL_QUANTITATIVE_PROMOTION_WITHOUT_PRIMARY_OR_DIGITIZED_SOURCE",
        ),
        (
            "PROBE_SUPPORT_COUNTER_OVERLAP",
            SynthesisClaimRecord(
                record_id="BAD-05",
                polarity="SCOPING_QUALIFICATION",
                target_context=target_ctx,
                supporting_tuples=("S104:O3",),
                counterevidence_tuples=("S104:O3",),
                prose_sentences=("In S104:O3 gastrointestinal events were 18.8% versus 9.4%.",),
                typed_propositions=(),
            ),
            "I6_SUPPORT_COUNTEREVIDENCE_OVERLAP",
        ),
    ]

    return DomainVignetteBundle(
        domain_name="drug_development",
        target_context=target_ctx,
        graph=graph,
        initial_corpus=initial_corpus,
        settled_corpus=settled_corpus,
        patch_events=patch_events,
        valid_records=valid_records,
        invalid_probe_records=invalid_probes,
    )


def build_sport_exercise_vignette() -> DomainVignetteBundle:
    """Build modular sport & exercise physiology swap-pack vignette."""
    cohort_poset = ContextAxisPoset(
        name="cohort_training_tier",
        dominates={
            "c2c12_myotube_vitro": frozenset({"c2c12_myotube_vitro"}),
            "human_whole_body_floor": frozenset({"c2c12_myotube_vitro", "human_whole_body_floor"}),
            "recreational_adults": frozenset(
                {"c2c12_myotube_vitro", "human_whole_body_floor", "recreational_adults"}
            ),
            "subelite_collegiate": frozenset(
                {"c2c12_myotube_vitro", "human_whole_body_floor", "subelite_collegiate"}
            ),
            "elite_endurance_cohort": frozenset(
                {
                    "c2c12_myotube_vitro",
                    "human_whole_body_floor",
                    "recreational_adults",
                    "subelite_collegiate",
                    "elite_endurance_cohort",
                }
            ),
        },
    )
    thermal_poset = ContextAxisPoset(
        name="environmental_stress",
        dominates={
            "thermoneutral_20C": frozenset({"thermoneutral_20C"}),
            "humid_heat_32C": frozenset({"thermoneutral_20C", "humid_heat_32C"}),
        },
    )
    lattice = ContextLatticeSpec(
        axis_posets={
            "cohort_training_tier": cohort_poset,
            "environmental_stress": thermal_poset,
        },
        bridge_floor={"cohort_training_tier": "human_whole_body_floor"},
    )
    target_ctx = ContextSignature(
        {"cohort_training_tier": "elite_endurance_cohort", "environmental_stress": "humid_heat_32C"}
    )

    ctx_vitro = ContextSignature(
        {"cohort_training_tier": "c2c12_myotube_vitro", "environmental_stress": "thermoneutral_20C"}
    )
    studies = {"E201": ctx_vitro, "E202": target_ctx}
    study_variables = {
        "E201": {"VAR:E201:Maltodextrin_Bath", "VAR:E201:Basal_Media"},
        "E202": {"VAR:E202:Alginate_CHO_90gh", "VAR:E202:Isoenergetic_Glucose_90gh"},
    }
    atoms = {
        "E201:O1": EvidenceAtom(
            atom_id="ATOM:E201:O1",
            outcome_tuple="E201:O1",
            study_id="E201",
            endpoint="Myotube glycogen synthase fractional velocity",
            context=ctx_vitro,
            contrast=MultiArmContrast(
                focal_arms=("VAR:E201:Maltodextrin_Bath",),
                comparator_arms=("VAR:E201:Basal_Media",),
                contrast_kind="pairwise",
            ),
            focal_quantity=QuantitySpec(
                kind="relative_change",
                relation="=",
                value="45.0",
                unit="%",
                unit_dimension="dimensionless_percentage",
                change_direction="increase",
            ),
            comparator_quantity=None,
            statistic_kind="fold_change",
            sample_size=6,
            p_value_or_interval="p=0.01",
            polarity="favorable",
            provenance=ProvenancePin(
                document_id="SYNTHETIC-DOI:E201",
                section_pointer="Table 1",
                verbatim_quote="Glycogen synthase fractional velocity increased by 45.0% (n=6, p=0.01).",
                requires_figure_digitization=False,
                primary_pdf_verified=True,
            ),
        ),
        "E202:O1": EvidenceAtom(
            atom_id="ATOM:E202:O1",
            outcome_tuple="E202:O1",
            study_id="E202",
            endpoint="15-minute time-trial mean power output after 120-minute preload at 32C",
            context=target_ctx,
            contrast=MultiArmContrast(
                focal_arms=("VAR:E202:Alginate_CHO_90gh",),
                comparator_arms=("VAR:E202:Isoenergetic_Glucose_90gh",),
                contrast_kind="pairwise",
            ),
            focal_quantity=QuantitySpec(
                kind="point_level",
                relation="=",
                value="298.0",
                unit="W",
                unit_dimension="power",
            ),
            comparator_quantity=QuantitySpec(
                kind="point_level",
                relation="=",
                value="284.0",
                unit="W",
                unit_dimension="power",
            ),
            statistic_kind="mean",
            sample_size=16,
            p_value_or_interval="p=0.009",
            polarity="favorable",
            provenance=ProvenancePin(
                document_id="SYNTHETIC-DOI:E202",
                section_pointer="Results p.8, Table 2",
                verbatim_quote="Mean 15-min time-trial power was 298.0 W with alginate-encapsulated CHO vs 284.0 W control (n=16, p=0.009).",
                requires_figure_digitization=False,
                primary_pdf_verified=True,
            ),
        ),
    }
    raw_mentioned_arms = {
        "E201:O1": ("VAR:E201:Maltodextrin_Bath", "VAR:E201:Basal_Media"),
        "E202:O1": ("VAR:E202:Alginate_CHO_90gh", "VAR:E202:Isoenergetic_Glucose_90gh"),
    }
    graph = ContextEvidenceGraph(
        lattice=lattice,
        studies=studies,
        study_variables=study_variables,
        arm_aliases={},
        atoms=atoms,
        raw_mentioned_arms_by_tuple=raw_mentioned_arms,
    )

    # Corpora and ledger: D_1 is the complete projection of the graph; D_0 carries one extraction defect.
    initial_corpus, settled_corpus, patch_events = _corpora_and_ledger(
        graph,
        "CARB_HYDROGEL_ENDURANCE_CORPUS",
        [("PATCH-EX-001", "E201", "/outcomes/0/quantity_kind", "point_level", "Reclassify C2C12 glycogen resynthesis % increase as relative_change.")],
    )
    valid_records = [
        SynthesisClaimRecord(
            record_id="SPORT-PROM-01",
            polarity="SUPPORTS",
            target_context=target_ctx,
            supporting_tuples=("E202:O1",),
            counterevidence_tuples=(),
            prose_sentences=(
                "In E202:O1 alginate-encapsulated carbohydrate maintained 298.0 W versus 284.0 W across 16 elite athletes.",
            ),
            typed_propositions=(
                TypedProposition(
                    proposition_id="PROP:SPORT-01",
                    outcome_tuple="E202:O1",
                    claimed_endpoint="15-minute time-trial mean power output after 120-minute preload at 32C",
                    claimed_focal_arms=("VAR:E202:Alginate_CHO_90gh",),
                    claimed_comparator_arms=("VAR:E202:Isoenergetic_Glucose_90gh",),
                    claimed_quantity=QuantitySpec(
                        kind="point_level",
                        relation="=",
                        value="298.0",
                        unit="W",
                        unit_dimension="power",
                    ),
                    claimed_statistic="mean",
                    claimed_sample_size=16,
                    claimed_comparator_quantity=QuantitySpec(
                        kind="point_level",
                        relation="=",
                        value="284.0",
                        unit="W",
                        unit_dimension="power",
                    ),
                ),
            ),
            requests_direct_quantitative_promotion=True,
        )
    ]
    invalid_probes = [
        (
            "SPORT_VITRO_OVERCLAIM",
            SynthesisClaimRecord(
                record_id="SPORT-BAD-01",
                polarity="SUPPORTS",
                target_context=target_ctx,
                supporting_tuples=("E201:O1",),
                counterevidence_tuples=(),
                prose_sentences=("In E201:O1 myotube synthase velocity rose 45.0%.",),
                typed_propositions=(),
                requests_direct_quantitative_promotion=True,
            ),
            "I4_INADMISSIBLE_CONTEXT_TRANSFER",
        )
    ]
    return DomainVignetteBundle(
        domain_name="sport_science",
        target_context=target_ctx,
        graph=graph,
        initial_corpus=initial_corpus,
        settled_corpus=settled_corpus,
        patch_events=patch_events,
        valid_records=valid_records,
        invalid_probe_records=invalid_probes,
    )


# ---------------------------------------------------------------------------
# Type 1 diabetes worked example (Section 5 of the manuscript).
#
# Eight peer-reviewed studies and ten findings taken from the OpenIslet corpus
# (release v19.0.0-R4S4-SOL61-FOLDIN). Every `quote` below is the corpus's
# verbatim quote for that outcome, copied by script; arm identifiers, figure
# flags and sample sizes are those of the corpus record. A sample size is given
# only where the corpus states one unambiguous number. `primary_pdf_verified`
# is False throughout: the corpus certifies no outcome against its primary PDF.
# `t1d/check_t1d_vignette.py` re-checks all of this against the corpus.
# ---------------------------------------------------------------------------

_T1D_FINDINGS: tuple[dict[str, Any], ...] = (
    {
        "tuple": "R1_0340:O2",
        "pmid": "41020949",
        "context": "population_t2d",
        "study_context": "population_t2d",
        "endpoint": "Type 2 diabetes risk in SLC30A8 LoF heterozygotes (additive model)",
        "focal": ["A2_LoF_heterozygotes"],
        "comparator": ["A1_noncarriers"],
        "contrast_kind": "pairwise",
        "quantity": {"kind": "relative_change", "relation": "=", "value": "37", "unit": "%", "change_direction": "decrease"},
        "statistic": "odds_ratio",
        "n": None,
        "p": "OR 0.62; 95% CI 0.53 to 0.72; p = 1.1e-9",
        "polarity": "favorable",
        "section": "results_text; Fig 2a",
        "figure": False,
        "quote": "The protective effect was also observed in heterozygotes, with a 37% decrease in type 2 diabetes risk (ORadditive=0.62; 95% CI 0.53, 0.72; p=1.1\u00d710\u20139)",
    },
    {
        "tuple": "R1_0141:O9",
        "pmid": "31676859",
        "context": "sc_invitro_basal",
        "study_context": "sc_invitro_basal",
        "endpoint": "Number of INS- and SLC30A8-transcript expressing cells in hiPSC-derived beta-like cells",
        "focal": ["A7_iPSC_Arg138_het", "A8_iPSC_Lys34_hom"],
        "comparator": ["A6_iPSC_control"],
        "contrast_kind": "multi_arm_joint",
        "quantity": {"kind": "missing_numeric", "missing_reason": "shown_in_figure_without_tabulated_value"},
        "statistic": "qualitative",
        "n": None,
        "p": None,
        "polarity": "unfavorable",
        "section": "results_text; Extended Data Fig 4d-g",
        "figure": True,
        "quote": "the number of INS- and SLC30A8-transcript expressing cells were reduced in clones with premature stop codons in SLC30A8 (Extended Data Fig. 4d-g), indicating a reduced formation of beta-like cells",
    },
    {
        "tuple": "R1_0116:O1",
        "pmid": "29679103",
        "context": "clinical_graft",
        "study_context": "clinical_graft",
        "endpoint": "Independent risk factors for failure to achieve insulin independence (multivariate analysis)",
        "focal": ["A1_islet_allograft_recipient_cohort"],
        "comparator": [],
        "contrast_kind": "single_arm",
        "quantity": {"kind": "missing_numeric", "missing_reason": "qualitative_association_with_p_value_range"},
        "statistic": "qualitative",
        "n": 40,
        "p": "p = 0.015-0.046 (range across the three factors)",
        "polarity": "unfavorable",
        "section": "abstract",
        "figure": False,
        "quote": "In multivariate analysis, HLA-A*24 positivity, presence of SLC30A8 CT or TT genotypes and BMI more than or equal to the group median (23.9 kg/m2) were independently associated with failure to achieve insulin independence (p = 0.015-0.046).",
    },
    {
        "tuple": "R1_0214:O4",
        "pmid": "36055241",
        "context": "humanized_mouse",
        "study_context": "humanized_mouse",
        "endpoint": "gRNA enrichment in hPi grafts in the whole-genome CRISPR screen",
        "focal": ["A3_LT_hPi"],
        "comparator": ["A4_LT_ctrl"],
        "contrast_kind": "pairwise",
        "quantity": {"kind": "missing_numeric", "missing_reason": "shown_in_figure_without_tabulated_value"},
        "statistic": "qualitative",
        "n": 12,
        "p": None,
        "polarity": "favorable",
        "section": "results_text; Fig 1G",
        "figure": True,
        "quote": "Most prominent were the enrichments of B2M, HLA-A, TAP1/2, STAT1, JAK1/2, and CXCL10 gRNAs in LT SC-islet hPi grafts (Figure 1G).",
    },
    {
        "tuple": "R1_0214:O12",
        "pmid": "36055241",
        "context": "sc_invitro_cytokine",
        "study_context": "humanized_mouse",
        "endpoint": "IFN-gamma-induced HLA, CXCL10 and inhibitory protein expression in STAT1 KO (ST1L) SC-islets",
        "focal": ["A16_ST1L"],
        "comparator": ["A15_GL_ctrl"],
        "contrast_kind": "pairwise",
        "quantity": {"kind": "missing_numeric", "missing_reason": "shown_in_figure_without_tabulated_value"},
        "statistic": "qualitative",
        "n": None,
        "p": None,
        "polarity": "neutral_or_mixed",
        "section": "results_text; Fig 4D/4E",
        "figure": True,
        "quote": "The absence of STAT1 in ST1L also led to desensitization to IFN\u03b3, as shown by the downregulation of HLA proteins and CXCL10 as well as inhibitory proteins (HLA-E, PD-L1, and SOCS1) (Figures 4D and 4E).",
    },
    {
        "tuple": "R1_0215:O7",
        "pmid": "36103539",
        "context": "sc_invitro_cytokine",
        "study_context": "sc_invitro_cytokine",
        "endpoint": "HLA-ABC expression after NLRC5 knockdown in iPSC-derived beta-like cells exposed to IFN-alpha",
        "focal": ["A10_iPSC_siNLRC5_IFNa"],
        "comparator": ["A9_iPSC_siCTL_IFNa"],
        "contrast_kind": "pairwise",
        "quantity": {"kind": "missing_numeric", "missing_reason": "shown_in_figure_without_tabulated_value"},
        "statistic": "qualitative",
        "n": None,
        "p": None,
        "polarity": "favorable",
        "section": "figure_caption; Fig 3K",
        "figure": True,
        "quote": "NLRC5 depletion decreased IFN\u03b1-induced expression of HLA-ABC (D, G, and K)",
    },
    {
        "tuple": "R1_0215:O27",
        "pmid": "36103539",
        "context": "sc_invitro_cytokine",
        "study_context": "sc_invitro_cytokine",
        "endpoint": "BCL2L1 (Bcl-XL) expression in alpha-like versus beta-like IFN-alpha-exposed iPSC-islet cells",
        "focal": ["A4_iPSC_alpha_like_IFNa"],
        "comparator": ["A3_iPSC_beta_like_IFNa"],
        "contrast_kind": "pairwise",
        "quantity": {"kind": "relative_change", "relation": "=", "value": "0.13", "unit": "log2_fc", "change_direction": "increase"},
        "statistic": "fold_change",
        "n": None,
        "p": "adjusted P << 0.0001",
        "polarity": "neutral_or_mixed",
        "section": "results_text; Fig 1H",
        "figure": False,
        "quote": "Compared to \u03b2-like cells, \u03b1-like cells also had a higher expression of the antiapoptotic family member BCL2L1 (also known as Bcl-XL) (log2FC = 0.13, adjusted P << 0.0001)",
    },
    {
        "tuple": "R1_0098:O8",
        "pmid": "27787618",
        "context": "sc_invitro_basal",
        "study_context": "sc_invitro_basal",
        "endpoint": "CD55 (DAF) complement regulator expression, hESC-EC vs hESC-PE",
        "focal": ["A3_hESC_EC"],
        "comparator": ["A1_hESC_PE"],
        "contrast_kind": "pairwise",
        "quantity": {"kind": "missing_numeric", "missing_reason": "shown_in_figure_without_tabulated_value"},
        "statistic": "qualitative",
        "n": None,
        "p": None,
        "polarity": "favorable",
        "section": "results_text; Fig 1c",
        "figure": True,
        "quote": "The differentiated hESC-ECs clearly expressed decay-accelerating factor (DAF; CD55), similar to PTEC line HK-2 and HEK293 cells, while this expression was lower on hESC-PEs (Fig. 1c).",
    },
    {
        "tuple": "R1_0202:O7",
        "pmid": "35666870",
        "context": "beta_cell_line",
        "study_context": "beta_cell_line",
        "endpoint": "GSIS in INS-1 cells after CD59 knockout",
        "focal": ["A6_INS1_CD59KO"],
        "comparator": ["A5_INS1_control"],
        "contrast_kind": "pairwise",
        "quantity": {"kind": "missing_numeric", "missing_reason": "shown_in_figure_without_tabulated_value"},
        "statistic": "qualitative",
        "n": 5,
        "p": None,
        "polarity": "unfavorable",
        "section": "results_text; Fig 3A",
        "figure": True,
        "quote": "Knockout of endogenous CD59 strongly inhibits GSIS in INS-1 cells",
    },
    {
        "tuple": "R1_0376:O15",
        "pmid": "42120510",
        "context": "sc_invitro_hypoxia",
        "study_context": "sc_invitro_hypoxia",
        "endpoint": "TXNIP in SC-beta cells under hypoxia",
        "focal": ["A4_SC_hypoxia"],
        "comparator": ["A3_SC_normoxia"],
        "contrast_kind": "pairwise",
        "quantity": {"kind": "missing_numeric", "missing_reason": "shown_in_figure_without_tabulated_value"},
        "statistic": "qualitative",
        "n": None,
        "p": None,
        "polarity": "unfavorable",
        "section": "results_text; Fig 4C",
        "figure": True,
        "quote": "Their hyperactive redox state was marked by upregulation of TXNIP, GPX4, and HSF1.",
    },
)


def build_t1d_vignette() -> DomainVignetteBundle:
    """Build the type 1 diabetes islet-replacement case study (real published studies)."""
    # Four-axis context map. `model_system` is a diamond with a bridge floor:
    #   immortalized_beta_line < primary_islet_or_sc_floor
    #     < {humanized_mouse_allograft || human_sc_or_donor_islet_invitro} < clinical_t1d_islet_graft
    # plus the off-branch `adult_population_t2d_genetics`, which shares no upper bound.
    model_poset = ContextAxisPoset(
        name="model_system",
        dominates={
            "immortalized_beta_line": frozenset({"immortalized_beta_line"}),
            "primary_islet_or_sc_floor": frozenset({"immortalized_beta_line", "primary_islet_or_sc_floor"}),
            "humanized_mouse_allograft": frozenset(
                {"immortalized_beta_line", "primary_islet_or_sc_floor", "humanized_mouse_allograft"}
            ),
            "human_sc_or_donor_islet_invitro": frozenset(
                {"immortalized_beta_line", "primary_islet_or_sc_floor", "human_sc_or_donor_islet_invitro"}
            ),
            "clinical_t1d_islet_graft": frozenset(
                {
                    "immortalized_beta_line",
                    "primary_islet_or_sc_floor",
                    "humanized_mouse_allograft",
                    "human_sc_or_donor_islet_invitro",
                    "clinical_t1d_islet_graft",
                }
            ),
            "adult_population_t2d_genetics": frozenset({"adult_population_t2d_genetics"}),
        },
    )
    immune_poset = ContextAxisPoset(
        name="immune_challenge",
        dominates={
            "unreconstituted_or_basal": frozenset({"unreconstituted_or_basal"}),
            "cytokine_or_single_effector_invitro": frozenset(
                {"unreconstituted_or_basal", "cytokine_or_single_effector_invitro"}
            ),
            "pbmc_allogeneic_reconstituted": frozenset(
                {"unreconstituted_or_basal", "cytokine_or_single_effector_invitro", "pbmc_allogeneic_reconstituted"}
            ),
            "immunocompetent_auto_plus_allo_t1d": frozenset(
                {
                    "unreconstituted_or_basal",
                    "cytokine_or_single_effector_invitro",
                    "pbmc_allogeneic_reconstituted",
                    "immunocompetent_auto_plus_allo_t1d",
                }
            ),
        },
    )
    engraft_poset = ContextAxisPoset(
        name="engraftment_microenvironment",
        dominates={
            "normoxic_2d_culture": frozenset({"normoxic_2d_culture"}),
            "acute_hypoxic_or_inflammatory_stress": frozenset(
                {"normoxic_2d_culture", "acute_hypoxic_or_inflammatory_stress"}
            ),
            "vascularized_marginal_mass_graft": frozenset(
                {"normoxic_2d_culture", "acute_hypoxic_or_inflammatory_stress", "vascularized_marginal_mass_graft"}
            ),
        },
    )
    readout_poset = ContextAxisPoset(
        name="readout_maturity",
        dominates={
            "transcript_or_surface_marker": frozenset({"transcript_or_surface_marker"}),
            "functional_gsis_exocytosis": frozenset({"transcript_or_surface_marker", "functional_gsis_exocytosis"}),
            "durable_c_peptide_and_insulin_independence": frozenset(
                {
                    "transcript_or_surface_marker",
                    "functional_gsis_exocytosis",
                    "durable_c_peptide_and_insulin_independence",
                }
            ),
        },
    )
    lattice = ContextLatticeSpec(
        axis_posets={
            "model_system": model_poset,
            "immune_challenge": immune_poset,
            "engraftment_microenvironment": engraft_poset,
            "readout_maturity": readout_poset,
        },
        bridge_floor={"model_system": "primary_islet_or_sc_floor"},
    )

    def ctx(model: str, immune: str, engraft: str, readout: str) -> ContextSignature:
        return ContextSignature(
            {
                "model_system": model,
                "immune_challenge": immune,
                "engraftment_microenvironment": engraft,
                "readout_maturity": readout,
            }
        )

    contexts = {
        "clinical_graft": ctx(
            "clinical_t1d_islet_graft",
            "immunocompetent_auto_plus_allo_t1d",
            "vascularized_marginal_mass_graft",
            "durable_c_peptide_and_insulin_independence",
        ),
        # Off-branch study. Only `model_system` decides its classification; the other
        # coordinates have no natural value for a population cohort.
        "population_t2d": ctx(
            "adult_population_t2d_genetics",
            "unreconstituted_or_basal",
            "normoxic_2d_culture",
            "durable_c_peptide_and_insulin_independence",
        ),
        "sc_invitro_basal": ctx(
            "human_sc_or_donor_islet_invitro",
            "unreconstituted_or_basal",
            "normoxic_2d_culture",
            "transcript_or_surface_marker",
        ),
        "sc_invitro_cytokine": ctx(
            "human_sc_or_donor_islet_invitro",
            "cytokine_or_single_effector_invitro",
            "acute_hypoxic_or_inflammatory_stress",
            "transcript_or_surface_marker",
        ),
        "sc_invitro_hypoxia": ctx(
            "human_sc_or_donor_islet_invitro",
            "unreconstituted_or_basal",
            "acute_hypoxic_or_inflammatory_stress",
            "transcript_or_surface_marker",
        ),
        "humanized_mouse": ctx(
            "humanized_mouse_allograft",
            "pbmc_allogeneic_reconstituted",
            "vascularized_marginal_mass_graft",
            "transcript_or_surface_marker",
        ),
        "beta_cell_line": ctx(
            "immortalized_beta_line",
            "unreconstituted_or_basal",
            "normoxic_2d_culture",
            "functional_gsis_exocytosis",
        ),
    }
    target_ctx = contexts["clinical_graft"]

    atoms: dict[str, EvidenceAtom] = {}
    studies: dict[str, ContextSignature] = {}
    study_variables: dict[str, set[str]] = {}
    settled_studies: dict[str, dict[str, Any]] = {}
    for f in _T1D_FINDINGS:
        sid = f["tuple"].split(":")[0]
        focal = tuple(f"VAR:{sid}:{a}" for a in f["focal"])
        comp = tuple(f"VAR:{sid}:{a}" for a in f["comparator"])
        atom = EvidenceAtom(
            atom_id=f"ATOM:{f['tuple']}",
            outcome_tuple=f["tuple"],
            study_id=sid,
            endpoint=f["endpoint"],
            context=contexts[f["context"]],
            contrast=MultiArmContrast(focal_arms=focal, comparator_arms=comp, contrast_kind=f["contrast_kind"]),
            focal_quantity=QuantitySpec(**f["quantity"]),
            comparator_quantity=None,
            statistic_kind=f["statistic"],
            sample_size=f["n"],
            p_value_or_interval=f["p"],
            polarity=f["polarity"],
            provenance=ProvenancePin(
                document_id=f"PMID:{f['pmid']}",
                section_pointer=f["section"],
                verbatim_quote=f["quote"],
                requires_figure_digitization=f["figure"],
                primary_pdf_verified=False,
            ),
        )
        atoms[f["tuple"]] = atom
        studies.setdefault(sid, contexts[f["study_context"]])
        study_variables.setdefault(sid, set()).update(focal + comp)
        # Complete corpus record: every atom field, nulls explicit (release 3.2).
        record = corpus_outcome_from_atom(atom)
        settled_studies.setdefault(
            sid, {"study_id": sid, "context": dict(studies[sid].axes), "outcomes": []}
        )["outcomes"].append(record)

    settled_corpus: dict[str, Any] = {
        "corpus_id": "OPENISLET_T1D_WORKED_EXAMPLE",
        "studies": list(settled_studies.values()),
    }
    position = {st["study_id"]: i for i, st in enumerate(settled_corpus["studies"])}

    # Raw extraction D_0: the settled corpus with three extraction defects written in
    # on purpose, so that the example has something to repair. The defects are
    # constructed for this example; they are not events of the production ledger.
    initial_corpus = copy.deepcopy(settled_corpus)
    initial_corpus["studies"][position["R1_0340"]]["outcomes"][0]["quantity_kind"] = "point_level"
    initial_corpus["studies"][position["R1_0141"]]["outcomes"][0]["focal_arms"] = ["VAR:R1_0141:A7_iPSC_Arg138_het"]
    initial_corpus["studies"][position["R1_0215"]]["outcomes"][0]["requires_figure_digitization"] = False

    work_doc = copy.deepcopy(initial_corpus)
    patch_events = [
        _make_patch_event(
            work_doc,
            "T1D-PATCH-001",
            position["R1_0340"],
            "/outcomes/0/quantity_kind",
            "relative_change",
            "Retype the 37% decrease in type 2 diabetes risk in R1_0340:O2 from a level to a relative change.",
        ),
        _make_patch_event(
            work_doc,
            "T1D-PATCH-002",
            position["R1_0141"],
            "/outcomes/0/focal_arms",
            ["VAR:R1_0141:A7_iPSC_Arg138_het", "VAR:R1_0141:A8_iPSC_Lys34_hom"],
            "Restore the second premature-stop clone arm dropped from R1_0141:O9.",
        ),
        _make_patch_event(
            work_doc,
            "T1D-PATCH-003",
            position["R1_0215"],
            "/outcomes/0/requires_figure_digitization",
            True,
            "Flag R1_0215:O7 as read from a figure panel that has not been digitized.",
        ),
    ]
    if work_doc != settled_corpus:
        raise RuntimeError("T1D_VIGNETTE_PATCHES_DO_NOT_REACH_SETTLED_CORPUS")

    graph = ContextEvidenceGraph(
        lattice=lattice,
        studies=studies,
        study_variables=study_variables,
        arm_aliases={},
        atoms=atoms,
        raw_mentioned_arms_by_tuple={
            t: tuple(a.contrast.focal_arms) + tuple(a.contrast.comparator_arms) for t, a in atoms.items()
        },
    )

    all_axes = ("engraftment_microenvironment", "immune_challenge", "model_system", "readout_maturity")

    def record(
        record_id: str,
        polarity: str,
        supporting: tuple[str, ...],
        counter: tuple[str, ...],
        typed: tuple[str, ...],
        gate: ProspectiveFalsifierGate | None = None,
        promote: bool = False,
        prose: tuple[str, ...] | None = None,
    ) -> SynthesisClaimRecord:
        return populate_claim_attestation(
            SynthesisClaimRecord(
                record_id=record_id,
                polarity=polarity,  # type: ignore[arg-type]
                target_context=target_ctx,
                supporting_tuples=supporting,
                counterevidence_tuples=counter,
                prose_sentences=(
                    prose if prose is not None else tuple(render_atom_sentence(atoms[t]) for t in supporting + counter)
                ),
                typed_propositions=tuple(
                    proposition_from_atom(atoms[t], f"PROP:{record_id}:{t}") for t in typed
                ),
                prospective_gate=gate,
                requests_direct_quantitative_promotion=promote,
            ),
            graph.atoms,
        )

    valid_records = [
        # The one finding from the target setting itself. It is qualitative, so nothing is promoted.
        record("T1D-CLIN-DIRECT-01", "SUPPORTS", ("R1_0116:O1",), (), ("R1_0116:O1",)),
        record(
            "T1D-CEG-H1-SLC30A8-INVERSION",
            "SCOPING_QUALIFICATION",
            ("R1_0141:O9", "R1_0116:O1"),
            ("R1_0340:O2",),
            ("R1_0340:O2", "R1_0141:O9", "R1_0116:O1"),
            gate=ProspectiveFalsifierGate(
                gate_id="GATE-H1-SLC30A8-MARGINAL-MASS",
                unresolved_context_axes=all_axes,
                head_to_head_arms=("SLC30A8_WT_SC_Islets", "SLC30A8_Arg138X_or_EpitopeEdited_SC_Islets"),
                primary_readout_and_unit="Stage-7 INS+NKX6.1+ beta-like yield (%) and 12-week stimulated C-peptide (ng/mL)",
                quantitative_pass_threshold=">=90% WT Stage-7 beta-like yield and non-inferior 12-week C-peptide at 1000 IEQ",
            ),
        ),
        record(
            "T1D-CEG-H2-NLRC5-UNCOUPLING",
            "CANDIDATE_UPGRADE",
            ("R1_0214:O4", "R1_0215:O7"),
            ("R1_0214:O12",),
            ("R1_0214:O4", "R1_0215:O7"),
            gate=ProspectiveFalsifierGate(
                gate_id="GATE-H2-NLRC5-VS-STAT1KO",
                unresolved_context_axes=all_axes,
                head_to_head_arms=("NLRC5_Attenuated_SC_Islets", "STAT1_KO_SC_Islets"),
                primary_readout_and_unit="IFN-induced HLA-A/B/C suppression (%) and preserved HLA-E/PD-L1 MFI (%)",
                quantitative_pass_threshold=">=75% HLA-A/B/C suppression with >=80% preserved HLA-E/PD-L1 MFI and NK survival",
            ),
        ),
        record(
            "T1D-CEG-H4-CD55-ARMOR-PRESERVE-CD59",
            "CANDIDATE_UPGRADE",
            ("R1_0098:O8",),
            ("R1_0202:O7",),
            ("R1_0098:O8", "R1_0202:O7"),
            gate=ProspectiveFalsifierGate(
                gate_id="GATE-H4-CD55-ECTOPIC-CD59-UNTOUCHED",
                unresolved_context_axes=all_axes,
                head_to_head_arms=("Ectopic_CD55_Intact_Endogenous_CD59_SC_Islets", "Endogenous_CD59_Edited_SC_Islets"),
                primary_readout_and_unit="Human serum C3b/MAC complement survival (%) and dynamic perifusion GSIS stimulation index",
                quantitative_pass_threshold=">=85% complement survival with >=95% preserved wild-type GSIS stimulation index",
            ),
        ),
        record(
            "T1D-CEG-H5-TXNIP-HYPOXIA-UNCOUPLING",
            "CANDIDATE_UPGRADE",
            ("R1_0215:O27",),
            ("R1_0376:O15",),
            ("R1_0215:O27",),
            gate=ProspectiveFalsifierGate(
                gate_id="GATE-H5-TXNIP-CIS-EDIT",
                unresolved_context_axes=all_axes,
                head_to_head_arms=("TXNIP_Promoter_Edited_SC_Islets", "TXNIP_CodingKO_and_WT_SC_Islets"),
                primary_readout_and_unit="Day-14 post-hypoxic INS+/GCG+ beta-to-alpha ratio (fold)",
                quantitative_pass_threshold=">=2.0-fold Day-14 INS+/GCG+ ratio vs WT under 1% O2 without basal Nrf2 drift",
            ),
        ),
    ]

    invalid_probes: list[tuple[str, SynthesisClaimRecord, str]] = [
        (
            "PROBE_T2D_GENETICS_PROMOTED_TO_T1D_GRAFT",
            record("T1D-BAD-01", "SUPPORTS", ("R1_0340:O2",), (), ("R1_0340:O2",), promote=True),
            "I4_INADMISSIBLE_CONTEXT_TRANSFER",
        ),
        (
            "PROBE_UPGRADE_WITHOUT_FALSIFIER_GATE",
            record("T1D-BAD-02", "CANDIDATE_UPGRADE", ("R1_0215:O7",), (), ()),
            "I4_MISSING_PROSPECTIVE_FALSIFIER_GATE",
        ),
        (
            # The odds ratio 0.62 misread as a 62% decrease; the source reports a 37% decrease.
            "PROBE_ODDS_RATIO_READ_AS_PERCENT_DECREASE",
            record(
                "T1D-BAD-03",
                "SCOPING_QUALIFICATION",
                ("R1_0340:O2",),
                (),
                (),
                prose=(
                    "In R1_0340:O2 the endpoint Type 2 diabetes risk in SLC30A8 LoF heterozygotes (additive model) "
                    "showed a relative decrease of 62 %.",
                ),
            ),
            "I3_TOK_UNGROUNDED_VALUE",
        ),
        (
            "PROBE_PROMOTION_FROM_UNVERIFIED_SOURCE",
            record("T1D-BAD-04", "SUPPORTS", ("R1_0116:O1",), (), ("R1_0116:O1",), promote=True),
            "I5_ILLEGAL_QUANTITATIVE_PROMOTION_WITHOUT_PRIMARY_OR_DIGITIZED_SOURCE",
        ),
    ]

    return DomainVignetteBundle(
        domain_name="t1d_islet_replacement",
        target_context=target_ctx,
        graph=graph,
        initial_corpus=initial_corpus,
        settled_corpus=settled_corpus,
        patch_events=patch_events,
        valid_records=valid_records,
        invalid_probe_records=invalid_probes,
    )


if __name__ == "__main__":
    import argparse
    import json

    parser = argparse.ArgumentParser(description="Verify CEG worked domain vignettes.")
    parser.add_argument(
        "--domain",
        choices=("drug_development", "sport_science", "t1d_islet_replacement", "both"),
        default="both",
    )
    args = parser.parse_args()
    out: dict[str, Any] = {}
    if args.domain in ("drug_development", "both"):
        out["drug_development"] = build_drug_development_vignette().run_all_checks()
    if args.domain in ("sport_science", "both"):
        out["sport_science"] = build_sport_exercise_vignette().run_all_checks()
    if args.domain == "t1d_islet_replacement":
        out["t1d_islet_replacement"] = build_t1d_vignette().run_all_checks()
    print(json.dumps(out, indent=2))
