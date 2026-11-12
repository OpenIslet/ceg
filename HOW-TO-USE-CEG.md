# How to Build and Query a Context–Evidence Graph in Any Field

A **Context–Evidence Graph (CEG)** turns a pile of primary papers into an auditable evidence graph that fits cleanly inside a modern AI context window — without stripping away the experimental details that explain why studies differ.

Nothing in the method is tied to biology or medicine. The same structure applies to clinical trials, oncology, ecology, materials science, chemistry, social science or machine-learning benchmarks.

---

## 1. The Core Idea in Plain English

Most literature tools fail in one of two ways:

1. **Flat text summaries** compress a paper into a single claim (*"Treatment X improved survival"*), dropping the control group, dose, species, measurement unit and timing. When two papers disagree because their conditions differed, a flat summary hallucinates a contradiction or averages incompatible numbers.
2. **Rigid ontologies** try to force every exploratory question through a database schema up front, stifling hypothesis generation.

CEG separates **exploration** from **saving**:

- **Regime 1 — Explore freely in context:** Load the structured graph into an AI assistant's context window. Ask open-ended questions, compare mechanisms across fields, look for hidden trade-offs, and draft candidate hypotheses without rigid templates blocking creative reasoning.
- **Regime 2 — Verify strictly at commit:** Whenever a new study extraction, correction or synthesis claim is saved back to the shared graph, pass it through `ceg_reference.py`. Deterministic checks enforce arm structure, unit compatibility, context transfer rules, source pointers and append-only history.

---

## 2. What Every Study Record Keeps

Instead of one flat summary per paper, extract each primary empirical paper into four linked pieces:

1. **Study metadata (`Study`):** Stable identifier (such as a DOI, PMID or arXiv ID), year and primary-source verification flag.
2. **Experimental and control groups (`Arm`):** Every treatment arm and its matched control arm listed separately (`arm_id`, intervention, dose, sample size `n`).
3. **Context coordinates (`ContextTuple`):** Named axes describing the experimental setting — for example:
   - *Biomedicine:* `model_system` (`immortalized_line`, `murine_in_vivo`, `primary_human_ex_vivo`, `clinical_trial`), `perturbation_regime`, `readout_time_horizon`, `microenvironment`.
   - *Materials science:* `sample_form` (`thin_film`, `bulk_pellet`, `device_stack`), `operating_temperature`, `atmosphere`, `cycling_horizon`.
   - *Machine learning:* `evaluation_split` (`in_domain_val`, `out_of_distribution`, `live_deployment`), `compute_budget`, `prompt_regime`.
4. **Measured outcomes (`EvidenceAtom`):** Endpoint name, compared arm IDs, direction or numeric value, exact measurement unit (checked for SI case sensitivity, e.g. `mM` vs `mm`, `µg` vs `mg`), and a **source pointer** (figure panel, table row or verified verbatim quote).

---

## 3. Why Experimental Settings Form a Partial Order (Not a 1–5 Score)

In real science, two experimental models often answer *different* parts of a question rather than sitting above and below each other on a single ladder:

- A **mouse graft** tests whole-body blood supply and immune recruitment, but in mouse biology.
- **Primary human cells in a dish** test human genetics and receptors, but without blood vessels or organs.

Neither setting dominates the other. In CEG, each context axis is a **partially ordered set (poset)**:

- If source context `c_src` is at least as close to the target along every axis, transfer is `DIRECT` or `MONOTONE_BOUNDED`.
- If one model is stronger on one axis and weaker on another (such as mouse *in vivo* vs human *in vitro*), `evaluate_transfer` classifies the relationship as `INCOMPARABLE` and keeps both pieces of evidence visible side by side rather than averaging them.

---

## 4. Minimal Python Example (`ceg_reference.py`)

Create a script in the repository folder (requires only Python 3.9+):

```python
from ceg_reference import (
    Arm,
    ContextPoset,
    ContextTuple,
    EvidenceAtom,
    Study,
    ContextEvidenceGraph,
    evaluate_transfer,
    verify_atom,
)

# 1. Define a partial order over experimental settings for your domain.
poset = ContextPoset(
    axes={
        "system": {
            "cell_line_dish": set(),
            "mouse_in_vivo": {"cell_line_dish"},
            "human_primary_dish": {"cell_line_dish"},
            "human_clinical": {"mouse_in_vivo", "human_primary_dish", "cell_line_dish"},
        }
    }
)

# 2. Compare two contexts against a clinical target: mouse_in_vivo and human_primary_dish
#    each refine cell_line_dish, yet remain incomparable with each other.
dish_human = ContextTuple({"system": "human_primary_dish"})
mouse_graft = ContextTuple({"system": "mouse_in_vivo"})
clinical = ContextTuple({"system": "human_clinical"})

print("Human dish -> Mouse graft:", evaluate_transfer(poset, dish_human, mouse_graft))
print("Human dish -> Clinical:", evaluate_transfer(poset, dish_human, clinical))
```

Inspect [`domain_vignettes.py`](domain_vignettes.py) for complete end-to-end graphs including:

- `build_antifibrotic_vignette()` — translational drug discovery (`NX-410`),
- `build_sport_science_vignette()` — human performance & exercise physiology,
- `build_t1d_vignette()` — type 1 diabetes islet engineering built from peer-reviewed studies.

---

## 5. Prompt Pattern for Extracting Papers with an LLM

When using Claude, ChatGPT, Gemini or another long-context model to extract primary studies into a Context–Evidence Graph, enforce four rules in the extraction prompt:

1. **Primary empirical studies only** for evidence atoms (keep reviews for background mapping, never as independent empirical counts).
2. **Every arm explicit:** Never record an effect without naming both the active arm and the comparator arm (`A1_treated` vs `A0_vehicle`) and their sample sizes (`n`).
3. **Exact units and timepoints:** Preserve case-sensitive units (`µM`, `mmHg`, `ng/mL`, `days`) and assay horizon (`24 h` vs `14 d`).
4. **Source pin on every outcome:** Require the exact table/figure coordinates (`Fig. 2C`, `Table 1`) and, where licence terms permit, an exact verbatim quote checked against the source text; never guess or paraphrase numbers.

After the model proposes candidate JSON records, run them through `ceg_reference.py` before committing them to the corpus. Any unit mismatch, missing comparator arm, confusable Unicode homoglyph, negated safety bound or broken hash link is flagged deterministically with an explicit violation code.

---

## 6. Scaling Up: Lessons from a 1,151-Study Corpus

Section 6.9 of the paper documents how a production Context–Evidence Graph behaves at corpus scale (`1,151` primary studies, `7,659` arms, `14,105` structured outcomes, `4.17 MB` of canonical `RFC 8785` JSON):

- **Fits in long context or modular slices:** Focused subgraphs for a specific mechanism or design question typically span tens to a few hundred kilobytes, fitting comfortably inside a single prompt window with full arm and unit provenance intact.
- **Corrections never erase history:** When an extraction mistake is caught later, record an append-only `JSONPatch` event (`verify_bijective_patch_replay`) rather than silently overwriting old numbers. Anyone can replay the corpus forward or backward and verify every digest.
- **Separate public rights from verification:** Keep SHA-256 digests and figure/table pointers on every outcome so public releases can cap or redact non-open-access quotes while remaining auditable against PubMed and DOI sources.
