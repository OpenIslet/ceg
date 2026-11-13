#!/usr/bin/env python3
"""Replay the 106-event production curation ledger of the OpenIslet type 1 diabetes corpus.

Two independent replays, then tamper controls, all over the full 1,151-study corpora:

  1. Production convention. Forward replay of the ledger from the first pinned master and reverse
     replay from the last must reproduce the pinned `settled_extractions` payloads exactly, with every
     old-value guard and every per-step study digest holding. The production digest of a study is
     sha256(json.dumps(study, sort_keys=True, ensure_ascii=False)).
  2. Reference implementation. The same ledger, adapted to the corpus shape that
     ceg_reference.verify_bijective_patch_replay expects (a `studies` array whose members carry a
     `study_id`, taken from the production `queue_id`; pointers rewritten from
     /settled_extractions/<i>/... to /studies/<i>/...; study digests recomputed in RFC 8785 form from
     the replayed states), is verified in Layer 1 (structure only) and Layer 2 (digest locks).
  3. Tamper controls through the reference verifier.

Run from this folder (about two minutes):

    PYTHONDONTWRITEBYTECODE=1 python3 -B replay_production_ledger.py --recompute OPENISLET_DIR [--write]
    PYTHONDONTWRITEBYTECODE=1 python3 -B replay_production_ledger.py --check ../CEG-JOURNAL-MANUSCRIPT-V3.2.md

The OpenIslet checkout is read, never written. ceg_reference.py is imported unmodified. The output
holds no timing and no absolute path, so a rerun reproduces it byte for byte. Standard library only.
"""
import sys

sys.dont_write_bytecode = True

import copy
import dataclasses
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
PKG = HERE.parent
sys.path.insert(0, str(PKG))
import ceg_reference as ref  # noqa: E402

STORED = HERE / "production_ledger_replay.json"
S3 = "reports/2026-11-08-stage1-full-3917-screening/stage3_conveyor_rounds"
V1 = f"{S3}/stage3_master_1151_settled_gold_extractions.json"
V4 = f"{S3}/stage3_master_1151_settled_gold_v4_derived.json"
LEDGER = f"{S3}/sol61_v1_to_v4_reversible_patch_ledger.json"


def prod_hash(obj) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def tokens(pointer: str) -> list:
    return [t.replace("~1", "/").replace("~0", "~") for t in pointer.split("/")[1:]]


def walk(doc, toks):
    cur = doc
    for t in toks:
        cur = cur[int(t)] if isinstance(cur, list) else cur[t]
    return cur


def put(doc, pointer: str, value) -> None:
    toks = tokens(pointer)
    parent = walk(doc, toks[:-1])
    if isinstance(parent, list):
        parent[int(toks[-1])] = value
    else:
        parent[toks[-1]] = value


def production_replay(d1: list, d4: list, events: list) -> dict:
    fwd, rev = copy.deepcopy(d1), copy.deepcopy(d4)
    failures = 0

    def study(doc, ev):
        return doc[int(tokens(ev["json_pointer"])[1])]

    for ev in events:
        st = study(fwd, ev)
        failures += st["queue_id"] != ev["queue_id"]
        failures += prod_hash(st) != ev["study_sha256_before"]
        cur = walk({"settled_extractions": fwd}, tokens(ev["json_pointer"]))
        failures += cur != ev["old_value"]
        put({"settled_extractions": fwd}, ev["json_pointer"], copy.deepcopy(ev["new_value"]))
        failures += prod_hash(study(fwd, ev)) != ev["study_sha256_after"]
    for ev in reversed(events):
        st = study(rev, ev)
        failures += prod_hash(st) != ev["study_sha256_after"]
        cur = walk({"settled_extractions": rev}, tokens(ev["json_pointer"]))
        failures += cur != ev["new_value"]
        put({"settled_extractions": rev}, ev["json_pointer"], copy.deepcopy(ev["old_value"]))
        failures += prod_hash(study(rev, ev)) != ev["study_sha256_before"]
    return {
        "events": len(events),
        "patched_studies": len({e["queue_id"] for e in events}),
        "forward_payload_equals_v4": fwd == d4,
        "reverse_payload_equals_v1": rev == d1,
        "guard_or_digest_failures": int(failures),
        "digest_convention": "sha256(json.dumps(study, sort_keys=True, ensure_ascii=False))",
    }


def adapt(arr: list) -> dict:
    return {"studies": [dict(s, study_id=s["queue_id"]) for s in arr]}


def build_events(c0: dict, events: list) -> list:
    work = copy.deepcopy(c0)
    out = []
    for ev in events:
        toks = ev["json_pointer"].split("/")[1:]
        pointer = "/studies/" + "/".join(toks[1:])
        pos = int(toks[1])
        if work["studies"][pos]["study_id"] != ev["queue_id"]:
            raise ValueError(f"ledger event {ev.get('event_index')} points at study position {pos}, which is not {ev['queue_id']}")
        pre = ref.sha256_canonical(work["studies"][pos])
        put(work, pointer, copy.deepcopy(ev["new_value"]))
        post = ref.sha256_canonical(work["studies"][pos])
        out.append(
            ref.ReversiblePatchEvent(
                event_id=f"EV-{ev['event_index']:03d}",
                study_id=ev["queue_id"],
                json_pointer=pointer,
                old_value=copy.deepcopy(ev["old_value"]),
                new_value=copy.deepcopy(ev["new_value"]),
                rationale=ev["reason"],
                study_sha256_before=pre,
                study_sha256_after=post,
            )
        )
    return out


def run(c0, cm, events, layer: int) -> dict:
    try:
        out = ref.verify_bijective_patch_replay(c0, cm, events, check_step_digests=(layer == 2))
        return {"verdict": "ACCEPTED", "events_replayed": out["events_replayed"], "distinct_patched_studies": out["distinct_patched_studies"]}
    except ValueError as exc:
        return {"verdict": "REJECTED", "code": str(exc).split(":")[0].split(" ")[0]}


def first_string_key(d: dict) -> str:
    return sorted(k for k, v in d.items() if isinstance(v, str) and v)[0]


def compute(root: Path) -> dict:
    ledger = json.load(open(root / LEDGER, encoding="utf-8"))
    d1 = json.load(open(root / V1, encoding="utf-8"))["settled_extractions"]
    d4 = json.load(open(root / V4, encoding="utf-8"))["settled_extractions"]
    prod_events = sorted(ledger["events"], key=lambda e: e["event_index"])
    result = {
        "reference_release": "v3.2",
        "reference_sha256": sha256_file(PKG / "ceg_reference.py"),
        "adapter": "study_id := queue_id; pointers /settled_extractions/<i>/... -> /studies/<i>/...; study digests recomputed (RFC 8785) from the replayed states",
        "production_convention_replay": production_replay(d1, d4, prod_events),
    }
    c0, cm = adapt(d1), adapt(d4)
    evs = build_events(c0, prod_events)
    result["reference_digests"] = {
        "initial_corpus_sha256": ref.sha256_canonical(c0),
        "settled_corpus_sha256": ref.sha256_canonical(cm),
        "ledger_sha256": ref.ledger_sha256(evs),
    }
    result["reference_replay"] = {"layer1": run(c0, cm, evs, 1), "layer2": run(c0, cm, evs, 2)}

    controls = []

    def control(cid, text, layer, c0_, cm_, evs_, expect):
        r = run(c0_, cm_, evs_, layer)
        controls.append({"id": cid, "tamper": text, "layer": layer, "expected": expect, **r})

    # C1 delete one event
    k = 52
    control("C1", "delete one event", 1, c0, cm, evs[:k] + evs[k + 1 :], "REJECTED")
    # C2 duplicate one event under a fresh identifier
    dup = dataclasses.replace(evs[k], event_id="EV-DUP")
    control("C2", "duplicate one event under a fresh identifier", 1, c0, cm, evs[: k + 1] + [dup] + evs[k + 1 :], "REJECTED")
    # C3 alter one string in the settled corpus
    cm3 = copy.deepcopy(cm)
    o = cm3["studies"][-1]["outcomes"][0]
    key = first_string_key(o)
    o[key] = o[key] + "x"
    control("C3", "alter one string of the settled corpus", 1, c0, cm3, evs, "REJECTED")
    # C4 alter one string in the initial corpus
    c04 = copy.deepcopy(c0)
    o = c04["studies"][0]["outcomes"][0]
    key = first_string_key(o)
    o[key] = o[key] + "x"
    control("C4", "alter one string of the initial corpus", 1, c04, cm, evs, "REJECTED")
    # C5 redirect one event pointer to a sibling field of the same record
    ev5 = evs[k]
    toks = tokens(ev5.json_pointer)
    parent = walk(c0, toks[:-1])
    sibling = sorted(x for x in parent if x != toks[-1] and parent[x] != ev5.old_value)[0]
    redirected = dataclasses.replace(ev5, json_pointer="/" + "/".join(toks[:-1] + [sibling]))
    control("C5", "redirect one pointer to a sibling field", 1, c0, cm, evs[:k] + [redirected] + evs[k + 1 :], "REJECTED")
    # C6 swap two adjacent events of one study that edit different fields
    swap = next(i for i in range(len(evs) - 1) if evs[i].study_id == evs[i + 1].study_id and evs[i].json_pointer != evs[i + 1].json_pointer)
    swapped = list(evs)
    swapped[swap], swapped[swap + 1] = swapped[swap + 1], swapped[swap]
    control("C6", "transpose two adjacent events of one study that edit different fields", 1, c0, cm, swapped, "ACCEPTED")
    control("C6", "transpose two adjacent events of one study that edit different fields", 2, c0, cm, swapped, "REJECTED")
    # C7 reword one rationale (replay-neutral)
    reworded = dataclasses.replace(evs[k], rationale=evs[k].rationale + " (reworded)")
    evs7 = evs[:k] + [reworded] + evs[k + 1 :]
    control("C7", "reword one rationale", 2, c0, cm, evs7, "ACCEPTED")
    controls[-1]["ledger_digest_changed"] = ref.ledger_sha256(evs7) != result["reference_digests"]["ledger_sha256"]
    # C8 overwrite one per-step digest (values untouched)
    forged = dataclasses.replace(evs[k], study_sha256_after="0" * 64)
    evs8 = evs[:k] + [forged] + evs[k + 1 :]
    control("C8", "overwrite one per-step study digest", 1, c0, cm, evs8, "ACCEPTED")
    control("C8", "overwrite one per-step study digest", 2, c0, cm, evs8, "REJECTED")
    result["controls"] = controls
    result["controls_as_expected"] = sum(c["verdict"] == c["expected"] for c in controls)
    result["controls_total"] = len(controls)
    return result


WORDS = {8: "eight", 10: "ten"}


def statements(r: dict) -> list:
    p, rr = r["production_convention_replay"], r["reference_replay"]
    ctl = r["controls"]
    structural = [c for c in ctl if c["id"] in ("C1", "C2", "C3", "C4", "C5")]
    return [
        f"{p['events']} reversible events",
        f"{rr['layer1']['distinct_patched_studies']} patched studies",
        "no old-value guard or study-digest failure",
        "accepts the same ledger",
        f"{sum(c['verdict'] == 'REJECTED' for c in structural)} of {len(structural)} structural tampers",
        f"{sum(c['verdict'] == 'REJECTED' for c in structural)}/{len(structural)} structural tampers",
        f"{WORDS[len({c['id'] for c in ctl})].capitalize()} kinds of tamper, run as {WORDS[len(ctl)]} controls",
    ]


def main(argv: list) -> int:
    if len(argv) >= 3 and argv[1] == "--check":
        r = json.load(open(STORED, encoding="utf-8"))
        text = Path(argv[2]).read_text(encoding="utf-8")
        bad = []
        p, rr = r["production_convention_replay"], r["reference_replay"]
        if not (p["forward_payload_equals_v4"] and p["reverse_payload_equals_v1"] and p["guard_or_digest_failures"] == 0):
            bad.append("stored production replay is not clean")
        if not (rr["layer1"]["verdict"] == "ACCEPTED" and rr["layer2"]["verdict"] == "ACCEPTED"):
            bad.append("stored reference replay is not accepted in both layers")
        if r["controls_as_expected"] != r["controls_total"]:
            bad.append("a stored control did not behave as expected")
        bad += [f"manuscript does not state: {s}" for s in statements(r) if s not in text]
        if bad:
            for b in bad:
                print(f"PRODUCTION LEDGER MISMATCH: {b}")
            return 1
        print(f"PRODUCTION LEDGER OK: {len(statements(r))} statements in {Path(argv[2]).name} match {STORED.name}")
        return 0
    if len(argv) >= 3 and argv[1] == "--recompute":
        root = Path(argv[2]).expanduser().resolve()
        r = compute(root)
        text = json.dumps(r, indent=2, sort_keys=True) + "\n"
        if "--write" in argv:
            STORED.write_text(text, encoding="utf-8")
            print(f"wrote {STORED.name}: controls as expected {r['controls_as_expected']}/{r['controls_total']}")
            return 0
        same = STORED.exists() and STORED.read_text(encoding="utf-8") == text
        print(f"recomputed results {'match' if same else 'DIFFER FROM'} {STORED.name}")
        return 0 if same else 1
    print(__doc__)
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv))
