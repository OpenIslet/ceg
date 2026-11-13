#!/usr/bin/env python3
"""Reference implementation of Context-Evidence Graphs (CEGs), version 3.1.

Zero-dependency Python 3 (3.9+) implementation of the structures and checks
defined in the accompanying manuscript:

  I1  Referential closure, atom well-formedness, context-poset validity and
      full-schema atom-to-corpus binding.
  I2  Reversible, state-guarded curation patch ledger over strict RFC 6901
      JSON pointers and an RFC 8785 (JCS) canonical form restricted to
      float-free documents.
  I3  Proposition grounding in three tiers:
        I3_tok    role-typed, Unicode- and homoglyph-hardened numeral grounding,
        I3_clause two-level citation scoping, parenthetical/range unit, arm-order
                  and polarity checks over synthesis prose (a lint, not a proof),
        I3_typed  exact field-wise equality between typed propositions and
                  evidence atoms, with promotion coverage.
  I4  Context-transfer classification on a finite product poset, with unconditional
      target-context and prospective-falsifier gate validation.
  I5  Primary-source and figure-digitization promotion gate.
  I6  Non-vacuity, positive-claim support requirement and support/counter-evidence
      disjointness.

The prose tiers are deliberately conservative heuristics with measured false
acceptances and false rejections (see the synthetic and real-text benchmarks).
Guarantees attach to the typed layer and to the ledger, not to free prose.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from decimal import Decimal, InvalidOperation
from functools import lru_cache
import hashlib
import json
from pathlib import Path
import re
import sys
import types
from typing import Any, Literal, Mapping, Sequence
import unicodedata

REFERENCE_VERSION = "3.1"


# ---------------------------------------------------------------------------
# Canonical JSON (RFC 8785 restricted to float-free documents) and hashing
# ---------------------------------------------------------------------------

_MAX_SAFE_INTEGER = 2**53 - 1


def _jcs_string(s: str) -> str:
    out = ['"']
    for ch in s:
        cp = ord(ch)
        if ch == '"':
            out.append('\\"')
        elif ch == "\\":
            out.append("\\\\")
        elif ch == "\b":
            out.append("\\b")
        elif ch == "\f":
            out.append("\\f")
        elif ch == "\n":
            out.append("\\n")
        elif ch == "\r":
            out.append("\\r")
        elif ch == "\t":
            out.append("\\t")
        elif cp < 0x20:
            out.append(f"\\u{cp:04x}")
        elif 0xD800 <= cp <= 0xDFFF:
            raise ValueError("CANONICAL_JSON_LONE_SURROGATE")
        else:
            out.append(ch)
    out.append('"')
    return "".join(out)


def _jcs(obj: Any) -> str:
    if obj is None:
        return "null"
    if obj is True:
        return "true"
    if obj is False:
        return "false"
    if type(obj) is int:
        if abs(obj) > _MAX_SAFE_INTEGER:
            raise ValueError(f"CANONICAL_JSON_INTEGER_OUT_OF_RANGE: {obj}")
        return str(obj)
    if isinstance(obj, float):
        raise ValueError("CANONICAL_JSON_FLOAT_FORBIDDEN: carry non-integers as decimal strings")
    if type(obj) is str:
        return _jcs_string(obj)
    if isinstance(obj, (list, tuple)):
        return "[" + ",".join(_jcs(x) for x in obj) + "]"
    if isinstance(obj, dict):
        encoded_items: list[tuple[bytes, str, Any]] = []
        for k, v in obj.items():
            if type(k) is not str:
                raise ValueError(f"CANONICAL_JSON_NON_STRING_KEY: {k!r}")
            kj = _jcs_string(k)
            encoded_items.append((k.encode("utf-16-be"), kj, v))
        encoded_items.sort(key=lambda item: item[0])
        return "{" + ",".join(kj + ":" + _jcs(v) for _, kj, v in encoded_items) + "}"
    raise ValueError(f"CANONICAL_JSON_UNSUPPORTED_TYPE: {type(obj).__name__}")


def canonical_json_bytes(obj: Any) -> bytes:
    """RFC 8785 canonical UTF-8 bytes for float-free JSON values.

    Object keys are ordered by UTF-16 code units as RFC 8785 requires. Floats
    are rejected so that no number-serialization rule is ever exercised:
    non-integer quantities are carried as decimal strings.
    """
    return _jcs(obj).encode("utf-8")


def sha256_canonical(obj: Any) -> str:
    """SHA-256 hex digest of the canonical JSON form."""
    return hashlib.sha256(canonical_json_bytes(obj)).hexdigest()


def _json_equal(a: Any, b: Any) -> bool:
    """Type-strict JSON equality (True != 1, "1" != 1)."""
    return canonical_json_bytes(a) == canonical_json_bytes(b)


def _json_clone(obj: Any) -> Any:
    """Independent copy as a JSON tree.

    A round trip through the canonical form removes any sharing of sub-objects
    between studies, so an edit to one study can never reach another through an
    aliased container (copy.deepcopy would preserve such sharing).
    """
    return json.loads(canonical_json_bytes(obj).decode("utf-8"))


# ---------------------------------------------------------------------------
# Zero-cache source-pinned module loading, single-read JSON & package hygiene
# ---------------------------------------------------------------------------

def _reject_nonfinite_json_constant(token: str) -> Any:
    raise ValueError(f"NON_FINITE_JSON_CONSTANT: {token!r}")


def _resolve_contained_path(root_dir: str | Path, rel_path: str | Path) -> Path:
    """Resolve `rel_path` strictly inside `root_dir`, rejecting absolute, `..` or symlink traversal."""
    root = Path(root_dir)
    if root.is_symlink():
        raise ValueError(f"HYGIENE_PATH_OR_SYMLINK_TRAVERSAL: symlink root {str(root)!r}")
    root_resolved = root.resolve()
    rel = Path(rel_path)
    if not str(rel) or rel.is_absolute() or ".." in rel.parts:
        raise ValueError(f"HYGIENE_PATH_OR_SYMLINK_TRAVERSAL: {str(rel)!r}")
    cur = root_resolved
    for part in rel.parts:
        cur = cur / part
        if cur.is_symlink():
            raise ValueError(f"HYGIENE_PATH_OR_SYMLINK_TRAVERSAL: symlink component {str(cur)!r}")
    resolved = cur.resolve()
    try:
        resolved.relative_to(root_resolved)
    except ValueError as exc:
        raise ValueError(f"HYGIENE_PATH_OR_SYMLINK_TRAVERSAL: escaped root {str(rel)!r}") from exc
    return resolved


def load_json_single_read(
    path: str | Path,
    *,
    expected_sha256: str | None = None,
    root_dir: str | Path | None = None,
) -> tuple[Any, str, int]:
    """Read a UTF-8 JSON file once (`read_bytes -> sha256 -> json.loads`), rejecting NaN/Infinity."""
    target = _resolve_contained_path(root_dir, path) if root_dir is not None else Path(path)
    if root_dir is None and target.is_symlink():
        raise ValueError(f"HYGIENE_PATH_OR_SYMLINK_TRAVERSAL: symlink target {str(target)!r}")
    raw_bytes = target.read_bytes()
    actual_sha256 = hashlib.sha256(raw_bytes).hexdigest()
    if expected_sha256 is not None and actual_sha256 != expected_sha256:
        raise ValueError(
            f"JSON_SHA256_MISMATCH: expected {expected_sha256}, got {actual_sha256} for {target.name!r}"
        )
    try:
        text = raw_bytes.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError(f"INVALID_UTF8_JSON_BYTES: {target.name!r}") from exc
    parsed = json.loads(text, parse_constant=_reject_nonfinite_json_constant)
    return parsed, actual_sha256, len(raw_bytes)


def load_source_pinned_module(
    module_name: str,
    source_path: str | Path,
    *,
    expected_sha256: str | None = None,
    root_dir: str | Path | None = None,
) -> types.ModuleType:
    """Load a local Python module directly from SHA-256-verified source bytes.

    Compiles `raw_bytes` via `compile(raw_bytes, str(path), "exec")` with
    `mod.__cached__ = None` rather than `importlib.util.spec_from_file_location`
    or `importlib.reload`, eliminating `.pyc` cache lookup even when Python runs
    without `-B` under a poisoned `PYTHONPYCACHEPREFIX`.
    """
    target = _resolve_contained_path(root_dir, source_path) if root_dir is not None else Path(source_path)
    if root_dir is None and target.is_symlink():
        raise ValueError(f"HYGIENE_PATH_OR_SYMLINK_TRAVERSAL: symlink module {str(target)!r}")
    target = target.resolve()
    raw_bytes = target.read_bytes()
    actual_sha256 = hashlib.sha256(raw_bytes).hexdigest()
    if expected_sha256 is not None and actual_sha256 != expected_sha256:
        raise ValueError(
            f"SOURCE_MODULE_SHA256_MISMATCH: expected {expected_sha256}, got {actual_sha256} for {target.name!r}"
        )
    raw_bytes.decode("utf-8")
    code = compile(raw_bytes, str(target), "exec", dont_inherit=True)
    mod = types.ModuleType(module_name)
    mod.__file__ = str(target)
    mod.__cached__ = None
    mod.__loader__ = None
    mod.__package__ = ""
    mod.__dict__["__source_sha256__"] = actual_sha256
    sys.modules[module_name] = mod
    exec(code, mod.__dict__)
    return mod


def verify_chunk_or_package_hygiene(
    root_dir: str | Path,
    manifest_entries: Sequence[Mapping[str, Any]],
    *,
    byte_budget: int,
    token_budget: int | None = None,
) -> list[str]:
    """Verify package/chunk manifest hygiene without bare `assert` (identical under `python3 -O`).

    Rejects non-positive byte/token budgets, path or symlink traversal outside
    `root_dir`, `utf8_bytes` drift against the single-read byte buffer, SHA-256
    drift, invalid UTF-8, and `NaN`/`Infinity` JSON constants.
    """
    errs: list[str] = []
    if type(byte_budget) is not int or byte_budget <= 0:
        errs.append(f"HYGIENE_NON_POSITIVE_BUDGET: byte_budget={byte_budget!r}")
    if token_budget is not None and (type(token_budget) is not int or token_budget <= 0):
        errs.append(f"HYGIENE_NON_POSITIVE_BUDGET: token_budget={token_budget!r}")
    if errs:
        return errs

    for idx, entry in enumerate(manifest_entries):
        rel = entry.get("path")
        if not isinstance(rel, str) or not rel:
            errs.append(f"HYGIENE_INVALID_ENTRY_PATH[{idx}]: {rel!r}")
            continue
        try:
            target = _resolve_contained_path(root_dir, rel)
        except ValueError as exc:
            errs.append(str(exc))
            continue
        if not target.is_file():
            errs.append(f"HYGIENE_MISSING_FILE: {rel!r}")
            continue
        raw = target.read_bytes()
        actual_len = len(raw)
        if actual_len > byte_budget:
            errs.append(f"HYGIENE_BYTE_BUDGET_EXCEEDED ({rel}): {actual_len} > {byte_budget}")
        expected_bytes = entry.get("utf8_bytes")
        if expected_bytes is not None:
            if type(expected_bytes) is not int or expected_bytes < 0 or expected_bytes != actual_len:
                errs.append(
                    f"HYGIENE_UTF8_BYTES_DRIFT ({rel}): declared {expected_bytes!r} != actual {actual_len}"
                )
        expected_sha = entry.get("sha256")
        actual_sha = hashlib.sha256(raw).hexdigest()
        if expected_sha is not None and expected_sha != actual_sha:
            errs.append(f"HYGIENE_SHA256_MISMATCH ({rel}): declared {expected_sha!r} != actual {actual_sha}")
        try:
            text = raw.decode("utf-8")
        except UnicodeDecodeError:
            errs.append(f"HYGIENE_INVALID_UTF8 ({rel})")
            continue
        if token_budget is not None:
            approx_tokens = len(text.split())
            if approx_tokens > token_budget:
                errs.append(f"HYGIENE_TOKEN_BUDGET_EXCEEDED ({rel}): {approx_tokens} > {token_budget}")
        if entry.get("require_json", rel.endswith(".json")):
            try:
                json.loads(text, parse_constant=_reject_nonfinite_json_constant)
            except ValueError as exc:
                errs.append(f"HYGIENE_NON_FINITE_OR_INVALID_JSON ({rel}): {exc}")
    return errs


# ---------------------------------------------------------------------------
# Strict RFC 6901 pointers and the reversible patch ledger (Invariant I2)
# ---------------------------------------------------------------------------

_ARRAY_INDEX_RE = re.compile(r"0|[1-9][0-9]*")
_BAD_ESCAPE_RE = re.compile(r"~(?![01])")


def parse_json_pointer(pointer: str) -> list[str]:
    """Parse a non-empty RFC 6901 pointer into reference tokens.

    The empty pointer (whole document) is rejected because a ledger event
    replaces one location inside a parent container.
    """
    if not isinstance(pointer, str) or not pointer.startswith("/"):
        raise ValueError(f"I2_INVALID_JSON_POINTER: {pointer!r}")
    tokens: list[str] = []
    for raw in pointer[1:].split("/"):
        if _BAD_ESCAPE_RE.search(raw):
            raise ValueError(f"I2_INVALID_JSON_POINTER_ESCAPE: {pointer!r}")
        tokens.append(raw.replace("~1", "/").replace("~0", "~"))
    return tokens


def resolve_json_pointer(doc: Any, pointer: str) -> tuple[Any, str | int]:
    """Resolve a pointer to (parent_container, key_or_index) of an existing location.

    Array indices must be canonical ("0" or digits without a leading zero), so
    each existing location has exactly one pointer.
    """
    tokens = parse_json_pointer(pointer)
    cur = doc
    for i, tok in enumerate(tokens):
        is_last = i == len(tokens) - 1
        if isinstance(cur, list):
            if not _ARRAY_INDEX_RE.fullmatch(tok):
                raise ValueError(f"I2_INVALID_ARRAY_INDEX: {tok!r} in {pointer!r}")
            idx = int(tok)
            if idx >= len(cur):
                raise ValueError(f"I2_POINTER_TARGET_MISSING: {pointer!r}")
            if is_last:
                return cur, idx
            cur = cur[idx]
        elif isinstance(cur, dict):
            if tok not in cur:
                raise ValueError(f"I2_POINTER_TARGET_MISSING: {pointer!r}")
            if is_last:
                return cur, tok
            cur = cur[tok]
        else:
            raise ValueError(f"I2_POINTER_THROUGH_SCALAR: {pointer!r}")
    raise ValueError(f"I2_INVALID_JSON_POINTER: {pointer!r}")


@dataclass(frozen=True)
class ReversiblePatchEvent:
    """One invertible curation edit: replace the value at a pointer inside one study."""

    event_id: str
    study_id: str
    json_pointer: str
    old_value: Any
    new_value: Any
    rationale: str
    study_sha256_before: str
    study_sha256_after: str


def ledger_sha256(events: Sequence[ReversiblePatchEvent]) -> str:
    """Digest of the whole ledger, rationales and identifiers included.

    Replay binds the corpora to the edits. It does not bind an event's rationale
    or identifier, and it accepts an inserted edit-and-revert pair. Publishing
    this digest beside the two corpus digests pins those as well.
    """
    return sha256_canonical(
        [
            {
                "event_id": e.event_id,
                "study_id": e.study_id,
                "json_pointer": e.json_pointer,
                "old_value": e.old_value,
                "new_value": e.new_value,
                "rationale": e.rationale,
                "study_sha256_before": e.study_sha256_before,
                "study_sha256_after": e.study_sha256_after,
            }
            for e in events
        ]
    )


def _study_positions(corpus: Mapping[str, Any]) -> dict[str, int]:
    studies = corpus.get("studies") if isinstance(corpus, Mapping) else None
    if not isinstance(studies, list):
        raise ValueError("I2_CORPUS_MISSING_STUDIES_ARRAY")
    pos: dict[str, int] = {}
    for i, st in enumerate(studies):
        sid = st.get("study_id") if isinstance(st, dict) else None
        if not isinstance(sid, str):
            raise ValueError(f"I2_STUDY_WITHOUT_ID: /studies/{i}")
        if sid in pos:
            raise ValueError(f"I2_DUPLICATE_STUDY_ID: {sid}")
        pos[sid] = i
    return pos


def _check_event_scope(ev: ReversiblePatchEvent, study_pos: int) -> None:
    """Bind the event's pointer to the study whose digests guard it."""
    tokens = parse_json_pointer(ev.json_pointer)
    if len(tokens) < 3 or tokens[0] != "studies" or tokens[1] != str(study_pos):
        raise ValueError(
            f"I2_POINTER_OUTSIDE_DECLARED_STUDY: {ev.event_id} declares {ev.study_id} (/studies/{study_pos}) "
            f"but points at {ev.json_pointer!r}"
        )
    if tokens[2:] == ["study_id"]:
        raise ValueError(f"I2_STUDY_ID_IMMUTABLE: {ev.event_id}")


def verify_bijective_patch_replay(
    initial_corpus: dict[str, Any],
    settled_corpus: dict[str, Any],
    events: Sequence[ReversiblePatchEvent],
    *,
    check_step_digests: bool = True,
) -> dict[str, Any]:
    """Verify forward replay D_0 -> D_m and exact inverse replay D_m -> D_0 (I2).

    With `check_step_digests=False`, structural projection closure still verifies
    every pointer, old_value -> new_value transition, and exact equality of the
    replayed corpora against `settled_corpus` and `initial_corpus`; with
    `check_step_digests=True` (default), per-step study SHA-256 guards are
    checked as an independent cryptographic layer. In both modes the event
    identifier, study identifier, pointer and rationale must be free of
    forbidden Unicode.
    Raises ValueError with an I2_* code on the first violation.
    """
    seen_ids: set[str] = set()
    for ev in events:
        if ev.event_id in seen_ids:
            raise ValueError(f"I2_DUPLICATE_EVENT_ID: {ev.event_id}")
        seen_ids.add(ev.event_id)
        if _json_equal(ev.old_value, ev.new_value):
            raise ValueError(f"I2_NOOP_EVENT: {ev.event_id}")
        # Ledger strings are screened like every other structured string (release 3.1).
        for label, text in (
            ("event_id", ev.event_id),
            ("study_id", ev.study_id),
            ("json_pointer", ev.json_pointer),
            ("rationale", ev.rationale),
        ):
            if isinstance(text, str) and contains_forbidden_unicode(text):
                raise ValueError(f"I2_FORBIDDEN_UNICODE_IN_LEDGER_STRING: {ev.event_id!r} ({label})")

    work = _json_clone(initial_corpus)
    pos = _study_positions(work)
    for ev in events:
        if ev.study_id not in pos:
            raise ValueError(f"I2_UNKNOWN_STUDY_IN_PATCH: {ev.event_id} -> {ev.study_id}")
        _check_event_scope(ev, pos[ev.study_id])
        st = work["studies"][pos[ev.study_id]]
        if check_step_digests and sha256_canonical(st) != ev.study_sha256_before:
            raise ValueError(f"I2_PRE_HASH_MISMATCH: {ev.event_id}")
        parent, key = resolve_json_pointer(work, ev.json_pointer)
        if not _json_equal(parent[key], ev.old_value):
            raise ValueError(f"I2_OLD_VALUE_MISMATCH: {ev.event_id}")
        parent[key] = _json_clone(ev.new_value)
        if check_step_digests and sha256_canonical(st) != ev.study_sha256_after:
            raise ValueError(f"I2_POST_HASH_MISMATCH: {ev.event_id}")

    if not _json_equal(work, settled_corpus):
        raise ValueError("I2_FORWARD_SETTLED_DIGEST_MISMATCH")
    expected_settled_sha = sha256_canonical(settled_corpus)

    rev_work = _json_clone(settled_corpus)
    rev_pos = _study_positions(rev_work)
    for ev in reversed(events):
        if ev.study_id not in rev_pos:
            raise ValueError(f"I2_UNKNOWN_STUDY_IN_PATCH: {ev.event_id} -> {ev.study_id}")
        _check_event_scope(ev, rev_pos[ev.study_id])
        st = rev_work["studies"][rev_pos[ev.study_id]]
        if check_step_digests and sha256_canonical(st) != ev.study_sha256_after:
            raise ValueError(f"I2_REVERSE_POST_HASH_MISMATCH: {ev.event_id}")
        parent, key = resolve_json_pointer(rev_work, ev.json_pointer)
        if not _json_equal(parent[key], ev.new_value):
            raise ValueError(f"I2_REVERSE_NEW_VALUE_MISMATCH: {ev.event_id}")
        parent[key] = _json_clone(ev.old_value)
        if check_step_digests and sha256_canonical(st) != ev.study_sha256_before:
            raise ValueError(f"I2_REVERSE_PRE_HASH_MISMATCH: {ev.event_id}")

    if not _json_equal(rev_work, initial_corpus):
        raise ValueError("I2_INVERSE_INITIAL_DIGEST_MISMATCH")
    expected_initial_sha = sha256_canonical(initial_corpus)

    return {
        "events_replayed": len(events),
        "distinct_patched_studies": len({e.study_id for e in events}),
        "initial_sha256": expected_initial_sha,
        "settled_sha256": expected_settled_sha,
        "ledger_sha256": ledger_sha256(events),
        "step_digests_checked": check_step_digests,
        "bijective_replay_verified": True,
    }


# ---------------------------------------------------------------------------
# Product context poset (C, <=_C) and transfer classification (Invariant I4)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ContextAxisPoset:
    """Finite partially ordered axis given by the down-set of each element."""

    name: str
    # element -> every element it dominates, itself included (reflexive-transitive closure)
    dominates: Mapping[str, frozenset[str]]

    def validate(self) -> None:
        """Check that `dominates` really is a partial order written as down-sets."""
        elems = set(self.dominates)
        for e, down in self.dominates.items():
            if e not in down:
                raise ValueError(f"I4_POSET_NOT_REFLEXIVE ({self.name}): {e!r}")
            undeclared = sorted(set(down) - elems)
            if undeclared:
                raise ValueError(f"I4_POSET_UNDECLARED_ELEMENT ({self.name}): {undeclared}")
        for e, down in self.dominates.items():
            for x in down:
                if x != e and e in self.dominates[x]:
                    raise ValueError(f"I4_POSET_NOT_ANTISYMMETRIC ({self.name}): {x!r} ~ {e!r}")
                if not set(self.dominates[x]) <= set(down):
                    raise ValueError(f"I4_POSET_NOT_TRANSITIVE ({self.name}): below {x!r} but not below {e!r}")

    def leq(self, lower: str, higher: str) -> bool:
        """True iff `lower` <=_k `higher`."""
        if lower not in self.dominates or higher not in self.dominates:
            raise ValueError(f"I4_UNKNOWN_AXIS_ELEMENT ({self.name}): {lower!r} or {higher!r}")
        return lower in self.dominates[higher]


@dataclass(frozen=True)
class ContextSignature:
    """Multi-axis experimental or clinical context c in C."""

    axes: Mapping[str, str]

    def __post_init__(self) -> None:
        object.__setattr__(self, "axes", dict(sorted(self.axes.items())))

    def get(self, axis: str) -> str:
        return self.axes[axis]


TransferClass = Literal["DIRECT_MATCH", "BOUNDED_BRIDGE", "INADMISSIBLE_TRANSFER"]


@dataclass(frozen=True)
class ContextLatticeSpec:
    """Product partial order over named context axes, with optional bridge floors."""

    axis_posets: Mapping[str, ContextAxisPoset]
    bridge_floor: Mapping[str, str] = field(default_factory=dict)

    def validate(self) -> None:
        for name, poset in self.axis_posets.items():
            if poset.name != name:
                raise ValueError(f"I4_AXIS_NAME_MISMATCH: {name!r} vs {poset.name!r}")
            poset.validate()
        for axis, floor in self.bridge_floor.items():
            if axis not in self.axis_posets:
                raise ValueError(f"I4_BRIDGE_FLOOR_ON_UNKNOWN_AXIS: {axis!r}")
            if floor not in self.axis_posets[axis].dominates:
                raise ValueError(f"I4_BRIDGE_FLOOR_UNKNOWN_ELEMENT ({axis}): {floor!r}")

    def check_signature(self, sig: ContextSignature) -> None:
        if set(sig.axes) != set(self.axis_posets):
            raise ValueError("I4_CONTEXT_SIGNATURE_AXIS_MISMATCH")
        for axis, val in sig.axes.items():
            if val not in self.axis_posets[axis].dominates:
                raise ValueError(f"I4_UNKNOWN_AXIS_ELEMENT ({axis}): {val!r}")

    def has_common_upper_bound(self, axis: str, a: str, b: str) -> bool:
        poset = self.axis_posets[axis]
        return any(a in down and b in down for down in poset.dominates.values())

    def evaluate_transfer(
        self,
        source: ContextSignature,
        target: ContextSignature,
    ) -> tuple[TransferClass, tuple[str, ...]]:
        """Classify the transfer from a source context to a target context.

        U = {k : target_k is not <=_k source_k} is the unresolved axis set.
          DIRECT_MATCH           U is empty (target <=_C source).
          INADMISSIBLE_TRANSFER  some k in U has no common upper bound of
                                 (source_k, target_k), or has a bridge floor
                                 that source_k does not reach.
          BOUNDED_BRIDGE         U is non-empty and no axis in U is inadmissible.
        Floors and common-upper-bound tests apply to unresolved axes only: an
        axis on which the source already covers the target needs no bridge.
        """
        self.check_signature(source)
        self.check_signature(target)
        unresolved: list[str] = []
        inadmissible = False
        for ax_name, poset in sorted(self.axis_posets.items()):
            s_val = source.axes[ax_name]
            t_val = target.axes[ax_name]
            if poset.leq(t_val, s_val):
                continue
            unresolved.append(ax_name)
            if not self.has_common_upper_bound(ax_name, s_val, t_val):
                inadmissible = True
            if ax_name in self.bridge_floor and not poset.leq(self.bridge_floor[ax_name], s_val):
                inadmissible = True
        if not unresolved:
            return "DIRECT_MATCH", ()
        if inadmissible:
            return "INADMISSIBLE_TRANSFER", tuple(unresolved)
        return "BOUNDED_BRIDGE", tuple(unresolved)


# ---------------------------------------------------------------------------
# Quantity sum type, multi-arm contrasts and evidence atoms
# ---------------------------------------------------------------------------

QuantityKind = Literal[
    "point_level",
    "relative_change",
    "interval_bound",
    "censored_observation",
    "missing_numeric",
]

RelationOp = Literal["=", "≈", "<", "<=", ">", ">="]
ChangeDirection = Literal["increase", "decrease", "neutral"]


_DECIMAL_LITERAL_RE = re.compile(r"-?(?:0|[1-9][0-9]*)(?:\.[0-9]+)?")
RELATION_OPS = ("=", "≈", "<", "<=", ">", ">=")
STATISTIC_KINDS = (
    "mean",
    "median",
    "proportion",
    "rate",
    "rate_ratio",
    "hazard_ratio",
    "odds_ratio",
    "difference",
    "fold_change",
    "qualitative",
)
ATOM_POLARITIES = ("favorable", "unfavorable", "neutral_or_mixed", "unresolved")
CONTRAST_KINDS = ("single_arm", "pairwise", "multi_arm_joint")
EPISTEMIC_ROLES = ("SUPPORTS", "CONTRADICTS", "SCOPING_QUALIFICATION", "CANDIDATE_UPGRADE", "FALSIFIED_GRAVEYARD")


def _parse_decimal_strict(val: str) -> Decimal:
    """Plain ASCII decimal literal: optional minus, digits, optional fraction. No exponent, sign or padding."""
    if not isinstance(val, str) or not _DECIMAL_LITERAL_RE.fullmatch(val):
        raise ValueError(f"NON_NUMERIC_QUANTITY_LITERAL: {val!r}")
    try:
        return Decimal(val)
    except InvalidOperation as exc:  # unreachable for a literal that matched
        raise ValueError(f"NON_NUMERIC_QUANTITY_LITERAL: {val!r}") from exc


@dataclass(frozen=True)
class QuantitySpec:
    """Typed quantity: the tag `kind` selects which fields are meaningful (disjoint union)."""

    kind: QuantityKind
    relation: RelationOp | None = None
    value: str | None = None
    lower: str | None = None
    upper: str | None = None
    unit: str | None = None
    unit_dimension: str | None = None
    change_direction: ChangeDirection | None = None
    missing_reason: str | None = None

    def validate(self) -> None:
        if self.kind == "point_level":
            if (
                self.relation not in ("=", "≈")
                or self.value is None
                or not (self.unit and self.unit.strip())
                or not (self.unit_dimension and self.unit_dimension.strip())
            ):
                raise ValueError(f"INVALID_POINT_LEVEL_QUANTITY: {self}")
            if self.change_direction is not None:
                raise ValueError("POINT_LEVEL_CANNOT_CARRY_RELATIVE_CHANGE_DIRECTION")
            if any(x is not None for x in (self.lower, self.upper, self.missing_reason)):
                raise ValueError(f"POINT_LEVEL_FORBIDDEN_CROSS_VARIANT_FIELD: {self}")
            _parse_decimal_strict(self.value)
        elif self.kind == "relative_change":
            if self.relation not in RELATION_OPS or self.value is None or self.unit not in ("%", "fold", "ratio", "log2_fc"):
                raise ValueError(f"INVALID_RELATIVE_CHANGE_QUANTITY: {self}")
            if self.change_direction not in ("increase", "decrease", "neutral"):
                raise ValueError(f"RELATIVE_CHANGE_REQUIRES_EXPLICIT_SIGN: {self}")
            if any(x is not None for x in (self.lower, self.upper, self.missing_reason)):
                raise ValueError(f"RELATIVE_CHANGE_FORBIDDEN_CROSS_VARIANT_FIELD: {self}")
            if self.unit_dimension is not None and not self.unit_dimension.strip():
                raise ValueError(f"EMPTY_UNIT_DIMENSION: {self}")
            _parse_decimal_strict(self.value)
        elif self.kind == "interval_bound":
            if (
                self.lower is None
                or self.upper is None
                or not (self.unit and self.unit.strip())
                or not (self.unit_dimension and self.unit_dimension.strip())
            ):
                raise ValueError(f"INVALID_INTERVAL_QUANTITY: {self}")
            if any(x is not None for x in (self.relation, self.value, self.change_direction, self.missing_reason)):
                raise ValueError(f"INTERVAL_BOUND_FORBIDDEN_CROSS_VARIANT_FIELD: {self}")
            lo = _parse_decimal_strict(self.lower)
            hi = _parse_decimal_strict(self.upper)
            if lo > hi:
                raise ValueError(f"INVERTED_INTERVAL_BOUNDS: {self.lower} > {self.upper}")
        elif self.kind == "censored_observation":
            if self.relation not in (">", ">=", "<", "<=") or self.value is None or not (self.unit and self.unit.strip()):
                raise ValueError(f"INVALID_CENSORED_QUANTITY: {self}")
            if any(x is not None for x in (self.lower, self.upper, self.change_direction, self.missing_reason)):
                raise ValueError(f"CENSORED_FORBIDDEN_CROSS_VARIANT_FIELD: {self}")
            if self.unit_dimension is not None and not self.unit_dimension.strip():
                raise ValueError(f"EMPTY_UNIT_DIMENSION: {self}")
            _parse_decimal_strict(self.value)
        elif self.kind == "missing_numeric":
            if any(
                x is not None
                for x in (
                    self.relation,
                    self.value,
                    self.lower,
                    self.upper,
                    self.unit,
                    self.unit_dimension,
                    self.change_direction,
                )
            ):
                raise ValueError(f"MISSING_NUMERIC_MUST_NOT_CARRY_VALUE: {self}")
            if not (self.missing_reason and self.missing_reason.strip()):
                raise ValueError("MISSING_NUMERIC_REQUIRES_REASON")
        else:
            raise ValueError(f"UNKNOWN_QUANTITY_KIND: {self.kind!r}")


@dataclass(frozen=True)
class MultiArmContrast:
    """Arm-role binding that must account for every arm the source comparison names."""

    focal_arms: tuple[str, ...]
    comparator_arms: tuple[str, ...]
    contrast_kind: Literal["single_arm", "pairwise", "multi_arm_joint"]

    def validate(self, study_variable_ids: set[str], raw_mentioned_arms: Sequence[str]) -> None:
        all_bound = list(self.focal_arms) + list(self.comparator_arms)
        if not self.focal_arms:
            raise ValueError("CONTRAST_REQUIRES_AT_LEAST_ONE_FOCAL_ARM")
        if set(self.focal_arms) & set(self.comparator_arms):
            raise ValueError("FOCAL_AND_COMPARATOR_ARMS_OVERLAP")
        if len(set(all_bound)) != len(all_bound):
            raise ValueError("DUPLICATE_ARM_IN_CONTRAST")
        for v in all_bound:
            if v not in study_variable_ids:
                raise ValueError(f"UNKNOWN_STUDY_ARM_VARIABLE: {v}")
        if self.contrast_kind not in CONTRAST_KINDS:
            raise ValueError(f"UNKNOWN_CONTRAST_KIND: {self.contrast_kind!r}")
        if self.contrast_kind == "single_arm" and (self.comparator_arms or len(self.focal_arms) != 1):
            raise ValueError("SINGLE_ARM_CONTRAST_REQUIRES_EXACTLY_ONE_FOCAL_ARM_AND_NO_COMPARATOR")
        if self.contrast_kind == "pairwise" and (len(self.focal_arms) != 1 or len(self.comparator_arms) != 1):
            raise ValueError("PAIRWISE_CONTRAST_REQUIRES_EXACTLY_ONE_FOCAL_AND_ONE_COMPARATOR")
        if self.contrast_kind == "multi_arm_joint" and len(all_bound) < 3:
            raise ValueError("MULTI_ARM_JOINT_CONTRAST_REQUIRES_AT_LEAST_THREE_ARMS")
        missing_arms = [a for a in raw_mentioned_arms if a not in set(all_bound)]
        if missing_arms:
            raise ValueError(f"MULTI_ARM_TRUNCATION_DETECTED: dropped arms {missing_arms}")


@dataclass(frozen=True)
class ProvenancePin:
    """Source location and primary-verification metadata for an extracted outcome."""

    document_id: str
    section_pointer: str
    verbatim_quote: str
    requires_figure_digitization: bool
    primary_pdf_verified: bool


def split_outcome_tuple(outcome_tuple: str) -> tuple[str, str]:
    """Split `<study_or_doc_id>:<outcome_id>` via colon-safe `rsplit(':', 1)`.

    Preserves colon-bearing document identifiers such as `PMID:24836310:O1` or
    `DOI:10.1056/NEJMoa1402582:O1` rather than truncating at the first colon.
    """
    if not isinstance(outcome_tuple, str) or ":" not in outcome_tuple:
        raise ValueError(f"INVALID_OUTCOME_TUPLE: {outcome_tuple!r}")
    study_part, outcome_part = outcome_tuple.rsplit(":", 1)
    if not study_part or not outcome_part:
        raise ValueError(f"INVALID_OUTCOME_TUPLE: {outcome_tuple!r}")
    return study_part, outcome_part


@dataclass(frozen=True)
class EvidenceAtom:
    """Typed empirical record bound to a single outcome."""

    atom_id: str
    outcome_tuple: str
    study_id: str
    endpoint: str
    context: ContextSignature
    contrast: MultiArmContrast
    focal_quantity: QuantitySpec
    comparator_quantity: QuantitySpec | None
    statistic_kind: Literal["mean", "median", "proportion", "rate", "hazard_ratio", "fold_change", "qualitative"]
    sample_size: int | None
    p_value_or_interval: str | None
    polarity: Literal["favorable", "unfavorable", "neutral_or_mixed", "unresolved"]
    provenance: ProvenancePin

    def validate(self, study_variable_ids: set[str], raw_mentioned_arms: Sequence[str]) -> None:
        if self.atom_id != f"ATOM:{self.outcome_tuple}":
            raise ValueError(f"ATOM_ID_MISMATCH: {self.atom_id} vs {self.outcome_tuple}")
        tuple_study_id, _ = split_outcome_tuple(self.outcome_tuple)
        if tuple_study_id != self.study_id:
            raise ValueError(f"CROSS_STUDY_ATOM_MISBINDING: {self.atom_id}")
        if self.statistic_kind not in STATISTIC_KINDS:
            raise ValueError(f"UNKNOWN_STATISTIC_KIND: {self.statistic_kind!r}")
        if self.polarity not in ATOM_POLARITIES:
            raise ValueError(f"UNKNOWN_ATOM_POLARITY: {self.polarity!r}")
        if self.sample_size is not None and (type(self.sample_size) is not int or self.sample_size < 0):
            raise ValueError(f"SAMPLE_SIZE_MUST_BE_A_NON_NEGATIVE_INTEGER: {self.sample_size!r}")
        if type(self.provenance.requires_figure_digitization) is not bool:
            raise ValueError(
                f"PROVENANCE_BOOLEAN_MUST_BE_BOOL (requires_figure_digitization): {self.provenance.requires_figure_digitization!r}"
            )
        if type(self.provenance.primary_pdf_verified) is not bool:
            raise ValueError(
                f"PROVENANCE_BOOLEAN_MUST_BE_BOOL (primary_pdf_verified): {self.provenance.primary_pdf_verified!r}"
            )
        if not self.endpoint.strip():
            raise ValueError(f"EMPTY_ENDPOINT_NAME: {self.atom_id}")
        self.contrast.validate(study_variable_ids, raw_mentioned_arms)
        self.focal_quantity.validate()
        if self.comparator_quantity is not None:
            self.comparator_quantity.validate()
        checked_texts = (
            ("atom_id", self.atom_id),
            ("outcome_tuple", self.outcome_tuple),
            ("study_id", self.study_id),
            ("endpoint", self.endpoint),
            ("unit", self.focal_quantity.unit or ""),
            ("comparator_unit", self.comparator_quantity.unit if self.comparator_quantity and self.comparator_quantity.unit else ""),
            ("p_value_or_interval", self.p_value_or_interval or ""),
            ("document_id", self.provenance.document_id),
            ("section_pointer", self.provenance.section_pointer),
            ("verbatim_quote", self.provenance.verbatim_quote),
        )
        for label, text in checked_texts:
            if contains_forbidden_unicode(text):
                raise ValueError(f"FORBIDDEN_CODE_POINT_IN_{label.upper()}: {self.atom_id}")
        # The endpoint name is screened unmasked (release 3.1). Release 3.0 passed the
        # endpoint as its own masking lexicon, which blanked the whole name before the
        # script scan, so Cyrillic, Greek-confusable and mixed-script endpoint names
        # were never reported.
        for label, text, lex in (
            ("endpoint", self.endpoint, frozenset()),
            ("unit", self.focal_quantity.unit or "", frozenset()),
            (
                "comparator_unit",
                self.comparator_quantity.unit if self.comparator_quantity and self.comparator_quantity.unit else "",
                frozenset(),
            ),
        ):
            if _depth_profile(text) is None:
                raise ValueError(f"UNBALANCED_BRACKETS_IN_{label.upper()}: {self.atom_id}")
            if contains_confusable_homoglyphs(text, lex):
                raise ValueError(f"CONFUSABLE_HOMOGLYPH_IN_{label.upper()}: {self.atom_id}")
        # Arm identifiers are screened as well (release 3.1); release 3.0 did not inspect them.
        for arm in list(self.contrast.focal_arms) + list(self.contrast.comparator_arms):
            if contains_forbidden_unicode(arm):
                raise ValueError(f"FORBIDDEN_CODE_POINT_IN_ARM_IDENTIFIER: {self.atom_id}")
            if contains_confusable_homoglyphs(arm):
                raise ValueError(f"CONFUSABLE_HOMOGLYPH_IN_ARM_IDENTIFIER: {self.atom_id}")
        if self.statistic_kind == "qualitative" and (
            self.focal_quantity.kind != "missing_numeric"
            or (self.comparator_quantity is not None and self.comparator_quantity.kind != "missing_numeric")
        ):
            raise ValueError(f"QUALITATIVE_OUTCOME_CANNOT_EMIT_NUMERIC_ATOM: {self.atom_id}")
        if self.provenance.requires_figure_digitization and (
            self.focal_quantity.kind == "point_level"
            or (self.comparator_quantity is not None and self.comparator_quantity.kind == "point_level")
        ):
            raise ValueError(f"UNDIGITIZED_FIGURE_CANNOT_EMIT_EXACT_POINT_ATOM: {self.atom_id}")


# ---------------------------------------------------------------------------
# Numeral, unit and number-word tokenization (I3_tok)
# ---------------------------------------------------------------------------

def _build_unit_table() -> dict[str, str]:
    """Surface spelling -> canonical unit. Matching is case-sensitive (mM is not mm, uM is not um)."""
    t: dict[str, str] = {"%": "%", "percent": "%", "per cent": "%", "fold": "fold", "×": "fold"}
    micro = ("u", "µ", "μ", "mc")
    mass = [("pg", "pg"), ("ng", "ng"), ("mg", "mg"), ("g", "g")] + [(m + "g", "ug") for m in micro]
    for surface, canon in mass:
        for d_surface, d_canon in (("mL", "mL"), ("ml", "mL"), ("dL", "dL"), ("dl", "dL"), ("L", "L")):
            t[f"{surface}/{d_surface}"] = f"{canon}/{d_canon}"
        t[f"{surface}/kg"] = f"{canon}/kg"
        t[f"{surface}/lung"] = f"{canon}/lung"
        t[surface] = canon
    t["kg"] = "kg"
    t["kg/lung"] = "kg/lung"
    for m in micro:
        t[m + "L"] = "uL"
        t[m + "M"] = "uM"
        t[m + "m"] = "um"
    for surface, canon in (("mL", "mL"), ("ml", "mL"), ("dL", "dL"), ("dl", "dL"), ("L", "L"), ("l", "L")):
        t[surface] = canon
    for d_surface in ("mL", "ml"):
        for t_surface in ("year", "yr"):
            t[f"{d_surface}/{t_surface}"] = "mL/year"
    for s in ("liter", "liters", "litre", "litres"):
        t[s] = "L"
    # Molarity (uppercase M) vs SI length (lowercase m) kept strictly distinct.
    for s in ("fM", "pM", "nM", "mM", "M"):
        t[s] = s
    for s in ("fm", "pm", "nm", "mm", "cm", "m", "km"):
        t[s] = s
    for s in ("mmHg", "kPa", "W", "kW"):
        t[s] = s
    for s in ("point", "points"):
        t[s] = "points"
    for s in ("min", "mins", "minute", "minutes"):
        t[s] = "min"
    for s in ("h", "hr", "hrs", "hour", "hours"):
        t[s] = "h"
    for s in ("d", "day", "days"):
        t[s] = "days"
    for s in ("wk", "wks", "week", "weeks"):
        t[s] = "weeks"
    for s in ("mo", "mos", "month", "months"):
        t[s] = "months"
    for s in ("yr", "yrs", "year", "years"):
        t[s] = "years"
    return t


UNIT_TABLE: dict[str, str] = _build_unit_table()
_UNIT_CASEFOLD: dict[str, set[str]] = {}
for _surface, _canon in UNIT_TABLE.items():
    _UNIT_CASEFOLD.setdefault(_surface.lower(), set()).add(_canon)


def canonical_unit(unit: str | None) -> str | None:
    """Canonical unit string, or the stripped input if the unit is not tabulated.

    Case-insensitive fallback applies only to multi-letter word units whose
    lowercase folding maps to a single unambiguous canonical unit (e.g. 'Months'
    -> 'months', never conflating 'mm' with 'mM' or 'um' with 'uM').
    """
    if unit is None:
        return None
    u = unit.strip()
    if u in UNIT_TABLE:
        return UNIT_TABLE[u]
    folded = _UNIT_CASEFOLD.get(u.lower(), set())
    if len(u) >= 3 and len(folded) == 1:
        return next(iter(folded))
    return u


# Horizontal spaces of any width, so a thin or no-break space cannot detach a unit.
_SPACE_CLASS = "[ \\t\\u00a0\\u1680\\u2000-\\u200a\\u202f\\u205f\\u3000]"
_SIGN_CLASS = "[-\\u2010\\u2011\\u2012\\u2013\\u2212+]"
# Single-letter ambiguous units ('d', 'h', 'l', 'm', 'g') require an explicit space or hyphen
# after the numeral so panel/list labels ('4d' in 'Fig. 4d', '7h' in 'Fig. 7h', '2d' in 'Table 2d')
# do not parse as '4 days' or '7 h'.
_SPACED_ONLY_UNITS = frozenset({"d", "h", "l", "m", "g"})
_DESIGNATOR_PREFIX_RE = re.compile(
    r"\b(?:fig(?:ure)?s?\.?|tab(?:le)?s?\.?|panels?|supp(?:lementary)?\.?)\s*$",
    re.IGNORECASE,
)
_DIRECT_UNITS = [s for s in UNIT_TABLE if s not in _SPACED_ONLY_UNITS]
_UNIT_DIRECT_ALT = "|".join(re.escape(s) for s in sorted(_DIRECT_UNITS, key=lambda s: (-len(s), s)))
_UNIT_SPACED_ALT = "|".join(re.escape(s) for s in sorted(_SPACED_ONLY_UNITS, key=lambda s: (-len(s), s)))
_UNIT_ALT = "|".join(re.escape(s) for s in sorted(UNIT_TABLE, key=lambda s: (-len(s), s)))
_UNIT_AFTER = (
    r"(?:(?:(?:" + _SPACE_CLASS + r"*|-)(?P<unit_direct>" + _UNIT_DIRECT_ALT + r")"
    r"|(?:" + _SPACE_CLASS + r"+|-)(?P<unit_spaced>" + _UNIT_SPACED_ALT + r"))(?![A-Za-z0-9/]))"
)
NUM_TOKEN_RE = re.compile(
    r"(?<![A-Za-z0-9._])"
    r"(?P<sign>" + _SIGN_CLASS + r"(?=[0-9.]))?"
    r"(?P<num>[1-9][0-9]{0,2}(?:,[0-9]{3})+(?:\.[0-9]+)?|[0-9]+(?:[.,][0-9]+)?|\.[0-9]+)"
    + _UNIT_AFTER + r"?"
)
_UNIT_AFTER_RE = re.compile(_UNIT_AFTER)
_PAREN_UNIT_RE = re.compile(r"\(\s*(?P<punit>" + _UNIT_ALT + r")\s*\)(?![A-Za-z0-9/])")
_MULTIPLIER_RE = re.compile(r"(?<![A-Za-z0-9])[x×]([0-9]+(?:\.[0-9]+)?)(?![A-Za-z0-9.])")
_GLUED_PREFIX_RE = re.compile(r"(?:[^\W\d_]|_)[0-9]+(?:[.,][0-9]+)?")

_ONES = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9}
_TENS = {"twenty": 20, "thirty": 30, "forty": 40, "fifty": 50, "sixty": 60, "seventy": 70, "eighty": 80, "ninety": 90}
WORD_NUMBERS: dict[str, str] = {
    "zero": "0", **{w: str(v) for w, v in _ONES.items()},
    "ten": "10", "eleven": "11", "twelve": "12", "thirteen": "13", "fourteen": "14", "fifteen": "15",
    "sixteen": "16", "seventeen": "17", "eighteen": "18", "nineteen": "19",
    **{w: str(v) for w, v in _TENS.items()},
    "hundred": "100", "thousand": "1000", "million": "1000000", "dozen": "12",
    "single": "1", "double": "2", "doubled": "2", "doubles": "2", "doubling": "2", "twice": "2",
    "triple": "3", "tripled": "3", "tripling": "3", "treble": "3", "trebled": "3", "trebling": "3", "thrice": "3",
    "quadruple": "4", "quadrupled": "4", "quadrupling": "4", "quintupled": "5",
    "halve": "0.5", "halved": "0.5", "halving": "0.5", "half": "0.5", "quarter": "0.25",
    "first": "1", "second": "2", "third": "3", "fourth": "4", "fifth": "5",
    "sixth": "6", "seventh": "7", "eighth": "8", "ninth": "9", "tenth": "10",
}
VULGAR_FRACTIONS: dict[str, str] = {
    "½": "0.5", "⅓": "0.333", "⅔": "0.667", "¼": "0.25", "¾": "0.75", "⅕": "0.2",
}

_WORD_ALT = "|".join(sorted(WORD_NUMBERS, key=lambda w: (-len(w), w)))
WORD_FOLD_RE = re.compile(r"(?<![A-Za-z])(" + _WORD_ALT + r")fold(?![A-Za-z])", re.IGNORECASE)
WORD_COMPOUND_RE = re.compile(
    r"(?<![A-Za-z])(" + "|".join(_TENS) + r")[- ](" + "|".join(_ONES) + r")(?![A-Za-z])", re.IGNORECASE
)
WORD_SINGLE_RE = re.compile(r"(?<![A-Za-z])(" + _WORD_ALT + r")(?![A-Za-z])", re.IGNORECASE)

# Fixed trial-design and statistics idioms whose number words do not denote a quantity.
IDIOM_RE = re.compile(
    r"(?<![A-Za-z])(?:"
    r"(?:single|double|triple)[- ]blind(?:ed)?|double[- ]dummy|"
    r"(?:first|second|third)[- ]line|first[- ]in[- ]human|"
    r"half[- ](?:life|lives|maximal)|"
    r"(?:one|two)[- ](?:way|sided|tailed)|"
    r"(?:zero|first|second)[- ]order|"
    r"single[- ](?:dose|cent(?:er|re)|arm|agent)|"
    r"(?:first|second|third)[- ]generation"
    r")(?![A-Za-z])",
    re.IGNORECASE,
)

_DOC_ID_PAT = r"(?:DOI:10\.[0-9]{4,9}/[A-Za-z0-9_.-]+|[A-Z][A-Za-z0-9_.-]*(?::(?!O\d+(?![A-Za-z0-9]))[A-Za-z0-9_.-]+)*)"
_TUPLE_PAT = _DOC_ID_PAT + r":O\d+[A-Za-z0-9_-]*"
TUPLE_RE = re.compile(r"(?<![A-Za-z0-9_])(" + _TUPLE_PAT + r")(?![A-Za-z0-9_])")
# Only outcome-tuple identifiers are masked by pattern, and an unknown tuple is
# rejected outright. Every other name that embeds digits must be registered in
# the graph (entity lexicon) to be masked: a mask that swallows arbitrary
# upper-case tokens or identifier-shaped strings also swallows numerals glued to them.
IDENTIFIER_MASK_RE = re.compile(r"(?<![A-Za-z0-9_])" + _TUPLE_PAT + r"(?![A-Za-z0-9_])")
_ENTITY_TOKEN_RE = re.compile(r"(?<![A-Za-z0-9])[A-Za-z][A-Za-z0-9]*(?:-[A-Za-z0-9]+)*")

# Unicode Default_Ignorable_Code_Point ranges: characters a renderer may draw with no width.
_DEFAULT_IGNORABLE_RANGES = (
    (0x00AD, 0x00AD), (0x034F, 0x034F), (0x061C, 0x061C), (0x115F, 0x1160), (0x17B4, 0x17B5),
    (0x180B, 0x180F), (0x200B, 0x200F), (0x202A, 0x202E), (0x2060, 0x206F), (0x3164, 0x3164),
    (0xFE00, 0xFE0F), (0xFEFF, 0xFEFF), (0xFFA0, 0xFFA0), (0xFFF0, 0xFFF8),
    (0x1BCA0, 0x1BCA3), (0x1D173, 0x1D17A), (0xE0000, 0xE0FFF),
)


_GREEK_CONFUSABLE_CODEPOINTS = frozenset({
    0x0391, 0x0392, 0x0395, 0x0396, 0x0397, 0x0399, 0x039A, 0x039C,
    0x039D, 0x039F, 0x03A1, 0x03A4, 0x03A5, 0x03A7, 0x03BD, 0x03BF,
})
_ALPHA_WORD_RE = re.compile(r"[^\W\d_]+")


def _is_numeral_or_ascii_ident_or_sep(ch: str) -> bool:
    return bool(ch) and (
        ch.isnumeric()
        or ("A" <= ch <= "Z")
        or ("a" <= ch <= "z")
        or ch in "_.,-+−–‐‑‒"
    )


def contains_forbidden_unicode(text: str) -> list[str]:
    """Code points that can hide, split or reorder what a reader sees (UTS #39).

    Format, control, surrogate, private-use and unassigned code points; non-ASCII
    spaces (`Zs`), line and paragraph separators (`Zl`/`Zp`); every default-ignorable
    code point (including variation selectors and tags); every enclosing mark (`Me`);
    and any combining mark (`Mn`/`Mc`/`Me`) splitting or adjoining an ASCII digit,
    ASCII identifier character, decimal separator, sign, whitespace, or boundary.
    """
    bad: list[str] = []
    base = ""
    n = len(text)
    for idx, ch in enumerate(text):
        if ch in ("\n", "\t", "\r", " "):
            base = ch
            continue
        cat = unicodedata.category(ch)
        cp = ord(ch)
        ignorable = any(lo <= cp <= hi for lo, hi in _DEFAULT_IGNORABLE_RANGES)
        next_ch = text[idx + 1] if idx + 1 < n else ""
        mark_hazard = (
            cat == "Me"
            or (
                cat.startswith("M")
                and (
                    not base
                    or base.isspace()
                    or _is_numeral_or_ascii_ident_or_sep(base)
                    or _is_numeral_or_ascii_ident_or_sep(next_ch)
                )
            )
        )
        if cat in ("Cf", "Cc", "Cs", "Co", "Cn", "Zl", "Zp", "Zs") or ignorable or mark_hazard:
            bad.append(f"U+{cp:04X}@pos{idx}")
        if not cat.startswith("M"):
            base = ch
    return bad


def contains_confusable_homoglyphs(text: str, lexicon: frozenset[str] = frozenset()) -> list[str]:
    """UTR #36 confusable scripts, mathematical/fullwidth look-alikes, and mixed-script words."""
    bad: list[str] = []
    for idx, ch in enumerate(text):
        cp = ord(ch)
        if (0x1D400 <= cp <= 0x1D7FF) or (0xFF01 <= cp <= 0xFF5E) or cp in (0x212A, 0x212B):
            bad.append(f"U+{cp:04X}@pos{idx}")
    masked = mask_identifiers(text, lexicon)
    for idx, ch in enumerate(masked):
        cp = ord(ch)
        if (
            (0x0400 <= cp <= 0x052F)
            or (0x2DE0 <= cp <= 0x2DFF)
            or (0xA640 <= cp <= 0xA69F)
            or (0x0530 <= cp <= 0x058F)
            or (0x13A0 <= cp <= 0x13FF)
            or (0xAB70 <= cp <= 0xABBF)
            or cp in _GREEK_CONFUSABLE_CODEPOINTS
        ):
            bad.append(f"U+{cp:04X}@pos{idx}")
    for m in _ALPHA_WORD_RE.finditer(masked):
        word = m.group(0)
        if word in UNIT_TABLE:
            continue
        has_latin = False
        has_non_latin = False
        for ch in word:
            if not unicodedata.category(ch).startswith("L"):
                continue
            if "a" <= ch <= "z" or "A" <= ch <= "Z":
                has_latin = True
            else:
                name = unicodedata.name(ch, "")
                if "LATIN" in name:
                    has_latin = True
                else:
                    has_non_latin = True
        if has_latin and has_non_latin:
            bad.append(f"mixed_script:{word}@pos{m.start()}")
    return bad


def normalize_numeric_literal(raw: str) -> str:
    """Canonical decimal string: ASCII minus, grouping commas removed, trailing zeros dropped."""
    s = raw.strip()
    for dash in ("‐", "‑", "‒", "–", "−"):
        s = s.replace(dash, "-")
    if re.fullmatch(r"[-+]?[1-9][0-9]{0,2}(?:,[0-9]{3})+(?:\.[0-9]+)?", s):
        s = s.replace(",", "")
    elif re.fullmatch(r"[-+]?[0-9]+,[0-9]+", s):
        s = s.replace(",", ".")
    if s.startswith("+"):
        s = s[1:]
    try:
        d = Decimal(s)
    except InvalidOperation:
        return s
    if not d.is_finite():
        return s
    norm = format(d.normalize(), "f")
    if "." in norm:
        norm = norm.rstrip("0").rstrip(".")
    return "0" if norm in ("-0", "") else norm


@dataclass(frozen=True)
class Mention:
    """A numeral found in prose, in reading order. `end` includes an attached unit."""

    start: int
    end: int
    value: str
    unit: str | None


def build_entity_lexicon(
    atoms_by_tuple: Mapping[str, "EvidenceAtom"],
    arm_aliases: Mapping[str, tuple[str, ...]] | None = None,
) -> frozenset[str]:
    """Registered names that embed digits (PRO-C3, COL1A1, NX-410), lower-cased.

    Harvested from atom endpoints, arm identifiers, arm aliases, source document
    identifiers and units that contain numeric characters. Only these exact strings
    are masked before numeral extraction.
    """
    terms: set[str] = set()

    def harvest(text: str) -> None:
        for m in _ENTITY_TOKEN_RE.finditer(text):
            tok = m.group(0)
            if any(c.isnumeric() for c in tok):
                terms.add(tok.lower())

    for atom in atoms_by_tuple.values():
        harvest(atom.endpoint)
        for arm in list(atom.contrast.focal_arms) + list(atom.contrast.comparator_arms):
            terms.add(arm.lower())
            for part in arm.split(":")[-1].split("_"):
                harvest(part)
        doc_id = atom.provenance.document_id.strip()
        if any(c.isnumeric() for c in doc_id):
            terms.add(doc_id.lower())
        for q in (atom.focal_quantity, atom.comparator_quantity):
            if q is not None and q.unit and any(c.isnumeric() for c in q.unit):
                terms.add(q.unit.strip().lower())
    for aliases in (arm_aliases or {}).values():
        for alias in aliases:
            harvest(alias)
    return frozenset(terms)


def build_endpoint_phrases(atoms_by_tuple: Mapping[str, "EvidenceAtom"]) -> frozenset[str]:
    """Full registered endpoint names, lower-cased.

    A verbatim endpoint name is masked in prose, so a digit, a superscript or a
    word such as "reduced" inside the name is not read as a numeral or a cue.
    """
    return frozenset(a.endpoint.strip().lower() for a in atoms_by_tuple.values() if a.endpoint.strip())


@lru_cache(maxsize=512)
def _lexicon_regex(lexicon: frozenset[str]) -> re.Pattern[str] | None:
    if not lexicon:
        return None
    alt = "|".join(re.escape(t) for t in sorted(lexicon, key=lambda t: (-len(t), t)))
    return re.compile(r"(?<![A-Za-z0-9])(?:" + alt + r")(?![A-Za-z0-9])(?![.,][0-9])", re.IGNORECASE)


def _blank(text: str, pattern: re.Pattern[str] | None) -> str:
    if pattern is None:
        return text
    return pattern.sub(lambda m: " " * len(m.group(0)), text)


def mask_identifiers(text: str, lexicon: frozenset[str] = frozenset()) -> str:
    """Replace outcome identifiers and registered names by spaces (offsets preserved)."""
    return _blank(_blank(text, IDENTIFIER_MASK_RE), _lexicon_regex(lexicon))


def _matched_unit_key(m: re.Match[str]) -> str | None:
    return m.group("unit_direct") or m.group("unit_spaced")


def find_noncanonical_numerals(text: str, lexicon: frozenset[str] = frozenset()) -> list[str]:
    """Numerals written in forms the tokenizer will not read as an ASCII decimal.

    Outside registered names: any numeric character that is not an ASCII digit
    or one of six tabulated vulgar fractions (other scripts' digits, superscripts,
    circled digits, Roman-numeral characters); an unmasked letter or underscore
    glued directly to a digit (except a standalone prefix multiplier such as x10);
    and a numeral with a letter suffix that is not a recognized unit (29.4k, 29.4e3,
    3d without space). These fail closed.
    """
    bad: list[str] = []
    masked = mask_identifiers(text, lexicon)
    for idx, ch in enumerate(masked):
        if unicodedata.category(ch) in ("Nd", "No", "Nl") and not ("0" <= ch <= "9") and ch not in VULGAR_FRACTIONS:
            bad.append(f"U+{ord(ch):04X}@pos{idx}")
    chars = list(masked)
    for mm in _MULTIPLIER_RE.finditer(masked):
        if not _UNIT_AFTER_RE.match(masked, mm.end()):
            chars[mm.start():mm.end()] = " " * (mm.end() - mm.start())
    masked_no_mult = "".join(chars)
    for m in _GLUED_PREFIX_RE.finditer(masked_no_mult):
        bad.append(f"glued:{m.group(0).strip()}@pos{m.start()}")
    for m in NUM_TOKEN_RE.finditer(masked):
        if _matched_unit_key(m) is None and m.end() < len(masked) and masked[m.end()].isalpha():
            bad.append(f"suffix:{masked[m.start():m.end() + 1]}@pos{m.start()}")
    return bad


def extract_mentions(text: str, lexicon: frozenset[str] = frozenset()) -> list[Mention]:
    """All numeral mentions in `text`, sorted by position."""
    masked = mask_identifiers(text, lexicon)
    found: list[Mention] = []

    def _is_designator_prefix(pos: int) -> bool:
        return bool(_DESIGNATOR_PREFIX_RE.search(masked[max(0, pos - 24):pos]))

    def with_unit(start: int, end: int, value: str) -> Mention:
        u = _UNIT_AFTER_RE.match(masked, end)
        ukey = _matched_unit_key(u) if u else None
        if u and ukey and not (ukey in ("d", "h") and _is_designator_prefix(start)):
            return Mention(start, u.end(), value, UNIT_TABLE[ukey])
        return Mention(start, end, value, None)

    for idx, ch in enumerate(masked):
        if ch in VULGAR_FRACTIONS:
            found.append(with_unit(idx, idx + 1, VULGAR_FRACTIONS[ch]))

    for m in NUM_TOKEN_RE.finditer(masked):
        raw = (m.group("sign") or "") + m.group("num")
        ukey = _matched_unit_key(m)
        if ukey in ("d", "h") and _is_designator_prefix(m.start()):
            found.append(Mention(m.start(), m.end("num"), normalize_numeric_literal(raw), None))
        else:
            unit = UNIT_TABLE[ukey] if ukey else None
            found.append(Mention(m.start(), m.end(), normalize_numeric_literal(raw), unit))

    for m in _MULTIPLIER_RE.finditer(masked):
        found.append(Mention(m.start(), m.end(), normalize_numeric_literal(m.group(1)), "fold"))

    def blank(pattern: re.Pattern[str], s: str) -> str:
        return pattern.sub(lambda m: " " * len(m.group(0)), s)

    words = blank(IDIOM_RE, masked)
    for m in WORD_FOLD_RE.finditer(words):
        found.append(Mention(m.start(), m.end(), WORD_NUMBERS[m.group(1).lower()], "fold"))
    words = blank(WORD_FOLD_RE, words)
    for m in WORD_COMPOUND_RE.finditer(words):
        total = _TENS[m.group(1).lower()] + _ONES[m.group(2).lower()]
        found.append(with_unit(m.start(), m.end(), str(total)))
    words = blank(WORD_COMPOUND_RE, words)
    for m in WORD_SINGLE_RE.finditer(words):
        found.append(with_unit(m.start(), m.end(), WORD_NUMBERS[m.group(1).lower()]))

    found.sort(key=lambda x: (x.start, x.end))
    return found


def extract_prose_numeric_mentions(sentence: str) -> list[tuple[str, str | None]]:
    """(value, canonical_unit) pairs in reading order."""
    return [(m.value, m.unit) for m in extract_mentions(sentence)]


def atom_numeric_pool(atom: EvidenceAtom) -> set[str]:
    """Untyped union of every numeral an atom carries.

    Used only by the bag-of-numbers baselines. The prose verifier uses the
    role-typed pools of `atom_role_pools`, because an untyped pool lets a sample
    size or a digit from an endpoint name stand in for a measured value.
    """
    pool: set[str] = set()
    for q in (atom.focal_quantity, atom.comparator_quantity):
        if q is None:
            continue
        for v in (q.value, q.lower, q.upper):
            if v is not None:
                pool.add(normalize_numeric_literal(v))
    if atom.sample_size is not None:
        pool.add(str(atom.sample_size))
    if atom.p_value_or_interval:
        for nm, _ in extract_prose_numeric_mentions(atom.p_value_or_interval):
            pool.add(nm)
    for m_ep in re.finditer(r"[0-9]+(?:\.[0-9]+)?", atom.endpoint):
        pool.add(normalize_numeric_literal(m_ep.group(0)))
    for arm_id in list(atom.contrast.focal_arms) + list(atom.contrast.comparator_arms):
        tail = arm_id.split(":")[-1]
        for m_arm in re.finditer(r"[0-9]+(?:\.[0-9]+)?", tail):
            pool.add(normalize_numeric_literal(m_arm.group(0)))
    return pool


def prose_unit(unit: str | None) -> str | None:
    """The canonical unit the tokenizer attaches when `unit` follows a numeral, else None.

    Value slots are keyed by this reading of the atom's own unit string, so an
    atom's canonical rendering always agrees with its slots, whatever its unit.
    """
    if not unit:
        return None
    m = NUM_TOKEN_RE.match("0 " + unit.strip())
    ukey = _matched_unit_key(m) if m else None
    return UNIT_TABLE[ukey] if (m and ukey) else None


@dataclass(frozen=True)
class AtomPools:
    """Numerals an atom licenses in prose, separated by the role they play."""

    value_slots: frozenset[tuple[str, str | None]]  # measured values with the unit the tokenizer reads
    sample: frozenset[str]                          # sample size
    aux: frozenset[tuple[str, str | None]]          # numerals in the statistic string and in registered names
    outcome_units: frozenset[str]
    raw_units: frozenset[str]                       # the atom's unit strings exactly as recorded
    interval_pairs: frozenset[tuple[str, str]] = frozenset()  # (lower, upper) pairs for interval_bound slots

    @property
    def value_numbers(self) -> frozenset[str]:
        return frozenset(v for v, _ in self.value_slots)

    @property
    def aux_numbers(self) -> frozenset[str]:
        return frozenset(v for v, _ in self.aux)


def atom_role_pools(
    atom: EvidenceAtom,
    arm_aliases: Mapping[str, tuple[str, ...]] | None = None,
    lexicon: frozenset[str] = frozenset(),
) -> AtomPools:
    slots: set[tuple[str, str | None]] = set()
    units: set[str] = set()
    raw_units: set[str] = set()
    interval_pairs: set[tuple[str, str]] = set()
    for q in (atom.focal_quantity, atom.comparator_quantity):
        if q is None:
            continue
        pu = prose_unit(q.unit)
        if pu:
            units.add(pu)
        if q.unit:
            raw_units.add(q.unit.strip())
        for v in (q.value, q.lower, q.upper):
            if v is not None:
                slots.add((normalize_numeric_literal(v), pu))
        if q.kind == "interval_bound" and q.lower is not None and q.upper is not None:
            interval_pairs.add((normalize_numeric_literal(q.lower), normalize_numeric_literal(q.upper)))
    sample = {str(atom.sample_size)} if atom.sample_size is not None else set()
    aux: set[tuple[str, str | None]] = set()
    sources: list[str] = [atom.endpoint]
    if atom.p_value_or_interval:
        sources.append(atom.p_value_or_interval)
    for arm in list(atom.contrast.focal_arms) + list(atom.contrast.comparator_arms):
        sources.extend(arm.split(":")[-1].split("_"))
        sources.extend((arm_aliases or {}).get(arm, ()))
    for src in sources:
        for m in extract_mentions(src, lexicon):
            aux.add((m.value, m.unit))
    return AtomPools(
        frozenset(slots),
        frozenset(sample),
        frozenset(aux),
        frozenset(units),
        frozenset(raw_units),
        frozenset(interval_pairs),
    )


def classify_mention(m: Mention, pools: AtomPools) -> str | None:
    """None if the atom licenses the mention, else 'UNGROUNDED', 'UNIT' or 'ROLE'."""
    v, u = m.value, m.unit
    in_value = v in pools.value_numbers
    if in_value and (u is None or (v, u) in pools.value_slots):
        return None
    aux_units = {au for av, au in pools.aux if av == v}
    if aux_units and (u is None or u in aux_units):
        return None
    if v in pools.sample and u is None:
        return None
    if in_value:
        return "UNIT"
    if aux_units or v in pools.sample:
        return "ROLE" if u in pools.outcome_units else "UNIT"
    return "UNGROUNDED"


# Words that may follow a measured value written without a unit.
_CONTINUATION_WORDS = frozenset(
    "versus vs in on to from and or with for at of across among whereas while but although though compared relative "
    "respectively was were than after before by during under as".split()
)
_NEXT_TOKEN_RE = re.compile(r"[^\s,;:()\[\]{}.!?]+")
_RANGE_REST_RE = re.compile(r"^[-–—−‐‑‒]\s*([0-9]+(?:[.,][0-9]+)?)")


def unread_unit_after(text: str, m: Mention, pools: AtomPools) -> str | None:
    """The token after a unit-less measured value if it may be an unread unit, else None.

    A value written with no tabulated unit must be followed by the atom's own unit
    string, by a registered interval-range upper bound ('50-150 µM'), by punctuation,
    or by a function word. Anything else (mcg/mL, litres, million, a per-mille sign)
    could be a unit or multiplier the tokenizer cannot read, so it is refused.
    """
    rest = text[m.end:].lstrip(" \t               　")
    if not rest:
        return None
    for raw in pools.raw_units:
        if rest.startswith(raw) and not rest[len(raw):len(raw) + 1].isalnum():
            return None
    rm = _RANGE_REST_RE.match(rest)
    if rm and (m.value, normalize_numeric_literal(rm.group(1))) in pools.interval_pairs:
        return None
    tok = _NEXT_TOKEN_RE.match(rest)
    if tok is None or tok.group(0).lower() in _CONTINUATION_WORDS:
        return None
    return tok.group(0)


# ---------------------------------------------------------------------------
# Citation scoping, units, arm order and polarity over prose (I3_clause)
# ---------------------------------------------------------------------------

STRONG_SPLIT_RE = re.compile(
    r"(?:;|\bwhereas\b|\bwhile\b|\bbut\b|\beven\s+though\b|\balthough\b|\bthough\b"
    r"|,\s*(?=(?:(?:in|for|under|within|across|per|from)\s+)?"
    + _TUPLE_PAT
    + r"|\b[A-Za-z0-9_-]+\s+(?:was|were|had|reached|showed)\b)"
    r"|\band\b(?=[^.;]*" + _TUPLE_PAT + r"))",
    re.IGNORECASE,
)
WEAK_SPLIT_RE = re.compile(
    r"\bin\s+contrast\s+to\b|\bas\s+opposed\s+to\b|\bcompared\s+(?:with|to)\b|\brelative\s+to\b"
    r"|\bversus\b|\bvs\.?(?![A-Za-z])",
    re.IGNORECASE,
)

_COMPARATIVES = ("higher", "greater", "longer", "lower", "shorter", "smaller", "fewer")
_INC_RE = re.compile(
    r"\b(?:increas(?:e|es|ed|ing)|rise|rises|risen|rising|rose|raised|raises|elevat(?:ed|es|ion|ions)"
    r"|higher|greater|longer|gain(?:ed|s)?|grew|climbed|augmented|enhanced|up-?regulat(?:ed|ion)"
    r"|expanded|prolonged|lengthened|extended|stabilized|amplified|doubled|tripled|quadrupled)\b",
    re.IGNORECASE,
)
_DEC_RE = re.compile(
    r"\b(?:decreas(?:e|es|ed|ing)|reduc(?:e|es|ed|ing|tion|tions)|lower(?:ed|s)?|shorter|smaller|fewer"
    r"|fell|fall|falls|fallen|drop(?:s|ped)?|declin(?:e|es|ed|ing)|diminish(?:ed|es)?|shortened|suppress(?:ed|es|ion)"
    r"|attenuat(?:ed|es|ion)|halved|deplet(?:ed|ion)|abrogated|down-?regulat(?:ed|ion))\b",
    re.IGNORECASE,
)
# A negation cue, then at most four words (commas allowed), then the direction cue; or a "non-" prefix.
_NEGATED_TAIL_RE = re.compile(
    r"(?:(?:\b(?:not|no|never|neither|nor|without|rather\s+than|failed\s+to|unable\s+to|lack\s+of|absence\s+of)|n't)"
    r"(?:[\s,]+[A-Za-z-]+){0,4}[\s,]*|\bnon-)$",
    re.IGNORECASE,
)
_FOCAL_DENIAL_RE = re.compile(
    r"\b(?:not|nor|never|rather\s+than)\s+(?:seen\s+|observed\s+|present\s+|found\s+)?"
    r"(?:(?:in|on|for|with|among)\s+)?(?:the\s+)?",
    re.IGNORECASE,
)
_CLAUSE_RESET_RE = re.compile(r"[;:]|\b(?:and|but|yet)\s+(?:the\s+)?[a-z0-9_-]+\s+$", re.IGNORECASE)
_ATTRIBUTION_RE = re.compile(r"\s*(?:in|on|for|with|among|of)\s+(?:the\s+)?", re.IGNORECASE)
_FROM_TO_WORD_RE = re.compile(r"\b(from|to)\b", re.IGNORECASE)
# An arm named only as the reference of a comparison ("compared with vehicle slices") is not the subject.
_REFERENCE_TAIL_RE = re.compile(
    r"(?:compared\s+(?:with|to)|in\s+contrast\s+to|as\s+opposed\s+to|versus|vs\.?|relative\s+to"
    r"|than(?:\s+(?:in|on|for|with|among))?|against|over)\s+(?:the\s+)?$",
    re.IGNORECASE,
)
_CONTRAST_CONNECTOR_RE = re.compile(
    r"\b(?:versus|vs\.?(?![A-Za-z])|compared\s+(?:with|to)|relative\s+to|against|than"
    r"|whereas|while|but|even\s+though|although|though|in\s+contrast\s+to|as\s+opposed\s+to|respectively)\b",
    re.IGNORECASE,
)
_CITATION_INTRODUCER_RE = re.compile(r"\b(?:see|cf|also|and|or|in|for|from|per|ref|refs)\b", re.IGNORECASE)
_CUE_DENIER_SUFFIX_RE = re.compile(r"^-(?:free|na[iï]ve|independent|deficient|negative)\b", re.IGNORECASE)
_CUE_DENIER_PREFIX_RE = re.compile(r"(?:non|un)-$", re.IGNORECASE)
_PAREN_UNIT_RE = re.compile(r"\(\s*([^()]+?)\s*\)")
_SAMPLE_PREFIX_RE = re.compile(r"\bn\s*=\s*$", re.IGNORECASE)
_COUNT_NOUN_AFTER_RE = re.compile(
    r"^\s+(?:patients|participants|subjects|mice|rats|animals|donors|slices|athletes|arms|cohorts|trials|studies)\b",
    re.IGNORECASE,
)
_PVAL_PREFIX_RE = re.compile(r"\bp\s*(?:value\s*)?(?:<=|>=|<|>|=|≤|≥)\s*$", re.IGNORECASE)

# Negated inequality prefixes are checked before unnegated ones so symbolic and
# verbal negations invert cleanly (e.g. "not < 5%" -> ">=", "not <= 5%" -> ">",
# "never > 10%" -> "<=", "not fewer than 5" / "never less than 5%" -> ">=").
_NEGATED_INEQ_PREFIX_RULES: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"\b(?:not|never|neither|nor)\s*(?:<=|≤)\s*$", re.IGNORECASE), ">"),
    (re.compile(r"\b(?:not|never|neither|nor)\s+at\s+most\s+$", re.IGNORECASE), ">"),
    (re.compile(r"\b(?:not|never|neither|nor)\s*<\s*$", re.IGNORECASE), ">="),
    (re.compile(r"\b(?:not|never|no|neither|nor)\s+(?:less|fewer)\s+than\s+$", re.IGNORECASE), ">="),
    (re.compile(r"\b(?:not|never|neither|nor)\s+(?:below|under)\s+$", re.IGNORECASE), ">="),
    (
        re.compile(
            r"\b(?:(?:did|does|do)\s+not|didn't|doesn't|never|failed\s+to)\s+fall\s+(?:below|under)\s+$",
            re.IGNORECASE,
        ),
        ">=",
    ),
    (re.compile(r"\b(?:not|never|neither|nor)\s*(?:>=|≥)\s*$", re.IGNORECASE), "<"),
    (re.compile(r"\b(?:not|never|neither|nor)\s+at\s+least\s+$", re.IGNORECASE), "<"),
    (re.compile(r"\b(?:not|never|neither|nor)\s*>\s*$", re.IGNORECASE), "<="),
    (
        re.compile(
            r"\b(?:not|never|no|neither|nor)\s+(?:more|greater|higher|longer)\s+than\s+$",
            re.IGNORECASE,
        ),
        "<=",
    ),
    (re.compile(r"\b(?:not|never|neither|nor)\s+(?:above|over|exceeding)\s+$", re.IGNORECASE), "<="),
    (
        re.compile(
            r"\b(?:(?:did|does|do)\s+not|didn't|doesn't|never|failed\s+to)\s+exceed(?:ed)?\s+$",
            re.IGNORECASE,
        ),
        "<=",
    ),
)
_PLAIN_INEQ_PREFIX_RULES: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"(?:>=|≥)\s*$|\bat\s+least\s+$", re.IGNORECASE), ">="),
    (re.compile(r"(?:<=|≤)\s*$|\bat\s+most\s+$", re.IGNORECASE), "<="),
    (re.compile(r">\s*$|\b(?:exceed(?:ed|s|ing)?|(?:more|greater)\s+than|above)\s+$", re.IGNORECASE), ">"),
    (re.compile(r"<\s*$|\b(?:(?:less|fewer)\s+than|below|under)\s+$", re.IGNORECASE), "<"),
    (re.compile(r"(?:≈|~)\s*$|\b(?:approximately|approx\.?|about|roughly|around)\s+$", re.IGNORECASE), "≈"),
)
_INEQ_PREFIX_RULES: tuple[tuple[re.Pattern[str], str], ...] = _NEGATED_INEQ_PREFIX_RULES + _PLAIN_INEQ_PREFIX_RULES
_STACKED_BOUND_NEG_RE = re.compile(
    r"(?:\b(?:not|never|without|hardly|scarcely|barely|(?:did|does|do)\s+not)|n't)\s+(?:ever\s+)?"
    r"(?:\b(?:not|never|no|neither|nor|without|hardly|scarcely|barely|fail(?:ed)?\s+to)\b)"
    r"(?:\s+|\s*(?:<=|>=|<|>|=|≤|≥|≈|~)"
    r"|\s+(?:at\s+(?:least|most)|(?:less|fewer|more|greater|higher|longer)\s+than"
    r"|below|under|above|over|exceed(?:ed|ing)?|fall\s+(?:below|under)))*\s*$",
    re.IGNORECASE,
)
_AMBIGUOUS_PRE_INEQ_RE = re.compile(
    r"(?:\b(?:hardly|scarcely|barely)\s*(?:<=|>=|<|>|=|≤|≥|≈|~|"
    r"\b(?:at\s+(?:least|most)|(?:less|fewer|more|greater|higher|longer)\s+than|below|under|above|over|exceed(?:ed|s|ing)?)\b)?\s*$"
    r"|\b(?:not|never|neither|nor|without)\s*(?:=|≈|~|\b(?:approximately|approx\.?|about|roughly|around|exactly)\b)\s*$)",
    re.IGNORECASE,
)
_RESIDUAL_BOUND_NEG_RE = re.compile(
    r"\b(?:not|never|neither|nor|hardly|scarcely|barely)\s*$",
    re.IGNORECASE,
)

PooledPolicy = Literal["reject", "permit"]

_OPENERS, _CLOSERS = "([{", ")]}"


def parse_prose_bound_operator(prefix: str) -> str | None:
    """Return the effective relational operator preceding a numeral, 'AMBIGUOUS_NEGATION', or None."""
    if _PVAL_PREFIX_RE.search(prefix):
        return None
    if _STACKED_BOUND_NEG_RE.search(prefix) or _AMBIGUOUS_PRE_INEQ_RE.search(prefix):
        return "AMBIGUOUS_NEGATION"
    for pat, op in _NEGATED_INEQ_PREFIX_RULES:
        if pat.search(prefix):
            return op
    if _RESIDUAL_BOUND_NEG_RE.search(prefix):
        return "AMBIGUOUS_NEGATION"
    for pat, op in _PLAIN_INEQ_PREFIX_RULES:
        if pat.search(prefix):
            return op
    return None


def parse_prose_quantity_bound(
    fragment: str,
    *,
    default_unit_dimension: str = "unspecified",
) -> QuantitySpec:
    """Parse a prose quantity/bound snippet conservatively into a `QuantitySpec`.

    Inverts clean symbolic and verbal negated inequalities (`'not <5%'` -> `>= 5 %`,
    `'never >10%'` -> `<= 10 %`, `'not fewer than 5 days'` -> `>= 5 days`), while
    bailing out conservatively to `missing_numeric` on forbidden Unicode, stacked
    or ambiguous residual negations (`'not 5%'`, `'not never <5%'`, `'hardly <5%'`),
    or unrecognized/missing units.
    """
    if contains_forbidden_unicode(fragment) or find_noncanonical_numerals(fragment):
        return QuantitySpec(kind="missing_numeric", missing_reason="noncanonical_or_forbidden_surface_form")
    mentions = extract_mentions(fragment)
    if len(mentions) != 1 or mentions[0].unit is None:
        return QuantitySpec(kind="missing_numeric", missing_reason="ambiguous_or_unitless_quantity_fragment")
    m = mentions[0]
    prefix = fragment[:m.start]
    op = parse_prose_bound_operator(prefix)
    if op == "AMBIGUOUS_NEGATION":
        return QuantitySpec(kind="missing_numeric", missing_reason="ambiguous_residual_negation")
    eff_op: RelationOp = "=" if op is None else op  # type: ignore[assignment]
    if eff_op in ("<", "<=", ">", ">="):
        return QuantitySpec(
            kind="censored_observation",
            relation=eff_op,
            value=m.value,
            unit=m.unit,
            unit_dimension=default_unit_dimension,
        )
    return QuantitySpec(
        kind="point_level",
        relation=eff_op,
        value=m.value,
        unit=m.unit,
        unit_dimension=default_unit_dimension,
    )


def _depth_profile(sentence: str) -> list[int] | None:
    """Bracket depth at each character (round, square, curly), or None if unbalanced."""
    depth = 0
    out: list[int] = []
    for ch in sentence:
        if ch in _OPENERS:
            depth += 1
            out.append(depth)
        elif ch in _CLOSERS:
            out.append(depth)
            depth -= 1
            if depth < 0:
                return None
        else:
            out.append(depth)
    return out if depth == 0 else None


def _split_at_depth0(text: str, offset: int, depth: list[int], pattern: re.Pattern[str]) -> list[tuple[int, int]]:
    """Spans of `text` (absolute offsets) separated by depth-0 matches of `pattern`."""
    spans: list[tuple[int, int]] = []
    last = 0
    for m in pattern.finditer(text):
        if depth[offset + m.start()] > 0:
            continue
        spans.append((offset + last, offset + m.start()))
        last = m.end()
    spans.append((offset + last, offset + len(text)))
    return spans


def _is_citation_only_segment(text: str) -> bool:
    """True when `text` carries only outcome anchors and citation connectives."""
    stripped = TUPLE_RE.sub(" ", text)
    stripped = _CITATION_INTRODUCER_RE.sub(" ", stripped)
    return not any(ch.isalnum() for ch in stripped)


def scope_segments(sentence: str) -> list[tuple[str, tuple[str, ...]]] | None:
    """Two-level citation scoping: [(segment_text, cited_tuples)], or None if unbalanced.

    1. Strong separators (; whereas, while, but, although, though, clause comma,
       'and' before a cited tuple) cut the sentence into statements.
    2. Inside a statement, weak contrast connectors (versus, compared with,
       relative to, ...) cut fragments. If two fragments cite different tuples
       each fragment keeps its own scope; otherwise the statement is one segment.
    3. A fragment or statement that cites nothing inherits the scope of the
       nearest preceding scoped segment, else the nearest following one.
    4. Pure citation-only fragments (e.g. "; see C500:O2 and C500:O1" or
       "and (C500:O1)") fold into their adjacent empirical segment so split
       trailing/leading pooled citations keep their joint tuple scope.
    Only separators at bracket depth 0 count.
    """
    depth = _depth_profile(sentence)
    if depth is None:
        return None

    def tuples_in(a: int, b: int) -> tuple[str, ...]:
        return tuple(dict.fromkeys(TUPLE_RE.findall(sentence[a:b])))

    # Each entry: [text_parts, scope]; scope == () marks an orphan awaiting inheritance.
    segments: list[list[Any]] = []
    for s_a, s_b in _split_at_depth0(sentence, 0, depth, STRONG_SPLIT_RE):
        if not sentence[s_a:s_b].strip():
            continue
        frags = [
            (f_a, f_b, tuples_in(f_a, f_b))
            for f_a, f_b in _split_at_depth0(sentence[s_a:s_b], s_a, depth, WEAK_SPLIT_RE)
            if sentence[f_a:f_b].strip()
        ]
        scoped = [f for f in frags if f[2]]
        if len({f[2] for f in scoped}) >= 2:
            local: list[list[Any]] = []
            pending: list[str] = []
            for f_a, f_b, f_scope in frags:
                text = sentence[f_a:f_b].strip()
                if f_scope:
                    local.append([pending + [text], f_scope])
                    pending = []
                elif local:
                    local[-1][0].append(text)
                else:
                    pending.append(text)
            segments.extend(local)
        else:
            segments.append([[sentence[s_a:s_b].strip()], tuples_in(s_a, s_b)])

    resolved: list[list[Any]] = []
    leading: list[str] = []
    for parts, scope in segments:
        if scope:
            resolved.append([leading + parts, scope])
            leading = []
        elif resolved:
            resolved[-1][0].extend(parts)
        else:
            leading.extend(parts)
    if leading and not resolved:
        resolved.append([leading, ()])

    merged: list[list[Any]] = []
    pending_parts: list[str] = []
    pending_scope: list[str] = []
    for parts, scope in resolved:
        joined = " ".join(parts)
        if scope and _is_citation_only_segment(joined):
            if merged:
                merged[-1][0].extend(parts)
                merged[-1][1] = tuple(dict.fromkeys(list(merged[-1][1]) + list(scope)))
            else:
                pending_parts.extend(parts)
                pending_scope.extend(scope)
        else:
            combined_scope = tuple(dict.fromkeys(pending_scope + list(scope)))
            merged.append([pending_parts + parts, combined_scope])
            pending_parts = []
            pending_scope = []
    if pending_parts:
        if merged:
            merged[-1][0].extend(pending_parts)
            merged[-1][1] = tuple(dict.fromkeys(list(merged[-1][1]) + pending_scope))
        else:
            merged.append([pending_parts, tuple(dict.fromkeys(pending_scope))])
    return [(" ".join(parts), tuple(scope)) for parts, scope in merged]


def _arm_cues(arms: Sequence[str], arm_aliases: Mapping[str, tuple[str, ...]]) -> list[str]:
    cues: list[str] = []
    for a in arms:
        for c in arm_aliases.get(a, (a.split(":")[-1],)):
            if c:
                cues.append(c.lower())
    return cues


def _cue_starts_at(haystack: str, cue: str, pos: int) -> bool:
    if not haystack.startswith(cue, pos):
        return False
    end = pos + len(cue)
    if pos > 0 and haystack[pos - 1].isalnum():
        return False
    if end < len(haystack) and haystack[end].isalnum():
        return False
    if _CUE_DENIER_PREFIX_RE.search(haystack[max(0, pos - 4):pos]):
        return False
    if _CUE_DENIER_SUFFIX_RE.match(haystack[end:]):
        return False
    return True


def _cue_hits(haystack: str, cues: Sequence[str]) -> list[tuple[int, int]]:
    hits: list[tuple[int, int]] = []
    for c in cues:
        start = 0
        while True:
            p = haystack.find(c, start)
            if p < 0:
                break
            if _cue_starts_at(haystack, c, p):
                hits.append((p, p + len(c)))
            start = p + 1
    return hits


def _cue_text(text: str, atom: EvidenceAtom) -> str:
    """Lower-cased text for cue search: outcome identifiers and the atom's own endpoint name blanked."""
    return mask_identifiers(text, frozenset({atom.endpoint.strip().lower()})).lower()


def _true_direction(atom: EvidenceAtom) -> Literal["increase", "decrease"] | None:
    fq, cq = atom.focal_quantity, atom.comparator_quantity
    if fq.kind == "relative_change":
        return fq.change_direction if fq.change_direction in ("increase", "decrease") else None
    if fq.kind == "point_level" and cq is not None and cq.kind == "point_level" and fq.value and cq.value:
        fv, cv = _parse_decimal_strict(fq.value), _parse_decimal_strict(cq.value)
        if fv < 0 or cv < 0 or re.search(r"\bchange\b", atom.endpoint, re.IGNORECASE):
            # Change-from-baseline endpoint: a verb may describe the within-arm
            # trajectory rather than the between-arm contrast, so polarity is not checked.
            return None
        if fv > cv:
            return "increase"
        if fv < cv:
            return "decrease"
    return None


def _check_direction(
    text: str,
    atom: EvidenceAtom,
    arm_aliases: Mapping[str, tuple[str, ...]],
    lexicon: frozenset[str] = frozenset(),
) -> list[str]:
    """Polarity check for one single-outcome segment.

    A verb or noun cue is ignored when the nearest arm mention before it is a
    comparator arm (it then describes the comparator). A comparative (higher,
    greater, longer, lower, shorter, smaller, fewer) attributed to the comparator
    arm, by position or by a following "in <comparator>", is read with its
    direction reversed. A negated cue of the true direction (including a post-hoc
    denial "not in <focal_arm>") is an error. Otherwise the governing cue is the
    last counted cue before the first mention of the focal value; if no cue
    precedes that value, the segment is wrong when it holds only cues of the
    opposite class.
    """
    true_dir = _true_direction(atom)
    if true_dir is None:
        return []
    low = _cue_text(text, atom)
    focal_cues = _arm_cues(atom.contrast.focal_arms, arm_aliases)
    comp_cues = _arm_cues(atom.contrast.comparator_arms, arm_aliases)
    focal_hits = _cue_hits(low, focal_cues)
    comp_hits = _cue_hits(low, comp_cues)

    def governed_by_comparator(pos: int) -> bool:
        f_before = max((b for a, b in focal_hits if a < pos), default=-1)
        c_before = max((b for a, b in comp_hits if a < pos), default=-1)
        if c_before <= f_before:
            return False
        return not bool(_CLAUSE_RESET_RE.search(low[c_before:pos]))

    def attributed_to_comparator(end: int) -> bool:
        m = _ATTRIBUTION_RE.match(low, end)
        return bool(m) and any(_cue_starts_at(low, c, m.end()) for c in comp_cues)

    flip = {"increase": "decrease", "decrease": "increase"}
    plain: list[tuple[int, str]] = []
    negated: set[str] = set()
    for kind, pattern in (("increase", _INC_RE), ("decrease", _DEC_RE)):
        for m in pattern.finditer(low):
            effective = kind
            on_comparator = governed_by_comparator(m.start()) or attributed_to_comparator(m.end())
            if on_comparator:
                if m.group(0).lower() not in _COMPARATIVES:
                    continue
                effective = flip[kind]
            if _NEGATED_TAIL_RE.search(low[max(0, m.start() - 64):m.start()]):
                negated.add(effective)
            else:
                plain.append((m.start(), effective))
    plain.sort()

    first_true_pos = next((pos for pos, kind in plain if kind == true_dir), None)
    if first_true_pos is not None and focal_cues:
        for dm in _FOCAL_DENIAL_RE.finditer(low, first_true_pos):
            if any(_cue_starts_at(low, fc, dm.end()) for fc in focal_cues):
                negated.add(true_dir)
                break

    t = atom.outcome_tuple
    opposite = flip[true_dir]
    errs: list[str] = []
    if true_dir in negated:
        errs.append(f"I3_CLAUSE_NEGATED_DIRECTION ({t}): negated {true_dir} cue on true {true_dir}")

    focal_value = normalize_numeric_literal(atom.focal_quantity.value or "")
    focal_pos = next((m.start for m in extract_mentions(text, lexicon) if m.value == focal_value), None)
    before = [kind for pos, kind in plain if focal_pos is not None and pos < focal_pos]
    kinds = {kind for _, kind in plain}
    if before:
        if before[-1] == opposite:
            errs.append(f"I3_CLAUSE_DIRECTION_REVERSAL ({t}): {opposite} cue governs the focal value on true {true_dir}")
    elif opposite in kinds and true_dir not in kinds:
        errs.append(f"I3_CLAUSE_DIRECTION_REVERSAL ({t}): {opposite} cue on true {true_dir}")
    return errs


def _check_inequalities(
    text: str,
    atom: EvidenceAtom,
    lexicon: frozenset[str],
) -> list[str]:
    """Verify relational bound operators (including inverted negated bounds) on measured values."""
    expected_by_value: dict[str, set[str]] = {}
    for q in (atom.focal_quantity, atom.comparator_quantity):
        if q is None or q.kind not in ("censored_observation", "point_level", "relative_change") or not q.value:
            continue
        expected_by_value.setdefault(normalize_numeric_literal(q.value), set()).add(q.relation)
    if not expected_by_value:
        return []

    masked = mask_identifiers(text, lexicon)
    errs: list[str] = []
    for m in extract_mentions(text, lexicon):
        allowed = expected_by_value.get(m.value)
        if not allowed:
            continue
        prefix = masked[max(0, m.start - 56):m.start]
        if _PVAL_PREFIX_RE.search(prefix) or _SAMPLE_PREFIX_RE.search(prefix):
            continue
        if m.unit is None and _COUNT_NOUN_AFTER_RE.match(masked[m.end:]):
            continue
        op = parse_prose_bound_operator(prefix) or "="
        if op not in allowed:
            errs.append(
                f"I3_CLAUSE_INEQUALITY_MISMATCH ({atom.outcome_tuple}): "
                f"effective bound {op!r} on {m.value!r} conflicts with {sorted(allowed)}"
            )
    return errs


def _trajectories(text: str, mentions: Sequence[Mention]) -> list[tuple[Mention, Mention]]:
    """(baseline, destination) pairs for 'from X ... to Y' and 'to Y ... from X' constructions.

    X and Y are the numerals that directly follow the two words; a clause
    separator between them breaks the construction.
    """

    def numeral_after(pos: int) -> Mention | None:
        for m in mentions:
            if m.start >= pos:
                return m if not text[pos:m.start].strip() else None
        return None

    words = [(w.group(1).lower(), w.start(), w.end()) for w in _FROM_TO_WORD_RE.finditer(text)]
    pairs: list[tuple[Mention, Mention]] = []
    for i, (word, _, end) in enumerate(words):
        first = numeral_after(end)
        if first is None:
            continue
        other = "to" if word == "from" else "from"
        for word2, start2, end2 in words[i + 1:]:
            if start2 < first.end or word2 != other:
                continue
            if ";" in text[first.end:start2]:
                break
            second = numeral_after(end2)
            if second is not None:
                pairs.append((first, second) if word == "from" else (second, first))
                break
    return pairs


def _has_contrast_connector(
    low: str,
    all_arm_hits: Sequence[tuple[int, int]],
    *,
    depth: Sequence[int] | None = None,
    require_depth0: bool = False,
) -> bool:
    for m in _CONTRAST_CONNECTOR_RE.finditer(low):
        if not require_depth0 or depth is None or depth[m.start()] == 0:
            return True
    return any(
        (not require_depth0 or depth is None or depth[a] == 0)
        and bool(_REFERENCE_TAIL_RE.search(low[max(0, a - 28):a]))
        for a, _ in all_arm_hits
    )


def bind_contrast_arms_conservatively(
    text: str,
    study_arm_aliases: Mapping[str, tuple[str, ...]],
) -> tuple[str | None, str | None]:
    """Bind `(focal_arm, comparator_arm)` conservatively from clause text.

    Returns `(None, None)` whenever:
      - brackets are unbalanced,
      - fewer than 2 or 3 or more distinct arms are named at bracket depth 0
        (never truncating >= 3 arms to the first pair), or
      - 2 arms are named without an explicit depth-0 contrast connector
        (e.g. symmetric `'Arm A and Arm B'` or a parenthetical side-note).
    """
    depth = _depth_profile(text)
    if depth is None:
        return (None, None)
    low = text.lower()
    arm_first_pos: list[tuple[int, str]] = []
    all_hits_d0: list[tuple[int, int]] = []
    for arm_id, aliases in study_arm_aliases.items():
        cues = [c.lower() for c in (aliases or (arm_id.split(":")[-1],)) if c]
        hits = [(a, b) for a, b in _cue_hits(low, cues) if depth[a] == 0]
        if hits:
            all_hits_d0.extend(hits)
            arm_first_pos.append((min(a for a, _ in hits), arm_id))
    if len(arm_first_pos) != 2:
        return (None, None)
    if not _has_contrast_connector(low, all_hits_d0, depth=depth, require_depth0=True):
        return (None, None)
    arm_first_pos.sort(key=lambda item: item[0])
    (pos0, arm0), (pos1, arm1) = arm_first_pos
    arm0_ref = bool(_REFERENCE_TAIL_RE.search(low[max(0, pos0 - 28):pos0]))
    arm1_ref = bool(_REFERENCE_TAIL_RE.search(low[max(0, pos1 - 28):pos1]))
    if arm0_ref and not arm1_ref:
        return (arm1, arm0)
    return (arm0, arm1)


def _check_arm_order(
    text: str,
    atom: EvidenceAtom,
    arm_aliases: Mapping[str, tuple[str, ...]],
    lexicon: frozenset[str],
) -> list[str]:
    """Agreement between where the arms' values stand and which arm they belong to.

    Enforces:
      1. a from/to construction over two point values must run comparator -> focal;
      2. when both focal and comparator arms are named in the clause, an explicit
         contrast connector is required (refusing symmetric 'ArmA and ArmB' or
         parenthetical arm side-notes), and naming >= 3 study arms in a pairwise
         clause is refused;
      3. a value directly attributed to an arm ("48.6 ng/mL in vehicle slices")
         must belong to that arm (including non-point comparators);
      4. with both arms named and both point values present, their orders agree;
      5. with both point values present and an arm unnamed at depth 0, focal comes first;
      6. a single stated value whose grammatical subject at depth 0 is solely the
         other arm (even when the true owner appears after 'than'/'versus' or inside
         parentheses, including comma-appositive constructions) is rejected.
    """
    t = atom.outcome_tuple
    depth = _depth_profile(text) or [0] * len(text)
    low = _cue_text(text, atom)
    focal_cues = _arm_cues(atom.contrast.focal_arms, arm_aliases)
    comp_cues = _arm_cues(atom.contrast.comparator_arms, arm_aliases)
    focal_hits = _cue_hits(low, focal_cues)
    comp_hits = _cue_hits(low, comp_cues)
    focal_hits_d0 = [(a, b) for a, b in focal_hits if depth[a] == 0]
    comp_hits_d0 = [(a, b) for a, b in comp_hits if depth[a] == 0]

    if atom.contrast.contrast_kind == "pairwise":
        bound_arms = set(atom.contrast.focal_arms) | set(atom.contrast.comparator_arms)
        study_prefix = f"VAR:{atom.study_id}:"
        extra_arms = [
            arm_id
            for arm_id in arm_aliases
            if arm_id.startswith(study_prefix)
            and arm_id not in bound_arms
            and any(depth[a] == 0 for a, _ in _cue_hits(low, _arm_cues((arm_id,), arm_aliases)))
        ]
        if extra_arms and (focal_hits_d0 or comp_hits_d0):
            return [f"I3_CLAUSE_MULTI_ARM_AMBIGUITY ({t}): unmodeled study arm(s) {sorted(extra_arms)} in pairwise clause"]

    fq, cq = atom.focal_quantity, atom.comparator_quantity
    mentions = extract_mentions(text, lexicon)
    has_point_fq = fq.kind == "point_level" and bool(fq.value)
    has_point_cq = cq is not None and cq.kind == "point_level" and bool(cq.value)
    fv_s = normalize_numeric_literal(fq.value) if has_point_fq and fq.value else None
    cv_s = normalize_numeric_literal(cq.value) if has_point_cq and cq and cq.value else None

    if has_point_fq and has_point_cq and fv_s is not None and cv_s is not None and fv_s != cv_s:
        relevant = [(b, d) for b, d in _trajectories(text, mentions) if {b.value, d.value} == {fv_s, cv_s}]
        if any(b.value == fv_s for b, _ in relevant):
            return [f"I3_CLAUSE_FROM_TO_SWAP ({t}): trajectory swaps baseline {cv_s} and focal {fv_s}"]
        if relevant:
            return []

    if focal_hits and comp_hits:
        both_d0 = bool(focal_hits_d0 and comp_hits_d0)
        check_hits = (focal_hits_d0 + comp_hits_d0) if both_d0 else (focal_hits + comp_hits)
        if not _has_contrast_connector(low, check_hits, depth=depth, require_depth0=both_d0):
            return [f"I3_CLAUSE_NON_CONTRAST_MULTI_ARM ({t}): multi-arm binding requires an explicit contrast connector"]

    if not has_point_fq or fv_s is None:
        return []
    if has_point_cq and fv_s == cv_s:
        return []

    target_vals = (fv_s, cv_s) if has_point_cq and cv_s is not None else (fv_s,)
    for m in mentions:
        if m.value not in target_vals:
            continue
        attribution = _ATTRIBUTION_RE.match(low, m.end)
        if not attribution:
            continue
        wrong = comp_cues if m.value == fv_s else focal_cues
        right = focal_cues if m.value == fv_s else comp_cues
        at = attribution.end()
        if any(_cue_starts_at(low, c, at) for c in wrong) and not any(_cue_starts_at(low, c, at) for c in right):
            return [f"I3_CLAUSE_ARM_VALUE_SWAP ({t}): value {m.value} is attributed to the other arm"]

    chars = list(low)
    for a, b in focal_hits + comp_hits:
        chars[a:b] = " " * (b - a)
    bare = extract_mentions("".join(chars), lexicon)
    fv_pos = next((m.start for m in bare if m.value == fv_s), -1)
    cv_pos = next((m.start for m in bare if cv_s is not None and m.value == cv_s), -1)

    def subject_position(hits: Sequence[tuple[int, int]]) -> int:
        for a, _ in sorted(hits):
            if not _REFERENCE_TAIL_RE.search(low[max(0, a - 28):a]):
                return a
        return -1

    f_subj = subject_position(focal_hits_d0)
    c_subj = subject_position(comp_hits_d0)

    if not has_point_cq:
        if fv_pos >= 0 and f_subj < 0 and 0 <= c_subj < fv_pos:
            return [f"I3_CLAUSE_ARM_VALUE_SWAP ({t}): focal value {fv_s} stated with comparator arm as subject"]
        return []

    eff_f_hits = focal_hits_d0 if focal_hits_d0 else focal_hits
    eff_c_hits = comp_hits_d0 if comp_hits_d0 else comp_hits
    f_pos = min((a for a, _ in eff_f_hits), default=-1)
    c_pos = min((a for a, _ in eff_c_hits), default=-1)

    if fv_pos >= 0 and cv_pos >= 0:
        if focal_hits_d0 and comp_hits_d0:
            if (f_pos < c_pos) != (fv_pos < cv_pos):
                return [f"I3_CLAUSE_ARM_VALUE_SWAP ({t}): focal/comparator values inverted relative to arm cues"]
        elif not focal_hits and not comp_hits:
            if cv_pos < fv_pos:
                return [f"I3_CLAUSE_UNLABELED_VALUE_ORDER ({t}): comparator value precedes focal value with arms unnamed"]
        elif (bool(focal_hits_d0) != bool(comp_hits_d0)) and cv_pos < fv_pos and not (0 <= c_pos < fv_pos):
            return [f"I3_CLAUSE_UNLABELED_VALUE_ORDER ({t}): comparator value precedes focal value with an arm unnamed"]
        elif focal_hits and comp_hits and (f_pos < c_pos) != (fv_pos < cv_pos):
            return [f"I3_CLAUSE_ARM_VALUE_SWAP ({t}): focal/comparator values inverted relative to arm cues"]
        return []

    if fv_pos >= 0 and f_subj < 0 and 0 <= c_subj < fv_pos:
        return [f"I3_CLAUSE_ARM_VALUE_SWAP ({t}): focal value {fv_s} stated with comparator arm as subject"]
    if cv_pos >= 0 and c_subj < 0 and 0 <= f_subj < cv_pos:
        return [f"I3_CLAUSE_ARM_VALUE_SWAP ({t}): comparator value {cv_s} stated with focal arm as subject"]
    return []


def _recognized_unit(raw: str) -> str | None:
    u = raw.strip()
    if u in UNIT_TABLE:
        return UNIT_TABLE[u]
    folded = _UNIT_CASEFOLD.get(u.lower(), set())
    if len(u) >= 3 and len(folded) == 1:
        return next(iter(folded))
    return None


_TRAILING_PAREN_UNIT_RE = re.compile(r"^\s*\(\s*([^()]+?)\s*\)")


def _unique_parenthesized_unit(masked_text: str) -> tuple[str, int] | None:
    """Return `(canonical_unit, close_paren_index)` when `masked_text` has one standalone `(unit)` header."""
    found: list[tuple[str, int]] = []
    for pm in _PAREN_UNIT_RE.finditer(masked_text):
        cu = _recognized_unit(pm.group(1))
        if cu is not None:
            found.append((cu, pm.end()))
    if len({u for u, _ in found}) == 1 and found:
        return found[0]
    return None


def _mention_errors(
    text: str,
    tup: str,
    pools_by_tuple: Mapping[str, AtomPools],
    lexicon: frozenset[str],
) -> list[str]:
    errs: list[str] = []
    pools = pools_by_tuple[tup]
    masked = mask_identifiers(text, lexicon)
    depth = _depth_profile(text) or [0] * len(text)
    header_unit = _unique_parenthesized_unit(masked)
    for m in extract_mentions(text, lexicon):
        eff = m
        if m.unit is None and depth[m.start] == 0:
            trailing_pm = _TRAILING_PAREN_UNIT_RE.match(masked[m.end:])
            trailing_cu = _recognized_unit(trailing_pm.group(1)) if trailing_pm is not None else None
            if trailing_cu is not None:
                eff = Mention(start=m.start, end=m.end, value=m.value, unit=trailing_cu)
            elif (
                header_unit is not None
                and not _SAMPLE_PREFIX_RE.search(masked[max(0, m.start - 12):m.start])
                and not _PVAL_PREFIX_RE.search(masked[max(0, m.start - 16):m.start])
                and not _COUNT_NOUN_AFTER_RE.match(masked[m.end:])
            ):
                eff = Mention(start=m.start, end=m.end, value=m.value, unit=header_unit[0])
        verdict = classify_mention(eff, pools)
        shown = eff.value + (f" {eff.unit}" if eff.unit else "")
        if verdict is None:
            measured_only = eff.value in pools.value_numbers and eff.value not in pools.sample and eff.value not in pools.aux_numbers
            if eff.unit is None and measured_only:
                stray = unread_unit_after(text, eff, pools)
                if stray is not None:
                    errs.append(f"I3_CLAUSE_UNREAD_UNIT ({tup}): value {shown!r} is followed by {stray!r}, not by its unit")
            continue
        if verdict == "UNGROUNDED":
            owners = sorted(o for o, p in pools_by_tuple.items() if o != tup and eff.value in p.value_numbers)
            if owners:
                errs.append(f"I3_CLAUSE_CROSS_OUTCOME_SWAP ({tup}): value {shown!r} belongs to sibling tuple {owners}")
            else:
                errs.append(f"I3_TOK_UNGROUNDED_VALUE ({tup}): {shown!r} is not licensed by the cited atom")
        elif verdict == "UNIT":
            errs.append(f"I3_CLAUSE_UNIT_MISMATCH ({tup}): {shown!r} conflicts with units {sorted(pools.outcome_units)}")
        else:
            errs.append(f"I3_TOK_ROLE_CONFUSION ({tup}): {shown!r} is a sample size or name numeral, not a measured value")
    return errs


def _endpoint_tokens(endpoint: str) -> set[str]:
    return {
        tok.lower()
        for tok in re.findall(r"[A-Za-z0-9]+(?:-[A-Za-z0-9]+)*", endpoint)
        if len(tok) >= 3 and tok.lower() not in {"the", "and", "for", "with", "mean", "total", "relative", "change"}
    }


def _verify_respectively(
    sentence: str,
    cited: Sequence[str],
    atoms_by_tuple: Mapping[str, EvidenceAtom],
    pools_by_tuple: Mapping[str, AtomPools],
    arm_aliases: Mapping[str, tuple[str, ...]],
    lexicon: frozenset[str],
) -> list[str]:
    """Index-aligned 'respectively' lists: value group i must belong to cited tuple i."""
    errs: list[str] = []
    mentions = extract_mentions(sentence, lexicon)
    all_values = set().union(*(pools_by_tuple[t].value_numbers for t in cited))
    candidates = [m for m in mentions if m.value in all_values]
    for m in mentions:
        if m.value in all_values:
            continue
        if all(classify_mention(m, pools_by_tuple[t]) is not None for t in cited):
            errs.append(f"I3_TOK_UNGROUNDED_VALUE ({list(cited)}): {m.value!r} is not licensed by any cited atom")
    k = len(cited)
    if not candidates:
        return errs
    if len(candidates) % k != 0:
        errs.append(
            f"I3_CLAUSE_RESPECTIVELY_ARITY ({list(cited)}): {len(candidates)} values cannot be aligned with {k} tuples"
        )
        return errs
    g = len(candidates) // k
    seg_start = 0
    for i, tup in enumerate(cited):
        group = candidates[i * g:(i + 1) * g]
        pools = pools_by_tuple[tup]
        for m in group:
            verdict = classify_mention(m, pools)
            if verdict == "UNGROUNDED":
                errs.append(f"I3_CLAUSE_RESPECTIVELY_SWAP ({tup}): value {m.value!r} is out of position")
            elif verdict is not None:
                errs.append(f"I3_CLAUSE_UNIT_MISMATCH ({tup}): {m.value!r} {m.unit!r} conflicts with the atom")
            elif m.unit is None:
                stray = unread_unit_after(sentence, m, pools)
                if stray is not None:
                    errs.append(f"I3_CLAUSE_UNREAD_UNIT ({tup}): value {m.value!r} is followed by {stray!r}, not by its unit")
        seg_end = group[-1].end
        seg_slice = sentence[seg_start:seg_end]
        errs.extend(_check_direction(seg_slice, atoms_by_tuple[tup], arm_aliases, lexicon))
        errs.extend(_check_inequalities(seg_slice, atoms_by_tuple[tup], lexicon))
        errs.extend(_check_arm_order(sentence, atoms_by_tuple[tup], arm_aliases, lexicon))
        seg_start = seg_end

    tail = sentence[candidates[-1].end:]
    if tail.strip():
        tail_chars = list(tail.lower())
        all_arm_cues: list[str] = []
        for tup in cited:
            atom = atoms_by_tuple[tup]
            all_arm_cues.extend(_arm_cues(atom.contrast.focal_arms, arm_aliases))
            all_arm_cues.extend(_arm_cues(atom.contrast.comparator_arms, arm_aliases))
        for a, b in _cue_hits("".join(tail_chars), all_arm_cues):
            tail_chars[a:b] = " " * (b - a)
        tail_words = set(re.findall(r"[a-z0-9]+(?:-[a-z0-9]+)*", "".join(tail_chars)))
        matched_tuples = [tup for tup in cited if _endpoint_tokens(atoms_by_tuple[tup].endpoint) & tail_words]
        for tup in (matched_tuples or list(cited)):
            errs.extend(_check_direction(tail, atoms_by_tuple[tup], arm_aliases, lexicon))
    return errs


def verify_prose_sentence(
    sentence: str,
    atoms_by_tuple: Mapping[str, EvidenceAtom],
    arm_aliases: Mapping[str, tuple[str, ...]] | None = None,
    *,
    pooled_numeral_policy: PooledPolicy = "reject",
    lexicon: frozenset[str] | None = None,
) -> list[str]:
    """Run I3_tok and I3_clause on one synthesis sentence; return violation strings.

    `pooled_numeral_policy` decides what happens to a numeral-bearing segment
    whose citation scope holds two or more tuples. Such a segment cannot be
    bound to one outcome from surface features (Theorem 2): "reject" refuses it,
    "permit" falls back to a union-pool check.
    """
    aliases = arm_aliases or {}
    errors: list[str] = []
    bad_uni = contains_forbidden_unicode(sentence)
    if bad_uni:
        return [f"I3_TOK_FORBIDDEN_UNICODE: {bad_uni}"]
    if lexicon is None:
        lexicon = build_entity_lexicon(atoms_by_tuple, aliases)
    # Prose is read with full endpoint names masked as well; atom pools are not,
    # so the 12 of a "12-week" endpoint still licenses "week 12".
    prose_lexicon = lexicon | build_endpoint_phrases(atoms_by_tuple)
    homoglyphs = contains_confusable_homoglyphs(sentence, prose_lexicon)
    if homoglyphs:
        return [f"I3_TOK_CONFUSABLE_HOMOGLYPH: {homoglyphs}"]
    odd = find_noncanonical_numerals(sentence, prose_lexicon)
    if odd:
        return [f"I3_TOK_NONCANONICAL_NUMERAL: {odd}"]

    cited_all = TUPLE_RE.findall(sentence)
    mentions = extract_mentions(sentence, prose_lexicon)
    if mentions and not cited_all:
        return [f"I3_TOK_UNCITED_EMPIRICAL_NUMERALS: {[m.value for m in mentions]}"]
    if not cited_all:
        return errors
    unknown = [t for t in cited_all if t not in atoms_by_tuple]
    if unknown:
        return [f"I1_UNKNOWN_OUTCOME_TUPLES: {unknown}"]
    if _depth_profile(sentence) is None:
        return ["I3_CLAUSE_UNBALANCED_BRACKETS: citation scope cannot be resolved"]

    cited = list(dict.fromkeys(cited_all))
    pools_by_tuple = {t: atom_role_pools(atoms_by_tuple[t], aliases, lexicon) for t in cited}

    if len(cited) >= 2 and re.search(r"\brespectively\b", sentence, re.IGNORECASE):
        return _verify_respectively(sentence, cited, atoms_by_tuple, pools_by_tuple, aliases, prose_lexicon)

    if len(cited) == 1:
        segments: list[tuple[str, tuple[str, ...]]] = [(sentence, (cited[0],))]
    else:
        segments = scope_segments(sentence) or []

    for text, scope in segments:
        if len(scope) == 1:
            tup = scope[0]
            errors.extend(_mention_errors(text, tup, pools_by_tuple, prose_lexicon))
            errors.extend(_check_direction(text, atoms_by_tuple[tup], aliases, prose_lexicon))
            errors.extend(_check_inequalities(text, atoms_by_tuple[tup], prose_lexicon))
            errors.extend(_check_arm_order(text, atoms_by_tuple[tup], aliases, prose_lexicon))
            continue
        seg_mentions = extract_mentions(text, prose_lexicon)
        if not seg_mentions:
            continue
        if pooled_numeral_policy == "reject":
            errors.append(
                f"I3_CLAUSE_AMBIGUOUS_POOLED_CITATION ({list(scope)}): numerals "
                f"{[m.value for m in seg_mentions]} cannot be bound to one cited outcome"
            )
            continue
        for m in seg_mentions:
            verdicts = [classify_mention(m, pools_by_tuple[t]) for t in scope]
            if any(v is None for v in verdicts):
                continue
            code = "I3_CLAUSE_UNION_UNIT_MISMATCH" if "UNIT" in verdicts or "ROLE" in verdicts else "I3_TOK_UNGROUNDED_VALUE"
            errors.append(f"{code} ({list(scope)}): {m.value!r} is not licensed by any pooled atom")
    return errors


def render_atom_sentence(atom: EvidenceAtom) -> str:
    """Canonical single-tuple sentence for an atom, focal value first.

    The endpoint name is masked as a registered phrase, each value carries the
    atom's own unit string, and no arm is named. The strict lint accepts the
    rendering of an atom that passes validation and also meets the rendering
    preconditions of section 4.3 of the manuscript. It is refused, for example,
    when the outcome identifier lacks the `O<digits>` form, when a
    `relative_change` atom carries an inequality relation (which this template
    omits), or when an endpoint or unit string contains a tuple identifier, a
    number word, a direction cue or a relation operator.
    """
    fq, cq = atom.focal_quantity, atom.comparator_quantity
    n = f" (n = {atom.sample_size})" if atom.sample_size is not None else ""

    def show(q: QuantitySpec) -> str:
        if q.kind == "interval_bound":
            return f"{q.lower} to {q.upper} {q.unit}"
        if q.kind == "censored_observation":
            return f"{q.relation} {q.value} {q.unit}"
        return f"{q.value} {q.unit}"

    head = f"In {atom.outcome_tuple} the endpoint {atom.endpoint}"
    if fq.kind == "missing_numeric":
        return f"{head} was reported without a numeric value{n}."
    if fq.kind == "relative_change":
        word = {"increase": "increase", "decrease": "decrease"}.get(fq.change_direction or "", "change")
        return f"{head} showed a relative {word} of {fq.value} {fq.unit}{n}."
    if cq is not None and cq.kind != "missing_numeric":
        return f"{head} was {show(fq)} versus {show(cq)}{n}."
    return f"{head} was {show(fq)}{n}."


# ---------------------------------------------------------------------------
# Typed propositions (I3_typed)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class TypedProposition:
    """Structured quantitative assertion about exactly one outcome."""

    proposition_id: str
    outcome_tuple: str
    claimed_endpoint: str
    claimed_focal_arms: tuple[str, ...]
    claimed_comparator_arms: tuple[str, ...]
    claimed_quantity: QuantitySpec
    claimed_statistic: str
    claimed_sample_size: int | None
    claimed_comparator_quantity: QuantitySpec | None = None


TYPED_COORDINATES = (
    "endpoint",
    "focal_arms",
    "comparator_arms",
    "focal_quantity",
    "comparator_quantity",
    "statistic",
    "sample_size",
)


_ASCII_LOWER = str.maketrans("ABCDEFGHIJKLMNOPQRSTUVWXYZ", "abcdefghijklmnopqrstuvwxyz")


def normalize_endpoint_name(name: str) -> str:
    """Trim ASCII spaces and tabs and fold ASCII letters only.

    Unicode case folding is avoided on purpose: it maps look-alikes such as the
    Kelvin sign onto ASCII letters.
    """
    return name.strip(" \t").translate(_ASCII_LOWER)


def verify_typed_proposition_homomorphism(
    prop: TypedProposition,
    atoms_by_tuple: Mapping[str, EvidenceAtom],
) -> list[str]:
    """I3_typed: the proposition must equal its atom on all seven coordinates."""
    if prop.outcome_tuple not in atoms_by_tuple:
        return [f"I1_UNKNOWN_ATOM_FOR_PROPOSITION ({prop.proposition_id}): {prop.outcome_tuple}"]
    atom = atoms_by_tuple[prop.outcome_tuple]
    pid = prop.proposition_id
    errs: list[str] = []
    if normalize_endpoint_name(prop.claimed_endpoint) != normalize_endpoint_name(atom.endpoint):
        errs.append(f"I3_TYPED_ENDPOINT_MISMATCH ({pid}): {prop.claimed_endpoint!r} != {atom.endpoint!r}")
    if len(set(prop.claimed_focal_arms)) != len(prop.claimed_focal_arms) or set(prop.claimed_focal_arms) != set(
        atom.contrast.focal_arms
    ):
        errs.append(f"I3_TYPED_FOCAL_ARM_MISMATCH ({pid}): {prop.claimed_focal_arms} != {atom.contrast.focal_arms}")
    if len(set(prop.claimed_comparator_arms)) != len(prop.claimed_comparator_arms) or set(
        prop.claimed_comparator_arms
    ) != set(atom.contrast.comparator_arms):
        errs.append(
            f"I3_TYPED_COMPARATOR_ARM_MISMATCH ({pid}): {prop.claimed_comparator_arms} != {atom.contrast.comparator_arms}"
        )
    for label, q in (("focal", prop.claimed_quantity), ("comparator", prop.claimed_comparator_quantity)):
        if q is None:
            continue
        try:
            q.validate()
        except ValueError as exc:
            errs.append(f"I3_TYPED_INVALID_QUANTITY ({pid}, {label}): {exc}")
            return errs
    if prop.claimed_quantity != atom.focal_quantity:
        errs.append(f"I3_TYPED_QUANTITY_MISMATCH ({pid}): {prop.claimed_quantity} != {atom.focal_quantity}")
    if prop.claimed_comparator_quantity != atom.comparator_quantity:
        errs.append(
            f"I3_TYPED_COMPARATOR_QUANTITY_MISMATCH ({pid}): {prop.claimed_comparator_quantity} != {atom.comparator_quantity}"
        )
    if prop.claimed_statistic != atom.statistic_kind:
        errs.append(f"I3_TYPED_STATISTIC_MISMATCH ({pid}): {prop.claimed_statistic!r} != {atom.statistic_kind!r}")
    n_claimed, n_atom = prop.claimed_sample_size, atom.sample_size
    same_n = (n_claimed is None and n_atom is None) or (type(n_claimed) is int and type(n_atom) is int and n_claimed == n_atom)
    if not same_n:
        errs.append(f"I3_TYPED_SAMPLE_SIZE_MISMATCH ({pid}): {n_claimed!r} != {n_atom!r}")
    return errs


def proposition_from_atom(atom: EvidenceAtom, proposition_id: str) -> TypedProposition:
    """The unique proposition that I3_typed accepts for `atom`."""
    return TypedProposition(
        proposition_id=proposition_id,
        outcome_tuple=atom.outcome_tuple,
        claimed_endpoint=atom.endpoint,
        claimed_focal_arms=atom.contrast.focal_arms,
        claimed_comparator_arms=atom.contrast.comparator_arms,
        claimed_quantity=atom.focal_quantity,
        claimed_statistic=atom.statistic_kind,
        claimed_sample_size=atom.sample_size,
        claimed_comparator_quantity=atom.comparator_quantity,
    )


# ---------------------------------------------------------------------------
# Synthesis records, falsifier gates and the record verifier (I1, I4-I6)
# ---------------------------------------------------------------------------

EpistemicPolarity = Literal[
    "SUPPORTS",
    "CONTRADICTS",
    "SCOPING_QUALIFICATION",
    "CANDIDATE_UPGRADE",
    "FALSIFIED_GRAVEYARD",
]

PromotionDecision = Literal[
    "PROMOTE_QUANTITATIVE_PARAMETER",
    "HOLD_FOR_PROSPECTIVE_FALSIFIER",
    "QUALITATIVE_SCOPING_ONLY",
    "REJECT_INADMISSIBLE_OR_INVALID",
]


@dataclass(frozen=True)
class ProspectiveFalsifierGate:
    """Prespecified experiment required before a bridged claim may be upgraded."""

    gate_id: str
    unresolved_context_axes: tuple[str, ...]
    head_to_head_arms: tuple[str, str]
    primary_readout_and_unit: str
    quantitative_pass_threshold: str

    def validate(self, declared_axes: Mapping[str, Any]) -> None:
        if not isinstance(self.gate_id, str) or not self.gate_id.strip():
            raise ValueError("ProspectiveFalsifierGate requires non-empty gate_id")
        if not isinstance(self.primary_readout_and_unit, str) or not self.primary_readout_and_unit.strip():
            raise ValueError("ProspectiveFalsifierGate requires non-empty primary_readout_and_unit")
        if not isinstance(self.quantitative_pass_threshold, str) or not self.quantitative_pass_threshold.strip():
            raise ValueError("ProspectiveFalsifierGate requires non-empty quantitative_pass_threshold")
        if (
            not isinstance(self.head_to_head_arms, tuple)
            or len(self.head_to_head_arms) != 2
            or not all(isinstance(a, str) and a.strip() for a in self.head_to_head_arms)
            or self.head_to_head_arms[0] == self.head_to_head_arms[1]
        ):
            raise ValueError(f"ProspectiveFalsifierGate requires two distinct non-empty head_to_head_arms: {self.head_to_head_arms!r}")
        if len(set(self.unresolved_context_axes)) != len(self.unresolved_context_axes):
            raise ValueError(f"Duplicate unresolved_context_axes in gate {self.gate_id!r}")
        unknown_axes = [ax for ax in self.unresolved_context_axes if ax not in declared_axes]
        if unknown_axes:
            raise ValueError(f"Undeclared unresolved_context_axes {unknown_axes} in gate {self.gate_id!r}")
        for field_name, val in (
            ("gate_id", self.gate_id),
            ("primary_readout_and_unit", self.primary_readout_and_unit),
            ("quantitative_pass_threshold", self.quantitative_pass_threshold),
            ("head_to_head_arms[0]", self.head_to_head_arms[0]),
            ("head_to_head_arms[1]", self.head_to_head_arms[1]),
        ):
            bad_uni = contains_forbidden_unicode(val)
            if bad_uni:
                raise ValueError(f"Forbidden Unicode in gate {field_name}: {bad_uni}")


ACTIVE_PROSE_KEYS: frozenset[str] = frozenset({"prose_sentences"})


@dataclass(frozen=True)
class SynthesisClaimRecord:
    """Synthesis claim: prose, typed propositions, anchors and transfer metadata."""

    record_id: str
    polarity: EpistemicPolarity
    target_context: ContextSignature
    supporting_tuples: tuple[str, ...]
    counterevidence_tuples: tuple[str, ...]
    prose_sentences: tuple[str, ...]
    typed_propositions: tuple[TypedProposition, ...]
    prospective_gate: ProspectiveFalsifierGate | None = None
    requests_direct_quantitative_promotion: bool = False
    attestation_sha256: str | None = None


def _quantity_to_dict(q: QuantitySpec | None) -> dict[str, Any] | None:
    if q is None:
        return None
    return {
        "kind": q.kind,
        "relation": q.relation,
        "value": q.value,
        "lower": q.lower,
        "upper": q.upper,
        "unit": q.unit,
        "unit_dimension": q.unit_dimension,
        "change_direction": q.change_direction,
        "missing_reason": q.missing_reason,
    }


def _proposition_to_dict(p: TypedProposition) -> dict[str, Any]:
    return {
        "proposition_id": p.proposition_id,
        "outcome_tuple": p.outcome_tuple,
        "claimed_endpoint": p.claimed_endpoint,
        "claimed_focal_arms": list(p.claimed_focal_arms),
        "claimed_comparator_arms": list(p.claimed_comparator_arms),
        "claimed_quantity": _quantity_to_dict(p.claimed_quantity),
        "claimed_comparator_quantity": _quantity_to_dict(p.claimed_comparator_quantity),
        "claimed_statistic": p.claimed_statistic,
        "claimed_sample_size": p.claimed_sample_size,
    }


def _gate_to_dict(g: ProspectiveFalsifierGate | None) -> dict[str, Any] | None:
    if g is None:
        return None
    return {
        "gate_id": g.gate_id,
        "unresolved_context_axes": list(g.unresolved_context_axes),
        "head_to_head_arms": list(g.head_to_head_arms),
        "primary_readout_and_unit": g.primary_readout_and_unit,
        "quantitative_pass_threshold": g.quantitative_pass_threshold,
    }


def record_to_envelope(
    rec: SynthesisClaimRecord,
    extra_metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Serialize `SynthesisClaimRecord` (plus optional extra metadata) as a float-free JSON envelope."""
    env: dict[str, Any] = {
        "record_id": rec.record_id,
        "polarity": rec.polarity,
        "target_context": dict(rec.target_context.axes),
        "supporting_tuples": list(rec.supporting_tuples),
        "counterevidence_tuples": list(rec.counterevidence_tuples),
        "prose_sentences": list(rec.prose_sentences),
        "typed_propositions": [_proposition_to_dict(p) for p in rec.typed_propositions],
        "prospective_gate": _gate_to_dict(rec.prospective_gate),
        "requests_direct_quantitative_promotion": rec.requests_direct_quantitative_promotion,
        "attestation_sha256": rec.attestation_sha256,
    }
    if extra_metadata:
        for k, v in extra_metadata.items():
            if k in env:
                raise ValueError(f"RESERVED_ENVELOPE_KEY: {k!r}")
            env[k] = v
    return env


def non_prose_surface_sha256(
    envelope: Mapping[str, Any],
    active_prose_keys: frozenset[str] = ACTIVE_PROSE_KEYS,
) -> str:
    """Digest over 100% of non-prose keys (`k not in active_prose_keys`)."""
    locked = {k: v for k, v in envelope.items() if k not in active_prose_keys}
    return sha256_canonical(locked)


def verify_non_prose_surface_lock(
    envelope: Mapping[str, Any],
    expected_non_prose_sha256: str,
    active_prose_keys: frozenset[str] = ACTIVE_PROSE_KEYS,
) -> list[str]:
    """Reject any mutation or unverified claim injection in non-prose metadata keys."""
    actual = non_prose_surface_sha256(envelope, active_prose_keys=active_prose_keys)
    if actual != expected_non_prose_sha256:
        return [
            f"I6_NON_PROSE_METADATA_TAMPER: expected {expected_non_prose_sha256}, got {actual}"
        ]
    return []


def compute_claim_attestation(
    rec: SynthesisClaimRecord,
    atoms_by_tuple: Mapping[str, EvidenceAtom],
    arm_aliases: Mapping[str, tuple[str, ...]] | None = None,
) -> dict[str, Any]:
    """Compute the deterministic claim attestation structure over prose mentions and anchors."""
    aliases = arm_aliases or {}
    lexicon = build_entity_lexicon(atoms_by_tuple, aliases) | build_endpoint_phrases(atoms_by_tuple)
    sentence_attestations: list[dict[str, Any]] = []
    for sent in rec.prose_sentences:
        cited = list(dict.fromkeys(TUPLE_RE.findall(sent)))
        mentions = [
            {"value": m.value, "unit": m.unit}
            for m in extract_mentions(sent, lexicon)
        ]
        sentence_attestations.append(
            {
                "sentence_sha256": sha256_canonical(sent),
                "cited_tuples": cited,
                "tuple_bound_numeric_attestations": mentions,
            }
        )
    return {
        "record_id": rec.record_id,
        "polarity": rec.polarity,
        "target_context": dict(rec.target_context.axes),
        "supporting_tuples": list(rec.supporting_tuples),
        "counterevidence_tuples": list(rec.counterevidence_tuples),
        "requests_direct_quantitative_promotion": rec.requests_direct_quantitative_promotion,
        "typed_propositions": [_proposition_to_dict(p) for p in rec.typed_propositions],
        "prospective_gate": _gate_to_dict(rec.prospective_gate),
        "sentence_attestations": sentence_attestations,
    }


def compute_claim_attestation_sha256(
    rec: SynthesisClaimRecord,
    atoms_by_tuple: Mapping[str, EvidenceAtom],
    arm_aliases: Mapping[str, tuple[str, ...]] | None = None,
) -> str:
    return sha256_canonical(compute_claim_attestation(rec, atoms_by_tuple, arm_aliases))


def populate_claim_attestation(
    rec: SynthesisClaimRecord,
    atoms_by_tuple: Mapping[str, EvidenceAtom],
    arm_aliases: Mapping[str, tuple[str, ...]] | None = None,
) -> SynthesisClaimRecord:
    """Return `rec` with a freshly computed `attestation_sha256` (used for Mode B self-attesting controls)."""
    return replace(
        rec,
        attestation_sha256=compute_claim_attestation_sha256(rec, atoms_by_tuple, arm_aliases),
    )


@dataclass
class CEGVerificationReport:
    """Result of verifying one SynthesisClaimRecord against a graph."""

    record_id: str
    passed_all_invariants: bool
    transfer_classes: dict[str, TransferClass]
    unresolved_axes_union: tuple[str, ...]
    promotion_decision: PromotionDecision
    violations: list[str] = field(default_factory=list)


@dataclass
class ContextEvidenceGraph:
    """Context-Evidence Graph container and verifier."""

    lattice: ContextLatticeSpec
    studies: dict[str, ContextSignature]
    study_variables: dict[str, set[str]]
    arm_aliases: dict[str, tuple[str, ...]]
    atoms: dict[str, EvidenceAtom]
    raw_mentioned_arms_by_tuple: dict[str, tuple[str, ...]]

    def verify_graph_integrity(self) -> list[str]:
        """I1 over the whole graph: poset validity, context signatures and atom well-formedness."""
        errs: list[str] = []
        try:
            self.lattice.validate()
        except ValueError as exc:
            return [f"I4_INVALID_CONTEXT_POSET: {exc}"]
        for sid, sig in sorted(self.studies.items()):
            try:
                self.lattice.check_signature(sig)
            except ValueError as exc:
                errs.append(f"I1_INVALID_STUDY_CONTEXT ({sid}): {exc}")
        for tup, atom in sorted(self.atoms.items()):
            if tup != atom.outcome_tuple:
                errs.append(f"I1_ATOM_KEY_MISMATCH: {tup} -> {atom.outcome_tuple}")
                continue
            if atom.study_id not in self.studies:
                errs.append(f"I1_DANGLING_STUDY_CONTEXT: {tup} -> {atom.study_id}")
                continue
            if tup not in self.raw_mentioned_arms_by_tuple:
                errs.append(f"I1_MISSING_SOURCE_ARM_MENTIONS: {tup}")
                continue
            try:
                # An outcome's context may refine its study's context (e.g. an in-vitro sub-assay).
                self.lattice.check_signature(atom.context)
                atom.validate(
                    self.study_variables.get(atom.study_id, set()),
                    self.raw_mentioned_arms_by_tuple[tup],
                )
            except ValueError as exc:
                errs.append(f"I1_ATOM_VALIDATION_ERROR ({tup}): {exc}")
        return errs

    def verify_synthesis_record(
        self,
        rec: SynthesisClaimRecord,
        *,
        pooled_numeral_policy: PooledPolicy = "reject",
        check_attestation_digest: bool = True,
    ) -> CEGVerificationReport:
        """Verify I1, I3_tok, I3_clause, I3_typed, I4, I5 and I6 on a synthesis record.

        When `check_attestation_digest=True` and `rec.attestation_sha256` is populated,
        a stale attestation adds `I6_STALE_CLAIM_ATTESTATION` without short-circuiting
        semantic checks, so Mode A reports both stale-attestation and semantic errors
        while Mode B (self-attesting adversary) is tested purely on semantic grounding.
        """
        violations: list[str] = []
        # The promotion flag must be a genuine boolean; any other type is refused and never requests promotion.
        prom = rec.requests_direct_quantitative_promotion is True
        if type(rec.requests_direct_quantitative_promotion) is not bool:
            violations.append(
                f"I1_INVALID_PROMOTION_BOOLEAN ({rec.record_id}): requests_direct_quantitative_promotion must be a bool, "
                f"got {type(rec.requests_direct_quantitative_promotion).__name__}"
            )
        if check_attestation_digest and rec.attestation_sha256 is not None:
            expected_att = compute_claim_attestation_sha256(rec, self.atoms, self.arm_aliases)
            if rec.attestation_sha256 != expected_att:
                violations.append(
                    f"I6_STALE_CLAIM_ATTESTATION ({rec.record_id}): expected {expected_att}, got {rec.attestation_sha256}"
                )
        if rec.polarity not in EPISTEMIC_ROLES:
            violations.append(f"I1_UNKNOWN_EPISTEMIC_ROLE: {rec.polarity!r}")
        # Record and proposition identifiers are screened for forbidden code points (release 3.1).
        if contains_forbidden_unicode(rec.record_id):
            violations.append(f"I1_FORBIDDEN_UNICODE_IN_RECORD_ID: {rec.record_id!r}")
        for prop in rec.typed_propositions:
            if contains_forbidden_unicode(prop.proposition_id):
                violations.append(f"I1_FORBIDDEN_UNICODE_IN_PROPOSITION_ID: {prop.proposition_id!r}")

        # I6: non-vacuity, polarity-role compatibility, and support/counter-evidence disjointness
        if not rec.supporting_tuples and not rec.counterevidence_tuples:
            violations.append("I6_VACUOUS_SYNTHESIS_RECORD_NO_ANCHORS")
        if rec.polarity in ("SUPPORTS", "CANDIDATE_UPGRADE") and not rec.supporting_tuples:
            violations.append(f"I6_MISSING_SUPPORTING_ANCHORS_FOR_POSITIVE_CLAIM: polarity={rec.polarity!r}")
        if prom and rec.polarity != "SUPPORTS":
            violations.append(f"I6_INCOMPATIBLE_POLARITY_FOR_PROMOTION: polarity={rec.polarity!r}")
        overlap = sorted(set(rec.supporting_tuples) & set(rec.counterevidence_tuples))
        if overlap:
            violations.append(f"I6_SUPPORT_COUNTEREVIDENCE_OVERLAP: {overlap}")

        all_anchors = list(dict.fromkeys(list(rec.supporting_tuples) + list(rec.counterevidence_tuples)))
        for t in all_anchors:
            if t not in self.atoms:
                violations.append(f"I1_DANGLING_ANCHOR_TUPLE: {t}")

        # I3_tok + I3_clause on every prose sentence, plus anchor-scope containment
        anchor_set = set(all_anchors)
        lexicon = build_entity_lexicon(self.atoms, self.arm_aliases)
        for sent in rec.prose_sentences:
            for cited_t in TUPLE_RE.findall(sent):
                if cited_t not in anchor_set:
                    violations.append(f"I1_UNDECLARED_PROSE_TUPLE: {cited_t} not in record anchors")
            violations.extend(
                verify_prose_sentence(
                    sent,
                    self.atoms,
                    self.arm_aliases,
                    pooled_numeral_policy=pooled_numeral_policy,
                    lexicon=lexicon,
                )
            )

        # I3_typed on every structured proposition
        for prop in rec.typed_propositions:
            if prop.outcome_tuple not in anchor_set:
                violations.append(f"I1_UNDECLARED_PROPOSITION_TUPLE: {prop.outcome_tuple}")
            violations.extend(verify_typed_proposition_homomorphism(prop, self.atoms))

        # Promotion needs a typed proposition for every supporting tuple
        if prom:
            if not rec.typed_propositions:
                violations.append("I3_TYPED_MISSING_FOR_QUANTITATIVE_PROMOTION")
            else:
                covered = {p.outcome_tuple for p in rec.typed_propositions}
                uncovered = [t for t in rec.supporting_tuples if t not in covered]
                if uncovered:
                    violations.append(f"I3_TYPED_PROMOTION_COVERAGE_GAP: no typed proposition for {uncovered}")

        # I4 (target context well-formedness and context transfer) and I5 (primary source and figure digitization)
        target_context_valid = True
        try:
            self.lattice.check_signature(rec.target_context)
        except ValueError as exc:
            target_context_valid = False
            violations.append(f"I4_TARGET_CONTEXT_INVALID: {exc}")

        transfer_map: dict[str, TransferClass] = {}
        unresolved_union: set[str] = set()
        any_unverified_or_undigitized = False

        for t in rec.supporting_tuples:
            if t not in self.atoms:
                continue
            atom = self.atoms[t]
            if (
                type(atom.provenance.requires_figure_digitization) is not bool
                or type(atom.provenance.primary_pdf_verified) is not bool
            ):
                any_unverified_or_undigitized = True
                violations.append(
                    f"I1_INVALID_PROVENANCE_BOOLEAN ({t}): "
                    f"requires_figure_digitization={atom.provenance.requires_figure_digitization!r}, "
                    f"primary_pdf_verified={atom.provenance.primary_pdf_verified!r}"
                )
            elif atom.provenance.requires_figure_digitization or not atom.provenance.primary_pdf_verified:
                any_unverified_or_undigitized = True
            if not target_context_valid:
                continue
            try:
                t_cls, unres = self.lattice.evaluate_transfer(atom.context, rec.target_context)
            except ValueError as exc:
                violations.append(f"I4_CONTEXT_SIGNATURE_INVALID ({t}): {exc}")
                continue
            transfer_map[t] = t_cls
            unresolved_union.update(unres)

        if rec.prospective_gate is not None:
            try:
                rec.prospective_gate.validate(self.lattice.axis_posets)
            except ValueError as exc:
                violations.append(f"I4_INVALID_PROSPECTIVE_FALSIFIER_GATE: {exc}")

        if prom:
            if any(v != "DIRECT_MATCH" for v in transfer_map.values()) or not transfer_map:
                violations.append(
                    f"I4_ILLEGAL_QUANTITATIVE_PROMOTION_ACROSS_CONTEXT_GAP: transfers={transfer_map}, unresolved={sorted(unresolved_union)}"
                )
            if any_unverified_or_undigitized:
                violations.append("I5_ILLEGAL_QUANTITATIVE_PROMOTION_WITHOUT_PRIMARY_OR_DIGITIZED_SOURCE")
            # A quantitative promotion needs a quantity: at least one supporting atom must hold a number.
            if transfer_map and not any(
                self.atoms[t].focal_quantity.kind != "missing_numeric"
                or (
                    self.atoms[t].comparator_quantity is not None
                    and self.atoms[t].comparator_quantity.kind != "missing_numeric"
                )
                for t in transfer_map
            ):
                violations.append("I5_ILLEGAL_QUANTITATIVE_PROMOTION_OF_MISSING_NUMERIC")

        if rec.polarity in ("SUPPORTS", "CANDIDATE_UPGRADE") and any(
            v == "INADMISSIBLE_TRANSFER" for v in transfer_map.values()
        ):
            violations.append(f"I4_INADMISSIBLE_CONTEXT_TRANSFER: transfers={transfer_map}")

        if unresolved_union:
            if rec.polarity == "CANDIDATE_UPGRADE" and rec.prospective_gate is None:
                violations.append(f"I4_MISSING_PROSPECTIVE_FALSIFIER_GATE: unresolved_axes={sorted(unresolved_union)}")
            elif rec.prospective_gate is not None:
                uncovered_axes = sorted(unresolved_union - set(rec.prospective_gate.unresolved_context_axes))
                if uncovered_axes:
                    violations.append(f"I4_INCOMPLETE_PROSPECTIVE_FALSIFIER_AXES: uncovered={uncovered_axes}")

        passed = len(violations) == 0
        if not passed:
            decision: PromotionDecision = "REJECT_INADMISSIBLE_OR_INVALID"
        elif prom:
            decision = "PROMOTE_QUANTITATIVE_PARAMETER"
        elif unresolved_union or any_unverified_or_undigitized:
            decision = (
                "HOLD_FOR_PROSPECTIVE_FALSIFIER"
                if rec.prospective_gate is not None
                else "QUALITATIVE_SCOPING_ONLY"
            )
        else:
            decision = "QUALITATIVE_SCOPING_ONLY"

        return CEGVerificationReport(
            record_id=rec.record_id,
            passed_all_invariants=passed,
            transfer_classes=transfer_map,
            unresolved_axes_union=tuple(sorted(unresolved_union)),
            promotion_decision=decision,
            violations=violations,
        )


# ---------------------------------------------------------------------------
# Atom-to-corpus binding (I1): atoms must agree with the settled corpus D_m
# ---------------------------------------------------------------------------

_RELATION_PREFIX_RE = re.compile(r"^(>=|<=|>|<|≈|=)?\s*(.+)$")
_ALLOWED_CORPUS_OUTCOME_KEYS = frozenset(
    {
        "outcome_id",
        "endpoint",
        "unit",
        "comparator_unit",
        "unit_dimension",
        "quantity_kind",
        "kind",
        "comparator_kind",
        "focal_value",
        "value",
        "relation",
        "lower",
        "focal_lower",
        "upper",
        "focal_upper",
        "change_direction",
        "missing_reason",
        "comparator_value",
        "comparator",
        "comparator_relation",
        "comparator_lower",
        "comparator_upper",
        "statistic_kind",
        "statistic",
        "requires_figure_digitization",
        "primary_pdf_verified",
        "document_id",
        "source_id",
        "section_pointer",
        "page_or_table_pin",
        "verbatim_quote",
        "verbatim_span",
        "comparator_arms",
        "focal_arms",
        "sample_size",
    }
)


def verify_atoms_bound_to_corpus(graph: ContextEvidenceGraph, settled_corpus: Mapping[str, Any]) -> list[str]:
    """Field-wise agreement between every atom and its record in the settled corpus.

    I2 certifies the path D_0 -> D_m; this check ties the atoms the synthesis is
    verified against to D_m. Every corpus outcome needs a unique atom and every
    atom a unique corpus outcome; every field carried by the corpus outcome record
    must be recognized and equal the atom's field without type coercion.
    """
    errs: list[str] = []
    seen: set[str] = set()
    seen_studies: set[str] = set()
    studies = settled_corpus.get("studies", [])
    if not isinstance(studies, list):
        return ["I1_INVALID_CORPUS_STUDIES_CONTAINER"]

    for st in studies:
        if not isinstance(st, Mapping) or not isinstance(st.get("study_id"), str):
            errs.append(f"I1_INVALID_CORPUS_STUDY_ENTRY: {st!r}")
            continue
        sid = st["study_id"]
        if sid in seen_studies:
            errs.append(f"I1_DUPLICATE_CORPUS_STUDY: {sid}")
        seen_studies.add(sid)
        outcomes = st.get("outcomes", [])
        if not isinstance(outcomes, list):
            errs.append(f"I1_INVALID_CORPUS_OUTCOMES_CONTAINER: {sid}")
            continue
        for oc in outcomes:
            if not isinstance(oc, Mapping) or not isinstance(oc.get("outcome_id"), str):
                errs.append(f"I1_INVALID_CORPUS_OUTCOME_ENTRY ({sid}): {oc!r}")
                continue
            tup = f"{sid}:{oc['outcome_id']}"
            if tup in seen:
                errs.append(f"I1_DUPLICATE_CORPUS_OUTCOME: {tup}")
                continue
            seen.add(tup)
            atom = graph.atoms.get(tup)
            if atom is None:
                errs.append(f"I1_CORPUS_OUTCOME_WITHOUT_ATOM: {tup}")
                continue
            unknown_keys = sorted(set(oc.keys()) - _ALLOWED_CORPUS_OUTCOME_KEYS)
            if unknown_keys:
                errs.append(f"I1_UNRECOGNIZED_CORPUS_OUTCOME_FIELD ({tup}): {unknown_keys}")

            fq, cq = atom.focal_quantity, atom.comparator_quantity

            def differ(field_name: str, corpus_val: Any, atom_val: Any) -> None:
                if not _json_equal(corpus_val, atom_val):
                    errs.append(f"I1_ATOM_CORPUS_FIELD_MISMATCH ({tup}.{field_name}): corpus {corpus_val!r} != atom {atom_val!r}")

            if "endpoint" in oc:
                differ("endpoint", oc["endpoint"], atom.endpoint)
            if "unit" in oc:
                differ("unit", oc["unit"], fq.unit)
                if cq is not None and cq.unit is not None and "comparator_unit" not in oc:
                    differ("unit (comparator)", oc["unit"], cq.unit)
            if "comparator_unit" in oc:
                differ("comparator_unit", oc["comparator_unit"], cq.unit if cq is not None else None)
            if "unit_dimension" in oc:
                differ("unit_dimension", oc["unit_dimension"], fq.unit_dimension)
            for key in ("quantity_kind", "kind"):
                if key in oc:
                    differ(key, oc[key], fq.kind)
            if "comparator_kind" in oc:
                differ("comparator_kind", oc["comparator_kind"], cq.kind if cq is not None else None)
            for key in ("focal_value", "value"):
                if key in oc:
                    raw_v = oc[key]
                    if raw_v is None and fq.kind == "missing_numeric":
                        differ(key, None, fq.value)
                    elif not isinstance(raw_v, str):
                        errs.append(f"I1_ATOM_CORPUS_FIELD_MISMATCH ({tup}.{key}): non-string corpus value {raw_v!r}")
                    else:
                        m = _RELATION_PREFIX_RE.match(raw_v)
                        relation, literal = (m.group(1), m.group(2)) if m else (None, raw_v)
                        differ(key, literal, fq.value)
                        if "relation" not in oc:
                            differ(f"{key}.relation", relation or "=", fq.relation)
                        elif relation is not None:
                            differ(f"{key}.relation", relation, fq.relation)
            if "relation" in oc:
                differ("relation", oc["relation"], fq.relation)
            for key in ("lower", "focal_lower"):
                if key in oc:
                    differ(key, oc[key], fq.lower)
            for key in ("upper", "focal_upper"):
                if key in oc:
                    differ(key, oc[key], fq.upper)
            if "change_direction" in oc:
                differ("change_direction", oc["change_direction"], fq.change_direction)
            if "missing_reason" in oc:
                differ("missing_reason", oc["missing_reason"], fq.missing_reason)
            for key in ("comparator_value", "comparator"):
                if key in oc:
                    raw_cv = oc[key]
                    if cq is None:
                        differ(key, raw_cv, None)
                    elif raw_cv is None and cq.kind == "missing_numeric":
                        differ(key, None, cq.value)
                    elif not isinstance(raw_cv, str):
                        errs.append(f"I1_ATOM_CORPUS_FIELD_MISMATCH ({tup}.{key}): non-string corpus value {raw_cv!r}")
                    else:
                        m = _RELATION_PREFIX_RE.match(raw_cv)
                        relation, literal = (m.group(1), m.group(2)) if m else (None, raw_cv)
                        differ(key, literal, cq.value)
                        if "comparator_relation" not in oc:
                            differ(f"{key}.relation", relation or "=", cq.relation)
                        elif relation is not None:
                            differ(f"{key}.relation", relation, cq.relation)
            if "comparator_relation" in oc:
                differ("comparator_relation", oc["comparator_relation"], cq.relation if cq is not None else None)
            if "comparator_lower" in oc:
                differ("comparator_lower", oc["comparator_lower"], cq.lower if cq is not None else None)
            if "comparator_upper" in oc:
                differ("comparator_upper", oc["comparator_upper"], cq.upper if cq is not None else None)
            for key in ("statistic_kind", "statistic"):
                if key in oc:
                    differ(key, oc[key], atom.statistic_kind)
            for bkey in ("requires_figure_digitization", "primary_pdf_verified"):
                if bkey in oc:
                    if type(oc[bkey]) is not bool:
                        errs.append(f"I1_ATOM_CORPUS_FIELD_MISMATCH ({tup}.{bkey}): non-bool {oc[bkey]!r}")
                    else:
                        differ(bkey, oc[bkey], getattr(atom.provenance, bkey))
            for key in ("document_id", "source_id"):
                if key in oc:
                    differ(key, oc[key], atom.provenance.document_id)
            for key in ("section_pointer", "page_or_table_pin"):
                if key in oc:
                    differ(key, oc[key], atom.provenance.section_pointer)
            for key in ("verbatim_quote", "verbatim_span"):
                if key in oc:
                    differ(key, oc[key], atom.provenance.verbatim_quote)
            if "comparator_arms" in oc:
                if not isinstance(oc["comparator_arms"], list):
                    errs.append(f"I1_ATOM_CORPUS_FIELD_MISMATCH ({tup}.comparator_arms): non-list {oc['comparator_arms']!r}")
                else:
                    differ("comparator_arms", oc["comparator_arms"], list(atom.contrast.comparator_arms))
            if "focal_arms" in oc:
                if not isinstance(oc["focal_arms"], list):
                    errs.append(f"I1_ATOM_CORPUS_FIELD_MISMATCH ({tup}.focal_arms): non-list {oc['focal_arms']!r}")
                else:
                    differ("focal_arms", oc["focal_arms"], list(atom.contrast.focal_arms))
            if "sample_size" in oc:
                if oc["sample_size"] is not None and type(oc["sample_size"]) is not int:
                    errs.append(f"I1_ATOM_CORPUS_FIELD_MISMATCH ({tup}.sample_size): non-int {oc['sample_size']!r}")
                else:
                    differ("sample_size", oc["sample_size"], atom.sample_size)
    for sid in sorted(graph.studies):
        if sid not in seen_studies:
            errs.append(f"I1_STUDY_WITHOUT_CORPUS_RECORD: {sid}")
    for sid in sorted(seen_studies):
        if sid not in graph.studies:
            errs.append(f"I1_CORPUS_STUDY_WITHOUT_GRAPH_RECORD: {sid}")
    for tup in sorted(graph.atoms):
        if tup not in seen:
            errs.append(f"I1_ATOM_WITHOUT_CORPUS_RECORD: {tup}")
    return errs


# ---------------------------------------------------------------------------
# Two-layer CEG package verification: structure-only vs digest-locked
# ---------------------------------------------------------------------------

def _atom_to_dict(atom: EvidenceAtom) -> dict[str, Any]:
    return {
        "atom_id": atom.atom_id,
        "outcome_tuple": atom.outcome_tuple,
        "study_id": atom.study_id,
        "endpoint": atom.endpoint,
        "context": dict(atom.context.axes),
        "contrast": {
            "focal_arms": list(atom.contrast.focal_arms),
            "comparator_arms": list(atom.contrast.comparator_arms),
            "contrast_kind": atom.contrast.contrast_kind,
        },
        "focal_quantity": _quantity_to_dict(atom.focal_quantity),
        "comparator_quantity": _quantity_to_dict(atom.comparator_quantity),
        "statistic_kind": atom.statistic_kind,
        "sample_size": atom.sample_size,
        "p_value_or_interval": atom.p_value_or_interval,
        "polarity": atom.polarity,
        "provenance": {
            "document_id": atom.provenance.document_id,
            "section_pointer": atom.provenance.section_pointer,
            "verbatim_quote": atom.provenance.verbatim_quote,
            "requires_figure_digitization": atom.provenance.requires_figure_digitization,
            "primary_pdf_verified": atom.provenance.primary_pdf_verified,
        },
    }


_CORE_NON_DIGEST_ENVELOPE_KEYS: tuple[str, ...] = (
    "record_id",
    "polarity",
    "target_context",
    "supporting_tuples",
    "counterevidence_tuples",
    "prose_sentences",
    "typed_propositions",
    "prospective_gate",
    "requests_direct_quantitative_promotion",
)


def verify_envelopes_bound_to_records(
    records: Sequence[SynthesisClaimRecord],
    record_envelopes: Sequence[Mapping[str, Any]],
    *,
    check_attestation_digest: bool = True,
) -> list[str]:
    """Verify bijective, order-preserving core-field equality between `records` and `record_envelopes`."""
    errs: list[str] = []
    if len(record_envelopes) != len(records):
        errs.append(
            f"I6_ENVELOPE_RECORD_COUNT_MISMATCH: expected {len(records)} envelopes, got {len(record_envelopes)}"
        )
    required_keys = (
        _CORE_NON_DIGEST_ENVELOPE_KEYS + ("attestation_sha256",)
        if check_attestation_digest
        else _CORE_NON_DIGEST_ENVELOPE_KEYS
    )
    for idx, (rec, env) in enumerate(zip(records, record_envelopes)):
        if not isinstance(env, Mapping):
            errs.append(f"I6_INVALID_RECORD_ENVELOPE ({idx}): expected mapping, got {type(env).__name__}")
            continue
        canonical_env = record_to_envelope(rec)
        for key in required_keys:
            if key not in env or not _json_equal(env[key], canonical_env[key]):
                errs.append(
                    f"I6_ENVELOPE_RECORD_MISMATCH ({rec.record_id}.{key}): "
                    f"envelope {env.get(key)!r} != record {canonical_env[key]!r}"
                )
    return errs


def compute_ceg_section_digests(
    graph: ContextEvidenceGraph,
    initial_corpus: Mapping[str, Any],
    settled_corpus: Mapping[str, Any],
    patch_events: Sequence[ReversiblePatchEvent],
    records: Sequence[SynthesisClaimRecord],
    *,
    record_envelopes: Sequence[Mapping[str, Any]] | None = None,
) -> dict[str, str]:
    """Compute independent cryptographic section digests for Layer-2 digest locks."""
    if record_envelopes is not None:
        env_errs = verify_envelopes_bound_to_records(
            records, record_envelopes, check_attestation_digest=True
        )
        if env_errs:
            raise ValueError(env_errs[0])
        envelopes = list(record_envelopes)
    else:
        envelopes = [record_to_envelope(r) for r in records]
    return {
        "initial_corpus_sha256": sha256_canonical(dict(initial_corpus)),
        "settled_corpus_sha256": sha256_canonical(dict(settled_corpus)),
        "ledger_sha256": ledger_sha256(patch_events),
        "atoms_sha256": sha256_canonical({t: _atom_to_dict(a) for t, a in sorted(graph.atoms.items())}),
        "records_non_prose_sha256": sha256_canonical([non_prose_surface_sha256(env) for env in envelopes]),
    }


def verify_ceg_package(
    graph: ContextEvidenceGraph,
    initial_corpus: dict[str, Any],
    settled_corpus: dict[str, Any],
    patch_events: Sequence[ReversiblePatchEvent],
    records: Sequence[SynthesisClaimRecord],
    *,
    section_digests: Mapping[str, str] | None = None,
    record_envelopes: Sequence[Mapping[str, Any]] | None = None,
    check_top_digests: bool = True,
    pooled_numeral_policy: PooledPolicy = "reject",
) -> dict[str, Any]:
    """Verify a full CEG package in either structure-only (`check_top_digests=False`) or two-layer mode.

    Layer 1 (`check_top_digests=False`) enforces structural projection closure
    from first principles: poset axioms, study/outcome catalogue counts, node/edge
    referential closure, interval bounds, atom-to-corpus field equality, forward
    and inverse value-transition patch replay, and `I1`/`I3`/`I4`/`I5`/`I6` over
    every synthesis record — with precomputed section and step digests disabled.
    Layer 2 (`check_top_digests=True`) adds independent cryptographic section,
    per-step study, per-claim attestation, and non-prose metadata digest locks.
    """
    violations: list[str] = []
    violations.extend(graph.verify_graph_integrity())
    try:
        verify_bijective_patch_replay(
            initial_corpus,
            settled_corpus,
            patch_events,
            check_step_digests=check_top_digests,
        )
    except ValueError as exc:
        violations.append(str(exc))
    violations.extend(verify_atoms_bound_to_corpus(graph, settled_corpus))

    reports: list[CEGVerificationReport] = []
    for rec in records:
        rep = graph.verify_synthesis_record(
            rec,
            pooled_numeral_policy=pooled_numeral_policy,
            check_attestation_digest=check_top_digests,
        )
        reports.append(rep)
        for v in rep.violations:
            violations.append(f"{rec.record_id}: {v}")

    env_errs: list[str] = []
    if record_envelopes is not None:
        env_errs = verify_envelopes_bound_to_records(
            records,
            record_envelopes,
            check_attestation_digest=check_top_digests,
        )
        violations.extend(env_errs)

    if check_top_digests and section_digests is not None:
        actual = compute_ceg_section_digests(
            graph,
            initial_corpus,
            settled_corpus,
            patch_events,
            records,
            record_envelopes=record_envelopes if not env_errs else None,
        )
        for key, expected_val in sorted(section_digests.items()):
            if actual.get(key) != expected_val:
                violations.append(
                    f"DIGEST_LOCK_MISMATCH ({key}): expected {expected_val}, got {actual.get(key)}"
                )

    return {
        "passed": len(violations) == 0,
        "check_top_digests": check_top_digests,
        "studies_verified": len(graph.studies),
        "atoms_verified": len(graph.atoms),
        "events_verified": len(patch_events),
        "records_verified": len(records),
        "violations": violations,
        "record_reports": reports,
    }
