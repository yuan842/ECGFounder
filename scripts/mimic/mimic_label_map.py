"""Map MIMIC-IV-ECG machine-interpretation free text to the 9 unified V3.1 heads.

MIMIC-IV-ECG ships per-record auto-interpretation statements from GE MUSE in
`machine_measurements.csv`, distributed across columns `report_0`–`report_17`.
Each cell contains a short free-text diagnostic phrase (e.g. "atrial
fibrillation", "premature ventricular complexes", "sinus bradycardia").

This module defines the 9 target heads and a substring-based mapper from
the MIMIC text to the corresponding 150-class Founder idx.

The 9 target heads (V3.1 single-head ontology minus ST Elevation, which has
insufficient data per the cross-mapping Excel):
  4 SINUS BRADYCARDIA
  5 ATRIAL FIBRILLATION
  6 SINUS TACHYCARDIA
  9 PREMATURE VENTRICULAR COMPLEXES
  16 PREMATURE ATRIAL COMPLEXES
  19 PREMATURE SUPRAVENTRICULAR COMPLEXES
  93 SUPRAVENTRICULAR TACHYCARDIA
  98 VENTRICULAR TACHYCARDIA
  142 WITH SINUS PAUSE

Each entry below pairs the target head with the set of LOWERCASE substring
matches that should drive its positive label. Substring matching is order-
sensitive — more-specific phrases are checked first (e.g. "supraventricular
tachycardia" before "tachycardia") via the order of the list.
"""
from __future__ import annotations
import re
from typing import Iterable

# (idx, canonical_head_name, [list of lowercase substrings that signal a positive])
HEAD_SPEC: list[tuple[int, str, list[str]]] = [
    # idx 4  — SINUS BRADYCARDIA
    (4, "SINUS BRADYCARDIA", [
        "sinus bradycardia",
        "marked sinus bradycardia",
        "bradycardia",        # broad fallback
    ]),
    # idx 5  — ATRIAL FIBRILLATION
    (5, "ATRIAL FIBRILLATION", [
        "atrial fibrillation",
        "atrial fib",
        "afib",
        "a-fib",
    ]),
    # idx 6  — SINUS TACHYCARDIA
    (6, "SINUS TACHYCARDIA", [
        "sinus tachycardia",
        "sinus tach",
    ]),
    # idx 9  — PREMATURE VENTRICULAR COMPLEXES
    (9, "PREMATURE VENTRICULAR COMPLEXES", [
        "premature ventricular complex",     # PVC
        "premature ventricular contraction",
        "ventricular ectopic",
        "ventricular premature",
        "pvc",
        "ventricular bigeminy",   # ambiguous but a positive for PVC head
        "ventricular trigeminy",  # same
        "ventricular couplet",
    ]),
    # idx 16 — PREMATURE ATRIAL COMPLEXES
    (16, "PREMATURE ATRIAL COMPLEXES", [
        "premature atrial complex",
        "premature atrial contraction",
        "atrial premature",
        "atrial ectopic",
        "atrial bigeminy",
        "atrial trigeminy",
        "pac",
    ]),
    # idx 19 — PREMATURE SUPRAVENTRICULAR COMPLEXES
    (19, "PREMATURE SUPRAVENTRICULAR COMPLEXES", [
        "premature supraventricular complex",
        "supraventricular premature",
        "supraventricular ectopic",
        "supraventricular couplet",
        "psvc",
    ]),
    # idx 93 — SUPRAVENTRICULAR TACHYCARDIA  (matched BEFORE generic "tachycardia")
    (93, "SUPRAVENTRICULAR TACHYCARDIA", [
        "supraventricular tachycardia",
        "svt",
        "atrioventricular nodal reentrant tachycardia",
        "avnrt",
        "atrioventricular reentrant tachycardia",
        "avrt",
    ]),
    # idx 98 — VENTRICULAR TACHYCARDIA   (matched BEFORE generic "tachycardia")
    (98, "VENTRICULAR TACHYCARDIA", [
        "ventricular tachycardia",
        "v-tach",
        "vt ",
        "nonsustained ventricular tachycardia",
        "nsvt",
        "vent tachycardia",
    ]),
    # idx 142 — WITH SINUS PAUSE
    (142, "WITH SINUS PAUSE", [
        "sinus pause",
        "sinoatrial pause",
        "sinus arrest",
        "asystole",
    ]),
]

# Head indices in canonical order (must match the column order of the labels matrix).
HEAD_IDX: list[int] = [h[0] for h in HEAD_SPEC]
HEAD_NAMES: list[str] = [h[1] for h in HEAD_SPEC]


_SUPRA_RE = re.compile(r"\bsupra(?:[\-\s]?ventricular)\b", re.IGNORECASE)


def label_from_text(text: str) -> tuple[int, ...]:
    """Return tuple of head idx for which the text triggers a positive.

    Empty tuple = no positive on any of the 9 heads.

    Implementation note: "ventricular X" appears as a literal substring of
    "supraventricular X" (e.g. "supraventricular tachycardia" ⊃
    "ventricular tachycardia"). To prevent the V-pattern heads (idx 9, 98)
    from firing on supra-patterns, we mask occurrences of "supraventricular"
    out of the normalized text BEFORE running the V-pattern substring checks.
    Supra-pattern heads (idx 16, 19, 93) are evaluated on the original text.
    """
    if not text:
        return ()
    norm = text.lower()
    # Build a version of the text where every "supraventricular" is replaced
    # with a sentinel that does not collide with any V-pattern keyword.
    norm_v_safe = _SUPRA_RE.sub("supravent_", norm)

    hit = []
    V_HEADS = {9, 98}  # use the masked text for these
    for idx, _name, kws in HEAD_SPEC:
        target = norm_v_safe if idx in V_HEADS else norm
        for kw in kws:
            if kw in target:
                hit.append(idx)
                break
    return tuple(hit)


def labels_from_report_columns(report_cells: Iterable[str]) -> tuple[int, ...]:
    """Apply `label_from_text` over multiple MIMIC report_0..report_17 cells and
    union the positives. Returns a tuple of head idx that fire."""
    hits: set[int] = set()
    for cell in report_cells:
        if cell is None:
            continue
        hits.update(label_from_text(str(cell)))
    return tuple(sorted(hits))


def labels_to_binary_vector(hits: Iterable[int]) -> list[int]:
    """Convert a set of fired head idx into a binary (9,) vector ordered by
    HEAD_IDX."""
    hit_set = set(hits)
    return [1 if i in hit_set else 0 for i in HEAD_IDX]


def labels_to_founder_vector(hits: Iterable[int]) -> list[int]:
    """Convert a set of fired head idx into a binary (150,) vector matching
    the full Founder vocabulary (other 141 heads = 0)."""
    vec = [0] * 150
    for i in hits:
        if 0 <= i < 150:
            vec[i] = 1
    return vec


if __name__ == '__main__':
    # Smoke tests on synthetic strings
    cases = [
        ("Atrial fibrillation present, rapid ventricular response.", {5}),
        ("Sinus bradycardia at 55 bpm.", {4}),
        ("Supraventricular tachycardia — non-sustained.", {93}),
        ("Ventricular tachycardia, 6-beat run.", {98}),
        ("Sinus rhythm with premature atrial complexes.", {16}),
        ("Frequent PVCs and ventricular bigeminy.", {9}),
        ("Normal sinus rhythm.", set()),
        ("Sinus pause >2.5 s detected.", {142}),
        ("Supraventricular couplet noted.", {19}),  # specific match — should NOT also hit idx 93
        ("Sinus tach at 110.", {6}),                # should NOT also hit 93 or 98
    ]
    for text, expected in cases:
        got = set(label_from_text(text))
        ok = "✓" if got == expected else "✗"
        print(f"  {ok}  {text!r} → got={sorted(got)}, expected={sorted(expected)}")
