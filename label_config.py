"""
label_config.py — Unified single-index label mapping (v3) for the
ECGFounder 150-class model.

Single source of truth for dataset → 150-class index mappings across:
  * ecg-tp-fzark and ecg-fp-doctor-removed (fzark / FP datasets)
  * MIT-BIH arrhythmia annotations
  * PTB-XL (active-class set)

v3 design principle: every supported event maps to **exactly one** ECGFounder
index. No multi-head logic, no MAX/AND operators, no partial-match heads.
Vocabulary-limited events (Prolonged RR Interval, Ventricular Bigeminy,
Ventricular Trigeminy) remain passed through — the 150-class output has no
head with strong activation on these.

v3.1 update (2026-05-28): three composite events are recovered via their
*constituent beat* head — Supraventricular Trigeminy and Supraventricular
Bigeminy route to PREMATURE ATRIAL COMPLEXES (idx 16, the same head used for
Isolated Supraventricular Beat); Ventricular Couplet routes to PREMATURE
VENTRICULAR COMPLEXES (idx 9, the same head used for Isolated Ventricular
Beat). The head fires on the constituent beats rather than the pattern, so
event-level recall is achieved while pattern discrimination (couplet vs.
isolated beat, trigeminy vs. bigeminy) is left to downstream beat-pattern
analysis. Recall lift measured on ecg_tp_fzark at t=0.5: SV-Trigeminy 5.3% →
96.1%, SV-Bigeminy 64.1% → 78.1%, V-Couplet 41.7% → 80.6%.

See `res/label_reclassification_plan.md` (v3) for the full design rationale
and `res/cross_dataset_supp_off/cross_dataset_performance_supp_off.md` §8 for
the head-correlation analysis that motivated the v3.1 recovery.
"""
from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from enum import Enum
from typing import Optional


# ═══════════════════════════════════════════════════════════════════════════
# 150-class vocabulary integrity check
# ═══════════════════════════════════════════════════════════════════════════

TASKS_FILE: str = "tasks.txt"

# SHA256 of the canonical tasks.txt — fail loudly if the vocabulary drifts.
# Computed once on 2026-05-27 against the unchanged upstream file
# (origin: Shenda Hong, 2025-06-09).
TASKS_SHA256: str = "7c087a0d383c7ea9ae9b818ad00a551b029334edfa5f1fbb5036bfbaa0df0c39"


def load_tasks(path: str = TASKS_FILE) -> list[str]:
    """Return the 150-class label list, after verifying the SHA256 hash.

    Raises:
        FileNotFoundError: if `path` does not exist.
        RuntimeError: if the file's SHA256 does not match `TASKS_SHA256`.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{path} not found. The 150-class vocabulary is required for "
            f"label_config to function."
        )
    with open(path, "rb") as f:
        data = f.read()
    actual = hashlib.sha256(data).hexdigest()
    if actual != TASKS_SHA256:
        raise RuntimeError(
            f"tasks.txt hash mismatch.\n"
            f"  expected: {TASKS_SHA256}\n"
            f"  actual:   {actual}\n"
            f"The 150-class vocabulary has changed; ontology mappings are invalid. "
            f"If the change was intentional, update TASKS_SHA256 in label_config.py."
        )
    return [line.strip() for line in data.decode().splitlines() if line.strip()]


# ═══════════════════════════════════════════════════════════════════════════
# Ontology schema (v3: single-index only)
# ═══════════════════════════════════════════════════════════════════════════

class ClinicalRiskTier(Enum):
    CRITICAL = "critical"   # life-threatening if missed (VT, pause, STEMI)
    HIGH     = "high"       # significant clinical event (AF, bradycardia)
    MODERATE = "moderate"   # actionable but lower urgency (couplets, runs)
    LOW      = "low"        # benign or contextual (isolated ectopy, sinus tach)


class ClinicalReliability(Enum):
    """Empirical reliability tier from the production fzark + ECG-FP cohort.

    Computed PPV = TP / (TP + FP) at the head's default fire threshold.
    Sourced from `Dataset Classification Cross-Mapping.xlsx` (2026-05-28).
    """
    RELIABLE          = "reliable"               # PPV >= 80%
    MODERATE_FP       = "moderate_fp"            # 20% <= PPV < 80%
    SEVERE_FP         = "severe_fp"              # PPV < 20%
    INSUFFICIENT_DATA = "insufficient_data"      # n < 5 in fzark


@dataclass(frozen=True)
class ClinicalOntologyNode:
    """One row of the fzark → ECGFounder ontology.

    v3 invariant: exactly one ECGFounder index per event. No operator field;
    no multi-head logic. Composite events live in FZARK_UNMAPPABLE.

    v3.1 fields (from `Dataset Classification Cross-Mapping.xlsx`):
      - fzark_ppv_pct         : empirical PPV on the production cohort
      - clinical_reliability  : reliability tier derived from PPV
      - clinical_support      : scientific / clinical description
    """
    canonical_name: str            # human-readable identifier
    ecgfounder_index: int          # exactly one 150-class index
    risk_tier: ClinicalRiskTier
    semantic_match: str            # "exact" | "good"
    notes: str = ""
    # — v3.1 reliability metadata (Excel cross-mapping) —
    fzark_ppv_pct: Optional[float] = None        # TP / (TP+FP) × 100, None if no fzark data
    clinical_reliability: Optional[ClinicalReliability] = None
    clinical_support: str = ""


# ═══════════════════════════════════════════════════════════════════════════
# fzark / ECG-FP ontology — 13 supported events (v3.1)
# ═══════════════════════════════════════════════════════════════════════════

# ── Reliability metadata sourced from
#    `Dataset Classification Cross-Mapping.xlsx` (2026-05-28) ──────────────
# Each head's PPV is computed from production fzark TP + ECG-FP cohorts.
# Heads that share an index in v3.1 (PAC head idx 16, PVC head idx 9) share
# the same PPV / reliability tier — they reflect the head, not the event.
_HEAD_RELIABILITY: dict[int, tuple[float, ClinicalReliability, str]] = {
    4:  (26.6, ClinicalReliability.MODERATE_FP,       "SA node pacing at a rate <60 bpm. Clinically maps directly to the generalized \"Bradycardia\" alerting event."),
    5:  (91.6, ClinicalReliability.RELIABLE,          "Irregularly irregular atrial rhythm lacking distinct P waves. Maps directly to \"Atrial Fibrillation\" continuous rhythm alerts."),
    6:  (0.0,  ClinicalReliability.SEVERE_FP,         "SA node pacing at a rate >100 bpm. Clinically maps directly to \"Sinus Tachycardia\" physiological arousal or stress alerts."),
    9:  (15.3, ClinicalReliability.SEVERE_FP,         "Ectopic beats originating from ventricles. Mapped to \"Isolated Ventricular Beat\" and \"Ventricular Couplet\" as these are the exact temporal event manifestations of PVC burden."),
    16: (7.6,  ClinicalReliability.SEVERE_FP,         "Ectopic supraventricular beats. Mapped to \"Isolated Supraventricular Beat\", \"Bigeminy\", and \"Trigeminy\", as these represent the isolated and patterned temporal variations of atrial ectopy."),
    19: (16.7, ClinicalReliability.SEVERE_FP,         "Broad category for early beats above the ventricles. Specifically mapped to \"Supraventricular Couplet\" to capture paired ectopic firing before escalating to a run/tachycardia."),
    68: (100.0, ClinicalReliability.INSUFFICIENT_DATA, "Acute transmural myocardial ischemia/injury. Clinically maps directly to the acute \"ST Elevation\" Fzark alert signaling potential STEMI."),
    93: (0.1,  ClinicalReliability.SEVERE_FP,         "Rapid rhythm (usually >150 bpm) originating above the ventricles. Mapped to \"Supraventricular Run\" which is the algorithmic detection of 3+ consecutive SVTs."),
    98: (62.2, ClinicalReliability.MODERATE_FP,       "Potentially lethal rapid rhythm originating from ventricles. Mapped to \"Ventricular Run\", the exact algorithmic trigger for 3+ consecutive PVCs."),
    142:(9.4,  ClinicalReliability.SEVERE_FP,         "Failure of the SA node to fire for >2 seconds. Mapped to \"Pause\", which is the critical algorithmic alert for transient asystole."),
}


def _r(idx: int) -> dict:
    """Helper: shorthand for the reliability triple of head `idx`."""
    ppv, rel, supp = _HEAD_RELIABILITY[idx]
    return dict(fzark_ppv_pct=ppv, clinical_reliability=rel, clinical_support=supp)


FZARK_ONTOLOGY: dict[str, ClinicalOntologyNode] = {
    "Atrial Fibrillation": ClinicalOntologyNode(
        canonical_name="ATRIAL FIBRILLATION",
        ecgfounder_index=5,
        risk_tier=ClinicalRiskTier.HIGH,
        semantic_match="exact",
        **_r(5),
    ),
    "Sinus Tachycardia": ClinicalOntologyNode(
        canonical_name="SINUS TACHYCARDIA",
        ecgfounder_index=6,
        risk_tier=ClinicalRiskTier.LOW,
        semantic_match="exact",
        **_r(6),
    ),
    "Bradycardia": ClinicalOntologyNode(
        canonical_name="SINUS BRADYCARDIA",
        ecgfounder_index=4,
        risk_tier=ClinicalRiskTier.HIGH,
        semantic_match="exact",
        notes="New in v2; was previously unmapped (n=2,155 TPs unrecovered).",
        **_r(4),
    ),
    "ST Elevation": ClinicalOntologyNode(
        canonical_name="ST ELEVATION NOW PRESENT IN",
        ecgfounder_index=68,
        risk_tier=ClinicalRiskTier.CRITICAL,
        semantic_match="exact",
        notes="New in v2; n=1 TP — detection metrics will not be meaningful.",
        **_r(68),
    ),
    "Isolated Ventricular Beat": ClinicalOntologyNode(
        canonical_name="PREMATURE VENTRICULAR COMPLEXES",
        ecgfounder_index=9,
        risk_tier=ClinicalRiskTier.LOW,
        semantic_match="exact",
        **_r(9),
    ),
    "Isolated Supraventricular Beat": ClinicalOntologyNode(
        canonical_name="PREMATURE ATRIAL COMPLEXES",
        ecgfounder_index=16,
        risk_tier=ClinicalRiskTier.LOW,
        semantic_match="exact",
        **_r(16),
    ),
    "Supraventricular Couplet": ClinicalOntologyNode(
        canonical_name="PREMATURE SUPRAVENTRICULAR COMPLEXES",
        ecgfounder_index=19,
        risk_tier=ClinicalRiskTier.MODERATE,
        semantic_match="good",
        notes="Off-by-one fix: was idx 20 (LBBB). Couplet = paired premature complexes.",
        **_r(19),
    ),
    "Ventricular Run": ClinicalOntologyNode(
        canonical_name="VENTRICULAR TACHYCARDIA",
        ecgfounder_index=98,
        risk_tier=ClinicalRiskTier.CRITICAL,
        semantic_match="good",
        notes="Off-by-one fix: was idx 99 (EARLY REPOLARIZATION). V run = brief non-sustained VT.",
        **_r(98),
    ),
    "Pause": ClinicalOntologyNode(
        canonical_name="WITH SINUS PAUSE",
        ecgfounder_index=142,
        risk_tier=ClinicalRiskTier.CRITICAL,
        semantic_match="good",
        notes="Off-by-one fix: was idx 143 (BIVENTRICULAR HYPERTROPHY).",
        **_r(142),
    ),
    "Supraventricular Run": ClinicalOntologyNode(
        canonical_name="SUPRAVENTRICULAR TACHYCARDIA",
        ecgfounder_index=93,
        risk_tier=ClinicalRiskTier.MODERATE,
        semantic_match="good",
        notes="New in v2; SV run = brief episode of SVT.",
        **_r(93),
    ),
    # ── v3.1 beat-level recoveries (composite events) ──────────────────────
    # These three events share their target head with a sibling event:
    # SV Trigeminy / SV Bigeminy share idx 16 with Isolated SV Beat (PAC head),
    # V Couplet shares idx 9 with Isolated V Beat (PVC head). The head fires
    # on the constituent beats rather than the pattern; downstream pattern
    # analysis is required to distinguish couplet/bigeminy/trigeminy from the
    # corresponding isolated-beat alert.
    "Supraventricular Trigeminy": ClinicalOntologyNode(
        canonical_name="PREMATURE ATRIAL COMPLEXES",
        ecgfounder_index=16,
        risk_tier=ClinicalRiskTier.LOW,
        semantic_match="good",
        notes="v3.1 recovery: head fires on constituent PACs; det@0.5 = 96.1% "
              "on fzark TPs (was 5.3% via passthrough).",
        **_r(16),
    ),
    "Supraventricular Bigeminy": ClinicalOntologyNode(
        canonical_name="PREMATURE ATRIAL COMPLEXES",
        ecgfounder_index=16,
        risk_tier=ClinicalRiskTier.LOW,
        semantic_match="good",
        notes="v3.1 recovery: head fires on constituent PACs; det@0.5 = 78.1% "
              "on fzark TPs (was 64.1% via passthrough).",
        **_r(16),
    ),
    "Ventricular Couplet": ClinicalOntologyNode(
        canonical_name="PREMATURE VENTRICULAR COMPLEXES",
        ecgfounder_index=9,
        risk_tier=ClinicalRiskTier.MODERATE,
        semantic_match="good",
        notes="v3.1 recovery: a V-couplet is two consecutive PVCs, so the PVC "
              "head fires directly; det@0.5 = 80.6% on fzark TPs (was 41.7% "
              "via passthrough).",
        **_r(9),
    ),
}


# Events that are NOT mapped to a single ECGFounder head.
# At runtime these produce no model prediction; pattern detection (if any)
# comes from the downstream FP suppression filter.
FZARK_UNMAPPABLE: frozenset[str] = frozenset({
    # Meta-categories (already pass-through in v1)
    "Multiple Event",
    "Unknown",
    "Custom Heart Rate",
    # v3.1: composite events with no head firing strongly on their constituent
    # beats or pattern. SV Trigeminy, SV Bigeminy, V Couplet were recovered in
    # v3.1 — see FZARK_ONTOLOGY above.
    "Ventricular Bigeminy",
    "Ventricular Trigeminy",
    "Prolonged RR Interval",
})


# ── Backward-compatibility dict ─────────────────────────────────────────────
# Existing scripts use `LABEL_MAP[event]` syntax. Expose a derived dict so
# they can `from label_config import FZARK_LABEL_MAP as LABEL_MAP` without
# further changes. The dict is read-only by convention.
FZARK_LABEL_MAP: dict[str, int] = {
    name: node.ecgfounder_index for name, node in FZARK_ONTOLOGY.items()
}


# ═══════════════════════════════════════════════════════════════════════════
# MIT-BIH single-index maps
# ═══════════════════════════════════════════════════════════════════════════

# Beat-annotation symbol → 150-class index.
# v3 change: S and j moved from idx 16 (PAC) to idx 19 (PSVC), aligning with
# the fzark Supraventricular Couplet mapping.
MITDB_BEAT_MAP: dict[str, int] = {
    "V": 9,    # PVC                          → PREMATURE VENTRICULAR COMPLEXES
    "L": 20,   # LBBB beat                    → LEFT BUNDLE BRANCH BLOCK
    "R": 11,   # RBBB beat                    → RIGHT BUNDLE BRANCH BLOCK
    "A": 16,   # Atrial premature             → PREMATURE ATRIAL COMPLEXES
    "a": 16,   # Aberrated atrial premature   → PREMATURE ATRIAL COMPLEXES
    "S": 19,   # Supraventricular premature   → PREMATURE SUPRAVENTRICULAR COMPLEXES
    "j": 19,   # Junctional premature         → PREMATURE SUPRAVENTRICULAR COMPLEXES
}


# Rhythm-annotation token → 150-class index.
# v3 adds VT / AFL / SVTA / SBR mappings; bigeminy/trigeminy intentionally
# omitted (composite events live in the suppression filter).
MITDB_RHYTHM_MAP: dict[str, int] = {
    "(AFIB": 5,   # Atrial fibrillation         → ATRIAL FIBRILLATION
    "(VT":   98,  # Ventricular tachycardia     → VENTRICULAR TACHYCARDIA
    "(AFL":  32,  # Atrial flutter              → ATRIAL FLUTTER
    "(SVTA": 93,  # SV tachyarrhythmia          → SUPRAVENTRICULAR TACHYCARDIA
    "(SBR":  4,   # Sinus bradycardia           → SINUS BRADYCARDIA
}


# Indices set to 1.0 when a segment contains no AFib rhythm AND no abnormal beats.
MITDB_DEFAULT_NORMAL: tuple[int, ...] = (1, 2)


# ═══════════════════════════════════════════════════════════════════════════
# PTB-XL active-class set
# ═══════════════════════════════════════════════════════════════════════════

# The 31 of 150 indices that have positive PTB-XL samples (n=21,799 records).
# PTB-XL labels are pre-computed in csv/ptbxl_label.csv as binary 150-vectors,
# so v3 introduces no on-the-fly mapping changes — this set is exposed for
# downstream filters that need to restrict metric averaging to active classes.
PTBXL_ACTIVE_CLASSES: frozenset[int] = frozenset({
    2, 3, 4, 5, 6, 9, 11, 12, 13, 15, 17, 20, 24, 26, 30, 32,
    36, 40, 50, 54, 60, 61, 70, 78, 79, 82, 93, 98, 101, 107, 112,
})


# ═══════════════════════════════════════════════════════════════════════════
# Detection scope — GLOBAL HARD RULE (6 events)
# ═══════════════════════════════════════════════════════════════════════════
# ┌─────────────────────────────────────────────────────────────────────────┐
# │ HARD RULE (2026-05-29): the system detects EXACTLY these 7 labels —       │
# │ 6 fzark events + NORMAL ECG. Nothing else is a valid detection target.    │
# │   Atrial Fibrillation (5) · Bradycardia (4) · Sinus Tachycardia (6)       │
# │   Supraventricular Run (93) · Ventricular Run (98) · Pause (142)          │
# │   Normal ECG (2)                                                          │
# │ This is the single source of truth; consistency is asserted at import.    │
# └─────────────────────────────────────────────────────────────────────────┘
# SCOPE_EVENT_TO_HEAD is the authoritative scope mapping (event/label → head).
# NORMAL ECG is a backbone head, NOT a fzark arrhythmia event (so it is NOT in
# FZARK_LABEL_MAP); it is exempt from the fzark-consistency check below.
SCOPE_EVENT_TO_HEAD: dict[str, int] = {
    "Atrial Fibrillation":   5,
    "Bradycardia":           4,
    "Sinus Tachycardia":     6,
    "Supraventricular Run":  93,
    "Ventricular Run":       98,
    "Pause":                 142,
    "Normal ECG":            2,    # backbone NORMAL ECG head (not a fzark event)
}
SCOPE_EVENTS: frozenset[str] = frozenset(SCOPE_EVENT_TO_HEAD)

# DETECTION_SCOPE maps each in-scope ECGFounder head index → its tasks.txt label.
# Heads 93/98/142 carry the head's own tasks.txt label (SVT / VT / sinus pause),
# a "good" — not exact — semantic match to SV Run / V Run / Pause.
# Out-of-scope heads (PVC 9, PAC 16, …) keep FZARK_ONTOLOGY entries for label
# MAPPING, but detect()/detect_index() return None for them and the FP suppressor
# and eval scripts skip them. (Names validated vs load_tasks() in tests.)
DETECTION_SCOPE: dict[int, str] = {
    2:   "NORMAL ECG",                     # not a fzark event — backbone head
    4:   "SINUS BRADYCARDIA",
    5:   "ATRIAL FIBRILLATION",
    6:   "SINUS TACHYCARDIA",
    93:  "SUPRAVENTRICULAR TACHYCARDIA",   # fzark: Supraventricular Run
    98:  "VENTRICULAR TACHYCARDIA",        # fzark: Ventricular Run
    142: "WITH SINUS PAUSE",               # fzark: Pause
}

# HARD invariant — fail loudly at import if the views ever drift apart. The set of
# SCOPE_EVENT_TO_HEAD values must equal DETECTION_SCOPE's keys; fzark-mapped scope
# events must agree with FZARK_LABEL_MAP (NORMAL ECG is exempt — not a fzark event).
# Enforced by _assert_scope_consistency() near the bottom.


# Per-head detection thresholds — override the 0.5 default for specific heads.
# Heads absent here use DEFAULT_THRESHOLD (0.5).
#
# DECISION (2026-05-29): ALL scope heads use the 0.5 default — no overrides.
# Calibration (scripts/calibrate_scope_thresholds.py, res/scope_threshold_cal/)
# found that heads 93 (SV Run) and 142 (Pause) only fire at noise-floor thresholds
# (0.040 / 0.006) — i.e. the base single-lead model cannot really detect them.
# Those tiny thresholds are fragile and device/cohort-specific, so they were
# REVERTED to 0.5: heads 93/98/142 stay effectively silent at single-lead (an
# honest "not detectable") rather than firing on near-noise. Detecting SV-Run /
# V-Run / Pause needs the fine-tuned/fuzzy head, not a low threshold.
# (The calibration record is kept for history; do not re-add the overrides without
# a deliberate per-device recalibration.)
DEFAULT_THRESHOLD: float = 0.5
HEAD_THRESHOLDS: dict[int, float] = {}


def head_threshold(index: int) -> float:
    """The detection threshold for `index` (per-head override or 0.5 default)."""
    return HEAD_THRESHOLDS.get(index, DEFAULT_THRESHOLD)


# ═══════════════════════════════════════════════════════════════════════════
# Detection helpers
# ═══════════════════════════════════════════════════════════════════════════

def in_scope(index: int) -> bool:
    """True iff `index` is an active detection head (in DETECTION_SCOPE)."""
    return index in DETECTION_SCOPE


def scope_indices() -> frozenset[int]:
    """The set of in-scope ECGFounder head indices."""
    return frozenset(DETECTION_SCOPE)


def scope_events() -> frozenset[str]:
    """The 6 in-scope fzark event names (the global hard rule)."""
    return SCOPE_EVENTS


def event_in_scope(event_name: str) -> bool:
    """True iff `event_name` is one of the 6 in-scope detection events."""
    return event_name in SCOPE_EVENTS


def require_in_scope(event_name: str) -> str:
    """Hard guard: return `event_name` if in scope, else raise ValueError.

    Use at boundaries that must never act on an out-of-scope event.
    """
    if event_name not in SCOPE_EVENTS:
        raise ValueError(
            f"{event_name!r} is out of detection scope. The hard rule allows only "
            f"{sorted(SCOPE_EVENTS)}."
        )
    return event_name


def get_index(event_name: str) -> Optional[int]:
    """Return the ECGFounder index for `event_name`, or None if unsupported."""
    node = FZARK_ONTOLOGY.get(event_name)
    return node.ecgfounder_index if node is not None else None


def is_supported(event_name: str) -> bool:
    """True iff the event has a single-head ECGFounder mapping in v3."""
    return event_name in FZARK_ONTOLOGY


def detect(
    probs,
    event_name: str,
    threshold: Optional[float] = None,
) -> Optional[bool]:
    """Single-head detection: True if probs[idx] > threshold.

    `threshold=None` (default) uses the per-head calibrated threshold
    (head_threshold); pass a float to override. Returns None ("no detection
    result available") if the event is unsupported OR its head is outside
    DETECTION_SCOPE — callers should treat None as out-of-scope, not a false
    negative.
    """
    # resolve head: scope map first (covers NORMAL ECG), else fzark ontology
    idx = SCOPE_EVENT_TO_HEAD.get(event_name)
    if idx is None:
        node = FZARK_ONTOLOGY.get(event_name)
        idx = node.ecgfounder_index if node is not None else None
    if idx is None or idx not in DETECTION_SCOPE:
        return None
    thr = head_threshold(idx) if threshold is None else threshold
    return bool(probs[idx] > thr)


def detect_index(
    probs,
    index: int,
    threshold: Optional[float] = None,
) -> Optional[bool]:
    """Index-based detection: True if probs[index] > threshold.

    `threshold=None` (default) uses the per-head calibrated threshold.
    Returns None if `index` is outside DETECTION_SCOPE.
    """
    if index not in DETECTION_SCOPE:
        return None
    thr = head_threshold(index) if threshold is None else threshold
    return bool(probs[index] > thr)


# ═══════════════════════════════════════════════════════════════════════════
# HARD-RULE invariant — enforced at import (fail loudly if the scope drifts)
# ═══════════════════════════════════════════════════════════════════════════
# Scope labels that are NOT fzark arrhythmia events (backbone heads) — exempt from
# the FZARK_LABEL_MAP agreement check.
_NON_FZARK_SCOPE: frozenset[str] = frozenset({"Normal ECG"})


def _assert_scope_consistency() -> None:
    """The 7-label detection scope (6 fzark events + NORMAL ECG) is a global hard
    rule. Guarantee the views stay in sync: SCOPE_EVENT_TO_HEAD values must equal
    DETECTION_SCOPE's keys, and each fzark-mapped scope event must agree with
    FZARK_LABEL_MAP (NORMAL ECG exempt — it is not a fzark event)."""
    if len(SCOPE_EVENT_TO_HEAD) != 7:
        raise AssertionError(f"SCOPE_EVENT_TO_HEAD must hold exactly 7 labels, got {len(SCOPE_EVENT_TO_HEAD)}")
    if set(SCOPE_EVENT_TO_HEAD.values()) != set(DETECTION_SCOPE):
        raise AssertionError(
            "Detection-scope hard rule violated: SCOPE_EVENT_TO_HEAD heads "
            f"{sorted(SCOPE_EVENT_TO_HEAD.values())} != DETECTION_SCOPE keys {sorted(DETECTION_SCOPE)}"
        )
    for ev, head in SCOPE_EVENT_TO_HEAD.items():
        if ev in _NON_FZARK_SCOPE:
            continue
        if FZARK_LABEL_MAP.get(ev) != head:
            raise AssertionError(
                f"Scope event {ev!r} head {head} disagrees with FZARK_LABEL_MAP "
                f"({FZARK_LABEL_MAP.get(ev)})"
            )


_assert_scope_consistency()
