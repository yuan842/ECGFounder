"""
label_config.py — Unified single-index label mapping (v3) for the
ECGFounder 150-class model.

Single source of truth for dataset → 150-class index mappings across:
  * ecg-tp-fzark and ecg-fp-doctor-removed (fzark / FP datasets)
  * MIT-BIH arrhythmia annotations
  * PTB-XL (active-class set)

v3 design principle: every supported event maps to **exactly one** ECGFounder
index. No multi-head logic, no MAX/AND operators, no partial-match heads.
Composite events (bigeminy / trigeminy / couplets) and vocabulary-limited
events (Prolonged RR Interval) are passed through to the downstream FP
suppression filter, which already handles pattern detection via motion + SNR.

See `res/label_reclassification_plan.md` (v3) for the full design rationale.
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


@dataclass(frozen=True)
class ClinicalOntologyNode:
    """One row of the fzark → ECGFounder ontology.

    v3 invariant: exactly one ECGFounder index per event. No operator field;
    no multi-head logic. Composite events live in FZARK_UNMAPPABLE.
    """
    canonical_name: str            # human-readable identifier
    ecgfounder_index: int          # exactly one 150-class index
    risk_tier: ClinicalRiskTier
    semantic_match: str            # "exact" | "good"
    notes: str = ""


# ═══════════════════════════════════════════════════════════════════════════
# fzark / ECG-FP ontology — 10 supported events
# ═══════════════════════════════════════════════════════════════════════════

FZARK_ONTOLOGY: dict[str, ClinicalOntologyNode] = {
    "Atrial Fibrillation": ClinicalOntologyNode(
        canonical_name="ATRIAL FIBRILLATION",
        ecgfounder_index=5,
        risk_tier=ClinicalRiskTier.HIGH,
        semantic_match="exact",
    ),
    "Sinus Tachycardia": ClinicalOntologyNode(
        canonical_name="SINUS TACHYCARDIA",
        ecgfounder_index=6,
        risk_tier=ClinicalRiskTier.LOW,
        semantic_match="exact",
    ),
    "Bradycardia": ClinicalOntologyNode(
        canonical_name="SINUS BRADYCARDIA",
        ecgfounder_index=4,
        risk_tier=ClinicalRiskTier.HIGH,
        semantic_match="exact",
        notes="New in v2; was previously unmapped (n=2,155 TPs unrecovered).",
    ),
    "ST Elevation": ClinicalOntologyNode(
        canonical_name="ST ELEVATION NOW PRESENT IN",
        ecgfounder_index=68,
        risk_tier=ClinicalRiskTier.CRITICAL,
        semantic_match="exact",
        notes="New in v2; n=1 TP — detection metrics will not be meaningful.",
    ),
    "Isolated Ventricular Beat": ClinicalOntologyNode(
        canonical_name="PREMATURE VENTRICULAR COMPLEXES",
        ecgfounder_index=9,
        risk_tier=ClinicalRiskTier.LOW,
        semantic_match="exact",
    ),
    "Isolated Supraventricular Beat": ClinicalOntologyNode(
        canonical_name="PREMATURE ATRIAL COMPLEXES",
        ecgfounder_index=16,
        risk_tier=ClinicalRiskTier.LOW,
        semantic_match="exact",
    ),
    "Supraventricular Couplet": ClinicalOntologyNode(
        canonical_name="PREMATURE SUPRAVENTRICULAR COMPLEXES",
        ecgfounder_index=19,
        risk_tier=ClinicalRiskTier.MODERATE,
        semantic_match="good",
        notes="Off-by-one fix: was idx 20 (LBBB). Couplet = paired premature complexes.",
    ),
    "Ventricular Run": ClinicalOntologyNode(
        canonical_name="VENTRICULAR TACHYCARDIA",
        ecgfounder_index=98,
        risk_tier=ClinicalRiskTier.CRITICAL,
        semantic_match="good",
        notes="Off-by-one fix: was idx 99 (EARLY REPOLARIZATION). V run = brief non-sustained VT.",
    ),
    "Pause": ClinicalOntologyNode(
        canonical_name="WITH SINUS PAUSE",
        ecgfounder_index=142,
        risk_tier=ClinicalRiskTier.CRITICAL,
        semantic_match="good",
        notes="Off-by-one fix: was idx 143 (BIVENTRICULAR HYPERTROPHY).",
    ),
    "Supraventricular Run": ClinicalOntologyNode(
        canonical_name="SUPRAVENTRICULAR TACHYCARDIA",
        ecgfounder_index=93,
        risk_tier=ClinicalRiskTier.MODERATE,
        semantic_match="good",
        notes="New in v2; SV run = brief episode of SVT.",
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
    # Dropped in v3 — composite events (require multi-head logic) or
    # partial-match heads that lose pattern information.
    "Ventricular Couplet",
    "Ventricular Bigeminy",
    "Supraventricular Bigeminy",
    "Ventricular Trigeminy",
    "Supraventricular Trigeminy",
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
# Detection helpers
# ═══════════════════════════════════════════════════════════════════════════

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
    threshold: float = 0.5,
) -> Optional[bool]:
    """Single-head detection: True if probs[idx] > threshold, None if unsupported.

    Callers should treat `None` as "no detection result available" and avoid
    logging the event as a false negative.
    """
    node = FZARK_ONTOLOGY.get(event_name)
    if node is None:
        return None
    return bool(probs[node.ecgfounder_index] > threshold)
