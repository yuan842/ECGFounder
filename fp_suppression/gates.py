"""Shared gate primitives for the split FP-suppression algorithms.

A `Gate` is one threshold condition on one feature (e.g. mean_motion <= 5.0).
An event's rule = a list of gates; **keep iff ALL gates pass** (suppress if any
fails). A missing feature fails OPEN (gate passes) — same fail-open semantics as
the production v2 suppressor (multiclass_fp_suppression._apply_rules).

The two families are distinguished only by which features their gates read:
  • MOTION  family → accelerometer-derived features  (mean_motion, std_motion, max_motion)
  • SQI     family → ECG-intrinsic features          (snr_proxy, mean_hr_bpm, clip_pct, flat_pct, ...)

This module is dependency-free (no numpy needed for evaluation).
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

# Operators — identical table to multiclass_fp_suppression._OPS
_OPS = {
    '<':  lambda v, t: v < t,
    '<=': lambda v, t: v <= t,
    '>':  lambda v, t: v > t,
    '>=': lambda v, t: v >= t,
    '==': lambda v, t: v == t,
}

# Which feature belongs to which trainable family. Used to validate that a
# motion profile never carries an ECG gate and vice-versa.
MOTION_FEATURES = {'mean_motion', 'std_motion', 'max_motion'}
SQI_FEATURES    = {'snr_proxy', 'mean_hr_bpm', 'clip_pct', 'flat_pct',
                   'baseline_drift', 'hf_noise', 'kurt', 'dyn_range'}


@dataclass(frozen=True)
class Gate:
    """One keep-condition. `keep` is True when the condition holds."""
    feature: str
    op: str
    threshold: float

    def family(self) -> str:
        if self.feature in MOTION_FEATURES:
            return 'motion'
        if self.feature in SQI_FEATURES:
            return 'sqi'
        return 'other'

    def evaluate(self, features: Dict[str, float]) -> Tuple[bool, bool]:
        """Return (passed, missing). Missing features fail OPEN → passed=True."""
        if self.feature not in features or features[self.feature] is None:
            return True, True
        fn = _OPS.get(self.op)
        if fn is None:
            raise ValueError(f"Unsupported operator: {self.op!r}")
        return bool(fn(features[self.feature], self.threshold)), False

    def __str__(self) -> str:
        return f"{self.feature} {self.op} {self.threshold:g}"


@dataclass
class Decision:
    """Result of applying one family's gates to one event."""
    keep: bool
    reason: str
    fired_gates: List[Gate] = field(default_factory=list)   # gates that FAILED → caused suppression


def apply_gates(event: str, gates: Optional[List[Gate]],
                features: Dict[str, float]) -> Decision:
    """Keep iff every gate passes; suppress on the first/any failing gate.

    No gates for this event  → passthrough (keep).
    Missing feature          → that gate passes (fail-open).
    """
    if not gates:
        return Decision(keep=True, reason='passthrough')
    failed, missing = [], []
    for g in gates:
        passed, is_missing = g.evaluate(features)
        if is_missing:
            missing.append(g.feature)
        elif not passed:
            failed.append(g)
    if failed:
        reason = '; '.join(f"{g.feature}={features[g.feature]:.2f} failed {g.op}{g.threshold:g}"
                           for g in failed)
        return Decision(keep=False, reason=reason, fired_gates=failed)
    if missing:
        return Decision(keep=True, reason=f"pass (missing: {','.join(missing)})")
    return Decision(keep=True, reason='pass_all_gates')
