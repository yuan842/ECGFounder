"""Shared interface types for the overlay subsystem (L1→L2 boundary)."""
from __future__ import annotations
from dataclasses import dataclass, field


@dataclass(frozen=True)
class ScopeScores:
    """L1 output / L2 input — the boundary contract.

    probs:      {head_idx: calibrated_prob} for the 6 scope heads {4,5,6,93,98,142}
    nsr_score:  head-1 (NORMAL SINUS RHYTHM) raw prob — an FP *feature* for L2,
                NOT a detection target (head 1 is out of scope).
    context:    optional physiology, e.g. {"hr_bpm": 48.0, "rr_irregularity": 0.3}
    quality:    informational signal-state flags {150: Noisy, 151: High-Motion};
                populated by the SQG, advisory only when the gate is OFF.
    """
    probs: dict[int, float]
    nsr_score: float = 0.0
    context: dict[str, float] = field(default_factory=dict)
    quality: dict[int, bool] = field(default_factory=dict)


@dataclass(frozen=True)
class Decision:
    """L2 output — one per scope head, with an audit reason string."""
    head: int
    fired: bool
    score: float
    reason: str


@dataclass(frozen=True)
class RunAlert:
    """Collapsed SV-Run/V-Run alert (Decision 2 = merge for alerting).

    Emitted when either run head survives L2. origin is 'uncertain' at single
    lead; acuity is escalated to the max of its members (treat as potential VT).
    """
    fired: bool
    score: float            # max(SVT, V-Run) surviving score
    origin: str = "uncertain (SV/V)"
    acuity: str = "critical"   # escalate to VT-level
    members: tuple[int, ...] = (93, 98)
