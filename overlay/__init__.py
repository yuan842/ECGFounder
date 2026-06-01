"""overlay — scoped detection + multi-label arbitration subsystem (Option B).

Design doc: res/OVERLAY_DESIGN.md. Pipeline:

    preprocessing
      → SignalQualityGate   (SQG; default OFF — pass-through)
      → Net1D backbone      (frozen CNN, 150 logits)
      → ScopeProjection     (L1; learned 150→6 calibrated scope probs)
      → arbitrate()         (L2; deterministic multi-label rules → decisions)

These are SKELETONS with the recommended defaults wired in:
  • L1 = projection (150→6), weights UNTRAINED (see OVERLAY_DESIGN §3 for training).
  • L2 = deterministic; SVT/VT merged for alerting, kept in the audit record.
  • SQG = OFF by default; thresholds uncalibrated placeholders.
"""
from overlay.types import ScopeScores, Decision, RunAlert
from overlay.signal_quality_gate import SignalQualityGate, GateResult
from overlay.arbiter import (arbitrate, to_alerts, SCOPE_HEADS, ArbiterConfig,
                             DEFAULT_CONFIG, GT_MATCHED_CONFIG, OFF_CONFIG,
                             EXCLUSION_GROUPS, COUPLE_GROUPS)

# inference (backbone→L1→L2) pulls in torch + checkpoints; import lazily so the
# lightweight pieces above stay importable without loading the backbone.
def __getattr__(name):  # PEP 562
    if name in ("ScopedDetector", "L1_HEADS", "BASE_HEADS", "load_l1"):
        from overlay import inference
        return getattr(inference, name)
    raise AttributeError(name)

__all__ = [
    "ScopeScores", "Decision", "RunAlert",
    "SignalQualityGate", "GateResult",
    "arbitrate", "to_alerts", "SCOPE_HEADS", "ArbiterConfig", "DEFAULT_CONFIG",
    "GT_MATCHED_CONFIG", "OFF_CONFIG", "EXCLUSION_GROUPS", "COUPLE_GROUPS",
    "ScopedDetector", "L1_HEADS", "BASE_HEADS", "load_l1",
]
