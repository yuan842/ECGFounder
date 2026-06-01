"""SQG — Signal-Quality Gate (SQI + Motion).  DEFAULT OFF.

Runs BETWEEN preprocessing and the CNN backbone (see res/OVERLAY_DESIGN.md §2).
Signal interpretability is a property of the input, so this gates the *signal*
before detection rather than suppressing head outputs afterward.

Semantics:
  • OFF (default): pure pass-through. Quality flags may still be computed for
    logging, but no segment is blocked — everything reaches the CNN.
  • ON: a segment failing the SQI/motion threshold is NOT sent to detection
    (passed=False → caller emits "uninterpretable"), since CNN output on
    corrupted signal is untrustworthy.

Thresholds below are UNCALIBRATED placeholders (Decision 3, OVERLAY_DESIGN §7):
re-derive on a labeled clean/noisy device set before enabling. PTB-XL has no
accelerometer, so motion is undefined there and OFF is the only valid setting.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Optional

# signal-state label indices (virtual; see label_config.SIGNAL_STATE_LABELS)
NOISY = 150
HIGH_MOTION = 151

# placeholder thresholds — only consulted when enabled=True
DEFAULT_MOTION_MG = 20.0     # mean_motion (mG) above this ⇒ High-Motion
DEFAULT_MIN_SNR = 1.0        # snr_proxy below this ⇒ Noisy


@dataclass
class GateResult:
    signal: Any                      # the (unchanged) signal, passed through
    quality: dict[int, bool]         # {150: Noisy?, 151: High-Motion?}
    passed: bool                     # True ⇒ proceed to CNN
    reason: str


class SignalQualityGate:
    def __init__(self, enabled: bool = False,
                 motion_mg: float = DEFAULT_MOTION_MG,
                 min_snr: float = DEFAULT_MIN_SNR):
        self.enabled = enabled
        self.motion_mg = motion_mg
        self.min_snr = min_snr

    def gate(self, signal: Any, *, accel: Optional[Any] = None,
             sqi: Optional[dict[str, float]] = None) -> GateResult:
        quality = self._flags(accel=accel, sqi=sqi)
        if not self.enabled:
            return GateResult(signal, quality, passed=True,
                              reason="SQG off (pass-through)")
        bad = quality.get(HIGH_MOTION, False) or quality.get(NOISY, False)
        return GateResult(
            signal, quality, passed=not bad,
            reason="ok" if not bad else
            f"gated: {'high-motion ' if quality.get(HIGH_MOTION) else ''}"
            f"{'noisy' if quality.get(NOISY) else ''}".strip(),
        )

    def _flags(self, *, accel, sqi) -> dict[int, bool]:
        """Compute signal-state flags. Fail-open: missing inputs ⇒ no flag.

        TODO (when enabling): wire to sqi.py metrics + accelerometer mean_motion.
        Currently a placeholder that flags only on explicitly-provided metrics.
        """
        flags: dict[int, bool] = {}
        if accel is not None:
            mean_motion = float(getattr(accel, "mean_motion_mg", accel)
                                if not isinstance(accel, dict)
                                else accel.get("mean_motion_mg", 0.0))
            flags[HIGH_MOTION] = mean_motion > self.motion_mg
        if sqi is not None and "snr_proxy" in sqi:
            flags[NOISY] = float(sqi["snr_proxy"]) < self.min_snr
        return flags


if __name__ == "__main__":
    g = SignalQualityGate()  # default OFF
    r = g.gate("<signal>", accel={"mean_motion_mg": 99.0})
    print("OFF :", r.passed, r.reason, r.quality)   # passed=True (pass-through)
    g_on = SignalQualityGate(enabled=True)
    r2 = g_on.gate("<signal>", accel={"mean_motion_mg": 99.0})
    print("ON  :", r2.passed, r2.reason, r2.quality) # passed=False (high-motion)
