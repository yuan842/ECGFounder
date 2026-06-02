"""SQG — Signal-Quality Gate (SQI + Motion).  DEFAULT OFF (2026-06-02 update).

Runs BETWEEN preprocessing and the CNN backbone (see res/OVERLAY_DESIGN.md §2),
on the S0 (raw) SQI taps. Signal interpretability is a property of the input, so
this gates the *signal* before detection rather than suppressing head outputs
afterward.

Semantics:
  • OFF (default): pure pass-through. Quality flags may still be computed for
    logging, but no segment is blocked. (Turned off pending device-specific L1
    calibration; re-enable per deployment after validation.)
  • ON: a segment failing the SQI threshold is classified **Noisy** and is NOT
    sent to detection (passed=False → caller emits "uninterpretable"/Noisy).
  • Fail-open: if the required SQI metric is absent (e.g. PTB-XL, no accel), no
    flag is raised and the segment passes.

Noisy rule (SNR only — baseline_drift gating DISABLED for now):

    NOISY  ⇔  snr_proxy < τ(device)

  device "default" → τ = 0.10 (= -10 dB), inclusive <=   (all devices, incl. AliveCor)

DEVICE_MIN_SNR is currently empty: every device — including AliveCor (Challenge
2017) — uses the default 0.10 threshold (the AliveCor-specific 0.1122 was retired
2026-06-02 in favor of device-specific L1 calibration). Add a (τ, inclusive) entry
to override a device. snr_proxy is the scale-free LINEAR variance ratio from
sqi.compute_band_sqi (SNR_dB = 10·log10·snr_proxy). Baseline-drift gating stays
disabled (max_baseline_drift=None): the threshold is amplitude/scale-dependent and
not yet validated. See res/challenge2017/SQI_NOISY_GATE_PROPOSAL.md.
"""
from __future__ import annotations
import math
from dataclasses import dataclass
from typing import Any, Optional

# signal-state label indices (virtual; see label_config.SIGNAL_STATE_LABELS)
NOISY = 150
HIGH_MOTION = 151

DEFAULT_MOTION_MG = 20.0          # mean_motion (mG) above this ⇒ High-Motion

# Device-specific SNR (linear snr_proxy) overrides → (threshold, inclusive).
# inclusive=False ⇒ noisy if snr <  τ ; inclusive=True ⇒ noisy if snr <= τ.
# Empty: all devices use the default below (AliveCor uses default as of 2026-06-02).
DEVICE_MIN_SNR: dict[str, tuple[float, bool]] = {}
DEFAULT_MIN_SNR = 0.10             # snr <= 0.10 (-10 dB) — every device
DEFAULT_MIN_SNR_INCLUSIVE = True

# Baseline-drift gating disabled by default (scale-dependent, not yet validated).
DEFAULT_MAX_BASELINE_DRIFT: Optional[float] = None   # mV ceiling; None ⇒ disabled


def snr_to_db(snr_proxy: float) -> float:
    """Convert the linear snr_proxy (power ratio) to decibels (10·log10)."""
    return 10.0 * math.log10(max(float(snr_proxy), 1e-12))


def device_snr_threshold(device: str) -> tuple[float, bool]:
    """(threshold, inclusive) SNR floor for a device; falls back to default."""
    return DEVICE_MIN_SNR.get(str(device).lower(), (DEFAULT_MIN_SNR, DEFAULT_MIN_SNR_INCLUSIVE))


@dataclass
class GateResult:
    signal: Any                      # the (unchanged) signal, passed through
    quality: dict[int, bool]         # {150: Noisy?, 151: High-Motion?}
    passed: bool                     # True ⇒ proceed to CNN
    reason: str


class SignalQualityGate:
    def __init__(self, enabled: bool = False, device: str = "default",
                 motion_mg: float = DEFAULT_MOTION_MG,
                 min_snr: Optional[float] = None, snr_inclusive: Optional[bool] = None,
                 max_baseline_drift: Optional[float] = DEFAULT_MAX_BASELINE_DRIFT):
        self.enabled = enabled
        self.device = device
        thr, inc = device_snr_threshold(device)
        # explicit overrides win over the device table
        self.min_snr = thr if min_snr is None else min_snr
        self.snr_inclusive = inc if snr_inclusive is None else snr_inclusive
        self.motion_mg = motion_mg
        self.max_baseline_drift = max_baseline_drift   # None ⇒ drift gating disabled

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

        Noisy ⇔ snr_proxy below the device SNR threshold (baseline-drift gating
        only applies when max_baseline_drift is set). NOISY is raised only when a
        consulted SQI metric is present.
        """
        flags: dict[int, bool] = {}
        if accel is not None:
            mean_motion = float(getattr(accel, "mean_motion_mg", accel)
                                if not isinstance(accel, dict)
                                else accel.get("mean_motion_mg", 0.0))
            flags[HIGH_MOTION] = mean_motion > self.motion_mg
        if sqi is not None:
            snr = sqi.get("snr_proxy")
            drift = sqi.get("baseline_drift")
            drift_on = self.max_baseline_drift is not None
            considered = (snr is not None) or (drift_on and drift is not None)
            if considered:
                noisy = False
                if snr is not None:
                    noisy |= (float(snr) <= self.min_snr) if self.snr_inclusive \
                             else (float(snr) < self.min_snr)
                if drift_on and drift is not None:
                    noisy |= float(drift) > self.max_baseline_drift
                flags[NOISY] = noisy
        return flags


if __name__ == "__main__":
    for dev in ("default", "alivecor"):
        g = SignalQualityGate(device=dev)
        op = "<=" if g.snr_inclusive else "<"
        print(f"[{dev}] noisy if snr {op} {g.min_snr:.4f} ({snr_to_db(g.min_snr):.1f} dB); "
              f"drift gating={'on' if g.max_baseline_drift is not None else 'off'}")
        print("   snr=0.05 :", g.gate("<sig>", sqi={"snr_proxy": 0.05}).reason)
        print("   snr=0.11 :", g.gate("<sig>", sqi={"snr_proxy": 0.11}).reason)
        print("   snr=0.9  :", g.gate("<sig>", sqi={"snr_proxy": 0.9, "baseline_drift": 0.5}).reason,
              "(drift ignored)")
        print("   no-sqi   :", g.gate("<sig>").reason)
