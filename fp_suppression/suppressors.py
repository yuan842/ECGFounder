"""The two independently-trainable FP-suppression algorithms.

  MotionFPSuppressor  — gates on accelerometer features; device-calibrated to a
                        device's motion-energy distribution.
  SQIFPSuppressor     — gates on ECG-intrinsic features (signal quality + rate
                        plausibility); device-calibrated to a device's electrode.

Each is constructed FROM a DeviceProfile (its own gate table slice) and exposes:
  .suppress(event, features)  -> Decision         (keep / drop for this family)
  .gates_for(event)           -> [Gate]
  .calibrate(...)             -> updated gate(s)   (see calibration.py)

They are used standalone (train/run one family alone) or composed by
FPSuppressionPipeline (run both, AND the keep decisions).

Note on HR (Bradycardia): heart rate is a *physiologic-rate* gate, not strictly
signal "quality", but it is ECG-DERIVED (not motion-derived), so within a
two-algorithm split it lives in the SQI/ECG-intrinsic algo. It can be promoted
to its own family later without touching the motion algo.
"""
from __future__ import annotations
from typing import Dict, List

from .gates import Gate, Decision, apply_gates
from .device_profiles import DeviceProfile, get_profile


class _GateSuppressor:
    """Common machinery; subclasses pick which gate table they read."""
    family: str = "base"

    def __init__(self, profile_or_device="fzark", *, enabled: bool = True):
        self.profile: DeviceProfile = (
            profile_or_device if isinstance(profile_or_device, DeviceProfile)
            else get_profile(profile_or_device))
        self.profile.validate()
        self.enabled = enabled

    # subclasses override
    def _table(self) -> Dict[str, List[Gate]]:
        raise NotImplementedError

    def gates_for(self, event: str) -> List[Gate]:
        return list(self._table().get(event, []))

    def covered_events(self) -> List[str]:
        return sorted(self._table())

    def suppress(self, event: str, features: Dict[str, float]) -> Decision:
        """Family-local keep/drop. Disabled or uncovered event → passthrough."""
        if not self.enabled:
            return Decision(keep=True, reason=f'{self.family}_disabled')
        return apply_gates(event, self._table().get(event), features)

    def set_gates(self, event: str, gates: List[Gate]) -> None:
        for g in gates:
            if g.family() != self.family:
                raise ValueError(f"{self.family} suppressor rejects {g.family} gate {g}")
        self._table()[event] = gates


class MotionFPSuppressor(_GateSuppressor):
    """Suppress motion-artifact false positives using accelerometer features."""
    family = "motion"

    def _table(self) -> Dict[str, List[Gate]]:
        return self.profile.motion_gates


class SQIFPSuppressor(_GateSuppressor):
    """Suppress false positives using ECG-intrinsic signal-quality / rate features."""
    family = "sqi"

    def _table(self) -> Dict[str, List[Gate]]:
        return self.profile.sqi_gates
