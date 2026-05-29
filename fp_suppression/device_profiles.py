"""Per-device parameter profiles for the two FP-suppression algorithms.

Training parameters are DEVICE-BASED: each device's electrode + accelerometer
has its own motion-energy and signal-quality distribution, so each gets its own
calibrated thresholds. A `DeviceProfile` carries two independent gate tables —
one for the motion algo, one for the SQI algo — each keyed by event name.

  motion_gates[event] -> [Gate(mean_motion, ...), ...]   (accelerometer features)
  sqi_gates[event]    -> [Gate(snr_proxy/mean_hr_bpm, ...), ...]  (ECG features)

The two tables are trained SEPARATELY (see calibration.py) and can be updated
independently — e.g. recalibrate motion for a new chest strap without touching
the SQI gates.

`fzark` is the production baseline: its gates are byte-for-byte the v2 rule set
(multiclass_fp_suppression.ACTIVE_RULES), so MotionFP ∧ SQIFP on `fzark`
reproduces production exactly (see tests/test_split_equivalence.py).
"""
from __future__ import annotations
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Dict, List

from .gates import Gate, MOTION_FEATURES, SQI_FEATURES


@dataclass
class DeviceProfile:
    device_id: str
    motion_gates: Dict[str, List[Gate]] = field(default_factory=dict)
    sqi_gates:    Dict[str, List[Gate]] = field(default_factory=dict)
    notes: str = ""

    def validate(self) -> None:
        """Guarantee motion gates only read motion features and SQI gates only
        read ECG-intrinsic features — the property that keeps the two algos
        independently trainable."""
        for ev, gs in self.motion_gates.items():
            for g in gs:
                if g.feature not in MOTION_FEATURES:
                    raise ValueError(f"[{self.device_id}] motion gate for {ev!r} "
                                     f"reads non-motion feature {g.feature!r}")
        for ev, gs in self.sqi_gates.items():
            for g in gs:
                if g.feature not in SQI_FEATURES:
                    raise ValueError(f"[{self.device_id}] sqi gate for {ev!r} "
                                     f"reads non-sqi feature {g.feature!r}")

    def copy(self, device_id: str) -> "DeviceProfile":
        return DeviceProfile(device_id=device_id,
                             motion_gates=deepcopy(self.motion_gates),
                             sqi_gates=deepcopy(self.sqi_gates),
                             notes=self.notes)


# ════════════════════════════════════════════════════════════════════════════
# fzark / Vivalink Holter — PRODUCTION baseline (== v2 ACTIVE_RULES, split)
# ════════════════════════════════════════════════════════════════════════════
#   AFib              motion ≤ 5.0                       (motion)
#   SV-Trigeminy      motion ≥ 15.0                      (motion, inverted)
#   V-Trigeminy       motion ≥ 24.0  AND  snr > 1.2      (motion + SQI — SPLIT)
#   Bradycardia       hr ≤ 56.3                          (SQI / physiologic)
FZARK = DeviceProfile(
    device_id="fzark",
    motion_gates={
        'Atrial Fibrillation':        [Gate('mean_motion', '<=', 5.0)],
        'Supraventricular Trigeminy': [Gate('mean_motion', '>=', 15.0)],
        'Ventricular Trigeminy':      [Gate('mean_motion', '>=', 24.0)],
    },
    sqi_gates={
        'Bradycardia':                [Gate('mean_hr_bpm', '<=', 56.3)],
        'Ventricular Trigeminy':      [Gate('snr_proxy',   '>',  1.2)],
    },
    notes="Production v2 baseline (ACC_SCALE_FACTOR=2048, Vivalink mG units).",
)

# ════════════════════════════════════════════════════════════════════════════
# move_chest_gel — MOVE chest strap, ecg:gel + chest accelerometer
# ════════════════════════════════════════════════════════════════════════════
# Chest motion is ~6× lower than fzark's distribution (median 1.0 mG, p95 0.8 mG
# at rest). The AFib gate is recalibrated to the rest-motion envelope
# (res/move_eval/MOTION_GATE_RECALIBRATION.md). The inverted SV/V-Trig motion
# gates and the SNR gate are NOT yet device-calibrated for MOVE — flagged below.
MOVE_CHEST_GEL = FZARK.copy("move_chest_gel")
MOVE_CHEST_GEL.motion_gates['Atrial Fibrillation'] = [Gate('mean_motion', '<=', 1.0)]
MOVE_CHEST_GEL.notes = (
    "MOVE chest ecg:gel. EDF already in g → no /2048 rescale. "
    "AFib motion gate recalibrated 5.0→1.0 mG (rest p99). "
    "TODO: SV/V-Trig motion (≥15/≥24) and SNR (>1.2) gates NOT device-calibrated "
    "for MOVE — fzark values carried over as placeholders. TP-retention unvalidated "
    "(MOVE has no labelled positives)."
)


_REGISTRY: Dict[str, DeviceProfile] = {p.device_id: p for p in (FZARK, MOVE_CHEST_GEL)}
DEFAULT_DEVICE = "fzark"


def get_profile(device_id: str = DEFAULT_DEVICE) -> DeviceProfile:
    if device_id not in _REGISTRY:
        raise KeyError(f"Unknown device {device_id!r}. Known: {sorted(_REGISTRY)}")
    p = _REGISTRY[device_id]
    p.validate()
    return p


def register_profile(profile: DeviceProfile) -> None:
    profile.validate()
    _REGISTRY[profile.device_id] = profile


def list_devices() -> List[str]:
    return sorted(_REGISTRY)
