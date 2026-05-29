"""Composer — run both algorithms and combine their keep decisions.

An alert is kept iff BOTH families keep it (logical AND). This reproduces the
production v2 semantics: production ANDs all of an event's gates together, and
the split simply partitions those gates by feature family; ANDing the two
per-family decisions is associative-equivalent to ANDing the whole set.

Either family can be toggled off (e.g. motion-only on a device without a usable
SQI gate, or SQI-only on a dataset with no accelerometer like PTB-XL).
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Optional

from .gates import Decision
from .suppressors import MotionFPSuppressor, SQIFPSuppressor
from .device_profiles import DeviceProfile, get_profile


@dataclass
class PipelineResult:
    keep: bool
    motion: Decision
    sqi: Decision

    @property
    def reason(self) -> str:
        if self.keep:
            return "kept"
        who = []
        if not self.motion.keep:
            who.append(f"motion[{self.motion.reason}]")
        if not self.sqi.keep:
            who.append(f"sqi[{self.sqi.reason}]")
        return "suppressed by " + " & ".join(who)


class FPSuppressionPipeline:
    def __init__(self, device="fzark", *, motion: bool = True, sqi: bool = True):
        self.profile: DeviceProfile = (
            device if isinstance(device, DeviceProfile) else get_profile(device))
        self.motion = MotionFPSuppressor(self.profile, enabled=motion)
        self.sqi    = SQIFPSuppressor(self.profile, enabled=sqi)

    def suppress(self, event: str, features: Dict[str, float]) -> PipelineResult:
        m = self.motion.suppress(event, features)
        s = self.sqi.suppress(event, features)
        return PipelineResult(keep=(m.keep and s.keep), motion=m, sqi=s)

    def covered_events(self) -> List[str]:
        return sorted(set(self.motion.covered_events()) | set(self.sqi.covered_events()))
