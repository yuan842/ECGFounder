"""End-to-end scoped inference: backbone → L1 (routed) → L2 arbiter.

Wires the trained L1 ScopeProjection into the detection path with the per-head
routing recommended in res/scope_overlay/REPORT.md:

    L1_HEADS  = {4 Brady, 5 AFib, 6 Sinus-Tachy}  → learned L1 projection
    BASE_HEADS= {93 SVT-Run, 98 V-Run, 142 Pause} → raw backbone head sigmoid

L1 helped the data-rich heads (Brady PR-AUC 0.35→0.60, Tachy PPV 0.33→0.58,
AFib all-up) but degraded the rare SVT/VT and can't learn Pause (0 PTB-XL GT) —
so those route straight from the backbone, and L2 merges the SVT/VT run cluster.

Pipeline (SQG default OFF):
    preprocessed signal → [SQG] → backbone(150) → routed ScopeScores → arbitrate()
"""
from __future__ import annotations
from typing import Optional

import torch

from checkpoints import load_ecgfounder
from device_utils import resolve_device
from overlay.scope_overlay import ScopeProjection, SCOPE_HEADS, NSR_HEAD, DEFAULT_CKPT
from overlay.signal_quality_gate import SignalQualityGate
from overlay.types import ScopeScores, Decision, RunAlert
from overlay.arbiter import arbitrate, to_alerts, ArbiterConfig

L1_HEADS: tuple[int, ...] = (4, 5, 6)        # routed through the learned projection
BASE_HEADS: tuple[int, ...] = (93, 98, 142)  # routed straight from the backbone head


class ScopedDetector:
    """backbone (frozen) + L1 (routed) + L2 arbiter, behind one call."""

    def __init__(self, projection_ckpt: str = DEFAULT_CKPT,
                 device=None, sqg: Optional[SignalQualityGate] = None,
                 arbiter_config: Optional[ArbiterConfig] = None,
                 backbone_ckpt: Optional[str] = None):
        self.device = device or resolve_device()
        self.backbone = load_ecgfounder(self.device, ckpt_path=backbone_ckpt)
        self.backbone.eval()
        self.l1 = load_l1(projection_ckpt, self.device)
        self.sqg = sqg or SignalQualityGate(enabled=False)        # OFF by default
        self.arbiter_config = arbiter_config or ArbiterConfig()   # L2 OFF by default

    @torch.no_grad()
    def score(self, signal: torch.Tensor, *, context: dict | None = None,
              accel=None, sqi=None) -> tuple[ScopeScores | None, str]:
        """signal: (1,1,5000) or (1,5000) preprocessed tensor.

        Returns (ScopeScores, reason). ScopeScores is None iff the SQG (when ON)
        rejects the segment — then `reason` explains why (caller emits 'low-quality').
        """
        g = self.sqg.gate(signal, accel=accel, sqi=sqi)
        if not g.passed:
            return None, g.reason
        x = signal.reshape(1, 1, -1).float().to(self.device)
        logits = self.backbone(x)                      # (1,150)
        l1_probs = self.l1(logits.cpu())               # (1,6), aligned to SCOPE_HEADS
        base = torch.sigmoid(logits).cpu()             # (1,150)
        probs: dict[int, float] = {}
        for i, h in enumerate(SCOPE_HEADS):
            probs[h] = float(l1_probs[0, i]) if h in L1_HEADS else float(base[0, h])
        ss = ScopeScores(
            probs=probs,
            nsr_score=float(base[0, NSR_HEAD]),        # head-1, FP feature for L2
            context=dict(context or {}),
            quality=g.quality,
        )
        return ss, "ok"

    def detect(self, signal: torch.Tensor, **kw) -> dict[int, Decision] | None:
        ss, _ = self.score(signal, **kw)
        return None if ss is None else arbitrate(ss, self.arbiter_config)

    def alerts(self, signal: torch.Tensor, **kw):
        d = self.detect(signal, **kw)
        return (None, None) if d is None else to_alerts(d, self.arbiter_config)


def load_l1(checkpoint_path: str = DEFAULT_CKPT, device="cpu") -> ScopeProjection:
    m = ScopeProjection()
    m.load_state_dict(torch.load(checkpoint_path, map_location=device))
    m.eval()
    return m


if __name__ == "__main__":
    import label_config as L
    det = ScopedDetector()
    x = torch.randn(1, 1, 5000)                        # stand-in preprocessed signal
    ss, why = det.score(x, context={"hr_bpm": 48.0})
    print("ScopeScores:", {h: round(v, 3) for h, v in ss.probs.items()},
          "| nsr=", round(ss.nsr_score, 3))
    print("source     :", {h: ("L1" if h in L1_HEADS else "base") for h in SCOPE_HEADS})
    print(f"L2 enabled : {det.arbiter_config.enabled}  (default OFF)")
    decisions = det.detect(x, context={"hr_bpm": 48.0})
    for h, d in decisions.items():
        print(f"  head {h:>3} {L.DETECTION_SCOPE[h]:<28} fired={d.fired!s:<5} {d.reason}")
    non_run, run = det.alerts(x, context={"hr_bpm": 48.0})
    print("alerts     :", [(L.DETECTION_SCOPE[d.head], round(d.score, 2)) for d in non_run],
          "| run:", run)
