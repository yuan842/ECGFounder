"""Device-specific overlay configuration (single file per device).

A device config records, per scope head, where its decision comes from and at
what threshold:
  source == "<name>-L1"  → routed through the named L1 checkpoint at `threshold`
  source == "base"       → raw backbone head at `threshold` (0.5; e.g. heads with
                           0 calibration events on this device — recalibrate when
                           a new device is identified)

The L1 weights live in `l1_checkpoint`. Built by scripts/train_l1_fzark.py
(fzark profile); the PTB profile remains separate (overlay.scope_overlay
DEFAULT_CKPT = fuzzySL.pth + POLICY_MAX_SENS_DROP). See res/scope_overlay/
device_configs/.
"""
from __future__ import annotations
import json
from dataclasses import dataclass


@dataclass(frozen=True)
class DeviceConfig:
    device: str
    l1_checkpoint: str
    heads: dict[int, dict]        # head idx → {source, threshold, fp_limit, ...}

    def l1_heads(self) -> list[int]:
        """Heads routed through the L1 checkpoint (source endswith '-L1')."""
        return [h for h, c in self.heads.items() if str(c.get("source", "")).endswith("-L1")]

    def fire_thresholds(self) -> dict[int, float]:
        """Per-head decision threshold for every scope head (base heads → 0.5)."""
        return {h: float(c.get("threshold", 0.5)) for h, c in self.heads.items()}


def load_device_config(path: str) -> DeviceConfig:
    with open(path) as f:
        d = json.load(f)
    heads = {int(h): c for h, c in d["heads"].items()}
    return DeviceConfig(device=d["device"], l1_checkpoint=d["l1_checkpoint"], heads=heads)
