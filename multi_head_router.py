"""Multi-checkpoint per-head router for ECGFounder.

Generalizes `DualHeadECGFounder`: instead of two checkpoints (base + fuzzy)
with a binary routing mask, this router supports an arbitrary set of
checkpoints with a `{head_idx: alias}` routing dict.

The intended production use is to combine:
  - 'base' → checkpoint/1_lead_ECGFounder.pth (V3.1 fzark production model)
  - 'v2'   → checkpoint/1_lead_ECGFounder_6head_v2.pth   (low-recall, high-FP-suppression PTB-XL specialist)
  - 'v3'   → checkpoint/1_lead_ECGFounder_6head_v3_posw.pth (recall-preserving PTB-XL specialist)

Default routing applies the recommendation in
[res/finetune_6head_v2/REGRESSION_REPORT.md](res/finetune_6head_v2):
  - idx 6 (SINUS TACHYCARDIA) → 'v2' (cleans 99% FP rate to 17.6% with no
                                       fzark TPs at risk for this class)
  - all other 149 heads      → 'base'

This default is provably non-regressing on every V3.1 fzark head: idx 6 has
no fzark TP positives, so no TP can be lost; and all other heads use base
weights byte-identically.

Cost: one forward pass per checkpoint that has at least one head routed to
it. Routing optimized to skip checkpoints with no active heads.
"""
from __future__ import annotations

import os
from typing import Optional, Union

import torch
import torch.nn as nn

from checkpoints import load_ecgfounder


# Default checkpoint paths
DEFAULT_CKPTS = {
    'base': "checkpoint/1_lead_ECGFounder.pth",
    'v2':   "checkpoint/1_lead_ECGFounder_6head_v2.pth",
    'v3':   "checkpoint/1_lead_ECGFounder_6head_v3_posw.pth",
}

# Recommended production routing for the 6-head core.
# Every head not in this dict defaults to 'base'.
DEFAULT_ROUTING_6HEAD: dict[int, str] = {
    6: 'v2',   # SINUS TACHYCARDIA  — base has 99% FP rate; v2 cuts to 17.6%
    # idx 4 (Bradycardia): v3 matches base anyway — base
    # idx 5 (AFib):        base + motion-gate already at 100% FP reduction — base
    # idx 9 (PVC):         v2 too conservative, v3 too aggressive — base
    # idx 93 (SVT):        v3 introduces FPs that didn't exist  — base
    # idx 98 (VT):         v3 introduces FPs that didn't exist  — base
}


class MultiHeadRouter(nn.Module):
    """Runs N Net1D checkpoints and merges per-head logits via a routing dict.

    Parameters
    ----------
    device : torch device or string
    checkpoints : {alias: path} mapping. Defaults to base+v2+v3.
    routing : {head_idx: alias} mapping. Heads not specified default to
              the `default_alias` parameter (which itself defaults to 'base').
    default_alias : alias to use for heads not in `routing`. Default 'base'.
    n_classes : output vector length. Default 150.
    """

    def __init__(
        self,
        device,
        checkpoints: Optional[dict[str, str]] = None,
        routing: Optional[dict[int, str]] = None,
        default_alias: str = 'base',
        n_classes: int = 150,
    ):
        super().__init__()
        self.device = device
        self.n_classes = n_classes

        ckpts  = dict(checkpoints) if checkpoints is not None else dict(DEFAULT_CKPTS)
        routing = dict(routing)     if routing     is not None else dict(DEFAULT_ROUTING_6HEAD)
        if default_alias not in ckpts:
            raise ValueError(
                f"default_alias={default_alias!r} not in checkpoints {list(ckpts)}"
            )
        for idx, alias in routing.items():
            if alias not in ckpts:
                raise ValueError(
                    f"routing[{idx}] = {alias!r} not in checkpoints {list(ckpts)}"
                )

        # Build the final per-head alias map (size n_classes).
        head_alias = [default_alias] * n_classes
        for idx, alias in routing.items():
            head_alias[idx] = alias

        # Identify which checkpoints are actually needed (≥1 head routes to it).
        needed = sorted(set(head_alias))
        unused = [a for a in ckpts if a not in needed]
        if unused:
            # We still load them lazily? No — be explicit. Drop unused to save memory.
            print(f"[MultiHeadRouter] skipping unused checkpoints: {unused}")
        ckpts = {a: ckpts[a] for a in needed}

        # Load each needed model
        self.models: dict[str, nn.Module] = {}
        for alias, path in ckpts.items():
            print(f"[MultiHeadRouter] loading {alias!r} ← {path}")
            self.models[alias] = load_ecgfounder(device, ckpt_path=path)
        # Register sub-modules so .to() / .eval() propagate; use ModuleDict.
        self._models = nn.ModuleDict(self.models)

        # Build a (n_classes,) LongTensor of source-model indices (for fast gather).
        alias_to_source = {a: i for i, a in enumerate(needed)}
        source_idx = torch.tensor([alias_to_source[a] for a in head_alias],
                                   dtype=torch.long, device=device)
        self.register_buffer('source_idx', source_idx)
        self._sources = needed
        self.head_alias = head_alias
        self.routing = routing
        self.default_alias = default_alias

        for p in self.parameters():
            p.requires_grad = False

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return merged logits (batch, n_classes).

        Calls every loaded model once on `x`, stacks the outputs into
        (n_sources, batch, n_classes), then per-head picks the source
        designated by `self.source_idx`.
        """
        outs = []
        for alias in self._sources:
            outs.append(self._models[alias](x))
        stacked = torch.stack(outs, dim=0)              # (S, B, n_classes)
        # Gather along the source axis: per-class index = self.source_idx
        # Expand source_idx to (1, B, n_classes) by tiling — actually we can
        # use advanced indexing per the source_idx vector.
        # For each column c, pick stacked[source_idx[c], :, c].
        # Shape trick: gather along dim 0.
        idx = self.source_idx.view(1, 1, -1).expand(1, stacked.shape[1], stacked.shape[2])
        return stacked.gather(0, idx).squeeze(0)        # (B, n_classes)

    def summary(self) -> str:
        """Human-readable summary of the routing."""
        per_src = {}
        for idx in range(self.n_classes):
            per_src.setdefault(self.head_alias[idx], []).append(idx)
        lines = [f"MultiHeadRouter({len(self._sources)} sources)"]
        for alias in self._sources:
            idxs = per_src.get(alias, [])
            lines.append(f"  {alias!r}: {len(idxs)} heads → {idxs if len(idxs) <= 15 else f'{idxs[:5]} … {idxs[-3:]}'}")
        return "\n".join(lines)

    def __repr__(self) -> str:
        return self.summary()
