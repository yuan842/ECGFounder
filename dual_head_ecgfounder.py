"""Two-checkpoint inference wrapper for ECGFounder.

Combines two Net1D checkpoints — typically `1_lead_ECGFounder.pth` (base) and
`1_lead_ECGFounder_fuzzy.pth` (fine-tuned on fuzzylead2 derived single-lead) —
and emits a merged 150-class output where each head is routed to the better
of the two models.

Why this exists
---------------
Fine-tuning ECGFounder on a dataset that only labels a subset of the 150
classes (fuzzylead2 labels 10 of 150) catastrophically forgets the other
140 heads. State-dict surgery on the classifier rows alone doesn't work —
the classifier rows are entangled with the backbone they trained against
(see `res/finetune_fuzzylead2/README.md` §"Root cause" and the B1/B2
experiments). The only zero-compromise solution is to run both models and
pick per-head.

Cost is roughly 2× a single inference pass. For batch inference at MPS rates
(~2 it/s base) that's a few hundred records / s, plenty for any realistic
workload.

Routing modes
-------------
- `'ptbxl_specific'` (default): 8 PTB-XL-specific labels from fuzzy
    [2, 18, 26, 32, 36, 62, 70, 82] — NORMAL ECG, IRBBB, LVH, AFL, LAFB,
    ILBBB, LPFB, RVH. The other 142 heads (including AFib idx 5 and Sinus
    Tachycardia idx 6) come from base, because the fuzzy training data
    has too few positives for those to remain reliable on fzark.

- `'all_active'`: all 10 fuzzylead2-active heads from fuzzy
    [2, 5, 6, 18, 26, 32, 36, 62, 70, 82]. Use only if your downstream
    task is on the fuzzylead2 / PTB-XL domain.

- pass a `head_routing: dict[int, str]` to override per-head:
    {5: 'base', 6: 'base', 2: 'fuzzy', ...}
"""
from __future__ import annotations

import os
from typing import Optional, Union

import torch
import torch.nn as nn

from checkpoints import load_ecgfounder


# Default routing dictionaries (head index → 'base' | 'fuzzy')
# Unspecified heads default to 'base'.
PTBXL_SPECIFIC_FUZZY_HEADS = (2, 18, 26, 32, 36, 62, 70, 82)
ALL_ACTIVE_FUZZY_HEADS     = (2, 5, 6, 18, 26, 32, 36, 62, 70, 82)


class DualHeadECGFounder(nn.Module):
    """Runs two Net1D models and merges per-head outputs.

    Parameters
    ----------
    device : torch device or string
    base_ckpt, fine_ckpt : paths to the two checkpoints.
    routing : 'ptbxl_specific' (default) | 'all_active' | dict[int, str]
        Per-head routing — see module docstring.
    """

    def __init__(
        self,
        device,
        base_ckpt: str = "checkpoint/1_lead_ECGFounder.pth",
        fine_ckpt: str = "checkpoint/1_lead_ECGFounder_fuzzy.pth",
        routing: Union[str, dict[int, str]] = 'ptbxl_specific',
        n_classes: int = 150,
    ):
        super().__init__()
        self.device = device
        self.n_classes = n_classes

        # Resolve routing → boolean mask of size (n_classes,), 1 = use fuzzy.
        if isinstance(routing, str):
            if routing == 'ptbxl_specific':
                fuzzy_idx = PTBXL_SPECIFIC_FUZZY_HEADS
            elif routing == 'all_active':
                fuzzy_idx = ALL_ACTIVE_FUZZY_HEADS
            else:
                raise ValueError(
                    f"Unknown routing string: {routing!r}. Expected "
                    f"'ptbxl_specific', 'all_active', or a dict."
                )
            mask = torch.zeros(n_classes, dtype=torch.bool)
            mask[list(fuzzy_idx)] = True
        elif isinstance(routing, dict):
            mask = torch.zeros(n_classes, dtype=torch.bool)
            for idx, who in routing.items():
                if who not in ('base', 'fuzzy'):
                    raise ValueError(f"head {idx}: who={who!r}, expected 'base'|'fuzzy'")
                if who == 'fuzzy':
                    mask[idx] = True
        else:
            raise TypeError(f"routing must be str or dict, got {type(routing)}")
        # Register as buffer so it moves with .to(device) and saves with state_dict.
        self.register_buffer('use_fuzzy_mask', mask.to(device))

        # Load both backbones. Each is set to eval() by load_ecgfounder.
        self.base = load_ecgfounder(device, ckpt_path=base_ckpt)
        self.fuzzy = load_ecgfounder(device, ckpt_path=fine_ckpt)
        for p in self.parameters():
            p.requires_grad = False

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return merged logits, shape (batch, 150)."""
        logits_base  = self.base(x)
        logits_fuzzy = self.fuzzy(x)
        # Broadcasting: mask is (150,), logits are (B, 150). torch.where picks
        # per-column (per-head) which source to use.
        return torch.where(self.use_fuzzy_mask, logits_fuzzy, logits_base)

    @property
    def fuzzy_head_indices(self) -> list[int]:
        """List the 150-class indices currently routed through the fuzzy model."""
        return torch.nonzero(self.use_fuzzy_mask, as_tuple=True)[0].tolist()

    def __repr__(self) -> str:
        return (
            f"DualHeadECGFounder("
            f"n_fuzzy_heads={int(self.use_fuzzy_mask.sum())}, "
            f"n_base_heads={int((~self.use_fuzzy_mask).sum())}, "
            f"fuzzy_idx={self.fuzzy_head_indices})"
        )
