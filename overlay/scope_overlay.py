"""L1 — learned scope projection (Decision 1 = projection, recommended).

Maps the frozen backbone's 150 head logits → 6 calibrated scope probabilities.
Reads the FULL 150-dim output (not just the 6 scope rows) so the descriptor
scaffold (ABNORMAL ECG, SINUS RHYTHM, LAE, run-adjacent heads, NSR) can serve
as features — the cross-head signal the in-place row-tuning form cannot use.

THIS IS A SKELETON: weights are randomly initialised. Training is the separate
next step — multi-dataset union (PTB-XL + fzark + MIMIC + MIT-BIH + CINC2015),
per-head pos_weight for rare heads, post-hoc per-head temperature calibration,
PTB-XL fold split respected, gated by an extended eval_6head_regression_gate.
See res/OVERLAY_DESIGN.md §3.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from label_config import DETECTION_SCOPE
from overlay.types import ScopeScores

SCOPE_HEADS: list[int] = sorted(DETECTION_SCOPE)   # [4, 5, 6, 93, 98, 142]
NSR_HEAD = 1                                        # FP feature passed to L2


class ScopeProjection(nn.Module):
    """150 → 6 linear projection + per-head temperature calibration.

    forward(logits150) -> (B, 6) calibrated sigmoid probabilities, column order
    = SCOPE_HEADS. Start linear (≈906 params, interpretable: read `proj.weight`
    as which of the 150 heads drives each scope head); swap in an MLP only if
    validation PR-AUC is short (OVERLAY_DESIGN §3 variant b).
    """

    def __init__(self, n_in: int = 150, heads: list[int] = SCOPE_HEADS):
        super().__init__()
        self.heads = list(heads)
        self.proj = nn.Linear(n_in, len(self.heads))
        # per-head temperature for post-hoc calibration (1.0 = identity until fit)
        self.register_buffer("temperature", torch.ones(len(self.heads)))
        # per-head decision threshold set by the L1 TRAINING POLICY (spec-optimised
        # under a max sensitivity-drop constraint). 0.5 until fit on validation.
        # See res/scope_overlay/L1_TRAINING_POLICY.md.
        self.register_buffer("decision_threshold", 0.5 * torch.ones(len(self.heads)))

    def forward(self, logits150: torch.Tensor) -> torch.Tensor:
        z = self.proj(logits150) / self.temperature
        return torch.sigmoid(z)

    def thresholds(self) -> dict[int, float]:
        """Per-head decision thresholds (policy-fit) keyed by tasks.txt index."""
        return {h: float(self.decision_threshold[i]) for i, h in enumerate(self.heads)}

    @torch.no_grad()
    def predict(self, logits150: torch.Tensor,
                context: dict[str, float] | None = None) -> ScopeScores:
        """Single-sample helper → the L1→L2 boundary object.

        logits150: shape (150,) or (1,150). nsr_score is read straight from the
        frozen backbone (head 1) — it is a feature for L2, never a detection.
        """
        x = logits150.reshape(1, -1).float()
        probs = self.forward(x).squeeze(0)
        nsr = float(torch.sigmoid(x[0, NSR_HEAD]))
        return ScopeScores(
            probs={h: float(probs[i]) for i, h in enumerate(self.heads)},
            nsr_score=nsr,
            context=dict(context or {}),
        )


# Production L1 = "fuzzySL" — iteration 2, trained on PTB-XL fuzzy derived-leads
# (45/60/75/90°). Chosen for its precision: best lead-II PPV (macro 0.66) and the
# most conservative / lowest-false-alert head behaviour. See
# res/scope_overlay/ITERATION_COMPARISON.md. Other iterations kept for reference:
#   scope_projection.pth        iter1 (lead-II)
#   scope_projection_fuzzy.pth  iter2 (fuzzy)  ← same weights as fuzzySL.pth
#   scope_projection_union.pth  iter3 (union)
DEFAULT_CKPT = "res/scope_overlay/fuzzySL.pth"


def load(checkpoint_path: str = DEFAULT_CKPT, device: str = "cpu") -> ScopeProjection:
    """Load the production L1 projection (fuzzySL; see res/scope_overlay/ITERATION_COMPARISON.md)."""
    m = ScopeProjection()
    m.load_state_dict(torch.load(checkpoint_path, map_location=device))
    m.eval()
    return m


if __name__ == "__main__":
    m = ScopeProjection()
    fake_logits = torch.randn(150)               # stand-in for backbone output
    ss = m.predict(fake_logits, context={"hr_bpm": 60.0})
    print("scope heads :", m.heads)
    print("probs       :", {h: round(v, 3) for h, v in ss.probs.items()})
    print("nsr_score   :", round(ss.nsr_score, 3), "(feature for L2, not a label)")
    print("NOTE: weights are random — this is a skeleton, not a trained model.")
