"""Calibrate the production L1's per-head decision thresholds (L1 TRAINING POLICY).

Policy (rev 2): choose the per-head threshold that MAXIMIZES F1, subject to
sensitivity dropping no more than MAX_SENS_DROP (default 0.05) below baseline,
held on the chosen validation distribution(s) (PTB-XL lead-II fold-9 and/or
fuzzy fold-9) with a safety MARGIN. Reports PR-AUC and the max-F1 threshold.
When multiple distributions are selected, the robust per-head threshold is the
MIN of the per-distribution max-F1 thresholds (keeps the sens floor on both).

Writes decision_threshold into the checkpoint (default: production fuzzySL).
See res/scope_overlay/L1_TRAINING_POLICY.md.
Run:  python3 -m scripts.calibrate_l1_policy           # fuzzy (deployment)
      python3 -m scripts.calibrate_l1_policy --dist leadII
"""
from __future__ import annotations
import os, sys, argparse
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import torch
from sklearn.metrics import average_precision_score

from checkpoints import load_ecgfounder
from device_utils import resolve_device
import label_config as L
from overlay.scope_overlay import ScopeProjection, SCOPE_HEADS, DEFAULT_CKPT
from scripts.train_scope_overlay import (
    load_ptbxl_leadii, label_lookup, fold_map, fuzzy_logits, FUZZY_DIR, POLICY_HEADS, drop_for)

ANGLES = [45, 60, 75, 90]


def sens_at(p, y, thr):
    f = p >= thr; n = int(y.sum())
    return float((f & (y == 1)).sum() / n) if n else float("nan")


def maxf1_threshold(p, y, base_sens, max_drop, margin):
    """argmax-F1 threshold s.t. sens ≥ base_sens − max_drop (+margin).

    Returns (tau, sens, spec, f1, pr_auc) at the chosen point."""
    npos = int(y.sum()); nneg = int((y == 0).sum())
    floor = max(0.0, base_sens - max_drop) + margin
    grid = np.unique(np.concatenate([p, np.linspace(0.01, 0.99, 99)]))
    best = (-1.0, 0.5, 0.0, 0.0)   # f1, tau, sens, spec
    for t in grid:
        f = p >= t
        tp = int((f & (y == 1)).sum()); fp = int((f & (y == 0)).sum())
        s = tp / npos if npos else 0.0
        if s < floor:
            continue
        ppv = tp / max(1, tp + fp)
        f1 = 2 * ppv * s / max(1e-9, ppv + s)
        sp = (nneg - fp) / max(1, nneg)
        if f1 > best[0]:
            best = (f1, float(t), s, sp)
    pr = float(average_precision_score(y, p)) if npos else float("nan")
    return float(np.clip(best[1], 0.01, 0.99)), best[2], best[3], best[0], pr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=DEFAULT_CKPT)
    ap.add_argument("--max-sens-drop", type=float, default=0.05)
    ap.add_argument("--margin", type=float, default=0.02, help="val→test safety margin (pp).")
    ap.add_argument("--dist", nargs="+", choices=["leadII", "fuzzy"],
                    default=["fuzzy"],
                    help="Validation distribution(s) to hold the sens floor on. "
                         "Default 'fuzzy' = the single-lead deployment target. Use "
                         "both for a robust (but spec-sacrificing) global threshold.")
    args = ap.parse_args()

    device = resolve_device()
    backbone = load_ecgfounder(device); backbone.eval()
    l1 = ScopeProjection(); l1.load_state_dict(torch.load(args.ckpt, map_location="cpu")); l1.eval()
    fm = fold_map(); lut = label_lookup()

    # lead-II val (fold 9)
    f0, l0, e0 = load_ptbxl_leadii()
    va0 = np.array([i for i, e in enumerate(e0) if fm.get(int(e)) == 9])
    with torch.no_grad():
        p0 = l1(torch.tensor(f0[va0], dtype=torch.float32)).numpy()
    b0 = 1.0 / (1.0 + np.exp(-f0[va0][:, SCOPE_HEADS])); Y0 = l0[va0][:, SCOPE_HEADS]

    # fuzzy val (fold 9, pooled angles)
    Ls, Ys = [], []
    for a in ANGLES:
        path = f"{FUZZY_DIR}/train_{a}deg.npz"; d = np.load(path)
        eid = d["ecg_ids"].astype(int); keep = np.array([fm.get(int(e)) == 9 for e in eid])
        Ls.append(fuzzy_logits(path, backbone, device)[keep])
        Ys.append(np.stack([lut[int(e)] for e in eid[keep]])[:, SCOPE_HEADS])
    Lf = np.concatenate(Ls); Yf = np.concatenate(Ys)
    with torch.no_grad():
        pf = l1(torch.tensor(Lf, dtype=torch.float32)).numpy()
    bf = 1.0 / (1.0 + np.exp(-Lf[:, SCOPE_HEADS]))

    thr = l1.decision_threshold.clone().numpy()
    from scripts.train_scope_overlay import POLICY_MAX_SENS_DROP
    print(f"Policy: criterion=max-F1, per-head sens-drop {POLICY_MAX_SENS_DROP} "
          f"(default {args.max_sens_drop:.0%}), margin {args.margin:.0%}, dist={args.dist}\n")
    for i, h in enumerate(SCOPE_HEADS):
        if h not in POLICY_HEADS:
            continue
        taus = {}
        sources = [("leadII", p0, b0, Y0), ("fuzzy", pf, bf, Yf)]
        for tag, p, b, Y in [s for s in sources if s[0] in args.dist]:
            y = Y[:, i]
            if y.sum() == 0:
                continue
            base_sens = sens_at(b[:, i], y, 0.5)
            tau, se, sp, f1, pr = maxf1_threshold(
                p[:, i], y, base_sens, drop_for(h, args.max_sens_drop), args.margin)
            taus[tag] = (tau, base_sens, se, sp, f1, pr)
        if not taus:
            continue
        tau = min(v[0] for v in taus.values())     # robust across selected dists
        thr[i] = tau
        det = " | ".join(f"{t}: base_sens={v[1]:.3f} tau*={v[0]:.3f} "
                         f"sens={v[2]:.3f} spec={v[3]:.3f} F1={v[4]:.3f} PR-AUC={v[5]:.3f}"
                         for t, v in taus.items())
        print(f"head {h:>3} {L.DETECTION_SCOPE[h]:<22} -> tau={tau:.3f}\n      {det}")

    l1.decision_threshold.copy_(torch.tensor(thr))
    torch.save(l1.state_dict(), args.ckpt)
    print(f"\nwrote calibrated thresholds into {args.ckpt}: "
          f"{ {h: round(float(thr[i]),3) for i,h in enumerate(SCOPE_HEADS) if h in POLICY_HEADS} }")


if __name__ == "__main__":
    main()
