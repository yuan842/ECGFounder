"""Per-head comparison WITH L2 ON: base vs iter1 vs iter3 vs production.

Each model at its own per-head policy thresholds (per deployment distribution),
then the GT-matched L2 arbiter applied (exclusion {4,5,6}, couple {93,98}).
Base = raw heads @0.5 + L2. Metrics from post-L2 decisions; PR-AUC/AUROC are
score-level (L2 does not change them). PTB-XL lead-II + fuzzy fold-10.

Writes res/scope_overlay/ITER_L2_COMPARISON.md.
Run:  python3 -m scripts.compare_iters_l2
"""
from __future__ import annotations
import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import torch
from sklearn.metrics import roc_auc_score, average_precision_score

from checkpoints import load_ecgfounder
from device_utils import resolve_device
import label_config as L
from overlay.scope_overlay import ScopeProjection, SCOPE_HEADS
from overlay.arbiter import arbitrate, ArbiterConfig
from overlay.types import ScopeScores
from scripts.train_scope_overlay import (
    load_ptbxl_leadii, label_lookup, fold_map, fuzzy_logits, FUZZY_DIR, POLICY_HEADS, drop_for)
from scripts.calibrate_l1_policy import maxf1_threshold, sens_at, ANGLES
from scripts.compare_overlay_iterations import probs_for

CKPTS = {"iter1": "res/scope_overlay/scope_projection.pth",
         "iter3": "res/scope_overlay/scope_projection_union.pth",
         "prod":  "res/scope_overlay/scope_projection_fuzzy.pth"}
HEADS = [4, 5, 6]
OUT = "res/scope_overlay/ITER_L2_COMPARISON.md"


def load(p):
    m = ScopeProjection(); m.load_state_dict(torch.load(p, map_location="cpu"), strict=False)
    m.eval(); return m


def fit(l1, vlog, vY):
    with torch.no_grad():
        p = l1(torch.tensor(vlog, dtype=torch.float32)).numpy()
    b = 1/(1+np.exp(-vlog[:, SCOPE_HEADS]))
    thr = {}
    for i, h in enumerate(SCOPE_HEADS):
        if h not in POLICY_HEADS or vY[:, i].sum() == 0:
            continue
        thr[h] = maxf1_threshold(p[:, i], vY[:, i], sens_at(b[:, i], vY[:, i], 0.5), drop_for(h), 0.02)[0]
    return thr


def l2_fire(probs, thr):
    """Run GT-matched L2 per row → (N,6) fired bool matrix."""
    fire = {h: 0.5 for h in SCOPE_HEADS}; fire.update(thr)
    cfg = ArbiterConfig(enabled=True, fire_threshold=fire)
    out = np.zeros((len(probs), len(SCOPE_HEADS)), bool)
    for r in range(len(probs)):
        d = arbitrate(ScopeScores(probs={h: float(probs[r, j]) for j, h in enumerate(SCOPE_HEADS)},
                                  nsr_score=0.0), cfg)
        out[r] = [d[h].fired for h in SCOPE_HEADS]
    return out


def panel(y, fired, score):
    y = y.astype(int)
    tp = int((fired & (y == 1)).sum()); fp = int((fired & (y == 0)).sum())
    fn = int((~fired & (y == 1)).sum()); tn = int((~fired & (y == 0)).sum())
    npos = tp + fn
    sens = tp/npos if npos else float("nan")
    spec = tn/(tn+fp) if tn+fp else float("nan")
    ppv = tp/(tp+fp) if tp+fp else float("nan")
    f1 = 2*ppv*sens/(ppv+sens) if (ppv == ppv and sens == sens and ppv+sens) else float("nan")
    pr = average_precision_score(y, score) if npos else float("nan")
    return dict(sens=sens, spec=spec, ppv=ppv, f1=f1, pr_auc=pr)


def main():
    device = resolve_device()
    backbone = load_ecgfounder(device); backbone.eval()
    models = {"base": None} | {k: load(v) for k, v in CKPTS.items()}
    fm = fold_map(); lut = label_lookup()

    f0, l0, e0 = load_ptbxl_leadii()
    va = np.array([i for i, e in enumerate(e0) if fm.get(int(e)) == 9])
    te = np.array([i for i, e in enumerate(e0) if fm.get(int(e)) == 10])
    LL = dict(vlog=f0[va], vY=l0[va][:, SCOPE_HEADS], tlog=f0[te], tY=l0[te][:, SCOPE_HEADS])

    Vl, Vy, Tl, Ty = [], [], [], []
    for a in ANGLES:
        tr = np.load(f"{FUZZY_DIR}/train_{a}deg.npz"); eid = tr["ecg_ids"].astype(int)
        kv = np.array([fm.get(int(e)) == 9 for e in eid])
        Vl.append(fuzzy_logits(f"{FUZZY_DIR}/train_{a}deg.npz", backbone, device)[kv])
        Vy.append(np.stack([lut[int(e)] for e in eid[kv]])[:, SCOPE_HEADS])
        fn = "val_60deg_split.npz" if a == 60 else f"val_{a}deg.npz"
        ve = np.load(f"{FUZZY_DIR}/{fn}"); et = ve["ecg_ids"].astype(int)
        Tl.append(fuzzy_logits(f"{FUZZY_DIR}/{fn}", backbone, device))
        Ty.append(np.stack([lut[int(e)] for e in et])[:, SCOPE_HEADS])
    FZ = dict(vlog=np.concatenate(Vl), vY=np.concatenate(Vy), tlog=np.concatenate(Tl), tY=np.concatenate(Ty))

    md = ["# Per-head with L2 ON — base vs iter1 vs iter3 vs production",
          "\nEach model at its per-head max-F1/5pp thresholds (per deployment dist), then "
          "GT-matched L2 (exclusion {4,5,6}, couple {93,98}). Base = raw heads @0.5 + L2. "
          "Metrics from post-L2 decisions; PR-AUC is score-level (L2-invariant). "
          "Heads 93/98/142 are base-routed and L2 keeps the couple → identical across models.\n"]

    for dname, D in (("PTB-XL lead-II", LL), ("FUZZY derived-lead (4 angles)", FZ)):
        decs = {}
        for k, m in models.items():
            thr = fit(m, D["vlog"], D["vY"]) if m is not None else {}
            probs = probs_for(D["tlog"], m)        # routed (base if m None)
            decs[k] = (l2_fire(probs, thr), probs, thr)
        md.append(f"\n## {dname} (test n={len(D['tY']):,})\n")
        md.append("| head | metric | base+L2 | iter1+L2 | iter3+L2 | prod+L2 |")
        md.append("|---|---|---:|---:|---:|---:|")
        for h in HEADS:
            i = SCOPE_HEADS.index(h)
            P = {k: panel(D["tY"][:, i], decs[k][0][:, i], decs[k][1][:, i]) for k in models}
            md.append(f"| {h} {L.DETECTION_SCOPE[h]} | thr | 0.50 | " +
                      " | ".join(f"{decs[k][2].get(h,0.5):.2f}" for k in ("iter1","iter3","prod")) + " |")
            for met in ("sens", "spec", "ppv", "f1", "pr_auc"):
                md.append(f"|  | {met} | " + " | ".join(f"{P[k][met]:.3f}" for k in models) + " |")
        for met in ("f1",):
            row = {k: np.mean([panel(D["tY"][:, SCOPE_HEADS.index(h)], decs[k][0][:, SCOPE_HEADS.index(h)], decs[k][1][:, SCOPE_HEADS.index(h)])[met] for h in HEADS]) for k in models}
            md.append(f"| **MACRO {met}** |  | " + " | ".join(f"{row[k]:.3f}" for k in models) + " |")

    with open(OUT, "w") as fh:
        fh.write("\n".join(md) + "\n")
    print("\n".join(md)); print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
