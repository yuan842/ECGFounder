"""iter1 vs iter3 vs production — all under the SAME L1 policy (max-F1, 5pp).

Each L1 model is calibrated with its own per-head max-F1 thresholds (per
deployment distribution), then evaluated at that operating point. Compares:
  base        raw backbone head @0.5
  iter1       L1 trained on lead-II          (scope_projection.pth)
  iter3       L1 trained on lead-II+fuzzy    (scope_projection_union.pth)
  prod(iter2) L1 trained on fuzzy = fuzzySL  (scope_projection_fuzzy.pth)
across PTB-XL lead-II and fuzzy (fold-10). L1 heads {4,5,6}; 93/98/142 are
base-routed (identical). Writes res/scope_overlay/ITER_POLICY_COMPARISON.md.

Run:  python3 -m scripts.compare_iters_policy
"""
from __future__ import annotations
import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import torch

from checkpoints import load_ecgfounder
from device_utils import resolve_device
import label_config as L
from overlay.scope_overlay import ScopeProjection, SCOPE_HEADS
from scripts.train_scope_overlay import (
    load_ptbxl_leadii, label_lookup, fold_map, fuzzy_logits, FUZZY_DIR, POLICY_HEADS, drop_for)
from scripts.calibrate_l1_policy import maxf1_threshold, sens_at, ANGLES
from scripts.compare_prod_vs_base_detailed import panel
from scripts.compare_overlay_iterations import probs_for

CKPTS = {"iter1": "res/scope_overlay/scope_projection.pth",
         "iter3": "res/scope_overlay/scope_projection_union.pth",
         "prod":  "res/scope_overlay/scope_projection_fuzzy.pth"}
HEADS = [4, 5, 6]
MARGIN = 0.02
OUT = "res/scope_overlay/ITER_POLICY_COMPARISON.md"


def load(p):
    # strict=False: iter1/iter3 predate the decision_threshold buffer (defaults
    # 0.5); thresholds are refit here per the policy anyway.
    m = ScopeProjection(); m.load_state_dict(torch.load(p, map_location="cpu"), strict=False)
    m.eval(); return m


def probs(l1, logits):
    with torch.no_grad():
        return l1(torch.tensor(logits, dtype=torch.float32)).numpy()


def fit(l1, vlog, vY):
    p = probs(l1, vlog); b = 1/(1+np.exp(-vlog[:, SCOPE_HEADS]))
    thr = {}
    for i, h in enumerate(SCOPE_HEADS):
        if h not in POLICY_HEADS or vY[:, i].sum() == 0:
            continue
        bs = sens_at(b[:, i], vY[:, i], 0.5)
        thr[h] = maxf1_threshold(p[:, i], vY[:, i], bs, drop_for(h), MARGIN)[0]
    return thr


def main():
    device = resolve_device()
    backbone = load_ecgfounder(device); backbone.eval()
    fm = fold_map(); lut = label_lookup()
    models = {k: load(v) for k, v in CKPTS.items()}

    # lead-II val/test
    f0, l0, e0 = load_ptbxl_leadii()
    va = np.array([i for i, e in enumerate(e0) if fm.get(int(e)) == 9])
    te = np.array([i for i, e in enumerate(e0) if fm.get(int(e)) == 10])
    LL = dict(vlog=f0[va], vY=l0[va][:, SCOPE_HEADS], tlog=f0[te], tY=l0[te][:, SCOPE_HEADS])

    # fuzzy val (fold9 of train_*deg) / test (val_*deg)
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
    FZ = dict(vlog=np.concatenate(Vl), vY=np.concatenate(Vy),
              tlog=np.concatenate(Tl), tY=np.concatenate(Ty))

    md = ["# iter1 vs iter3 vs production — same L1 policy (max-F1, 5pp)",
          "\nEach L1 model calibrated with its own per-head max-F1 thresholds on the "
          "matching validation distribution, then evaluated at that operating point. "
          "Base = raw head @0.5. L1 heads {4,5,6}; 93/98/142 base-routed (identical).\n"]

    for dname, D in (("PTB-XL lead-II", LL), ("FUZZY derived-lead (4 angles)", FZ)):
        # per-model thresholds fit on this distribution's val
        thr = {k: fit(m, D["vlog"], D["vY"]) for k, m in models.items()}
        base_p = probs_for(D["tlog"], None)
        prod_p = {k: probs_for(D["tlog"], m) for k, m in models.items()}
        md.append(f"\n## {dname} (test n={len(D['tY']):,})\n")
        md.append("| head | metric | base | iter1 | iter3 | prod(fuzzySL) |")
        md.append("|---|---|---:|---:|---:|---:|")
        for h in HEADS:
            i = SCOPE_HEADS.index(h)
            mb = panel(D["tY"][:, i], base_p[:, i], 0.5)
            mm = {k: panel(D["tY"][:, i], prod_p[k][:, i], thr[k].get(h, 0.5)) for k in models}
            md.append(f"| {h} {L.DETECTION_SCOPE[h]} | thr | 0.50 "
                      f"| {thr['iter1'].get(h,.5):.2f} | {thr['iter3'].get(h,.5):.2f} | {thr['prod'].get(h,.5):.2f} |")
            for met in ("sens", "spec", "ppv", "f1", "pr_auc"):
                md.append(f"|  | {met} | {mb[met]:.3f} | " +
                          " | ".join(f"{mm[k][met]:.3f}" for k in models) + " |")
        # macro F1 / PR-AUC over heads
        for met in ("f1", "pr_auc"):
            mbm = np.mean([panel(D["tY"][:, SCOPE_HEADS.index(h)], base_p[:, SCOPE_HEADS.index(h)], 0.5)[met] for h in HEADS])
            row = [np.mean([panel(D["tY"][:, SCOPE_HEADS.index(h)], prod_p[k][:, SCOPE_HEADS.index(h)], thr[k].get(h,.5))[met] for h in HEADS]) for k in models]
            md.append(f"| **MACRO** | {met} | {mbm:.3f} | " + " | ".join(f"{v:.3f}" for v in row) + " |")

    with open(OUT, "w") as fh:
        fh.write("\n".join(md) + "\n")
    print("\n".join(md))
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
