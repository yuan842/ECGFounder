"""Detailed baseline vs production (fuzzySL) comparison, PTB-XL + fuzzy.

Production = deployed overlay routing: fuzzySL L1 for {4,5,6}, base head for
{93,98,142}, L2 OFF. Baseline = raw backbone heads. Full metric panel per head
at threshold 0.5 (sens/spec/PPV/NPV/F1 + confusion counts) plus threshold-free
AUROC and PR-AUC. Test = PTB-XL fold-10: lead-II and fuzzy (4 angles pooled).

Run:  python3 -m scripts.compare_prod_vs_base_detailed
"""
from __future__ import annotations
import os, sys, csv
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import torch
from sklearn.metrics import roc_auc_score, average_precision_score

from checkpoints import load_ecgfounder
from device_utils import resolve_device
import label_config as L
from overlay.scope_overlay import SCOPE_HEADS
from overlay.inference import L1_HEADS, load_l1
from scripts.train_scope_overlay import (
    load_ptbxl_leadii, label_lookup, fold_map, fuzzy_logits, FUZZY_DIR, OUT_DIR, drop_for)
from scripts.compare_overlay_iterations import probs_for, ANGLE_FILES

def panel(y, p, thr=0.5):
    """Full metric dict at decision threshold `thr` + threshold-free AUROC/PR-AUC."""
    y = y.astype(int); fired = p >= thr
    tp = int((fired & (y == 1)).sum()); fp = int((fired & (y == 0)).sum())
    fn = int((~fired & (y == 1)).sum()); tn = int((~fired & (y == 0)).sum())
    npos = tp + fn
    sens = tp / npos if npos else float("nan")
    spec = tn / (tn + fp) if (tn + fp) else float("nan")
    ppv = tp / (tp + fp) if (tp + fp) else float("nan")
    npv = tn / (tn + fn) if (tn + fn) else float("nan")
    f1 = (2 * ppv * sens / (ppv + sens)) if (ppv + sens) else float("nan")
    try:
        au = roc_auc_score(y, p) if 0 < npos < len(y) else float("nan")
        pr = average_precision_score(y, p) if npos else float("nan")
    except Exception:
        au = pr = float("nan")
    return dict(n_pos=npos, tp=tp, fp=fp, fn=fn, tn=tn, sens=sens, spec=spec,
               ppv=ppv, npv=npv, f1=f1, auroc=au, pr_auc=pr)


def eval_set(name, logits, Y, rows):
    base = probs_for(logits, None)
    prod = probs_for(logits, PROD)
    thr = PROD.thresholds()                       # per-head policy thresholds
    print(f"\n################  {name}  (n={len(Y)})  ################")
    hdr = ["head", "model", "thr", "sens", "spec", "PPV", "NPV", "F1", "AUROC", "PR-AUC"]
    print("  ".join(f"{h:>7}" for h in hdr) + "   policy")
    for i, h in enumerate(SCOPE_HEADS):
        src = "L1*" if h in L1_HEADS else "base"
        mb = panel(Y[:, i], base[:, i], 0.5)
        tprod = thr[h] if h in L1_HEADS else 0.5
        mp = panel(Y[:, i], prod[:, i], tprod)
        for tag, m, t in (("base", mb, 0.5), ("prod", mp, tprod)):
            print(f"{h:>7} {tag:>7} {t:>7.2f} " +
                  " ".join(f"{m[k]:>7.3f}" if m[k] == m[k] else f"{'—':>7}"
                           for k in ("sens", "spec", "ppv", "npv", "f1", "auroc", "pr_auc")))
            rows.append([name, h, L.DETECTION_SCOPE[h], src, tag, f"{t:.3f}", m["n_pos"],
                         m["tp"], m["fp"], m["fn"], m["tn"],
                         *[f"{m[k]:.4f}" for k in ("sens","spec","ppv","npv","f1","auroc","pr_auc")]])
        # policy compliance (L1 heads only, heads with positives)
        if src == "L1*" and mb["n_pos"] > 0:
            dsens = mp["sens"] - mb["sens"]; dspec = mp["spec"] - mb["spec"]
            df1 = mp["f1"] - mb["f1"]
            budget = drop_for(h, 0.05)
            ok = "PASS" if (dsens >= -budget and dspec > 0) else "FAIL"
            print(f"{'':>7} {'Δ':>7} {'':>7} {dsens:>+7.3f} {dspec:>+7.3f} {'':>15} {df1:>+7.3f}"
                  f" {'':>15} -> {ok} (Δsens≥-{budget:.2f} & Δspec>0; ΔF1 shown)")


def main():
    global PROD
    device = resolve_device()
    backbone = load_ecgfounder(device); backbone.eval()
    PROD = load_l1()                       # production fuzzySL via DEFAULT_CKPT
    lut = label_lookup(); fm = fold_map()
    rows = []

    f0, l0, e0 = load_ptbxl_leadii()
    te = np.array([i for i, e in enumerate(e0) if fm.get(int(e)) == 10])
    eval_set("PTB-XL lead-II (fold-10)", f0[te], l0[te][:, SCOPE_HEADS], rows)

    Ls, Ys = [], []
    for fn in ANGLE_FILES.values():
        path = f"{FUZZY_DIR}/{fn}"; d = np.load(path)
        Ls.append(fuzzy_logits(path, backbone, device))
        Ys.append(np.stack([lut[int(e)] for e in d["ecg_ids"].astype(int)])[:, SCOPE_HEADS])
    eval_set("FUZZY derived-lead (fold-10, 4 angles pooled)",
             np.concatenate(Ls), np.concatenate(Ys), rows)

    with open(f"{OUT_DIR}/prod_vs_base_detailed.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["test_set","head","name","source","model","thr","n_pos","tp","fp","fn","tn",
                    "sens","spec","ppv","npv","f1","auroc","pr_auc"])
        w.writerows(rows)
    print(f"\nsaved {OUT_DIR}/prod_vs_base_detailed.csv")


if __name__ == "__main__":
    main()
