"""Detailed per-head comparison: baseline vs production (fuzzySL, max-F1 policy).

Production uses the DEPLOYMENT-APPROPRIATE policy thresholds per distribution
(lead-II thresholds on lead-II, fuzzy thresholds on fuzzy — the per-deployment-
lead policy). Full metric panel per head; writes res/scope_overlay/PER_HEAD_DETAILED.md.

Run:  python3 -m scripts.report_prod_per_head
"""
from __future__ import annotations
import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import torch

from checkpoints import load_ecgfounder
from device_utils import resolve_device
import label_config as L
from overlay.scope_overlay import ScopeProjection, SCOPE_HEADS, DEFAULT_CKPT
from overlay.inference import L1_HEADS
from scripts.train_scope_overlay import (
    load_ptbxl_leadii, label_lookup, fold_map, fuzzy_logits, FUZZY_DIR, POLICY_HEADS)
from scripts.calibrate_l1_policy import maxf1_threshold, sens_at, ANGLES
from scripts.compare_prod_vs_base_detailed import panel
from scripts.compare_overlay_iterations import probs_for

MAX_DROP, MARGIN = 0.05, 0.02
OUT = "res/scope_overlay/PER_HEAD_DETAILED.md"


def fit_thresholds(p, b, Y):
    """max-F1/5% per-head thresholds on a validation distribution."""
    thr = {}
    for i, h in enumerate(SCOPE_HEADS):
        if h not in POLICY_HEADS or Y[:, i].sum() == 0:
            continue
        bs = sens_at(b[:, i], Y[:, i], 0.5)
        tau, *_ = maxf1_threshold(p[:, i], Y[:, i], bs, MAX_DROP, MARGIN)
        thr[h] = tau
    return thr


def block(md, name, logits, Y, l1, thr):
    base = probs_for(logits, None)
    prod = probs_for(logits, l1)
    md.append(f"\n## {name} (n={len(Y):,})\n")
    md.append("| head | model | thr | Sens | Spec | PPV | NPV | F1 | AUROC | PR-AUC |")
    md.append("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for i, h in enumerate(SCOPE_HEADS):
        src = "L1" if h in L1_HEADS else "base"
        t = thr.get(h, 0.5) if src == "L1" else 0.5
        mb = panel(Y[:, i], base[:, i], 0.5)
        mp = panel(Y[:, i], prod[:, i], t)
        nm = L.DETECTION_SCOPE[h]
        if mb["n_pos"] == 0:
            md.append(f"| {h} {nm} | — | — | — | 1.000 | — | 1.000 | — | — | — | (0 positives) |")
            continue
        def row(tag, m, tt):
            cells = " | ".join(
                (f"{m[k]:.3f}" if m[k] == m[k] else "—") for k in
                ("sens", "spec", "ppv", "npv", "f1", "auroc", "pr_auc"))
            return f"| {h} {nm} ({src}) | {tag} | {tt:.2f} | {cells} |"
        md.append(row("base", mb, 0.5))
        md.append(row("**prod**", mp, t))
        if src == "L1":
            md.append(f"| | Δ | | {mp['sens']-mb['sens']:+.3f} | {mp['spec']-mb['spec']:+.3f} "
                      f"| {mp['ppv']-mb['ppv']:+.3f} | {mp['npv']-mb['npv']:+.3f} "
                      f"| {mp['f1']-mb['f1']:+.3f} | {mp['auroc']-mb['auroc']:+.3f} "
                      f"| {mp['pr_auc']-mb['pr_auc']:+.3f} |")


def main():
    device = resolve_device()
    backbone = load_ecgfounder(device); backbone.eval()
    l1 = ScopeProjection(); l1.load_state_dict(torch.load(DEFAULT_CKPT, map_location="cpu")); l1.eval()
    fm = fold_map(); lut = label_lookup()

    # lead-II val + test
    f0, l0, e0 = load_ptbxl_leadii()
    va0 = np.array([i for i, e in enumerate(e0) if fm.get(int(e)) == 9])
    te0 = np.array([i for i, e in enumerate(e0) if fm.get(int(e)) == 10])
    with torch.no_grad():
        p0v = l1(torch.tensor(f0[va0], dtype=torch.float32)).numpy()
    thr_leadii = fit_thresholds(p0v, 1/(1+np.exp(-f0[va0][:, SCOPE_HEADS])), l0[va0][:, SCOPE_HEADS])

    # fuzzy val + test
    Lv, Yv, Lt, Yt = [], [], [], []
    for a in ANGLES:
        tr = np.load(f"{FUZZY_DIR}/train_{a}deg.npz"); eid = tr["ecg_ids"].astype(int)
        kv = np.array([fm.get(int(e)) == 9 for e in eid])
        Lv.append(fuzzy_logits(f"{FUZZY_DIR}/train_{a}deg.npz", backbone, device)[kv])
        Yv.append(np.stack([lut[int(e)] for e in eid[kv]])[:, SCOPE_HEADS])
        fn = "val_60deg_split.npz" if a == 60 else f"val_{a}deg.npz"
        ve = np.load(f"{FUZZY_DIR}/{fn}"); eidt = ve["ecg_ids"].astype(int)
        Lt.append(fuzzy_logits(f"{FUZZY_DIR}/{fn}", backbone, device))
        Yt.append(np.stack([lut[int(e)] for e in eidt])[:, SCOPE_HEADS])
    Lv, Yv = np.concatenate(Lv), np.concatenate(Yv)
    with torch.no_grad():
        pfv = l1(torch.tensor(Lv, dtype=torch.float32)).numpy()
    thr_fuzzy = fit_thresholds(pfv, 1/(1+np.exp(-Lv[:, SCOPE_HEADS])), Yv)

    md = ["# Per-head detail — baseline vs production (fuzzySL, max-F1 / 5% policy)",
          "\nProduction = overlay routing (L1 for {4,5,6}, base for {93,98,142}), L2 OFF, "
          "per-head thresholds from the L1 policy (max-F1 s.t. sens loss <5pp), calibrated "
          "**per deployment distribution**. Baseline = raw backbone head at 0.5. "
          "PTB-XL fold-10 test. `thr` = decision threshold used.",
          f"\nProduction thresholds — lead-II: {{ {', '.join(f'{h}:{t:.3f}' for h,t in thr_leadii.items())} }}; "
          f"fuzzy: {{ {', '.join(f'{h}:{t:.3f}' for h,t in thr_fuzzy.items())} }}."]
    block(md, "PTB-XL lead-II", f0[te0], l0[te0][:, SCOPE_HEADS], l1, thr_leadii)
    block(md, "FUZZY derived-lead (4 angles pooled)", np.concatenate(Lt),
          np.concatenate(Yt), l1, thr_fuzzy)
    with open(OUT, "w") as fh:
        fh.write("\n".join(md) + "\n")
    print("\n".join(md))
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
