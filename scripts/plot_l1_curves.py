"""F1-vs-threshold and Precision-Recall curves per L1 head, with the policy τ* marked.

For the production L1 heads {4 Brady, 5 AFib, 6 Sinus-Tachy} on PTB-XL fold-10
lead-II and fuzzy test. Shows: (left) F1 across thresholds with τ* (max-F1 under
the sens<5pp constraint) and the sens-floor-forbidden region; (right) the PR
curve with the τ* operating point and the base-head 0.5 point. PR-AUC is the area
(threshold-free) — τ* is just one point on it.

Writes res/scope_overlay/l1_curves_{leadII,fuzzy}.png.
Run:  python3 -m scripts.plot_l1_curves
"""
from __future__ import annotations
import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, average_precision_score

from checkpoints import load_ecgfounder
from device_utils import resolve_device
import label_config as L
from overlay.scope_overlay import ScopeProjection, SCOPE_HEADS, DEFAULT_CKPT
from scripts.train_scope_overlay import (
    load_ptbxl_leadii, label_lookup, fold_map, fuzzy_logits, FUZZY_DIR)
from scripts.calibrate_l1_policy import maxf1_threshold, sens_at, ANGLES

HEADS = [4, 5, 6]
MAX_DROP, MARGIN = 0.05, 0.02
GRID = np.linspace(0.01, 0.99, 197)


def f1_curve(p, y):
    npos = int(y.sum()); nneg = int((y == 0).sum())
    f1s, sens, spec = [], [], []
    for t in GRID:
        f = p >= t
        tp = int((f & (y == 1)).sum()); fp = int((f & (y == 0)).sum())
        s = tp / npos; ppv = tp / max(1, tp + fp)
        f1s.append(2 * ppv * s / max(1e-9, ppv + s)); sens.append(s)
        spec.append((nneg - fp) / max(1, nneg))
    return np.array(f1s), np.array(sens), np.array(spec)


def at(p, y, t):
    f = p >= t; tp = int((f & (y == 1)).sum()); fp = int((f & (y == 0)).sum())
    s = tp / max(1, int(y.sum())); ppv = tp / max(1, tp + fp)
    return s, ppv, 2 * ppv * s / max(1e-9, ppv + s)


def make_fig(name, probs, Y, taus, base_sens):
    fig, ax = plt.subplots(len(HEADS), 2, figsize=(11, 3.2 * len(HEADS)))
    for r, h in enumerate(HEADS):
        i = SCOPE_HEADS.index(h); y = Y[:, i]; p = probs[:, i]
        tau = taus[h]; floor = max(0.0, base_sens[h] - MAX_DROP)
        f1s, sens, spec = f1_curve(p, y)
        prec, rec, thr = precision_recall_curve(y, p)
        ap = average_precision_score(y, p)
        nm = L.DETECTION_SCOPE[h]

        # left: F1 / sens / spec vs threshold
        a = ax[r, 0]
        a.plot(GRID, f1s, "b-", label="F1")
        a.plot(GRID, sens, "g--", lw=1, label="Sens")
        a.plot(GRID, spec, "r:", lw=1, label="Spec")
        a.axhline(floor, color="gray", ls=":", lw=1)
        a.fill_betweenx([0, 1], 0, 0, alpha=0)  # noop
        # forbidden region: thresholds where sens < floor
        bad = GRID[sens < floor]
        if len(bad):
            a.axvspan(bad.min(), 1.0, color="red", alpha=0.06)
        a.axvline(tau, color="k", lw=1.5)
        s_t, ppv_t, f1_t = at(p, y, tau)
        a.plot([tau], [f1_t], "ko", ms=7)
        a.annotate(f"τ*={tau:.3f}\nF1={f1_t:.3f}", (tau, f1_t),
                   textcoords="offset points", xytext=(8, -2), fontsize=8)
        a.axvline(0.5, color="orange", ls="--", lw=1, alpha=0.7)
        a.set_title(f"{h} {nm} — F1 vs threshold", fontsize=9)
        a.set_xlabel("threshold"); a.set_ylim(0, 1.02); a.set_xlim(0, 1)
        if r == 0:
            a.legend(fontsize=7, loc="center right")

        # right: PR curve with τ* and 0.5 points
        b = ax[r, 1]
        b.plot(rec, prec, "b-", label=f"PR (AP={ap:.3f})")
        b.plot([s_t], [ppv_t], "ko", ms=8, label=f"τ*={tau:.2f}")
        s5, ppv5, _ = at(p, y, 0.5)
        b.plot([s5], [ppv5], "s", color="orange", ms=7, label="τ=0.50")
        b.set_title(f"{h} {nm} — PR curve", fontsize=9)
        b.set_xlabel("Recall (Sens)"); b.set_ylabel("Precision (PPV)")
        b.set_xlim(0, 1); b.set_ylim(0, 1.02); b.legend(fontsize=7, loc="upper right")
    fig.suptitle(f"L1 production (fuzzySL) — {name} (PTB-XL fold-10)", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    out = f"res/scope_overlay/l1_curves_{name}.png"
    fig.savefig(out, dpi=120); plt.close(fig)
    print(f"wrote {out}")


def main():
    device = resolve_device()
    backbone = load_ecgfounder(device); backbone.eval()
    l1 = ScopeProjection(); l1.load_state_dict(torch.load(DEFAULT_CKPT, map_location="cpu")); l1.eval()
    fm = fold_map(); lut = label_lookup()

    # lead-II val (fit τ*) + test
    f0, l0, e0 = load_ptbxl_leadii()
    va = np.array([i for i, e in enumerate(e0) if fm.get(int(e)) == 9])
    te = np.array([i for i, e in enumerate(e0) if fm.get(int(e)) == 10])
    with torch.no_grad():
        p0v = l1(torch.tensor(f0[va], dtype=torch.float32)).numpy()
        p0t = l1(torch.tensor(f0[te], dtype=torch.float32)).numpy()
    Yv0 = l0[va][:, SCOPE_HEADS]; Yt0 = l0[te][:, SCOPE_HEADS]
    bsl = {h: sens_at(1/(1+np.exp(-f0[va][:, h])), Yv0[:, SCOPE_HEADS.index(h)], 0.5) for h in HEADS}
    tau_ll = {h: maxf1_threshold(p0v[:, SCOPE_HEADS.index(h)], Yv0[:, SCOPE_HEADS.index(h)],
                                 bsl[h], MAX_DROP, MARGIN)[0] for h in HEADS}
    # baseline sens on TEST (for the floor line reference shown on the test curve)
    bst_ll = {h: sens_at(1/(1+np.exp(-f0[te][:, h])), Yt0[:, SCOPE_HEADS.index(h)], 0.5) for h in HEADS}
    make_fig("leadII", p0t, Yt0, tau_ll, bst_ll)

    # fuzzy test + shipped thresholds
    Lt, Yt = [], []
    for a in ANGLES:
        fn = "val_60deg_split.npz" if a == 60 else f"val_{a}deg.npz"
        d = np.load(f"{FUZZY_DIR}/{fn}")
        Lt.append(fuzzy_logits(f"{FUZZY_DIR}/{fn}", backbone, device))
        Yt.append(np.stack([lut[int(e)] for e in d["ecg_ids"].astype(int)])[:, SCOPE_HEADS])
    Lt = np.concatenate(Lt); Yt = np.concatenate(Yt)
    with torch.no_grad():
        pft = l1(torch.tensor(Lt, dtype=torch.float32)).numpy()
    tau_fz = {h: l1.thresholds()[h] for h in HEADS}
    bst_fz = {h: sens_at(1/(1+np.exp(-Lt[:, h])), Yt[:, SCOPE_HEADS.index(h)], 0.5) for h in HEADS}
    make_fig("fuzzy", pft, Yt, tau_fz, bst_fz)


if __name__ == "__main__":
    main()
