"""Three-way comparison: base vs iter1 (lead-II L1) vs iter2 (fuzzy L1).

Evaluates on both test distributions (PTB-XL fold-10):
  • lead-II      — from base_probs_full.npy
  • fuzzy        — 4 derived-lead angles {45,60,75,90}°, pooled
Focus on the L1-affected heads {4 Brady, 5 AFib, 6 Sinus-Tachy}; heads
{93,98,142} route to base in every model (identical), shown for reference.

Run:  python3 -m scripts.compare_overlay_iterations
"""
from __future__ import annotations
import os, sys, csv
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import torch
import statistics as st

from checkpoints import load_ecgfounder
from device_utils import resolve_device
import label_config as L
from overlay.scope_overlay import ScopeProjection, SCOPE_HEADS
from overlay.inference import L1_HEADS
from scripts.train_scope_overlay import (
    load_ptbxl_leadii, label_lookup, fold_map, fuzzy_logits, metrics, logit, FUZZY_DIR, OUT_DIR)

ANGLE_FILES = {45: "val_45deg.npz", 60: "val_60deg_split.npz", 75: "val_75deg.npz", 90: "val_90deg.npz"}
ITER1 = f"{OUT_DIR}/scope_projection.pth"            # lead-II
ITER2 = f"{OUT_DIR}/scope_projection_fuzzy.pth"      # fuzzy
ITER3 = f"{OUT_DIR}/scope_projection_union.pth"      # lead-II + fuzzy union
L1COLS = [SCOPE_HEADS.index(h) for h in (4, 5, 6)]


def load_l1(path):
    m = ScopeProjection(); m.load_state_dict(torch.load(path, map_location="cpu")); m.eval(); return m


def probs_for(logits, l1):
    """Return (N,6) aligned to SCOPE_HEADS: L1 for {4,5,6}, base for rest."""
    base = 1.0 / (1.0 + np.exp(-logits))
    out = np.stack([base[:, h] for h in SCOPE_HEADS], 1).astype(np.float32)
    if l1 is not None:
        with torch.no_grad():
            p = l1(torch.tensor(logits, dtype=torch.float32)).numpy()
        for i, h in enumerate(SCOPE_HEADS):
            if h in L1_HEADS:
                out[:, i] = p[:, i]
    return out


def eval_set(name, logits, Y, models):
    print(f"\n################  {name}  (n={len(Y)})  ################")
    M = {tag: metrics(Y, probs_for(logits, l1)) for tag, l1 in models.items()}
    print(f"{'head':<16}{'metric':<8}" + "".join(f"{t:>15}" for t in models))
    for h in (4, 5, 6):
        for k in ("auroc", "pr_auc", "ppv"):
            print(f"{L.DETECTION_SCOPE[h][:15]:<16}{k:<8}" +
                  "".join(f"{M[t][h][k]:>15.3f}" for t in models))
    # macro over 4,5,6
    print(f"{'MACRO(4,5,6)':<16}{'pr_auc':<8}" +
          "".join(f"{st.mean(M[t][h]['pr_auc'] for h in (4,5,6)):>15.3f}" for t in models))
    print(f"{'':<16}{'ppv':<8}" +
          "".join(f"{st.mean(M[t][h]['ppv'] for h in (4,5,6)):>15.3f}" for t in models))
    return M


def main():
    device = resolve_device()
    backbone = load_ecgfounder(device); backbone.eval()
    models = {"base": None, "iter1(leadII)": load_l1(ITER1),
              "iter2(fuzzy)": load_l1(ITER2), "iter3(union)": load_l1(ITER3)}
    lut = label_lookup(); fm = fold_map()

    # lead-II fold-10
    f0, l0, e0 = load_ptbxl_leadii()
    te = np.array([i for i, e in enumerate(e0) if fm.get(int(e)) == 10])
    eval_set("PTB-XL lead-II (fold-10)", f0[te], l0[te][:, SCOPE_HEADS], models)

    # fuzzy fold-10 pooled
    Ls, Ys = [], []
    for ang, fn in ANGLE_FILES.items():
        path = f"{FUZZY_DIR}/{fn}"; d = np.load(path)
        Ls.append(fuzzy_logits(path, backbone, device))
        Ys.append(np.stack([lut[int(e)] for e in d["ecg_ids"].astype(int)])[:, SCOPE_HEADS])
    eval_set("FUZZY derived-lead (fold-10, 4 angles pooled)",
             np.concatenate(Ls), np.concatenate(Ys), models)


if __name__ == "__main__":
    main()
