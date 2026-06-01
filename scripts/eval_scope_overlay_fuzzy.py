"""Evaluate the PTB-XL-trained overlay on the FUZZY derived-lead test set.

Generalization test: L1 was trained on lead-II only; here we score the fold-10
records rendered as derived single-lead at {45,60,75,90}°. Compares:
  baseline   = raw backbone head sigmoid
  this model = deployed routing — L1 for {4,5,6}, base head for {93,98,142}
on each angle + pooled. Labels from the authoritative csv/ptbxl_label.csv.

Run:  python3 -m scripts.eval_scope_overlay_fuzzy
"""
from __future__ import annotations
import csv
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import torch

from checkpoints import load_ecgfounder
from device_utils import resolve_device
import label_config as L
from overlay.scope_overlay import SCOPE_HEADS
from overlay.inference import L1_HEADS, load_l1
from scripts.train_scope_overlay import label_lookup, fuzzy_logits, metrics, FUZZY_DIR, OUT_DIR

ANGLE_FILES = {45: "val_45deg.npz", 60: "val_60deg_split.npz",
               75: "val_75deg.npz", 90: "val_90deg.npz"}


def routed_and_base(logits: np.ndarray, l1) -> tuple[np.ndarray, np.ndarray]:
    base150 = 1.0 / (1.0 + np.exp(-logits))                 # (N,150)
    with torch.no_grad():
        l1p = l1(torch.tensor(logits, dtype=torch.float32)).numpy()  # (N,6) ~ SCOPE_HEADS
    model = np.zeros((len(logits), len(SCOPE_HEADS)), dtype=np.float32)
    base = np.zeros_like(model)
    for i, h in enumerate(SCOPE_HEADS):
        base[:, i] = base150[:, h]
        model[:, i] = l1p[:, i] if h in L1_HEADS else base150[:, h]
    return model, base


def main():
    device = resolve_device()
    backbone = load_ecgfounder(device); backbone.eval()
    l1 = load_l1(device="cpu")
    lut = label_lookup()
    print(f"Device: {device} | scope heads {SCOPE_HEADS} | L1 heads {L1_HEADS}")

    pooled_Y, pooled_model, pooled_base = [], [], []
    rows_out = []
    for ang, fn in ANGLE_FILES.items():
        path = f"{FUZZY_DIR}/{fn}"
        d = np.load(path)
        eids = d["ecg_ids"].astype(int)
        logits = fuzzy_logits(path, backbone, device)        # cached
        Y = np.stack([lut[int(e)] for e in eids])[:, SCOPE_HEADS]
        model_p, base_p = routed_and_base(logits, l1)
        pooled_Y.append(Y); pooled_model.append(model_p); pooled_base.append(base_p)
        mb, mm = metrics(Y, base_p), metrics(Y, model_p)
        print(f"\n=== {ang}°  (n={len(eids)}) baseline → model ===")
        for h in SCOPE_HEADS:
            b, m = mb[h], mm[h]
            src = "L1" if h in L1_HEADS else "base"
            if b["n_pos"] == 0:
                print(f"  {h:>3} {L.DETECTION_SCOPE[h]:<22} npos=0"); continue
            print(f"  {h:>3} {L.DETECTION_SCOPE[h]:<22} npos={b['n_pos']:<4} {src:>4} | "
                  f"AUROC {b['auroc']:.3f}->{m['auroc']:.3f} | "
                  f"PR {b['pr_auc']:.3f}->{m['pr_auc']:.3f} | "
                  f"PPV {b['ppv']:.3f}->{m['ppv']:.3f}")
            for tag, mm_ in (("base", b), ("model", m)):
                rows_out.append([ang, h, b["n_pos"], tag,
                                 f"{mm_['auroc']:.4f}", f"{mm_['pr_auc']:.4f}",
                                 f"{mm_['sens']:.4f}", f"{mm_['ppv']:.4f}"])

    # pooled across all 4 angles
    Y = np.concatenate(pooled_Y); MP = np.concatenate(pooled_model); BP = np.concatenate(pooled_base)
    mb, mm = metrics(Y, BP), metrics(Y, MP)
    print(f"\n=== POOLED 4 angles (n={len(Y)}) baseline → model ===")
    import statistics as st
    ev = [4, 5, 6, 93, 98]
    for h in SCOPE_HEADS:
        b, m = mb[h], mm[h]
        if b["n_pos"] == 0:
            print(f"  {h:>3} {L.DETECTION_SCOPE[h]:<22} npos=0"); continue
        src = "L1" if h in L1_HEADS else "base"
        print(f"  {h:>3} {L.DETECTION_SCOPE[h]:<22} npos={b['n_pos']:<5} {src:>4} | "
              f"AUROC {b['auroc']:.3f}->{m['auroc']:.3f} | PR {b['pr_auc']:.3f}->{m['pr_auc']:.3f} | "
              f"PPV {b['ppv']:.3f}->{m['ppv']:.3f}")
    for k in ("auroc", "pr_auc", "ppv", "sens"):
        vb = st.mean(mb[h][k] for h in ev); vm = st.mean(mm[h][k] for h in ev)
        print(f"  macro {k:<7} {vb:.3f} -> {vm:.3f}  (Δ {vm-vb:+.3f})")

    with open(f"{OUT_DIR}/metrics_fuzzy.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["angle", "head", "n_pos", "model", "auroc", "pr_auc", "sens", "ppv"])
        w.writerows(rows_out)
    print(f"\nsaved {OUT_DIR}/metrics_fuzzy.csv")


if __name__ == "__main__":
    main()
