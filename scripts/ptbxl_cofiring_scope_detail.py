"""Detailed per-head co-firing analysis RESTRICTED TO THE 7 SCOPE HEADS.

Universe = {2 Normal, 4 Brady, 5 AFib, 6 Sinus Tachy, 93 SVT-Run, 98 V-Run,
142 Pause}. The out-of-scope descriptor scaffold (heads 0/1/3/13 …) is ignored
entirely — every co-fire statistic below is scope-vs-scope only.

Inputs (split-exempt base backbone — not trained on PTB-XL):
  res/ptbxl_baseline_sqi/base_probs_full.npy  (21799,150) sigmoid probs
  csv/ptbxl_label.csv  `label` column         (21799,150) GT 0/1

Outputs (res/ptbxl_cofiring/):
  scope_per_head_profile.csv        one rich row per scope head
  scope_conditional_directional.csv P(col fires | row fires), 7x7 asymmetric
  scope_accompaniment_dist.csv      per head: dist of #other scope heads firing
  scope_cofire_tp_vs_fp.csv         per head: co-fire profile split by TP vs FP firing

Run:  python3 -m scripts.ptbxl_cofiring_scope_detail
"""
from __future__ import annotations
import ast
import csv
import os

import numpy as np

from label_config import load_tasks, DETECTION_SCOPE

PROBS_PATH = "res/ptbxl_baseline_sqi/base_probs_full.npy"
LABEL_CSV = "csv/ptbxl_label.csv"
OUT_DIR = "res/ptbxl_cofiring"
THR = 0.5
SCOPE = sorted(DETECTION_SCOPE)  # [2,4,5,6,93,98,142]


def load_gt(n: int) -> np.ndarray:
    rows = []
    with open(LABEL_CSV) as f:
        for rec in csv.DictReader(f):
            rows.append(ast.literal_eval(rec["label"]))
    gt = np.asarray(rows, dtype=np.int8)
    assert gt.shape == (n, 150)
    return gt


def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    tasks = load_tasks()
    name = {i: tasks[i] for i in SCOPE}
    probs = np.load(PROBS_PATH)
    n = probs.shape[0]
    gt = load_gt(n)

    fired = probs >= THR                       # (n,150) bool
    Sfire = fired[:, SCOPE]                     # (n,7) bool — scope predictions
    Sgt = (gt[:, SCOPE] == 1)                   # (n,7) bool — scope GT
    idx_of = {h: j for j, h in enumerate(SCOPE)}

    # ── per-head profile ───────────────────────────────────────────────────
    with open(f"{OUT_DIR}/scope_per_head_profile.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "head", "name", "n_gt", "n_pred", "n_tp", "n_fp",
            "sensitivity", "ppv",
            "n_fire_solo_in_scope", "pct_solo_when_fired",
            "mean_other_scope_when_fired", "max_other_scope_when_fired",
            "top_scope_companion", "companion_rate",
        ])
        for h in SCOPE:
            j = idx_of[h]
            col = Sfire[:, j]
            gtcol = Sgt[:, j]
            n_pred = int(col.sum())
            n_gt = int(gtcol.sum())
            n_tp = int((col & gtcol).sum())
            n_fp = n_pred - n_tp
            sens = n_tp / n_gt if n_gt else float("nan")
            ppv = n_tp / n_pred if n_pred else float("nan")
            if n_pred == 0:
                w.writerow([h, name[h], n_gt, 0, 0, 0,
                            f"{sens:.3f}" if n_gt else "nan", "nan",
                            0, "nan", "nan", "nan", "", ""])
                continue
            sub = Sfire[col]                              # (n_pred,7) rows where h fired
            others = sub.copy()
            others[:, j] = False
            other_cnt = others.sum(axis=1)
            n_solo = int((other_cnt == 0).sum())
            comp = others.sum(axis=0)                     # co-fire counts per scope head
            comp[j] = 0
            top_j = int(np.argmax(comp))
            top_h = SCOPE[top_j]
            comp_rate = comp[top_j] / n_pred
            w.writerow([
                h, name[h], n_gt, n_pred, n_tp, n_fp,
                f"{sens:.3f}" if n_gt else "nan", f"{ppv:.3f}",
                n_solo, f"{n_solo/n_pred:.3f}",
                f"{other_cnt.mean():.3f}", int(other_cnt.max()),
                f"{top_h}:{name[top_h]}" if comp[top_j] > 0 else "(none)",
                f"{comp_rate:.3f}" if comp[top_j] > 0 else "0",
            ])

    # ── directional conditional P(col fires | row fires) ────────────────────
    S = Sfire.astype(np.int64)
    co = S.T @ S                                          # (7,7)
    diag = np.diag(co)
    with open(f"{OUT_DIR}/scope_conditional_directional.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["P(col|row)  row=given"] + [f"{h}:{name[h]}" for h in SCOPE])
        for a, ha in enumerate(SCOPE):
            row = []
            for b in range(len(SCOPE)):
                row.append(round(co[a, b] / diag[a], 4) if diag[a] else 0.0)
            w.writerow([f"{ha}:{name[ha]}"] + row)

    # ── accompaniment distribution: P(#other scope heads = m | head fires) ──
    with open(f"{OUT_DIR}/scope_accompaniment_dist.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["head", "name", "n_pred",
                    "k0_solo", "k1", "k2", "k3", "k4plus"])
        for h in SCOPE:
            j = idx_of[h]
            col = Sfire[:, j]
            n_pred = int(col.sum())
            if n_pred == 0:
                w.writerow([h, name[h], 0, 0, 0, 0, 0, 0])
                continue
            sub = Sfire[col].copy()
            sub[:, j] = False
            kc = sub.sum(axis=1)
            buckets = [int((kc == m).sum()) for m in range(4)]
            buckets.append(int((kc >= 4).sum()))
            w.writerow([h, name[h], n_pred] + buckets)

    # ── co-fire profile split by TP-firing vs FP-firing ─────────────────────
    with open(f"{OUT_DIR}/scope_cofire_tp_vs_fp.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["head", "name", "fire_type", "n",
                    "mean_other_scope", "pct_solo", "top_companion", "rate"])
        for h in SCOPE:
            j = idx_of[h]
            col = Sfire[:, j]
            gtcol = Sgt[:, j]
            for tag, mask in (("TP", col & gtcol), ("FP", col & ~gtcol)):
                m = int(mask.sum())
                if m == 0:
                    w.writerow([h, name[h], tag, 0, "nan", "nan", "", ""])
                    continue
                sub = Sfire[mask].copy()
                sub[:, j] = False
                oc = sub.sum(axis=1)
                comp = sub.sum(axis=0)
                tj = int(np.argmax(comp))
                w.writerow([
                    h, name[h], tag, m,
                    f"{oc.mean():.3f}", f"{(oc==0).sum()/m:.3f}",
                    f"{SCOPE[tj]}:{name[SCOPE[tj]]}" if comp[tj] > 0 else "(none)",
                    f"{comp[tj]/m:.3f}" if comp[tj] > 0 else "0",
                ])

    print(f"wrote 4 scope-detail tables to {OUT_DIR}/")


if __name__ == "__main__":
    main()
