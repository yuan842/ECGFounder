"""Are co-fire firings REAL (annotated)? And does main+co-fire count match GT?

For each scope head, split its firings into MAIN (top-scoring fired scope head
on the segment) vs CO-FIRE (fired but not top), then check each firing against
the PTB-XL annotation (GT). Answers:

  (a) precision of co-fire firings  → are co-fires real or spurious?
  (b) total fired (main+co-fire) vs GT count → does counting all firings
      reproduce the database annotation?

Inputs (split-exempt base backbone):
  res/ptbxl_baseline_sqi/base_probs_full.npy  (21799,150)
  csv/ptbxl_label.csv `label`                 (21799,150) GT
Outputs (res/ptbxl_cofiring/):
  scope_realness_by_role.csv   per head x role: n, tp, fp, precision, mean score
  scope_count_vs_gt.csv        per head: GT count vs main/cofire/total fired + ratio

Run:  python3 -m scripts.ptbxl_cofiring_realness
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
SCOPE = sorted(DETECTION_SCOPE)


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

    P = probs[:, SCOPE]
    fired = P >= THR
    Sgt = (gt[:, SCOPE] == 1)
    idx_of = {h: j for j, h in enumerate(SCOPE)}

    masked = np.where(fired, P, -1.0)
    any_fired = fired.any(axis=1)
    main_j = masked.argmax(axis=1)

    # ── (a) realness by role ───────────────────────────────────────────────
    with open(f"{OUT_DIR}/scope_realness_by_role.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["head", "name", "role", "n_fired", "n_tp_annotated",
                    "n_fp_not_annotated", "precision",
                    "mean_score_tp", "mean_score_fp"])
        for h in SCOPE:
            j = idx_of[h]
            col = fired[:, j]
            gtcol = Sgt[:, j]
            is_main = any_fired & (main_j == j) & col
            is_cofire = col & ~is_main
            for role, mask in (("main", is_main), ("cofire", is_cofire),
                               ("all", col)):
                m = int(mask.sum())
                if m == 0:
                    w.writerow([h, name[h], role, 0, 0, 0, "nan", "nan", "nan"])
                    continue
                tp_mask = mask & gtcol
                fp_mask = mask & ~gtcol
                tp = int(tp_mask.sum())
                fp = m - tp
                sc_tp = P[tp_mask, j]
                sc_fp = P[fp_mask, j]
                w.writerow([
                    h, name[h], role, m, tp, fp, round(tp / m, 4),
                    round(float(sc_tp.mean()), 4) if sc_tp.size else "nan",
                    round(float(sc_fp.mean()), 4) if sc_fp.size else "nan",
                ])

    # ── (b) count vs GT ─────────────────────────────────────────────────────
    with open(f"{OUT_DIR}/scope_count_vs_gt.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["head", "name", "n_gt", "n_main", "n_cofire",
                    "n_total_fired", "total/gt", "n_tp", "recall",
                    "n_fp", "comment"])
        for h in SCOPE:
            j = idx_of[h]
            col = fired[:, j]
            gtcol = Sgt[:, j]
            is_main = any_fired & (main_j == j) & col
            n_main = int(is_main.sum())
            n_total = int(col.sum())
            n_cofire = n_total - n_main
            n_gt = int(gtcol.sum())
            n_tp = int((col & gtcol).sum())
            n_fp = n_total - n_tp
            recall = n_tp / n_gt if n_gt else float("nan")
            ratio = n_total / n_gt if n_gt else float("nan")
            if n_gt == 0:
                comment = "GT has 0 — any firing is FP"
            elif ratio > 2:
                comment = "over-counts GT >2x"
            elif ratio < 0.8:
                comment = "under-counts GT"
            else:
                comment = "roughly matches GT count"
            w.writerow([h, name[h], n_gt, n_main, n_cofire, n_total,
                        round(ratio, 3) if n_gt else "inf",
                        n_tp, round(recall, 3) if n_gt else "nan",
                        n_fp, comment])

    print(f"wrote realness + count-vs-gt tables to {OUT_DIR}/")


if __name__ == "__main__":
    main()
