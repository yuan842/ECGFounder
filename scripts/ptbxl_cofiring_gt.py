"""Ground-truth co-fire confusion mapping among the 8 scope heads (PTB-XL).

Uses ONLY the PTB-XL annotations (`csv/ptbxl_label.csv` `label`), not model
predictions. Answers: on a 10-second segment, which of the 8 scope heads does
the database actually co-label together? This is the REFERENCE the model's
predicted co-firing should be judged against.

Scope = {1 NSR, 2 Normal ECG, 4 Brady, 5 AFib, 6 Sinus Tachy, 93 SVT-Run,
98 V-Run, 142 Pause}.

Inputs:
  csv/ptbxl_label.csv `label`  (21799,150) GT 0/1
  (optional) res/ptbxl_baseline_sqi/base_probs_full.npy for a pred-vs-GT contrast
Outputs (res/ptbxl_cofiring/):
  gt_labels_per_segment.csv        dist of #scope GT labels per segment
  gt_cofire_counts.csv             8x8 GT co-occurrence counts
  gt_cofire_jaccard.csv            8x8 GT Jaccard
  gt_conditional_directional.csv   P(GT col | GT row)
  gt_vs_pred_cofire.csv            per scope pair: GT vs predicted co-fire (Jaccard + counts)

Run:  python3 -m scripts.ptbxl_cofiring_gt
"""
from __future__ import annotations
import ast
import csv
import os

import numpy as np

from label_config import load_tasks, DETECTION_SCOPE

LABEL_CSV = "csv/ptbxl_label.csv"
PROBS_PATH = "res/ptbxl_baseline_sqi/base_probs_full.npy"
OUT_DIR = "res/ptbxl_cofiring"
THR = 0.5
SCOPE = sorted(DETECTION_SCOPE)  # [1,2,4,5,6,93,98,142]


def load_gt(n_expected: int | None = None) -> np.ndarray:
    rows = []
    with open(LABEL_CSV) as f:
        for rec in csv.DictReader(f):
            rows.append(ast.literal_eval(rec["label"]))
    gt = np.asarray(rows, dtype=np.int8)
    assert gt.shape[1] == 150
    if n_expected is not None:
        assert gt.shape[0] == n_expected
    return gt


def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    tasks = load_tasks()
    name = {i: tasks[i] for i in SCOPE}
    gt = load_gt()
    n = gt.shape[0]

    G = (gt[:, SCOPE] == 1).astype(np.int64)     # (n,8) GT scope labels
    counts = {h: int(G[:, j].sum()) for j, h in enumerate(SCOPE)}
    print("GT scope label counts:")
    for h in SCOPE:
        print(f"  head {h:>3} {name[h]:<28} n_gt={counts[h]}")

    # ── 1. #scope GT labels per segment ────────────────────────────────────
    per_seg = G.sum(axis=1)
    from collections import Counter
    dist = Counter(per_seg.tolist())
    with open(f"{OUT_DIR}/gt_labels_per_segment.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["n_scope_gt_labels", "n_segments"])
        for k in range(0, int(per_seg.max()) + 1):
            w.writerow([k, dist.get(k, 0)])
    print(f"\nGT scope labels/segment: mean={per_seg.mean():.3f} "
          f"median={int(np.median(per_seg))} max={int(per_seg.max())} "
          f"| segments with >=2 scope labels: {(per_seg>=2).sum()}")

    # ── 2. GT co-occurrence counts + Jaccard ───────────────────────────────
    co = G.T @ G                                  # (8,8)
    diag = np.diag(co)
    with open(f"{OUT_DIR}/gt_cofire_counts.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["GT a\\GT b"] + [f"{h}:{name[h]}" for h in SCOPE])
        for a, ha in enumerate(SCOPE):
            w.writerow([f"{ha}:{name[ha]}"] + [int(co[a, b]) for b in range(len(SCOPE))])
    with open(f"{OUT_DIR}/gt_cofire_jaccard.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["GT a\\GT b"] + [f"{h}:{name[h]}" for h in SCOPE])
        for a, ha in enumerate(SCOPE):
            row = []
            for b in range(len(SCOPE)):
                union = diag[a] + diag[b] - co[a, b]
                row.append(round(co[a, b] / union, 4) if union else 0.0)
            w.writerow([f"{ha}:{name[ha]}"] + row)

    # ── 3. directional P(GT b | GT a) ──────────────────────────────────────
    with open(f"{OUT_DIR}/gt_conditional_directional.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["P(GTcol|GTrow)"] + [f"{h}:{name[h]}" for h in SCOPE])
        for a, ha in enumerate(SCOPE):
            row = [round(co[a, b] / diag[a], 4) if diag[a] else 0.0
                   for b in range(len(SCOPE))]
            w.writerow([f"{ha}:{name[ha]}"] + row)

    # ── 4. GT vs predicted co-fire, per scope pair ─────────────────────────
    if os.path.exists(PROBS_PATH):
        probs = np.load(PROBS_PATH)
        P = (probs[:, SCOPE] >= THR).astype(np.int64)
        copred = P.T @ P
        dpred = np.diag(copred)
        with open(f"{OUT_DIR}/gt_vs_pred_cofire.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["head_a", "name_a", "head_b", "name_b",
                        "gt_cofire_n", "gt_jaccard",
                        "pred_cofire_n", "pred_jaccard", "note"])
            for a, ha in enumerate(SCOPE):
                for b in range(a + 1, len(SCOPE)):
                    hb = SCOPE[b]
                    gco = int(co[a, b])
                    gun = diag[a] + diag[b] - co[a, b]
                    gj = round(gco / gun, 4) if gun else 0.0
                    pco = int(copred[a, b])
                    pun = dpred[a] + dpred[b] - copred[a, b]
                    pj = round(pco / pun, 4) if pun else 0.0
                    if gco == 0 and pco > 0:
                        note = "model co-fires; GT never does"
                    elif gco > 0 and pj > gj * 3:
                        note = "model over-couples vs GT"
                    elif gco > 0:
                        note = "real co-occurrence in GT"
                    else:
                        note = "neither"
                    w.writerow([ha, name[ha], hb, name[hb],
                                gco, gj, pco, pj, note])
        print("wrote GT tables + gt_vs_pred_cofire.csv")
    else:
        print("wrote GT tables (probs npy absent — skipped gt_vs_pred)")


if __name__ == "__main__":
    main()
