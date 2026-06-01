"""Co-firing analysis of the base ECGFounder model on PTB-XL (single-lead, t=0.5).

Inputs (precomputed, split-exempt base backbone — not trained on PTB-XL):
  - res/ptbxl_baseline_sqi/base_probs_full.npy  (21799, 150) sigmoid probs
  - csv/ptbxl_label.csv  `label` column        (21799, 150) ground-truth 0/1

Outputs (res/ptbxl_cofiring/):
  - heads_per_segment.csv         distribution of #heads firing per segment
  - cofire_counts_scope.csv       7x7 scope-head co-fire count matrix (pred)
  - cofire_jaccard_scope.csv      7x7 scope-head Jaccard overlap (pred)
  - confusion_per_gt_class.csv    per GT scope class: what predicted heads co-fire
  - top_cofire_pairs.csv          top co-firing head pairs over all 150 heads
  - SUMMARY.md                    human-readable writeup

Run from repo root:  python3 -m scripts.ptbxl_cofiring_analysis
"""
from __future__ import annotations
import ast
import csv
import os
from collections import Counter, defaultdict

import numpy as np

from label_config import load_tasks, DETECTION_SCOPE

PROBS_PATH = "res/ptbxl_baseline_sqi/base_probs_full.npy"
LABEL_CSV = "csv/ptbxl_label.csv"
OUT_DIR = "res/ptbxl_cofiring"
THR = 0.5

SCOPE = sorted(DETECTION_SCOPE)  # [2,4,5,6,93,98,142]


def load_gt(n_expected: int) -> np.ndarray:
    rows = []
    with open(LABEL_CSV) as f:
        r = csv.DictReader(f)
        for rec in r:
            rows.append(ast.literal_eval(rec["label"]))
    gt = np.asarray(rows, dtype=np.int8)
    assert gt.shape[0] == n_expected, f"GT rows {gt.shape[0]} != probs {n_expected}"
    assert gt.shape[1] == 150, f"GT width {gt.shape[1]} != 150"
    return gt


def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    tasks = load_tasks()
    probs = np.load(PROBS_PATH)
    n, k = probs.shape
    assert k == 150
    gt = load_gt(n)

    fired = probs >= THR                       # (n,150) bool — predicted positives
    name = {i: tasks[i] for i in range(150)}

    # ── alignment sanity check vs run.log GT counts ────────────────────────
    # reference (full 150-vector): AFib5=1514 Brady4=637 Tachy6=826 SVT93=42
    # VT98=42 Pause142=0 (NormalECG2=9514, NSR1=0 — both out of scope since 2026-06-01)
    gt_counts = {i: int(gt[:, i].sum()) for i in SCOPE}
    print("GT scope counts:")
    for i in SCOPE:
        print(f"  head {i:>3} {name[i]:<28} n_pos={gt_counts[i]}")

    # ── 1. heads firing per segment (all 150, and scope-only) ──────────────
    per_seg_all = fired.sum(axis=1)
    per_seg_scope = fired[:, SCOPE].sum(axis=1)
    dist_all = Counter(per_seg_all.tolist())
    dist_scope = Counter(per_seg_scope.tolist())
    n_scope = len(SCOPE)
    with open(f"{OUT_DIR}/heads_per_segment.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["n_heads_firing", "n_segments_all150",
                    f"n_segments_scope{n_scope}"])
        for kk in range(0, max(dist_all) + 1):
            w.writerow([kk, dist_all.get(kk, 0), dist_scope.get(kk, 0)])
    print(f"\nheads/segment (all 150): mean={per_seg_all.mean():.2f} "
          f"median={int(np.median(per_seg_all))} max={per_seg_all.max()}")
    print(f"heads/segment (scope {n_scope}): mean={per_seg_scope.mean():.2f} "
          f"median={int(np.median(per_seg_scope))} max={per_seg_scope.max()}")

    # ── 2. scope x scope co-fire count + Jaccard (prediction co-occurrence)─
    S = fired[:, SCOPE].astype(np.int64)                 # (n,7)
    cofire = S.T @ S                                     # (7,7) counts
    with open(f"{OUT_DIR}/cofire_counts_scope.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["head\\head"] + [f"{i}:{name[i]}" for i in SCOPE])
        for a, ia in enumerate(SCOPE):
            w.writerow([f"{ia}:{name[ia]}"] + [int(cofire[a, b]) for b in range(len(SCOPE))])
    # Jaccard = |A∩B| / |A∪B|
    diag = np.diag(cofire)
    with open(f"{OUT_DIR}/cofire_jaccard_scope.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["head\\head"] + [f"{i}:{name[i]}" for i in SCOPE])
        for a, ia in enumerate(SCOPE):
            row = []
            for b, ib in enumerate(SCOPE):
                union = diag[a] + diag[b] - cofire[a, b]
                row.append(round(cofire[a, b] / union, 4) if union else 0.0)
            w.writerow([f"{ia}:{name[ia]}"] + row)

    # ── 3. per GT scope class → predicted co-firing profile ────────────────
    # For records whose GROUND TRUTH includes scope head g, when the model fires
    # g (true positive), what else does it co-fire? Report mean #extra heads and
    # the top-5 most frequent co-fired heads (any of 150).
    with open(f"{OUT_DIR}/confusion_per_gt_class.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["gt_head", "gt_name", "n_gt_pos", "n_tp_fired",
                    "mean_total_heads_when_fired", "mean_extra_heads",
                    "top5_cofired_heads"])
        for g in SCOPE:
            gt_mask = gt[:, g] == 1
            tp_mask = gt_mask & fired[:, g]
            n_gt = int(gt_mask.sum())
            n_tp = int(tp_mask.sum())
            if n_tp == 0:
                w.writerow([g, name[g], n_gt, 0, 0, 0, ""])
                continue
            sub = fired[tp_mask]                          # (n_tp,150)
            total = sub.sum(axis=1)
            co = sub.sum(axis=0)
            co[g] = 0                                     # exclude self
            top = np.argsort(co)[::-1][:5]
            top_str = "; ".join(
                f"{int(i)}:{name[int(i)]}({int(co[int(i)])}, "
                f"{co[int(i)]/n_tp*100:.0f}%)"
                for i in top if co[int(i)] > 0
            )
            w.writerow([g, name[g], n_gt, n_tp,
                        round(total.mean(), 2), round(total.mean() - 1, 2),
                        top_str])

    # ── 4. top co-firing pairs across ALL 150 heads ────────────────────────
    F = fired.astype(np.int64)
    full = F.T @ F                                        # (150,150)
    pairs = []
    for a in range(150):
        for b in range(a + 1, 150):
            c = int(full[a, b])
            if c > 0:
                pairs.append((c, a, b))
    pairs.sort(reverse=True)
    with open(f"{OUT_DIR}/top_cofire_pairs.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["cofire_count", "head_a", "name_a", "head_b", "name_b",
                    "jaccard"])
        for c, a, b in pairs[:50]:
            union = int(full[a, a]) + int(full[b, b]) - c
            w.writerow([c, a, name[a], b, name[b],
                        round(c / union, 4) if union else 0.0])

    print(f"\nwrote outputs to {OUT_DIR}/")


if __name__ == "__main__":
    main()
