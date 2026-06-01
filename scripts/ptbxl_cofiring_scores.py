"""Score distributions: a scope head as MAIN fire vs as CO-FIRE (PTB-XL, t=0.5).

For each segment, among the fired SCOPE heads (prob >= 0.5) the highest-scoring
one is the MAIN fire; every other fired scope head is a CO-FIRE (a secondary
detection riding along under a stronger head). Question answered:

  When a head fires as a co-fire, how confident is it (score distribution),
  versus when the same head is the main fire?

and the directional pairwise version:

  When head A is the main fire and head B co-fires, what is B's score
  distribution — i.e. are B's ride-along firings weak or strong?

Inputs (split-exempt base backbone):
  res/ptbxl_baseline_sqi/base_probs_full.npy  (21799,150)
Outputs (res/ptbxl_cofiring/):
  scope_score_main_vs_cofire.csv   per head: score quantiles as main vs co-fire
  scope_pair_cofire_scores.csv     ordered (main A, cofire B): B's score quantiles

Run:  python3 -m scripts.ptbxl_cofiring_scores
"""
from __future__ import annotations
import csv
import os

import numpy as np

from label_config import load_tasks, DETECTION_SCOPE

PROBS_PATH = "res/ptbxl_baseline_sqi/base_probs_full.npy"
OUT_DIR = "res/ptbxl_cofiring"
THR = 0.5
SCOPE = sorted(DETECTION_SCOPE)
PCTS = [0, 5, 10, 25, 50, 75, 90, 100]


def qrow(vals: np.ndarray) -> list:
    if vals.size == 0:
        return ["nan"] * (len(PCTS) + 1)
    q = np.percentile(vals, PCTS)
    return [round(float(vals.mean()), 4)] + [round(float(x), 4) for x in q]


def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    tasks = load_tasks()
    name = {i: tasks[i] for i in SCOPE}
    probs = np.load(PROBS_PATH)
    n = probs.shape[0]

    P = probs[:, SCOPE]                         # (n,7) scope probs
    fired = P >= THR                            # (n,7)
    idx_of = {h: j for j, h in enumerate(SCOPE)}

    # main scope head per segment = argmax prob AMONG fired scope heads.
    # masked prob: -1 where not fired so argmax picks a fired head; segments
    # with no fired scope head have no main (handled by any_fired mask).
    masked = np.where(fired, P, -1.0)
    any_fired = fired.any(axis=1)
    main_j = masked.argmax(axis=1)              # valid only where any_fired

    qhdr = ["mean"] + [f"p{p}" for p in PCTS]

    # ── per-head: main vs co-fire score distribution ───────────────────────
    with open(f"{OUT_DIR}/scope_score_main_vs_cofire.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["head", "name", "role", "n"] + qhdr)
        for h in SCOPE:
            j = idx_of[h]
            is_main = any_fired & (main_j == j) & fired[:, j]
            is_cofire = fired[:, j] & ~is_main
            for role, mask in (("main", is_main), ("cofire", is_cofire)):
                vals = P[mask, j]
                w.writerow([h, name[h], role, int(mask.sum())] + qrow(vals))

    # ── directional pairwise: A main, B co-fires → B's score distribution ──
    with open(f"{OUT_DIR}/scope_pair_cofire_scores.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["main_head", "main_name", "cofire_head", "cofire_name",
                    "n"] + qhdr + ["cofire_head_solo_score_median_for_ref"])
        # reference: B's score when B is itself the main fire (median)
        ref_median = {}
        for h in SCOPE:
            j = idx_of[h]
            is_main = any_fired & (main_j == j) & fired[:, j]
            vals = P[is_main, j]
            ref_median[h] = round(float(np.median(vals)), 4) if vals.size else float("nan")

        for a in SCOPE:
            ja = idx_of[a]
            a_is_main = any_fired & (main_j == ja) & fired[:, ja]
            for b in SCOPE:
                if b == a:
                    continue
                jb = idx_of[b]
                mask = a_is_main & fired[:, jb]      # A main AND B co-fires
                vals = P[mask, jb]
                if vals.size == 0:
                    continue
                w.writerow([a, name[a], b, name[b], int(vals.size)]
                           + qrow(vals) + [ref_median[b]])

    print(f"wrote score-distribution tables to {OUT_DIR}/")


if __name__ == "__main__":
    main()
