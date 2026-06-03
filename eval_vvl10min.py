"""Evaluate the v3.1 single-lead detector on VVL10min Subject 088 (AFib head only).

Scope (decided 2026-06-02):
- AFib-only, 30-s segments. GT comes from `seg_majority` (one label per segment,
  one of: 'Has Afibs', 'Too Noisy', 'No Afibs', 'Others'). Positive = 'Has Afibs',
  negative = 'No Afibs'. 'Too Noisy' and 'Others' are excluded (uninterpretable /
  not a clean non-AFib reference).
- Detector = ScopedDetector (production v3.1 stack: backbone + L1 + L2, SQG ON by default).
- The 6-event detection scope is unchanged; we just read head-5 (AFib) per segment.
- The model takes 10-s windows. Each 30-s GT segment is scored with 3 contiguous
  non-overlapping 10-s windows; segment AFib score = max over windows. Segment
  fires if any window's head-5 Decision.fired is True.
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from preprocessing import ECGPreprocessor
from overlay.inference import ScopedDetector
from label_config import DETECTION_SCOPE

AFIB_HEAD = 5
assert DETECTION_SCOPE[AFIB_HEAD] == "ATRIAL FIBRILLATION", DETECTION_SCOPE[AFIB_HEAD]

DATA_ROOT = ROOT / "data" / "VVL10min" / "blocks_10min"
DEFAULT_SUBJECT_DIR = DATA_ROOT / "88"
LABELS_CSV = DATA_ROOT / "block_labels.csv"

SEG_STATES = ["Has Afibs", "Too Noisy", "No Afibs", "Others"]  # indices 0..3
POSITIVE_STATE_IDX = SEG_STATES.index("Has Afibs")
NEGATIVE_STATE_IDX = SEG_STATES.index("No Afibs")
NOISY_STATE_IDX = SEG_STATES.index("Too Noisy")
OTHERS_STATE_IDX = SEG_STATES.index("Others")
SCORABLE_IDX = (POSITIVE_STATE_IDX, NEGATIVE_STATE_IDX)


def split_into_segments(signal: np.ndarray, fs: int, seg_s: float = 30.0) -> list[np.ndarray]:
    n = int(round(seg_s * fs))
    return [signal[i * n:(i + 1) * n] for i in range(len(signal) // n)]


def split_into_windows(seg: np.ndarray, fs: int, win_s: float = 10.0) -> list[np.ndarray]:
    n = int(round(win_s * fs))
    return [seg[i * n:(i + 1) * n] for i in range(len(seg) // n)]


def score_block(detector: ScopedDetector, prep: ECGPreprocessor,
                block_path: Path) -> list[dict]:
    d = np.load(block_path, allow_pickle=True)
    sig = d["signal"].astype(np.float32)
    fs = int(d["fs"])
    seg_majority = d["seg_majority"].astype(int)
    segs = split_into_segments(sig, fs, seg_s=30.0)
    assert len(segs) == len(seg_majority), (len(segs), len(seg_majority), block_path)
    rows = []
    for s_idx, (seg, gt_idx) in enumerate(zip(segs, seg_majority)):
        windows = split_into_windows(seg, fs, win_s=10.0)
        win_probs, win_fired, win_gated = [], [], 0
        for w in windows:
            x = prep.process(w, fs_in=fs)               # (1, 5000) float32
            decisions = detector.detect(x)
            if decisions is None:
                win_gated += 1
                continue
            dec5 = decisions.get(AFIB_HEAD)
            if dec5 is None:
                continue
            win_probs.append(float(dec5.score))
            win_fired.append(bool(dec5.fired))
        seg_prob = max(win_probs) if win_probs else float("nan")
        seg_fire = any(win_fired) if win_fired else False
        rows.append(dict(
            block=block_path.name, seg_idx=s_idx,
            gt_state=SEG_STATES[gt_idx], gt_idx=int(gt_idx),
            n_windows=len(windows), n_gated=win_gated,
            seg_prob=seg_prob, seg_fire=seg_fire,
        ))
    return rows


def confusion(rows: list[dict]) -> dict:
    keep = [r for r in rows if r["gt_idx"] in SCORABLE_IDX
            and not (isinstance(r["seg_prob"], float) and np.isnan(r["seg_prob"]))]
    tp = sum(1 for r in keep if r["gt_idx"] == POSITIVE_STATE_IDX and r["seg_fire"])
    fn = sum(1 for r in keep if r["gt_idx"] == POSITIVE_STATE_IDX and not r["seg_fire"])
    fp = sum(1 for r in keep if r["gt_idx"] == NEGATIVE_STATE_IDX and r["seg_fire"])
    tn = sum(1 for r in keep if r["gt_idx"] == NEGATIVE_STATE_IDX and not r["seg_fire"])
    sens = tp / (tp + fn) if (tp + fn) else float("nan")
    spec = tn / (tn + fp) if (tn + fp) else float("nan")
    ppv = tp / (tp + fp) if (tp + fp) else float("nan")
    npv = tn / (tn + fn) if (tn + fn) else float("nan")
    f1 = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else float("nan")
    n_drop_noisy = sum(1 for r in rows if r["gt_idx"] == NOISY_STATE_IDX)
    n_drop_others = sum(1 for r in rows if r["gt_idx"] == OTHERS_STATE_IDX)
    n_drop_gated = sum(1 for r in rows if r["gt_idx"] in SCORABLE_IDX
                       and isinstance(r["seg_prob"], float) and np.isnan(r["seg_prob"]))
    return dict(tp=tp, fn=fn, fp=fp, tn=tn, n_scored=len(keep),
                n_drop_noisy=n_drop_noisy, n_drop_others=n_drop_others,
                n_drop_gated=n_drop_gated,
                sens=sens, spec=spec, ppv=ppv, npv=npv, f1=f1)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--subject-dir", default=str(DEFAULT_SUBJECT_DIR))
    p.add_argument("--out", default=str(ROOT / "res" / "vvl10min" / "subject88_afib_seg_eval.csv"))
    p.add_argument("--limit-blocks", type=int, default=0,
                   help="0 = all blocks; otherwise score the first N blocks (smoke run)")
    args = p.parse_args()

    sub_dir = Path(args.subject_dir)
    out_csv = Path(args.out); out_csv.parent.mkdir(parents=True, exist_ok=True)

    prep = ECGPreprocessor.for_fzark()
    detector = ScopedDetector()              # default v3.1 stack: backbone + L1 + L2, SQG ON

    block_files = sorted(sub_dir.glob("block_*.npz"))
    if args.limit_blocks:
        block_files = block_files[: args.limit_blocks]
    print(f"[eval] subject={sub_dir.name}  blocks={len(block_files)}  out={out_csv}")

    all_rows = []
    for i, bp in enumerate(block_files, 1):
        rows = score_block(detector, prep, bp)
        all_rows.extend(rows)
        print(f"  [{i:>2}/{len(block_files)}] {bp.name}  segs={len(rows)}", flush=True)

    with out_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
        w.writeheader(); w.writerows(all_rows)

    m = confusion(all_rows)
    print()
    print(f"=== VVL10min Subject {sub_dir.name} — AFib head-5 @ 30-s segment ===")
    print(f"  total segments      : {len(all_rows)}")
    print(f"  dropped (Too Noisy) : {m['n_drop_noisy']}")
    print(f"  dropped (Others)    : {m['n_drop_others']}")
    print(f"  dropped (SQG gate)  : {m['n_drop_gated']}")
    print(f"  scored              : {m['n_scored']}")
    print(f"  confusion           : TP={m['tp']} FN={m['fn']} FP={m['fp']} TN={m['tn']}")
    print(f"  sens (recall)       : {m['sens']:.3f}")
    print(f"  spec                : {m['spec']:.3f}")
    print(f"  PPV (precision)     : {m['ppv']:.3f}")
    print(f"  NPV                 : {m['npv']:.3f}")
    print(f"  F1                  : {m['f1']:.3f}")
    print(f"  csv saved to {out_csv}")


if __name__ == "__main__":
    main()
