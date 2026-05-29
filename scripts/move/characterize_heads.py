"""Comprehensive per-header SQI + motion characterization on MOVE.

For each V3.1 head, against the gravity-removed chest motion and the SQI panel:
  - FP rate (MOVE is all rhythm-negative → every fire is a false positive)
  - firing rate vs MOTION bin (rest → extreme)
  - SQI of windows where the head FIRES vs does NOT fire
  - correlation of P(head) with motion and with key SQIs
Plus a dataset-wide table of how each SQI behaves as motion rises.

Joins: res/move_eval/base_probs.npy + move_segmented metadata + move_sqi_prototype.csv
(all row-aligned over the 5158 non-padded windows).
"""
import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
import numpy as np, pandas as pd
from label_config import FZARK_ONTOLOGY

OUT, EVAL = "res/move_segmented", "res/move_eval"

# unique V3.1 heads (idx → events sharing it)
HEAD_EVENTS = {}
for ev, node in FZARK_ONTOLOGY.items():
    HEAD_EVENTS.setdefault(node.ecgfounder_index, []).append(ev)

MOTION_BINS = [0, 0.5, 1.0, 2.0, 5.0, np.inf]
BIN_LABELS  = ["rest <0.5", "low .5-1", "mod 1-2", "high 2-5", "extreme >5"]
SQIS = ["snr_proxy", "flat_pct", "clip_pct", "hf_noise_ratio", "kurt",
        "baseline_drift", "pct_physiologic_hr", "sqi_score_ecg"]


def main():
    base = np.load(f"{EVAL}/base_probs.npy")
    s = pd.read_csv(f"{OUT}/move_sqi_prototype.csv")
    motion = s["mean_motion_mg"].to_numpy(float)
    mbin = pd.cut(motion, MOTION_BINS, labels=BIN_LABELS, right=False)
    N = len(s)
    print(f"MOVE windows: {N} (all rhythm-negative). Motion bins (n): "
          + ", ".join(f"{l}={int((mbin==l).sum())}" for l in BIN_LABELS))

    # ── Dataset-wide: how each SQI behaves vs motion ─────────────────────────
    print("\n" + "="*92)
    print("A. SQI characteristics vs MOTION (dataset-wide, mean per motion bin)")
    print("="*92)
    tbl = s.assign(mbin=mbin).groupby("mbin", observed=True)[SQIS + ["mean_motion_mg"]].mean()
    print(tbl.reindex(BIN_LABELS).round(3).to_string())

    # ── Per-header characterization ──────────────────────────────────────────
    print("\n" + "="*92)
    print("B. Per-header: FP rate, motion of firing vs not, correlations")
    print("="*92)
    rows = []
    for idx in sorted(HEAD_EVENTS):
        p = base[:, idx]
        fired = p >= 0.5
        ev = "; ".join(HEAD_EVENTS[idx])
        fr = float(fired.mean())
        mot_fire = float(np.nanmean(motion[fired])) if fired.any() else np.nan
        mot_not  = float(np.nanmean(motion[~fired])) if (~fired).any() else np.nan
        # correlations of the continuous probability with motion + key SQIs
        cm = np.corrcoef(p, motion)[0, 1]
        csnr = np.corrcoef(p, s["snr_proxy"].fillna(s["snr_proxy"].mean()))[0, 1]
        cflat = np.corrcoef(p, s["flat_pct"])[0, 1]
        rows.append(dict(idx=idx, event=ev[:38], fp_pct=100*fr,
                         motion_fire=mot_fire, motion_notfire=mot_not,
                         corr_motion=cm, corr_snr=csnr, corr_flat=cflat))
    cdf = pd.DataFrame(rows)
    pd.set_option("display.width", 200)
    print(cdf.round(3).to_string(index=False))

    # ── Per-header FP rate vs MOTION bin (the "against motion" core) ─────────
    print("\n" + "="*92)
    print("C. Per-header FALSE-POSITIVE rate (%) by MOTION bin")
    print("="*92)
    grid = {}
    for idx in sorted(HEAD_EVENTS):
        fired = base[:, idx] >= 0.5
        row = {}
        for l in BIN_LABELS:
            mm = np.asarray(mbin == l)
            row[l] = 100*fired[mm].mean() if mm.sum() else np.nan
        grid[f"{idx} {HEAD_EVENTS[idx][0][:16]}"] = row
    gdf = pd.DataFrame(grid).T[BIN_LABELS]
    print(gdf.round(1).to_string())

    # ── Per-header SQI of FIRING vs NON-FIRING windows ───────────────────────
    print("\n" + "="*92)
    print("D. SQI of FIRING vs non-firing windows (fire | not), per header")
    print("="*92)
    print(f"  {'head':<26}" + "".join(f"{q:>16}" for q in ["snr_proxy","flat_pct","clip_pct","sqi_ecg"]))
    for idx in sorted(HEAD_EVENTS):
        fired = base[:, idx] >= 0.5
        if fired.sum() < 5:
            print(f"  {str(idx)+' '+HEAD_EVENTS[idx][0][:22]:<26}  (n_fire={int(fired.sum())} — too few)")
            continue
        cells = []
        for q in ["snr_proxy", "flat_pct", "clip_pct", "sqi_score_ecg"]:
            f = s[q][fired].mean(); n = s[q][~fired].mean()
            cells.append(f"{f:6.2f}|{n:6.2f}")
        print(f"  {str(idx)+' '+HEAD_EVENTS[idx][0][:22]:<26}" + "".join(f"{c:>16}" for c in cells))

    cdf.to_csv(f"{EVAL}/move_head_characterization.csv", index=False)
    gdf.to_csv(f"{EVAL}/move_head_fp_by_motion.csv")
    print(f"\nArtifacts → {EVAL}/move_head_characterization.csv, move_head_fp_by_motion.csv")


if __name__ == "__main__":
    main()
