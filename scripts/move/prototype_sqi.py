"""Prototype: compute the SQI panel on MOVE raw ECG windows and validate it
against the ground-truth activity label + the independent chest-motion channel.

Implements §6/§9 of docs/SQI_DESIGN.md. Now imports the consolidated, fixed
panel from the repo-root `sqi.py` (single source of truth) rather than
re-implementing metrics inline. Reads the RAW segmented ECG windows + the
synchronized chest-ACC channel, applies correct-stage SQI, and tests:
  H1/H2  how SQI varies by activity phase
  H3     SQI vs the independent gravity-removed chest motion
"""
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import numpy as np
import pandas as pd
from sqi import compute_sqi      # fixed panel: clip-decoupled, gravity-removed motion, dual composite

FS = 500
OUT = "res/move_segmented"


def main():
    df = pd.read_csv(f"{OUT}/move_window_metadata.csv")
    clean = df[df["ecg_pad_fraction"] == 0].reset_index(drop=True)   # exclude padded
    print(f"Windows: {len(df)} total, {len(clean)} non-padded → computing SQI ...")

    rows = []
    for i, r in clean.iterrows():
        sig = np.load(os.path.join(OUT, r["ecg_path"])).astype(np.float64)
        acc = np.load(os.path.join(OUT, r["acc_chest_path"]))   # (3,5000) → motion-fused composite
        s = compute_sqi(sig, fs=FS, acc_xyz=acc)
        s["activity_label"] = r["activity_label"]
        s["subject"] = r["subject"]
        rows.append(s)
        if (i + 1) % 1000 == 0:
            print(f"  {i+1}/{len(clean)}")
    sdf = pd.DataFrame(rows)
    sdf.to_csv(f"{OUT}/move_sqi_prototype.csv", index=False)
    print(f"SQI table → {OUT}/move_sqi_prototype.csv\n")

    # ─── H1/H2: SQI by activity ──────────────────────────────────────────
    order = ["baseline", "lift", "greetings", "gesticulate",
             "walk_before", "run", "walk_after", "unknown"]
    metrics = ["sqi_score_amb","sqi_score_ecg", "snr_proxy", "clip_pct", "flat_pct",
               "hf_noise_ratio", "baseline_drift", "kurt", "rr_cv",
               "pct_physiologic_hr", "mean_motion_mg"]
    g = sdf.groupby("activity_label")[metrics].mean().reindex(
        [o for o in order if o in sdf["activity_label"].unique()])
    print("═══ Mean SQI by activity (H1/H2) ═══")
    print(g.round(3).to_string())

    # ─── H1 explicit verdict ─────────────────────────────────────────────
    base = sdf[sdf.activity_label == "baseline"]
    run  = sdf[sdf.activity_label == "run"]
    print("\n═══ H1 verdict: baseline (rest) vs run (motion) ═══")
    for m in ["sqi_score_amb", "sqi_score_ecg", "snr_proxy", "clip_pct", "flat_pct", "hf_noise_ratio"]:
        b, rn = base[m].mean(), run[m].mean()
        arrow = "↓ worse" if m in ("sqi_score_amb","sqi_score_ecg","snr_proxy") and rn < b else \
                "↑ worse" if m not in ("sqi_score","snr_proxy") and rn > b else "?"
        print(f"  {m:<18} baseline={b:8.3f}  run={rn:8.3f}   {arrow}")

    # ─── H3: SQI vs independent chest motion ─────────────────────────────
    print("\n═══ H3 verdict: correlation of SQI with independent chest_motion_mg ═══")
    sub = sdf.dropna(subset=["mean_motion_mg"])
    for m in ["sqi_score_amb", "snr_proxy", "clip_pct", "hf_noise_ratio", "rr_cv"]:
        c = np.corrcoef(sub["mean_motion_mg"], sub[m].fillna(sub[m].mean()))[0, 1]
        print(f"  corr(mean_motion_mg, {m:<16}) = {c:+.3f}")


if __name__ == "__main__":
    main()
