"""Visualize one representative recording per cohort (AFib / Normal / Noisy)
across the preprocessing pipeline: S0 raw → S1 notch → S2 bandpass → S3 baseline
→ S4 resample(500 Hz) → S5 winsorize-normalize.

Grid = 6 stages (rows) × 3 cohorts (cols). Each cohort's representative is the
recording whose RAW composite SQI (sqi_score_ecg) is closest to its group median,
so the picks are typical, not cherry-picked. Per-panel annotation shows the stage
SQI composite and SNR.

Output: docs/sqi_stages_challenge2017.png
Run:  python3 -m scripts.viz_sqi_stages_challenge2017
"""
from __future__ import annotations
import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

import sqi
from preprocessing import ECGPreprocessor
from scripts.eval_sqi_stages_challenge2017 import (
    center_window, stage_signals, WIN0, FS0, FS1, GROUPS, STAGES, DATA, SPLIT)

OUT = "docs/sqi_stages_challenge2017.png"
COL = {"A": "#C0392B", "N": "#1C7293", "~": "#7A6F9B"}   # AFib red, Normal teal, Noisy purple-grey
CAP, SEED = 80, 42


def pick_representative(data, ids):
    """Recording whose RAW composite SQI is nearest the group median."""
    rng = np.random.RandomState(SEED)
    sample = ids if len(ids) <= CAP else rng.choice(ids, CAP, replace=False)
    scores = []
    for rid in sample:
        w = center_window(data[rid], WIN0)
        if np.std(w) < 1e-9:
            scores.append((rid, np.nan)); continue
        scores.append((rid, sqi.compute_sqi(w, FS0)["sqi_score_ecg"]))
    arr = np.array([s for _, s in scores], float)
    med = np.nanmedian(arr)
    j = int(np.nanargmin(np.abs(arr - med)))
    return scores[j][0]


def main():
    data = np.load(DATA, allow_pickle=True)
    sp = pd.read_csv(SPLIT)
    prep = ECGPreprocessor(powerline_hz=60, normalize="winsorize")

    picks = {g: pick_representative(data, sp[sp.label_str == g].idx.values) for g in GROUPS}
    print("picked:", {GROUPS[g]: int(picks[g]) for g in GROUPS})

    # precompute stage signals + panels per group
    seq = {}
    for g in GROUPS:
        raw10 = center_window(data[picks[g]], WIN0)
        stages = stage_signals(raw10, prep)
        seq[g] = [(name, x, fs, sqi.compute_sqi(x, fs)) for name, x, fs in stages]

    nrow, ncol = len(STAGES), len(GROUPS)
    fig, ax = plt.subplots(nrow, ncol, figsize=(15, 13))
    fig.suptitle("Challenge 2017 — one representative recording per cohort across preprocessing stages",
                 fontsize=15, fontweight="bold", y=0.995)

    for ci, g in enumerate(GROUPS):
        # column header
        rid = picks[g]
        ax[0, ci].set_title(f"{GROUPS[g]}  (rec {int(rid)})", fontsize=14,
                            fontweight="bold", color=COL[g], pad=22)
        for ri, (name, x, fs, panel) in enumerate(seq[g]):
            a = ax[ri, ci]
            t = np.arange(len(x)) / fs
            a.plot(t, x, color=COL[g], lw=0.6)
            a.set_xlim(0, 10)
            a.margins(y=0.08)
            # annotate stage SQI
            snr = panel.get("snr_proxy", np.nan)
            comp = panel.get("sqi_score_ecg", np.nan)
            a.text(0.985, 0.93, f"SQI {comp:.2f} · SNR {snr:.2f}", transform=a.transAxes,
                   ha="right", va="top", fontsize=8.5, color="#333",
                   bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="#CCC", alpha=0.85))
            # row label (stage) on the left column
            if ci == 0:
                unit = "mV (raw scale)" if ri < 5 else "z (winsorized)"
                a.set_ylabel(f"{name}\n[{unit}]", fontsize=10, fontweight="bold")
            a.tick_params(labelsize=7)
            if ri < nrow - 1:
                a.set_xticklabels([])
            else:
                a.set_xlabel("time (s)", fontsize=9)
            for sp_ in a.spines.values():
                sp_.set_color("#DDD")

    fig.text(0.5, 0.965,
             "Rows = pipeline stages (top→bottom). Notch≈no-op here; band-pass+baseline removal "
             "flatten drift & lift QRS; resample is morphology-neutral; winsorize rescales to z.",
             ha="center", fontsize=9.5, color="#555", style="italic")
    plt.tight_layout(rect=[0, 0, 1, 0.955])
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    plt.savefig(OUT, dpi=130, bbox_inches="tight")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
