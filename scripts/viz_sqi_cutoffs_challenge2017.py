"""Per-characteristic cut-off comparison across AFib / Normal / Noisy.

For each SQI characteristic we draw, per cohort, the requested distribution
cut-offs — p10 / p15 / p20 (lower) and p80 / p85 / p90 (upper) — as a graded band
with the median marked. The p10–p90 span is the wide band; inner ticks mark
p15/p20/p80/p85. Shown at S0 (prior to any preprocessing), the diagnostic stage
where group separation drives quality gating.

Output: docs/sqi_cutoffs_challenge2017.png
Run:  python3 -m scripts.viz_sqi_cutoffs_challenge2017
"""
from __future__ import annotations
import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

import sqi
from scripts.eval_sqi_stages_challenge2017 import center_window, WIN0, FS0, GROUPS, DATA, SPLIT

OUT = "docs/sqi_cutoffs_challenge2017.png"
COL = {"A": "#C0392B", "N": "#1C7293", "~": "#7A6F9B"}
CAP, SEED = 500, 42
PCTS = [10, 15, 20, 80, 85, 90]
# (metric, pretty label, log-x?)
METRICS = [
    ("snr_proxy", "SNR proxy  (band/residual var)", True),
    ("hf_noise_ratio", "HF-noise ratio  (>40 Hz / total)", True),
    ("baseline_drift", "Baseline drift  (σ of <0.5 Hz)", True),
    ("dynamic_range_mv", "Dynamic range (raw units)", True),
    ("kurt", "Kurtosis  (QRS peakedness)", False),
    ("rr_cv", "RR-interval CV  (rhythm irregularity)", False),
    ("mean_hr_bpm", "Mean HR (bpm)", False),
    ("sqi_score_ecg", "Composite SQI score", False),
]


def main():
    data = np.load(DATA, allow_pickle=True)
    sp = pd.read_csv(SPLIT)
    vals = {g: {m: [] for m, _, _ in METRICS} for g in GROUPS}
    for g in GROUPS:
        ids = sp[sp.label_str == g].idx.values
        rng = np.random.RandomState(SEED)
        ids = ids if len(ids) <= CAP else rng.choice(ids, CAP, replace=False)
        for rid in ids:
            w = center_window(data[rid], WIN0)
            if np.std(w) < 1e-9:
                continue
            p = sqi.compute_sqi(w, FS0)
            for m, _, _ in METRICS:
                vals[g][m].append(p.get(m, np.nan))

    fig, axes = plt.subplots(2, 4, figsize=(17, 8))
    fig.suptitle("Challenge 2017 — SQI characteristics at S0 (prior to preprocessing): "
                 "p10/15/20–p80/85/90 cut-offs by cohort", fontsize=15, fontweight="bold", y=0.99)
    glist = list(GROUPS)                       # ["A","N","~"]
    ypos = {g: len(glist) - i for i, g in enumerate(glist)}   # A=3 top … ~=1 bottom

    for ax, (m, label, logx) in zip(axes.ravel(), METRICS):
        for g in glist:
            v = np.array(vals[g][m], float); v = v[np.isfinite(v)]
            if logx:
                v = v[v > 0]
            if len(v) == 0:
                continue
            p10, p15, p20, p80, p85, p90 = np.percentile(v, PCTS)
            p50 = np.median(v)
            y = ypos[g]
            # wide band = p10–p90; darker inner band = p20–p80
            ax.plot([p10, p90], [y, y], color=COL[g], lw=6, alpha=0.25, solid_capstyle="round")
            ax.plot([p20, p80], [y, y], color=COL[g], lw=6, alpha=0.45, solid_capstyle="round")
            # cut-off ticks at each requested percentile
            for xp in (p10, p15, p20, p80, p85, p90):
                ax.plot([xp, xp], [y - 0.15, y + 0.15], color=COL[g], lw=1.3)
            ax.plot(p50, y, "o", color=COL[g], ms=10, zorder=3, mec="white", mew=1.2)
            xt = "{:.3g}".format
            ax.text(p10, y + 0.24, xt(p10), color=COL[g], fontsize=7, ha="center")
            ax.text(p90, y + 0.24, xt(p90), color=COL[g], fontsize=7, ha="center")
            ax.text(p50, y - 0.32, xt(p50), color=COL[g], fontsize=8, ha="center", fontweight="bold")
        if logx:
            ax.set_xscale("log")
        ax.set_title(label, fontsize=11, fontweight="bold")
        ax.set_yticks([ypos[g] for g in glist])
        ax.set_yticklabels([GROUPS[g] for g in glist], fontsize=10)
        ax.set_ylim(0.5, 3.6)
        ax.grid(axis="x", color="#EEE", lw=0.6)
        for s in ax.spines.values():
            s.set_color("#DDD")
        ax.tick_params(labelsize=8)

    handles = [plt.Line2D([0], [0], color=COL[g], lw=7, alpha=0.5, label=f"{GROUPS[g]}") for g in glist]
    fig.legend(handles=handles, loc="lower center", ncol=3, fontsize=11, frameon=False,
               bbox_to_anchor=(0.5, -0.02))
    fig.text(0.5, 0.94, "Light band = p10–p90, dark band = p20–p80; ticks = p10/15/20 & p80/85/90 cut-offs; "
             "dot = median. log-x where marked. n: AFib 500, Normal 500, Noisy 279.",
             ha="center", fontsize=9.5, color="#555", style="italic")
    plt.tight_layout(rect=[0, 0.03, 1, 0.93])
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    plt.savefig(OUT, dpi=130, bbox_inches="tight")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
