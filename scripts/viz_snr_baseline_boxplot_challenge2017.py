"""Deep drill — S0 SNR (in dB) and baseline drift: boxplot separation of
AFib / Normal / Noisy on Challenge 2017.

SNR proxy (sqi.compute_band_sqi), computed on the center 10 s RAW window (S0):

    bp        = bandpass(sig, 5–25 Hz)            # QRS / in-band ECG energy
    residual  = sig − bp                          # out-of-band (noise) energy
    snr_proxy = var(bp) / var(residual)           # linear power ratio
    SNR_dB    = 10 · log10(snr_proxy)             # ← decibel form (power ratio)

Baseline drift (sqi.compute_raw_sqi):

    lp             = lowpass(sig, <0.5 Hz)
    baseline_drift = std(lp)                       # mV of sub-0.5 Hz wander

All records per label (no model → no split): AFib 758, Normal 5076, Noisy 279.

Outputs:
  docs/snr_baseline_boxplot_challenge2017.png  — side-by-side boxplots + separation
  res/challenge2017/SNR_BASELINE_DRILL.md      — formula, dB cut-offs, separation stats
Run:  python3 -m scripts.viz_snr_baseline_boxplot_challenge2017
"""
from __future__ import annotations
import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt
from sklearn.metrics import roc_auc_score

from scripts.eval_sqi_stages_challenge2017 import center_window, WIN0, FS0, GROUPS, DATA, SPLIT

OUT_PNG = "docs/snr_baseline_boxplot_challenge2017.png"
OUT_MD = "res/challenge2017/SNR_BASELINE_DRILL.md"
COL = {"A": "#C0392B", "N": "#1C7293", "~": "#7A6F9B"}
PCTS = [5, 10, 15, 20, 80, 85, 90, 95]


def snr_proxy(sig, fs):
    b, a = butter(2, [5.0 / (fs / 2), 25.0 / (fs / 2)], btype="bandpass")
    bp = filtfilt(b, a, sig)
    return float(np.var(bp) / (np.var(sig - bp) + 1e-9))


def baseline_drift(sig, fs):
    b, a = butter(2, 0.5 / (fs / 2), btype="low")
    return float(np.std(filtfilt(b, a, sig)))


def cohen_d(x, y):
    nx, ny = len(x), len(y)
    sp = np.sqrt(((nx - 1) * np.var(x, ddof=1) + (ny - 1) * np.var(y, ddof=1)) / (nx + ny - 2))
    return (np.mean(x) - np.mean(y)) / (sp + 1e-12)


def main():
    data = np.load(DATA, allow_pickle=True)
    sp = pd.read_csv(SPLIT)
    snr_db = {g: [] for g in GROUPS}     # 10*log10(snr_proxy)
    drift = {g: [] for g in GROUPS}      # mV
    for g in GROUPS:
        for rid in sp[sp.label_str == g].idx.values:
            w = center_window(data[rid], WIN0) / 1000.0     # ADC → mV
            if np.std(w) < 1e-9:
                continue
            snr_db[g].append(10.0 * np.log10(max(snr_proxy(w, FS0), 1e-12)))
            drift[g].append(baseline_drift(w, FS0) * 1000.0)     # mV → µV
    for g in GROUPS:
        snr_db[g] = np.array(snr_db[g]); drift[g] = np.array(drift[g])
    n = {g: len(snr_db[g]) for g in GROUPS}
    print("n:", {GROUPS[g]: n[g] for g in GROUPS})

    # ── separation stats ──
    def auc(metric, pos, neg, higher_pos=True):
        x = np.r_[metric[pos], metric[neg]]
        y = np.r_[np.ones(len(metric[pos])), np.zeros(len(metric[neg]))]
        return roc_auc_score(y, x if higher_pos else -x)
    # Noisy is LOW snr / HIGH drift → orient so "noisy=positive"
    clean_snr = np.r_[snr_db["A"], snr_db["N"]]
    clean_drift = np.r_[drift["A"], drift["N"]]
    auc_snr_noisy = roc_auc_score(np.r_[np.ones(n["~"]), np.zeros(len(clean_snr))],
                                  np.r_[-snr_db["~"], -clean_snr])
    auc_drift_noisy = roc_auc_score(np.r_[np.ones(n["~"]), np.zeros(len(clean_drift))],
                                    np.r_[drift["~"], clean_drift])

    # ── boxplot figure ──
    glist = list(GROUPS)
    fig, (axA, axB) = plt.subplots(1, 2, figsize=(13.5, 6.2))
    fig.suptitle("Challenge 2017 — S0 (raw) SNR and baseline drift: AFib vs Normal vs Noisy",
                 fontsize=15, fontweight="bold", y=0.98)

    def box(ax, vals, ylabel, logy=False):
        bp = ax.boxplot([vals[g] for g in glist], notch=True, widths=0.55, patch_artist=True,
                        showfliers=False, medianprops=dict(color="black", lw=2))
        for patch, g in zip(bp["boxes"], glist):
            patch.set_facecolor(COL[g]); patch.set_alpha(0.55)
        ax.set_xticklabels([f"{GROUPS[g]}\n(n={n[g]})" for g in glist], fontsize=11)
        ax.set_ylabel(ylabel, fontsize=12)
        if logy:
            ax.set_yscale("log")
        ax.grid(axis="y", color="#EEE", lw=0.6)
        for s in ax.spines.values():
            s.set_color("#CCC")
        # annotate medians
        for i, g in enumerate(glist, start=1):
            m = np.median(vals[g])
            ax.text(i + 0.32, m, f"{m:.2f}", color=COL[g], fontsize=9, fontweight="bold", va="center")
        return bp

    box(axA, snr_db, "SNR  (dB) = 10·log₁₀(snr_proxy)")
    axA.set_title(f"SNR in dB    — Noisy-vs-clean AUC = {auc_snr_noisy:.3f}", fontsize=12, fontweight="bold")
    axA.axhline(0, color="#999", lw=0.8, ls="--")

    box(axB, drift, "Baseline drift  (µV, σ of <0.5 Hz)", logy=True)
    axB.set_title(f"Baseline drift — Noisy-vs-clean AUC = {auc_drift_noisy:.3f}", fontsize=12, fontweight="bold")

    fig.text(0.5, 0.005, "Notched box = median ±95% CI, IQR box, whiskers 1.5×IQR (outliers hidden). "
             "Noisy: low SNR / high drift.", ha="center", fontsize=9.5, color="#555", style="italic")
    plt.tight_layout(rect=[0, 0.03, 1, 0.94])
    os.makedirs(os.path.dirname(OUT_PNG), exist_ok=True)
    plt.savefig(OUT_PNG, dpi=140, bbox_inches="tight")
    print(f"wrote {OUT_PNG}")

    # ── markdown drill ──
    def stat(v):
        d = {f"p{q}": float(np.percentile(v, q)) for q in PCTS}
        d["median"] = float(np.median(v))
        return d
    md = ["# S0 deep drill — SNR (dB) and baseline drift across AFib / Normal / Noisy", "",
          "## SNR proxy → decibels", "",
          "```", "bp        = bandpass(sig, 5–25 Hz)        # in-band ECG (QRS) energy",
          "residual  = sig − bp                      # out-of-band (noise) energy",
          "snr_proxy = var(bp) / var(residual)       # linear power ratio",
          "SNR_dB    = 10 · log10(snr_proxy)         # decibel (power) form", "```", "",
          "Because `snr_proxy` is a **power** (variance) ratio, the decibel conversion uses "
          "the 10·log10 form (not 20·log10). SNR_dB = 0 dB ⇔ equal in-band and out-of-band "
          "power; >0 dB ⇔ ECG dominates; <0 dB ⇔ noise dominates.", "",
          f"Baseline drift = `std(lowpass<0.5 Hz of raw)` in µV (raw ADC÷1000→mV, ×1000→µV). n: "
          f"AFib {n['A']}, Normal {n['N']}, Noisy {n['~']}.", "",
          "## SNR in dB — cut-offs (p5 / p10 / p15 / p20 · median · p80 / p85 / p90 / p95)", "",
          "| group | p5 | p10 | p15 | p20 | median | p80 | p85 | p90 | p95 |",
          "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for g in glist:
        s = stat(snr_db[g])
        md.append(f"| {GROUPS[g]} | {s['p5']:.2f} | {s['p10']:.2f} | {s['p15']:.2f} | {s['p20']:.2f} | "
                  f"**{s['median']:.2f}** | {s['p80']:.2f} | {s['p85']:.2f} | {s['p90']:.2f} | {s['p95']:.2f} |")
    md += ["", "_(values in dB)_", "",
           "## Baseline drift (µV) — cut-offs", "",
           "| group | p5 | p10 | p15 | p20 | median | p80 | p85 | p90 | p95 |",
           "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for g in glist:
        s = stat(drift[g])
        md.append(f"| {GROUPS[g]} | {s['p5']:.1f} | {s['p10']:.1f} | {s['p15']:.1f} | {s['p20']:.1f} | "
                  f"**{s['median']:.1f}** | {s['p80']:.1f} | {s['p85']:.1f} | {s['p90']:.1f} | {s['p95']:.1f} |")
    md += ["", "## Separation (boxplot read-out)", "",
           "| comparison | metric | AUC | Cohen's d |", "|---|---|---:|---:|",
           f"| Noisy vs clean | SNR (dB) | {auc_snr_noisy:.3f} | {cohen_d(snr_db['~'], clean_snr):+.2f} |",
           f"| Noisy vs clean | baseline drift | {auc_drift_noisy:.3f} | {cohen_d(drift['~'], clean_drift):+.2f} |",
           f"| AFib vs Normal | SNR (dB) | {roc_auc_score(np.r_[np.ones(n['A']),np.zeros(n['N'])], np.r_[snr_db['A'],snr_db['N']]):.3f} | {cohen_d(snr_db['A'], snr_db['N']):+.2f} |",
           f"| AFib vs Normal | baseline drift | {roc_auc_score(np.r_[np.ones(n['A']),np.zeros(n['N'])], np.r_[drift['A'],drift['N']]):.3f} | {cohen_d(drift['A'], drift['N']):+.2f} |",
           "",
           "**Read:** SNR (dB) and baseline drift both separate **Noisy from clean** well "
           "(AUC ≈ 0.83–0.84) and in opposite directions (Noisy = low SNR, high drift). "
           "Neither separates **AFib from Normal** (AUC ≈ 0.5–0.6) — as intended, both are "
           "clean. AFib sits slightly *higher* in SNR than Normal here, so a Noisy gate on "
           "these features is gentle on the AFib target.", "",
           f"Figure: `{OUT_PNG}`."]
    with open(OUT_MD, "w") as f:
        f.write("\n".join(md) + "\n")
    print(f"wrote {OUT_MD}")
    print(f"\nSNR(dB) medians:  " + "  ".join(f"{GROUPS[g]}={np.median(snr_db[g]):.2f}" for g in glist))
    print(f"drift(mV) medians: " + "  ".join(f"{GROUPS[g]}={np.median(drift[g]):.1f}" for g in glist))


if __name__ == "__main__":
    main()
