"""Stage-by-stage Signal-Quality (SQI) evaluation across the preprocessing pipeline,
comparing AFib / Normal / Noisy cohorts on PhysioNet/CinC Challenge 2017.

We tap the SQI panel (sqi.py) at every stage of the production preprocessing:

  S0 raw          center 10 s raw window @300 Hz (PRIOR to any preprocessing)
  S1 notch        + 60 Hz powerline notch
  S2 bandpass     + 0.67-40 Hz Butterworth band-pass
  S3 baseline     + 0.4 s median baseline-wander removal
  S4 resample     + resample 300 → 500 Hz (= final 5000-sample length)
  S5 normalize    + winsorized z-score normalization (the model's input tensor)

For each stage and each cohort {A=AFib, N=Normal, ~=Noisy} we report every panel
characteristic (amplitude / saturation / flat / baseline-drift / HF-noise / SNR /
kurtosis / rhythm) as distribution **cut-offs at p10 / p15 / p20 · median ·
p80 / p85 / p90**, and the composite sqi_score_ecg.

NOTE (sqi.py stage discipline): amplitude/SNR metrics are physically meaningful
only up to S4; on S5 the signal is unit-variance winsorized, so absolute-scale
metrics (dynamic_range, baseline_drift) become scale-relative — reported for
completeness and flagged.

Writes res/challenge2017/SQI_STAGES.md + SQI_STAGES_cutoffs.csv.
Run:  python3 -m scripts.eval_sqi_stages_challenge2017
"""
from __future__ import annotations
import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import pandas as pd
from scipy.signal import filtfilt, medfilt

import sqi
from preprocessing import (ECGPreprocessor, _notch_coeffs, _bandpass_coeffs, _odd_kernel,
                           DEFAULT_BANDPASS, BASELINE_WINDOW_S)

DATA = "data/challenge2017/Challenge2017/raw_data.npy"
SPLIT = "csv/challenge2017_split.csv"
OUT = "res/challenge2017/SQI_STAGES.md"
FS0, FS1 = 300, 500
WIN0 = 3000          # 10 s @300 Hz
CAP, SEED = 500, 42
GROUPS = {"A": "AFib", "N": "Normal", "~": "Noisy"}
STAGES = ["S0 raw", "S1 notch", "S2 bandpass", "S3 baseline", "S4 resample", "S5 normalize"]
PCTS = [10, 15, 20, 80, 85, 90]                 # requested cut-offs
# panel keys to summarize
KEYS = ["dynamic_range_mv", "flat_pct", "clip_pct", "baseline_drift", "hf_noise_ratio",
        "snr_proxy", "kurt", "n_peaks", "mean_hr_bpm", "rr_cv", "pct_physiologic_hr",
        "sqi_score_ecg"]
PRETTY = {"dynamic_range_mv": "Dynamic range (mV)", "flat_pct": "Flat %", "clip_pct": "Clip %",
          "baseline_drift": "Baseline drift", "hf_noise_ratio": "HF-noise ratio (>40 Hz)",
          "snr_proxy": "SNR proxy", "kurt": "Kurtosis", "n_peaks": "QRS peak count",
          "mean_hr_bpm": "Mean HR (bpm)", "rr_cv": "RR CV", "pct_physiologic_hr": "% physiologic HR",
          "sqi_score_ecg": "SQI composite (ECG)"}


def center_window(sig, n):
    sig = np.asarray(sig, float)
    if len(sig) <= n:
        return np.pad(sig, (0, n - len(sig)))
    s = (len(sig) - n) // 2
    return sig[s:s + n]


def stage_signals(raw10, prep):
    """Return [(stage_name, signal, fs), ...] tapping each preprocessing step."""
    out = [("S0 raw", raw10.copy(), FS0)]
    x = raw10.astype(np.float64)
    b, a = _notch_coeffs(FS0, prep.powerline_hz); x = filtfilt(b, a, x)
    out.append(("S1 notch", x.copy(), FS0))
    b, a = _bandpass_coeffs(FS0, *DEFAULT_BANDPASS); x = filtfilt(b, a, x)
    out.append(("S2 bandpass", x.copy(), FS0))
    k = _odd_kernel(FS0, BASELINE_WINDOW_S); x = x - medfilt(x, kernel_size=k)
    out.append(("S3 baseline", x.copy(), FS0))
    xr = prep._resample(x, FS0)
    out.append(("S4 resample", xr.copy(), FS1))
    xn = prep._normalize(xr)
    out.append(("S5 normalize", xn.copy(), FS1))
    return out


def main():
    data = np.load(DATA, allow_pickle=True)
    sp = pd.read_csv(SPLIT)
    prep = ECGPreprocessor(powerline_hz=60, normalize="winsorize")

    # balanced sample per group (use all of Noisy; cap others)
    idx_by_g = {}
    for g in GROUPS:
        ids = sp[sp.label_str == g].idx.values
        rng = np.random.RandomState(SEED)
        idx_by_g[g] = ids if len(ids) <= CAP else rng.choice(ids, CAP, replace=False)
    print({GROUPS[g]: len(v) for g, v in idx_by_g.items()})

    # records[g][stage] -> list of panels
    rec = {g: {st: [] for st in STAGES} for g in GROUPS}
    for g, ids in idx_by_g.items():
        for rid in ids:
            raw10 = center_window(data[rid], WIN0)
            if np.std(raw10) < 1e-9:
                continue
            for st, x, fs in stage_signals(raw10, prep):
                rec[g][st].append(sqi.compute_sqi(x, fs))

    def agg(panels, key):
        """Return dict(median, n, p10..p90) over finite values, or None."""
        v = np.array([p.get(key, np.nan) for p in panels], float)
        v = v[np.isfinite(v)]
        if len(v) == 0:
            return None
        d = dict(median=float(np.median(v)), n=int(len(v)))
        for q in PCTS:
            d[f"p{q}"] = float(np.percentile(v, q))
        return d

    def cell(a, intfmt=False):
        if a is None:
            return "—"
        f = (lambda x: f"{x:.0f}") if intfmt else (lambda x: f"{x:.3g}")
        return (f"{f(a['p10'])} / {f(a['p15'])} / {f(a['p20'])} · "
                f"**{f(a['median'])}** · {f(a['p80'])} / {f(a['p85'])} / {f(a['p90'])}")

    # ── report ──
    md = ["# Challenge 2017 — stage-by-stage SQI across preprocessing: AFib vs Normal vs Noisy",
          "",
          f"Cohorts (center 10 s window/recording): "
          f"{', '.join(f'{GROUPS[g]} n={len(idx_by_g[g])}' for g in GROUPS)}. "
          "SQI panel (`sqi.py`) tapped at every preprocessing stage. **Each cell is "
          "`p10 / p15 / p20 · median · p80 / p85 / p90`** — the requested distribution "
          "cut-offs. Composite `sqi_score_ecg` ∈ [0,1] (higher = cleaner). Full "
          "machine-readable table: `SQI_STAGES_cutoffs.csv`.",
          ""]

    # 1) HEADLINE: composite SQI score by group × stage
    md += ["## 1. Composite SQI (`sqi_score_ecg`) — group × stage", "",
           "Cells: p10 / p15 / p20 · **median** · p80 / p85 / p90.", "",
           "| stage | " + " | ".join(GROUPS[g] for g in GROUPS) + " |",
           "|---|" + "---|" * len(GROUPS)]
    for st in STAGES:
        md.append(f"| {st} | " + " | ".join(cell(agg(rec[g][st], "sqi_score_ecg")) for g in GROUPS) + " |")
    md.append("")

    # 2) Per-characteristic, one table per metric (all stages, all groups)
    md.append("## 2. Per-characteristic cut-offs — every metric × stage × group\n")
    md.append("Cells: p10 / p15 / p20 · **median** · p80 / p85 / p90.\n")
    for key in KEYS:
        md.append(f"### {PRETTY[key]}  —  `{key}`\n")
        md.append("| stage | AFib | Normal | Noisy |")
        md.append("|---|---|---|---|")
        for st in STAGES:
            cells = [cell(agg(rec[g][st], key), intfmt=(key == "n_peaks")) for g in GROUPS]
            md.append(f"| {st} | " + " | ".join(cells) + " |")
        md.append("")

    # 3) Evolution of key noise/quality metrics across stages (median only)
    md.append("## 3. Metric evolution through the pipeline (medians)\n")
    for key in ["hf_noise_ratio", "snr_proxy", "baseline_drift", "kurt"]:
        md.append(f"### {PRETTY[key]}\n")
        md.append("| group | " + " | ".join(STAGES) + " |")
        md.append("|---|" + "---|" * len(STAGES))
        for g in GROUPS:
            cells = []
            for st in STAGES:
                a = agg(rec[g][st], key)
                cells.append(f"{a['median']:.3g}" if a else "—")
            md.append(f"| {GROUPS[g]} | " + " | ".join(cells) + " |")
        md.append("")

    # 4) separation summary at RAW (prior) — the diagnostic stage
    md.append("## 4. Group separation at S0 (prior to preprocessing)\n")
    md.append("Noisy should be worst; AFib vs Normal should differ mainly in rhythm "
              "irregularity (`rr_cv`), not signal quality. Cells: p10/p15/p20 · "
              "**median** · p80/p85/p90.\n")
    diag = ["hf_noise_ratio", "snr_proxy", "clip_pct", "flat_pct", "rr_cv",
            "pct_physiologic_hr", "sqi_score_ecg"]
    md.append("| metric | AFib | Normal | Noisy | Noisy vs clean |")
    md.append("|---|---|---|---|---|")
    for key in diag:
        a = agg(rec["A"]["S0 raw"], key); n = agg(rec["N"]["S0 raw"], key); no = agg(rec["~"]["S0 raw"], key)
        clean = np.nanmean([a["median"], n["median"]])
        worse = "▲ higher" if no["median"] > clean else "▼ lower"
        md.append(f"| {PRETTY[key]} | {cell(a)} | {cell(n)} | {cell(no)} | {worse} in Noisy |")
    md.append("")

    # 5) sqi_class distribution per group at S0
    md.append("## 5. SQI class distribution at S0 (good ≥0.7 / acceptable ≥0.4 / poor)\n")
    md.append("| group | good | acceptable | poor |")
    md.append("|---|---:|---:|---:|")
    for g in GROUPS:
        cls = [p["sqi_class"] for p in rec[g]["S0 raw"]]
        tot = max(1, len(cls))
        md.append(f"| {GROUPS[g]} | {cls.count('good')/tot:.1%} | "
                  f"{cls.count('acceptable')/tot:.1%} | {cls.count('poor')/tot:.1%} |")
    md.append("")
    md.append("> S5 metrics are on the winsorized z-scored tensor — amplitude-scale "
              "metrics (dynamic_range_mv, baseline_drift) are scale-relative there, not "
              "physical mV; SNR/rhythm/kurtosis remain interpretable.")

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as f:
        f.write("\n".join(md) + "\n")

    # long-form CSV: stage, group, metric, n, p10, p15, p20, median, p80, p85, p90
    csv_rows = []
    for st in STAGES:
        for g in GROUPS:
            for key in KEYS:
                a = agg(rec[g][st], key)
                if a is None:
                    continue
                row = dict(stage=st, group=GROUPS[g], metric=key, n=a["n"],
                           p10=a["p10"], p15=a["p15"], p20=a["p20"], median=a["median"],
                           p80=a["p80"], p85=a["p85"], p90=a["p90"])
                csv_rows.append({k: (round(v, 4) if isinstance(v, float) else v) for k, v in row.items()})
    csv_path = OUT.replace(".md", "_cutoffs.csv")
    pd.DataFrame(csv_rows).to_csv(csv_path, index=False)
    print(f"wrote {OUT}\nwrote {csv_path}  ({len(csv_rows)} rows)")
    # console: composite SQI medians by stage × group
    print("\nComposite SQI (median) by stage × group:")
    for st in STAGES:
        print(f"  {st:<14} " + "  ".join(f"{GROUPS[g]}={agg(rec[g][st],'sqi_score_ecg')['median']:.3f}" for g in GROUPS))


if __name__ == "__main__":
    main()
