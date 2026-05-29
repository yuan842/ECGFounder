"""Step-by-step SQI through the preprocessing pipeline, compared across activities.

Empirical version of the stage-tapping rule (docs/SQI_DESIGN.md §2): compute the
SQI panel at EACH preprocessing stage and show how every metric evolves, broken
down by MOVE activity phase. Demonstrates exactly which stage masks/rescues which
quality dimension — and therefore why each SQI must be tapped at a specific stage.

Pipeline stages replicated from preprocessing.ECGPreprocessor.process (MOVE: EU
50 Hz, native 500 Hz so resample + crop are no-ops):
  S0 raw          → the lead-selected window, untouched
  S1 +notch       → 50 Hz iirnotch (Q=30)
  S2 +bandpass    → 0.67–40 Hz Butterworth (order 4)
  S3 +baseline    → minus median-filtered baseline (0.4 s kernel)
  S4 +normalize   → winsorize [1.5, 98.5] pctl + z-score   (the model input)

Channel analyzed: ecg:gel.
"""
import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
import numpy as np, pandas as pd
from scipy.signal import filtfilt, iirnotch, butter, medfilt, welch
from scipy.stats import kurtosis

from preprocessing import (NOTCH_Q, DEFAULT_BANDPASS, BASELINE_WINDOW_S, WIN_LIMITS)

OUT = "res/move_segmented"
FS = 500
PER_ACTIVITY = 200          # sample for speed; means are stable
ACTS = ["baseline", "walk_before", "run", "walk_after"]   # rest→motion gradient


# ─── stage operators (faithful to preprocessing.py) ─────────────────────────
def stage_signals(raw):
    s0 = raw.astype(np.float64)
    bn, an = iirnotch(50, NOTCH_Q, FS)
    s1 = filtfilt(bn, an, s0)
    bb, ab = butter(4, [DEFAULT_BANDPASS[0]/(FS/2), DEFAULT_BANDPASS[1]/(FS/2)], btype="bandpass")
    s2 = filtfilt(bb, ab, s1)
    k = int(BASELINE_WINDOW_S * FS);  k += (k % 2 == 0)
    s3 = s2 - medfilt(s2, kernel_size=k)
    lo, hi = np.percentile(s3, WIN_LIMITS)
    s4 = np.clip(s3, lo, hi)
    s4 = (s4 - s4.mean()) / (s4.std() + 1e-8)
    return {"S0 raw": s0, "S1 notch": s1, "S2 bandpass": s2, "S3 baseline": s3, "S4 normalize": s4}


# ─── a compact SQI panel (cheap; computed at every stage) ───────────────────
def sqi(sig):
    rng = float(sig.max() - sig.min())
    w = max(1, int(0.1 * FS))
    roll = np.lib.stride_tricks.sliding_window_view(sig, w).std(axis=1)
    flat = float(np.mean(roll < 1e-4) * 100)
    # rail-based, flat-gated clip (same logic as sqi.py)
    if rng < 0.05:
        clip = 0.0
    else:
        tol = 0.005 * rng
        at = (sig >= sig.max()-tol) | (sig <= sig.min()+tol)
        clip = float(np.mean(at) * 100)
    try:
        bb, ab = butter(2, [5/(FS/2), 25/(FS/2)], btype="bandpass")
        bp = filtfilt(bb, ab, sig); snr = float(np.var(bp)/(np.var(sig-bp)+1e-9))
    except Exception:
        snr = np.nan
    try:
        bl, al = butter(2, 0.5/(FS/2), btype="low"); bd = float(np.std(filtfilt(bl, al, sig)))
    except Exception:
        bd = np.nan
    try:
        f, p = welch(sig, fs=FS, nperseg=min(1024, len(sig))); hf = float(p[f>40].sum()/(p.sum()+1e-12))
    except Exception:
        hf = np.nan
    kt = float(kurtosis(sig, fisher=True)) if np.std(sig) > 1e-6 else 0.0
    return dict(dyn_range=rng, flat_pct=flat, clip_pct=clip, snr_proxy=snr,
                baseline_drift=bd, hf_noise=hf, kurt=kt)


def main():
    meta = pd.read_csv(f"{OUT}/move_window_metadata.csv")
    meta = meta[meta.ecg_pad_fraction == 0]
    rng = np.random.default_rng(0)
    rows = []
    for a in ACTS:
        sub = meta[meta.activity_label == a]
        take = sub.sample(n=min(PER_ACTIVITY, len(sub)), random_state=0)
        for _, r in take.iterrows():
            raw = np.load(os.path.join(OUT, r["ecg_path"]))
            for stage, sig in stage_signals(raw).items():
                d = sqi(sig); d["stage"] = stage; d["activity"] = a
                rows.append(d)
    df = pd.DataFrame(rows)
    df.to_csv("res/move_eval/move_sqi_by_stage.csv", index=False)

    stages = ["S0 raw", "S1 notch", "S2 bandpass", "S3 baseline", "S4 normalize"]
    metrics = ["clip_pct", "flat_pct", "baseline_drift", "hf_noise", "snr_proxy", "kurt", "dyn_range"]
    print(f"Sampled {len(meta[meta.activity_label.isin(ACTS)])}→{PER_ACTIVITY}/activity; "
          f"stages × activities means.\n")
    for mname in metrics:
        piv = df.pivot_table(index="stage", columns="activity", values=mname, aggfunc="mean").reindex(stages)[ACTS]
        print(f"═══ {mname}  (stage ↓ × activity →) ═══")
        print(piv.round(3).to_string())
        print()


if __name__ == "__main__":
    main()
