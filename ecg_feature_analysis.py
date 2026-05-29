#!/usr/bin/env python3
"""
ecg_feature_analysis.py

Evaluate non-motion ECG features on AFib TP vs FP cohorts:
  1. RR-interval irregularity   (RMSSD, pNN50, sample entropy, CV-RR)
  2. P-wave detection score      (PR-segment morphology score)
  3. Signal-quality index        (flat-line %, baseline drift, kurtosis, SNR proxy)
  4. Multi-window persistence    (consecutive-window classification stability)

Workflow:
  - Walk through each JSON record's `data.ecg` lists (128 Hz)
  - Concatenate into a continuous strip
  - Detect R-peaks via Pan-Tompkins-like filter
  - Derive features per event, then aggregate per cohort
"""
import os, json, glob, math
import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt, find_peaks
from scipy.stats import kurtosis

FS = 128  # sampling frequency from JSON

# ──────────────────────────────────────────────────────────────────────────
# ECG loading
# ──────────────────────────────────────────────────────────────────────────
def load_ecg_strip(json_path):
    """Concatenate all EcgRaw segments into one continuous signal (mV-ish)."""
    try:
        with open(json_path) as f:
            data = json.load(f)
    except Exception:
        return None, None
    if not isinstance(data, list):
        return None, None
    chunks = []
    sf = FS
    mag = 1000
    for item in data:
        if not isinstance(item, dict): continue
        d = item.get('data', {})
        ecg = d.get('ecg')
        if ecg is None: continue
        sf = d.get('sf', sf) or sf
        mag = d.get('magnification', mag) or mag
        chunks.extend(ecg)
    if not chunks: return None, None
    sig = np.asarray(chunks, dtype=float) / float(mag)  # normalize by magnification
    return sig, sf

# ──────────────────────────────────────────────────────────────────────────
# R-peak detection (Pan-Tompkins-style)
# ──────────────────────────────────────────────────────────────────────────
def bandpass(sig, fs, lo=5.0, hi=20.0, order=2):
    nyq = fs / 2.0
    b, a = butter(order, [lo/nyq, hi/nyq], btype='band')
    return filtfilt(b, a, sig)

def detect_r_peaks(sig, fs):
    """Returns sample indices of R-peaks."""
    if len(sig) < fs * 2: return np.array([])
    # Bandpass + derivative + square + integrate
    bp = bandpass(sig, fs)
    deriv = np.diff(bp, prepend=bp[0])
    sq = deriv ** 2
    win = max(1, int(0.150 * fs))  # 150 ms integration window
    integ = np.convolve(sq, np.ones(win)/win, mode='same')
    # Adaptive threshold
    thresh = 0.4 * np.percentile(integ, 99)
    min_dist = int(0.25 * fs)  # 250 ms refractory → max 240 bpm
    peaks, _ = find_peaks(integ, height=thresh, distance=min_dist)
    # Refine to local max of original signal within ±50 ms
    win_r = int(0.05 * fs)
    refined = []
    for p in peaks:
        a = max(0, p - win_r); b = min(len(sig), p + win_r + 1)
        if b > a:
            refined.append(a + int(np.argmax(sig[a:b])))
    return np.asarray(refined, dtype=int)

# ──────────────────────────────────────────────────────────────────────────
# Feature 1: RR-interval irregularity
# ──────────────────────────────────────────────────────────────────────────
def sample_entropy(x, m=2, r=None):
    """Sample entropy of a 1-D sequence."""
    x = np.asarray(x, dtype=float)
    N = len(x)
    if N < m + 2: return np.nan
    if r is None: r = 0.2 * np.std(x)
    if r <= 0: return np.nan
    def _phi(mm):
        templates = np.array([x[i:i+mm] for i in range(N - mm + 1)])
        C = 0
        for i in range(len(templates)):
            d = np.max(np.abs(templates - templates[i]), axis=1)
            C += np.sum(d <= r) - 1  # exclude self-match
        return C
    A = _phi(m + 1)
    B = _phi(m)
    if B == 0 or A == 0: return np.nan
    return -math.log(A / B)

def rr_irregularity_features(rr_ms):
    """rr_ms = array of RR intervals in ms."""
    rr = np.asarray(rr_ms, dtype=float)
    if len(rr) < 3:
        return dict(n_rr=len(rr), mean_rr=np.nan, std_rr=np.nan, cv_rr=np.nan,
                    rmssd=np.nan, pnn50=np.nan, samp_en=np.nan)
    diff = np.diff(rr)
    rmssd = float(np.sqrt(np.mean(diff**2)))
    pnn50 = float(np.mean(np.abs(diff) > 50.0) * 100.0)
    mean_rr = float(np.mean(rr)); std_rr = float(np.std(rr))
    cv_rr = std_rr / mean_rr if mean_rr > 0 else np.nan
    se = sample_entropy(rr, m=2)
    return dict(n_rr=len(rr), mean_rr=mean_rr, std_rr=std_rr, cv_rr=cv_rr,
                rmssd=rmssd, pnn50=pnn50, samp_en=se)

# ──────────────────────────────────────────────────────────────────────────
# Feature 2: P-wave detection score
# ──────────────────────────────────────────────────────────────────────────
def pwave_features(sig, fs, r_peaks):
    """
    Locate the PR-window (60–200 ms before each R-peak) and quantify:
      - p_amp_mean: mean abs amplitude of largest peak in PR-window
      - p_present_pct: fraction of beats with a clear positive deflection > local baseline
      - p_consistency: 1 - std(p_amp)/mean(p_amp)
    AFib should show LOW p_present_pct and LOW p_consistency.
    """
    if len(r_peaks) < 3 or len(sig) == 0:
        return dict(p_amp_mean=np.nan, p_present_pct=np.nan, p_consistency=np.nan)
    pr_start = int(0.20 * fs)   # 200 ms before R
    pr_end   = int(0.06 * fs)   # 60 ms before R
    amps = []; present_flags = []
    for r in r_peaks:
        s = r - pr_start; e = r - pr_end
        if s < 0 or e <= s: continue
        win = sig[s:e]
        baseline = np.median(win)
        peak_amp = float(np.max(win) - baseline)
        amps.append(peak_amp)
        # baseline noise ~ MAD of the win
        mad = float(np.median(np.abs(win - baseline)) + 1e-9)
        present_flags.append(peak_amp > 3.0 * mad)
    if not amps:
        return dict(p_amp_mean=np.nan, p_present_pct=np.nan, p_consistency=np.nan)
    amps = np.asarray(amps); flags = np.asarray(present_flags)
    p_amp_mean = float(np.mean(amps))
    p_present_pct = float(np.mean(flags) * 100.0)
    p_cons = float(1.0 - (np.std(amps) / (np.mean(amps) + 1e-9))) if np.mean(amps) > 0 else 0.0
    return dict(p_amp_mean=p_amp_mean, p_present_pct=p_present_pct, p_consistency=p_cons)

# ──────────────────────────────────────────────────────────────────────────
# Feature 3: Signal-quality index
# ──────────────────────────────────────────────────────────────────────────
def sqi_features(sig, fs):
    """
    Returns:
      - flat_pct       : fraction of samples where local std < eps
      - baseline_drift : std of slow envelope (0.5 Hz low-pass)
      - kurt           : excess kurtosis (high in clean ECG due to R-peaks)
      - snr_proxy      : variance(bandpass) / variance(residual)
      - clip_pct       : fraction of samples at the signal extremes
    """
    if len(sig) < fs * 2:
        return dict(flat_pct=np.nan, baseline_drift=np.nan, kurt=np.nan,
                    snr_proxy=np.nan, clip_pct=np.nan)
    # Flat-line: rolling std over 100 ms windows
    w = max(1, int(0.1 * fs))
    s2 = np.lib.stride_tricks.sliding_window_view(sig, w)
    roll_std = s2.std(axis=1)
    eps = 1e-4
    flat_pct = float(np.mean(roll_std < eps) * 100.0)
    # Baseline drift: low-pass below 0.5 Hz
    try:
        b, a = butter(2, 0.5/(fs/2), btype='low')
        bd = filtfilt(b, a, sig)
        baseline_drift = float(np.std(bd))
    except Exception:
        baseline_drift = np.nan
    kurt = float(kurtosis(sig, fisher=True))
    # SNR proxy = variance(QRS-band 5-25Hz) / variance(non-QRS residual)
    try:
        bp = bandpass(sig, fs, 5.0, 25.0)
        residual = sig - bp
        snr = float(np.var(bp) / (np.var(residual) + 1e-9))
    except Exception:
        snr = np.nan
    # Clipping
    rng = np.max(sig) - np.min(sig)
    if rng > 0:
        upper = np.percentile(sig, 99.5)
        lower = np.percentile(sig, 0.5)
        clip_pct = float(np.mean((sig >= upper) | (sig <= lower)) * 100.0)
    else:
        clip_pct = 100.0
    return dict(flat_pct=flat_pct, baseline_drift=baseline_drift,
                kurt=kurt, snr_proxy=snr, clip_pct=clip_pct)

# ──────────────────────────────────────────────────────────────────────────
# Feature 4: Multi-window persistence
# ──────────────────────────────────────────────────────────────────────────
def persistence_features(sig, fs, win_sec=10.0, hop_sec=5.0):
    """
    Split signal into overlapping windows; for each window flag 'AFib-like'
    if its RR-irregularity (cv_rr or rmssd) exceeds typical sinus thresholds.
    Persistence = fraction of windows flagged + longest consecutive run.
    """
    win = int(win_sec * fs); hop = int(hop_sec * fs)
    n_win = max(0, (len(sig) - win) // hop + 1)
    flags = []
    for i in range(n_win):
        seg = sig[i*hop : i*hop + win]
        peaks = detect_r_peaks(seg, fs)
        if len(peaks) < 4:
            flags.append(0); continue
        rr_ms = np.diff(peaks) / fs * 1000.0
        if len(rr_ms) < 3:
            flags.append(0); continue
        rmssd_w = float(np.sqrt(np.mean(np.diff(rr_ms)**2)))
        # AFib literature: RMSSD > 100 ms is highly suggestive
        flags.append(int(rmssd_w > 100.0))
    if not flags:
        return dict(n_windows=0, persistence_pct=np.nan,
                    longest_run=np.nan, persistence_score=np.nan)
    flags = np.asarray(flags)
    persistence_pct = float(np.mean(flags) * 100.0)
    # longest consecutive run of 1s
    longest = 0; cur = 0
    for f in flags:
        cur = cur + 1 if f else 0
        longest = max(longest, cur)
    # Composite persistence score: fraction × normalized longest run
    persistence_score = float(persistence_pct/100.0 * (longest / len(flags)))
    return dict(n_windows=int(len(flags)),
                persistence_pct=persistence_pct,
                longest_run=int(longest),
                persistence_score=persistence_score)

# ──────────────────────────────────────────────────────────────────────────
# Per-event combined feature extraction
# ──────────────────────────────────────────────────────────────────────────
def extract_features_for_event(json_path):
    sig, fs = load_ecg_strip(json_path)
    if sig is None or len(sig) < fs * 5:
        return None
    peaks = detect_r_peaks(sig, fs)
    rr_ms = np.diff(peaks) / fs * 1000.0 if len(peaks) >= 2 else np.array([])
    out = {}
    out.update(rr_irregularity_features(rr_ms))
    out.update(pwave_features(sig, fs, peaks))
    out.update(sqi_features(sig, fs))
    out.update(persistence_features(sig, fs))
    out['n_peaks'] = int(len(peaks))
    out['duration_s'] = float(len(sig) / fs)
    return out

# ──────────────────────────────────────────────────────────────────────────
# Run on AFib TP and FP cohorts
# ──────────────────────────────────────────────────────────────────────────
def gather(csv_path, data_dir, event_type='Atrial Fibrillation', max_n=None):
    df = pd.read_csv(csv_path)
    df.columns = [c.lower() for c in df.columns]
    # event-type column
    et_col = 'event_type' if 'event_type' in df.columns else ('event' if 'event' in df.columns else None)
    if et_col:
        df = df[df[et_col].astype(str).str.strip().str.lower() == event_type.lower()].copy()
    paths = []
    pcol = None
    for c in ['json_path', 'path', 'json', 'file', 'filename', 'json file', 'json_file']:
        if c in df.columns:
            pcol = c; break
    if pcol is None:
        paths = glob.glob(os.path.join(data_dir, event_type, '*', '*.json'))
    else:
        for p in df[pcol].astype(str):
            p_norm = p.replace('\\', '/')
            if not os.path.isabs(p_norm):
                p_norm = os.path.join(data_dir, p_norm)
            paths.append(p_norm)
    paths = [p for p in paths if isinstance(p, str) and p.endswith('.json') and os.path.exists(p)]
    if max_n: paths = paths[:max_n]
    print(f"  → {len(paths)} files to process")
    rows = []
    for i, p in enumerate(paths):
        if i % 50 == 0: print(f"     {i}/{len(paths)}")
        feats = extract_features_for_event(p)
        if feats:
            feats['path'] = p
            rows.append(feats)
    return pd.DataFrame(rows)

if __name__ == '__main__':
    OUT = 'res/motion_analysis/data'
    os.makedirs(OUT, exist_ok=True)

    print("\n[1/2] Extracting ECG features from AFib TP cohort...")
    tp_df = gather('./data/ecg-tp_rex/summary.csv',
                   './data/ecg-tp_rex',
                   event_type='Atrial Fibrillation',
                   max_n=200)
    tp_df.to_csv(f'{OUT}/afib_tp_ecg_features.csv', index=False)
    print(f"  Saved {len(tp_df)} TP feature rows")

    print("\n[2/2] Extracting ECG features from AFib FP cohort...")
    fp_df = gather('./data/ecg_fp_doctor removed1/summary.csv',
                   './data/ecg_fp_doctor removed1',
                   event_type='Atrial Fibrillation',
                   max_n=46)  # all available
    fp_df.to_csv(f'{OUT}/afib_fp_ecg_features.csv', index=False)
    print(f"  Saved {len(fp_df)} FP feature rows")

    # ──────────────────────────────────────────────────────────────────
    # Comparison summary
    # ──────────────────────────────────────────────────────────────────
    print("\n" + "="*80)
    print("AFib TP vs FP — Non-Motion ECG Feature Comparison")
    print("="*80)
    metrics = ['rmssd', 'pnn50', 'samp_en', 'cv_rr', 'mean_rr',
               'p_amp_mean', 'p_present_pct', 'p_consistency',
               'flat_pct', 'baseline_drift', 'kurt', 'snr_proxy', 'clip_pct',
               'persistence_pct', 'longest_run', 'persistence_score']
    from scipy.stats import mannwhitneyu, ttest_ind
    print(f"\n{'metric':>18}  {'TP_med':>9}  {'FP_med':>9}  {'TP_p25':>8}  {'TP_p75':>8}  {'FP_p25':>8}  {'FP_p75':>8}  {'p_value':>9}  {'sep':>5}")
    rows = []
    for m in metrics:
        if m not in tp_df.columns or m not in fp_df.columns: continue
        tp_v = tp_df[m].dropna().values
        fp_v = fp_df[m].dropna().values
        if len(tp_v) < 3 or len(fp_v) < 3: continue
        try:
            _, p = mannwhitneyu(tp_v, fp_v, alternative='two-sided')
        except Exception:
            p = np.nan
        ratio = (np.median(tp_v) + 1e-9) / (np.median(fp_v) + 1e-9)
        sep = "***" if p<0.001 else ("**" if p<0.01 else ("*" if p<0.05 else "ns"))
        tp_med, fp_med = np.median(tp_v), np.median(fp_v)
        rows.append([m, tp_med, fp_med, np.percentile(tp_v,25), np.percentile(tp_v,75),
                     np.percentile(fp_v,25), np.percentile(fp_v,75), p, sep])
        print(f"{m:>18}  {tp_med:>9.2f}  {fp_med:>9.2f}  {np.percentile(tp_v,25):>8.2f}  "
              f"{np.percentile(tp_v,75):>8.2f}  {np.percentile(fp_v,25):>8.2f}  {np.percentile(fp_v,75):>8.2f}  "
              f"{p:>9.2e}  {sep:>5}")

    cmp_df = pd.DataFrame(rows, columns=['metric','tp_median','fp_median','tp_p25','tp_p75',
                                         'fp_p25','fp_p75','p_value','sig'])
    cmp_df.to_csv(f'{OUT}/afib_ecg_feature_comparison.csv', index=False)
    print(f"\nComparison saved to {OUT}/afib_ecg_feature_comparison.csv")
