#!/usr/bin/env python3
"""
compare_tp_fp_sqi_motion.py
============================

Compare SQI (signal quality) and motion/accelerometer characteristics
between the TP dataset (ecg_tp_fzark) and FP dataset (ecg_fp_doctor removed1),
broken down by event type.

Features extracted per record:
  Motion:  mean_motion, max_motion, std_motion, median_motion, peak_ratio,
           zero_motion_pct  (all in mG, DC-removed via 5-sample rolling mean)
  SQI:     kurt, baseline_drift, snr_proxy  (from ECG signal)
  Rhythm:  mean_hr_bpm, rr_cv, samp_en, n_peaks

Output:
  res/tp_fp_sqi_motion/
    ├── tp_features.csv
    ├── fp_features.csv
    ├── per_class_comparison.csv
    ├── aggregate_comparison.csv
    └── SQI_MOTION_COMPARISON_REPORT.md

Usage:
  python compare_tp_fp_sqi_motion.py [--per-class N] [--workers N]
"""
from __future__ import annotations
import argparse
import json
import math
import os
import sys
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats as sp_stats
from scipy.signal import butter, filtfilt, find_peaks
from scipy.stats import kurtosis as scipy_kurtosis

warnings.filterwarnings('ignore')

# ── paths ──────────────────────────────────────────────────────────────────
TP_CSV     = './data/ecg_tp_fzark/summary.csv'
TP_DIR     = './data/ecg_tp_fzark'
FP_CSV     = './data/ecg_fp_doctor removed1/summary.csv'
FP_DIR     = './data/ecg_fp_doctor removed1'
OUT_DIR    = './res/tp_fp_sqi_motion'

# ── constants ──────────────────────────────────────────────────────────────
FS_DEFAULT         = 128
ACC_SCALE_FACTOR   = 2048.0
DC_REMOVAL_WINDOW  = 5
MOTION_UNITS       = 1000.0      # G → mG
ACC_SAMPLING_RATE  = 5.0         # Hz


# ═══════════════════════════════════════════════════════════════════════════
# Feature extraction (self-contained — no dependency on other project files)
# ═══════════════════════════════════════════════════════════════════════════

def _bandpass(sig, fs, lo=5.0, hi=20.0, order=2):
    nyq = fs / 2.0
    b, a = butter(order, [lo / nyq, hi / nyq], btype='band')
    return filtfilt(b, a, sig)


def _detect_r_peaks(sig, fs):
    if len(sig) < fs * 2:
        return np.array([], dtype=int)
    bp = _bandpass(sig, fs)
    deriv = np.diff(bp, prepend=bp[0])
    sq = deriv ** 2
    win = max(1, int(0.150 * fs))
    integ = np.convolve(sq, np.ones(win) / win, mode='same')
    thresh = 0.4 * np.percentile(integ, 99)
    min_dist = int(0.25 * fs)
    peaks, _ = find_peaks(integ, height=thresh, distance=min_dist)
    win_r = int(0.05 * fs)
    refined = []
    for p in peaks:
        a = max(0, p - win_r); b = min(len(sig), p + win_r + 1)
        if b > a:
            refined.append(a + int(np.argmax(sig[a:b])))
    return np.asarray(refined, dtype=int)


def _sample_entropy(x, m=2, r=None):
    x = np.asarray(x, dtype=float)
    N = len(x)
    if N < m + 2:
        return np.nan
    if r is None:
        r = 0.2 * np.std(x)
    if r <= 0:
        return np.nan
    def _phi(mm):
        T = np.array([x[i:i + mm] for i in range(N - mm + 1)])
        C = 0
        for i in range(len(T)):
            d = np.max(np.abs(T - T[i]), axis=1)
            C += np.sum(d <= r) - 1
        return C
    A = _phi(m + 1)
    B = _phi(m)
    if B == 0 or A == 0:
        return np.nan
    return -math.log(A / B)


def extract_all_features(json_path: str) -> Optional[Dict[str, float]]:
    """
    Extract motion + SQI + rhythm features from a single event JSON.
    Returns dict or None if the file can't be parsed.
    """
    try:
        with open(json_path) as f:
            data = json.load(f)
    except Exception:
        return None
    if not isinstance(data, list) or len(data) == 0:
        return None

    # ── Motion ─────────────────────────────────────────────────────────────
    acc_rows = []
    ecg_chunks = []
    fs = FS_DEFAULT
    mag = 1000
    for item in data:
        if not isinstance(item, dict):
            continue
        d = item.get('data', {})
        # accelerometer
        acc = d.get('acc')
        if acc:
            for a in acc:
                if isinstance(a, dict):
                    acc_rows.append([a.get('x', 0), a.get('y', 0), a.get('z', 0)])
                elif isinstance(a, (list, tuple)) and len(a) >= 3:
                    acc_rows.append([a[0], a[1], a[2]])
        # ECG
        ecg = d.get('ecg')
        if ecg is not None:
            fs = d.get('sf', fs) or fs
            mag = d.get('magnification', mag) or mag
            ecg_chunks.extend(ecg)

    out = {}

    # Motion features
    if acc_rows:
        arr = np.asarray(acc_rows, dtype=float) / ACC_SCALE_FACTOR
        vm = np.sqrt(np.sum(arr * arr, axis=1))
        w = DC_REMOVAL_WINDOW
        if len(vm) >= w:
            kern = np.ones(w) / w
            baseline = np.convolve(vm, kern, mode='same')
            dyn = np.abs(vm - baseline) * MOTION_UNITS
        else:
            dyn = np.abs(vm - vm.mean()) * MOTION_UNITS
        out['mean_motion']     = float(np.mean(dyn))
        out['max_motion']      = float(np.max(dyn))
        out['std_motion']      = float(np.std(dyn))
        out['median_motion']   = float(np.median(dyn))
        out['peak_ratio']      = float(np.max(dyn) / (np.mean(dyn) + 1e-6))
        out['zero_motion_pct'] = float(100.0 * np.mean(dyn < 1.0))
    else:
        for k in ['mean_motion', 'max_motion', 'std_motion', 'median_motion',
                   'peak_ratio', 'zero_motion_pct']:
            out[k] = np.nan

    # ── ECG-based features ─────────────────────────────────────────────────
    sig = np.asarray(ecg_chunks, dtype=float) / float(mag) if ecg_chunks else np.array([])

    if len(sig) >= fs * 3:
        peaks = _detect_r_peaks(sig, fs)
        rr_ms = np.diff(peaks) / fs * 1000.0 if len(peaks) >= 2 else np.array([])
        out['n_peaks'] = int(len(peaks))

        # RR / rhythm features
        if len(rr_ms) >= 3:
            mean_rr = float(np.mean(rr_ms))
            out['mean_hr_bpm']  = 60000.0 / mean_rr if mean_rr > 0 else np.nan
            out['rr_cv']        = float(np.std(rr_ms) / (mean_rr + 1e-9))
            out['samp_en']      = float(_sample_entropy(rr_ms, m=2))
        else:
            out['mean_hr_bpm'] = np.nan
            out['rr_cv']       = np.nan
            out['samp_en']     = np.nan

        # SQI features
        try:
            out['kurt'] = float(scipy_kurtosis(sig, fisher=True))
        except Exception:
            out['kurt'] = np.nan
        try:
            b, a = butter(2, 0.5 / (fs / 2), btype='low')
            out['baseline_drift'] = float(np.std(filtfilt(b, a, sig)))
        except Exception:
            out['baseline_drift'] = np.nan
        try:
            bp = _bandpass(sig, fs, 5.0, 25.0)
            out['snr_proxy'] = float(np.var(bp) / (np.var(sig - bp) + 1e-9))
        except Exception:
            out['snr_proxy'] = np.nan
    else:
        for k in ['n_peaks', 'mean_hr_bpm', 'rr_cv', 'samp_en',
                   'kurt', 'baseline_drift', 'snr_proxy']:
            out[k] = np.nan

    # Sanitize: replace inf with nan
    for k, v in list(out.items()):
        if isinstance(v, float) and (math.isinf(v)):
            out[k] = np.nan
    return out


def _worker(args):
    """Worker for parallel extraction."""
    json_path, event_type = args
    feats = extract_all_features(json_path)
    if feats is not None:
        feats['event_type'] = event_type
        feats['json_path'] = json_path
    return feats


# ═══════════════════════════════════════════════════════════════════════════
# Dataset loading & sampling
# ═══════════════════════════════════════════════════════════════════════════

def load_and_sample(csv_path: str, data_dir: str, per_class: int,
                    label: str) -> pd.DataFrame:
    """Load summary CSV, stratified sample up to per_class per event type."""
    df = pd.read_csv(csv_path)
    print(f"\n[{label}] Loaded {len(df)} records from {csv_path}")

    sampled = []
    for et in sorted(df['Event Type'].unique()):
        sub = df[df['Event Type'] == et]
        if len(sub) > per_class:
            sub = sub.sample(n=per_class, random_state=42)
        sampled.append(sub)
    df_s = pd.concat(sampled, ignore_index=True)
    print(f"[{label}] Sampled {len(df_s)} records across {df_s['Event Type'].nunique()} event types")
    return df_s


def extract_dataset_features(df: pd.DataFrame, data_dir: str,
                             label: str, workers: int = 8) -> pd.DataFrame:
    """Extract features from all records in df."""
    tasks = []
    for _, row in df.iterrows():
        jp = os.path.join(data_dir, row['JSON File'].replace('\\', '/'))
        tasks.append((jp, row['Event Type']))

    results = []
    done = 0
    total = len(tasks)
    print(f"[{label}] Extracting features from {total} records (workers={workers})...")

    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_worker, t): t for t in tasks}
        for fut in as_completed(futures):
            done += 1
            if done % 500 == 0 or done == total:
                print(f"  [{label}] {done}/{total}")
            r = fut.result()
            if r is not None:
                results.append(r)

    feat_df = pd.DataFrame(results)
    print(f"[{label}] Successfully extracted {len(feat_df)}/{total} records")
    return feat_df


# ═══════════════════════════════════════════════════════════════════════════
# Statistical comparison
# ═══════════════════════════════════════════════════════════════════════════

METRICS = [
    # Motion
    'mean_motion', 'max_motion', 'std_motion', 'median_motion',
    'peak_ratio', 'zero_motion_pct',
    # SQI
    'kurt', 'baseline_drift', 'snr_proxy',
    # Rhythm
    'mean_hr_bpm', 'rr_cv', 'samp_en', 'n_peaks',
]

METRIC_GROUPS = {
    'Motion':  ['mean_motion', 'max_motion', 'std_motion', 'median_motion',
                'peak_ratio', 'zero_motion_pct'],
    'SQI':     ['kurt', 'baseline_drift', 'snr_proxy'],
    'Rhythm':  ['mean_hr_bpm', 'rr_cv', 'samp_en', 'n_peaks'],
}


def compare_metric(fp_vals: np.ndarray, tp_vals: np.ndarray) -> Dict:
    """Compare a single metric between FP and TP populations."""
    fp = fp_vals[~np.isnan(fp_vals)]
    tp = tp_vals[~np.isnan(tp_vals)]
    out = dict(
        fp_n=len(fp), tp_n=len(tp),
        fp_mean=np.nan, tp_mean=np.nan,
        fp_median=np.nan, tp_median=np.nan,
        fp_p25=np.nan, fp_p75=np.nan, fp_p95=np.nan,
        tp_p25=np.nan, tp_p75=np.nan, tp_p95=np.nan,
        diff_mean=np.nan, ratio=np.nan,
        cohens_d=np.nan,
        ttest_p=np.nan, mannwhitney_p=np.nan,
    )
    if len(fp) < 3 or len(tp) < 3:
        return out

    out['fp_mean']   = float(np.mean(fp))
    out['tp_mean']   = float(np.mean(tp))
    out['fp_median'] = float(np.median(fp))
    out['tp_median'] = float(np.median(tp))
    out['fp_p25']    = float(np.percentile(fp, 25))
    out['fp_p75']    = float(np.percentile(fp, 75))
    out['fp_p95']    = float(np.percentile(fp, 95))
    out['tp_p25']    = float(np.percentile(tp, 25))
    out['tp_p75']    = float(np.percentile(tp, 75))
    out['tp_p95']    = float(np.percentile(tp, 95))

    out['diff_mean'] = out['fp_mean'] - out['tp_mean']
    denom = out['tp_mean'] if abs(out['tp_mean']) > 1e-9 else 1e-9
    out['ratio'] = out['fp_mean'] / denom

    # Cohen's d
    pooled_std = math.sqrt((np.var(fp) + np.var(tp)) / 2.0)
    out['cohens_d'] = (out['fp_mean'] - out['tp_mean']) / pooled_std if pooled_std > 1e-9 else np.nan

    # Statistical tests
    try:
        _, p = sp_stats.ttest_ind(fp, tp, equal_var=False)
        out['ttest_p'] = float(p)
    except Exception:
        pass
    try:
        _, p = sp_stats.mannwhitneyu(fp, tp, alternative='two-sided')
        out['mannwhitney_p'] = float(p)
    except Exception:
        pass

    return out


def build_comparison(fp_features: pd.DataFrame, tp_features: pd.DataFrame,
                     event_types: List[str]) -> pd.DataFrame:
    """Build the per-class, per-metric comparison table."""
    rows = []
    for et in event_types:
        fp_sub = fp_features[fp_features['event_type'] == et]
        tp_sub = tp_features[tp_features['event_type'] == et]
        for metric in METRICS:
            fp_vals = fp_sub[metric].values.astype(float) if metric in fp_sub.columns else np.array([])
            tp_vals = tp_sub[metric].values.astype(float) if metric in tp_sub.columns else np.array([])
            comp = compare_metric(fp_vals, tp_vals)
            comp['event_type'] = et
            comp['metric'] = metric
            rows.append(comp)

    return pd.DataFrame(rows)


# ═══════════════════════════════════════════════════════════════════════════
# Report generation
# ═══════════════════════════════════════════════════════════════════════════

def _sig_stars(p):
    if p != p or p is None:
        return ''
    if p < 0.001:
        return '***'
    if p < 0.01:
        return '**'
    if p < 0.05:
        return '*'
    return ''


def _fmt(v, decimals=2):
    if v != v or v is None:
        return '—'
    return f'{v:.{decimals}f}'


def generate_report(comp_df: pd.DataFrame,
                    fp_features: pd.DataFrame, tp_features: pd.DataFrame,
                    event_types: List[str], per_class: int) -> str:
    """Generate the markdown report."""
    lines = []
    lines.append('# SQI & Motion Comparison: TP (fzark) vs FP (doctor-removed)')
    lines.append('')
    lines.append(f'**TP Dataset**: `ecg_tp_fzark` — clinician-confirmed true positives')
    lines.append(f'**FP Dataset**: `ecg_fp_doctor removed1` — clinician-removed false positives')
    lines.append(f'**Sampling**: up to {per_class} records per event type per dataset (random_state=42)')
    lines.append(f'**TP records extracted**: {len(tp_features)}')
    lines.append(f'**FP records extracted**: {len(fp_features)}')
    lines.append('')
    lines.append('**Interpretation**: FP events are noise the model fires on incorrectly; ')
    lines.append('TP events are real arrhythmias. Differences in SQI/motion help explain ')
    lines.append('*why* the model makes mistakes and guide suppressor design.')
    lines.append('')
    lines.append('---')
    lines.append('')

    # ── Section 1: Aggregate overview ──────────────────────────────────────
    lines.append('## 1. Aggregate overview (all event types pooled)')
    lines.append('')
    for group_name, group_metrics in METRIC_GROUPS.items():
        lines.append(f'### {group_name} features')
        lines.append('')
        lines.append('| Metric | FP mean | FP median | TP mean | TP median | Ratio (FP/TP) | Cohen\'s d | p-value |')
        lines.append('|---|---|---|---|---|---|---|---|')
        for metric in group_metrics:
            fp_vals = fp_features[metric].values.astype(float) if metric in fp_features.columns else np.array([])
            tp_vals = tp_features[metric].values.astype(float) if metric in tp_features.columns else np.array([])
            c = compare_metric(fp_vals, tp_vals)
            p_str = _fmt(c['mannwhitney_p'], 4) + _sig_stars(c['mannwhitney_p'])
            lines.append(f'| {metric} | {_fmt(c["fp_mean"])} | {_fmt(c["fp_median"])} '
                         f'| {_fmt(c["tp_mean"])} | {_fmt(c["tp_median"])} '
                         f'| {_fmt(c["ratio"])} | {_fmt(c["cohens_d"])} | {p_str} |')
        lines.append('')

    lines.append('---')
    lines.append('')

    # ── Section 2: Per-class motion comparison ─────────────────────────────
    lines.append('## 2. Per-class motion comparison')
    lines.append('')
    lines.append('| Event Type | FP n | TP n | FP mean motion (mG) | TP mean motion (mG) | Ratio | FP max motion | TP max motion | FP zero% | TP zero% | p-value |')
    lines.append('|---|---|---|---|---|---|---|---|---|---|---|')

    for et in event_types:
        sub = comp_df[comp_df['event_type'] == et]
        mm = sub[sub['metric'] == 'mean_motion'].iloc[0] if len(sub[sub['metric'] == 'mean_motion']) else None
        mx = sub[sub['metric'] == 'max_motion'].iloc[0] if len(sub[sub['metric'] == 'max_motion']) else None
        zm = sub[sub['metric'] == 'zero_motion_pct'].iloc[0] if len(sub[sub['metric'] == 'zero_motion_pct']) else None
        if mm is None:
            continue
        p_str = _fmt(mm['mannwhitney_p'], 4) + _sig_stars(mm['mannwhitney_p'])
        lines.append(
            f'| {et} | {mm["fp_n"]:.0f} | {mm["tp_n"]:.0f} '
            f'| {_fmt(mm["fp_mean"])} | {_fmt(mm["tp_mean"])} | {_fmt(mm["ratio"])} '
            f'| {_fmt(mx["fp_mean"] if mx is not None else np.nan)} '
            f'| {_fmt(mx["tp_mean"] if mx is not None else np.nan)} '
            f'| {_fmt(zm["fp_mean"] if zm is not None else np.nan)} '
            f'| {_fmt(zm["tp_mean"] if zm is not None else np.nan)} '
            f'| {p_str} |'
        )
    lines.append('')
    lines.append('---')
    lines.append('')

    # ── Section 3: Per-class SQI comparison ────────────────────────────────
    lines.append('## 3. Per-class SQI comparison')
    lines.append('')
    lines.append('| Event Type | FP n | TP n | FP snr_proxy | TP snr_proxy | FP kurt | TP kurt | FP baseline_drift | TP baseline_drift | SNR p-value |')
    lines.append('|---|---|---|---|---|---|---|---|---|---|')

    for et in event_types:
        sub = comp_df[comp_df['event_type'] == et]
        snr = sub[sub['metric'] == 'snr_proxy'].iloc[0] if len(sub[sub['metric'] == 'snr_proxy']) else None
        krt = sub[sub['metric'] == 'kurt'].iloc[0] if len(sub[sub['metric'] == 'kurt']) else None
        bd  = sub[sub['metric'] == 'baseline_drift'].iloc[0] if len(sub[sub['metric'] == 'baseline_drift']) else None
        if snr is None:
            continue
        p_str = _fmt(snr['mannwhitney_p'], 4) + _sig_stars(snr['mannwhitney_p'])
        lines.append(
            f'| {et} | {snr["fp_n"]:.0f} | {snr["tp_n"]:.0f} '
            f'| {_fmt(snr["fp_mean"])} | {_fmt(snr["tp_mean"])} '
            f'| {_fmt(krt["fp_mean"])} | {_fmt(krt["tp_mean"])} '
            f'| {_fmt(bd["fp_mean"], 4)} | {_fmt(bd["tp_mean"], 4)} '
            f'| {p_str} |'
        )
    lines.append('')
    lines.append('---')
    lines.append('')

    # ── Section 4: Per-class rhythm comparison ─────────────────────────────
    lines.append('## 4. Per-class rhythm comparison')
    lines.append('')
    lines.append('| Event Type | FP n | TP n | FP HR (bpm) | TP HR (bpm) | FP rr_cv | TP rr_cv | FP samp_en | TP samp_en | HR p-value |')
    lines.append('|---|---|---|---|---|---|---|---|---|---|')

    for et in event_types:
        sub = comp_df[comp_df['event_type'] == et]
        hr   = sub[sub['metric'] == 'mean_hr_bpm'].iloc[0] if len(sub[sub['metric'] == 'mean_hr_bpm']) else None
        rrcv = sub[sub['metric'] == 'rr_cv'].iloc[0] if len(sub[sub['metric'] == 'rr_cv']) else None
        se   = sub[sub['metric'] == 'samp_en'].iloc[0] if len(sub[sub['metric'] == 'samp_en']) else None
        if hr is None:
            continue
        p_str = _fmt(hr['mannwhitney_p'], 4) + _sig_stars(hr['mannwhitney_p'])
        lines.append(
            f'| {et} | {hr["fp_n"]:.0f} | {hr["tp_n"]:.0f} '
            f'| {_fmt(hr["fp_mean"], 1)} | {_fmt(hr["tp_mean"], 1)} '
            f'| {_fmt(rrcv["fp_mean"], 3)} | {_fmt(rrcv["tp_mean"], 3)} '
            f'| {_fmt(se["fp_mean"])} | {_fmt(se["tp_mean"])} '
            f'| {p_str} |'
        )
    lines.append('')
    lines.append('---')
    lines.append('')

    # ── Section 5: Per-class detail (full percentile breakdown) ────────────
    lines.append('## 5. Detailed per-class percentile breakdown')
    lines.append('')

    for et in event_types:
        fp_sub = fp_features[fp_features['event_type'] == et]
        tp_sub = tp_features[tp_features['event_type'] == et]
        if len(fp_sub) < 3 and len(tp_sub) < 3:
            continue
        lines.append(f'### {et} (FP n={len(fp_sub)}, TP n={len(tp_sub)})')
        lines.append('')
        lines.append('| Metric | Dataset | Mean | Median | P25 | P75 | P95 | Std |')
        lines.append('|---|---|---|---|---|---|---|---|')

        for metric in METRICS:
            for ds_label, sub_df in [('FP', fp_sub), ('TP', tp_sub)]:
                if metric not in sub_df.columns:
                    continue
                vals = sub_df[metric].dropna().values
                if len(vals) < 1:
                    lines.append(f'| {metric} | {ds_label} | — | — | — | — | — | — |')
                    continue
                lines.append(
                    f'| {metric} | {ds_label} '
                    f'| {_fmt(np.mean(vals))} | {_fmt(np.median(vals))} '
                    f'| {_fmt(np.percentile(vals, 25))} | {_fmt(np.percentile(vals, 75))} '
                    f'| {_fmt(np.percentile(vals, 95))} | {_fmt(np.std(vals))} |'
                )
        lines.append('')

    lines.append('---')
    lines.append('')

    # ── Section 6: Key observations ────────────────────────────────────────
    lines.append('## 6. Key observations')
    lines.append('')
    lines.append(_generate_observations(comp_df, fp_features, tp_features, event_types))
    lines.append('')

    lines.append('---')
    lines.append('')
    lines.append(f'Full data: `{OUT_DIR}/per_class_comparison.csv`')
    lines.append('')
    lines.append('*Generated by `compare_tp_fp_sqi_motion.py`*')

    return '\n'.join(lines)


def _generate_observations(comp_df, fp_features, tp_features, event_types):
    """Auto-generate key observations from the data."""
    obs = []

    # 1. Overall motion difference
    fp_mm = fp_features['mean_motion'].dropna()
    tp_mm = tp_features['mean_motion'].dropna()
    if len(fp_mm) > 0 and len(tp_mm) > 0:
        ratio = fp_mm.mean() / (tp_mm.mean() + 1e-9)
        obs.append(
            f'1. **FP events have {ratio:.1f}× higher mean motion than TPs overall** '
            f'(FP: {fp_mm.mean():.1f} mG vs TP: {tp_mm.mean():.1f} mG). '
            f'FP median: {fp_mm.median():.1f} mG vs TP median: {tp_mm.median():.1f} mG. '
            f'This confirms motion artefacts are a major FP driver.'
        )

    # 2. Zero-motion comparison
    fp_zm = fp_features['zero_motion_pct'].dropna()
    tp_zm = tp_features['zero_motion_pct'].dropna()
    if len(fp_zm) > 0 and len(tp_zm) > 0:
        obs.append(
            f'2. **TP events are much more static**: {tp_zm.mean():.1f}% of TP samples '
            f'have zero-motion (<1 mG) vs {fp_zm.mean():.1f}% for FP. True arrhythmias '
            f'tend to occur at rest or during minimal activity.'
        )

    # 3. SNR comparison
    fp_snr = fp_features['snr_proxy'].dropna()
    tp_snr = tp_features['snr_proxy'].dropna()
    if len(fp_snr) > 0 and len(tp_snr) > 0:
        obs.append(
            f'3. **TP events have higher signal quality**: SNR proxy FP={fp_snr.mean():.2f} '
            f'vs TP={tp_snr.mean():.2f}. '
            f'FP median SNR: {fp_snr.median():.2f} vs TP median: {tp_snr.median():.2f}. '
            f'Higher SNR in TPs indicates cleaner ECG recordings.'
        )

    # 4. Kurtosis
    fp_k = fp_features['kurt'].dropna()
    tp_k = tp_features['kurt'].dropna()
    if len(fp_k) > 0 and len(tp_k) > 0:
        higher = 'FP' if fp_k.mean() > tp_k.mean() else 'TP'
        obs.append(
            f'4. **{higher} events have higher kurtosis** (FP: {fp_k.mean():.1f} vs '
            f'TP: {tp_k.mean():.1f}). Higher kurtosis indicates more peaky/impulsive signals. '
            f'FP 95th percentile: {np.percentile(fp_k, 95):.1f} vs TP: {np.percentile(tp_k, 95):.1f}.'
        )

    # 5. Find the class with largest FP/TP motion gap
    biggest_gap = None
    biggest_ratio = 0
    for et in event_types:
        sub = comp_df[(comp_df['event_type'] == et) & (comp_df['metric'] == 'mean_motion')]
        if len(sub) == 0:
            continue
        row = sub.iloc[0]
        if row['fp_n'] >= 10 and row['tp_n'] >= 10:
            r = row['ratio'] if not np.isnan(row['ratio']) else 0
            if r > biggest_ratio:
                biggest_ratio = r
                biggest_gap = (et, row)
    if biggest_gap:
        et, row = biggest_gap
        obs.append(
            f'5. **{et} has the largest FP/TP motion gap** '
            f'({biggest_ratio:.1f}× ratio, FP mean: {row["fp_mean"]:.1f} mG, '
            f'TP mean: {row["tp_mean"]:.1f} mG). '
            f'Motion-based filtering would be especially effective for this class.'
        )

    # 6. Find class where SQI differs most
    biggest_snr_gap = None
    biggest_snr_d = 0
    for et in event_types:
        sub = comp_df[(comp_df['event_type'] == et) & (comp_df['metric'] == 'snr_proxy')]
        if len(sub) == 0:
            continue
        row = sub.iloc[0]
        if row['fp_n'] >= 10 and row['tp_n'] >= 10:
            d = abs(row['cohens_d']) if not np.isnan(row['cohens_d']) else 0
            if d > biggest_snr_d:
                biggest_snr_d = d
                biggest_snr_gap = (et, row)
    if biggest_snr_gap:
        et, row = biggest_snr_gap
        obs.append(
            f'6. **{et} has the largest SQI divergence** '
            f'(Cohen\'s d={row["cohens_d"]:.2f}, FP SNR: {row["fp_mean"]:.2f}, '
            f'TP SNR: {row["tp_mean"]:.2f}). '
            f'SQI-based filtering may benefit this class most.'
        )

    # 7. HR comparison for relevant types
    tachy_types = ['Sinus Tachycardia', 'Atrial Fibrillation']
    for et in tachy_types:
        sub = comp_df[(comp_df['event_type'] == et) & (comp_df['metric'] == 'mean_hr_bpm')]
        if len(sub) == 0:
            continue
        row = sub.iloc[0]
        if row['fp_n'] >= 10 and row['tp_n'] >= 3:
            obs.append(
                f'7. **{et} HR comparison**: FP mean HR={row["fp_mean"]:.0f} bpm '
                f'vs TP mean HR={row["tp_mean"]:.0f} bpm '
                f'(p={row["mannwhitney_p"]:.4f}{_sig_stars(row["mannwhitney_p"])}). '
            )
            break  # only one HR observation

    return '\n\n'.join(obs)


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description='Compare TP vs FP SQI & motion')
    parser.add_argument('--per-class', type=int, default=300,
                        help='Max records per event type per dataset (default 300)')
    parser.add_argument('--workers', type=int, default=8,
                        help='Parallel extraction workers (default 8)')
    args = parser.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)

    # ── Load & sample ─────────────────────────────────────────────────────
    tp_df = load_and_sample(TP_CSV, TP_DIR, args.per_class, 'TP')
    fp_df = load_and_sample(FP_CSV, FP_DIR, args.per_class, 'FP')

    # ── Extract features ──────────────────────────────────────────────────
    tp_features = extract_dataset_features(tp_df, TP_DIR, 'TP', args.workers)
    fp_features = extract_dataset_features(fp_df, FP_DIR, 'FP', args.workers)

    # ── Save raw features ─────────────────────────────────────────────────
    tp_features.to_csv(os.path.join(OUT_DIR, 'tp_features.csv'), index=False)
    fp_features.to_csv(os.path.join(OUT_DIR, 'fp_features.csv'), index=False)
    print(f"\nSaved feature CSVs to {OUT_DIR}/")

    # ── Common event types ────────────────────────────────────────────────
    tp_types = set(tp_features['event_type'].unique())
    fp_types = set(fp_features['event_type'].unique())
    common = sorted(tp_types & fp_types)
    all_types = sorted(tp_types | fp_types)
    print(f"\nCommon event types ({len(common)}): {common}")
    print(f"TP-only: {sorted(tp_types - fp_types)}")
    print(f"FP-only: {sorted(fp_types - tp_types)}")

    # ── Build comparison ──────────────────────────────────────────────────
    comp_df = build_comparison(fp_features, tp_features, all_types)
    comp_df.to_csv(os.path.join(OUT_DIR, 'per_class_comparison.csv'), index=False)

    # Aggregate comparison (all pooled)
    agg_rows = []
    for metric in METRICS:
        fp_vals = fp_features[metric].values.astype(float) if metric in fp_features.columns else np.array([])
        tp_vals = tp_features[metric].values.astype(float) if metric in tp_features.columns else np.array([])
        c = compare_metric(fp_vals, tp_vals)
        c['metric'] = metric
        agg_rows.append(c)
    agg_df = pd.DataFrame(agg_rows)
    agg_df.to_csv(os.path.join(OUT_DIR, 'aggregate_comparison.csv'), index=False)

    # ── Generate report ───────────────────────────────────────────────────
    report = generate_report(comp_df, fp_features, tp_features, all_types, args.per_class)
    report_path = os.path.join(OUT_DIR, 'SQI_MOTION_COMPARISON_REPORT.md')
    with open(report_path, 'w') as f:
        f.write(report)
    print(f"\nReport: {report_path}")

    # ── Console summary ───────────────────────────────────────────────────
    print('\n' + '=' * 70)
    print('SUMMARY — Per-class mean_motion (mG)')
    print('=' * 70)
    print(f'{"Event Type":<35} {"FP mean":>10} {"TP mean":>10} {"Ratio":>8} {"p-value":>12}')
    print('-' * 75)
    for et in all_types:
        sub = comp_df[(comp_df['event_type'] == et) & (comp_df['metric'] == 'mean_motion')]
        if len(sub) == 0:
            continue
        r = sub.iloc[0]
        print(f'{et:<35} {r["fp_mean"]:10.2f} {r["tp_mean"]:10.2f} '
              f'{r["ratio"]:8.2f} {r["mannwhitney_p"]:12.4f}{_sig_stars(r["mannwhitney_p"])}')

    print()
    print(f'{"Event Type":<35} {"FP SNR":>10} {"TP SNR":>10} {"FP kurt":>10} {"TP kurt":>10}')
    print('-' * 75)
    for et in all_types:
        snr_sub = comp_df[(comp_df['event_type'] == et) & (comp_df['metric'] == 'snr_proxy')]
        krt_sub = comp_df[(comp_df['event_type'] == et) & (comp_df['metric'] == 'kurt')]
        if len(snr_sub) == 0:
            continue
        s = snr_sub.iloc[0]
        k = krt_sub.iloc[0] if len(krt_sub) > 0 else None
        print(f'{et:<35} {s["fp_mean"]:10.2f} {s["tp_mean"]:10.2f} '
              f'{k["fp_mean"] if k is not None else float("nan"):10.2f} '
              f'{k["tp_mean"] if k is not None else float("nan"):10.2f}')

    print(f'\nDone. Full report: {report_path}')


if __name__ == '__main__':
    main()
