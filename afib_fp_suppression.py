"""
afib_fp_suppression.py
======================

Production AFib false-positive suppression module.

Implements the staged filter recommended by the AFib motion + ECG analysis:

    Layer 1 — S5 tiered interpretable rules
              (zero-motion bypass | tachycardic AFib | bradycardic AFib rescue)
    Layer 2 — S6 logistic-regression ensemble over 18 motion + ECG features
              (probabilistic ranker, threshold-tunable)

Public API
----------
    AFibFPSuppressor(weights_path=None)
        .extract_features(json_path)  → dict
        .layer1_keep(feats)           → bool       (S5 rule)
        .layer2_score(feats)          → float in [0, 1]   (P(FP))
        .suppress(p_afib, json_path)  → SuppressionResult
            - final_prob   (model prob if kept, 0 if rejected)
            - keep         (bool)
            - layer1_pass  (bool)
            - layer2_pass  (bool)
            - reason       (str — first failing rule)

Reference: AFIB_NONMOTION_FEATURE_ANALYSIS.md
           AFIB_FILTER_RECOMMENDATIONS.md
           system_performance.py  (S5 + S6 design)
"""
from __future__ import annotations
import json
import math
import os
from dataclasses import dataclass
from typing import Optional, Dict

import numpy as np

# Try to import the existing motion utility; fall back to inline implementation
try:
    from motion_analysis_utilities import extract_motion_features
    _HAS_MOTION_UTIL = True
except Exception:
    _HAS_MOTION_UTIL = False

# scipy is only needed for the ECG branch
try:
    from scipy.signal import butter, filtfilt, find_peaks
    from scipy.stats import kurtosis as scipy_kurtosis
    _HAS_SCIPY = True
except Exception:
    _HAS_SCIPY = False


# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

FS_DEFAULT = 128                  # Holter ECG sampling rate
ACC_SCALE_FACTOR = 2048.0         # Raw → G
DC_REMOVAL_WINDOW = 5             # 5-sample rolling mean
MOTION_UNITS = 1000.0             # G → mG

# 18-feature order MUST match training of the S6 logistic regression
FEATURE_ORDER = [
    'mean_motion', 'max_motion', 'std_motion', 'median_motion',
    'peak_ratio', 'zero_motion_pct',
    'mean_rr', 'rmssd', 'pnn50', 'samp_en', 'cv_rr', 'kurt',
    'baseline_drift', 'snr_proxy',
    'p_present_pct', 'p_consistency',
    'persistence_pct', 'n_peaks',
]

# Bundled hard-rule thresholds (from S5 design)
S5_THRESHOLDS = dict(
    quiet_motion_max   = 1.0,    # mG — zero-motion auto-keep
    tachy_mean_rr_max  = 850.0,  # ms
    tachy_kurt_max     = 6.0,
    tachy_motion_max   = 30.0,   # mG
    rescue_mean_rr_max = 1100.0,
    rescue_samp_en_min = 1.3,
    rescue_p_max       = 30.0,
)


# ─────────────────────────────────────────────────────────────────────────────
# Result dataclass
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class SuppressionResult:
    final_prob: float
    keep: bool
    layer1_pass: bool
    layer2_pass: bool
    layer2_score: float        # P(FP) from the ranker (0 if N/A)
    reason: str                # short human-readable explanation
    features: Optional[Dict[str, float]] = None


# ─────────────────────────────────────────────────────────────────────────────
# ECG feature helpers (re-used from ecg_feature_analysis.py, inlined to avoid
# a circular import — pure numpy/scipy)
# ─────────────────────────────────────────────────────────────────────────────

def _bandpass(sig, fs, lo=5.0, hi=20.0, order=2):
    nyq = fs / 2.0
    b, a = butter(order, [lo/nyq, hi/nyq], btype='band')
    return filtfilt(b, a, sig)

def _detect_r_peaks(sig, fs):
    if not _HAS_SCIPY or len(sig) < fs * 2:
        return np.array([], dtype=int)
    bp = _bandpass(sig, fs)
    deriv = np.diff(bp, prepend=bp[0])
    sq = deriv ** 2
    win = max(1, int(0.150 * fs))
    integ = np.convolve(sq, np.ones(win)/win, mode='same')
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
    if N < m + 2: return np.nan
    if r is None: r = 0.2 * np.std(x)
    if r <= 0: return np.nan
    def _phi(mm):
        T = np.array([x[i:i+mm] for i in range(N - mm + 1)])
        C = 0
        for i in range(len(T)):
            d = np.max(np.abs(T - T[i]), axis=1)
            C += np.sum(d <= r) - 1
        return C
    A = _phi(m + 1); B = _phi(m)
    if B == 0 or A == 0: return np.nan
    return -math.log(A / B)


# ─────────────────────────────────────────────────────────────────────────────
# Inline motion feature extraction (when motion_analysis_utilities not present)
# ─────────────────────────────────────────────────────────────────────────────

def _extract_motion_inline(json_path):
    try:
        with open(json_path) as f:
            data = json.load(f)
    except Exception:
        return None
    if not isinstance(data, list):
        return None
    rows = []
    for item in data:
        if not isinstance(item, dict): continue
        d = item.get('data', {})
        acc = d.get('acc')
        if not acc: continue
        for a in acc:
            if isinstance(a, dict):
                rows.append([a.get('x', 0), a.get('y', 0), a.get('z', 0)])
            elif isinstance(a, (list, tuple)) and len(a) >= 3:
                rows.append([a[0], a[1], a[2]])
    if not rows: return None
    arr = np.asarray(rows, dtype=float) / ACC_SCALE_FACTOR
    mag = np.sqrt(np.sum(arr * arr, axis=1))
    # DC removal: subtract rolling mean
    w = DC_REMOVAL_WINDOW
    if len(mag) >= w:
        kern = np.ones(w) / w
        baseline = np.convolve(mag, kern, mode='same')
        dyn = np.abs(mag - baseline) * MOTION_UNITS
    else:
        dyn = np.abs(mag - mag.mean()) * MOTION_UNITS
    return dict(
        mean_motion=float(np.mean(dyn)),
        max_motion=float(np.max(dyn)),
        std_motion=float(np.std(dyn)),
        median_motion=float(np.median(dyn)),
        peak_ratio=float(np.max(dyn) / (np.mean(dyn) + 1e-6)),
        zero_motion_pct=float(100.0 * np.mean(dyn < 1.0)),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Main suppressor class
# ─────────────────────────────────────────────────────────────────────────────

class AFibFPSuppressor:
    """
    AFib FP suppressor — combines:
      Layer 1: S5 tiered interpretable rules    (hard gate)
      Layer 2: S6 logistic-regression ensemble  (probabilistic ranker)
    """

    def __init__(self, weights_path: Optional[str] = None,
                 layer2_threshold: float = 0.5,
                 use_layer2: bool = True,
                 fs: int = FS_DEFAULT):
        """
        weights_path: NPZ file with keys {'mean', 'scale', 'coef', 'intercept'}
                      produced by train_afib_suppressor.py.  If None, Layer 2
                      is disabled (Layer 1 only).
        layer2_threshold: P(FP) above which Layer 2 rejects the event.
        use_layer2: master switch for the LR ranker.
        fs: ECG sampling rate.
        """
        self.fs = fs
        self.layer2_threshold = layer2_threshold
        self.use_layer2 = use_layer2 and (weights_path is not None) and os.path.exists(weights_path)
        self._mean = self._scale = self._coef = self._intercept = None
        if self.use_layer2:
            w = np.load(weights_path)
            self._mean      = w['mean']
            self._scale     = w['scale']
            self._coef      = w['coef']
            self._intercept = float(w['intercept'])

    # ── feature extraction ────────────────────────────────────────────────
    def extract_features(self, json_path: str) -> Optional[Dict[str, float]]:
        """Extract the 18-feature vector (motion + ECG) from a single event JSON."""
        # Motion
        if _HAS_MOTION_UTIL:
            mfeat = extract_motion_features(json_path) or {}
        else:
            mfeat = _extract_motion_inline(json_path) or {}
        if not mfeat:
            return None

        # ECG signal
        try:
            with open(json_path) as f:
                records = json.load(f)
            chunks, fs, mag = [], self.fs, 1000
            for item in records:
                if not isinstance(item, dict): continue
                d = item.get('data', {})
                ecg = d.get('ecg')
                if ecg is None: continue
                fs  = d.get('sf', fs) or fs
                mag = d.get('magnification', mag) or mag
                chunks.extend(ecg)
            sig = np.asarray(chunks, dtype=float) / float(mag)
        except Exception:
            sig = np.array([])

        feats = dict(mfeat)
        if len(sig) >= fs * 3 and _HAS_SCIPY:
            feats.update(self._compute_ecg_features(sig, fs))
        else:
            for k in ['mean_rr','rmssd','pnn50','samp_en','cv_rr','kurt',
                      'baseline_drift','snr_proxy','p_present_pct',
                      'p_consistency','persistence_pct','n_peaks']:
                feats[k] = 0.0
        return feats

    def _compute_ecg_features(self, sig, fs):
        peaks = _detect_r_peaks(sig, fs)
        rr_ms = np.diff(peaks) / fs * 1000.0 if len(peaks) >= 2 else np.array([])
        out = dict(n_peaks=int(len(peaks)))
        # RR-irregularity
        if len(rr_ms) >= 3:
            diff = np.diff(rr_ms)
            out['mean_rr']  = float(np.mean(rr_ms))
            out['rmssd']    = float(np.sqrt(np.mean(diff**2)))
            out['pnn50']    = float(np.mean(np.abs(diff) > 50.0) * 100.0)
            out['cv_rr']    = float(np.std(rr_ms) / (np.mean(rr_ms) + 1e-9))
            out['samp_en']  = float(_sample_entropy(rr_ms, m=2))
        else:
            out.update(dict(mean_rr=0.0, rmssd=0.0, pnn50=0.0,
                            cv_rr=0.0, samp_en=0.0))
        # P-wave
        if len(peaks) >= 3:
            pr_s = int(0.20 * fs); pr_e = int(0.06 * fs)
            amps, flags = [], []
            for r in peaks:
                a = r - pr_s; b = r - pr_e
                if a < 0 or b <= a: continue
                win = sig[a:b]
                base = np.median(win)
                amp = float(np.max(win) - base)
                amps.append(amp)
                mad = float(np.median(np.abs(win - base)) + 1e-9)
                flags.append(amp > 3.0 * mad)
            if amps:
                amps = np.asarray(amps)
                out['p_present_pct'] = float(np.mean(flags) * 100.0)
                out['p_consistency'] = float(
                    1.0 - (np.std(amps) / (np.mean(amps) + 1e-9))) if np.mean(amps) > 0 else 0.0
            else:
                out['p_present_pct'] = 0.0; out['p_consistency'] = 0.0
        else:
            out['p_present_pct'] = 0.0; out['p_consistency'] = 0.0
        # SQI
        try:
            out['kurt'] = float(scipy_kurtosis(sig, fisher=True))
            b, a = butter(2, 0.5/(fs/2), btype='low')
            out['baseline_drift'] = float(np.std(filtfilt(b, a, sig)))
            bp = _bandpass(sig, fs, 5.0, 25.0)
            out['snr_proxy'] = float(np.var(bp) / (np.var(sig - bp) + 1e-9))
        except Exception:
            out['kurt'] = 0.0; out['baseline_drift'] = 0.0; out['snr_proxy'] = 0.0
        # Persistence
        win_n = int(10 * fs); hop_n = int(5 * fs)
        nw = max(0, (len(sig) - win_n) // hop_n + 1)
        flags_w = []
        for i in range(nw):
            seg = sig[i*hop_n : i*hop_n + win_n]
            pk = _detect_r_peaks(seg, fs)
            if len(pk) < 4:
                flags_w.append(0); continue
            rr_w = np.diff(pk) / fs * 1000.0
            rmssd_w = float(np.sqrt(np.mean(np.diff(rr_w)**2))) if len(rr_w) >= 2 else 0.0
            flags_w.append(int(rmssd_w > 100.0))
        out['persistence_pct'] = float(np.mean(flags_w) * 100.0) if flags_w else 0.0
        # Replace any NaN with 0 to keep the LR happy
        for k, v in list(out.items()):
            if v is None or (isinstance(v, float) and (math.isnan(v) or math.isinf(v))):
                out[k] = 0.0
        return out

    # ── Layer 1: S5 tiered hard rule ──────────────────────────────────────
    def layer1_keep(self, feats: Dict[str, float]) -> (bool, str):
        T = S5_THRESHOLDS
        mm = feats.get('mean_motion', 0.0) or 0.0
        # Tier 1: quiet auto-keep
        if mm < T['quiet_motion_max']:
            return True, 'quiet_motion'
        # Tier 2: tachycardic AFib path
        if (feats.get('mean_rr', 0.0) < T['tachy_mean_rr_max']
                and feats.get('kurt', 9999.0) < T['tachy_kurt_max']
                and mm < T['tachy_motion_max']):
            return True, 'tachy_afib'
        # Tier 3: bradycardic AFib rescue
        if (feats.get('mean_rr', 9999.0) < T['rescue_mean_rr_max']
                and feats.get('samp_en', 0.0) >= T['rescue_samp_en_min']
                and feats.get('p_present_pct', 100.0) <= T['rescue_p_max']):
            return True, 'brady_rescue'
        return False, 'layer1_reject'

    # ── Layer 2: S6 logistic-regression ranker ────────────────────────────
    def layer2_score(self, feats: Dict[str, float]) -> float:
        if not self.use_layer2:
            return 0.0
        x = np.array([float(feats.get(k, 0.0) or 0.0) for k in FEATURE_ORDER],
                     dtype=np.float64)
        xs = (x - self._mean) / np.where(self._scale == 0, 1, self._scale)
        z = float(np.dot(xs, self._coef) + self._intercept)
        # sigmoid
        return 1.0 / (1.0 + math.exp(-z))

    # ── Top-level decision ────────────────────────────────────────────────
    def suppress(self, p_afib: float, json_path: str,
                 layer2_threshold: Optional[float] = None,
                 require_both_layers: bool = True) -> SuppressionResult:
        """
        Apply the full suppression pipeline to a single event.

        Parameters
        ----------
        p_afib  : model AFib probability for this event (sigmoid output).
        json_path : path to the original JSON record (for feature extraction).
        layer2_threshold : override default LR cutoff.
        require_both_layers : if True (default), event is kept only if BOTH
            layers pass; if False, EITHER passing is enough (more permissive).

        Returns
        -------
        SuppressionResult
        """
        t2 = layer2_threshold if layer2_threshold is not None else self.layer2_threshold
        feats = self.extract_features(json_path)
        if feats is None:
            # Cannot extract features → fail open (keep) but flag reason
            return SuppressionResult(
                final_prob=float(p_afib), keep=True,
                layer1_pass=True, layer2_pass=True, layer2_score=0.0,
                reason='no_features', features=None)

        l1_pass, l1_reason = self.layer1_keep(feats)
        l2_score = self.layer2_score(feats)
        l2_pass  = (l2_score < t2) if self.use_layer2 else True

        if require_both_layers:
            keep = l1_pass and l2_pass
        else:
            keep = l1_pass or l2_pass

        reason = ''
        if keep:
            reason = l1_reason
        else:
            if not l1_pass and not l2_pass:
                reason = f'both_layers_reject (l1={l1_reason}, l2_p={l2_score:.3f})'
            elif not l1_pass:
                reason = f'layer1_reject (l2_p={l2_score:.3f})'
            else:
                reason = f'layer2_reject (l1={l1_reason}, p={l2_score:.3f})'

        return SuppressionResult(
            final_prob=float(p_afib) if keep else 0.0,
            keep=keep,
            layer1_pass=l1_pass,
            layer2_pass=l2_pass,
            layer2_score=l2_score,
            reason=reason,
            features=feats,
        )

    # ── batch helper ──────────────────────────────────────────────────────
    def suppress_batch(self, probs, json_paths, **kw):
        return [self.suppress(float(p), jp, **kw) for p, jp in zip(probs, json_paths)]
