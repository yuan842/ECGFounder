"""
multiclass_fp_suppression.py  (v2)
===================================

FP-suppression filter based on simple, data-driven feature gates.

Design doc: ``fp_suppression_filter_v2.md``
Data source: ``SQI_MOTION_COMPARISON_REPORT.md``

Active rules (4 classes):
    Atrial Fibrillation        → mean_motion <= 5.0 mG           (max TP retention)
    Bradycardia                → mean_hr_bpm <= 56.3 bpm         (max TP retention)
    Supraventricular Trigeminy → mean_motion >= 15.0 mG          (balanced)
    Ventricular Trigeminy      → mean_motion >= 24.0 AND snr > 1.2  (balanced)

All other classes pass through unsuppressed.

Logic: AND across all conditions per class.
       Missing features → fail-open (PASS).

Public API  (backward-compatible with v1)
-----------------------------------------
    MultiClassFPSuppressor()
        .suppress_alert(alert_class, p, json_path) -> SuppressionResult
        .supported_classes() -> list[str]

Imports kept from v1 for consumers:
    CLASS_INDEX, INDEX_TO_CLASS, SuppressionResult
"""
from __future__ import annotations
import json
import math
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# ─── Re-export SuppressionResult for backward compatibility ───────────────
# v1 consumers do:  from multiclass_fp_suppression import SuppressionResult
from afib_fp_suppression import SuppressionResult  # noqa: F401


# ═══════════════════════════════════════════════════════════════════════════
# Constants
# ═══════════════════════════════════════════════════════════════════════════

FS_DEFAULT       = 128
ACC_SCALE_FACTOR = 2048.0
DC_REMOVAL_WINDOW = 5
MOTION_UNITS     = 1000.0          # G → mG

# v2 suppressed event types  (no model-head dependency — feature gates only)
# CLASS_INDEX kept for backward compatibility with comparison scripts.
# Only AFib has a known 150-class model head (index 5); the others are
# suppressed by feature gates and don't need model-head routing.
CLASS_INDEX = {
    'Atrial Fibrillation':              5,
    'Bradycardia':                     -1,   # no dedicated model head
    'Supraventricular Trigeminy':      -1,   # no dedicated model head
    'Ventricular Trigeminy':           -1,   # no dedicated model head
}
INDEX_TO_CLASS = {v: k for k, v in CLASS_INDEX.items() if v >= 0}


# ═══════════════════════════════════════════════════════════════════════════
# Rule registry  (from fp_suppression_filter_v2.md §4)
# ═══════════════════════════════════════════════════════════════════════════

ACTIVE_RULES: Dict[str, List[Tuple[str, str, float]]] = {
    # Tier: MAX_TP_RETENTION
    'Atrial Fibrillation': [
        ('mean_motion', '<=', 5.0),
    ],
    # Tier: MAX_TP_RETENTION
    'Bradycardia': [
        ('mean_hr_bpm', '<=', 56.3),
    ],
    # Tier: BALANCED  (inverted motion gate — TP has HIGH motion)
    'Supraventricular Trigeminy': [
        ('mean_motion', '>=', 15.0),
    ],
    # Tier: BALANCED  (inverted motion + SNR gate)
    'Ventricular Trigeminy': [
        ('mean_motion', '>=', 24.0),
        ('snr_proxy',   '>',  1.2),
    ],
}

PASSTHROUGH_CLASSES = {
    'Unknown',
    'ST Elevation',
    'Supraventricular Bigeminy',
    'Ventricular Bigeminy',
    'Ventricular Tachycardia',
    'Sinus Tachycardia',
    'Custom Heart Rate',
    # Classes pending rule development:
    'Isolated Supraventricular Beat',
    'Isolated Ventricular Beat',
    'Multiple Event',
    'Pause',
    'Prolonged RR Interval',
    'Ventricular Couplet',
    'Supraventricular Couplet',
    'Supraventricular Run',
    'Ventricular Run',
}


# ═══════════════════════════════════════════════════════════════════════════
# Feature extraction  (self-contained — no dependency on old suppressor)
# ═══════════════════════════════════════════════════════════════════════════

try:
    from scipy.signal import butter, filtfilt, find_peaks
    from scipy.stats import kurtosis as scipy_kurtosis
    _HAS_SCIPY = True
except ImportError:
    _HAS_SCIPY = False


def _bandpass(sig, fs, lo=5.0, hi=20.0, order=2):
    nyq = fs / 2.0
    b, a = butter(order, [lo / nyq, hi / nyq], btype='band')
    return filtfilt(b, a, sig)


def _detect_r_peaks(sig, fs):
    if not _HAS_SCIPY or len(sig) < fs * 2:
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


def extract_features(json_path: str) -> Optional[Dict[str, float]]:
    """
    Extract the minimal feature set needed by v2 rules:
      mean_motion, snr_proxy, mean_hr_bpm

    Also returns max_motion, std_motion, kurt, baseline_drift for logging.
    Returns None only if the JSON cannot be loaded at all.
    """
    try:
        with open(json_path) as f:
            data = json.load(f)
    except Exception:
        return None
    if not isinstance(data, list) or len(data) == 0:
        return None

    # ── Motion from accelerometer ──────────────────────────────────────────
    acc_rows = []
    ecg_chunks = []
    fs = FS_DEFAULT
    mag = 1000
    for item in data:
        if not isinstance(item, dict):
            continue
        d = item.get('data', {})
        acc = d.get('acc')
        if acc:
            for a in acc:
                if isinstance(a, dict):
                    acc_rows.append([a.get('x', 0), a.get('y', 0), a.get('z', 0)])
                elif isinstance(a, (list, tuple)) and len(a) >= 3:
                    acc_rows.append([a[0], a[1], a[2]])
        ecg = d.get('ecg')
        if ecg is not None:
            fs = d.get('sf', fs) or fs
            mag = d.get('magnification', mag) or mag
            ecg_chunks.extend(ecg)

    out: Dict[str, float] = {}

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
        out['mean_motion'] = float(np.mean(dyn))
        out['max_motion']  = float(np.max(dyn))
        out['std_motion']  = float(np.std(dyn))

    # ── ECG-based features (SNR, HR) ──────────────────────────────────────
    sig = np.asarray(ecg_chunks, dtype=float) / float(mag) if ecg_chunks else np.array([])

    if len(sig) >= fs * 3 and _HAS_SCIPY:
        # SNR proxy
        try:
            bp = _bandpass(sig, fs, 5.0, 25.0)
            out['snr_proxy'] = float(np.var(bp) / (np.var(sig - bp) + 1e-9))
        except Exception:
            pass
        # Kurtosis + baseline drift (for logging)
        try:
            out['kurt'] = float(scipy_kurtosis(sig, fisher=True))
        except Exception:
            pass
        try:
            b, a = butter(2, 0.5 / (fs / 2), btype='low')
            out['baseline_drift'] = float(np.std(filtfilt(b, a, sig)))
        except Exception:
            pass
        # HR from R-peaks
        peaks = _detect_r_peaks(sig, fs)
        if len(peaks) >= 2:
            rr_ms = np.diff(peaks) / fs * 1000.0
            if len(rr_ms) >= 3:
                mean_rr = float(np.mean(rr_ms))
                if mean_rr > 0:
                    out['mean_hr_bpm'] = 60000.0 / mean_rr

    # Sanitize
    for k, v in list(out.items()):
        if isinstance(v, float) and (math.isinf(v) or math.isnan(v)):
            del out[k]  # remove rather than set NaN — treat as missing

    return out


# ═══════════════════════════════════════════════════════════════════════════
# Gate evaluation engine
# ═══════════════════════════════════════════════════════════════════════════

_OPS = {
    '<':  lambda v, t: v < t,
    '<=': lambda v, t: v <= t,
    '>':  lambda v, t: v > t,
    '>=': lambda v, t: v >= t,
    '==': lambda v, t: v == t,
}


def _evaluate_condition(value: float, operator: str, threshold: float) -> bool:
    fn = _OPS.get(operator)
    if fn is None:
        raise ValueError(f"Unsupported operator: {operator!r}")
    return fn(value, threshold)


def _apply_rules(
    event_type: str,
    features: Dict[str, float],
) -> Tuple[bool, str, List[Tuple]]:
    """
    Evaluate the v2 rule set for an event type.

    Returns (passed, reason, failed_conditions).
    """
    if event_type in PASSTHROUGH_CLASSES or event_type not in ACTIVE_RULES:
        return True, 'passthrough', []

    rules = ACTIVE_RULES[event_type]
    failed = []
    missing = []

    for feat_name, op, thresh in rules:
        if feat_name not in features:
            missing.append(feat_name)
            continue  # fail-open
        val = features[feat_name]
        if not _evaluate_condition(val, op, thresh):
            failed.append((feat_name, op, thresh, val))

    if not failed:
        if missing:
            return True, f'pass (missing: {",".join(missing)})', []
        return True, 'pass_all_gates', []

    # Build reason string
    parts = []
    for feat_name, op, thresh, val in failed:
        parts.append(f'{feat_name}={val:.2f} failed {op}{thresh}')
    reason = '; '.join(parts)
    return False, reason, failed


# ═══════════════════════════════════════════════════════════════════════════
# Main suppressor class  (backward-compatible API)
# ═══════════════════════════════════════════════════════════════════════════

class MultiClassFPSuppressor:
    """
    v2 FP suppressor using simple feature gates.

    Drop-in replacement for the v1 MultiClassFPSuppressor.
    Constructor arguments from v1 are accepted but ignored (no LR ranker).
    """

    def __init__(self, **kwargs):
        # Accept and ignore v1 constructor args for backward compatibility
        pass

    def supported_classes(self) -> list:
        """Return the list of event types with an active suppression rule."""
        return list(ACTIVE_RULES.keys())

    def suppress_alert(
        self,
        alert_class: str | int,
        p: float,
        json_path: str,
    ) -> SuppressionResult:
        """
        Apply the v2 feature-gate filter for the given alert class.

        Parameters
        ----------
        alert_class : event-type string ('Atrial Fibrillation') or
                      a 150-class integer index.
        p           : model probability for that class.
        json_path   : path to the event JSON record.

        Returns
        -------
        SuppressionResult  (same dataclass as v1 — backward compatible)
        """
        # Resolve class name
        if isinstance(alert_class, int):
            cls = INDEX_TO_CLASS.get(alert_class)
        else:
            cls = alert_class

        # No rule for this class → pass through
        if cls is None or cls in PASSTHROUGH_CLASSES or cls not in ACTIVE_RULES:
            return SuppressionResult(
                final_prob=float(p), keep=True,
                layer1_pass=True, layer2_pass=True, layer2_score=0.0,
                reason='passthrough', features=None,
            )

        # Extract features
        features = extract_features(json_path)
        if features is None:
            return SuppressionResult(
                final_prob=float(p), keep=True,
                layer1_pass=True, layer2_pass=True, layer2_score=0.0,
                reason='no_features (fail-open)', features=None,
            )

        # Evaluate gates
        passed, reason, failed = _apply_rules(cls, features)

        return SuppressionResult(
            final_prob=float(p) if passed else 0.0,
            keep=passed,
            layer1_pass=passed,       # v2 has no layer distinction
            layer2_pass=True,         # no LR ranker in v2
            layer2_score=0.0,
            reason=reason,
            features=features,
        )
