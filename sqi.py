"""sqi.py — comprehensive per-window ECG Signal Quality Index.

Consolidates the quality metrics previously scattered in
`ecg_feature_analysis.py` / `compare_tp_fp_sqi_motion.py`, with three
corrections from the MOVE validation (docs/SQI_DESIGN.md §9):

  Fix 1  clip_pct is rail-based + flat-gated (no longer mis-fires on flat
         signals or on isolated R-peak tips).
  Fix 2  gravity-removed dynamic motion, matching the production FP feature
         `multiclass_fp_suppression.extract_features` exactly — so MOVE /
         wearable motion is directly comparable to the deployed motion gates.
         (NOT std-of-magnitude — that was a mis-diagnosis.)
  Fix 3  two composites:
           sqi_score_ecg  — ECG-intrinsic only (clinical resting data)
           sqi_score_amb  — ECG-intrinsic penalized by gravity-removed motion
                            from a synchronized accelerometer (wearable data)

Stage discipline (docs/SQI_DESIGN.md §2): amplitude / clip / flat / baseline
metrics are computed on the RAW lead; SNR / morphology / rhythm metrics on a
filtered copy. Never on the final winsorized+z-scored tensor.
"""
from __future__ import annotations
import numpy as np
from scipy.signal import butter, filtfilt, find_peaks, welch
from scipy.stats import kurtosis

# Production motion constants (mirror multiclass_fp_suppression.py)
_DC_REMOVAL_WINDOW = 5
_MOTION_UNITS = 1000.0           # g → mG
_MIN_DYN_RANGE_MV = 0.05         # below this the window is "flat", not clipping
_RAIL_RUN = 3                    # consecutive at-rail samples to count as clipping


# ─── motion (fix 2 — matches production exactly) ────────────────────────────
def gravity_removed_motion(acc_xyz: np.ndarray, w: int = _DC_REMOVAL_WINDOW) -> dict:
    """acc_xyz: (3, T) in g. Returns gravity-removed dynamic motion in mG,
    identical to `multiclass_fp_suppression.extract_features`'s `mean_motion`."""
    acc = np.asarray(acc_xyz, dtype=float)
    vm = np.sqrt((acc * acc).sum(axis=0))            # magnitude incl. gravity
    if len(vm) >= w:
        baseline = np.convolve(vm, np.ones(w) / w, mode="same")   # local DC ≈ gravity
        dyn = np.abs(vm - baseline) * _MOTION_UNITS
    else:
        dyn = np.abs(vm - vm.mean()) * _MOTION_UNITS
    return dict(mean_motion_mg=float(dyn.mean()),
                std_motion_mg=float(dyn.std()),
                max_motion_mg=float(dyn.max()))


def _bandpass(sig, fs, lo, hi, order=2):
    b, a = butter(order, [lo / (fs / 2), hi / (fs / 2)], btype="bandpass")
    return filtfilt(b, a, sig)


# ─── raw-stage SQI (amplitude / saturation / flat / baseline / powerline) ───
def compute_raw_sqi(sig: np.ndarray, fs: int) -> dict:
    sig = np.asarray(sig, dtype=float)
    out = {}
    rng = float(sig.max() - sig.min())
    out["dynamic_range_mv"] = rng

    # flat-line / lead-off: rolling 100 ms std < eps
    w = max(1, int(0.1 * fs))
    if len(sig) >= w:
        roll = np.lib.stride_tricks.sliding_window_view(sig, w).std(axis=1)
        out["flat_pct"] = float(np.mean(roll < 1e-4) * 100)
    else:
        out["flat_pct"] = 100.0

    # FIX 1: clip = sustained rail-sticking, flat-gated and run-length-gated.
    #   - if the window is essentially flat (tiny range) it is NOT clipping → 0
    #   - count only samples within tol of the rail that occur in runs of >=3
    #     (isolated R-peak tips are excluded; true saturation forms plateaus)
    if rng < _MIN_DYN_RANGE_MV:
        out["clip_pct"] = 0.0
    else:
        tol = 0.005 * rng
        at_rail = (sig >= sig.max() - tol) | (sig <= sig.min() + tol)
        # keep only at-rail samples that belong to a run of length >= _RAIL_RUN
        clip_mask = np.zeros_like(at_rail)
        run = 0
        for i, v in enumerate(at_rail):
            if v:
                run += 1
            else:
                if run >= _RAIL_RUN:
                    clip_mask[i - run:i] = True
                run = 0
        if run >= _RAIL_RUN:
            clip_mask[len(at_rail) - run:] = True
        out["clip_pct"] = float(np.mean(clip_mask) * 100)

    # baseline drift: std of <0.5 Hz envelope (on raw — stage-4 would remove it)
    try:
        b, a = butter(2, 0.5 / (fs / 2), btype="low")
        out["baseline_drift"] = float(np.std(filtfilt(b, a, sig)))
    except Exception:
        out["baseline_drift"] = np.nan

    # HF noise ratio: power >40 Hz / total
    try:
        f, p = welch(sig, fs=fs, nperseg=min(1024, len(sig)))
        out["hf_noise_ratio"] = float(p[f > 40].sum() / (p.sum() + 1e-12))
    except Exception:
        out["hf_noise_ratio"] = np.nan
    return out


# ─── filtered-stage SQI (SNR / morphology / rhythm) ─────────────────────────
def compute_band_sqi(sig: np.ndarray, fs: int) -> dict:
    sig = np.asarray(sig, dtype=float)
    out = {}
    try:
        bp = _bandpass(sig, fs, 5.0, 25.0)
        out["snr_proxy"] = float(np.var(bp) / (np.var(sig - bp) + 1e-9))
    except Exception:
        out["snr_proxy"] = np.nan
    try:
        out["kurt"] = float(kurtosis(sig, fisher=True)) if np.std(sig) > 1e-6 else 0.0
    except Exception:
        out["kurt"] = np.nan
    # rhythm plausibility
    try:
        qrs = _bandpass(sig, fs, 5.0, 20.0)
        pk, _ = find_peaks(qrs, distance=int(0.3 * fs), height=np.std(qrs))
        out["n_peaks"] = int(len(pk))
        if len(pk) >= 2:
            rr = np.diff(pk) / fs
            hr = 60.0 / rr
            out["mean_hr_bpm"] = float(np.median(hr))
            out["rr_cv"] = float(np.std(rr) / (np.mean(rr) + 1e-9))
            out["pct_physiologic_hr"] = float(np.mean((hr >= 40) & (hr <= 200)) * 100)
        else:
            out["mean_hr_bpm"] = np.nan
            out["rr_cv"] = np.nan
            out["pct_physiologic_hr"] = 0.0
    except Exception:
        out.update(n_peaks=0, mean_hr_bpm=np.nan, rr_cv=np.nan, pct_physiologic_hr=np.nan)
    return out


# ─── composites (fix 3) ─────────────────────────────────────────────────────
def _clamp(v):
    return float(max(0.0, min(1.0, v)))


def sqi_score_ecg(panel: dict) -> float:
    """ECG-intrinsic composite ∈ [0,1] — for clinical resting data (no ACC)."""
    snr  = _clamp(np.tanh((panel.get("snr_proxy") or 0) / 3.0))
    clip = _clamp(1 - (panel.get("clip_pct") or 0) / 100)
    flat = _clamp(1 - (panel.get("flat_pct") or 0) / 100)
    kurt = _clamp((panel.get("kurt") or 0) / 10.0)
    hr   = _clamp((panel.get("pct_physiologic_hr") or 0) / 100)
    return _clamp(0.30 * snr + 0.20 * clip + 0.20 * flat + 0.15 * kurt + 0.15 * hr)


def sqi_score_amb(panel: dict, mean_motion_mg: float) -> float:
    """Motion-fused composite ∈ [0,1] — for wearable data with a synchronized
    accelerometer. Starts from the ECG-intrinsic score and penalizes motion.
    Penalty: linear ramp, full penalty by ~50 mG gravity-removed dynamic motion."""
    base = sqi_score_ecg(panel)
    if mean_motion_mg is None or np.isnan(mean_motion_mg):
        return base
    motion_penalty = _clamp(mean_motion_mg / 50.0)       # 0 mG→0 … 50 mG→1
    return _clamp(base * (1.0 - 0.5 * motion_penalty))   # motion can halve the score


def sqi_class(score: float) -> str:
    return "good" if score >= 0.7 else "acceptable" if score >= 0.4 else "poor"


# ─── one-call panel ─────────────────────────────────────────────────────────
def compute_sqi(sig: np.ndarray, fs: int,
                acc_xyz: np.ndarray | None = None) -> dict:
    """Full per-window SQI panel. Pass acc_xyz (3,T) to enable the motion-fused
    composite (wearable data); omit it for clinical resting ECG."""
    panel = {}
    panel.update(compute_raw_sqi(sig, fs))          # raw-stage taps
    panel.update(compute_band_sqi(sig, fs))         # filtered-stage taps
    panel["sqi_score_ecg"] = sqi_score_ecg(panel)
    if acc_xyz is not None:
        mot = gravity_removed_motion(acc_xyz)
        panel.update(mot)
        panel["sqi_score_amb"] = sqi_score_amb(panel, mot["mean_motion_mg"])
        panel["sqi_class"] = sqi_class(panel["sqi_score_amb"])
    else:
        panel["sqi_class"] = sqi_class(panel["sqi_score_ecg"])
    return panel
