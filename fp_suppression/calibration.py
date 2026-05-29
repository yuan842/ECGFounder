"""Per-device threshold calibration — the "training" for each algorithm.

Each algorithm is trained SEPARATELY and parameters are DEVICE-BASED: you fit a
device's motion gates on (motion-feature, TP/FP-label) pairs from that device,
and its SQI gates on (ECG-feature, label) pairs — two independent jobs that need
two different data streams.

Two calibration modes:

  fit_gate_supervised(values, is_tp, feature, op)
      Needs labels (TP vs FP). Sweeps candidate thresholds and picks the one
      maximising Youden's J for the KEEP decision: J = keep-rate(TP) + drop-rate(FP) − 1.
      Works for both gate directions (op '<='/'>=').

  fit_gate_envelope(rest_values, feature, op, pct)
      Label-free. Uses the resting-subject feature envelope: a window with more
      <feature> than a resting subject ever shows is contaminated. For a
      suppress-high gate ('<=') the threshold is the rest pct percentile (the
      method in res/move_eval/MOTION_GATE_RECALIBRATION.md). Inverted gates
      ('>=', TP has HIGH feature) cannot be fit from rest alone → needs a TP cohort.

Both return a fitted `Gate`. `calibrate_profile_event` writes it into a profile's
motion or sqi table (chosen by the gate's family), leaving the other family
untouched.
"""
from __future__ import annotations
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .gates import Gate, _OPS
from .device_profiles import DeviceProfile


def _keep_mask(values: np.ndarray, op: str, thr: float) -> np.ndarray:
    fn = _OPS[op]
    return np.array([fn(v, thr) for v in values], dtype=bool)


def fit_gate_supervised(values: Sequence[float], is_tp: Sequence[bool],
                        feature: str, op: str,
                        grid: int = 200) -> Tuple[Gate, Dict[str, float]]:
    """Find the threshold maximising Youden's J for the keep decision.

    keep == op(value, thr). Good gate keeps TP, drops FP:
        J = (#TP kept / #TP) + (#FP dropped / #FP) − 1
    Returns (fitted_gate, stats).
    """
    v = np.asarray(values, dtype=float)
    y = np.asarray(is_tp, dtype=bool)
    ok = np.isfinite(v)
    v, y = v[ok], y[ok]
    if v.size == 0 or y.sum() == 0 or (~y).sum() == 0:
        raise ValueError("need finite values with both TP and FP present")

    lo, hi = float(np.min(v)), float(np.max(v))
    cands = np.unique(np.concatenate([v, np.linspace(lo, hi, grid)]))
    n_tp, n_fp = int(y.sum()), int((~y).sum())

    best = None
    for thr in cands:
        keep = _keep_mask(v, op, thr)
        tp_keep = int((keep & y).sum())
        fp_drop = int((~keep & ~y).sum())
        j = tp_keep / n_tp + fp_drop / n_fp - 1.0
        if best is None or j > best[1]:
            best = (thr, j, tp_keep, fp_drop)
    thr, j, tp_keep, fp_drop = best
    stats = dict(youden_j=j, n_tp=n_tp, n_fp=n_fp,
                 tp_retained=tp_keep / n_tp, fp_removed=fp_drop / n_fp,
                 threshold=float(thr))
    return Gate(feature, op, float(thr)), stats


def fit_gate_envelope(rest_values: Sequence[float], feature: str, op: str,
                      pct: float = 99.0) -> Tuple[Gate, Dict[str, float]]:
    """Label-free fit from the resting-subject envelope.

    For a suppress-high gate (op '<='): threshold = percentile(rest, pct).
    Inverted gates ('>=') are not fittable from rest alone (TP has high feature).
    """
    v = np.asarray(rest_values, dtype=float)
    v = v[np.isfinite(v)]
    if v.size == 0:
        raise ValueError("no finite rest values")
    if op not in ('<=', '<'):
        raise ValueError(f"envelope fit only supports suppress-high gates "
                         f"('<='/'<'); {op!r} (inverted) needs a TP cohort")
    thr = float(np.percentile(v, pct))
    stats = dict(threshold=thr, rest_n=int(v.size),
                 rest_p95=float(np.percentile(v, 95)),
                 rest_p99=float(np.percentile(v, 99)),
                 rest_median=float(np.median(v)))
    return Gate(feature, op, thr), stats


def calibrate_profile_event(profile: DeviceProfile, event: str, gate: Gate,
                            *, replace_feature: bool = True) -> DeviceProfile:
    """Write a fitted gate into the right family table of `profile` (in place).

    replace_feature=True swaps any existing gate on the same feature; otherwise
    appends. The other family's table is never touched.
    """
    table = profile.motion_gates if gate.family() == 'motion' else profile.sqi_gates
    existing = table.get(event, [])
    if replace_feature:
        existing = [g for g in existing if g.feature != gate.feature]
    table[event] = existing + [gate]
    profile.validate()
    return profile
