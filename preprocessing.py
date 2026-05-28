"""
preprocessing.py — Unified ECG preprocessing for the 1-lead ECGFounder.

Replaces three older code paths:
  - inline `robust_preprocess()` in eval scripts (fzark / ECG-FP / JSON)
  - `UniversalECGAdapter` (PTB-XL, MIT-BIH; plain z-score)
  - `RobustECGAdapter` (MIT-BIH; winsorized z-score)

Pipeline (in order):
    load → select_lead → notch → bandpass → baseline → resample
         → crop_pad → normalize → torch.float32 (1, target_len)

Public surface:

    ECGPreprocessor(target_fs=500, target_len=5000, ...)
    ECGPreprocessor.for_fzark()    # 128 Hz, EU 50 Hz
    ECGPreprocessor.for_ptbxl()    # 500 Hz, EU 50 Hz
    ECGPreprocessor.for_mitdb()    # 360 Hz, US 60 Hz

    prep.process(signal, fs_in, source_leads=None) -> torch.Tensor
    prep.from_fzark_json(path)                     -> torch.Tensor
    prep.from_wfdb(record_path)                    -> torch.Tensor
"""
from __future__ import annotations

import json
import os
from functools import lru_cache
from typing import Optional, Sequence, Tuple

import numpy as np
import torch
from scipy.interpolate import interp1d
from scipy.signal import butter, filtfilt, iirnotch, medfilt

# ── Module-level constants (single source of truth) ─────────────────────────

TARGET_FS: int            = 500       # ECGFounder native sampling rate (Hz)
TARGET_LEN: int           = 5000      # 10 s @ 500 Hz
DEFAULT_BANDPASS: Tuple[float, float] = (0.67, 40.0)
NOTCH_Q: float            = 30.0
BASELINE_WINDOW_S: float  = 0.4
WIN_LIMITS: Tuple[float, float] = (1.5, 98.5)
FZARK_MAGNIFICATION: int  = 1000      # int → mV scale for Vivalink JSON


# ── Cached filter coefficient builders ──────────────────────────────────────
# Coefficients depend only on fs_in (and constants). Cache to avoid recomputing
# on every call. lru_cache returns the same tuple each time — values are
# read-only numpy arrays, safe to share across threads.

@lru_cache(maxsize=16)
def _notch_coeffs(fs_in: int, powerline_hz: int) -> Tuple[np.ndarray, np.ndarray]:
    return iirnotch(powerline_hz, NOTCH_Q, fs_in)


@lru_cache(maxsize=16)
def _bandpass_coeffs(fs_in: int,
                     lo: float = DEFAULT_BANDPASS[0],
                     hi: float = DEFAULT_BANDPASS[1]) -> Tuple[np.ndarray, np.ndarray]:
    return butter(N=4, Wn=[lo, hi], btype='bandpass', fs=fs_in)


def _odd_kernel(fs_in: int, window_s: float) -> int:
    """Return the nearest-or-larger odd kernel size for a given window."""
    k = int(window_s * fs_in)
    return k if (k % 2 == 1) else k + 1


# ═══════════════════════════════════════════════════════════════════════════
# ECGPreprocessor
# ═══════════════════════════════════════════════════════════════════════════

class ECGPreprocessor:
    """Unified preprocessing for the 1-lead ECGFounder model."""

    def __init__(
        self,
        target_fs: int = TARGET_FS,
        target_len: int = TARGET_LEN,
        target_lead: str = "II",
        powerline_hz: int = 50,
        apply_notch: bool = True,
        apply_bandpass: bool = True,
        bandpass_hz: Tuple[float, float] = DEFAULT_BANDPASS,
        apply_baseline: bool = True,
        baseline_window_s: float = BASELINE_WINDOW_S,
        normalize: str = "winsorize",          # "winsorize" | "zscore" | "none"
        win_limits: Tuple[float, float] = WIN_LIMITS,
    ):
        if normalize not in ("winsorize", "zscore", "none"):
            raise ValueError(
                f"normalize must be one of 'winsorize', 'zscore', 'none'; "
                f"got {normalize!r}"
            )
        if powerline_hz not in (50, 60):
            raise ValueError(f"powerline_hz must be 50 or 60; got {powerline_hz}")

        self.target_fs         = target_fs
        self.target_len        = target_len
        self.target_lead       = target_lead
        self.powerline_hz      = powerline_hz
        self.apply_notch       = apply_notch
        self.apply_bandpass    = apply_bandpass
        self.bandpass_hz       = bandpass_hz
        self.apply_baseline    = apply_baseline
        self.baseline_window_s = baseline_window_s
        self.normalize         = normalize
        self.win_limits        = win_limits

    # ─── Factory methods ────────────────────────────────────────────────────

    @classmethod
    def for_fzark(cls) -> "ECGPreprocessor":
        """fzark / ECG-FP / Vivalink Holter recordings @ 128 Hz, EU 50 Hz grid."""
        return cls(powerline_hz=50, normalize="winsorize")

    @classmethod
    def for_ptbxl(cls) -> "ECGPreprocessor":
        """PTB-XL @ 500 Hz native, German 50 Hz grid.

        Apply full filtering for consistency with the training-time
        preprocessing manifold (historically PTB-XL eval skipped filtering).
        """
        return cls(powerline_hz=50, normalize="winsorize")

    @classmethod
    def for_mitdb(cls) -> "ECGPreprocessor":
        """MIT-BIH @ 360 Hz native, US 60 Hz grid. Uses MLII as Lead II."""
        return cls(powerline_hz=60, target_lead="II", normalize="winsorize")

    # ─── Core pipeline ──────────────────────────────────────────────────────

    def process(
        self,
        signal: np.ndarray,
        fs_in: int,
        source_leads: Optional[Sequence[str]] = None,
    ) -> torch.Tensor:
        """Run the full preprocessing pipeline.

        Args:
            signal: 1-D (N,) or 2-D (C, N) numpy array in physical units (mV).
            fs_in: Input sampling rate (Hz).
            source_leads: Required when signal has >1 channel. Length must
                          equal signal.shape[0]. Used to pick `self.target_lead`.

        Returns:
            torch.float32 tensor of shape (1, target_len).
        """
        x = np.asarray(signal, dtype=np.float64)   # filtfilt prefers float64

        # 1. Lead selection — produce a 1-D signal
        x_1d = self._select_lead(x, source_leads)

        # 2. Notch
        if self.apply_notch:
            b, a = _notch_coeffs(int(fs_in), self.powerline_hz)
            x_1d = filtfilt(b, a, x_1d)

        # 3. Bandpass
        if self.apply_bandpass:
            b, a = _bandpass_coeffs(int(fs_in), *self.bandpass_hz)
            x_1d = filtfilt(b, a, x_1d)

        # 4. Median baseline removal
        if self.apply_baseline:
            k = _odd_kernel(int(fs_in), self.baseline_window_s)
            baseline = medfilt(x_1d, kernel_size=k)
            x_1d = x_1d - baseline

        # 5. Resample to target_fs
        x_1d = self._resample(x_1d, fs_in)

        # 6. Center-crop or zero-pad to target_len
        x_1d = self._crop_or_pad(x_1d)

        # 7. Normalize
        x_1d = self._normalize(x_1d)

        return torch.from_numpy(x_1d.astype(np.float32))[None, :]

    # ─── Stage helpers (private) ────────────────────────────────────────────

    def _select_lead(
        self,
        signal: np.ndarray,
        source_leads: Optional[Sequence[str]],
    ) -> np.ndarray:
        """Reduce multi-channel input to a single 1-D channel."""
        if signal.ndim == 1:
            return signal
        if signal.ndim != 2:
            raise ValueError(f"signal must be 1-D or 2-D; got shape {signal.shape}")

        n_ch = signal.shape[0]
        if n_ch == 1:
            return signal[0]

        if source_leads is None:
            raise ValueError(
                f"signal has {n_ch} channels but source_leads is None; "
                f"cannot determine which one to use as '{self.target_lead}'"
            )
        if len(source_leads) != n_ch:
            raise ValueError(
                f"source_leads has {len(source_leads)} entries but signal has "
                f"{n_ch} channels"
            )

        target_norm = self.target_lead.lower().strip()
        source_norm = [str(l).lower().strip() for l in source_leads]

        # Exact match first
        if target_norm in source_norm:
            return signal[source_norm.index(target_norm)]

        # MIT-BIH compatibility: MLII == Lead II
        if target_norm == "ii" and "mlii" in source_norm:
            return signal[source_norm.index("mlii")]

        raise ValueError(
            f"target_lead {self.target_lead!r} not found in source_leads "
            f"{list(source_leads)}"
        )

    def _resample(self, x: np.ndarray, fs_in: int) -> np.ndarray:
        if fs_in == self.target_fs:
            return x
        t_total = len(x) / fs_in
        n_out   = int(t_total * self.target_fs)
        x_old   = np.linspace(0.0, t_total, num=len(x), endpoint=True)
        x_new   = np.linspace(0.0, t_total, num=n_out, endpoint=True)
        f       = interp1d(x_old, x, kind='linear', fill_value='extrapolate')
        return f(x_new)

    def _crop_or_pad(self, x: np.ndarray) -> np.ndarray:
        n = len(x)
        if n == self.target_len:
            return x
        if n > self.target_len:
            start = (n - self.target_len) // 2
            return x[start : start + self.target_len]
        # Pad symmetrically with zeros
        out = np.zeros(self.target_len, dtype=x.dtype)
        pad = (self.target_len - n) // 2
        out[pad : pad + n] = x
        return out

    def _normalize(self, x: np.ndarray) -> np.ndarray:
        if self.normalize == "none":
            return x
        if self.normalize == "winsorize":
            lo, hi = np.percentile(x, self.win_limits)
            x = np.clip(x, lo, hi)
        # plain z-score uses the (possibly clipped) signal
        mean = float(np.mean(x))
        std  = float(np.std(x))
        return (x - mean) / (std + 1e-8)

    # ─── Format-specific loaders ────────────────────────────────────────────

    def from_fzark_json(self, json_path: str) -> torch.Tensor:
        """Load a Vivalink-style JSON sidecar and run the pipeline.

        Returns float32 tensor (1, target_len). Native fs = 128 Hz.
        """
        with open(json_path) as f:
            records = json.load(f)
        raw = np.concatenate(
            [np.array(r['data']['ecg'], dtype=np.float32) for r in records]
        )
        raw = raw / FZARK_MAGNIFICATION   # → mV
        return self.process(raw, fs_in=128)

    def from_wfdb(self, record_path: str) -> torch.Tensor:
        """Load a WFDB record and run the pipeline.

        `record_path` is the prefix (without `.dat`/`.hea`). Reads the
        sampling rate and signal names from the WFDB header.
        """
        import wfdb   # local import — WFDB datasets only
        sig, meta = wfdb.rdsamp(record_path)        # sig: (N, C) physical units
        if sig.ndim == 1:
            sig = sig[:, None]
        return self.process(
            sig.T,                                  # → (C, N)
            fs_in=int(meta['fs']),
            source_leads=meta['sig_name'],
        )


# ═══════════════════════════════════════════════════════════════════════════
# Backward-compatibility shims (emit DeprecationWarning)
# Remove once all callers are migrated.
# ═══════════════════════════════════════════════════════════════════════════

def _deprecation(name: str, replacement: str) -> None:
    import warnings
    warnings.warn(
        f"{name} is deprecated; use {replacement} instead.",
        DeprecationWarning,
        stacklevel=3,
    )


class UniversalECGAdapter:
    """Deprecated. Use `ECGPreprocessor` instead."""

    def __init__(self, target_leads=None, target_length=TARGET_LEN, target_fs=TARGET_FS):
        _deprecation("UniversalECGAdapter", "ECGPreprocessor(normalize='zscore')")
        target_lead = target_leads[0] if target_leads else "II"
        self._prep = ECGPreprocessor(
            target_fs=target_fs, target_len=target_length,
            target_lead=target_lead, normalize="zscore",
        )

    def standardize(self, raw_signal, source_leads, fs_in, powerline_hz=50):
        self._prep.powerline_hz = powerline_hz
        return self._prep.process(raw_signal, fs_in=fs_in, source_leads=source_leads)


class RobustECGAdapter:
    """Deprecated. Use `ECGPreprocessor` instead."""

    def __init__(self, target_leads=None, target_length=TARGET_LEN, target_fs=TARGET_FS,
                 win_limits=WIN_LIMITS):
        _deprecation("RobustECGAdapter", "ECGPreprocessor(normalize='winsorize')")
        target_lead = target_leads[0] if target_leads else "II"
        self._prep = ECGPreprocessor(
            target_fs=target_fs, target_len=target_length,
            target_lead=target_lead, normalize="winsorize", win_limits=win_limits,
        )

    def standardize(self, raw_signal, source_leads, fs_in, powerline_hz=50):
        self._prep.powerline_hz = powerline_hz
        return self._prep.process(raw_signal, fs_in=fs_in, source_leads=source_leads)
