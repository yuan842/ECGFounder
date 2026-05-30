"""Realistic ECG noise augmentation — close the clean-train → noisy-deploy gap.

⏸ FROZEN / NOT IN USE (2026-05-30). An A/B showed this does NOT improve performance on
the noisy fzark cohort (mean ΔAUROC +0.001) — fzark's gap is device/lead, not recoverable
additive noise. The module is correct (SNR exact to <0.01 dB) and kept for reuse, but is
OPT-IN and OFF by default; do not enable in training/production without a new hypothesis.
See docs/NOISE_AUGMENTATION_LESSON.md (and res/finetune_6head_v2/NOISE_AUG_AB_FZARK.md).


Clean training data (PTB-XL) makes heads that work on clean ECG but fail on noisy
ambulatory ECG (we measured SVT/VT AUROC 0.99 on PTB-XL vs 0.36-0.61 on fzark single
lead). This module mixes deployment-realistic noise into clean signals at a CONTROLLED
SNR, so the model sees the same morphology under corruption.

Two noise sources:
  • Synthetic models (default, NO data needed): baseline wander, muscle/EMG, electrode
    motion, powerline, white. Physics-flavored, parameterised, reproducible.
  • Optional real noise: MIT-BIH Noise Stress Test DB (NSTDB) records bw/ma/em — pass
    --nstdb-dir to mix real recorded noise instead of/alongside the synthetic models.

SNR is controlled in dB: noise is scaled so 10·log10(P_signal / P_noise) == target.

Use as a training-time transform on CLEAN data only (never on val/test), ideally at the
raw stage before band-pass; works on already-processed signals too (scale-invariant mix),
just less physically literal for sub-band noise (baseline wander on a band-passed signal).
"""
from __future__ import annotations
from typing import Sequence
import numpy as np
from scipy.signal import butter, filtfilt

NOISE_TYPES = ("baseline", "muscle", "electrode", "powerline", "white")


# ─── synthetic noise generators (unit-ish amplitude; SNR scaling done later) ───
def _rng(rng): return rng if rng is not None else np.random.default_rng()


def baseline_wander(n: int, fs: int, rng=None) -> np.ndarray:
    """Low-frequency drift: sum of 0.05–0.5 Hz sinusoids, random phase/amp."""
    rng = _rng(rng); t = np.arange(n) / fs; x = np.zeros(n)
    for _ in range(rng.integers(2, 5)):
        f = rng.uniform(0.05, 0.5); x += rng.uniform(0.5, 1.5) * np.sin(2*np.pi*f*t + rng.uniform(0, 2*np.pi))
    return x


def muscle_artifact(n: int, fs: int, rng=None) -> np.ndarray:
    """EMG: band-passed white noise (~20–100 Hz), bursty envelope."""
    rng = _rng(rng); w = rng.standard_normal(n)
    hi = min(100.0, fs/2 - 1)
    b, a = butter(4, [20/(fs/2), hi/(fs/2)], btype="bandpass")
    x = filtfilt(b, a, w)
    # bursty: amplitude-modulate with a slow random envelope
    env = np.clip(filtfilt(*butter(2, 1.0/(fs/2), btype="low"), rng.standard_normal(n)) + 0.5, 0, None)
    return x * (0.5 + env)


def electrode_motion(n: int, fs: int, rng=None) -> np.ndarray:
    """Electrode-motion: occasional large low-frequency transients (baseline jumps)."""
    rng = _rng(rng); x = np.zeros(n)
    for _ in range(rng.integers(1, 4)):
        c = rng.integers(0, n); w = int(rng.uniform(0.1, 0.6) * fs)
        lo, hi = max(0, c-w), min(n, c+w)
        bump = np.hanning(hi-lo) * rng.uniform(-3, 3)
        x[lo:hi] += bump
    b, a = butter(2, 5.0/(fs/2), btype="low")               # smooth → low-freq motion
    return filtfilt(b, a, x)


def powerline(n: int, fs: int, hz: float = 50.0, rng=None) -> np.ndarray:
    """Mains interference at `hz` (+ small 2nd/3rd harmonics)."""
    rng = _rng(rng); t = np.arange(n) / fs; ph = rng.uniform(0, 2*np.pi)
    x = np.sin(2*np.pi*hz*t + ph)
    x += 0.3 * np.sin(2*np.pi*2*hz*t + ph) + 0.15 * np.sin(2*np.pi*3*hz*t + ph)
    return x


def white_noise(n: int, fs: int, rng=None) -> np.ndarray:
    return _rng(rng).standard_normal(n)


_GEN = {"baseline": baseline_wander, "muscle": muscle_artifact,
        "electrode": electrode_motion, "powerline": powerline, "white": white_noise}


# ─── SNR-controlled mixing ────────────────────────────────────────────────────
def add_noise_at_snr(clean: np.ndarray, noise: np.ndarray, snr_db: float) -> np.ndarray:
    """Return clean + scaled noise such that achieved SNR == snr_db (per-sample power)."""
    ps = float(np.mean(clean.astype(np.float64) ** 2))
    pn = float(np.mean(noise.astype(np.float64) ** 2))
    if pn <= 0 or ps <= 0:
        return clean
    target_pn = ps / (10.0 ** (snr_db / 10.0))
    return (clean + noise * np.sqrt(target_pn / pn)).astype(clean.dtype if hasattr(clean, "dtype") else np.float32)


# ─── the augmenter (training-time transform) ──────────────────────────────────
class NoiseAugmenter:
    """Callable: clean (C,N) or (N,) → noisy, mixing 1–`max_types` noise types at a
    random SNR in [snr_min, snr_max] dB, applied with probability `p`.

    Deployment-matched defaults: ambulatory single-lead ECG typically sits ~5–20 dB;
    calibrate snr_min/max to your measured fzark/MOVE SNR. Set p<1 so the model still
    sees clean examples (mixed-condition robustness).
    """
    def __init__(self, fs: int = 500, snr_min: float = 5.0, snr_max: float = 20.0,
                 types: Sequence[str] = NOISE_TYPES, max_types: int = 2, p: float = 0.7,
                 powerline_hz: float = 50.0, nstdb_dir: str | None = None, seed: int | None = None):
        self.fs, self.snr_min, self.snr_max = fs, snr_min, snr_max
        self.types = [t for t in types if t in _GEN]
        self.max_types, self.p, self.powerline_hz = max_types, p, powerline_hz
        self.rng = np.random.default_rng(seed)
        self._nstdb = _load_nstdb(nstdb_dir, fs) if nstdb_dir else None   # {name: 1-D array}

    def _make_noise(self, n: int) -> np.ndarray:
        k = int(self.rng.integers(1, self.max_types + 1))
        chosen = self.rng.choice(self.types, size=min(k, len(self.types)), replace=False)
        noise = np.zeros(n)
        for t in chosen:
            if self._nstdb and t in self._nstdb:           # real NSTDB segment if available
                seg = self._nstdb[t]
                s = int(self.rng.integers(0, max(1, len(seg) - n)))
                g = seg[s:s + n]
                g = np.pad(g, (0, n - len(g))) if len(g) < n else g
                noise += g / (np.std(g) + 1e-8)
            elif t == "powerline":
                noise += _GEN[t](n, self.fs, self.powerline_hz, self.rng)
            else:
                noise += _GEN[t](n, self.fs, rng=self.rng)
        return noise

    def __call__(self, sig):
        is_torch = hasattr(sig, "detach")
        arr = sig.detach().cpu().numpy() if is_torch else np.asarray(sig)
        if self.rng.random() < self.p:
            snr = float(self.rng.uniform(self.snr_min, self.snr_max))
            out = np.empty_like(arr, dtype=np.float32)
            flat = arr.reshape(-1, arr.shape[-1])
            o = out.reshape(-1, arr.shape[-1])
            for i in range(flat.shape[0]):
                o[i] = add_noise_at_snr(flat[i].astype(np.float32), self._make_noise(arr.shape[-1]), snr)
        else:
            out = arr.astype(np.float32)
        if is_torch:
            import torch
            return torch.from_numpy(out)
        return out


def _load_nstdb(nstdb_dir, fs):
    """Optional: load NSTDB bw/ma/em records → {synthetic-type: 1-D noise} (resampled)."""
    import os
    try:
        import wfdb
        from scipy.signal import resample_poly
        from math import gcd
    except Exception:
        return None
    name_map = {"bw": "baseline", "ma": "muscle", "em": "electrode"}
    out = {}
    for rec, t in name_map.items():
        p = os.path.join(nstdb_dir, rec)
        if os.path.exists(p + ".dat"):
            r = wfdb.rdrecord(p); s = r.p_signal[:, 0].astype(np.float64)
            if r.fs != fs:
                g = gcd(int(r.fs), fs); s = resample_poly(s, fs // g, int(r.fs) // g)
            out[t] = s
    return out or None


def _self_test():
    fs, n = 500, 5000
    rng = np.random.default_rng(0)
    clean = np.sin(2*np.pi*1.2*np.arange(n)/fs).astype(np.float32)   # 72 bpm proxy
    print("achieved vs target SNR (single noise types):")
    for t in NOISE_TYPES:
        noise = (_GEN[t](n, fs, fs, rng) if t == "powerline" else _GEN[t](n, fs, rng=rng))
        for tgt in (0.0, 10.0, 20.0):
            mixed = add_noise_at_snr(clean, noise, tgt)
            res = (mixed.astype(np.float64) - clean.astype(np.float64))
            ach = 10*np.log10(np.mean(clean.astype(np.float64)**2)/np.mean(res**2))
            assert abs(ach - tgt) < 1e-2, (t, tgt, ach)   # float32-mix tolerance
        print(f"  {t:<10} SNR exact to <0.01 dB ✓")
    aug = NoiseAugmenter(fs=fs, snr_min=10, snr_max=10, p=1.0, seed=1)
    out = aug(clean[None, :])                                        # (1,N)
    res = out[0] - clean
    ach = 10*np.log10(np.mean(clean**2)/np.mean(res**2))
    print(f"\nNoiseAugmenter (1,N) @target 10 dB → achieved {ach:.2f} dB; shape {out.shape} ✓")
    # p=0 → passthrough
    assert np.allclose(NoiseAugmenter(p=0.0, seed=2)(clean), clean)
    print("p=0 passthrough ✓\nself-test OK ✓")


if __name__ == "__main__":
    _self_test()
