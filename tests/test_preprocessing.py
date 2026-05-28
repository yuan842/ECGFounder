"""Tests for the unified `preprocessing.ECGPreprocessor`.

Run with:  python3 -m pytest tests/test_preprocessing.py -v
"""
import sys
import os

import numpy as np
import pytest
import torch

# Make the repo root importable when pytest is run from anywhere
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from preprocessing import (
    ECGPreprocessor,
    TARGET_FS,
    TARGET_LEN,
    WIN_LIMITS,
    FZARK_MAGNIFICATION,
)


# ─── Fixtures ───────────────────────────────────────────────────────────────

@pytest.fixture
def rng():
    return np.random.default_rng(seed=42)


# ─── Shape / dtype invariants ──────────────────────────────────────────────

class TestShapeAndDtype:
    @pytest.mark.parametrize("fs_in", [128, 250, 360, 500, 1000])
    def test_output_shape_independent_of_input_fs(self, fs_in, rng):
        sig = rng.standard_normal(fs_in * 10).astype(np.float32)
        out = ECGPreprocessor().process(sig, fs_in=fs_in)
        assert out.shape == (1, TARGET_LEN)
        assert out.dtype == torch.float32

    @pytest.mark.parametrize("duration_s", [3, 5, 10, 15, 30])
    def test_short_and_long_inputs_both_produce_target_len(self, duration_s, rng):
        fs_in = 128
        sig = rng.standard_normal(fs_in * duration_s).astype(np.float32)
        out = ECGPreprocessor().process(sig, fs_in=fs_in)
        assert out.shape == (1, TARGET_LEN)

    def test_2d_single_channel_accepted(self, rng):
        sig = rng.standard_normal((1, 1280)).astype(np.float32)
        out = ECGPreprocessor().process(sig, fs_in=128)
        assert out.shape == (1, TARGET_LEN)

    def test_3d_input_rejected(self, rng):
        sig = rng.standard_normal((1, 1, 1280))
        with pytest.raises(ValueError, match="1-D or 2-D"):
            ECGPreprocessor().process(sig, fs_in=128)


# ─── Numerical robustness ──────────────────────────────────────────────────

class TestNumericalRobustness:
    def test_zero_input_produces_no_nan(self):
        sig = np.zeros(1280, dtype=np.float32)
        out = ECGPreprocessor().process(sig, fs_in=128)
        assert not torch.isnan(out).any()
        assert not torch.isinf(out).any()

    def test_constant_input_produces_no_nan(self):
        sig = np.full(1280, 3.7, dtype=np.float32)
        out = ECGPreprocessor().process(sig, fs_in=128)
        assert not torch.isnan(out).any()

    def test_winsorize_clips_extreme_outliers(self, rng):
        # Inject a 1000-sigma spike. After winsorize+zscore, max should not
        # be dominated by it (legacy threshold: |x|<10).
        sig = rng.standard_normal(1280).astype(np.float32)
        sig[100] = 1000.0
        out = ECGPreprocessor(normalize="winsorize").process(sig, fs_in=128)
        assert torch.abs(out).max() < 10.0

    def test_zscore_is_dominated_by_outlier(self, rng):
        # Without winsorize, the spike dominates → demonstrates why winsorize
        # is the safer default.
        sig = rng.standard_normal(1280).astype(np.float32)
        sig[100] = 1000.0
        out = ECGPreprocessor(normalize="zscore").process(sig, fs_in=128)
        # The spike should still produce a visibly large value
        assert torch.abs(out).max() > 5.0


# ─── Factory presets ───────────────────────────────────────────────────────

class TestFactoryPresets:
    def test_for_fzark_settings(self):
        p = ECGPreprocessor.for_fzark()
        assert p.powerline_hz == 50
        assert p.normalize == "winsorize"
        assert p.target_fs == TARGET_FS
        assert p.target_len == TARGET_LEN

    def test_for_ptbxl_settings(self):
        p = ECGPreprocessor.for_ptbxl()
        assert p.powerline_hz == 50
        assert p.normalize == "winsorize"
        assert p.apply_notch is True
        assert p.apply_bandpass is True

    def test_for_mitdb_settings(self):
        p = ECGPreprocessor.for_mitdb()
        assert p.powerline_hz == 60
        assert p.target_lead == "II"
        assert p.normalize == "winsorize"


# ─── Constructor validation ────────────────────────────────────────────────

class TestConstructorValidation:
    def test_invalid_normalize_raises(self):
        with pytest.raises(ValueError, match="normalize"):
            ECGPreprocessor(normalize="oops")

    def test_invalid_powerline_raises(self):
        with pytest.raises(ValueError, match="powerline_hz"):
            ECGPreprocessor(powerline_hz=42)


# ─── Lead alignment ────────────────────────────────────────────────────────

class TestLeadAlignment:
    def test_lead_ii_picked_from_12_lead_signal(self, rng):
        leads = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF',
                 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
        sig = rng.standard_normal((12, 5000)).astype(np.float32) * 0.1
        # Make Lead II a unique 1 Hz sine wave (recognizable post-pipeline)
        t = np.arange(5000) / 500.0
        sig[1] = np.sin(2 * np.pi * 1.0 * t).astype(np.float32)

        prep = ECGPreprocessor(target_lead="II", apply_notch=False,
                               apply_bandpass=False, apply_baseline=False,
                               normalize="none")
        out = prep.process(sig, fs_in=500, source_leads=leads)
        # Output should look like the input Lead II (a clean 1 Hz sine)
        out_np = out[0].numpy()
        assert out_np.shape == (TARGET_LEN,)
        # Energy at the sine wave's fundamental should dominate
        fft_mag = np.abs(np.fft.rfft(out_np))
        peak_idx = int(np.argmax(fft_mag))
        peak_hz = peak_idx * 500.0 / TARGET_LEN
        assert abs(peak_hz - 1.0) < 0.5, f"peak at {peak_hz} Hz, expected ~1 Hz"

    def test_mlii_accepted_as_lead_ii(self, rng):
        sig = rng.standard_normal((2, 3600)).astype(np.float32)
        prep = ECGPreprocessor(target_lead="II")
        out = prep.process(sig, fs_in=360, source_leads=['MLII', 'V1'])
        assert out.shape == (1, TARGET_LEN)

    def test_missing_lead_raises(self, rng):
        sig = rng.standard_normal((2, 3600))
        with pytest.raises(ValueError, match="not found"):
            ECGPreprocessor(target_lead="V99").process(
                sig, fs_in=360, source_leads=['MLII', 'V1']
            )

    def test_multichannel_without_source_leads_raises(self, rng):
        sig = rng.standard_normal((3, 1000))
        with pytest.raises(ValueError, match="source_leads is None"):
            ECGPreprocessor().process(sig, fs_in=500, source_leads=None)

    def test_source_leads_length_mismatch_raises(self, rng):
        sig = rng.standard_normal((3, 1000))
        with pytest.raises(ValueError, match="has 2 entries but signal has 3"):
            ECGPreprocessor().process(sig, fs_in=500, source_leads=['I', 'II'])


# ─── Stage toggles ─────────────────────────────────────────────────────────

class TestStageToggles:
    def test_no_filtering_passes_signal_through_amplitude_only(self, rng):
        sig = rng.standard_normal(5000).astype(np.float32)
        prep = ECGPreprocessor(
            apply_notch=False, apply_bandpass=False, apply_baseline=False,
            normalize="none",
        )
        out = prep.process(sig, fs_in=500)
        # fs_in == target_fs and len == target_len → no resample, no crop
        np.testing.assert_allclose(out[0].numpy(), sig, rtol=1e-5)

    def test_resample_passthrough_when_fs_matches(self, rng):
        sig = rng.standard_normal(5000).astype(np.float32)
        prep = ECGPreprocessor()
        out = prep.process(sig, fs_in=500)
        assert out.shape == (1, TARGET_LEN)


# ─── Byte-parity with legacy implementation ─────────────────────────────────

class TestLegacyParity:
    """Match the inline `robust_preprocess` from the old eval scripts."""

    @staticmethod
    def legacy_preprocess(signal_1d, fs_in=128):
        """Copied verbatim from the old eval_ecg_tprex.py."""
        from scipy.signal import medfilt, iirnotch, filtfilt, butter
        from scipy.interpolate import interp1d
        x = signal_1d[np.newaxis, :].astype(np.float64)
        # Notch
        b, a = iirnotch(50, 30, fs_in); x = filtfilt(b, a, x, axis=1)
        # Bandpass
        b, a = butter(4, [0.67, 40.0], btype='bandpass', fs=fs_in)
        x = filtfilt(b, a, x, axis=1)
        # Baseline
        kernel = int(0.4 * fs_in) | 1
        baseline = medfilt(x[0], kernel_size=kernel); x[0] -= baseline
        # Resample
        t = x.shape[1] / fs_in
        n_out = int(t * 500)
        x_old = np.linspace(0, t, x.shape[1], endpoint=True)
        x_new = np.linspace(0, t, n_out, endpoint=True)
        f = interp1d(x_old, x[0], kind='linear', fill_value='extrapolate')
        x = f(x_new)[np.newaxis, :]
        # Crop/pad
        cur = x.shape[1]
        out = np.zeros((1, 5000), dtype=np.float32)
        if cur >= 5000:
            s = (cur - 5000) // 2; out[0] = x[0, s:s+5000]
        else:
            p = (5000 - cur) // 2; out[0, p:p+cur] = x[0]
        # Winsorized z-score
        lo, hi = np.percentile(out[0], (1.5, 98.5))
        clipped = np.clip(out[0], lo, hi)
        m, s = np.mean(clipped), np.std(clipped)
        out[0] = (clipped - m) / (s + 1e-8)
        return out[0]   # 1-D

    def test_parity_with_legacy_robust_preprocess(self, rng):
        # 10 seconds of synthetic ECG-like data @ 128 Hz
        fs_in = 128
        sig = rng.standard_normal(fs_in * 10).astype(np.float32) * 0.5
        # Add a 1 Hz sinusoid to mimic a heartbeat structure
        t = np.arange(len(sig)) / fs_in
        sig += 0.3 * np.sin(2 * np.pi * 1.0 * t).astype(np.float32)

        legacy_out = self.legacy_preprocess(sig, fs_in=fs_in)
        new_out = ECGPreprocessor.for_fzark().process(sig, fs_in=fs_in)[0].numpy()

        np.testing.assert_allclose(new_out, legacy_out, atol=1e-5, rtol=1e-4)


# ─── Loader smoke tests (skip if fixtures missing) ─────────────────────────

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

class TestLoaders:
    def test_from_fzark_json_smoke(self):
        # Find any fzark JSON sample
        sample_dirs = [
            os.path.join(REPO_ROOT, "data/ecg_tp_fzark"),
            os.path.join(REPO_ROOT, "data/ecg-tp_rex"),
        ]
        sample = None
        for d in sample_dirs:
            if not os.path.isdir(d):
                continue
            for root, _, files in os.walk(d):
                for f in files:
                    if f.endswith(".json"):
                        sample = os.path.join(root, f)
                        break
                if sample:
                    break
            if sample:
                break
        if sample is None:
            pytest.skip("No fzark JSON test fixture available")

        out = ECGPreprocessor.for_fzark().from_fzark_json(sample)
        assert out.shape == (1, TARGET_LEN)
        assert out.dtype == torch.float32
        assert not torch.isnan(out).any()


# ─── Deprecation shim wiring ───────────────────────────────────────────────

class TestDeprecationShims:
    def test_universal_adapter_emits_warning_and_works(self, rng):
        from preprocessing import UniversalECGAdapter
        with pytest.warns(DeprecationWarning):
            adapter = UniversalECGAdapter(target_leads=['II'])
        sig = rng.standard_normal((1, 1280)).astype(np.float32)
        out = adapter.standardize(sig, source_leads=['II'], fs_in=128)
        assert out.shape == (1, TARGET_LEN)

    def test_robust_adapter_emits_warning_and_works(self, rng):
        from preprocessing import RobustECGAdapter
        with pytest.warns(DeprecationWarning):
            adapter = RobustECGAdapter(target_leads=['II'])
        sig = rng.standard_normal((1, 1280)).astype(np.float32)
        out = adapter.standardize(sig, source_leads=['II'], fs_in=128)
        assert out.shape == (1, TARGET_LEN)
