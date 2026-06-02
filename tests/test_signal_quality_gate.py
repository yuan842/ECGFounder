"""S0 Signal-Quality Gate (Noisy policy) — unit tests.

SNR-only rule (DEFAULT OFF as of 2026-06-02; baseline-drift gating disabled):
  • every device (incl. AliveCor): NOISY ⇔ snr_proxy <= 0.10 (-10 dB), inclusive
Noisy ⇒ passed=False ⇒ caller skips detection. Fail-open when SQI absent.
"""
import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from overlay.signal_quality_gate import (
    SignalQualityGate, snr_to_db, device_snr_threshold, NOISY, HIGH_MOTION,
    DEFAULT_MIN_SNR, DEVICE_MIN_SNR)


def test_threshold_table():
    assert DEFAULT_MIN_SNR == 0.10
    assert DEVICE_MIN_SNR == {}                                  # no device overrides
    assert device_snr_threshold("default") == (0.10, True)
    assert device_snr_threshold("fzark") == (0.10, True)         # unknown ⇒ default
    assert device_snr_threshold("alivecor") == (0.10, True)      # AliveCor now uses default
    assert abs(snr_to_db(0.10) - (-10.0)) < 1e-6


def test_default_is_off():
    g = SignalQualityGate()
    assert g.enabled is False                                    # gate OFF by default
    assert g.min_snr == 0.10 and g.snr_inclusive is True
    assert g.max_baseline_drift is None                          # drift gating off


def test_default_device_inclusive_boundary():
    g = SignalQualityGate(enabled=True, device="default")
    assert g.gate(None, sqi={"snr_proxy": 0.10}).passed is False   # <= 0.10 ⇒ noisy
    assert g.gate(None, sqi={"snr_proxy": 0.1001}).passed is True  # just above ⇒ clean
    assert g.gate(None, sqi={"snr_proxy": 0.05}).passed is False


def test_alivecor_uses_default_threshold():
    g = SignalQualityGate(enabled=True, device="alivecor")
    assert g.min_snr == 0.10 and g.snr_inclusive is True
    assert g.gate(None, sqi={"snr_proxy": 0.11}).passed is True    # >0.10 ⇒ clean (was noisy under old 0.1122)
    assert g.gate(None, sqi={"snr_proxy": 0.10}).passed is False


def test_noisy_classifies_and_reasons():
    g = SignalQualityGate(enabled=True)
    r = g.gate(None, sqi={"snr_proxy": 0.02})
    assert r.quality[NOISY] is True and r.passed is False and "noisy" in r.reason


def test_baseline_drift_ignored_by_default():
    g = SignalQualityGate(enabled=True)                          # drift gating OFF
    r = g.gate(None, sqi={"snr_proxy": 0.9, "baseline_drift": 1.0})
    assert r.quality[NOISY] is False and r.passed is True


def test_baseline_drift_when_explicitly_enabled():
    g = SignalQualityGate(enabled=True, max_baseline_drift=0.1)   # opt-in re-enable
    assert g.gate(None, sqi={"snr_proxy": 0.9, "baseline_drift": 0.15}).passed is False
    assert g.gate(None, sqi={"snr_proxy": 0.9, "baseline_drift": 0.05}).passed is True


def test_fail_open_without_sqi():
    g = SignalQualityGate(enabled=True)
    assert g.gate(None).passed is True and NOISY not in g.gate(None).quality


def test_explicit_min_snr_override():
    g = SignalQualityGate(enabled=True, min_snr=0.3, snr_inclusive=False)
    assert g.gate(None, sqi={"snr_proxy": 0.25}).passed is False
    assert g.gate(None, sqi={"snr_proxy": 0.35}).passed is True


def test_off_is_passthrough():
    g = SignalQualityGate(enabled=False)
    r = g.gate(None, sqi={"snr_proxy": 0.0})
    assert r.passed is True and "off" in r.reason
    assert r.quality[NOISY] is True                              # flag still computed


def test_motion_independent_of_noisy():
    g = SignalQualityGate(enabled=True)
    r = g.gate(None, accel={"mean_motion_mg": 99.0}, sqi={"snr_proxy": 1.0})
    assert r.quality[HIGH_MOTION] is True and r.passed is False   # motion still gates
