"""Regression test: split pipeline (fzark profile) ≡ production v2 suppressor.

For every active event and a grid of feature dicts that straddle each gate's
boundary, the AND of (MotionFP, SQIFP) keep-decisions must equal the production
MultiClassFPSuppressor / _apply_rules keep-decision. Also checks the new
calibration fitters and the family-independence invariant.
"""
import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np

from multiclass_fp_suppression import _apply_rules, ACTIVE_RULES
from fp_suppression import (FPSuppressionPipeline, MotionFPSuppressor,
                            SQIFPSuppressor, get_profile, fit_gate_supervised,
                            fit_gate_envelope)

EVENTS = list(ACTIVE_RULES.keys())

# feature grids spanning every boundary used by the rules
MOTIONS = [0.0, 1.0, 4.9, 5.0, 5.1, 14.9, 15.0, 23.9, 24.0, 24.1, 30.0, 100.0]
SNRS    = [0.0, 1.0, 1.19, 1.2, 1.21, 2.0, 5.0]
HRS     = [30.0, 56.2, 56.3, 56.4, 70.0, 120.0]


def test_equivalence():
    pipe = FPSuppressionPipeline(device="fzark")
    n = 0
    for ev in EVENTS:
        for mot in MOTIONS:
            for snr in SNRS:
                for hr in HRS:
                    feats = {'mean_motion': mot, 'snr_proxy': snr, 'mean_hr_bpm': hr}
                    prod_keep, _, _ = _apply_rules(ev, feats)
                    split_keep = pipe.suppress(ev, feats).keep
                    assert prod_keep == split_keep, (
                        f"MISMATCH {ev}: motion={mot} snr={snr} hr={hr} "
                        f"prod={prod_keep} split={split_keep}")
                    n += 1
    print(f"[equivalence] {n} feature-combos × events all match production ✓")


def test_missing_feature_fail_open():
    """Missing feature → gate passes (fail-open), matching production."""
    pipe = FPSuppressionPipeline(device="fzark")
    for ev in EVENTS:
        feats = {}  # nothing present
        prod_keep, _, _ = _apply_rules(ev, feats)
        assert pipe.suppress(ev, feats).keep == prod_keep == True
    print("[fail-open] missing features keep all events ✓")


def test_families_independent():
    """Toggling SQI off must not change motion-only events, and vice-versa."""
    motion = MotionFPSuppressor("fzark")
    sqi = SQIFPSuppressor("fzark")
    feats = {'mean_motion': 100.0, 'snr_proxy': 5.0, 'mean_hr_bpm': 80.0}
    # AFib is motion-only: SQI passes-through, motion suppresses (100>5)
    assert motion.suppress('Atrial Fibrillation', feats).keep is False
    assert sqi.suppress('Atrial Fibrillation', feats).keep is True
    # Bradycardia is SQI-only: motion passes-through, SQI suppresses (80>56.3)
    assert motion.suppress('Bradycardia', feats).keep is True
    assert sqi.suppress('Bradycardia', feats).keep is False
    print("[independence] motion/sqi tables disjoint per family ✓")


def test_vtrig_split_is_and():
    """V-Trig = (motion>=24) AND (snr>1.2) — both halves must hold to keep."""
    pipe = FPSuppressionPipeline(device="fzark")
    cases = {
        (30.0, 2.0): True,    # both pass → keep
        (30.0, 1.0): False,   # snr fails → drop
        (10.0, 2.0): False,   # motion fails → drop
        (10.0, 1.0): False,   # both fail → drop
    }
    for (mot, snr), want in cases.items():
        got = pipe.suppress('Ventricular Trigeminy',
                            {'mean_motion': mot, 'snr_proxy': snr}).keep
        assert got == want, f"V-Trig motion={mot} snr={snr}: want {want} got {got}"
    print("[v-trig] motion∧snr split reproduces AND gate ✓")


def test_calibration_supervised():
    """Supervised fit recovers a separating threshold (TP low-motion, FP high)."""
    rng = np.random.default_rng(0)
    tp = rng.normal(1.0, 0.3, 200).clip(0)      # real AFib: low motion
    fp = rng.normal(8.0, 1.5, 200).clip(0)      # motion-artifact FP: high motion
    vals = np.concatenate([tp, fp])
    is_tp = np.array([True]*200 + [False]*200)
    gate, stats = fit_gate_supervised(vals, is_tp, 'mean_motion', '<=')
    assert 1.5 < gate.threshold < 7.0, gate.threshold
    assert stats['tp_retained'] > 0.9 and stats['fp_removed'] > 0.9, stats
    print(f"[fit-supervised] thr={gate.threshold:.2f} "
          f"TP-kept={stats['tp_retained']:.2f} FP-dropped={stats['fp_removed']:.2f} ✓")


def test_fzark_sqi_profile_gates():
    """The prototype fzark_sqi profile adds SQI quality gates for Bradycardia + Pause
    without disturbing fzark (equivalence test above still passes)."""
    sqi = SQIFPSuppressor("fzark_sqi")
    brady = {g.feature: g for g in sqi.gates_for("Bradycardia")}
    assert brady["snr_proxy"].op == ">=" and brady["snr_proxy"].threshold == 0.84
    assert brady["mean_hr_bpm"].threshold == 56.3                # HR gate retained
    pause = sqi.gates_for("Pause")
    assert len(pause) == 1 and pause[0].feature == "hf_noise" and pause[0].op == "<="
    # low-SNR bradycardia (good rate) is now dropped; clean one kept
    assert sqi.suppress("Bradycardia", {"mean_hr_bpm": 50, "snr_proxy": 0.3}).keep is False
    assert sqi.suppress("Bradycardia", {"mean_hr_bpm": 50, "snr_proxy": 2.0}).keep is True
    # HF-noisy pause dropped; clean pause kept
    assert sqi.suppress("Pause", {"hf_noise": 0.10}).keep is False
    assert sqi.suppress("Pause", {"hf_noise": 0.0}).keep is True
    # fzark (production) profile is unchanged — no Pause gate, Brady = HR only
    prod = SQIFPSuppressor("fzark")
    assert prod.gates_for("Pause") == []
    assert [g.feature for g in prod.gates_for("Bradycardia")] == ["mean_hr_bpm"]
    print("[fzark_sqi] SQI quality gates present; fzark profile unchanged ✓")


def test_calibration_envelope():
    """Label-free rest-envelope fit on MOVE-like rest motion → ~1 mG."""
    rng = np.random.default_rng(0)
    rest = rng.normal(0.5, 0.2, 500).clip(0)
    gate, stats = fit_gate_envelope(rest, 'mean_motion', '<=', pct=99.0)
    assert 0.6 < gate.threshold < 1.5, gate.threshold
    print(f"[fit-envelope] rest-p99 thr={gate.threshold:.2f} mG ✓")


if __name__ == "__main__":
    test_equivalence()
    test_missing_feature_fail_open()
    test_families_independent()
    test_vtrig_split_is_and()
    test_calibration_supervised()
    test_fzark_sqi_profile_gates()
    test_calibration_envelope()
    print("\nALL SPLIT-FP TESTS PASSED ✓")
