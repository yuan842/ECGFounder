"""L2 arbiter — unit tests for the GT-derived exclusion/couple rules.

The rules are a direct image of the PTB-XL GT co-occurrence matrix:
  - {Brady(4), AFib(5), Sinus-Tachy(6)} pairwise GT=0 → exclusion group (≤1 fires)
  - SVT-Run(93) ≡ V-Run(98) GT Jaccard 1.0 → couple group (kept, merged for alerts)
"""
import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from overlay.arbiter import (
    arbitrate, to_alerts, ArbiterConfig, DEFAULT_CONFIG, GT_MATCHED_CONFIG, OFF_CONFIG,
    EXCLUSION_GROUPS, COUPLE_GROUPS, BRADY, AFIB, TACHY, SVT, VRUN, PAUSE)
from overlay.types import ScopeScores

ON = GT_MATCHED_CONFIG


def ss(probs, nsr=0.0, hr=None):
    return ScopeScores(probs=probs, nsr_score=nsr,
                       context=({"hr_bpm": hr} if hr is not None else {}))


def fired(d):
    return {h for h, dec in d.items() if dec.fired}


# ── master switch ────────────────────────────────────────────────────────────
def test_explicit_off_is_passthrough():
    s = ss({BRADY: 0.9, AFIB: 0.9, TACHY: 0.9})
    d = arbitrate(s, OFF_CONFIG)
    assert fired(d) == {BRADY, AFIB, TACHY}          # nothing suppressed when OFF
    assert all("L2 off" in d[h].reason for h in (BRADY, AFIB, TACHY))

def test_default_is_on():
    # DEFAULT_CONFIG is now L2 ON → exclusion group keeps one of {4,5,6}
    s = ss({BRADY: 0.9, AFIB: 0.92, TACHY: 0.9})
    assert len(fired(arbitrate(s, DEFAULT_CONFIG)) & {BRADY, AFIB, TACHY}) == 1


# ── exclusion group {4,5,6}: at most one fires ──────────────────────────────
def test_exclusion_group_keeps_one():
    s = ss({BRADY: 0.9, AFIB: 0.92, TACHY: 0.9})
    f = fired(arbitrate(s, ON))
    assert len(f & {BRADY, AFIB, TACHY}) == 1

def test_afib_beats_tachy_directionally():
    # even when Tachy scores higher, AFib wins (directional AF-RVR rule)
    s = ss({AFIB: 0.55, TACHY: 0.99})
    f = fired(arbitrate(s, ON))
    assert AFIB in f and TACHY not in f

def test_exclusion_keeps_highest_margin():
    # Brady vs AFib (no directional rule): higher margin-over-threshold wins.
    cfg = ArbiterConfig(enabled=True, fire_threshold={BRADY: 0.5, AFIB: 0.5})
    s = ss({BRADY: 0.95, AFIB: 0.60})
    f = fired(arbitrate(s, cfg))
    assert BRADY in f and AFIB not in f

def test_brady_tachy_never_both():
    s = ss({BRADY: 0.8, TACHY: 0.8})
    assert len(fired(arbitrate(s, ON)) & {BRADY, TACHY}) == 1


# ── couple group {93,98}: kept together, merged for alerting ────────────────
def test_couple_group_kept_and_merged():
    s = ss({SVT: 0.7, VRUN: 0.7})
    d = arbitrate(s, ON)
    assert SVT in fired(d) and VRUN in fired(d)       # both kept in the record
    non_run, run = to_alerts(d, ON)
    assert run is not None and run.fired               # merged to one run alert
    assert not any(x.head in (SVT, VRUN) for x in non_run)

def test_runs_not_in_exclusion_group():
    # a run may co-fire with a rate head (runs/pause are free heads)
    s = ss({AFIB: 0.9, VRUN: 0.7})
    f = fired(arbitrate(s, ON))
    assert AFIB in f and VRUN in f


# ── NSR contradiction (FP feature) ──────────────────────────────────────────
def test_nsr_contradiction_suppresses_weak_arrhythmia():
    s = ss({AFIB: 0.6}, nsr=0.9)                       # weak AFib + high NSR
    assert AFIB not in fired(arbitrate(s, ON))
    s2 = ss({AFIB: 0.95}, nsr=0.9)                     # strong AFib survives
    assert AFIB in fired(arbitrate(s2, ON))


# ── HR plausibility ─────────────────────────────────────────────────────────
def test_hr_plausibility_blocks_fast_brady():
    s = ss({BRADY: 0.9}, hr=80)
    assert BRADY not in fired(arbitrate(s, ON))
    s2 = ss({BRADY: 0.9}, hr=45)
    assert BRADY in fired(arbitrate(s2, ON))


# ── uncalibrated (base-routed) heads excluded from rate-exclusion ───────────
def test_uncalibrated_head_excluded_from_rate_arbitration():
    # On a device where TACHY is base-routed (uncalibrated, base@0.5), TACHY
    # must NOT suppress AFib via the directional rule, nor lose by margin —
    # it is simply excluded from the arbitration and flows through unchanged.
    cfg = ArbiterConfig(enabled=True, uncalibrated_heads=frozenset({TACHY}))
    # directional case: AFib + Tachy both fire — neither suppresses the other
    s = ss({AFIB: 0.6, TACHY: 0.99})
    f = fired(arbitrate(s, cfg))
    assert AFIB in f and TACHY in f
    # margin case: Brady + Tachy both fire — uncalibrated Tachy flows through
    s2 = ss({BRADY: 0.95, TACHY: 0.99}, hr=45)
    f2 = fired(arbitrate(s2, cfg))
    assert BRADY in f2 and TACHY in f2

def test_uncalibrated_head_does_not_suppress_calibrated_peer():
    # If AFib were uncalibrated, it must NOT win the AFib▸Tachy directional rule.
    cfg = ArbiterConfig(enabled=True, uncalibrated_heads=frozenset({AFIB}))
    s = ss({AFIB: 0.99, TACHY: 0.6})
    f = fired(arbitrate(s, cfg))
    assert AFIB in f and TACHY in f


# ── structure sanity ─────────────────────────────────────────────────────────
def test_groups_match_gt_structure():
    assert (BRADY, AFIB, TACHY) in EXCLUSION_GROUPS
    assert (SVT, VRUN) in COUPLE_GROUPS
    # exclusion and couple groups are disjoint
    excl = {h for g in EXCLUSION_GROUPS for h in g}
    coup = {h for g in COUPLE_GROUPS for h in g}
    assert excl.isdisjoint(coup)
