"""Smoke tests for the per-recording reporting / annotation modules.

Two layers, both fast and data-free:

1. merge_events() — the pure episode-merging core of scripts/report_recording.py
   (gap tolerance, min-duration, onset/offset/peak rollup).

2. Scope-drift guard — scripts/viz_report_formats.py hardcodes EF_HEADS, the head
   list it plots. When NORMAL/NSR (head 2) left DETECTION_SCOPE (2026-06-01) the
   reporting writer dropped its p_2 column but the visualizer still asked for it →
   KeyError 'p_2'. This test statically asserts EF_HEADS == scope so any future
   scope change that drifts the visualizer fails here, loudly, instead of at runtime.
"""
import os, sys, ast
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import label_config as L
from scripts.report_recording import merge_events

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
VIZ = os.path.join(REPO, "scripts", "viz_report_formats.py")


# ── merge_events: episode rollup ─────────────────────────────────────────────
def test_merge_events_single_episode_rollup():
    t0 = np.array([0, 10, 20, 30, 40], float)
    t1 = np.array([10, 20, 30, 40, 50], float)
    fired = np.array([False, True, True, True, False])
    probs = np.array([0.1, 0.6, 0.9, 0.7, 0.2])
    ev = merge_events(t0, t1, fired, probs, max_gap=1, min_win=1)
    assert len(ev) == 1
    e = ev[0]
    assert e["onset_s"] == 10.0 and e["offset_s"] == 40.0
    assert e["duration_s"] == 30.0
    assert e["n_windows"] == 3 and e["n_fired"] == 3
    assert e["peak_prob"] == 0.9


def test_merge_events_gap_tolerance():
    # fired-gap-fired: bridged into one episode when max_gap>=1, split when 0.
    t0 = np.array([0, 10, 20], float); t1 = np.array([10, 20, 30], float)
    fired = np.array([True, False, True]); probs = np.array([0.6, 0.1, 0.7])
    merged = merge_events(t0, t1, fired, probs, max_gap=1, min_win=1)
    assert len(merged) == 1 and merged[0]["n_windows"] == 3 and merged[0]["n_fired"] == 2
    split = merge_events(t0, t1, fired, probs, max_gap=0, min_win=1)
    assert len(split) == 2


def test_merge_events_min_window_filter():
    t0 = np.array([0, 10, 20], float); t1 = np.array([10, 20, 30], float)
    fired = np.array([False, True, False]); probs = np.array([0.1, 0.6, 0.1])
    assert merge_events(t0, t1, fired, probs, max_gap=1, min_win=2) == []   # too short
    assert len(merge_events(t0, t1, fired, probs, max_gap=1, min_win=1)) == 1


def test_merge_events_empty():
    z = np.zeros(4, float)
    assert merge_events(z, z, np.zeros(4, bool), z) == []


# ── scope-drift guard (the p_2 regression class) ─────────────────────────────
def _viz_ef_heads():
    """Statically extract the EF_HEADS literal from viz_report_formats.py without
    executing its module-level plotting code (which needs data files)."""
    tree = ast.parse(open(VIZ).read())
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for tgt in node.targets:
                if isinstance(tgt, ast.Name) and tgt.id == "EF_HEADS":
                    return ast.literal_eval(node.value)
    raise AssertionError("EF_HEADS assignment not found in scripts/viz_report_formats.py")


def test_viz_heads_match_detection_scope():
    heads = {t[0] for t in _viz_ef_heads()}
    scope = set(L.scope_indices())
    assert heads == scope, (
        f"viz EF_HEADS {sorted(heads)} drifted from DETECTION_SCOPE {sorted(scope)}. "
        "A head in EF_HEADS but not scope crashes the visualizer (KeyError 'p_<h>'); "
        "a scope head missing from EF_HEADS is silently never plotted. "
        "Update scripts/viz_report_formats.py EF_HEADS to match.")
