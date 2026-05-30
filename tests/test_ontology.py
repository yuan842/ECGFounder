"""Tests for the v3 single-index label ontology in `label_config`.

These tests would have caught the four off-by-one bugs at commit time, and
they enforce the v3 single-head invariant going forward.
"""
import os
import sys

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from label_config import (
    ClinicalOntologyNode,
    ClinicalRiskTier,
    DETECTION_SCOPE,
    SCOPE_EVENTS,
    HEAD_THRESHOLDS,
    head_threshold,
    scope_events,
    event_in_scope,
    require_in_scope,
    _assert_scope_consistency,
    FZARK_LABEL_MAP,
    FZARK_ONTOLOGY,
    FZARK_UNMAPPABLE,
    MITDB_BEAT_MAP,
    MITDB_DEFAULT_NORMAL,
    MITDB_RHYTHM_MAP,
    PTBXL_ACTIVE_CLASSES,
    TASKS_SHA256,
    detect,
    detect_index,
    get_index,
    in_scope,
    is_supported,
    load_tasks,
    scope_indices,
)


# ─── detection scope ────────────────────────────────────────────────────────

class TestDetectionScope:
    def test_scope_indices(self):
        assert scope_indices() == frozenset({4, 5, 6, 93, 98, 142})

    def test_scope_events_hard_rule(self):
        # global hard rule: exactly these 6 fzark events
        assert scope_events() == frozenset({
            "Atrial Fibrillation", "Bradycardia", "Sinus Tachycardia",
            "Supraventricular Run", "Ventricular Run", "Pause"})
        assert len(SCOPE_EVENTS) == 6
        assert event_in_scope("Atrial Fibrillation")
        assert not event_in_scope("Isolated Ventricular Beat")

    def test_require_in_scope_raises_out_of_scope(self):
        assert require_in_scope("Pause") == "Pause"
        with pytest.raises(ValueError):
            require_in_scope("Isolated Ventricular Beat")

    def test_scope_consistency_invariant_holds(self):
        _assert_scope_consistency()  # the import-time hard check, callable
        # the two views must agree: scope-event heads == DETECTION_SCOPE keys
        assert {FZARK_LABEL_MAP[e] for e in SCOPE_EVENTS} == set(DETECTION_SCOPE)

    def test_scope_names_match_tasks_txt(self):
        tasks = load_tasks()
        for idx, name in DETECTION_SCOPE.items():
            assert tasks[idx] == name, (idx, name, tasks[idx])

    def test_scope_indices_match_ontology(self):
        # the run/pause heads in scope must agree with FZARK_LABEL_MAP
        assert FZARK_LABEL_MAP["Supraventricular Run"] == 93
        assert FZARK_LABEL_MAP["Ventricular Run"] == 98
        assert FZARK_LABEL_MAP["Pause"] == 142

    def test_in_scope_heads_detect(self):
        probs = [1.0] * 150
        assert detect(probs, "Atrial Fibrillation") is True     # head 5
        assert detect(probs, "Bradycardia") is True             # head 4
        assert detect(probs, "Sinus Tachycardia") is True       # head 6
        assert detect(probs, "Supraventricular Run") is True    # head 93
        assert detect(probs, "Ventricular Run") is True         # head 98
        assert detect(probs, "Pause") is True                   # head 142

    def test_out_of_scope_heads_return_none(self):
        probs = [1.0] * 150
        # mapped events whose heads are NOT in scope → None (not a false negative)
        assert detect(probs, "Isolated Ventricular Beat") is None     # head 9
        assert detect(probs, "Isolated Supraventricular Beat") is None  # head 16
        assert detect_index(probs, 9) is None
        assert detect_index(probs, 98) is True
        assert in_scope(142) and not in_scope(9)

    def test_per_head_threshold_values(self):
        assert head_threshold(142) == 0.006   # Pause (calibrated)
        assert head_threshold(93) == 0.040    # SV Run (calibrated)
        assert head_threshold(98) == 0.5      # VT — NOT overridden (worse-than-chance)
        assert head_threshold(5) == 0.5       # AFib — default

    def test_calibrated_threshold_applied_by_default(self):
        # Pause head 142 fires at a low score (0.01 > 0.006) under its calibrated thr,
        probs = [0.0] * 150
        probs[142] = 0.01
        assert detect(probs, "Pause") is True
        assert detect_index(probs, 142) is True
        # ...but the same 0.01 at an un-overridden head (AFib) stays below 0.5.
        probs2 = [0.0] * 150
        probs2[5] = 0.01
        assert detect(probs2, "Atrial Fibrillation") is False
        # explicit threshold override still wins
        assert detect(probs, "Pause", threshold=0.5) is False


# ─── tasks.txt integrity ────────────────────────────────────────────────────

class TestTasksIntegrity:
    def test_sha256_is_pinned_not_placeholder(self):
        assert TASKS_SHA256 != "<pin to known-good hash>", (
            "TASKS_SHA256 must be pinned to the real hash."
        )
        assert len(TASKS_SHA256) == 64, "SHA256 hash must be 64 hex chars"

    def test_load_tasks_returns_150_entries(self):
        tasks = load_tasks()
        assert len(tasks) == 150, f"expected 150 classes, got {len(tasks)}"

    def test_load_tasks_strips_empty_lines(self):
        tasks = load_tasks()
        assert all(line and not line.isspace() for line in tasks)


# ─── Ontology contents (would have caught the off-by-one bugs) ──────────────

# A test fails if the head an event maps to does NOT contain the expected
# substring. The substring is case-insensitive.
EXPECTED_LABEL_SUBSTRINGS = {
    "Atrial Fibrillation":            "atrial fibrillation",
    "Sinus Tachycardia":              "sinus tachycardia",
    "Bradycardia":                    "sinus bradycardia",
    "ST Elevation":                   "st elevation",
    "Isolated Ventricular Beat":      "premature ventricular",
    "Isolated Supraventricular Beat": "premature atrial",
    "Supraventricular Couplet":       "premature supraventricular",
    "Ventricular Run":                "ventricular tachycardia",
    "Pause":                          "sinus pause",
    "Supraventricular Run":           "supraventricular tachycardia",
    # v3.1 beat-level recoveries — route to the constituent-beat head.
    "Supraventricular Trigeminy":     "premature atrial",
    "Supraventricular Bigeminy":      "premature atrial",
    "Ventricular Couplet":            "premature ventricular",
}


class TestOntologyContents:
    @pytest.mark.parametrize("event_name", list(EXPECTED_LABEL_SUBSTRINGS))
    def test_index_resolves_to_expected_label(self, event_name):
        tasks = load_tasks()
        node = FZARK_ONTOLOGY[event_name]
        label = tasks[node.ecgfounder_index].lower()
        needle = EXPECTED_LABEL_SUBSTRINGS[event_name]
        assert needle in label, (
            f"{event_name} → idx {node.ecgfounder_index} ({label!r}) "
            f"does not contain {needle!r}"
        )

    def test_all_indices_in_range(self):
        tasks = load_tasks()
        n = len(tasks)
        for name, node in FZARK_ONTOLOGY.items():
            assert 0 <= node.ecgfounder_index < n, (
                f"{name}: idx {node.ecgfounder_index} out of range [0, {n})"
            )

    def test_event_set_matches_spec(self):
        """v3.1 commits to exactly 13 supported events. Drift triggers failure."""
        assert len(FZARK_ONTOLOGY) == 13
        expected = set(EXPECTED_LABEL_SUBSTRINGS)
        assert set(FZARK_ONTOLOGY) == expected, (
            f"Stage v3.1 event set drift. "
            f"Missing: {expected - set(FZARK_ONTOLOGY)}, "
            f"Unexpected: {set(FZARK_ONTOLOGY) - expected}"
        )

    def test_all_semantic_matches_are_exact_or_good(self):
        for name, node in FZARK_ONTOLOGY.items():
            assert node.semantic_match in ("exact", "good"), (
                f"{name}: semantic_match={node.semantic_match!r}"
            )

    def test_risk_tier_is_valid_enum_value(self):
        for name, node in FZARK_ONTOLOGY.items():
            assert isinstance(node.risk_tier, ClinicalRiskTier), (
                f"{name}: risk_tier must be a ClinicalRiskTier enum"
            )


# ─── v3 invariants (single-head only; no composite events) ─────────────────

# v3.1: SV Trigeminy, SV Bigeminy, V Couplet were promoted out of this set
# because their constituent-beat head (idx 16 PAC / idx 9 PVC) fires strongly
# enough on the fzark TP cohort to be useful as event-level detectors. The
# events below remain unmappable because no 150-class head shows strong
# activation on them.
COMPOSITE_EVENTS = {
    "Ventricular Bigeminy",
    "Ventricular Trigeminy",
    "Prolonged RR Interval",
}

META_CATEGORIES = {"Multiple Event", "Unknown", "Custom Heart Rate"}


class TestV3Invariants:
    def test_ontology_and_unmappable_are_disjoint(self):
        overlap = set(FZARK_ONTOLOGY) & FZARK_UNMAPPABLE
        assert not overlap, (
            f"Events in both supported and unmappable sets: {overlap}"
        )

    def test_no_composite_events_in_supported_ontology(self):
        """v3 invariant: composite events must remain in FZARK_UNMAPPABLE.

        A future engineer cannot accidentally re-add a multi-head event to the
        ontology without removing it from this guard, which is a deliberate
        decision rather than an accident.
        """
        leaked = COMPOSITE_EVENTS & set(FZARK_ONTOLOGY)
        assert not leaked, (
            f"Composite events leaked into single-head ontology: {leaked}. "
            f"Per v3 these must stay in FZARK_UNMAPPABLE."
        )

    def test_all_composite_events_are_unmappable(self):
        missing = COMPOSITE_EVENTS - FZARK_UNMAPPABLE
        assert not missing, (
            f"Composite events should be unmappable but aren't: {missing}"
        )

    def test_all_meta_categories_are_unmappable(self):
        missing = META_CATEGORIES - FZARK_UNMAPPABLE
        assert not missing, (
            f"Meta-categories should be unmappable but aren't: {missing}"
        )


# ─── Backward-compat dict (FZARK_LABEL_MAP) ────────────────────────────────

class TestLabelMapDict:
    def test_label_map_matches_ontology_indices(self):
        for name, idx in FZARK_LABEL_MAP.items():
            assert FZARK_ONTOLOGY[name].ecgfounder_index == idx

    def test_label_map_has_exactly_the_supported_events(self):
        assert set(FZARK_LABEL_MAP) == set(FZARK_ONTOLOGY)


# ─── MIT-BIH maps ──────────────────────────────────────────────────────────

class TestMitdbMaps:
    def test_beat_map_S_j_route_to_psvc_not_pac(self):
        """v3 change vs v1: S and j must hit idx 19 (PSVC), not idx 16 (PAC)."""
        assert MITDB_BEAT_MAP["S"] == 19
        assert MITDB_BEAT_MAP["j"] == 19
        assert MITDB_BEAT_MAP["A"] == 16
        assert MITDB_BEAT_MAP["a"] == 16

    def test_beat_map_indices_are_in_range(self):
        tasks = load_tasks()
        for sym, idx in MITDB_BEAT_MAP.items():
            assert 0 <= idx < len(tasks), f"{sym}: idx {idx} out of range"

    def test_rhythm_map_includes_v3_additions(self):
        """v3 adds VT / AFL / SVTA / SBR; AFIB carries over from v1."""
        for token in ("(AFIB", "(VT", "(AFL", "(SVTA", "(SBR"):
            assert token in MITDB_RHYTHM_MAP, f"missing rhythm {token}"

    def test_rhythm_map_excludes_composite_events(self):
        """v3 invariant: bigeminy / trigeminy rhythms must NOT have mappings."""
        for token in ("(B", "(T", "(AB"):
            assert token not in MITDB_RHYTHM_MAP, (
                f"{token} is a composite rhythm; must remain unmapped in v3"
            )

    def test_rhythm_indices_match_label_substrings(self):
        """Each MITDB rhythm must point to a 150-class entry with a related name."""
        tasks = load_tasks()
        expected = {
            "(AFIB": "atrial fibrillation",
            "(VT":   "ventricular tachycardia",
            "(AFL":  "atrial flutter",
            "(SVTA": "supraventricular tachycardia",
            "(SBR":  "sinus bradycardia",
        }
        for token, substring in expected.items():
            idx = MITDB_RHYTHM_MAP[token]
            label = tasks[idx].lower()
            assert substring in label, (
                f"{token} → idx {idx} ({label!r}) does not contain {substring!r}"
            )

    def test_default_normal_indices(self):
        assert MITDB_DEFAULT_NORMAL == (1, 2)


# ─── PTB-XL active class set ───────────────────────────────────────────────

class TestPtbxlActive:
    def test_active_set_has_31_classes(self):
        assert len(PTBXL_ACTIVE_CLASSES) == 31

    def test_active_indices_are_in_range(self):
        for idx in PTBXL_ACTIVE_CLASSES:
            assert 0 <= idx < 150


# ─── Detection helpers ─────────────────────────────────────────────────────

class TestDetectionHelpers:
    def test_get_index_returns_correct_index_for_supported(self):
        assert get_index("Atrial Fibrillation") == 5
        assert get_index("Ventricular Run") == 98
        assert get_index("Pause") == 142

    def test_get_index_returns_none_for_unsupported(self):
        assert get_index("Ventricular Bigeminy") is None
        assert get_index("Prolonged RR Interval") is None
        assert get_index("not a real event") is None

    def test_is_supported(self):
        assert is_supported("Atrial Fibrillation")
        assert not is_supported("Ventricular Bigeminy")
        assert not is_supported("Unknown")

    def test_v31_recoveries_are_supported(self):
        """v3.1: SV Trigeminy / SV Bigeminy / V Couplet routed via beat head."""
        assert get_index("Supraventricular Trigeminy") == 16
        assert get_index("Supraventricular Bigeminy") == 16
        assert get_index("Ventricular Couplet") == 9

    def test_detect_returns_bool_for_supported(self):
        probs = [0.0] * 150
        probs[5] = 0.9
        assert detect(probs, "Atrial Fibrillation", threshold=0.5) is True
        probs[5] = 0.1
        assert detect(probs, "Atrial Fibrillation", threshold=0.5) is False

    def test_detect_returns_none_for_unsupported(self):
        probs = [0.0] * 150
        assert detect(probs, "Ventricular Trigeminy") is None
        assert detect(probs, "Unknown") is None
