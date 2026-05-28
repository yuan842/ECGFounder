"""Tests for the `MultiClassFPSuppressor` on/off toggle.

The suppressor must be a true no-op when disabled — no JSON I/O, no feature
extraction, every alert preserved exactly.
"""
import os
import sys
from unittest.mock import patch

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from multiclass_fp_suppression import (
    ACTIVE_RULES,
    MultiClassFPSuppressor,
    SuppressionResult,
)


class TestEnabledDefault:
    def test_default_is_enabled(self):
        s = MultiClassFPSuppressor()
        assert s.enabled is True

    def test_supported_classes_when_enabled(self):
        s = MultiClassFPSuppressor()
        assert sorted(s.supported_classes()) == sorted(ACTIVE_RULES.keys())


class TestDisabled:
    def test_supported_classes_is_empty_when_disabled(self):
        s = MultiClassFPSuppressor(enabled=False)
        assert s.supported_classes() == []

    def test_disabled_returns_keep_true_for_supported_class(self):
        """When disabled, even AFib alerts (which DO have an active rule) pass."""
        s = MultiClassFPSuppressor(enabled=False)
        # Use a path that doesn't exist — if `enabled=False` short-circuits
        # correctly, feature extraction is never attempted.
        result = s.suppress_alert(
            "Atrial Fibrillation",
            p=0.95,
            json_path="/nonexistent/path/never_read.json",
        )
        assert isinstance(result, SuppressionResult)
        assert result.keep is True
        assert result.final_prob == 0.95
        assert result.reason == "disabled"
        assert result.features is None

    def test_disabled_short_circuits_before_feature_extraction(self):
        """The whole point of the toggle: zero JSON I/O when off."""
        s = MultiClassFPSuppressor(enabled=False)
        with patch("multiclass_fp_suppression.extract_features") as mock_extract:
            s.suppress_alert(
                "Atrial Fibrillation", p=0.95,
                json_path="/nonexistent/path.json",
            )
            mock_extract.assert_not_called()

    def test_disabled_preserves_prob_for_passthrough_class(self):
        """Even pass-through-only classes (no rule) work the same way."""
        s = MultiClassFPSuppressor(enabled=False)
        result = s.suppress_alert(
            "Sinus Tachycardia",  # no ACTIVE_RULES entry → already passthrough
            p=0.42,
            json_path="/nonexistent/path.json",
        )
        assert result.keep is True
        assert result.final_prob == 0.42
        assert result.reason == "disabled"

    @pytest.mark.parametrize("event_class", list(ACTIVE_RULES.keys()))
    def test_disabled_keeps_every_active_class(self, event_class):
        """Round-trip: every class that WOULD be filtered passes when off."""
        s = MultiClassFPSuppressor(enabled=False)
        with patch("multiclass_fp_suppression.extract_features") as mock_extract:
            result = s.suppress_alert(event_class, p=0.7, json_path="/x.json")
            mock_extract.assert_not_called()
        assert result.keep is True
        assert result.final_prob == 0.7


class TestEnabledStillFiltersWhenAppropriate:
    """Sanity: enabling the suppressor does NOT accidentally make it pass through."""

    def test_enabled_attempts_feature_extraction_for_active_class(self):
        s = MultiClassFPSuppressor(enabled=True)
        with patch(
            "multiclass_fp_suppression.extract_features", return_value=None,
        ) as mock_extract:
            result = s.suppress_alert(
                "Atrial Fibrillation", p=0.95,
                json_path="/some/path.json",
            )
            mock_extract.assert_called_once()
        # Feature extraction returned None → fail-open path: keep=True with the
        # "no_features" reason (distinct from the "disabled" reason).
        assert result.keep is True
        assert result.reason == "no_features (fail-open)"
