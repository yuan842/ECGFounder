"""Feature extraction split into the two trainable families.

The production extractor (multiclass_fp_suppression.extract_features) reads one
JSON window and returns motion + ECG features together. Here we expose them as
TWO separate functions so each algorithm can be trained on its own data source:

  • extract_motion_features → needs ONLY the accelerometer stream
  • extract_sqi_features    → needs ONLY the ECG stream

Both delegate to the validated production extractor and slice the result, so the
numbers match production exactly. A caller that already has features (e.g. a
training DataFrame row) can skip these and pass a plain dict straight to a
suppressor's .suppress().
"""
from __future__ import annotations
from typing import Dict, Optional

from multiclass_fp_suppression import extract_features as _extract_all
from .gates import MOTION_FEATURES, SQI_FEATURES


def extract_motion_features(json_path: str) -> Optional[Dict[str, float]]:
    feats = _extract_all(json_path)
    if feats is None:
        return None
    return {k: v for k, v in feats.items() if k in MOTION_FEATURES}


def extract_sqi_features(json_path: str) -> Optional[Dict[str, float]]:
    feats = _extract_all(json_path)
    if feats is None:
        return None
    return {k: v for k, v in feats.items() if k in SQI_FEATURES}


def extract_all_features(json_path: str) -> Optional[Dict[str, float]]:
    """Full panel — convenience for the pipeline (one JSON read, both families)."""
    return _extract_all(json_path)
