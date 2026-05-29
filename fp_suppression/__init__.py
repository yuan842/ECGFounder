"""Split FP-suppression: two independently-trainable, device-parameterized algos.

    from fp_suppression import FPSuppressionPipeline, MotionFPSuppressor, SQIFPSuppressor

    pipe = FPSuppressionPipeline(device="fzark")          # both families, prod params
    res  = pipe.suppress("Atrial Fibrillation", feats)    # res.keep, res.reason

    motion_only = MotionFPSuppressor("move_chest_gel")    # train/run one family alone
    sqi_only    = SQIFPSuppressor("fzark")

Architecture
------------
  gates.py            Gate primitive + AND keep-logic (fail-open on missing feature)
  device_profiles.py  per-device gate tables (fzark = production; move_chest_gel)
  features.py         extract_motion_features / extract_sqi_features (split data sources)
  suppressors.py      MotionFPSuppressor, SQIFPSuppressor
  pipeline.py         FPSuppressionPipeline — AND-composes both families
  calibration.py      per-device threshold fitting (supervised + label-free envelope)

`fzark` profile through the pipeline == production v2 (multiclass_fp_suppression);
verified by tests/test_split_equivalence.py.
"""
from .gates import Gate, Decision, apply_gates, MOTION_FEATURES, SQI_FEATURES
from .device_profiles import (DeviceProfile, get_profile, register_profile,
                              list_devices, FZARK, MOVE_CHEST_GEL, DEFAULT_DEVICE)
from .features import (extract_motion_features, extract_sqi_features,
                       extract_all_features)
from .suppressors import MotionFPSuppressor, SQIFPSuppressor
from .pipeline import FPSuppressionPipeline, PipelineResult
from .calibration import (fit_gate_supervised, fit_gate_envelope,
                          calibrate_profile_event)

__all__ = [
    "Gate", "Decision", "apply_gates", "MOTION_FEATURES", "SQI_FEATURES",
    "DeviceProfile", "get_profile", "register_profile", "list_devices",
    "FZARK", "MOVE_CHEST_GEL", "DEFAULT_DEVICE",
    "extract_motion_features", "extract_sqi_features", "extract_all_features",
    "MotionFPSuppressor", "SQIFPSuppressor",
    "FPSuppressionPipeline", "PipelineResult",
    "fit_gate_supervised", "fit_gate_envelope", "calibrate_profile_event",
]
