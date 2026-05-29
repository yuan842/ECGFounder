# `fp_suppression` — split FP-suppression (motion ⟂ SQI), device-parameterized

Splits the monolithic v2 false-positive suppressor (`multiclass_fp_suppression.py`)
into **two independently-trainable algorithms** whose parameters are **per-device**:

| Algorithm | Reads | Suppresses | Trained on |
|---|---|---|---|
| **MotionFPSuppressor** | accelerometer features (`mean_motion`, `std_motion`, `max_motion`) | motion-artifact FPs | (motion, label) pairs from a device's accelerometer |
| **SQIFPSuppressor** | ECG-intrinsic features (`snr_proxy`, `mean_hr_bpm`, `clip_pct`, …) | poor-signal / rate-implausible FPs | (ECG-feature, label) pairs from a device's electrode |

The two never share features, so they are trained and updated **separately** — recalibrate
a new chest strap's motion gate without touching the SQI gates, or run SQI-only on a
dataset with no accelerometer (e.g. PTB-XL).

## How the split maps to the production rules

Production keeps an alert iff **all** of an event's gates pass. We partition those gates
by feature family and **AND the two per-family decisions** — algebraically identical:

| Event | Motion gate | SQI gate |
|---|---|---|
| Atrial Fibrillation | `mean_motion ≤ T` | — |
| Supraventricular Trigeminy | `mean_motion ≥ T` | — |
| Ventricular Trigeminy | `mean_motion ≥ T` | `snr_proxy > T` |
| Bradycardia | — | `mean_hr_bpm ≤ T` |

> **HR note:** heart rate is a *physiologic-rate* gate, not signal "quality", but it is
> ECG-derived (not motion-derived), so it lives in the SQI/ECG-intrinsic algo within a
> two-way split. Promotable to its own family later without touching the motion algo.

`tests/test_split_equivalence.py` proves the `fzark` profile through the pipeline reproduces
`multiclass_fp_suppression` across 2,016 boundary-spanning feature combos.

## Device profiles (`device_profiles.py`)

- **`fzark`** — production baseline (Vivalink Holter; AFib≤5, SV-Trig≥15, V-Trig≥24∧snr>1.2, Brady≤56.3).
- **`move_chest_gel`** — MOVE chest `ecg:gel`; AFib motion gate recalibrated **5.0 → 1.0 mG**
  (rest-envelope p99, see `res/move_eval/MOTION_GATE_RECALIBRATION.md`). SV/V-Trig motion and
  the SNR gate are **not yet device-calibrated** for MOVE (flagged in the profile notes).

## Training (`calibration.py`)

```python
from fp_suppression import fit_gate_supervised, fit_gate_envelope, calibrate_profile_event, get_profile

# supervised (needs TP/FP labels) — picks threshold maximising Youden's J for keep/drop
gate, stats = fit_gate_supervised(motion_vals, is_tp, 'mean_motion', '<=')

# label-free (rest-motion envelope) — for suppress-high gates only
gate, stats = fit_gate_envelope(rest_motion_vals, 'mean_motion', '<=', pct=99.0)

# write into a device's correct family table (other family untouched)
prof = get_profile('move_chest_gel').copy('move_chest_gel_v2')
calibrate_profile_event(prof, 'Atrial Fibrillation', gate)
```

## Usage

```python
from fp_suppression import FPSuppressionPipeline, MotionFPSuppressor, SQIFPSuppressor

pipe = FPSuppressionPipeline(device='fzark')            # both families
res  = pipe.suppress('Atrial Fibrillation', feats)      # res.keep, res.reason

MotionFPSuppressor('move_chest_gel').suppress(ev, feats)  # one family alone
FPSuppressionPipeline('fzark', motion=False)              # SQI-only (no-accelerometer datasets)
```

Status: new package, side-by-side with the unchanged production `multiclass_fp_suppression.py`.
Wiring the pipeline into the inference path is a follow-up (not done here).
