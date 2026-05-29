# SQI quality gate prototype — Bradycardia + Pause

**Date**: 2026-05-29
**Motivation**: the TP/FP characterization ([res/sqi_motion_tp_fp/SQI_MOTION_TP_FP.md](../sqi_motion_tp_fp/SQI_MOTION_TP_FP.md))
showed Bradycardia and Pause false positives separate on **signal quality, not motion** — the
production motion/HR gates miss them. This adds an SQI quality gate for both.
**Calibrated on**: `res/sqi_motion_tp_fp/per_window.csv` (fzark TP vs fp_doctor), Youden-J via
`fp_suppression.fit_gate_supervised`. TP = keep, FP = drop. **No detection / inference.**
**Script**: [scripts/calibrate_sqi_gate.py](../../scripts/calibrate_sqi_gate.py) · thresholds in `sqi_gate_thresholds.csv`.

## Calibration

| event | candidate gate | threshold | TP-kept | FP-removed | Youden J |
|---|---|---|---|---|---|
| **Bradycardia** | `snr_proxy >= T` | **0.84** | **0.89** | **0.79** | 0.680 |
| Bradycardia | `hf_noise <= T` | 0.0129 | 0.79 | 0.66 | 0.450 |
| Bradycardia | snr ∧ hf (AND) | — | 0.73 | 0.92 | — |
| **Pause** | `hf_noise <= T` | **0.000257** | **1.00** | **1.00** | **0.998** |
| Pause | `snr_proxy >= T` | 0.802 | 0.80 | 0.85 | 0.653 |
| Pause | snr ∧ hf (AND) | — | 0.80 | 1.00 | — |

- **Pause is near-perfectly separable by HF-noise alone** — false pauses are HF-noise artifacts
  (TP hf-noise ≈ 0, FP median 0.14). `hf_noise <= 0.000257` keeps all 82 TP and removes all 500 FP.
- **Bradycardia's best single gate is SNR ≥ 0.84** (J=0.68). The AND with HF is more aggressive
  (FP-removed 0.92) but costs more TP (0.73) — we keep the single SNR gate as the prototype.

## What was wired in (`fp_suppression`)

New **`fzark_sqi`** device profile (production `fzark` left **unchanged** — equivalence test still passes):

```python
FZARK_SQI.sqi_gates['Bradycardia'] = [Gate('mean_hr_bpm','<=',56.3),   # production HR gate
                                      Gate('snr_proxy','>=',0.84)]      # + SQI quality gate
FZARK_SQI.sqi_gates['Pause']       = [Gate('hf_noise','<=',0.000257)]  # HF-noise artifact gate
```

Keep iff **all** of an event's gates pass — so Bradycardia is now kept only if it is both a
bradycardic rate **and** adequate SNR; Pause is kept only if HF-noise is negligible.

`hf_noise` was added to `multiclass_fp_suppression.extract_features` (welch >40 Hz power ratio,
same definition as `sqi.compute_raw_sqi`) so the gate is runnable from the production feature path.

## End-to-end validation (through the package, on the calibration data)

| event | TP-kept | FP-removed |
|---|---|---|
| Bradycardia (SQI gate, HR held bradycardic) | 0.88 | 0.79 |
| Pause | 1.00 | 1.00 |

## Caveats
- **Prototype, not production.** Thresholds are **fzark-device-specific** (raw-ECG welch / 5–25 Hz SNR).
- **Pause TP n=82** (small) — the perfect separation is on this sample; the 0.000257 threshold is
  effectively "any detectable HF noise → drop". Validate on more TP pauses before deployment.
- The Bradycardia SNR gate compounds with the HR gate (AND) — it trims ~11 % of TP for ~79 % of the
  remaining FP. Tune the operating point per recall requirement.
- Lives in `fzark_sqi`; the production `fzark` profile is untouched.
