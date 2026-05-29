# Suppressor Recalibration Report

**TP source**: `ecg_tp_fzark` (clinician-confirmed true positives)
**FP source**: `ecg_fp_doctor removed1` (clinician-removed false positives)
**Persist**: NO — dry run

---

## Summary

| Class | n_TP | n_FP | Current thr | Current TP% | Current FP% | Proposed thr | Proposed TP% | Proposed FP% | Δ TP | Δ FP |
|---|---|---|---|---|---|---|---|---|---|---|
| Atrial Fibrillation | 800 | 300 | 0.475 | 56.9% | 96.7% | **0.787** | **66.6%** | **91.7%** | +9.8 | -5.0 |
| Isolated Ventricular Beat | 800 | 300 | 0.675 | 89.9% | 80.3% | **0.527** | **86.6%** | **88.0%** | -3.2 | +7.7 |
| Isolated Supraventricular Beat | 800 | 300 | 0.600 | 90.5% | 11.7% | **0.600** | **90.5%** | **11.7%** | +0.0 | +0.0 |

---

## Methodology

1. Extract features from both TP and FP cohorts using the existing suppressor pipeline
2. Measure Layer 1 (hard rules) pass rates on both cohorts
3. For events passing Layer 1, collect Layer 2 (LR) scores
4. Sweep Layer 2 threshold to find optimal operating points:
   - **AFib**: Target ≥95% TP retention (was 55% — severely over-suppressing)
   - **IVB**: Target ≥90% TP retention (balanced)
   - **ISB**: Keep current (already ≥97% TP retention)
5. Report combined (L1 + L2) performance at proposed thresholds

Per-class threshold sweep CSVs are in `res/suppressor_recalibration/`.
