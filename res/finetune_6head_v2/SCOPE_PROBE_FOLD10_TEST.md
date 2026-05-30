# Scope-aligned 6-head linear probe — clean fold-10 test

**Date**: 2026-05-30
**Model**: ECGFounder base backbone (frozen) + linear probe on the **scope-aligned** heads
`[2, 4, 5, 6, 93, 98]` (= 7-label detection scope − Pause, which has 0 PTB-XL positives).
**Split**: the global rule — **train folds 1-8 / val fold 9 / test fold 10** (`ptbxl_splits`),
patient-stratified, 0 leakage. Data = PTB-XL 4-angle fuzzy set {45,60,75,90°}.
**Training**: 5 epochs, pos-weight BCE (caps imbalance), only 6,150 params trainable, backbone +
144 non-target classifier rows frozen. **Non-target row drift = 0.00e+00 every epoch** (the other
144 heads are byte-identical to base — production-safe).
**Checkpoint**: `checkpoint/1_lead_ECGFounder_6head_scope.pth`.
**Test set is touched once**, on the model selected by fold-9 validation.

## Fold-10 TEST — base vs scope-aligned probe (ROC, threshold-free)

| head | base ROC | **probe ROC** | Δ | probe PR-AUC | unique test pos |
|---|---|---|---|---|---|
| AFib (5) | 0.9757 | **0.9894** | +0.014 | 0.938 | 152 |
| Sinus Tachycardia (6) | 0.9857 | **0.9944** | +0.009 | 0.887 | 82 |
| Bradycardia (4) | 0.9440 | **0.9583** | +0.014 | 0.626 | 64 |
| **NORMAL ECG (2)** | 0.8133 | **0.8837** | **+0.070** | 0.833 | 963 |
| SV Run / SVT (93) | 0.9964 | 0.9976 | +0.001 | 0.338 | **5** ⚠️ |
| V Run / VT (98) | 0.9967 | 0.9977 | +0.001 | 0.349 | **5** ⚠️ |
| **6-head macro** | — | **0.9702** | — | 0.662 | — |

(Probe `pos` shown as **unique records**; the script's printed counts are ×4 angle-replicated, e.g. NORMAL 3852 = 963×4.)

## Reading

- **The probe improves every well-powered head**, and leaves the 144 others bit-identical:
  - **NORMAL ECG is the headline: +0.070 ROC (0.81 → 0.88)** — the most under-served head in the base model, now the biggest gain. (PR-AUC 0.83.)
  - AFib +0.014 (→0.989), Sinus Tachy +0.009 (→0.994), Bradycardia +0.014 (→0.958) — small but consistent lifts on already-strong heads.
- **SVT (93) / VT (98): ROC ≈ 0.998 but rest on only 5 unique positives** → statistically empty (95% CI on a rate at n=5 is ±~40%; their **PR-AUC ≈ 0.34** already exposes the precision problem). Treat as exploratory, not a real number. These need MIMIC pooling to reach a meaningful test count.
- **Validation (fold 9) was stable across epochs** (macro ROC 0.972 → 0.973); the model selected is epoch 5.

## Verdict

- **Clean, leakage-free fold-10 test: 6-head macro ROC 0.970.** On the four well-powered heads the
  scope-aligned probe beats the frozen base, headlined by **Normal ECG +0.07**, while keeping the
  other 144 heads exactly unchanged.
- **SVT/VT numbers are not trustworthy (n=5)** — the only honest fix is more positives from an
  external source (MIMIC-IV-ECG pipeline). Pause (142) remains untrainable/untestable on PTB-XL.

## Caveats
- Base ROC computed on PTB-XL fold-10 lead II (`for_ptbxl`); probe ROC on the 4-angle fuzzy test —
  both are base-backbone-feature readouts, so the ROC delta reflects the retrained head, but the
  preprocessing/angle sets are not byte-identical.
- pos-weight caps at 50; threshold-free metrics (ROC/PR) reported — operating-point (sens/spec at a
  threshold) not tuned here.
