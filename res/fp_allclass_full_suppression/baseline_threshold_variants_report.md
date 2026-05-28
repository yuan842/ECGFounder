# Baseline Threshold Variant Comparison

**Cohort**: 2641 records, up to 200/event-type
**Model**: Baseline (`./checkpoint/1_lead_ECGFounder.pth`)
**Thresholds**: 0.5, 0.6, 0.7
**Suppressors active**: ['Atrial Fibrillation', 'Bradycardia', 'Supraventricular Trigeminy', 'Ventricular Trigeminy']

All events are clinician-removed false positives → **lower alert rate is better**.

---

## Per-class results

| Event Type | n | route | t=0.5 raw% | t=0.5 post% | t=0.6 raw% | t=0.6 post% | t=0.7 raw% | t=0.7 post% |
|---|---|---|---|---|---|---|---|---|
| Atrial Fibrillation | 200 | Atrial Fibrillation | 71.0 | 0.0 | 59.5 | 0.0 | 41.0 | 0.0 |
| Isolated Supraventricular Beat | 200 | — | 11.5 | 11.5 | 7.5 | 7.5 | 5.0 | 5.0 |
| Isolated Ventricular Beat | 200 | — | 31.0 | 31.0 | 25.0 | 25.0 | 18.5 | 18.5 |
| Pause | 200 | — | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| Prolonged RR Interval | 17 | — | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| Sinus Tachycardia | 200 | — | 99.5 | 99.5 | 99.5 | 99.5 | 98.5 | 98.5 |
| Supraventricular Couplet | 200 | — | 1.5 | 1.5 | 0.5 | 0.5 | 0.5 | 0.5 |
| Ventricular Couplet | 200 | — | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| Ventricular Run | 200 | — | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| Bradycardia | 200 | Bradycardia | 69.5 | 12.0 | 63.5 | 10.5 | 51.5 | 8.5 |
| Custom Heart Rate | 22 | — | 81.8 | 81.8 | 77.3 | 77.3 | 50.0 | 50.0 |
| Multiple Event | 200 | — | 58.0 | 58.0 | 53.0 | 53.0 | 43.5 | 43.5 |
| Supraventricular Bigeminy | 1 | — | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| Supraventricular Run | 200 | — | 48.0 | 48.0 | 41.5 | 41.5 | 36.0 | 36.0 |
| Supraventricular Trigeminy | 200 | Supraventricular Trigeminy | 35.5 | 3.0 | 30.5 | 2.5 | 20.5 | 2.5 |
| Ventricular Tachycardia | 1 | — | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 |
| Ventricular Trigeminy | 200 | Ventricular Trigeminy | 58.0 | 0.5 | 45.0 | 0.5 | 32.5 | 0.5 |

---

## Aggregate (mapped/suppressed classes)

| Threshold | Raw FP rate | Post-suppression FP rate |
|---|---|---|
| 0.5 | 26.5% (429/1617) | **17.7%** (287/1617) |
| 0.6 | 23.7% (384/1617) | **16.4%** (265/1617) |
| 0.7 | 20.2% (327/1617) | **15.2%** (245/1617) |

### Suppressed classes only

| Threshold | Raw FP rate | Post-suppression FP rate |
|---|---|---|
| 0.5 | 58.5% (468/800) | **3.9%** (31/800) |
| 0.6 | 49.6% (397/800) | **3.4%** (27/800) |
| 0.7 | 36.4% (291/800) | **2.9%** (23/800) |

---

## Per-class post-suppression comparison (native-class heads only)

| Class | t=0.5 raw→post | t=0.6 raw→post | t=0.7 raw→post |
|---|---|---|---|
| Atrial Fibrillation | 71.0% → **0.0%** | 59.5% → **0.0%** | 41.0% → **0.0%** |
| Bradycardia | 69.5% → **12.0%** | 63.5% → **10.5%** | 51.5% → **8.5%** |
| Supraventricular Trigeminy | 35.5% → **3.0%** | 30.5% → **2.5%** | 20.5% → **2.5%** |
| Ventricular Trigeminy | 58.0% → **0.5%** | 45.0% → **0.5%** | 32.5% → **0.5%** |

---

## Key observations

**Suppressor version**: v2 feature-gate filters
**Active rules**: ['Atrial Fibrillation', 'Bradycardia', 'Supraventricular Trigeminy', 'Ventricular Trigeminy']

1. **Suppressed-class FP elimination is dramatic**: Aggregate FP rate for suppressed classes drops from 58.5% to 3.9% at t=0.5 — a 93% relative reduction. Holds across thresholds: 49.6% → 3.4% at t=0.6, 36.4% → 2.9% at t=0.7.

2. **Atrial Fibrillation: 100% FP suppression at all thresholds**: The mean_motion ≤ 5.0 mG gate eliminates every AFib false positive (71.0% → 0.0% at t=0.5, 59.5% → 0.0% at t=0.6, 41.0% → 0.0% at t=0.7).

3. **Ventricular Trigeminy: near-total FP suppression**: 58.0% → 0.5% at t=0.5 (99.1% of FPs eliminated). The dual gate (mean_motion ≥ 24.0 AND snr_proxy > 1.2) is highly effective — only 1 FP escapes at every threshold.

4. **Supraventricular Trigeminy: strong FP suppression**: 35.5% → 3.0% at t=0.5 (91.5% of FPs eliminated). The inverted motion gate (mean_motion ≥ 15.0 mG) correctly identifies that FP SV Trigeminy events have low motion.

5. **Bradycardia: significant but incomplete suppression**: 69.5% → 12.0% at t=0.5 (82.7% of FPs eliminated). The mean_hr_bpm ≤ 56.3 gate removes most FPs but ~12% persist. Residual FP rate improves slightly at higher thresholds (10.5% at t=0.6, 8.5% at t=0.7).

6. **Non-suppressed classes are unaffected**: All passthrough classes show raw% = post% across all thresholds, confirming zero collateral damage from v2 gates.

7. **Sinus Tachycardia FPs remain problematic**: 99.5% FP rate at t=0.5 (199/200 FPs detected) with no suppression rule active. This class may need its own feature gate in a future iteration.

8. **Threshold alone is insufficient for FP control**: Raising threshold from 0.5 to 0.7 reduces raw FP rate by only 6.3pp (26.5% → 20.2%) for mapped classes. The v2 gates provide far more FP reduction (26.5% → 17.7%) than any threshold adjustment alone.


Full CSV: `./res/fp_allclass_full_suppression/baseline_threshold_variants.csv`

*Generated from the same inference run as the main comparison.*