# Baseline Threshold Variant Comparison

**Cohort**: 6541 records, up to 500/event-type
**Model**: Baseline (`./checkpoint/1_lead_ECGFounder.pth`)
**Thresholds**: 0.5, 0.6, 0.7
**Suppressors active**: ['Atrial Fibrillation', 'Bradycardia', 'Supraventricular Trigeminy', 'Ventricular Trigeminy']

All events are clinician-removed false positives → **lower alert rate is better**.

---

## Per-class results

| Event Type | n | route | t=0.5 raw% | t=0.5 post% | t=0.6 raw% | t=0.6 post% | t=0.7 raw% | t=0.7 post% |
|---|---|---|---|---|---|---|---|---|
| Atrial Fibrillation | 500 | — | 72.0 | 72.0 | 61.8 | 61.8 | 49.2 | 49.2 |
| Bradycardia | 500 | — | 3.2 | 3.2 | 2.6 | 2.6 | 2.2 | 2.2 |
| Isolated Supraventricular Beat | 500 | — | 30.6 | 30.6 | 21.0 | 21.0 | 13.6 | 13.6 |
| Isolated Ventricular Beat | 500 | — | 25.4 | 25.4 | 18.6 | 18.6 | 12.4 | 12.4 |
| Pause | 500 | — | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| Sinus Tachycardia | 500 | — | 99.0 | 99.0 | 98.6 | 98.6 | 98.2 | 98.2 |
| Supraventricular Bigeminy | 1 | — | 100.0 | 100.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| Supraventricular Couplet | 500 | — | 2.2 | 2.2 | 1.0 | 1.0 | 0.4 | 0.4 |
| Supraventricular Run | 500 | — | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| Supraventricular Trigeminy | 500 | — | 67.4 | 67.4 | 54.8 | 54.8 | 40.6 | 40.6 |
| Ventricular Couplet | 500 | — | 19.6 | 19.6 | 15.2 | 15.2 | 10.6 | 10.6 |
| Ventricular Run | 500 | — | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| Custom Heart Rate | 22 | — | 81.8 | 81.8 | 68.2 | 68.2 | 59.1 | 59.1 |
| Multiple Event | 500 | — | 62.4 | 62.4 | 57.6 | 57.6 | 50.4 | 50.4 |
| Prolonged RR Interval | 17 | — | 17.6 | 17.6 | 5.9 | 5.9 | 0.0 | 0.0 |
| Ventricular Tachycardia | 1 | — | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 |
| Ventricular Trigeminy | 500 | — | 37.8 | 37.8 | 26.8 | 26.8 | 19.6 | 19.6 |

---

## Aggregate (mapped/suppressed classes)

| Threshold | Raw FP rate | Post-suppression FP rate |
|---|---|---|
| 0.5 | 29.0% (1598/5501) | **29.0%** (1598/5501) |
| 0.6 | 24.9% (1368/5501) | **24.9%** (1368/5501) |
| 0.7 | 20.7% (1136/5501) | **20.7%** (1136/5501) |

### Suppressed classes only

| Threshold | Raw FP rate | Post-suppression FP rate |
|---|---|---|
| 0.5 | nan% (0/0) | **nan%** (0/0) |
| 0.6 | nan% (0/0) | **nan%** (0/0) |
| 0.7 | nan% (0/0) | **nan%** (0/0) |

---

## Per-class post-suppression comparison (native-class heads only)

| Class | t=0.5 raw→post | t=0.6 raw→post | t=0.7 raw→post |
|---|---|---|---|
| Atrial Fibrillation | 72.0% → **72.0%** | 61.8% → **61.8%** | 49.2% → **49.2%** |
| Bradycardia | 3.2% → **3.2%** | 2.6% → **2.6%** | 2.2% → **2.2%** |
| Supraventricular Trigeminy | 67.4% → **67.4%** | 54.8% → **54.8%** | 40.6% → **40.6%** |
| Ventricular Trigeminy | 37.8% → **37.8%** | 26.8% → **26.8%** | 19.6% → **19.6%** |

---

## Key observations

*(Auto-populated from the data above — see per-class and aggregate tables.)*

**Suppressor version**: v2 feature-gate filters
**Active rules**: ['Atrial Fibrillation', 'Bradycardia', 'Supraventricular Trigeminy', 'Ventricular Trigeminy']


Full CSV: `./res/fp_allclass_full_suppression/baseline_threshold_variants_supp_off.csv`

*Generated from the same inference run as the main comparison.*