# TP Retention — Baseline Threshold Variant Comparison

**Dataset**: `ecg_tp_fzark` — clinician-confirmed true positives
**Cohort**: 4088 records, up to 500/event-type (stratified sample)
**Model**: Baseline (`./checkpoint/1_lead_ECGFounder.pth`)
**Thresholds**: 0.5, 0.6, 0.7
**Suppression**: OFF
**Suppressors active**: []

All events are clinician-confirmed true positives → **higher detection & retention is better**.

---

## 1. Baseline at default threshold

| Event Type | n | Baseline detect% | Baseline retain% |
|---|---|---|---|
| Atrial Fibrillation | 500 | 97.8 | 97.8 |
| Bradycardia | 500 | 94.4 | 94.4 |
| Isolated Supraventricular Beat | 500 | 53.6 | 53.6 |
| Isolated Ventricular Beat | 500 | 78.6 | 78.6 |
| Pause | 82 | 0.0 | 0.0 |
| ST Elevation | 1 | 0.0 | 0.0 |
| Supraventricular Bigeminy | 64 | 78.1 | 78.1 |
| Supraventricular Couplet | 500 | 4.4 | 4.4 |
| Supraventricular Run | 26 | 0.0 | 0.0 |
| Supraventricular Trigeminy | 76 | 96.1 | 96.1 |
| Ventricular Couplet | 36 | 80.6 | 80.6 |
| Ventricular Run | 500 | 0.0 | 0.0 |
| Multiple Event | 116 | 56.9 | 56.9 |
| Prolonged RR Interval | 169 | 18.3 | 18.3 |
| Unknown | 500 | 91.0 | 91.0 |
| Ventricular Bigeminy | 2 | 50.0 | 50.0 |
| Ventricular Trigeminy | 16 | 6.2 | 6.2 |

---

## 2. Baseline threshold variant comparison

| Event Type | n | t=0.5 det% | t=0.5 ret% | t=0.5 loss% | t=0.6 det% | t=0.6 ret% | t=0.6 loss% | t=0.7 det% | t=0.7 ret% | t=0.7 loss% |
|---|---|---|---|---|---|---|---|---|---|---|
| Atrial Fibrillation | 500 | 97.8 | 97.8 | 0.0 | 96.2 | 96.2 | 0.0 | 94.0 | 94.0 | 0.0 |
| Bradycardia | 500 | 94.4 | 94.4 | 0.0 | 94.0 | 94.0 | 0.0 | 93.6 | 93.6 | 0.0 |
| Isolated Supraventricular Beat | 500 | 53.6 | 53.6 | 0.0 | 43.6 | 43.6 | 0.0 | 32.0 | 32.0 | 0.0 |
| Isolated Ventricular Beat | 500 | 78.6 | 78.6 | 0.0 | 74.4 | 74.4 | 0.0 | 69.8 | 69.8 | 0.0 |
| Pause | 82 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| ST Elevation | 1 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| Supraventricular Bigeminy | 64 | 78.1 | 78.1 | 0.0 | 65.6 | 65.6 | 0.0 | 51.6 | 51.6 | 0.0 |
| Supraventricular Couplet | 500 | 4.4 | 4.4 | 0.0 | 1.4 | 1.4 | 0.0 | 0.8 | 0.8 | 0.0 |
| Supraventricular Run | 26 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| Supraventricular Trigeminy | 76 | 96.1 | 96.1 | 0.0 | 89.5 | 89.5 | 0.0 | 76.3 | 76.3 | 0.0 |
| Ventricular Couplet | 36 | 80.6 | 80.6 | 0.0 | 61.1 | 61.1 | 0.0 | 55.6 | 55.6 | 0.0 |
| Ventricular Run | 500 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| Multiple Event | 116 | 56.9 | 56.9 | 0.0 | 52.6 | 52.6 | 0.0 | 48.3 | 48.3 | 0.0 |
| Prolonged RR Interval | 169 | 18.3 | 18.3 | 0.0 | 11.2 | 11.2 | 0.0 | 5.9 | 5.9 | 0.0 |
| Unknown | 500 | 91.0 | 91.0 | 0.0 | 77.6 | 77.6 | 0.0 | 56.0 | 56.0 | 0.0 |
| Ventricular Bigeminy | 2 | 50.0 | 50.0 | 0.0 | 50.0 | 50.0 | 0.0 | 50.0 | 50.0 | 0.0 |
| Ventricular Trigeminy | 16 | 6.2 | 6.2 | 0.0 | 6.2 | 6.2 | 0.0 | 6.2 | 6.2 | 0.0 |

---

## 3. Aggregate TP retention

### All classes

| Threshold | Detection rate | Retention rate | Suppressor TP loss |
|---|---|---|---|
| 0.5 | 57.5% (2350/4088) | **57.5%** (2350/4088) | 0.0% (0/2350) |
| 0.6 | 52.6% (2150/4088) | **52.6%** (2150/4088) | 0.0% (0/2150) |
| 0.7 | 46.7% (1910/4088) | **46.7%** (1910/4088) | 0.0% (0/1910) |

### Suppressed classes only (n=0)

| Threshold | Detection rate | Retention rate | Suppressor TP loss |
|---|---|---|---|
| 0.5 | nan% (0/0) | **nan%** (0/0) | 0.0% (0/0) |
| 0.6 | nan% (0/0) | **nan%** (0/0) | 0.0% (0/0) |
| 0.7 | nan% (0/0) | **nan%** (0/0) | 0.0% (0/0) |

---

## 4. Per-class retention (native-class heads with suppressors)

| Class | n | t=0.5 det→ret | t=0.6 det→ret | t=0.7 det→ret |
|---|---|---|---|---|
| Atrial Fibrillation | 500 | 97.8% → **97.8%** | 96.2% → **96.2%** | 94.0% → **94.0%** |
| Bradycardia | 500 | 94.4% → **94.4%** | 94.0% → **94.0%** | 93.6% → **93.6%** |
| Supraventricular Trigeminy | 76 | 96.1% → **96.1%** | 89.5% → **89.5%** | 76.3% → **76.3%** |
| Ventricular Trigeminy | 16 | 6.2% → **6.2%** | 6.2% → **6.2%** | 6.2% → **6.2%** |

---

## 5. Key observations

*(Populated after run — see console output for details.)*


Full CSV: `./res/tp_fzark_full_suppression/tp_baseline_threshold_variants_supp_off.csv`

*Generated from `ecg_tp_fzark` using the baseline model.*