# TP Retention — Baseline Threshold Variant Comparison

**Dataset**: `ecg_tp_fzark` — clinician-confirmed true positives
**Cohort**: 1988 records, up to 200/event-type (stratified sample)
**Model**: Baseline (`./checkpoint/1_lead_ECGFounder.pth`)
**Thresholds**: 0.5, 0.6, 0.7
**Suppressors active**: ['Atrial Fibrillation', 'Bradycardia', 'Supraventricular Trigeminy', 'Ventricular Trigeminy']

All events are clinician-confirmed true positives → **higher detection & retention is better**.

---

## 1. Baseline vs Fine-tuned at default thresholds

| Event Type | n | Baseline detect% | Baseline retain% | Fine-tuned detect% | Fine-tuned retain% |
|---|---|---|---|---|---|
| Atrial Fibrillation | 200 | 96.5 | 96.5 | 97.0 | 97.0 |
| Isolated Supraventricular Beat | 200 | 62.0 | 62.0 | 75.5 | 75.5 |
| Isolated Ventricular Beat | 200 | 82.0 | 82.0 | 92.0 | 92.0 |
| Pause | 82 | 0.0 | 0.0 | 0.0 | 0.0 |
| Prolonged RR Interval | 169 | 0.0 | 0.0 | 0.0 | 0.0 |
| Supraventricular Couplet | 200 | 0.5 | 0.5 | 2.0 | 2.0 |
| Ventricular Couplet | 36 | 0.0 | 0.0 | 0.0 | 0.0 |
| Ventricular Run | 200 | 0.0 | 0.0 | 0.0 | 0.0 |
| Bradycardia | 200 | 11.5 | 9.5 | 3.0 | 1.0 |
| Multiple Event | 116 | 62.9 | 62.9 | 48.3 | 48.3 |
| ST Elevation | 1 | 0.0 | 0.0 | 0.0 | 0.0 |
| Supraventricular Bigeminy | 64 | 92.2 | 92.2 | 75.0 | 75.0 |
| Supraventricular Run | 26 | 73.1 | 73.1 | 57.7 | 57.7 |
| Supraventricular Trigeminy | 76 | 22.4 | 22.4 | 3.9 | 3.9 |
| Unknown | 200 | 93.5 | 93.5 | 70.0 | 70.0 |
| Ventricular Bigeminy | 2 | 50.0 | 50.0 | 50.0 | 50.0 |
| Ventricular Trigeminy | 16 | 25.0 | 12.5 | 6.2 | 0.0 |

---

## 2. Baseline threshold variant comparison

| Event Type | n | t=0.5 det% | t=0.5 ret% | t=0.5 loss% | t=0.6 det% | t=0.6 ret% | t=0.6 loss% | t=0.7 det% | t=0.7 ret% | t=0.7 loss% |
|---|---|---|---|---|---|---|---|---|---|---|
| Atrial Fibrillation | 200 | 96.5 | 96.5 | 0.0 | 93.5 | 93.5 | 0.0 | 88.5 | 88.5 | 0.0 |
| Isolated Supraventricular Beat | 200 | 62.0 | 62.0 | 0.0 | 55.5 | 55.5 | 0.0 | 41.0 | 41.0 | 0.0 |
| Isolated Ventricular Beat | 200 | 82.0 | 82.0 | 0.0 | 79.0 | 79.0 | 0.0 | 73.5 | 73.5 | 0.0 |
| Pause | 82 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| Prolonged RR Interval | 169 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| Supraventricular Couplet | 200 | 0.5 | 0.5 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| Ventricular Couplet | 36 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| Ventricular Run | 200 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| Bradycardia | 200 | 11.5 | 9.5 | 17.4 | 9.0 | 7.0 | 22.2 | 6.0 | 4.5 | 25.0 |
| Multiple Event | 116 | 62.9 | 62.9 | 0.0 | 56.0 | 56.0 | 0.0 | 42.2 | 42.2 | 0.0 |
| ST Elevation | 1 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| Supraventricular Bigeminy | 64 | 92.2 | 92.2 | 0.0 | 90.6 | 90.6 | 0.0 | 89.1 | 89.1 | 0.0 |
| Supraventricular Run | 26 | 73.1 | 73.1 | 0.0 | 73.1 | 73.1 | 0.0 | 46.2 | 46.2 | 0.0 |
| Supraventricular Trigeminy | 76 | 22.4 | 22.4 | 0.0 | 11.8 | 11.8 | 0.0 | 5.3 | 5.3 | 0.0 |
| Unknown | 200 | 93.5 | 93.5 | 0.0 | 90.5 | 90.5 | 0.0 | 81.5 | 81.5 | 0.0 |
| Ventricular Bigeminy | 2 | 50.0 | 50.0 | 0.0 | 50.0 | 50.0 | 0.0 | 50.0 | 50.0 | 0.0 |
| Ventricular Trigeminy | 16 | 25.0 | 12.5 | 50.0 | 12.5 | 0.0 | 100.0 | 12.5 | 0.0 | 100.0 |

---

## 3. Aggregate TP retention

### All classes

| Threshold | Detection rate | Retention rate | Suppressor TP loss |
|---|---|---|---|
| 0.5 | 43.5% (865/1988) | **43.2%** (859/1988) | 0.7% (6/865) |
| 0.6 | 40.7% (809/1988) | **40.4%** (803/1988) | 0.7% (6/809) |
| 0.7 | 35.5% (706/1988) | **35.3%** (701/1988) | 0.7% (5/706) |

### Suppressed classes only (n=492)

| Threshold | Detection rate | Retention rate | Suppressor TP loss |
|---|---|---|---|
| 0.5 | 48.2% (237/492) | **47.0%** (231/492) | 2.5% (6/237) |
| 0.6 | 43.9% (216/492) | **42.7%** (210/492) | 2.8% (6/216) |
| 0.7 | 39.6% (195/492) | **38.6%** (190/492) | 2.6% (5/195) |

---

## 4. Per-class retention (native-class heads with suppressors)

| Class | n | t=0.5 det→ret | t=0.6 det→ret | t=0.7 det→ret |
|---|---|---|---|---|
| Atrial Fibrillation | 200 | 96.5% → **96.5%** | 93.5% → **93.5%** | 88.5% → **88.5%** |
| Bradycardia | 200 | 11.5% → **9.5%** | 9.0% → **7.0%** | 6.0% → **4.5%** |
| Supraventricular Trigeminy | 76 | 22.4% → **22.4%** | 11.8% → **11.8%** | 5.3% → **5.3%** |
| Ventricular Trigeminy | 16 | 25.0% → **12.5%** | 12.5% → **0.0%** | 12.5% → **0.0%** |

---

## 5. Key observations

1. **Suppressor TP loss is minimal overall**: Only 0.7% of detected TPs are lost to suppression at t=0.5 (6/865), holding steady across thresholds (0.7% at t=0.6, 0.7% at t=0.7). The v2 feature gates are highly selective.

2. **Atrial Fibrillation retention is perfect**: 0% TP loss across all thresholds — the mean_motion ≤ 5.0 mG gate preserves every clinician-confirmed AFib event.

3. **Supraventricular Trigeminy retention is perfect**: 0% TP loss at all thresholds — the inverted motion gate (mean_motion ≥ 15.0 mG) correctly identifies that TP SV Trigeminy events have high motion signatures.

4. **Bradycardia has moderate TP loss**: 17.4% of detected Bradycardia TPs are suppressed at t=0.5 (2 of 23 detected). The mean_hr_bpm ≤ 56.3 gate trades some TP retention for FP suppression (69.5% → 12.0% FP rate). Overall Bradycardia detection is already low (11.5%), so the absolute TP count affected is small.

5. **Ventricular Trigeminy has high TP loss but tiny sample**: 50% TP loss at t=0.5 (2 of 4 detected TPs suppressed). Total n=16 with only 25% baseline detection — results are statistically fragile. Re-evaluate when TP n > 50.

6. **Threshold increases erode detection far more than suppression**: Raising from t=0.5 to t=0.7 drops aggregate detection from 43.5% to 35.5% (−18.4% relative), but suppressor TP loss stays flat at ~0.7%. The dominant factor in TP retention is the model's detection threshold, not the v2 gates.

7. **Fine-tuned model outperforms baseline on key classes**: ISB (75.5% vs 62.0%), IVB (92.0% vs 82.0%), and AFib (97.0% vs 96.5%). However, it loses ground on SV Bigeminy (75.0% vs 92.2%), SV Run (57.7% vs 73.1%), and Unknown (70.0% vs 93.5%).

8. **Zero-detection classes remain undetectable**: Pause, Prolonged RR, V Couplet, V Run, and ST Elevation show 0% detection across both models and all thresholds — these event types are not in the 150-class model's effective vocabulary.


Full CSV: `./res/tp_fzark_full_suppression/tp_baseline_threshold_variants.csv`

*Generated from `ecg_tp_fzark` using the baseline model.*