# ECG-FP-Doctor-Removed1 — Per-Class Performance Comparison

**Cohort**: 2641 records, stratified up to 200/event-type
**Device**: mps
**Baseline threshold**: 0.5  |  **Fine-tuned threshold**: 0.2646

Every record is a clinician-removed false positive — lower alert rate is better.

## Per-class results

| Event Type | n | class idx | Baseline raw% | Baseline post% | Fine-tuned raw% | Fine-tuned post% |
|---|---|---|---|---|---|---|
| Atrial Fibrillation | 200 | 5 | 71.0 | 3.5 | 47.0 | 2.0 |
| Isolated Supraventricular Beat | 200 | 16 | 11.5 | nan | 40.0 | nan |
| Isolated Ventricular Beat | 200 | 9 | 31.0 | nan | 69.5 | nan |
| Pause | 200 | 143 | 0.0 | nan | 0.0 | nan |
| Prolonged RR Interval | 17 | 81 | 0.0 | nan | 0.0 | nan |
| Sinus Tachycardia | 200 | 6 | 99.5 | nan | 100.0 | nan |
| Supraventricular Couplet | 200 | 20 | 1.5 | nan | 4.5 | nan |
| Ventricular Couplet | 200 | 91 | 0.0 | nan | 0.5 | nan |
| Ventricular Run | 200 | 99 | 0.0 | nan | 0.0 | nan |
| Bradycardia  *(AFib-head only)* | 200 | 5 | 69.5 | 6.5 | 56.5 | 5.0 |
| Custom Heart Rate  *(AFib-head only)* | 22 | 5 | 81.8 | 0.0 | 54.5 | 0.0 |
| Multiple Event  *(AFib-head only)* | 200 | 5 | 58.0 | 4.0 | 50.5 | 3.5 |
| Supraventricular Bigeminy  *(AFib-head only)* | 1 | 5 | 0.0 | 0.0 | 0.0 | 0.0 |
| Supraventricular Run  *(AFib-head only)* | 200 | 5 | 48.0 | 1.0 | 42.0 | 1.0 |
| Supraventricular Trigeminy  *(AFib-head only)* | 200 | 5 | 35.5 | 0.0 | 14.5 | 0.0 |
| Ventricular Tachycardia  *(AFib-head only)* | 1 | 5 | 100.0 | 0.0 | 100.0 | 0.0 |
| Ventricular Trigeminy  *(AFib-head only)* | 200 | 5 | 58.0 | 4.0 | 26.5 | 1.5 |

Full CSV: `./res/fp_allclass_comparison/allclass_comparison.csv`