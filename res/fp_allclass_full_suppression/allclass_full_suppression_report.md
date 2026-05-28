# All-Class FP Comparison with Full Per-Class Suppression

**Cohort**: 2641 records, up to 200/event-type
**Device**: mps
**Suppressors active**: ['Atrial Fibrillation', 'Bradycardia', 'Supraventricular Trigeminy', 'Ventricular Trigeminy']

## Per-class results

| Event Type | n | route | Baseline raw% | Baseline post% | Fine-tuned raw% | Fine-tuned post% |
|---|---|---|---|---|---|---|
| Atrial Fibrillation | 200 | Atrial Fibrillation | 71.0 | 0.0 | 47.0 | 0.0 |
| Isolated Supraventricular Beat | 200 | — | 11.5 | nan | 40.0 | nan |
| Isolated Ventricular Beat | 200 | — | 31.0 | nan | 69.5 | nan |
| Pause | 200 | — | 0.0 | nan | 0.0 | nan |
| Prolonged RR Interval | 17 | — | 0.0 | nan | 0.0 | nan |
| Sinus Tachycardia | 200 | — | 99.5 | nan | 100.0 | nan |
| Supraventricular Couplet | 200 | — | 1.5 | nan | 4.5 | nan |
| Ventricular Couplet | 200 | — | 0.0 | nan | 0.5 | nan |
| Ventricular Run | 200 | — | 0.0 | nan | 0.0 | nan |
| Bradycardia | 200 | Bradycardia | 69.5 | 12.0 | 56.5 | 8.5 |
| Custom Heart Rate | 22 | — | 81.8 | nan | 54.5 | nan |
| Multiple Event | 200 | — | 58.0 | nan | 50.5 | nan |
| Supraventricular Bigeminy | 1 | — | 0.0 | nan | 0.0 | nan |
| Supraventricular Run | 200 | — | 48.0 | nan | 42.0 | nan |
| Supraventricular Trigeminy | 200 | Supraventricular Trigeminy | 35.5 | 3.0 | 14.5 | 2.0 |
| Ventricular Tachycardia | 1 | — | 100.0 | nan | 100.0 | nan |
| Ventricular Trigeminy | 200 | Ventricular Trigeminy | 58.0 | 0.5 | 26.5 | 0.5 |

Full CSV: `./res/fp_allclass_full_suppression/allclass_full_suppression.csv`