# All-Class FP Comparison with Full Per-Class Suppression

**Cohort**: 6541 records, up to 500/event-type
**Device**: mps
**Suppression**: ON
**Suppressors active**: ['Atrial Fibrillation', 'Bradycardia', 'Supraventricular Trigeminy', 'Ventricular Trigeminy']

## Per-class results

| Event Type | n | route | Baseline raw% | Baseline post% |
|---|---|---|---|---|
| Atrial Fibrillation | 500 | Atrial Fibrillation | 72.0 | 0.0 |
| Bradycardia | 500 | Bradycardia | 3.2 | 1.8 |
| Isolated Supraventricular Beat | 500 | — | 30.6 | nan |
| Isolated Ventricular Beat | 500 | — | 25.4 | nan |
| Pause | 500 | — | 0.0 | nan |
| Sinus Tachycardia | 500 | — | 99.0 | nan |
| Supraventricular Bigeminy | 1 | — | 100.0 | nan |
| Supraventricular Couplet | 500 | — | 2.2 | nan |
| Supraventricular Run | 500 | — | 0.0 | nan |
| Supraventricular Trigeminy | 500 | Supraventricular Trigeminy | 67.4 | 0.8 |
| Ventricular Couplet | 500 | — | 19.6 | nan |
| Ventricular Run | 500 | — | 0.0 | nan |
| Custom Heart Rate | 22 | — | 81.8 | nan |
| Multiple Event | 500 | — | 62.4 | nan |
| Prolonged RR Interval | 17 | — | 17.6 | nan |
| Ventricular Tachycardia | 1 | — | 100.0 | nan |
| Ventricular Trigeminy | 500 | Ventricular Trigeminy | 37.8 | 0.6 |

Full CSV: `./res/fp_allclass_full_suppression/allclass_full_suppression.csv`