# ecg-tp_rex — base ECGFounder single-lead sensitivity (TP-only cohort)

- Val rows: **264**; files present on disk: **98**; missing (not evaluable): **166**.
- Every row is `True Event=True` → metric is **sensitivity** (recall) at the head threshold; ROC/PR are not defined (no per-class negatives).
- `sens_base` = fire rate at base@0.5; `sens_scope` = at the device scope threshold (scope heads only).

## Files present per class (val split)

| event | present/total |
|---|---|
| Atrial Fibrillation | 98/98 |
| Isolated Ventricular Beat | 0/89 |
| Pause | 0/6 |
| Prolonged RR Interval | 0/70 |
| Ventricular Couplet | 0/1 |

## Per-class sensitivity (evaluable classes)

| event | head | n | routing | sens base@0.5 | sens @scope-τ | mean p | median p |
|---|---:|---:|---|---:|---:|---:|---:|
| Atrial Fibrillation | 5 | 98 | scope | 1.000 | 1.000 | 0.944 | 0.961 |

> ⚠ 166 records across 4 classes (Isolated Ventricular Beat, Pause, Prolonged RR Interval, Ventricular Couplet) have no JSON on disk — not evaluated. Stage the files to extend coverage.
