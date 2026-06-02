# Recording report — 4JF9

- Duration **3767 s** (62.8 min), 377 × 10 s windows
- Model **base** · FP device profile **move_chest_gel** · scope [4, 5, 6, 93, 98, 142]
- ⚠ MOVE is rhythm-NEGATIVE (activity only) — detections here are effectively false positives.

## Per-label burden (after FP suppression)

| label | episodes | burden | % rec | longest | windows raw→FP |
|---|---|---|---|---|---|
| Bradycardia | 0 | 0s | 0.0% | 0s | 6→0 |
| Atrial Fibrillation | 12 | 220s | 5.8% | 40s | 90→20 |
| Sinus Tachycardia | 11 | 1520s | 40.4% | 490s | 147→147 |
| Supraventricular Run | 3 | 890s | 23.6% | 490s | 88→88 |
| Ventricular Run | 5 | 1040s | 27.6% | 490s | 102→102 |
| Pause | 0 | 0s | 0.0% | 0s | 0→0 |

## Event intervals (after FP suppression)

| onset | offset | dur | label | peak p |
|---|---|---|---|---|
| 252s | 262s | 10s | Atrial Fibrillation | 0.714 |
| 272s | 312s | 40s | Sinus Tachycardia | 0.998 |
| 282s | 312s | 30s | Atrial Fibrillation | 0.562 |
| 462s | 472s | 10s | Atrial Fibrillation | 0.635 |
| 622s | 662s | 40s | Atrial Fibrillation | 0.609 |
| 682s | 712s | 30s | Atrial Fibrillation | 0.92 |
| 872s | 902s | 30s | Atrial Fibrillation | 0.958 |
| 1112s | 1122s | 10s | Atrial Fibrillation | 0.82 |
| 1132s | 1492s | 360s | Sinus Tachycardia | 0.995 |
| 1152s | 1492s | 340s | Ventricular Run | 0.983 |
| 1162s | 1492s | 330s | Supraventricular Run | 0.972 |
| 1512s | 1522s | 10s | Atrial Fibrillation | 0.73 |
| 1522s | 2012s | 490s | Sinus Tachycardia | 0.991 |
| 1522s | 2012s | 490s | Supraventricular Run | 0.971 |
| 1522s | 2012s | 490s | Ventricular Run | 0.976 |
| 3022s | 3042s | 20s | Atrial Fibrillation | 0.964 |
| 3022s | 3362s | 340s | Sinus Tachycardia | 0.993 |
| 3032s | 3112s | 80s | Ventricular Run | 0.945 |
| 3042s | 3112s | 70s | Supraventricular Run | 0.89 |
| 3142s | 3262s | 120s | Ventricular Run | 0.843 |
| 3332s | 3342s | 10s | Ventricular Run | 0.632 |
| 3402s | 3432s | 30s | Sinus Tachycardia | 0.623 |
| 3472s | 3612s | 140s | Sinus Tachycardia | 0.738 |
| 3642s | 3662s | 20s | Sinus Tachycardia | 0.782 |
| 3662s | 3672s | 10s | Atrial Fibrillation | 0.994 |
| 3692s | 3702s | 10s | Atrial Fibrillation | 0.727 |
| 3702s | 3712s | 10s | Sinus Tachycardia | 0.793 |
| 3782s | 3792s | 10s | Atrial Fibrillation | 0.754 |
| 3802s | 3872s | 70s | Sinus Tachycardia | 0.84 |
| 3912s | 3922s | 10s | Sinus Tachycardia | 0.576 |
| 3952s | 3962s | 10s | Sinus Tachycardia | 0.812 |