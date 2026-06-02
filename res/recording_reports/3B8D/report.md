# Recording report — 3B8D

- Duration **3467 s** (57.8 min), 347 × 10 s windows
- Model **base** · FP device profile **move_chest_gel** · scope [4, 5, 6, 93, 98, 142]
- ⚠ MOVE is rhythm-NEGATIVE (activity only) — detections here are effectively false positives.

## Per-label burden (after FP suppression)

| label | episodes | burden | % rec | longest | windows raw→FP |
|---|---|---|---|---|---|
| Bradycardia | 0 | 0s | 0.0% | 0s | 0→0 |
| Atrial Fibrillation | 1 | 10s | 0.3% | 10s | 129→1 |
| Sinus Tachycardia | 8 | 1790s | 51.6% | 740s | 178→178 |
| Supraventricular Run | 3 | 70s | 2.0% | 30s | 7→7 |
| Ventricular Run | 18 | 960s | 27.7% | 330s | 84→84 |
| Pause | 0 | 0s | 0.0% | 0s | 0→0 |

## Event intervals (after FP suppression)

| onset | offset | dur | label | peak p |
|---|---|---|---|---|
| 643s | 663s | 20s | Sinus Tachycardia | 0.997 |
| 1453s | 1933s | 480s | Sinus Tachycardia | 0.979 |
| 1453s | 1513s | 60s | Ventricular Run | 0.709 |
| 1533s | 1543s | 10s | Ventricular Run | 0.617 |
| 1603s | 1933s | 330s | Ventricular Run | 0.79 |
| 1613s | 1643s | 30s | Supraventricular Run | 0.552 |
| 1993s | 2733s | 740s | Sinus Tachycardia | 0.987 |
| 2023s | 2073s | 50s | Ventricular Run | 0.682 |
| 2283s | 2293s | 10s | Ventricular Run | 0.539 |
| 2363s | 2383s | 20s | Ventricular Run | 0.55 |
| 2433s | 2443s | 10s | Ventricular Run | 0.533 |
| 2463s | 2543s | 80s | Ventricular Run | 0.684 |
| 2583s | 2613s | 30s | Ventricular Run | 0.713 |
| 2633s | 2643s | 10s | Ventricular Run | 0.747 |
| 2663s | 2733s | 70s | Ventricular Run | 0.681 |
| 2763s | 3163s | 400s | Sinus Tachycardia | 0.991 |
| 2773s | 2803s | 30s | Ventricular Run | 0.585 |
| 2823s | 2873s | 50s | Ventricular Run | 0.6 |
| 2893s | 2903s | 10s | Ventricular Run | 0.69 |
| 2923s | 2933s | 10s | Ventricular Run | 0.556 |
| 2953s | 2973s | 20s | Ventricular Run | 0.572 |
| 2993s | 3143s | 150s | Ventricular Run | 0.81 |
| 3003s | 3033s | 30s | Supraventricular Run | 0.64 |
| 3063s | 3073s | 10s | Supraventricular Run | 0.509 |
| 3183s | 3193s | 10s | Atrial Fibrillation | 0.597 |
| 3183s | 3193s | 10s | Sinus Tachycardia | 0.959 |
| 3223s | 3233s | 10s | Sinus Tachycardia | 0.951 |
| 3283s | 3293s | 10s | Sinus Tachycardia | 0.987 |
| 3373s | 3493s | 120s | Sinus Tachycardia | 0.99 |
| 3373s | 3383s | 10s | Ventricular Run | 0.512 |