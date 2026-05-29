# All-Class FP Comparison with Full Per-Class Suppression
## Baseline vs Fine-tuned Model on `ecg_fp_doctor removed1`

**Date**: 2026-05-26
**Cohort**: 2,641 records, stratified up to 200 per event-type, 17 event types
**Device**: MPS (~40 s end-to-end)
**Suppressors active**:
- `Atrial Fibrillation` → `AFibFPSuppressor` (L1 rules + LR ranker, threshold 0.475)
- `Isolated Ventricular Beat` → `PerClassFPSuppressor` (L1 safety net + fzark-trained LR, threshold 0.675)
- `Isolated Supraventricular Beat` → `PerClassFPSuppressor` (L1 safety net + fzark-trained LR, threshold 0.600)
- `Sinus Tachycardia` → `PerClassFPSuppressor` (rule-only, **12 mG default**)
- Other classes → AFib suppressor applied to the AFib head (cross-class FP filter)

All events in this dataset are clinician-removed false positives → **lower alert rate is better**.

---

## 1. Headline table — native-class evaluation
*Each event scored against its own class head, then routed to the matching suppressor.*

| Event Type | n | Suppressor | Baseline raw → post | Fine-tuned raw → post |
|---|---|---|---|---|
| **Atrial Fibrillation** | 200 | AFib | 71.0 % → **3.5 %** | 47.0 % → **2.0 %** |
| **Isolated Supraventricular Beat** | 200 | ISB | 11.5 % → **9.5 %** | 40.0 % → 34.5 % |
| **Isolated Ventricular Beat** | 200 | IVB | 31.0 % → **8.0 %** | 69.5 % → 17.5 % |
| **Sinus Tachycardia** | 200 | Sinus Tachy (12 mG) | 99.5 % → **24.5 %** | 100.0 % → **24.5 %** |
| Pause | 200 | — | 0.0 % | 0.0 % |
| Prolonged RR Interval | 17 | — | 0.0 % | 0.0 % |
| Supraventricular Couplet | 200 | — | 1.5 % | 4.5 % |
| Ventricular Couplet | 200 | — | 0.0 % | 0.5 % |
| Ventricular Run | 200 | — | 0.0 % | 0.0 % |

## 2. Cross-class evaluation — AFib false-fires + AFib suppressor

| Event Type | n | Baseline raw → post | Fine-tuned raw → post |
|---|---|---|---|
| Bradycardia | 200 | 69.5 % → **6.5 %** | 56.5 % → **5.0 %** |
| Custom Heart Rate | 22 | 81.8 % → **0.0 %** | 54.5 % → **0.0 %** |
| Multiple Event | 200 | 58.0 % → **4.0 %** | 50.5 % → **3.5 %** |
| Supraventricular Run | 200 | 48.0 % → **1.0 %** | 42.0 % → **1.0 %** |
| Supraventricular Trigeminy | 200 | 35.5 % → **0.0 %** | 14.5 % → **0.0 %** |
| Ventricular Trigeminy | 200 | 58.0 % → **4.0 %** | 26.5 % → **1.5 %** |
| Supraventricular Bigeminy | 1 | 0 % | 0 % |
| Ventricular Tachycardia | 1 | 100 % → 0 % | 100 % → 0 % |

## 3. Aggregate over mapped classes (weighted by n)

| Model | Raw FP rate | Post-suppression FP rate (only on suppressed classes) |
|---|---|---|
| Baseline | 26.5 % (429/1,617) | **11.4 %** (91 / 800) |
| Fine-tuned | 32.3 % (523/1,617) | **19.6 %** (157 / 800) |

The **baseline** delivers the lower aggregate post-suppression FP rate after the full pipeline — fine-tuning regressed IVB and ISB heads so badly that even strong per-class suppression cannot fully recover those classes.

---

## 4. Key findings

### Finding 1 — Sinus Tachycardia: 100 % → 24.5 % with the new 12 mG default
The 12 mG `motion_max` default brings sinus-tachy false-alert rate down from 100 % to **24.5 %** for both models. Suppressor decisions are model-independent (purely signal-based), so baseline and fine-tuned land at identical post rates. Compared with the previous 50 mG default (42 % post rate), this is a **17.5 percentage-point** further reduction.

### Finding 2 — Cross-class AFib suppression is dramatically effective
For non-AFib events that the model falsely flagged as AFib, the AFib suppressor cuts the false-fire rate by **90–100 %** in every cohort: Custom HR (81.8 → 0), SV Trigeminy (35.5 → 0), V Trigeminy (58 → 4), SV Run (48 → 1), Multiple Event (58 → 4), Bradycardia (69.5 → 6.5). The motion + RR + kurtosis discriminators of the AFib suppressor generalize remarkably well to suppressing falsely-fired-as-AFib events from other arrhythmia classes.

### Finding 3 — Native-class suppression is strong where it has TP-fzark data
| Class | Baseline raw → post | Fine-tuned raw → post |
|---|---|---|
| AFib | 71 → 3.5 (**95 %** ↓) | 47 → 2 (**96 %** ↓) |
| IVB | 31 → 8 (74 % ↓) | 69.5 → 17.5 (75 % ↓) |
| ISB | 11.5 → 9.5 (17 % ↓ — note 1) | 40 → 34.5 (14 % ↓ — note 1) |

> Note 1: ISB suppression is weak in this evaluation because the model flags ISB on only 11.5 % of events to begin with — the few that survive the model's own gate are the most "ISB-like" and harder for the suppressor to reject. On the full 300-event ISB FP cohort (independent of model gate), the suppressor achieves 93.7 % suppression.

### Finding 4 — Fine-tuning still regresses non-AFib heads
Even with full suppressor stack, fine-tuned IVB post-rate is **17.5 %** vs baseline's **8.0 %** — fine-tuning's catastrophic forgetting on the IVB head adds ~10 pp of residual FP that no downstream filter can recover. ISB shows the same pattern (34.5 % vs 9.5 %). For these classes, **baseline + suppressor outperforms fine-tuned + suppressor**.

### Finding 5 — End-to-end FP rate by class is now in single digits for most categories
After the full pipeline:

| Class | Baseline post | Fine-tuned post |
|---|---|---|
| AFib | 3.5 % | 2.0 % |
| IVB | 8.0 % | 17.5 % |
| ISB | 9.5 % | 34.5 % |
| Sinus Tachy | 24.5 % | 24.5 % |
| Bradycardia (AFib fire) | 6.5 % | 5.0 % |
| Multiple Event (AFib fire) | 4.0 % | 3.5 % |
| SV Run (AFib fire) | 1.0 % | 1.0 % |
| SV Trigeminy (AFib fire) | 0 % | 0 % |
| V Trigeminy (AFib fire) | 4.0 % | 1.5 % |
| Custom HR (AFib fire) | 0 % | 0 % |

---

## 5. Recommendation matrix

| Use case | Configuration |
|---|---|
| **Multi-class production deployment** | **Baseline model** + `MultiClassFPSuppressor` (with current per-class defaults) |
| **AFib-only deployment** | Fine-tuned model + AFib suppressor |
| **Reducing cross-class false-AFib alerts** | AFib suppressor (works equally well with either model) |
| **Sinus tachycardia alarm fatigue** | `default` mode (12 mG) — 75.5 pp reduction in FP rate |
| **Mixed deployment** | Use baseline + suppressors as primary; fine-tuned + AFib suppressor as a parallel AFib pathway for hard cases |

---

## 6. Comparison vs the previous run (only AFib suppressor)

| Class | Previous post-FP (AFib suppressor only) | This run (full per-class) | Improvement |
|---|---|---|---|
| Atrial Fibrillation (baseline) | 3.5 % | 3.5 % | unchanged |
| Atrial Fibrillation (fine-tuned) | 2.0 % | 2.0 % | unchanged |
| **Isolated Ventricular Beat (baseline)** | 31.0 % (no suppressor) | **8.0 %** | **−23 pp** |
| **Isolated Ventricular Beat (fine-tuned)** | 69.5 % | **17.5 %** | **−52 pp** |
| **Isolated Supraventricular Beat (baseline)** | 11.5 % | **9.5 %** | −2 pp |
| **Isolated Supraventricular Beat (fine-tuned)** | 40.0 % | **34.5 %** | −5.5 pp |
| **Sinus Tachycardia (both models)** | 42 % (50 mG default) | **24.5 %** (12 mG default) | **−17.5 pp** |

Adding the fzark-trained per-class suppressors and tightening the Sinus-Tachy threshold delivers double-digit improvements on IVB, single-digit improvements on ISB, and 17.5 pp on Sinus Tachycardia.

---

## 7. Reproduction

```bash
# defaults to MPS on Apple Silicon
python3 compare_all_classes_with_full_suppression.py --per-class 200
```

Files:
- `res/fp_allclass_full_suppression/allclass_full_suppression.csv`
- `res/fp_allclass_full_suppression/allclass_full_suppression_report.md` (auto-generated)
- `res/fp_allclass_full_suppression/{baseline,finetuned}_probs_full.npy`

*Generated 2026-05-26 from the full per-class suppression pipeline.*
