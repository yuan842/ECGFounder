# ECG-FP-Doctor-Removed1 — All-Class Performance Comparison

**Date**: 2026-05-26
**Cohort**: 2,641 records, stratified sample up to 200 / event-type, across **17 event types** in the dataset
**Device**: MPS (Apple M-series; ~50 s total runtime for both models — down from 13 min on CPU)
**Models**:
- **Baseline**: `1_lead_ECGFounder.pth`, threshold = 0.5
- **Fine-tuned**: `finetuned_afib_model.pth`, threshold = 0.2646
**Suppressor**: `afib_fp_suppressor_lr.npz`, L2 threshold = 0.475

Every record is a clinician-removed false positive → **lower alert rate = better.**

---

## 1. Headline table

### Direct-class evaluation (event-type → its own 150-class head)
*Model is asked: "Is this `<event_type>`?". All examples are FPs, so a low rate is good.*

| Event Type | n | class | Baseline raw% | Fine-tuned raw% | Δ raw |
|---|---|---|---|---|---|
| **Atrial Fibrillation** | 200 | 5 | 71.0% **→ 3.5%** *(post-supp)* | 47.0% **→ 2.0%** *(post-supp)* | −24.0 pp ✓ |
| Isolated Supraventricular Beat | 200 | 16 | 11.5% | 40.0% | **+28.5 pp ✗** |
| Isolated Ventricular Beat | 200 | 9 | 31.0% | 69.5% | **+38.5 pp ✗** |
| Pause | 200 | 143 | 0.0% | 0.0% | 0 |
| Prolonged RR Interval | 17 | 81 | 0.0% | 0.0% | 0 |
| Sinus Tachycardia | 200 | 6 | 99.5% | 100.0% | +0.5 pp |
| Supraventricular Couplet | 200 | 20 | 1.5% | 4.5% | +3.0 pp |
| Ventricular Couplet | 200 | 91 | 0.0% | 0.5% | +0.5 pp |
| Ventricular Run | 200 | 99 | 0.0% | 0.0% | 0 |

### Cross-class AFib evaluation (does the AFib head falsely fire on a non-AFib event?)
*Suppressor applied; "post" reflects model+suppressor pipeline.*

| Event Type | n | Baseline AFib raw% | Baseline AFib post% | Fine-tuned AFib raw% | Fine-tuned AFib post% | Suppressor gain |
|---|---|---|---|---|---|---|
| Bradycardia | 200 | 69.5% | **6.5%** | 56.5% | **5.0%** | ~91% |
| Custom Heart Rate | 22 | 81.8% | **0.0%** | 54.5% | **0.0%** | 100% |
| Multiple Event | 200 | 58.0% | **4.0%** | 50.5% | **3.5%** | ~93% |
| Supraventricular Bigeminy | 1 | 0.0% | 0.0% | 0.0% | 0.0% | — |
| Supraventricular Run | 200 | 48.0% | **1.0%** | 42.0% | **1.0%** | ~98% |
| Supraventricular Trigeminy | 200 | 35.5% | **0.0%** | 14.5% | **0.0%** | 100% |
| Ventricular Tachycardia | 1 | 100.0% | 0.0% | 100.0% | 0.0% | 100% |
| Ventricular Trigeminy | 200 | 58.0% | **4.0%** | 26.5% | **1.5%** | ~94% |

---

## 2. Three key findings

### Finding 1 — Fine-tuning HURTS specificity on non-AFib classes (catastrophic forgetting)
The fine-tune script (`finetune_afib.py`) unfreezes the last residual block and the full dense layer, but its loss is computed only on the AFib logit. Backprop through the shared feature extractor disrupts the representations relied on by other class heads.

Observed regression:
- **Isolated Ventricular Beat**: 31% → **69.5%** false-alert rate (+38.5 pp worse)
- **Isolated Supraventricular Beat**: 11.5% → **40%** (+28.5 pp worse)
- **Ventricular Couplet / Supraventricular Couplet**: small but consistent degradation

This is exactly what catastrophic forgetting looks like — single-task fine-tuning erodes the multi-class capability of the foundation model.

### Finding 2 — Fine-tuning improves AFib-head specificity in *every* arrhythmia population
Looking at the AFib head only (cross-class evaluation), fine-tuning consistently lowers the false-AFib rate on non-AFib events:

| Cohort | Baseline → Fine-tuned (AFib raw%) |
|---|---|
| Atrial Fibrillation (target) | 71.0% → 47.0% (-24 pp) |
| Bradycardia | 69.5% → 56.5% (-13 pp) |
| Custom Heart Rate | 81.8% → 54.5% (-27 pp) |
| Multiple Event | 58.0% → 50.5% (-7 pp) |
| Supraventricular Run | 48.0% → 42.0% (-6 pp) |
| Supraventricular Trigeminy | 35.5% → 14.5% (-21 pp) |
| Ventricular Trigeminy | 58.0% → 26.5% (-31 pp) |

The fine-tuned AFib head is **uniformly more conservative** than the baseline AFib head — exactly the intended training outcome.

### Finding 3 — The AFib FP suppressor is highly effective as a **cross-class** filter
Even though the suppressor was trained only on AFib-vs-AFib-FP discrimination, it generalizes remarkably well to suppressing falsely-fired-as-AFib events from other arrhythmia classes:

| Event Type | Baseline AFib raw% | Baseline AFib post-supp% | Suppressor gain |
|---|---|---|---|
| Bradycardia | 69.5% | 6.5% | 90.6% |
| Custom Heart Rate | 81.8% | 0.0% | 100.0% |
| Multiple Event | 58.0% | 4.0% | 93.1% |
| Supraventricular Run | 48.0% | 1.0% | 97.9% |
| Supraventricular Trigeminy | 35.5% | 0.0% | 100.0% |
| Ventricular Trigeminy | 58.0% | 4.0% | 93.1% |

The suppressor's motion + RR + kurtosis rules naturally reject bradycardic and motion-corrupted events (regardless of the event's "true" label), and its LR ranker carries the rest.

---

## 3. Aggregate performance

**Direct-class evaluation, weighted by sample size over the 9 mapped event types:**

| Model | Raw alerts | Total n | Weighted raw FP rate |
|---|---|---|---|
| Baseline | 429 | 1,617 | **26.5%** |
| Fine-tuned | 523 | 1,617 | **32.3%** |

**Fine-tuning *increased* the aggregate FP rate by ~6 pp on native-class heads** — driven almost entirely by IVB and ISB regressions. Sinus Tachycardia at ~100% in both models also pulls the aggregate up regardless of fine-tuning.

**AFib-head pipeline (model + suppressor) — every event type combined (n=2,641):**

| | Baseline | Fine-tuned |
|---|---|---|
| Raw AFib alerts | 1,496 | 1,160 |
| Post-suppression AFib alerts | ~85 (estimated from per-class) | ~50 (estimated) |
| Effective AFib FP rate | ~3.2% | ~1.9% |

The **fine-tuned + suppressor stack remains the strongest configuration for AFib-specific FP suppression** across the entire FP dataset.

---

## 4. Per-class deployment recommendation

| If you are deploying for… | Use |
|---|---|
| **AFib detection only** | Fine-tuned model + FP suppressor (best AFib-head specificity in every cohort) |
| **Multi-class arrhythmia detection** | **Baseline model** for non-AFib classes + **fine-tuned + suppressor** specifically for the AFib head (composite scoring) |
| **Single-model deployment, multi-class** | Baseline — fine-tuned regresses on IVB / ISB / VC / SVC |
| Reducing alarm fatigue from any arrhythmia mistakenly flagged as AFib | Add the FP suppressor — it cuts cross-class false AFib alerts by 90–100% in every cohort |

---

## 5. Sinus Tachycardia — the irreducible class

Both models alert ~100% of Sinus Tachycardia events as Sinus Tachycardia (baseline 99.5%, fine-tuned 100%). These are all clinician-removed, but the ECG **truly does** show sinus tachycardia — the clinician's removal reflects a decision that the event was not actionable, not that the model was wrong about the rhythm. No purely motion/ECG filter can fix this; it requires context (HR trend, patient baseline, activity log) the model doesn't have access to.

---

## 6. Output files

- `res/fp_allclass_comparison/allclass_comparison.csv` — per-class table
- `res/fp_allclass_comparison/allclass_comparison_report.md` — auto-generated short report
- `res/fp_allclass_comparison/baseline_probs_full.npy` — full (N, 150) prediction matrix
- `res/fp_allclass_comparison/finetuned_probs_full.npy` — same for fine-tuned

---

## 7. Reproduction

```bash
# Defaults to MPS on Apple Silicon
python3 compare_all_classes_on_fp_dataset.py --per-class 200

# Force CPU or CUDA
python3 compare_all_classes_on_fp_dataset.py --per-class 200 --device cpu
```

Total runtime on MPS: **~50 seconds** for both models × 2,641 records.
