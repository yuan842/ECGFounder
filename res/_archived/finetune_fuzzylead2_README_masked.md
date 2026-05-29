# Fuzzylead2 Fine-Tune — **Masked-Loss** Variant

**Date**: 2026-05-28
**Checkpoint**: `checkpoint/1_lead_ECGFounder_fuzzy_masked.pth` (118 MB)
**Training script**: [scripts/finetune_fuzzylead2.py](../../scripts/finetune_fuzzylead2.py) `--masked-loss`
**Reason for variant**: the vanilla full fine-tune (`1_lead_ECGFounder_fuzzy.pth`) catastrophically forgot 7 of 9 V3.1 fzark heads — see [README.md](README.md). The masked-loss recipe zeroes the per-element BCE on the 140 inactive heads so no gradient flows through them.

## Recipe difference

| Setting | Vanilla fuzzy | Masked-loss fuzzy |
|---|---|---|
| Loss | `BCEWithLogitsLoss(reduction='mean')` | `BCEWithLogitsLoss(reduction='none')`, multiplied by a `(150,)` mask = 1 at active heads (10) and 0 elsewhere, then summed / (batch × 10) |
| Backbone | Trainable | **Still trainable** (no freezing) |
| Active heads | 150 (all) | 10 (idx 2, 5, 6, 18, 26, 32, 36, 62, 70, 82) |
| Epochs | 5 | 2 (training killed mid-epoch 3, best-saved checkpoint is epoch 2) |

## Training curve on `val_75deg`

| Epoch | Train loss | macro ROC (9 active) | macro PR | micro F1 |
|---|---|---|---|---|
| Baseline | — | 0.7296 | 0.1804 | 0.0931 |
| 1 (masked) | 0.1068 | 0.8841 | 0.3773 | 0.0904 |
| 2 (masked) — best | 0.0924 | **0.8888** | 0.3737 | 0.0441 |

Train loss is naturally ~10× higher than vanilla because we're not averaging over the (zero-loss) 140 inactive heads. Best macro ROC (0.8888) essentially matches the vanilla fine-tune's epoch-2 best (0.8891) — the 10 active heads learned the same fuzzy-lead representations.

## Per-class ROC on `val_75deg` — BASE vs vanilla FUZZY vs MASKED

| Class | pos | BASE | FUZZY | MASKED |
|---|---|---|---|---|
| NORMAL ECG | 963 | 0.804 | 0.884 | **0.885** |
| ATRIAL FIBRILLATION | 8 | 0.952 | 0.970 | **0.976** |
| INCOMPLETE RBBB | 112 | 0.736 | 0.785 | **0.788** |
| LEFT VENTRICULAR HYPERTROPHY | 214 | 0.714 | 0.819 | **0.822** |
| ATRIAL FLUTTER | 4 | 0.998 | **1.000** | 0.999 |
| LEFT ANTERIOR FASCICULAR BLOCK | 162 | 0.380 | **0.972** | 0.969 |
| INCOMPLETE LBBB | 8 | 0.818 | 0.910 | **0.925** |
| LEFT POSTERIOR FASCICULAR BLOCK | 18 | 0.392 | **0.810** | 0.803 |
| RV HYPERTROPHY | 12 | 0.773 | **0.852** | 0.832 |

Masked-loss matches vanilla on every active class (±0.01 ROC). PR-AUC also matches (NORMAL 0.845 vs 0.841, AFib 0.205 vs 0.100 — masked is **better** for the very-rare AFib head).

## fzark TP cohort — base preservation check

This was the whole point of the masked variant.

| V3.1 head | BASE det@0.5 | FUZZY det@0.5 | **MASKED det@0.5** | Δ vs base |
|---|---|---|---|---|
| Atrial Fibrillation (idx 5, *active*) | 97.8% | 0.2% | **6.8%** | −91.0 pp ⚠️ |
| Bradycardia (idx 4, inactive) | 94.4% | 0.8% | **94.2%** | −0.2 pp ✅ |
| Isolated Supraventricular Beat (idx 16, inactive) | 53.6% | 0.0% | **90.8%** | +37.2 pp |
| Isolated Ventricular Beat (idx 9, inactive) | 78.6% | 0.0% | **97.4%** | +18.8 pp |
| Pause (idx 142, inactive) | 0.0% | 0.0% | **0.0%** | 0 pp (mean 0.010 → 0.436) |
| Supraventricular Couplet (idx 19, inactive) | 4.4% | 0.0% | **81.8%** | +77.4 pp |
| Supraventricular Run (idx 93, inactive) | 0.0% | 0.0% | **0.0%** | 0 pp (mean 0.062 → 0.454) |
| Supraventricular Trigeminy (idx 16, inactive) | 96.1% | 0.0% | **100.0%** | +3.9 pp |
| Supraventricular Bigeminy (idx 16, inactive) | 78.1% | 0.0% | **100.0%** | +21.9 pp |
| Ventricular Couplet (idx 9, inactive) | 80.6% | 0.0% | **100.0%** | +19.4 pp |
| Ventricular Run (idx 98, inactive) | 0.0% | 0.0% | **2.8%** | +2.8 pp (mean 0.066 → 0.486) |
| ST Elevation (idx 68, inactive) | 0.0% | 0.0% | **0.0%** | 0 pp (mean 0.025 → 0.472) |

Aggregate (V3.1 weighted by n):

| Threshold | BASE | FUZZY | MASKED | Δ vs base |
|---|---|---|---|---|
| t=0.5 | 54.7% | 0.2% | **62.3%** | +7.6 pp |
| t=0.6 | 51.1% | 0.0% | 35.7% | −15.5 pp |
| t=0.7 | 47.5% | 0.0% | 15.9% | −31.7 pp |

CSV: [fzark_base_vs_fuzzy_masked.csv](fzark_base_vs_fuzzy_masked.csv).

## What worked and what didn't

### ✅ Worked
- **No more catastrophic collapse.** Vanilla fuzzy: aggregate V3.1 = 0.2% at t=0.5. Masked: 62.3%. The masked recipe eliminated the "every head pushed to 0" failure.
- **The active heads learned identically** — masked matches vanilla on every val_75deg class to ±0.01 ROC.
- **Bradycardia head perfectly preserved** — 94.4% → 94.2% at t=0.5.
- **Some inactive heads got *better* than base** at t=0.5: ISB (+37 pp), V Couplet (+19 pp), SV Couplet (+77 pp). The backbone's fuzzy-lead training apparently sharpened the underlying QRS-morphology representations.

### ⚠️ Didn't fully work
- **AFib head broke (97.8% → 6.8%).** AFib is *active* in training (120 positives in 58,803 = 0.2% rate). The model learned that AFib is rare in the fuzzy domain and shifted its decision boundary up. On fzark (where AFib TPs are common and high-signal) the head no longer fires above 0.5. Mean prob 0.913 → 0.161. This is the price of "active but rare" — masked-loss couldn't protect it because its loss *is* computed.
- **Calibration drift on silent heads.** Pause, ST Elevation, SV Run, V Run heads sat at near-zero mean prob (0.010–0.066) under base. Under masked they sit at 0.43–0.49 — just below 0.5. They didn't *fire* (so V3.1 detection is unchanged at t=0.5) but they're now poised to fire at slightly lower thresholds. The shared backbone drifted — the masked loss only froze gradients into the heads' linear weights, not the upstream features.
- **At t=0.7 the masked model is worse than base** (15.9% vs 47.5% aggregate). High-confidence detection collapsed because the model became less confident overall.

## Root cause of the partial success

The masking only blocks gradient through the **150-class linear layer** (the very last layer). The backbone (Net1D) is still trained against the active-class loss, which shifts the underlying features that all 150 heads read from. The inactive heads survived (no direct gradient) but are seeing a different feature distribution than they were originally trained against.

To **fully** preserve the inactive heads we'd need one of:
1. **Freeze the backbone**, train only the 10 active heads' linear weights (true linear probe restricted to the active subset)
2. **State-dict surgery**: take vanilla fuzzy weights but replace the 10 active heads' linear weights with base, OR vice versa — take base everywhere but copy the 10 active heads from fuzzy
3. **Knowledge-distillation regularization** pulling all 150 logits toward base-model logits with a stop-gradient on the base side
4. **LoRA / adapter** that adds low-rank updates instead of modifying the backbone directly

## Use this checkpoint when

✅ The same use cases as vanilla fuzzy (PTB-XL active label set on single-lead) **and** you need any of: Bradycardia, IVB/PVC, ISB/PAC, SV Couplet/Bigeminy/Trigeminy, V Couplet to still work. Those are now preserved or even improved over base.

⚠️ If your downstream task uses **AFib**, you should stick with the base model — the masked variant's AFib head is broken too (different failure mode from vanilla: under-confident rather than completely silent).

❌ At thresholds ≥ 0.6 the masked model is meaningfully less confident than base across most heads. For high-confidence operating points, stick with the base model.

## Reproducing

```bash
python3 scripts/finetune_fuzzylead2.py --epochs 5 --batch-size 32 --lr 1e-4 --masked-loss
python3 scripts/eval_fuzzy_vs_base.py --ckpt checkpoint/1_lead_ECGFounder_fuzzy_masked.pth --label fuzzy_masked
```
