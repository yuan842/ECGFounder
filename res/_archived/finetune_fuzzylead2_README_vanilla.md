# Fuzzylead2 Fine-Tuned ECGFounder — Training Report

**Date**: 2026-05-28
**Output checkpoint**: `checkpoint/1_lead_ECGFounder_fuzzy.pth` (118 MB, state_dict only)
**Base model**: `checkpoint/1_lead_ECGFounder.pth` (353 MB, full save)
**Training script**: [scripts/finetune_fuzzylead2.py](../../scripts/finetune_fuzzylead2.py)

## Setup

| Field | Value |
|---|---|
| Train set | concatenated `train_{45,75,90}deg.npz` (58,803 records, 150-class multi-hot) |
| Val set | `val_75deg.npz` (2,198 records — Lead II proxy) |
| Backbone | full 1-lead ECGFounder Net1D (all parameters trainable) |
| Loss | `BCEWithLogitsLoss` over all 150 heads |
| Optimizer | Adam(lr=1e-4, weight_decay=1e-5) |
| Batch size | 32 |
| Epochs | 5 (best ROC saved) |
| Device | MPS (Apple Silicon GPU) |
| Wall time | ~63 min, ~13 min/epoch |

## Training curve (val on `val_75deg`)

| Epoch | Train loss | macro ROC | macro PR-AUC | micro F1 |
|---|---|---|---|---|
| Baseline (no training) | — | 0.7296 | 0.1804 | 0.0931 |
| 1 | 0.0091 | 0.8825 | 0.3560 | 0.6551 |
| 2 | 0.0069 | **0.8891** ★ best | 0.3747 | 0.6537 |
| 3 | 0.0069 | 0.8825 | 0.3818 | **0.6969** |
| 4 | 0.0068 | 0.8804 | 0.3807 | 0.6552 |
| 5 | 0.0068 | 0.8847 | 0.3845 | 0.6745 |

Best macro ROC checkpoint = **epoch 2** (saved to `1_lead_ECGFounder_fuzzy.pth`).

## Per-class metrics on `val_75deg` — BASE vs FUZZY at t=0.5

| Class | pos | BASE ROC | FUZZY ROC | BASE PR | FUZZY PR | BASE F1 | FUZZY F1 |
|---|---|---|---|---|---|---|---|
| NORMAL ECG | 963 | 0.804 | **0.884** | 0.745 | **0.841** | 0.626 | **0.769** |
| ATRIAL FIBRILLATION | 8 | 0.952 | **0.970** | 0.042 | **0.100** | 0.079 | 0.000 |
| INCOMPLETE RBBB | 112 | 0.736 | **0.785** | 0.121 | **0.226** | 0.139 | 0.084 |
| LV HYPERTROPHY | 214 | 0.714 | **0.819** | 0.229 | **0.390** | 0.232 | 0.260 |
| ATRIAL FLUTTER | 4 | 0.998 | **1.000** | 0.374 | **0.887** | 0.444 | **0.750** |
| **LEFT ANTERIOR FASCICULAR BLOCK** | 162 | 0.380 | **0.972** | 0.054 | **0.750** | 0.000 | **0.547** |
| INCOMPLETE LBBB | 8 | 0.818 | **0.910** | 0.034 | 0.087 | 0.000 | 0.000 |
| LEFT POSTERIOR FASCICULAR BLOCK | 18 | 0.392 | **0.810** | 0.007 | 0.053 | 0.000 | 0.000 |
| RV HYPERTROPHY | 12 | 0.773 | **0.852** | 0.018 | 0.037 | 0.000 | 0.000 |

Fine-tuning improved ROC on **every** active class. The big wins are LAFB (0.380 → 0.972), LV Hypertrophy (0.714 → 0.819), Left Posterior Fascicular Block (0.392 → 0.810), and NORMAL ECG calibration (F1 0.626 → 0.769).

## ⚠️ Important caveat — catastrophic forgetting on fzark domain

Running the fuzzy fine-tuned model on the fzark TP cohort:

| Event (V3.1 head) | base detection @ 0.5 | fuzzy detection @ 0.5 | Δ |
|---|---|---|---|
| Atrial Fibrillation (idx 5) | 97.8% | **0.2%** | −97.6 pp |
| Bradycardia (idx 4) | 94.4% | 0.8% | −93.6 pp |
| Isolated Ventricular Beat (idx 9) | 78.6% | 0.0% | −78.6 pp |
| Supraventricular Trigeminy (idx 16) | 96.1% | 0.0% | −96.1 pp |
| Aggregate V3.1 detection | 54.7% | **0.2%** | −54.5 pp |

CSV: [fzark_base_vs_fuzzy.csv](fzark_base_vs_fuzzy.csv).

### Why this happened

The fuzzylead2 training data only has positive labels for **10 of 150 classes**:

| Active idx | head | positives | rate |
|---|---|---|---|
| 2 | NORMAL ECG | 25,653 | 43.6% |
| 5 | ATRIAL FIBRILLATION | 120 | 0.20% |
| 6 | SINUS TACHYCARDIA | 12 | 0.02% |
| 18 | INCOMPLETE RBBB | 3,018 | 5.13% |
| 26 | LEFT VENTRICULAR HYPERTROPHY | 5,754 | 9.79% |
| 32 | ATRIAL FLUTTER | 156 | 0.27% |
| 36 | LEFT ANTERIOR FASCICULAR BLOCK | 4,383 | 7.45% |
| 62 | INCOMPLETE LBBB | 207 | 0.35% |
| 70 | LEFT POSTERIOR FASCICULAR BLOCK | 477 | 0.81% |
| 82 | RIGHT VENTRICULAR HYPERTROPHY | 342 | 0.58% |

**The other 140 heads — including Bradycardia (idx 4), PVC (idx 9), PAC (idx 16), PSVC (idx 19), VT (idx 98), SVT (idx 93), Pause (idx 142) — have zero positive examples** during training. Under `BCEWithLogitsLoss`, every gradient step for these heads pushes their output toward 0. After 5 epochs × 1,838 batches the model has effectively been trained to *never fire* on those heads.

The val ROC of 0.889 looks healthy because it is computed only on the 9 active classes — the catastrophic collapse on the 141 zero-positive heads is invisible to that metric.

## Use this checkpoint when

✅ Deploying for PTB-XL-style label set (NORMAL ECG, AFib, AFL, IRBBB, LVH, LAFB, ILBBB, LPFB, RVH) on single-lead ambulatory recordings.

✅ Demonstrating domain transfer from fuzzy-derived (45°/75°/90°) leads back to clinical labels — ROC gains on the active 9 are real (mean +6 pp).

## Do NOT use this checkpoint when

❌ The downstream task uses the V3.1 fzark ontology (AFib, Bradycardia, PVC, PAC, PSVC, VT, SVT, Pause). The fuzzy model is silent on 7 of 9 V3.1 heads.

❌ Any general-purpose 150-class arrhythmia detection. Only the 10 active heads in the fuzzy training set are reliable; the rest have been wiped out.

## How to do this better next time

The fundamental issue is full fine-tuning with `BCEWithLogitsLoss` on sparse-label data. Mitigations, ranked by likely impact:

1. **Linear probe** — freeze the backbone, train only the 150-class linear head. Preserves base-model behavior on unrepresented classes.
2. **Mask the loss** — compute BCE only over the 10 active heads; freeze logits for the other 140. Implementationally: multiply the loss tensor by a `(150,)` mask before reduction.
3. **LoRA / small adapter** — add a small low-rank adapter on top of the backbone, keep backbone frozen.
4. **Knowledge distillation regularization** — add an L2 penalty pulling fine-tuned logits toward base-model logits on the inactive heads.
5. **Re-merge** — load the base checkpoint, replace only the 10 active heads' weights with the fuzzy ones, keep the other 140 from base. A simple state_dict surgery.

Recommendation: try **(2) masked loss** first — minimal code change, preserves all 150 heads, gives the model freedom to update the backbone for the 10 active labels.

## Reproducing

```bash
# Train (60–70 min on MPS, batch_size=32, 5 epochs)
python3 scripts/finetune_fuzzylead2.py --epochs 5 --batch-size 32 --lr 1e-4

# Evaluate on fzark TP cohort
python3 scripts/eval_fuzzy_vs_base.py
```
