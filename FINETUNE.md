# Fine-Tune Strategy — Paper Alignment + Project Verdict

**Status**: reference memo, captured 2026-05-28 from the ECGFounder paper review.
**Companion**: [PROJECT_STATE.md §3](PROJECT_STATE.md) — the diagnostic record of fine-tune attempts that led to "do not touch the deep model" as the production answer.
**Source paper**: *An Electrocardiogram Foundation Model Built on over 10 Million Recordings with External Evaluation across Multiple Domains*, Li et al., arXiv:2410.04133v4 (Aug 2025).

---

## 1. What the paper's fine-tune strategy actually is

From §2.5 of the paper, plus supplementary §S4.2 and §S5.4:

| Aspect | Paper's choice |
|---|---|
| Original 150-class head | **Discarded** (replaced with a new linear layer sized to the downstream task's class count) |
| Two modes offered | **Linear probing** (backbone frozen, only new head trains) or **Full fine-tuning** (all weights trainable) |
| Optimizer / LR | Adam, `lr = 1e-4`, batch size 256 |
| Schedule | 30 epochs, ReduceLROnPlateau (patience 10, factor 0.1) |
| Checkpoint selection | Save best-validation-AUC weights |
| Linear-probe details (§S4.2) | `lr = 5e-3`, 100 epochs (separate hyperparams) |
| Headline finding (§S5.4) | *"Full fine-tuning consistently achieves better performance compared to linear probing, particularly for the ECGFounder model"* |

### Key quote from §2.5

> "When adapting to specific ECG downstream tasks, we needed to retain the parameters of the base model and **discard the initial classification linear layer**. The number of classes in the downstream task determines the number of neurons needed in the final layer of the new linear layer."

---

## 2. How it prevents backbone degradation — the honest answer

**It doesn't, because the paper doesn't try to.** The paper's fine-tuning is *adapt-to-new-task*, not *preserve-original-task*. Two pathways:

| Mode | Backbone behavior | "Degradation" in the original-task sense |
|---|---|---|
| **Linear probing** | Frozen — gradients don't flow into it | **Impossible by construction.** New head learns; backbone literally unchanged. Original 150-class outputs would still be available — but the paper replaces the head anyway. |
| **Full fine-tuning** | All weights updated against the new task | **Accepted as the cost.** The original 150-class output is discarded, so degradation of those heads is irrelevant — the paper doesn't measure or claim to preserve them. |

The paper offers no technique for the harder problem we attempted in our project — **fine-tuning on a new dataset while keeping the original 150-class output functional on fzark**. That was never the paper's claim or design goal.

---

## 3. What this means for our project

Our prior fine-tune attempts (catastrophic forgetting, masked-loss drift, 6-head linear-probe regression, state-dict surgery — see [PROJECT_STATE.md §3](PROJECT_STATE.md) for the full table) were trying to do something the ECGFounder paper does not support:

> *Preserve the 150-class head while adding new training signal.*

The paper would discard the head; we were trying to keep it. Our work is therefore not a failure to follow the paper — it's a different problem the paper didn't address.

Specifically:

1. **Our "linear probing on 6 heads with frozen backbone + frozen 144 non-target rows"** is paper-aligned in spirit (backbone frozen → no degradation possible), but adds the extra constraint of preserving 144 of 150 heads bit-identically. Our regression gate failed not because the backbone degraded, but because the *6 trained heads* recalibrated away from the t=0.5 production threshold.

2. **Our "vanilla full fine-tune"** (PROJECT_STATE.md §3 attempt #1) was paper-style full FT, but the paper would have evaluated only the new task — it would never have re-tested the original 150 classes. We did, and that's how we discovered the catastrophic forgetting. From the paper's design perspective, that "forgetting" is by intent.

3. **The two-checkpoint inference pattern** ([dual_head_ecgfounder.py](dual_head_ecgfounder.py)) IS the paper-aligned solution if you want both:
   - Original 150-class outputs → use the base model
   - Specialized new-task outputs → use a fine-tuned checkpoint
   - Per-head routing decides which model serves which prediction

---

## 4. Paper-aligned recipe (if a future fine-tune is needed)

If a new fine-tune task arises (e.g., predict a new biomarker, predict a new arrhythmia category not covered by the 150-class vocabulary), the paper-aligned recipe is:

```
1. Discard the 150-class head entirely.
2. Create a NEW linear head for the specific task
     - 1 output for a binary task (e.g., new AFib detector)
     - K outputs for a K-way classification
     - 1 output (no sigmoid) for a regression task (e.g., LVEF)
3. Choose:
     - Linear probing — safer, smaller LR (1e-4), frozen backbone
     - Full FT — better per paper §S5.4, lr=1e-4 batch=256, 30 epochs
4. Optimizer: Adam, lr=1e-4, ReduceLROnPlateau(patience=10, factor=0.1).
5. Evaluate on the new task only — do NOT try to preserve the original 150-class predictions.
6. Ship as a parallel checkpoint and use DualHeadECGFounder to route
   queries to base (original 150) vs. fine-tuned (new task) per-call.
```

This is the only fine-tune path that is both paper-compliant AND preserves our v3.1 fzark production behavior. The trick: don't repurpose the original head; add a new specialist alongside.

---

## 5. Conclusion

**No new fine-tune work is recommended for the v3.1 fzark production pipeline.** PROJECT_STATE.md §3 already concludes *"the cleanest 'preserve both worlds' answer is to NOT touch the deep model"*. The ECGFounder paper, by virtue of explicitly discarding the head on every fine-tune, agrees implicitly — the paper offers no technique for our specific preservation problem because that problem was never in its scope.

Our production stack stays as documented in [PROJECT_STATE.md §0](PROJECT_STATE.md):
- Base model (unchanged)
- v3.1 ontology in `label_config.py`
- v2 FP suppression (`multiclass_fp_suppression.py`)
- `DualHeadECGFounder` for any per-head routing needs

Future fine-tunes — if any — should follow §4 above: new head, new task, new checkpoint, route via `DualHeadECGFounder`.

---

## 6. Citations

| Section in paper | Topic | Page (approx) |
|---|---|---|
| §2.5 | Fine-tuning strategy | 6–7 |
| §S4.2 | Linear probing details | P10 (supplementary) |
| §S5.4 | Full FT vs linear probing comparison | P14 (supplementary) |
| Figure S6 | Visual: full FT > LP on ECGFounder | (supplementary) |

Paper: arXiv:2410.04133v4. Code: https://github.com/PKUDigitalHealth/ECGFounder.

---

*This memo replaces the need to re-read the paper for fine-tune-strategy questions. Pair with [PROJECT_STATE.md §3](PROJECT_STATE.md) for the project's diagnostic record of why we stopped fine-tuning.*
