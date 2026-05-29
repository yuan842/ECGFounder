# Fine-Tune Strategy v2 — Preserve fzark, Improve Safely

**Date**: 2026-05-28
**Status**: proposal, not yet executed (the work below explicitly retires the v1 vanilla & masked recipes)
**Goal**: improve the model on supplementary cohorts (fuzzylead2, future datasets) **without disrupting the V3.1 fzark deployment** that's already in production.

---

## 1. The constraint that shapes everything

**fzark V3.1 detection must not regress.** The current production pipeline (`1_lead_ECGFounder.pth` + v3.1 ontology + v2 FP suppression) delivers:

| Cohort | Metric | Current value |
|---|---|---|
| ECG-TP fzark | V3.1 aggregate detection @ t=0.5, suppression off | 54.7 % |
| ECG-TP fzark | V3.1 aggregate retention @ t=0.5, suppression on | 54.0 % |
| ECG-FP doctor | Raw FP @ t=0.5, suppression on | 16.32 % |
| Combined binary | PPV @ t=0.5, suppression on | 0.815 |
| Combined binary | ROC-AUC @ t=0.5, suppression on | 0.772 |
| Per-head | AFib detection @ t=0.5 | 97.8 % |
| Per-head | Bradycardia detection @ t=0.5 | 94.4 % |

Any fine-tune that ships must hit at least 99 % of every per-head number above on the cached fzark cohort, and must not increase the FP rate of any V3.1 head. This is the **regression gate**.

## 2. What v1 attempts taught us

Four serious attempts on the fuzzylead2 cohort, summarized:

| Attempt | Backbone | Loss scope | fzark agg @ 0.5 | val_75deg ROC | Verdict |
|---|---|---|---|---|---|
| Base (no FT) | frozen | n/a | 54.7 % | 0.7296 | the bar we must not regress under |
| Vanilla FT | trainable | BCE × all 150 heads | 0.2 % | 0.8891 | catastrophic forgetting; 7/9 V3.1 heads dead |
| Masked-loss FT | trainable | BCE × {10 active} heads only | 62.3 % @ 0.5 / 15.9 % @ 0.7 | 0.8888 | backbone still drifted; AFib head 97.8 % → 6.8 % |
| State-dict B1 (head surgery) | frozen base | only 10 fuzzy head rows pasted in | 54.6 % | 0.810 ← LAFB barely moved | safe but no gain |
| Dual-checkpoint inference | both | both, route per head | 54.7 % bit-identical | 0.884 on fuzzy heads | ✅ zero-risk fzark, full fuzzy gain, 2× inference cost |

The decisive findings:

1. **The 150-head BCE loss is a poison pill on sparse-label data.** 140 heads with zero positives get pushed to 0 by gradient descent. Any recipe that lets gradient flow through inactive heads' linear weights will collapse them.
2. **Even masking the loss isn't enough** — the backbone still drifts to optimize the active task, and that drift propagates to all 150 heads through the shared feature representation. Pause / SV-Run / V-Run mean probabilities moved from ~0.01–0.07 (silent) to ~0.43–0.49 (poised to fire on the wrong inputs).
3. **State-dict surgery on classifier rows alone doesn't transfer gains.** The 10 fuzzy classifier rows are tuned to fuzzy backbone features; pasted onto base backbone features they don't see the right inputs. Most of fuzzy's lift lives in the backbone, not the head.
4. **The cleanest "preserve both worlds" answer is to NOT touch the deep model.** Either run two checkpoints in parallel (Option B / DualHead, already shipped) or build downstream gates on top of model output (the proven path from `PERCLASS_FP_FINETUNED.md`).

## 3. Three principles for the next attempt

Drawn from §2 plus the cross-dataset mapping in [GLOBAL_LABEL_MAP.md](../GLOBAL_LABEL_MAP.md):

1. **Preserve, don't replace.** The base model has 30+ heads with diverse training data we cannot reproduce. Any new training that requires the base to be overwritten in place is the wrong default. Additive structures (LoRA, adapter, parallel checkpoint, downstream LR ranker) are correct.
2. **No gradient should ever flow through a V3.1 fzark head.** The protected indices are {4, 5, 6, 9, 16, 19, 68, 93, 98, 142} (10 unique heads, 13 events). The classifier rows for these must be frozen — their bias/weight values must be bit-identical after training.
3. **Always run the regression gate before adoption.** Every candidate checkpoint must produce fzark per-head detection rates within ±0.5 pp of base at t=0.5 and ±1.0 pp at t=0.7. This needs to be a CI test, not a manual check.

## 4. Strategy — four tiers, ranked by risk

### Tier 0 — already shipped (zero risk to fzark)

| Component | File | Purpose |
|---|---|---|
| **Dual-checkpoint inference** | [dual_head_ecgfounder.py](../../dual_head_ecgfounder.py) | Route 8 PTB-XL-specific labels through fuzzy checkpoint, 142 others through base. fzark heads are bit-identical to base. **Use this for any production deployment that needs PTB-XL labels.** |
| **v2 FP suppression** | [multiclass_fp_suppression.py](../../multiclass_fp_suppression.py) | Motion/HR/SNR feature gates on AFib, Bradycardia, SV-Trigeminy, V-Trigeminy. 44 % aggregate FP reduction at 1.3 % TP cost. |
| **v3.1 head-sharing recoveries** | [label_config.py](../../label_config.py) | SV-Trig, SV-Big, V-Couplet routed through their constituent-beat heads. +1.8 pp fzark detection without retraining. |

### Tier 1 — preserve-fzark-by-construction (low risk, ~hours of compute)

**Tier 1a: Frozen-backbone linear probe on non-fzark heads only.**

Goal: improve specific PTB-XL labels (LAFB, LVH, LPFB, etc.) on single-lead derived-lead data without touching anything fzark needs.

Recipe:
- Load `1_lead_ECGFounder.pth`; freeze the entire backbone.
- Freeze the 10 fzark classifier rows ({4, 5, 6, 9, 16, 19, 68, 93, 98, 142}) — manually `requires_grad=False` on those slices, OR clone them before training and overwrite after every step.
- Train only the remaining 140 classifier rows on fuzzylead2 with masked BCE over the 8 PTB-XL-specific active heads {2, 18, 26, 32, 36, 62, 70, 82} — the same masked-loss recipe but with the additional fzark-row freeze.
- Expected outcome: fzark heads bit-identical to base (because backbone frozen + fzark rows frozen). Active PTB-XL heads should match dual-checkpoint inference within a small δ — possibly slightly worse than dual, because the fuzzy backbone is not used.

Cost: ~10 minutes on MPS (only the 8 × 1024 = 8,192 head parameters are trained; the rest is forward-pass only).
Output: `checkpoint/1_lead_ECGFounder_lp_non_fzark.pth`.
Risk: low. Worst case it doesn't help — fzark is mathematically guaranteed unchanged.

**Tier 1b: Per-class downstream LR ranker on TP+FP feature data.**

Goal: improve precision on specific V3.1 classes (AFib, IVB, ISB, etc.) where we have paired TP+FP cohorts. This is the proven recipe from `PERCLASS_FP_FINETUNED.md` (IVB → 89.9 % retention / 88.7 % FP suppression; ISB → 90.5 % / 93.7 %).

The deep model is never touched. The ranker reads model output + 17 hand-crafted features and applies a per-class threshold. Already documented; extend to additional classes as TP cohorts arrive.

Risk: zero on model side. Cost: seconds per class once features are extracted.

### Tier 2 — LoRA / adapter on the backbone (medium risk, with safeguards)

Tier 2 is what we reach for when Tier 1 isn't enough — when the gain we want clearly requires the *backbone* to see different data, but we still cannot afford to overwrite the base weights.

**Recipe: LoRA on Net1D conv layers, fzark heads doubly protected.**

- Load `1_lead_ECGFounder.pth`; freeze every weight in the model.
- Wrap each conv layer (or just the deeper stages — see ablation note below) with a LoRA adapter at rank 8–16. Only the adapter weights train; base weights stay byte-identical.
- Apply the same loss-mask as Tier 1 (BCE on 8 PTB-XL-specific heads only), AND freeze the 10 fzark classifier rows.
- Add a **distillation regularizer**: `L_total = L_BCE_active + λ · MSE(student_logits[V3.1_indices], base_logits[V3.1_indices].detach())`. λ ≈ 0.5–1.0. This actively pushes V3.1 heads toward base, even though the LoRA architecture already structurally limits their drift.
- Add a **frozen-head sanity check** inside the training loop: every N steps, compute `(student_logits[V3.1] - base_logits[V3.1]).abs().max()`; abort if it exceeds, e.g., 0.1.

Two ablation variants to compare:
1. LoRA on **all 7 Net1D stages** — fullest expressiveness.
2. LoRA on **only stages 5–6** (the deeper representations) — leave low-level QRS-feature extractors fully shared with base. Likely safer because early-stage features are shared across all 150 heads.

Cost: ~30–60 min on MPS for 5 epochs (LoRA only adds maybe 1–2 % of total params).
Output: `checkpoint/1_lead_ECGFounder_lora_<config>.pth` plus LoRA-adapter `.safetensors`.
Risk: medium. The architecture prevents the base weights from being overwritten, but the LoRA delta can still shift the V3.1 head outputs at inference time. The distillation regularizer + frozen-head sanity check + post-hoc regression gate are the three guardrails.

### Tier 3 — re-train, never deployed without explicit approval (high risk)

For completeness, the recipes we will *not* use as defaults but might revisit if the field changes:

- Full fine-tune (vanilla): never, given the catastrophic-forgetting demonstration.
- Masked-loss full fine-tune: never, given the backbone-drift demonstration (Tier 2 with LoRA dominates this).
- Single-class fine-tune on AFib (like the previous "AFIB Fuzzy"): the prior result showed −1.2–3.2 pp regression even on the target metric. Tier 1b LR ranker dominates this.

## 5. CI regression-gate spec

Every candidate checkpoint goes through a `pytest` gate before promotion. The test compares the candidate's fzark predictions to the cached base predictions:

```python
# tests/test_fzark_regression_gate.py
def test_fzark_per_head_detection_within_tolerance(candidate_ckpt):
    base = np.load("res/tp_fzark_full_suppression/baseline_probs_tp.npy")
    cand = run_inference(candidate_ckpt, fzark_cohort)
    for et, idx in FZARK_LABEL_MAP.items():
        mask = (df['Event Type'] == et).values
        if mask.sum() == 0: continue
        base_det = (base[mask, idx] >= 0.5).mean()
        cand_det = (cand[mask, idx] >= 0.5).mean()
        # Allow ±0.5 pp at t=0.5, ±1 pp at t=0.7
        assert abs(cand_det - base_det) <= 0.005, (et, base_det, cand_det)

def test_fzark_inactive_heads_bit_identical(candidate_ckpt):
    # For Tier 1 (frozen backbone + frozen fzark rows): outputs MUST be byte-identical
    base = np.load(...)
    cand = run_inference(...)
    fzark_idx = list(FZARK_LABEL_MAP.values())
    max_diff = np.abs(cand[:, fzark_idx] - base[:, fzark_idx]).max()
    assert max_diff < 1e-6, max_diff
```

The first test bounds drift for Tier 2 (LoRA — small drift expected, but capped). The second test is the absolute test for Tier 1 (frozen + linear probe — must be bit-identical).

## 6. Recommended execution order

1. **Now**: confirm `DualHeadECGFounder` is the production wrapper for any task that needs PTB-XL labels. fzark stays on base. (Already done.)
2. **Sprint +0**: implement Tier 1a (frozen-backbone linear probe on non-fzark heads). Compare against dual-checkpoint inference on val_75deg; if it matches within 1 pp ROC, prefer Tier 1a (single checkpoint, half the inference cost).
3. **Sprint +1**: implement Tier 1b extensions — write LR rankers for AFib, Sin Tach FPs (the remaining V3.1 classes with both TP and FP cohorts). Target the same 90/90 bar as IVB / ISB. Stack with v2 suppression.
4. **Sprint +2 (only if needed)**: Tier 2 LoRA, but only after a concrete metric gap is identified that Tier 1 cannot close. Run the two ablations (all-stage vs stages-5–6) and report. Promote only if the regression gate passes.
5. **Tier 3 is off-limits** without an explicit approval — document the proposal, run small-cohort ablation, do not ship.

## 7. What this strategy explicitly does NOT do

- **Does not overwrite `1_lead_ECGFounder.pth`** under any tier. Base is forever read-only.
- **Does not retrain on data that lacks positives for fzark's V3.1 classes** without the loss mask + classifier-row freeze in place. The fuzzylead2 lesson is permanent.
- **Does not promote a checkpoint that fails the regression gate** even if it improves a supplementary metric. The asymmetric cost (fzark regression hurts production users; supplementary metric gains help future use cases) makes the gate non-negotiable.

## 8. Open questions for the operator

1. Is the 2× inference cost of `DualHeadECGFounder` acceptable in the target deployment environment? If yes, Tier 1a may not even be needed for the PTB-XL-label use case.
2. Is there demand for any V3.1 class beyond IVB/ISB to be hardened via Tier 1b LR ranker? AFib already has the v2 motion-gate at 100 % FP reduction, but Sin Tach (99 % raw FP rate) is the next obvious target.
3. Will additional TP cohorts arrive for Sinus Tachycardia, Pause, ST Elevation, V-Run, SV-Run, V-Trigeminy? Those are the dead-or-thin V3.1 heads in §3.4 of the cross-dataset performance report. None can be meaningfully fine-tuned without them.
4. Is there a fzark V-Trigeminy expansion in the pipeline? n=16 currently is too small to characterize the model's 87.5 % PAC-firing — a larger cohort would let us decide whether to route V-Trigeminy through the PAC head (like SV-Trigeminy) or keep it passthrough.

---

*Drafted on 2026-05-28 after v1 vanilla / masked / surgery / dual-checkpoint experiments. Refresh this doc when Tier 1a is run or when new cohorts arrive.*
