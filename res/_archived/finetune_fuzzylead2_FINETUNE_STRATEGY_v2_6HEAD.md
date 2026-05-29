# Fine-Tune Strategy v2 — 6-Head Core (Scoped)

**Date**: 2026-05-28
**Supersedes**: [FINETUNE_STRATEGY.md](FINETUNE_STRATEGY.md) (the all-fuzzy-active variant) for this specific scope
**Status**: proposal, not executed
**Scope**: fine-tune **only** the 6 Founder heads validated in BOTH PTB-XL and Fzark.

---

## 1. The 6-head target set

| Backbone idx | ECGFounder head | PTB-XL label | Fzark event(s) | Mapping note |
|---|---|---|---|---|
| 4 | SINUS BRADYCARDIA | SINUS BRADYCARDIA | Bradycardia | Exact match |
| 5 | ATRIAL FIBRILLATION | ATRIAL FIBRILLATION | Atrial Fibrillation | Exact match |
| 6 | SINUS TACHYCARDIA | SINUS TACHYCARDIA | Sinus Tachycardia | Exact match |
| 9 | PREMATURE VENTRICULAR COMPLEXES | PVC | Isolated Ventricular Beat **+** Ventricular Couplet (v3.1) | Shared PVC head — fires on constituent beats |
| 93 | SUPRAVENTRICULAR TACHYCARDIA | SVT | Supraventricular Run | Good match (brief SVT) |
| 98 | VENTRICULAR TACHYCARDIA | VT | Ventricular Run | Good match (brief NSVT) |

These 6 are simultaneously the **production-critical V3.1 fzark heads** (idx 4/5/6/9/93/98 all appear in `FZARK_LABEL_MAP`) and the **PTB-XL active classes** with the most ground-truth labels (1,514 + 826 + 637 + 1,143 + 42 + 42 = 4,204 PTB-XL positives in n=21,799).

The fine-tune target is to **improve these 6 heads on the production (ambulatory) distribution** using fzark's paired TP+FP cohort — without touching anything else in the model.

## 2. Why this scope changes the previous calculus

[FINETUNE_STRATEGY.md](FINETUNE_STRATEGY.md) v1 said "never touch fzark heads" because the data source (fuzzylead2) had no labels for most V3.1 events. **That constraint vanishes with the new scope** because:

1. **We now have paired TP and FP examples from the production distribution.** Fzark's clinician-confirmed TPs (29,278 total across the 6 heads) and `ecg_fp_doctor removed1`'s clinician-rejected FPs are the perfect supervised pairs. The model can learn to *separate* them rather than just "fire when the head looks like X."
2. **PTB-XL positives add a complementary clinical-recording distribution** (4,204 positives across the same 6 heads). Combining gives ambulatory + clinical generalization in one training run.
3. **The remaining 144 heads have zero labels in this restricted training set by construction** — so the freezing trick from v1 (mask the loss + freeze inactive classifier rows) is even cleaner here: we freeze 144 / 150 heads, the loss naturally never touches them, gradient cannot affect them.

This is the canonical, well-posed fine-tune problem: **target heads have paired labels in production distribution, all other heads are explicitly out of scope.**

## 3. Available training data (already on disk)

### Positives per head (would constitute target class = 1)

| idx | Head | PTB-XL pos | Fzark TPs | Combined |
|---|---|---|---|---|
| 4 | SINUS BRADYCARDIA | 637 | 2,155 | **2,792** |
| 5 | ATRIAL FIBRILLATION | 1,514 | 19,566 | **21,080** |
| 6 | SINUS TACHYCARDIA | 826 | 0 (no TPs in fzark) | **826** ⚠️ no production-distribution positives |
| 9 | PVC | 1,143 | 2,319 (IVB 2,283 + V Couplet 36) | **3,462** |
| 93 | SVT | 42 | 26 (SV Run) | **68** ⚠️ very thin |
| 98 | VT | 42 | 5,212 (V Run) | **5,254** |

### Negatives per head (would constitute target class = 0)

For each head, available negatives come from:
- **Fzark FPs** of the same event type — clinician-confirmed: model fired but it was wrong. These are gold for teaching the model "don't fire here."
- **All other Fzark TPs** (any record whose true label is a different V3.1 event) — naturally a hard negative for this head.
- **All other PTB-XL records** without this label.

| idx | Head | Fzark FPs (this event) | Production-distribution negatives (rough) |
|---|---|---|---|
| 4 | SINUS BRADYCARDIA | 5,941 | strong |
| 5 | ATRIAL FIBRILLATION | 1,793 | strong |
| 6 | SINUS TACHYCARDIA | 530 | weak (also no TPs — sparse signal both ways) |
| 9 | PVC | 12,877 (IVB 10,875 + V Couplet 2,002) | very strong |
| 93 | SVT | 21,263 (SV Run) | very strong |
| 98 | VT | 3,170 (V Run) | strong |

**Notable issues to flag upfront:**

- **idx 6 (Sin Tach) has no fzark TPs.** Only PTB-XL gives 826 positives. Production-distribution positives are missing entirely. Improvement here is fundamentally bounded by what PTB-XL teaches.
- **idx 93 (SVT) has only 68 combined positives.** Statistical power is thin. The fzark side has 26 SV-Run TPs — same problem as the cross-dataset analysis showed.
- **idx 9 (PVC) is the dual-event shared head.** Training treats IVB and V-Couplet as the same positive class; downstream pattern detection (couplet vs isolated beat) is delegated to the FP suppression layer.
- **idx 4 (Sinus Bradycardia)** interacts with the v2 FP suppression HR-gate (`mean_hr_bpm ≤ 56.3`). If the head's score distribution shifts, the HR-gate threshold may need re-calibration — the regression gate must catch this.

## 4. Strategy — three tiers ranked by risk

### Tier 1 — Linear probe on the 6 target heads (DEFAULT)

**This is the recommended starting point.** It is mathematically guaranteed to leave all 144 other heads bit-identical to base.

Recipe:

```python
model = load_ecgfounder(device)
for p in model.parameters():
    p.requires_grad = False

# Train ONLY the 6 rows of dense.weight + dense.bias indexed by 4, 5, 6, 9, 93, 98.
TARGET_IDX = [4, 5, 6, 9, 93, 98]
model.dense.weight.requires_grad = True
model.dense.bias.requires_grad   = True

# Mask the loss: BCE only over the 6 target heads
loss_mask = torch.zeros(150); loss_mask[TARGET_IDX] = 1.0

# Surgically zero the gradient on rows we don't want to update
def zero_non_target_rows(grad):
    new_grad = grad.clone()
    new_grad[[i for i in range(150) if i not in TARGET_IDX]] = 0
    return new_grad
model.dense.weight.register_hook(zero_non_target_rows)
model.dense.bias.register_hook(zero_non_target_rows)
```

Backbone is fully frozen, all 144 non-target classifier rows are gradient-blocked. The 6 target rows update against `L = BCE(logits[TARGET_IDX], labels[TARGET_IDX])`.

- **Trainable params**: 6 × 1024 + 6 = 6,150 (~0.005 % of the full model).
- **Compute**: <5 min on MPS for ~50k samples × 5 epochs.
- **Guarantee**: output on the other 144 heads is bit-identical to base (verified by post-training inference + `np.allclose`).
- **Risk**: cannot drift fzark behavior on non-target heads. The 6 target heads can change; the regression gate (§5) bounds how much.

### Tier 2 — Backbone LoRA + 6-head linear probe

Only adopt if Tier 1 plateaus and the 6 heads aren't improving enough. The backbone can shift, but architecturally:

- Add LoRA adapters at rank 8 on each Net1D stage (or just stages 5–6).
- Freeze base weights everywhere.
- Train: target-head BCE + distillation regularizer pulling the 144 non-target heads' logits toward base values: `L_total = L_BCE + λ · MSE(student_logits[non_target], base_logits[non_target].detach())` with λ ≈ 1.0.
- Same 144-head classifier-row freeze as Tier 1.
- **Mid-training assert**: every N steps, check `(student_logits[non_target] - base_logits[non_target]).abs().max() < 0.1`; abort otherwise.

Risk: the LoRA delta + distillation regularizer should keep non-target heads close to base but not bit-identical. The post-hoc regression gate is the deciding test.

### Tier 3 — full backbone fine-tune (NOT RECOMMENDED)

Documented for completeness only. The vanilla / masked v1 results in [README.md](README.md) and [README_masked.md](README_masked.md) show this path damages production. Do not ship without explicit approval.

## 5. CI regression gate (mandatory before adoption)

The regression gate is the asymmetric test that makes this safe:

```python
TARGET_IDX = [4, 5, 6, 9, 93, 98]
NON_TARGET_IDX = [i for i in range(150) if i not in TARGET_IDX]

def test_non_target_heads_unchanged(candidate_ckpt):
    """Tier 1: bit-identical. Tier 2: bounded drift."""
    base = np.load("res/tp_fzark_full_suppression/baseline_probs_tp.npy")
    cand = run_inference(candidate_ckpt, fzark_cohort)
    diff = np.abs(cand[:, NON_TARGET_IDX] - base[:, NON_TARGET_IDX]).max()
    if TIER == 1:
        assert diff < 1e-6, f"Tier 1 must be byte-identical: {diff}"
    else:  # Tier 2
        assert diff < 0.05, f"Tier 2 drift exceeded 0.05 on non-target head: {diff}"

def test_target_heads_no_regression(candidate_ckpt):
    """Per-head TP detection must not regress on fzark."""
    base = np.load(...); cand = run_inference(...)
    for idx in TARGET_IDX:
        # Per-event-type detection (handles the shared idx 9 = IVB + V Couplet)
        for et in [e for e, i in FZARK_LABEL_MAP.items() if i == idx]:
            mask = (df['Event Type'] == et).values
            if mask.sum() == 0: continue
            base_det = (base[mask, idx] >= 0.5).mean()
            cand_det = (cand[mask, idx] >= 0.5).mean()
            assert cand_det >= base_det - 0.005, (et, base_det, cand_det)

def test_target_heads_no_FP_increase(candidate_ckpt):
    """Per-head FP rate must not increase on fp_doctor_removed."""
    base_fp = np.load("res/fp_allclass_full_suppression/baseline_probs_full.npy")
    cand_fp = run_inference(candidate_ckpt, fp_cohort)
    for idx in TARGET_IDX:
        for et in [e for e, i in FZARK_LABEL_MAP.items() if i == idx]:
            mask = (fp_df['Event Type'] == et).values
            if mask.sum() == 0: continue
            base_rate = (base_fp[mask, idx] >= 0.5).mean()
            cand_rate = (cand_fp[mask, idx] >= 0.5).mean()
            assert cand_rate <= base_rate + 0.005, (et, base_rate, cand_rate)
```

Three guarantees, in order of importance:
1. The 144 non-target heads must be unchanged (Tier 1: bit-identical; Tier 2: ≤ 5 pp drift).
2. The 6 target heads' TP detection must not regress on fzark (within 0.5 pp tolerance).
3. The 6 target heads' FP rate must not increase on the doctor-removed cohort (within 0.5 pp tolerance).

**The candidate is adopted only if all three pass.** If a candidate improves a TP rate by 3 pp but raises the corresponding FP rate by 1 pp, it fails.

## 6. Data construction recipe

```python
# Build training set: positives from PTB-XL + Fzark TPs, negatives from Fzark FPs + PTB-XL negatives.
TARGET_IDX = [4, 5, 6, 9, 93, 98]

def build_labels(record_info):
    """Return a (6,) binary vector for the 6 target heads."""
    labels = np.zeros(6, dtype=np.float32)
    # Fzark TPs: positive on the head their event_type maps to
    if record_info['source'] == 'fzark_tp':
        idx = FZARK_LABEL_MAP.get(record_info['event_type'])
        if idx in TARGET_IDX:
            labels[TARGET_IDX.index(idx)] = 1.0
    # Fzark FPs: explicit negative on the head their event_type maps to.
    # All other 5 heads: unknown — must NOT contribute to the loss.
    elif record_info['source'] == 'fzark_fp':
        idx = FZARK_LABEL_MAP.get(record_info['event_type'])
        # idx is known to be 0 (FP) on that head; other 5 heads are ambiguous.
        # Use a per-record loss mask to skip the 5 ambiguous heads.
    # PTB-XL: 150-vector binary labels are pre-computed; restrict to 6 indices.
    elif record_info['source'] == 'ptbxl':
        labels = record_info['label_vec'][TARGET_IDX].astype(np.float32)
    return labels
```

Critical detail for FP records: a fzark FP labeled "Atrial Fibrillation" tells you the AFib head should be 0 on this strip, but says **nothing** about the other 5 heads (Bradycardia, Sin Tach, PVC, SVT, VT). The loss must mask the other 5 heads on FP records — only the AFib output is supervised. Without this, the model receives spurious gradient pushing other heads down on every fzark FP.

Sample composition target: rough 1:1 positive:negative ratio per head, oversampling small classes (Sin Tach, SVT). Random-sample with a fixed seed; persist sample IDs for reproducibility.

## 7. What this strategy explicitly does NOT do

- **Does not fine-tune on fuzzylead2 directly.** That dataset is out of scope here — its labels do not align with the 6 target heads cleanly (only AFib and Sin Tach overlap, and Sin Tach has only 12 fuzzylead2 positives).
- **Does not touch any of the 144 non-target heads' classifier rows.** Tier 1 enforces this by gradient hook; Tier 2 enforces this by classifier-row freeze + LoRA architecture.
- **Does not overwrite `1_lead_ECGFounder.pth`.** The new checkpoint is written separately (e.g. `1_lead_ECGFounder_6head_v2.pth`).
- **Does not bypass the regression gate.** No matter how promising a candidate's val_75deg ROC looks, if it fails any of the three CI tests it stays in `res/` as a record, not in `checkpoint/`.

## 8. Recommended execution order

1. **Build the data loader** (§6 — combined PTB-XL + Fzark TP + Fzark FP with per-record loss-mask). Smoke-test on a 1k subset.
2. **Run Tier 1** linear probe, 5 epochs. Run the CI regression gate.
3. **If Tier 1 produces meaningful improvement** (any of the 6 heads' TP det ↑ or FP rate ↓ by > 1 pp without violating the gate), ship that checkpoint as v2.
4. **If Tier 1 plateaus**, escalate to Tier 2 LoRA. Two ablations: all-stage LoRA vs stages-5–6 only. Same gate.
5. **Do not consider Tier 3** without explicit operator approval and a different data scope.

## 9. Expected outcomes per head (honest priors)

Reading from the data summary in §3:

| idx | Head | Realistic Tier 1 outcome |
|---|---|---|
| 5 | AFib | Best candidate — 21,080 positives + 1,793 hard FPs. Likely +0–1 pp TP, −2–5 pp FP. |
| 9 | PVC | Strong candidate — 3,462 positives + 12,877 FPs. Likely meaningful FP reduction on IVB. |
| 98 | VT | Strong candidate for FP reduction — 5,254 fzark V-Run TPs + 3,170 FPs. **However**, base V-Run TP detection is currently 0.0 % at t=0.5. The model would need to actually learn to fire here, not just suppress FPs. **This is the head most likely to either soar or fail outright.** |
| 4 | Bradycardia | Modest gain expected (already 94.4 % detection, 3.2 % FP). Watch for HR-gate interaction. |
| 93 | SVT | Thin data (68 combined positives). May not move statistically. |
| 6 | Sin Tach | Stuck without fzark TPs. PTB-XL alone is unlikely to fix the 99 % FP rate on fp_doctor — the cohort is too clinical, too different from ambulatory motion artifact. |

The aggregate-level prediction: **+0–2 pp TP retention, −3–8 pp aggregate FP rate**. Modest but real, all within the safety envelope.

## 10. Open questions

1. Is there a strong preference for single-checkpoint deployment vs. `DualHeadECGFounder`? If yes, this Tier 1 path produces exactly that — one checkpoint, fzark-preserved, 6 production heads improved.
2. Should idx 6 (Sin Tach) and idx 93 (SVT) be dropped from the target set given their thin data? Training them adds noise; freezing them is also clean. The Sin Tach 99 % FP rate is a known production pain point so dropping it has real cost.
3. What's the acceptable threshold for "no regression"? §5 uses 0.5 pp on per-head metrics. Tightening to 0 pp would block any candidate that has a mild noise drift on a borderline case; loosening to 1 pp would admit slightly-worse candidates if they improve elsewhere.
4. Does the v2 FP suppression layer's HR-gate (56.3 bpm for Bradycardia) need re-calibration once the Bradycardia head changes? The candidate evaluation should include suppression-on metrics too, not just raw head outputs.

---

*Drafted 2026-05-28 after the cross-dataset Founder ↔ PTB-XL ↔ Fzark mapping made clear which 6 heads have paired labels in both cohorts. The v1 strategy (FINETUNE_STRATEGY.md) remains valid for any task targeting PTB-XL-only labels.*
