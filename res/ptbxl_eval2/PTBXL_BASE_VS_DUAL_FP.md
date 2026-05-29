# PTB-XL — Base vs DualHead (fine-tune), FP-algo OFF/ON

**Date**: 2026-05-29
**Cohort**: PTB-XL random sample n=4,000 (seed 0), fully labelled → supervised metrics vs ground truth.
**Models**: base backbone vs DualHeadECGFounder (production). Lead II, `for_ptbxl()` preprocessing.
**Script**: [scripts/eval_ptbxl_base_vs_dual.py](../../scripts/eval_ptbxl_base_vs_dual.py); artifacts in `res/ptbxl_eval2/`.

---

## Two facts that frame this eval

1. **DualHead = base on 142 of 150 heads.** It routes only 8 PTB-XL-specific heads through the fuzzy checkpoint (NORMAL ECG, IRBBB, LVH, AFL, LAFB, ILBBB, LPFB, RVH). On every other head it is **bit-identical to base** (verified: max\|Δ\| = 0.00e+00 across the other 142 heads). So the entire fine-tune effect lives in those 8 heads.
2. **The FP-suppression algo is motion-based; PTB-XL has no accelerometer.** AFib/SV-Trig/V-Trig gates are *inapplicable*. Only the Bradycardia HR-gate runs. "FP off/on" is therefore not a meaningful axis on PTB-XL — reported only for completeness.

---

## 1. Fine-tune effect (base vs DualHead) on the 8 fuzzy-routed heads — supervised, threshold 0.5

| head | n_pos | BASE AUROC | DUAL AUROC | ΔAUROC | BASE F1 | DUAL F1 |
|---|---|---|---|---|---|---|
| **LAFB** | 315 | 0.391 | **0.938** | **+0.547** | 0.00 | 0.25 |
| LPFB | 43 | 0.539 | **0.747** | +0.208 | 0.02 | 0.05 |
| RVH | 31 | 0.688 | 0.818 | +0.130 | 0.01 | 0.00 |
| NORMAL ECG | 1717 | 0.846 | 0.883 | +0.037 | 0.28 | 0.15 |
| LVH | 417 | 0.693 | 0.725 | +0.032 | 0.30 | 0.31 |
| ATRIAL FLUTTER | 23 | 0.987 | 0.988 | +0.001 | 0.42 | **0.57** |
| INCOMPLETE RBBB (18) | **0** | — | — | — | — | — |
| INCOMPLETE LBBB (62) | **0** | — | — | — | — | — |

**mean ΔAUROC (dual − base) on the evaluable heads: +0.159.**

- **LAFB is the headline: AUROC 0.391 → 0.938 (+0.55).** Base is *worse than chance* on LAFB at single-lead; the fuzzy fine-tune fixes it. This confirms the `val_75deg` finding (LAFB 0.38→0.97) generalizes to the PTB-XL test data.
- LPFB (+0.21), RVH (+0.13), NORMAL ECG (+0.04), LVH (+0.03) all improve. Atrial Flutter AUROC was already 0.99; the fine-tune improves its **F1 0.42→0.57** (better PPV at threshold).
- **IRBBB (18) and ILBBB (62) are not labelled in PTB-XL** (not in `PTBXL_ACTIVE_CLASSES`) → 0 positives, not evaluable here. They're fuzzylead2-active heads only; their improvement is validated on `val_75deg`, not PTB-XL.
- **All 142 non-fuzzy heads: base ≡ DualHead exactly** → the fine-tune is surgically confined and production-safe.

---

## 2. FP-algo OFF vs ON

**Motion gates inapplicable** — PTB-XL has no accelerometer channel, so the AFib (≤5 mG), SV-Trig (≥15 mG), and V-Trig (≥24 mG) gates cannot run. The only applicable rule is the **Bradycardia HR-gate** (suppress if HR > 56.3 bpm, HR computed from the ECG):

| Bradycardia (idx 4), n_pos=134 | TP | FP | sens | ppv |
|---|---|---|---|---|
| FP algo OFF | 127 | 560 | 0.948 | 0.185 |
| FP algo ON (HR-gate) | 76 | 210 | 0.567 | 0.266 |

The HR-gate removes 350 FP but also **51 TP** — sensitivity collapses 0.95 → 0.57 for only a modest PPV gain (0.19 → 0.27). On clean resting PTB-XL, the HR-gate is a poor trade (many PTB-XL "Bradycardia"-labelled records have detected HR above the 56.3 cut, so the gate suppresses real positives). **Every other head: FP-on ≡ FP-off** (no applicable rule).

**Conclusion: the FP algo is effectively a no-op on PTB-XL** — it is a motion-artifact suppressor and PTB-XL has neither motion artifact nor an accelerometer. The off/on comparison is meaningful only on the ambulatory cohorts (fzark, MOVE).

---

## Net verdict

- **Fine-tune (DualHead) measurably improves the PTB-XL-specific heads** — mean +0.159 AUROC across the 6 evaluable ones, headlined by LAFB (+0.55, from worse-than-chance to 0.94), and improves Atrial Flutter F1 — while leaving all 142 other heads bit-identical. The fuzzylead2 fine-tune generalizes from its `val_75deg` validation to the PTB-XL test data.
- **FP-algo OFF/ON is not a useful axis on PTB-XL** — it's motion-based and PTB-XL has no motion sensor; the lone applicable HR-gate hurts more than it helps. Reserve FP off/on comparisons for fzark / MOVE.

## Caveats
- Single random sample (seed 0), threshold 0.5, no bootstrap CIs.
- IRBBB / ILBBB not labelled in PTB-XL → not evaluable here (validated on val_75deg only).
- Rare heads (RVH n=31, AFL n=23, LPFB n=43) have wide sampling noise on F1/PPV; AUROC is the more stable comparison.
