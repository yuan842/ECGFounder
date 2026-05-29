# Archived Documents — Historical Reference Only

**⚠️ DO NOT CONSULT THESE FOR CURRENT BEHAVIOR.**
The sole live label-mapping guide is [res/GLOBAL_LABEL_MAP.md](../GLOBAL_LABEL_MAP.md). Anything below is preserved for historical context and superseded by the current code + GLOBAL_LABEL_MAP.

---

## Manifest

| File | Original location | Superseded by | Date archived | Summary |
|---|---|---|---|---|
| **Dataset Classification Cross-Mapping.xlsx** | (user-supplied) | [GLOBAL_LABEL_MAP.md](../GLOBAL_LABEL_MAP.md) | 2026-05-28 | Source-of-truth Excel ratifying the v3.1 Founder ↔ PTB-XL ↔ Fzark mapping plus PPV / reliability / clinical-support columns. Kept for permanent record. |
| label_reclassification_plan.md | res/ | v3 ontology in [label_config.py](../../label_config.py) + GLOBAL_LABEL_MAP.md §1 | (earlier) | The original v3 single-head reclassification plan. Implemented; document obsolete. |
| label_map_off_by_one_bug_report.md | res/ | v3 ontology in [label_config.py](../../label_config.py) | (earlier) | Bug report on the original v1/v2 off-by-one mistakes (idx 99→98, 143→142, etc.). All fixes shipped in v3; the test suite (test_ontology.py) now guards against regression. |
| finetune_fuzzylead2_FINETUNE_STRATEGY.md | res/finetune_fuzzylead2/FINETUNE_STRATEGY.md | (no replacement — strategy retired) | 2026-05-28 | v1 fuzzylead2 fine-tune strategy. Catastrophic-forgetting results led to v2/6-head rewrite, which itself was rejected by the regression gate. Approach abandoned in favor of `DualHeadECGFounder` + per-class downstream gates. |
| finetune_fuzzylead2_FINETUNE_STRATEGY_v2_6HEAD.md | res/finetune_fuzzylead2/FINETUNE_STRATEGY_v2_6HEAD.md | (no replacement — strategy retired) | 2026-05-28 | v2 scoped-to-6-heads fine-tune strategy. Linear probe ran, two candidate checkpoints produced (`_v2`, `_v3_posw`), both failed regression gate. Conclusion: PTB-XL alone cannot retrain these heads without distribution-shift damage. Use the existing base model + downstream FP suppression + per-class LR rankers instead. |
| finetune_fuzzylead2_README_vanilla.md | res/finetune_fuzzylead2/README.md | n/a | 2026-05-28 | Training report for the vanilla full-fine-tune on fuzzylead2 (5 epochs, BCE×150). Catastrophic forgetting on fzark V3.1 heads (aggregate detection 54.7 % → 0.2 %). Retained for the diagnostic table. |
| finetune_fuzzylead2_README_masked.md | res/finetune_fuzzylead2/README_masked.md | n/a | 2026-05-28 | Training report for the masked-loss variant (gradient zeroed on 140 inactive heads). Partial recovery — backbone still drifted, AFib head broke (97.8 % → 6.8 %), high-confidence detection collapsed. Retained for the diagnostic table. |

## Why retired

The v1 / v2 / masked / state-dict-surgery experiments all explored ways to add training signal on top of the base 1-lead ECGFounder using the fuzzylead2 PTB-XL-derived single-lead cohort. Each variant either:
- destroyed fzark V3.1 performance outright (vanilla full FT), or
- shifted calibration enough that the strict regression gate rejected the checkpoint (masked-loss, 6-head v2, 6-head v3 pos-weight).

The fzark cohort's recording distribution (ambulatory, motion-rich) differs from PTB-XL's (clinical resting) in ways that BCE-with-logits training cannot reconcile without either (a) sharing the backbone with another distribution actively being learned, or (b) introducing a calibration shift the production threshold of 0.5 cannot absorb.

The currently-shipped production stack is:
- **Base model** `1_lead_ECGFounder.pth` — unchanged from upstream, the V3.1 ontology is the only addition.
- **v2 FP suppression** — motion/HR/SNR feature gates ([multiclass_fp_suppression.py](../../multiclass_fp_suppression.py)).
- **DualHead two-checkpoint inference** ([dual_head_ecgfounder.py](../../dual_head_ecgfounder.py)) — runs base + an optional supplementary checkpoint with per-head routing, when a PTB-XL-label use case calls for it.
- **Per-class LR rankers** (proven path: IVB/ISB at 90/90 TP retention / FP suppression — see `PERCLASS_FP_FINETUNED.md` at the project root).

Nothing in the archived files reflects current production behavior.

## When to read these anyway

- Reproducing the v1/v2/masked failure modes for a paper / writeup.
- Debugging a similar catastrophic-forgetting pattern in a future fine-tune attempt — the diagnostic tables and gradient-hook bug walk-through may save time.
- Audit trail for why a given approach was abandoned.

If you find yourself reading any of these to plan **new** work, stop and read [GLOBAL_LABEL_MAP.md](../GLOBAL_LABEL_MAP.md) instead.
