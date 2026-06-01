# L1 iterations — base vs iter1 (lead-II) vs iter2 (fuzzy) vs iter3 (union)

> **PRODUCTION MODEL = iter2, designated `fuzzySL`** (`res/scope_overlay/fuzzySL.pth`, copy of `scope_projection_fuzzy.pth`). It is the overlay's `DEFAULT_CKPT` ([overlay/scope_overlay.py](../../overlay/scope_overlay.py)), so `ScopedDetector`/`load_l1` load it by default. Chosen for **precision**: best lead-II macro PPV (0.66) and the most conservative, lowest-false-alert head behaviour — the right bias for a single-lead alerting product. Routing unchanged (L1 for {4,5,6}; base for {93,98,142}); L2 OFF.

Four L1 regimes compared on both PTB-XL fold-10 test distributions. Only the L1-affected heads `{4 Brady, 5 AFib, 6 Sinus-Tachy}` shown (93/98/142 route to base in every model → identical). L2 OFF. Eval: [scripts/compare_overlay_iterations.py](../../scripts/compare_overlay_iterations.py).

- **base** — raw backbone scope head
- **iter1** — L1 trained on PTB-XL **lead-II** (`scope_projection.pth`)
- **iter2** — L1 trained on PTB-XL **fuzzy** 45/60/75/90° (`scope_projection_fuzzy.pth`)
- **iter3** — L1 trained on **lead-II + fuzzy union**, fresh from base (`scope_projection_union.pth`)

## Macro over Brady / AFib / Sinus-Tachy

| test set | metric | base | iter1 (leadII) | iter2 (fuzzy) | iter3 (union) |
|---|---|---:|---:|---:|---:|
| **lead-II** | PR-AUC | 0.690 | **0.776** | 0.745 | 0.767 |
| | PPV@0.5 | 0.391 | 0.492 | **0.660** | 0.503 |
| **fuzzy** | PR-AUC | 0.700 | **0.784** | 0.772 | 0.770 |
| | PPV@0.5 | **0.532** | 0.465 | 0.463 | 0.437 |

## Per-head (lead-II fold-10, n=2,198)

| head | metric | base | iter1 | iter2 | iter3 |
|---|---|---:|---:|---:|---:|
| Bradycardia | PR-AUC | 0.346 | **0.601** | 0.489 | 0.555 |
| | PPV | 0.162 | 0.168 | **0.647** | 0.248 |
| AFib | PR-AUC | 0.909 | **0.922** | 0.920 | 0.918 |
| | PPV | 0.683 | **0.724** | 0.641 | 0.643 |
| Sinus-Tachy | PR-AUC | 0.813 | 0.805 | 0.825 | **0.828** |
| | PPV | 0.328 | 0.583 | **0.691** | 0.618 |

## Per-head (fuzzy fold-10, 4 angles pooled, n=8,792)

| head | metric | base | iter1 | iter2 | iter3 |
|---|---|---:|---:|---:|---:|
| Bradycardia | PR-AUC | 0.369 | **0.628** | 0.582 | 0.591 |
| AFib | PR-AUC | **0.927** | 0.917 | 0.899 | 0.888 |
| | PPV | **0.799** | 0.750 | 0.685 | 0.640 |
| Sinus-Tachy | PR-AUC | 0.803 | 0.807 | **0.835** | 0.831 |

## Verdict — the union does not combine the best of both; it averages them

1. **Ranking (PR-AUC) is saturated across all three L1 iterations** — lead-II ≈0.75–0.78, fuzzy ≈0.77–0.78. Adding fuzzy to training (iter3) does **not** beat the lead-II-only iter1 on either distribution. The projection's ranking gain over base (+0.08–0.09 macro) is real and robust, but **which leads it trained on barely matters for ranking**.

2. **iter3 (union) is a middle ground that wins nothing outright.** Lead-II PR-AUC 0.767 (just under iter1's 0.776); lead-II PPV 0.503 (well under iter2's 0.660); fuzzy PPV 0.437 (the **lowest** of all four). Mixing the two distributions dilutes the per-head calibration rather than reconciling it.

3. **The differentiator is precision (PPV@0.5), and it's calibration-bound:**
   - best **lead-II** PPV → **iter2** (0.660, the conservative fuzzy-trained model)
   - best **fuzzy** PPV → **base** (0.532)
   - iter1 and iter3 sit in between on both, mastering neither.

4. **AUROC flat across all four** everywhere — discrimination is a backbone property; the iterations only move calibration and the operating-point precision/recall balance.

## Recommendation

- **No union benefit** — iter3 does not justify replacing iter1. For a single lead-II model, **iter1** remains the best (highest PR-AUC, best AFib).
- If **precision** is the goal, **iter2** is strongest on lead-II; on rotated leads the **base** head is actually the most precise.
- The genuine cross-angle fix is still **per-deployment-lead temperature recalibration**, not the training-lead mix — all three L1 models rank within ~0.01–0.02 of each other on each distribution; only the threshold calibration separates them, and it's distribution-specific.
- 93/98/142 unchanged (base) in every iteration; Pause still needs a non-PTB-XL source.
