# L1 training policy — max-F1, per-head bounded sensitivity loss

**Policy (rev 4, 2026-06-01).** For each L1 detection head, choose the operating point that **maximizes F1** while **improving specificity**, subject to **sensitivity dropping no more than the per-head budget** below the baseline backbone head (at its 0.5 operating point). Report **PR-AUC** and the **max-F1 threshold** per head.

**Per-head sensitivity-loss budget** (`POLICY_MAX_SENS_DROP`) — currently uniform 5 pp:

| head | budget |
|---|---:|
| 4 Bradycardia | 5 pp |
| 5 AFib | 5 pp |
| 6 Sinus-Tachy | 5 pp |

> rev 1: max-spec at sens floor, 10 pp global. rev 2: max-F1, 5 pp global. rev 3: max-F1, per-head (Brady 10 pp). **rev 4 (current): Brady reverted to 5 pp — uniform 5 pp.** The per-head map is retained so any head can get a different budget without code changes.

## How it's operationalized

The BCE training gives the head's *ranking*; the policy sets the *decision threshold*. Per L1 head (Brady 4, AFib 5, Sinus-Tachy 6 — the heads the overlay routes through L1), on the validation split:

1. `base_sens` = baseline head sensitivity at 0.5.
2. `floor = base_sens − drop(h) + margin` where `drop(h)` is the per-head budget (margin 0.02 for val→test drift).
3. **τ\* = argmax F1** over thresholds **subject to sens(τ) ≥ floor**. (Among operating points that keep recall within budget, take the best precision/recall balance.)
4. Report PR-AUC (threshold-free ranking quality) and τ\*.

τ\* is stored per head in the checkpoint (`ScopeProjection.decision_threshold`), applied at inference via the arbiter's per-head `fire_threshold`. Heads routed to base (93/98/142) stay at 0.5. The per-head budget lives in `POLICY_MAX_SENS_DROP` in [scripts/train_scope_overlay.py](../../scripts/train_scope_overlay.py); `--max-sens-drop` (default 0.05) is the fallback for heads not listed; `--margin` (0.02). Implemented in [scripts/calibrate_l1_policy.py](../../scripts/calibrate_l1_policy.py) (post-training, authoritative) and the trainer (`--spec-opt`).

## Per-deployment-lead (unchanged finding)

A single global threshold cannot serve both lead-II and the rotated (fuzzy) leads — the score distributions conflict (Bradycardia). **Thresholds are calibrated for the deployment distribution.** `--dist fuzzy` (default) for the single-lead product; `--dist leadII` for a clean-lead-II deployment; `--dist leadII fuzzy` for a robust global threshold (min of the two, less spec gain).

## Production model — fuzzySL, fuzzy-calibrated (max-F1, uniform 5 pp)

`fuzzySL.pth` ships with thresholds **{Brady 0.431, AFib 0.863, Sinus-Tachy 0.860}**. Fold-10 fuzzy test, baseline (0.5) → production:

| head | Sens | Spec | PPV | F1 | PR-AUC | Δsens | Δspec | ΔF1 | policy |
|---|---|---|---|---|---|---:|---:|---:|---|
| Brady | 0.969→0.930 | 0.842→**0.867** | 0.155→0.173 | 0.268→**0.292** | 0.369→**0.582** | −0.039 | +0.025 | +0.024 | ✅ PASS |
| AFib | 0.934→0.891 | 0.983→**0.989** | 0.799→**0.863** | 0.861→**0.877** | 0.927→0.899 | −0.043 | +0.007 | +0.016 | ✅ PASS |
| Sinus-Tachy | 0.976→0.936 | 0.979→**0.985** | 0.643→**0.706** | 0.775→**0.805** | 0.803→**0.835** | −0.040 | +0.006 | +0.030 | ✅ PASS |

All heads: **F1 up, specificity up, sensitivity loss < 5 pp**. (SVT-Run/V-Run/Pause route to base, unchanged; Pause has no PTB-XL positives. PR-AUC is the L1-head ranking; AUROC unchanged by the threshold.)

**lead-II** (lead-II-calibrated, for reference): Brady 0.953→0.938 sens / 0.852→0.858 spec / F1 0.277→0.280 (thr 0.05); AFib PPV 0.68→0.84 / F1 +0.09; Tachy PPV 0.33→0.69 / F1 +0.30 — see [PER_HEAD_DETAILED.md](PER_HEAD_DETAILED.md).

## Deploying on lead-II

The fuzzy thresholds are tuned for rotated leads. For a lead-II deployment refit:
```
python3 -m scripts.calibrate_l1_policy --dist leadII
```

## Notes
- AUROC/PR-AUC are unaffected by threshold choice — the policy only moves the operating point.
- Verification + full panel (incl. PR-AUC and the chosen threshold per head): [scripts/compare_prod_vs_base_detailed.py](../../scripts/compare_prod_vs_base_detailed.py) → [PROD_VS_BASE_DETAILED.md](PROD_VS_BASE_DETAILED.md) / `prod_vs_base_detailed.csv`.
