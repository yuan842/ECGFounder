# L2 design — GT-matched deterministic arbiter

L2 ([overlay/arbiter.py](../../overlay/arbiter.py)) is the deterministic multi-label rule layer: 6 scope probs (+ NSR feature + HR) → per-head fire/no-fire Decisions. **Suppress/merge only**, no weights, unit-tested, **OFF by default**. Its structural rules are a **direct image of the PTB-XL GT co-occurrence matrix** ([gt_cofire_counts.csv](../ptbxl_cofiring/gt_cofire_counts.csv)) — so the emitted multi-label distribution matches GT.

## GT structure → rules

| GT observation (6-head scope) | rule |
|---|---|
| {Brady, AFib, Sinus-Tachy} pairwise co-occur **0×** (physiologically exclusive) | **EXCLUSION_GROUPS = [(4,5,6)]** — at most one fires |
| SVT-Run ≡ V-Run co-extensive (**Jaccard 1.0**) | **COUPLE_GROUPS = [(93,98)]** — co-fire kept, merged to one alert |
| Pause / runs co-occur freely (critical, rare) | **free heads** — never suppressed for co-firing |

### Exclusion resolution (when ≥2 of {4,5,6} fire)
1. **Directional override** `DIRECTIONAL = [(AFib, Tachy)]` — AFib beats Tachy (AF-RVR; co-firing Tachy is the FP, realness PPV ~1%).
2. **Keep best margin** — among the rest, keep the head with the largest `prob − τ` (confidence over its own policy threshold); suppress the others.

This collapses the bulk of spurious co-fires the raw model produces (AFib+Tachy, Brady+AFib, Brady+Tachy — all GT=0).

## Config

`ArbiterConfig`: master `enabled` (default **False** → pure pass-through), per-rule toggles (`nsr_contradiction`, `rate_exclusion`, `hr_plausibility`, `merge_runs`), thresholds. `GT_MATCHED_CONFIG = ArbiterConfig(enabled=True)` is the ready-to-use ON config. `ScopedDetector(..., enable_l2=True)` turns it on while reusing the per-head L1 policy thresholds; default stays OFF.

## Verification — emitted-vs-GT co-fire ([scripts/eval_l2_cofire.py](../../scripts/eval_l2_cofire.py))

PTB-XL fold-10, routed production probs, pairwise co-fire counts GT / L2-off / L2-on:

| pair | GT | L2 off | L2 on | |
|---|---:|---:|---:|---|
| Brady+AFib | 0 | 20 (ll) / 19 (fz) | **0 / 0** | rate-excl ✅ |
| Brady+Tachy | 0 | 3 / — | **0** | rate-excl ✅ |
| AFib+Tachy | 0 | 14 / 46 | **0 / 0** | rate-excl ✅ |
| SVT+V-Run | 5 (ll) / 20 (fz) | 5 / 37 | **5 / 37** | couple kept ✅ |

**Result: PASS** on both lead-II and fuzzy — every GT-zero rate pair eliminated, the coupled pair preserved. Residual AFib×run / Tachy×run co-fires persist by design (runs are free, critical heads; GT≈0 but we never suppress a run). Unit tests: [tests/test_arbiter.py](../../tests/test_arbiter.py) (10 tests).

## Caveats
- The rate-head exclusion is **physiology** (you can't be brady & tachy; AFib is its own rhythm), confirmed by GT — so it generalizes beyond PTB-XL. The SVT≡VT couple is partly a PTB-XL label-encoding artifact + single-lead inseparability, so "merge" is an honesty decision, not a clinical-identity claim.
- PTB-XL is resting 12-lead (one dominant rhythm/record); ambulatory single-lead may have more genuine co-occurrence (AFib with PVC runs/pauses) — which is why runs/pause are left free rather than forced exclusive.
- L2 is **suppress/merge only** and **OFF by default**; turning it on changes the operating point, not the L1 ranking.
