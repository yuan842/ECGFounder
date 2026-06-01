# Baseline vs Production (fuzzySL) — detailed metrics, PTB-XL + fuzzy

**Baseline:** raw frozen-backbone scope heads. **Production:** deployed overlay — **fuzzySL** L1 (iter2, fuzzy-trained) for {4 Brady, 5 AFib, 6 Sinus-Tachy}; base head for {93 SVT-Run, 98 V-Run, 142 Pause}; **L2 OFF**. Point metrics at **threshold 0.5**; AUROC/PR-AUC threshold-free. Test = PTB-XL fold-10 (held out, patient-stratified). Eval: [scripts/compare_prod_vs_base_detailed.py](../../scripts/compare_prod_vs_base_detailed.py); data: [prod_vs_base_detailed.csv](prod_vs_base_detailed.csv).

`L1*` = head served by fuzzySL; the rest are base in both models (so identical).

## PTB-XL lead-II (fold-10, n=2,198)

| head | src | model | n_pos | Sens | Spec | PPV | NPV | F1 | AUROC | PR-AUC |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 4 Brady | L1* | base | 64 | 0.953 | 0.852 | 0.162 | 0.998 | 0.277 | 0.944 | 0.346 |
| | | **prod** | 64 | 0.516 | **0.992** | **0.647** | 0.986 | **0.574** | 0.939 | **0.489** |
| 5 AFib | L1* | base | 152 | 0.934 | 0.968 | 0.683 | 0.995 | 0.789 | 0.976 | 0.909 |
| | | **prod** | 152 | 0.941 | 0.961 | 0.641 | 0.995 | 0.763 | 0.977 | **0.920** |
| 6 Sinus-Tachy | L1* | base | 82 | 0.976 | 0.922 | 0.328 | 0.999 | 0.491 | 0.986 | 0.813 |
| | | **prod** | 82 | 0.927 | **0.984** | **0.691** | 0.997 | **0.792** | 0.985 | **0.825** |
| 93 SVT-Run | base | base=prod | 5 | 0.200 | 0.998 | 0.200 | 0.998 | 0.200 | 0.996 | 0.319 |
| 98 V-Run | base | base=prod | 5 | 0.600 | 0.997 | 0.300 | 0.999 | 0.400 | 0.997 | 0.341 |
| 142 Pause | base | base=prod | 0 | — | 1.000 | — | 1.000 | — | — | — |

## FUZZY derived-lead (fold-10, 4 angles pooled, n=8,792)

| head | src | model | n_pos | Sens | Spec | PPV | NPV | F1 | AUROC | PR-AUC |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 4 Brady | L1* | base | 256 | 0.969 | 0.842 | 0.155 | 0.999 | 0.268 | 0.949 | 0.369 |
| | | **prod** | 256 | 0.910 | 0.877 | 0.182 | 0.997 | 0.303 | 0.947 | **0.582** |
| 5 AFib | L1* | base | 608 | 0.934 | 0.983 | **0.799** | 0.995 | **0.861** | 0.979 | **0.927** |
| | | prod | 608 | 0.933 | 0.968 | 0.685 | 0.995 | 0.790 | 0.982 | 0.899 |
| 6 Sinus-Tachy | L1* | base | 328 | 0.976 | 0.979 | **0.643** | 0.999 | **0.775** | 0.991 | 0.803 |
| | | prod | 328 | 0.970 | 0.966 | 0.521 | 0.999 | 0.678 | 0.991 | **0.835** |
| 93 SVT-Run | base | base=prod | 20 | 0.650 | 0.997 | 0.333 | 0.999 | 0.441 | 0.997 | 0.330 |
| 98 V-Run | base | base=prod | 20 | 0.700 | 0.996 | 0.311 | 0.999 | 0.431 | 0.996 | 0.310 |
| 142 Pause | base | base=prod | 0 | — | 1.000 | — | 1.000 | — | — | — |

## Interpretation

**The production model trades recall for precision/specificity — a deliberate low-false-alert bias.** This is clearest on lead-II:

- **Bradycardia (lead-II):** the standout. base fires on almost everything (Sens 0.95 but PPV 0.16, Spec 0.85 → 5 of 6 alerts false). Production tightens to Sens 0.52 / **Spec 0.99 / PPV 0.65**, lifting **F1 0.28→0.57** and PR-AUC 0.35→0.49. Far fewer false brady alerts, at the cost of missing ~half the true ones at 0.5.
- **Sinus-Tachy (lead-II):** same pattern — **PPV 0.33→0.69, Spec 0.92→0.98, F1 0.49→0.79**, Sens 0.98→0.93. A big precision win for ~5 pts of recall.
- **AFib (lead-II):** ≈neutral (already strong) — Sens 0.93→0.94, PPV 0.68→0.64, PR-AUC 0.909→**0.920**, F1 ~unchanged.

**On the fuzzy (rotated) leads the trade reverses for AFib/Sinus-Tachy** because the baseline is already high-precision there:

- **AFib (fuzzy):** production is **worse** — PPV 0.80→0.69, F1 0.86→0.79, PR-AUC 0.927→0.899. fuzzySL's conservative calibration over-fires relative to an already-sharp baseline.
- **Sinus-Tachy (fuzzy):** mixed — PR-AUC up (0.803→**0.835**) but threshold PPV/F1 down (0.64→0.52 PPV).
- **Bradycardia (fuzzy):** still a ranking win (PR-AUC 0.37→**0.58**) with a small F1 gain (0.27→0.30); PPV stays low because base brady is pathologically over-firing at every angle.

**Other observations:**
- **AUROC is flat** for every head on both datasets (±0.006) — discrimination is the backbone's; the overlay moves the operating point, not the ranking.
- **NPV ≈ 0.99–1.00 everywhere** — these are rare events, so true negatives dominate; NPV is not a discriminating metric here.
- **Specificity is where the production gain concentrates on lead-II** (Brady 0.85→0.99, Tachy 0.92→0.98) — i.e. fewer false positives, the metric a single-lead alerting product cares about most.
- **SVT-Run / V-Run / Pause identical** (routed to base) — Pause has 0 PTB-XL positives (Sens/PPV undefined, Spec 1.0); SVT/VT n=5–20, not meaningful.

## Bottom line

On its deployment target (**lead-II**) the production model is a clear improvement on the false-alert axis: **Bradycardia F1 +0.30, Sinus-Tachy F1 +0.30, specificity → 0.98–0.99**, AFib neutral-to-better — exactly the conservative profile chosen for fuzzySL. On **rotated leads** the recall-for-precision trade is unfavourable for AFib/Sinus-Tachy (baseline already precise there), so PPV/F1 regress even as ranking holds — reinforcing that **per-deployment-lead threshold/temperature recalibration** is the remaining lever.
