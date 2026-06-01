# This model vs baseline — FUZZY derived-lead (PTB-XL fold-10)

**Generalization test.** L1 was trained on **lead-II only**; here the held-out fold-10 records are scored as derived single-lead at **{45, 60, 75, 90}°**. Same routing as deployed: L1 for {4 Brady, 5 AFib, 6 Sinus-Tachy}, base head for {93 SVT-Run, 98 V-Run, 142 Pause}; **L2 OFF**. Labels from `csv/ptbxl_label.csv`. Eval: [scripts/eval_scope_overlay_fuzzy.py](../../scripts/eval_scope_overlay_fuzzy.py); data: [metrics_fuzzy.csv](metrics_fuzzy.csv).

## Pooled across 4 angles (n=8,792 = 2,198 records × 4) — baseline → model

| head | n_pos | src | AUROC | PR-AUC | PPV@0.5 |
|---|---:|---|---|---|---|
| 4 Bradycardia | 256 | L1 | 0.949→0.949 | 0.369→**0.628** | 0.155→0.157 |
| 5 AFib | 608 | L1 | 0.979→0.981 | 0.927→0.917 | 0.799→**0.750** |
| 6 Sinus-Tachy | 328 | L1 | 0.991→0.985 | 0.803→0.807 | 0.643→**0.488** |
| 93 SVT-Run | 20 | base | 0.997→0.997 | 0.330 | 0.333 (unchanged) |
| 98 V-Run | 20 | base | 0.996→0.996 | 0.310 | 0.311 (unchanged) |
| 142 Pause | 0 | base | — no positives — | | |

**Macro (5 evaluable heads):** AUROC 0.983→0.982 (−0.001) · PR-AUC 0.548→**0.598 (+0.051)** · PPV 0.448→0.408 (**−0.040**) · Sens 0.846→0.839 (−0.007). Per-angle numbers are stable (Brady PR 0.63–0.66, Tachy PPV 0.46–0.52 across angles) — see `metrics_fuzzy.csv`.

## The key finding — gains transfer partially, and unevenly

This is **not** the same story as lead-II. Compared to the lead-II test:

| | Bradycardia PR-AUC | AFib PPV | Sinus-Tachy PPV |
|---|---|---|---|
| **lead-II** (train dist.) | 0.35→0.60 ✅ | 0.68→0.72 ✅ | 0.33→0.58 ✅ |
| **fuzzy** (rotated) | 0.37→0.63 ✅ | 0.80→0.75 ❌ | 0.64→0.49 ❌ |

1. **Bradycardia's PR-AUC win is robust** — nearly doubles on both lead-II and fuzzy, at every angle. The strongest, most transferable improvement.
2. **AFib / Sinus-Tachy precision gains do NOT transfer to rotated leads — they reverse.** Why: on the fuzzy leads the *baseline* is already sharper/higher-PPV (AFib base PPV 0.80, Tachy 0.64 — vs 0.68 / 0.33 on lead-II). L1's temperature/calibration was fit to lead-II's operating point, where the raw heads under-fired; applied to the rotated leads it over-fires, pushing more positives and **lowering PPV**. The PR-AUC (ranking) is roughly preserved — it's the **0.5-threshold calibration that doesn't carry over**.
3. **AUROC ≈ flat everywhere** — discrimination/ranking generalizes across angle fine; only the threshold-dependent precision is distribution-specific.
4. **SVT/VT/Pause unchanged** — routed to base, so identical to baseline at every angle (n=5/angle, not meaningful).

## Conclusion

On its **training distribution (lead-II)** the overlay is a clear win (macro PR-AUC +0.05, PPV +0.06). On the **rotated derived-leads** the *ranking* gain persists (macro PR-AUC still +0.05, driven by Bradycardia) but the **precision gain does not — macro PPV drops −0.04** because L1's calibration is lead-II-specific and the baseline is already well-calibrated at other angles.

**Implication.** The lead-II-only L1 is the right model *for lead-II*. For robust performance across arbitrary electrode angle, either (a) **train L1 with the fuzzy augmentation** (the union run, which the user deferred), or (b) **re-fit the per-head temperature per deployment lead**. As-is, deploy the overlay on lead-II-like signals; on rotated leads prefer the baseline head for AFib/Sinus-Tachy precision, or keep L1 only for Bradycardia.
