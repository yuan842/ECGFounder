# This model vs baseline — PTB-XL (fold-10 test)

**Baseline:** raw frozen backbone — `sigmoid(150-logit)` read at the 6 scope heads.
**This model:** the deployed overlay — backbone → **L1 (routed)** → L2 (**OFF**). Per-head routing: L1 calibrated probs for {4 Brady, 5 AFib, 6 Sinus-Tachy}; raw backbone head for {93 SVT-Run, 98 V-Run, 142 Pause}. L2 rule arbiter is OFF, so this compares the *scoring* layer only.
**Test set:** PTB-XL fold 10, 2,198 held-out records (patient-stratified 1-8/9/10; base backbone is split-exempt). Threshold 0.5.

## Per-head (baseline → this model)

| head | n_pos | source | AUROC | PR-AUC | PPV@0.5 | Sens@0.5 |
|---|---:|---|---|---|---|---|
| 4 Bradycardia | 64 | L1 | 0.944→**0.960** | 0.346→**0.601** | 0.162→0.168 | 0.953→0.922 |
| 5 AFib | 152 | L1 | 0.976→**0.982** | 0.909→**0.922** | 0.683→**0.725** | 0.934→0.934 |
| 6 Sinus-Tachy | 82 | L1 | 0.986→0.980 | 0.813→0.805 | 0.328→**0.583** | 0.976→0.939 |
| 93 SVT-Run | 5 | base | 0.996→0.996 | 0.319→0.319 | 0.200→0.200 | 0.200→0.200 |
| 98 V-Run | 5 | base | 0.997→0.997 | 0.341→0.341 | 0.300→0.300 | 0.600→0.600 |
| 142 Pause | 0 | base | — no positives in PTB-XL — | | | |

## Macro-average (5 evaluable heads: 4,5,6,93,98)

| metric | baseline | this model | Δ |
|---|---:|---:|---:|
| AUROC | 0.980 | 0.983 | **+0.003** |
| PR-AUC | 0.546 | 0.598 | **+0.052** |
| PPV@0.5 | 0.335 | 0.395 | **+0.061** |
| Sens@0.5 | 0.733 | 0.719 | −0.014 |

## Read

- **The model improves the three data-rich heads and leaves the rare heads untouched.** Because SVT/VT/Pause route to the base head, the deployed model **never incurs L1's degradation on those** (the rows are identical to baseline) — the per-head routing is doing its job.
- **Biggest wins:** Bradycardia PR-AUC nearly doubles (0.35→0.60); Sinus-Tachy PPV +0.26 (0.33→0.58, far fewer false tachy calls at 0.5); AFib improves on AUROC, PR-AUC and PPV simultaneously.
- **Cost:** a slight recall give-back on Brady/Tachy (−0.03) in exchange for the large precision/PR-AUC gains — a favorable trade at the 0.5 operating point. AFib sensitivity unchanged (0.934).
- **AUROC ≈ flat** — the backbone's ranking was already strong; L1's contribution is **calibration and precision at the operating threshold**, not discrimination.
- **SVT-Run / V-Run numbers are n=5** — too few PTB-XL positives to be meaningful; reported for completeness. Pause is absent from PTB-XL entirely.

**Net:** on PTB-XL the overlay is a clear improvement over the raw backbone — macro PR-AUC +0.05 and PPV +0.06 — concentrated where the labels are dense (Brady, AFib, Sinus-Tachy), with zero regression on the heads it routes to base. Source: [metrics.csv](metrics.csv) / [REPORT.md](REPORT.md).
