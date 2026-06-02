# Challenge 2017 — S0 (raw) snr_proxy: distribution, CIs & Noisy cutoff

snr_proxy on the center 10 s RAW window (prior to preprocessing). 95% CIs are bootstrap percentile intervals (10,000 resamples, seed 42).

| group | n | mean | 95% CI (mean) | median | 95% CI (median) | IQR |
|---|---:|---:|---|---:|---|---|
| AFib | 758 | 1.352 | [1.288, 1.420] | 1.192 | [1.123, 1.279] | [0.664–1.858] |
| Normal | 5076 | 1.104 | [1.080, 1.128] | 0.890 | [0.863, 0.921] | [0.442–1.557] |
| Noisy | 279 | 0.381 | [0.325, 0.441] | 0.153 | [0.124, 0.198] | [0.074–0.494] |

## Pairwise separation (Mann–Whitney U, two-sided)

| pair | U p-value |
|---|---|
| AFib vs Normal | 2.39e-15 |
| AFib vs Noisy | 2.10e-70 |
| Normal vs Noisy | 1.57e-68 |

## Recommended cutoff — Noisy vs clean (AFib + Normal)

Rule: **flag NOISY if snr_proxy < τ**. snr_proxy as a noise detector has **ROC-AUC = 0.817** (clean n=5834, noisy n=279).

| criterion | τ | sensitivity (Noisy) | specificity (clean) | Youden J | precision | F1 |
|---|---:|---:|---:|---:|---:|---:|
| **Youden-optimal (recommended)** | **0.30** | 0.674 | 0.846 | 0.520 | 0.173 | 0.276 |
| balanced (sens≈spec) | 0.47 | 0.742 | 0.742 | 0.483 | 0.121 | 0.208 |

### Sweep near the recommendation

| τ | sens (Noisy) | spec (clean) | Youden J |
|---:|---:|---:|---:|
| 0.10 | 0.348 | 0.963 | 0.311 |
| 0.30 | 0.674 | 0.846 | 0.520 |
| 0.50 | 0.753 | 0.726 | 0.479 |
| 0.80 | 0.846 | 0.562 | 0.408 |
| 1.30 | 0.925 | 0.350 | 0.274 |

> **Recommendation: gate on `snr_proxy < 0.30` at S0** — catches 67% of Noisy while keeping 85% of clean AFib/Normal. Lower τ → fewer clean rejects; higher τ → more noise caught. The Noisy and clean distributions overlap (AUC≈0.82), so SNR alone is a useful but not perfect gate — combine with baseline-drift for a stronger rule.
