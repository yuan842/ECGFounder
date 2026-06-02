# S0 deep drill — SNR (dB) and baseline drift across AFib / Normal / Noisy

## SNR proxy → decibels

```
bp        = bandpass(sig, 5–25 Hz)        # in-band ECG (QRS) energy
residual  = sig − bp                      # out-of-band (noise) energy
snr_proxy = var(bp) / var(residual)       # linear power ratio
SNR_dB    = 10 · log10(snr_proxy)         # decibel (power) form
```

Because `snr_proxy` is a **power** (variance) ratio, the decibel conversion uses the 10·log10 form (not 20·log10). SNR_dB = 0 dB ⇔ equal in-band and out-of-band power; >0 dB ⇔ ECG dominates; <0 dB ⇔ noise dominates.

Baseline drift = `std(lowpass<0.5 Hz of raw)` in µV (raw ADC÷1000→mV, ×1000→µV). n: AFib 758, Normal 5076, Noisy 279.

## SNR in dB — cut-offs (p5 / p10 / p15 / p20 · median · p80 / p85 / p90 / p95)

| group | p5 | p10 | p15 | p20 | median | p80 | p85 | p90 | p95 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| AFib | -8.05 | -5.14 | -3.88 | -2.74 | **0.76** | 3.06 | 3.45 | 4.15 | 5.00 |
| Normal | -9.41 | -7.13 | -5.55 | -4.47 | **-0.50** | 2.43 | 2.97 | 3.52 | 4.36 |
| Noisy | -16.56 | -14.24 | -13.06 | -12.16 | **-8.15** | -1.90 | -0.94 | 0.17 | 1.92 |

_(values in dB)_

## Baseline drift (µV) — cut-offs

| group | p5 | p10 | p15 | p20 | median | p80 | p85 | p90 | p95 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| AFib | 5.9 | 7.5 | 8.7 | 9.7 | **17.1** | 40.3 | 48.7 | 57.3 | 84.9 |
| Normal | 5.8 | 7.4 | 8.8 | 10.2 | **20.3** | 46.2 | 55.1 | 70.1 | 98.2 |
| Noisy | 16.6 | 22.8 | 30.8 | 34.7 | **78.6** | 149.9 | 169.2 | 204.6 | 236.0 |

## Separation (boxplot read-out)

| comparison | metric | AUC | Cohen's d |
|---|---|---:|---:|
| Noisy vs clean | SNR (dB) | 0.817 | -1.51 |
| Noisy vs clean | baseline drift | 0.840 | +1.78 |
| AFib vs Normal | SNR (dB) | 0.589 | +0.28 |
| AFib vs Normal | baseline drift | 0.463 | -0.12 |

**Read:** SNR (dB) and baseline drift both separate **Noisy from clean** well (AUC ≈ 0.83–0.84) and in opposite directions (Noisy = low SNR, high drift). Neither separates **AFib from Normal** (AUC ≈ 0.5–0.6) — as intended, both are clean. AFib sits slightly *higher* in SNR than Normal here, so a Noisy gate on these features is gentle on the AFib target.

Figure: `docs/snr_baseline_boxplot_challenge2017.png`.
