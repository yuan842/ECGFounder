# Step-by-Step SQI Through Preprocessing, by Activity — MOVE

**Date**: 2026-05-29
**Channel**: `ecg:gel` (chest gel electrode).
**Cohort**: MOVE, 200 windows/activity sampled (seed 0), rest→motion gradient.
**Script**: [scripts/move/sqi_by_stage.py](../../scripts/move/sqi_by_stage.py) · data: [move_sqi_by_stage.csv](move_sqi_by_stage.csv)

This is the empirical version of the stage-tapping rule ([docs/SQI_DESIGN.md §2](../../docs/SQI_DESIGN.md)): compute the SQI panel **at every preprocessing stage** and watch each metric evolve, per activity. It shows exactly which stage masks/rescues which quality dimension — and therefore which stage each SQI must be measured at.

## Pipeline stages (faithful to `ECGPreprocessor.process`, MOVE = 50 Hz / 500 Hz)

| Stage | Operation |
|---|---|
| S0 raw | lead-selected window, untouched |
| S1 +notch | 50 Hz iirnotch (Q=30) |
| S2 +bandpass | 0.67–40 Hz Butterworth (order 4) |
| S3 +baseline | minus median-filtered baseline (0.4 s) |
| S4 +normalize | winsorize [1.5, 98.5] pctl + z-score → **model input** |

(resample + crop are no-ops on MOVE: already 500 Hz / 5000 samples.)

---

## clip_pct (saturation) — *must be measured on RAW*

| stage | baseline | walk_before | run | walk_after |
|---|---|---|---|---|
| **S0 raw** | **4.25** | **4.31** | **6.16** | **2.74** |
| S1 notch | 0.11 | 0.66 | 0.41 | 0.47 |
| S2 bandpass | 0.04 | 0.05 | 0.07 | 0.08 |
| S3 baseline | 0.04 | 0.05 | 0.06 | 0.07 |
| S4 normalize | 1.02 | 2.00 | 2.64 | 3.07 |

**The headline stage-tapping proof.** Genuine saturation is visible only on **S0 raw** (run 6.2 %). By S2 it's ~0 — the bandpass smooths the rails away. Then S4 winsorize+z-score *re-introduces* an artificial ~1–3 % "clip" (the percentile clip + rescale pushes samples back to new extremes). **Measuring clip on the model-input (S4) tensor would both miss the real saturation AND report a fake one** — exactly the bug this rule prevents. Clip is a raw-stage metric, full stop.

## baseline_drift — *must be measured PRE-baseline-removal (S0–S1)*

| stage | baseline | walk_before | run | walk_after |
|---|---|---|---|---|
| S0 raw / S1 notch | 0.074 | 0.099 | 0.104 | 0.126 |
| S2 bandpass | 0.004 | 0.009 | 0.026 | 0.026 |
| **S3 baseline** | 0.005 | 0.011 | 0.041 | 0.033 |
| S4 normalize | 0.022 | 0.045 | 0.074 | 0.090 |

Drift is real on S0 (rises rest→motion, as expected). By S3 it is **removed by construction** (that stage's whole job). Measuring it after S2/S3 reports ≈0 regardless of the true drift → must tap on S0/S1.

## hf_noise (>40 Hz power ratio) — *raw, before the 40 Hz bandpass*

| stage | baseline | walk_before | run | walk_after |
|---|---|---|---|---|
| **S0 raw** | 0.002 | 0.004 | **0.023** | 0.008 |
| S2 bandpass | 0.000 | 0.000 | 0.001 | 0.001 |
| S4 normalize | 0.002 | 0.003 | 0.004 | 0.003 |

High-frequency (muscle/EMG) noise peaks during `run` (0.023) on S0, then the 0.67–40 Hz bandpass erases it (S2 ≈0). Tap on raw or it reads ~0 everywhere.

## snr_proxy (QRS-band ratio) — *post-bandpass is correct; it IMPROVES through filtering*

| stage | baseline | walk_before | run | walk_after |
|---|---|---|---|---|
| S0 raw | 0.304 | 0.769 | 1.057 | 1.057 |
| S2 bandpass | 0.528 | 1.058 | 1.314 | 1.390 |
| **S3 baseline** | 0.738 | 1.442 | 1.633 | 1.798 |
| S4 normalize | 0.700 | 1.358 | 1.640 | 1.815 |

SNR rises stage-by-stage as noise is removed (S0→S3 roughly doubles it), then is z-score-invariant (S3≈S4). It is meaningful from S2 onward. Note SNR still *increases* with activity at every stage — the motion-energy-as-signal confound (§ SQI_DESIGN §9) persists regardless of stage; that's a metric limitation, not a stage issue.

## kurt (R-peak peakedness) — *bandpass-dependent, normalize-stable*

| stage | baseline | walk_before | run | walk_after |
|---|---|---|---|---|
| S0 raw | 3.33 | 5.41 | 4.55 | 3.39 |
| S2 bandpass | 5.72 | 7.09 | 4.53 | 3.76 |
| S3 baseline | 8.58 | 9.51 | 6.01 | 5.15 |
| S4 normalize | 5.90 | 9.91 | 4.87 | 3.98 |

Kurtosis is amplified by bandpass+baseline (R-peaks sharpened relative to a flattened baseline); winsorize at S4 trims it back. Best read at S2/S3.

## dyn_range (mV) — *raw only; z-score destroys absolute scale*

| stage | baseline | walk_before | run | walk_after |
|---|---|---|---|---|
| **S0 raw** | 0.72 | 1.52 | 2.57 | 2.91 |
| S4 normalize | 2.17 | 4.07 | 4.45 | 5.45 |

Real amplitude (rest 0.72 mV → motion 2.9 mV) is only on S0; after z-score the "range" is in arbitrary σ-units and no longer comparable across windows.

---

## Cross-activity reading (the rest→motion gradient)

| dimension | rest (baseline) | running | what it means |
|---|---|---|---|
| flat_pct (raw) | **73 %** | 21 % | gel electrode loses contact at dead rest; movement seats it |
| dyn_range (raw) | 0.72 mV | 2.57 mV | amplitude grows with motion (signal + artifact) |
| clip_pct (raw) | 4.3 % | 6.2 % | saturation worst during motion |
| baseline_drift (raw) | 0.074 | 0.104 | drift rises with motion |
| hf_noise (raw) | 0.002 | 0.023 | muscle noise peaks during run |

The two ends of the activity axis fail differently: **rest fails by flat-line/lead-off** (73 % flat, 0.72 mV), **motion fails by saturation + HF noise + drift**. A single quality threshold can't catch both — they're orthogonal failure modes, and each is only visible at its own pipeline stage.

---

## Conclusions (validates the stage-tapping rule empirically)

1. **clip_pct, dyn_range, baseline_drift, hf_noise → tap on S0 (raw).** Filtering/normalization erase or fake them by S2–S4. clip is the sharpest example: real on S0 (6 %), ~0 by S2, *artificially* 1–3 % at S4.
2. **snr_proxy, kurt → tap on S2/S3 (filtered).** They're defined on the QRS band and improve through filtering; meaningless on raw.
3. **Never compute any SQI on S4 (the model-input tensor).** Winsorize + z-score both masks real artifacts (clip→0 at S2) and manufactures fake ones (clip→1–3 % at S4; dyn_range in σ-units).
4. **Rest and motion are distinct failure regimes** — flat-line at rest, saturation/noise at motion — so the SQI panel needs both raw-amplitude metrics (catch rest) and raw-noise metrics (catch motion); a composite must include both families.

These results are exactly why `sqi.py` splits `compute_raw_sqi` (S0 taps) from `compute_band_sqi` (S2/S3 taps) and never touches the normalized output.
