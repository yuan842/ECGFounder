# PTB-XL — Baseline model (FP OFF) performance + comprehensive ECG SQI

**Date**: 2026-05-29
**Cohort**: **full PTB-XL, 21,799 records** (0 failed reads), lead II @ 500 Hz, fully labelled.
**Model**: base ECGFounder (`1_lead_ECGFounder.pth`), single-lead, **FP suppression OFF**, threshold 0.5.
**Scope**: the 7 global-rule labels {2,4,5,6,93,98,142}.
**Script**: [scripts/ptbxl_baseline_sqi.py](../../scripts/ptbxl_baseline_sqi.py) · CSVs in `res/ptbxl_baseline_sqi/`.

### Two framing facts
1. **PTB-XL has no accelerometer** → the FP-suppression algo is *inapplicable*. "FP off" is the only meaningful state (raw base model); there is no off/on axis to compare.
2. **SQI is ECG-only** here — `sqi.compute_sqi(..., acc=None)`: raw-stage (clip/flat/drift/HF/range) + band-stage (SNR/kurt/HR/rr_cv) + `sqi_score_ecg`. No motion / `sqi_score_amb`.

---

## 1. Baseline performance (FP off, t = 0.5) — 7 scope labels

| label | head | n_pos | sens | spec | PPV | F1 | AUROC | PR-AUC |
|---|---|---|---|---|---|---|---|---|
| **Atrial Fibrillation** | 5 | 1,514 | **0.976** | 0.968 | 0.692 | **0.810** | **0.990** | 0.936 |
| **Sinus Tachycardia** | 6 | 826 | 0.981 | 0.918 | 0.320 | 0.483 | **0.991** | 0.891 |
| **Bradycardia** | 4 | 637 | 0.970 | 0.852 | 0.165 | 0.282 | 0.941 | 0.335 |
| Supraventricular Run | 93 | 42 | 0.571 | 0.999 | 0.436 | 0.495 | **0.992** | 0.480 |
| Ventricular Run | 98 | 42* | 0.643 | 0.998 | 0.380 | 0.478 | 0.983 | 0.478 |
| **Normal ECG** | 2 | 9,514 | **0.167** | 0.979 | 0.858 | 0.279 | 0.836 | 0.763 |
| Pause | 142 | **0** | — | 1.000 | — | — | — | — |

\* the **same 42 records** carry both SV-Run (93) and V-Run (98) labels (identical columns — a PTB-XL→founder mapping artifact); treat 93/98 as one rare cohort.

**Reading by head:**
- **AFib & Sinus Tachycardia are excellent** — AUROC 0.99, sens ≈ 0.98. AFib PPV 0.69 (good even against 20k negatives).
- **Bradycardia**: high recall (0.97) but low PPV (0.17) — over-fires against the huge normal majority (spec 0.85 × 21k negatives = many FPs).
- **SV-Run / V-Run heads actually WORK on PTB-XL** — AUROC **0.99 / 0.98** — a sharp contrast with the fzark single-lead cohort where they were 0.61 / **0.36 (worse-than-chance)**. The heads *can* detect SVT/VT; the fzark failure is a **clean-clinical-data vs noisy-ambulatory-data** gap, not a broken head. (n=42, so wide CIs.) Sens is only 0.57/0.64 at 0.5 — a lower threshold would recover recall on clean data.
- **NORMAL ECG head is very conservative at 0.5**: sens **0.167** but PPV 0.858 and AUROC 0.836 — it *ranks* normal well and is almost always right when it fires, but rarely crosses 0.5. **0.5 is the wrong operating point for the Normal head** (it under-detects); its value lives at a lower threshold.
- **Pause has zero PTB-XL labels** — not evaluable on this dataset (only present in fzark).

---

## 2. ECG SQI — overall (full PTB-XL, median / IQR / mean)

| metric | median | IQR | mean |
|---|---|---|---|
| sqi_score_ecg | **0.781** | 0.684–0.846 | 0.766 |
| snr_proxy | 1.054 | 0.569–1.690 | 1.222 |
| dynamic_range_mv | 1.127 | 0.877–1.450 | 1.249 |
| clip_pct | **0.000** | 0–0 | 0.031 |
| flat_pct | 0.141 | 0.081–0.303 | 0.288 |
| baseline_drift | 0.029 | 0.014–0.053 | 0.043 |
| hf_noise_ratio | 0.009 | 0.005–0.017 | 0.016 |
| kurt | 9.63 | 4.5–16.0 | 11.1 |
| mean_hr_bpm | 72.5 | 63.8–84.5 | 76.6 |
| rr_cv | 0.031 | 0.015–0.099 | 0.081 |
| pct_physiologic_hr | 100 | 100–100 | 99.8 |

**PTB-XL is clinical-grade clean**: essentially zero saturation/flat-line, low HF-noise (0.009), low drift, `pct_physiologic_hr` 100 %, composite `sqi_score_ecg` 0.78. This is the high-quality end of the spectrum — contrast with MOVE (gel chest, rest flat_pct ~73 %, run clip ~6 %) and fzark ambulatory (more motion/noise).

---

## 3. ECG SQI by scope label — quality is uniform; rhythm features track the label

| label | n | sqi_score_ecg | snr | hf_noise | **mean_hr** | **rr_cv** | kurt |
|---|---|---|---|---|---|---|---|
| Normal ECG | 9,514 | **0.798** | 1.14 | 0.009 | 69.8 | 0.030 | 11.0 |
| Atrial Fibrillation | 1,514 | 0.769 | 0.93 | 0.009 | 90.7 | **0.203** | 9.0 |
| Bradycardia | 637 | 0.775 | 0.87 | 0.008 | **49.9** | 0.041 | 13.7 |
| Sinus Tachycardia | 826 | 0.736 | 1.14 | 0.011 | **108.7** | 0.016 | 5.0 |
| SV-Run / V-Run | 42 | 0.721 | 1.17 | 0.016 | **152.3** | 0.018 | 3.3 |

- **Signal quality barely varies by label** (sqi_score_ecg 0.72–0.80) — because PTB-XL is uniformly clean and (unlike fzark) has no false-positive cohort. **SQI is not a TP/FP discriminator here** — there are no FPs to separate.
- **The rhythm features cleanly track the diagnosis**, exactly as physiology predicts:
  - **AFib → rr_cv 0.203** vs Normal 0.030 (≈7×) — the irregular-irregular signature.
  - **Bradycardia → HR 49.9** (<60), **Sinus Tachy → HR 108.7** (>100), **SV/V-Run → HR 152** (fast).
  - Sharp QRS kurtosis drops with rate (Normal 11 → Run 3.3).
- So on PTB-XL the useful "quality" axis is really **rate/rhythm plausibility**, not noise — the opposite of the ambulatory cohorts where motion/HF-noise dominated.

---

## 4. PTB-XL vs the ambulatory cohorts (context)

| | PTB-XL (clinical, 12-lead resting) | fzark (ambulatory single-lead) | MOVE (chest gel + ACC) |
|---|---|---|---|
| Accelerometer | none → **FP algo N/A** | yes → motion gate active | yes |
| SQI composite (median) | **0.78** | TP 0.72 / FP 0.64 | rest flat-line heavy |
| clip / flat | ~0 % / 0.14 % | low | clip ~6 % (run), flat ~73 % (rest) |
| SVT/VT head AUROC | **0.99 / 0.98** | 0.61 / **0.36** | n/a (silent) |
| What separates events | **rate/rhythm** (HR, rr_cv) | **motion + SQI** (TP cleaner) | motion + HF-noise |

The single biggest cross-dataset finding: **the SVT/VT heads work on clean PTB-XL (AUROC ~0.99) but fail on noisy fzark single-lead (≤0.61)** — confirming those heads need clean signal, and the fzark deficit is a data-quality/lead problem, addressable by the fine-tuned head rather than by lowering thresholds into the noise floor.

---

## Key findings

1. **AFib and Sinus Tachycardia are production-grade on PTB-XL unaided** (AUROC ~0.99). Bradycardia has high recall but low precision against the normal majority.
2. **SV-Run / V-Run heads are strong on clean PTB-XL (AUROC 0.99/0.98)** — the opposite of their fzark single-lead behaviour — so the heads are fundamentally capable; the fzark gap is signal quality. (Same 42 records, small n.)
3. **The NORMAL ECG head under-detects at 0.5** (sens 0.17, AUROC 0.84, PPV 0.86) — it needs a lower operating point to be useful as a "normal" flag.
4. **PTB-XL is uniformly clean** (sqi_score_ecg ≈ 0.78, ~0 % clip) → SQI does not discriminate events here; instead **rate/rhythm features (HR, rr_cv) are the informative axis**, cleanly tracking each diagnosis (AFib rr_cv 0.20, Brady HR 50, Tachy 109, Run 152).
5. **FP suppression is moot on PTB-XL** (no accelerometer) — off/on comparisons belong on fzark/MOVE.

## Caveats
- Threshold fixed at 0.5 (per the global rule); AFib/Tachy/SVT/VT all show AUROC ≫ 0.5-operating-point sens, i.e. recall is threshold-limited, not signal-limited.
- SV-Run/V-Run n=42 (identical records) → wide CIs. Pause n=0 (no PTB-XL label).
- SQI computed on the single processed lead II (the model's input lead), not all 12 leads.
