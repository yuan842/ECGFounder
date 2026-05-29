# AFib Non-Motion ECG Feature Characterization
## TP (Atrial Fibrillation, n=200) vs FP (n=46) — Comparative Analysis

**Date**: 2026-05-26
**Goal**: Evaluate four candidate non-motion discriminators and design a filter that achieves >90% TP retention AND >90% FP suppression.

---

## Headline Result

A **single non-motion feature — `mean_rr` (mean R-R interval)** — achieves **91.0% TP retention with 97.8% FP suppression**, decisively crossing the 90/90 target that motion features alone could not reach. Three of the four feature families are highly discriminative; one (multi-window persistence) is weakly informative.

| Approach | TP retention | FP suppression |
|---|---|---|
| Motion features alone (best) | 92.9% | 80.4% |
| **`mean_rr ≤ 700 ms`** (single rule) | **91.0%** | **97.8%** |
| **`mean_rr ≤ 850 ms`** (balanced) | **92.0%** | **93.5%** |
| Logistic regression (all 12 ECG features) | 98.9% | 100.0% |
| Decision tree (depth 2, ECG features) | 96.6% | 100.0% |

---

## Methodology

ECG strips (~30 s @ 128 Hz) were reconstructed from each event's JSON. R-peaks were detected with a Pan-Tompkins-style detector (5–20 Hz bandpass → derivative → squaring → 150 ms integration → adaptive threshold → 250 ms refractory). All four feature families were then computed:

1. **RR-irregularity** — RMSSD, pNN50, sample entropy (m=2), CV-RR, mean RR
2. **P-wave detection** — amplitude / presence / consistency in the 60–200 ms pre-R window
3. **Signal-quality index (SQI)** — flat-line %, baseline drift, kurtosis, SNR proxy, clip %
4. **Multi-window persistence** — overlapping 10 s windows, AFib-flagged if RMSSD > 100 ms; report fraction flagged, longest consecutive run

Mann–Whitney U was used for significance (sample sizes asymmetric and distributions non-normal).

---

## 1. RR-Interval Irregularity

| Metric | TP median (p25–p75) | FP median (p25–p75) | p-value | Direction |
|---|---|---|---|---|
| **mean_rr (ms)** | **599 (564–622)** | **1133 (1060–1220)** | 1.9e-19 *** | **TP < FP — strongest** |
| rmssd (ms) | 181 (162–210) | 418 (335–650) | 4.3e-13 *** | FP > TP (unexpected) |
| pnn50 (%) | 70 (64–76) | 86 (75–92) | 2.5e-07 *** | FP > TP |
| samp_en | 1.82 (1.55–2.22) | 0.92 (0.55–1.39) | 5.5e-08 *** | TP > FP |
| cv_rr | 0.21 (0.19–0.23) | 0.27 (0.22–0.41) | 3.4e-04 *** | FP > TP |

### Interpretation
- **`mean_rr` is the single dominant discriminator.** TP AFib events have HR ≈ 100 bpm (599 ms); FPs are **bradycardic** (HR ≈ 53 bpm, mean_rr 1133 ms). The FP cohort in this dataset is overwhelmingly **bradyarrhythmias and sinus pauses mis-flagged as AFib**, not motion artifacts.
- **Sample entropy splits the cohorts cleanly in the opposite direction**: TP AFib is genuinely chaotic (entropy 1.82), while FP RR sequences are *more repetitive* (entropy 0.92) — long flat intervals broken by occasional pauses produce high RMSSD but low entropy. This is a powerful signature.
- **Counter-intuitive RMSSD/pNN50**: these are higher in FP than TP because long pauses inflate beat-to-beat differences. A naive "high RMSSD = AFib" rule would actually fail; **direction matters**.

### Single-feature operating points
| Filter | TP retention | FP suppression |
|---|---|---|
| `mean_rr ≤ 704 ms` | 96.6% | 100.0% (on cleaned subset n=179/21) |
| `mean_rr ≤ 700 ms` (NaN→reject) | **91.0%** | **97.8%** (full n=200/46) |
| `mean_rr ≤ 850 ms` | 92.0% | 93.5% |
| `samp_en ≥ 1.43` | 81.6% | 76.2% |

---

## 2. P-Wave Detection Score

| Metric | TP median | FP median | p-value |
|---|---|---|---|
| p_amp_mean | 0.06 | 0.01 | 1.5e-04 *** |
| p_present_pct (%) | 17.0 | 29.0 | 2.8e-07 *** |
| p_consistency | 0.39 | 0.61 | 8.4e-03 ** |

### Interpretation
Classic AFib lacks discrete P-waves, so we'd expect TP < FP on these metrics — and indeed **`p_present_pct` and `p_consistency` confirm it** (FP shows more "P-wave-like" content). But the discrimination is weaker than RR features because:
- The PR window can contain T-wave remnants from the prior beat at high HR, falsely positive
- AFib f-waves sometimes mimic low-amplitude P-waves
- Heavy motion noise in some FPs pollutes the PR window

**Best single P-wave threshold**: `p_present_pct ≤ 20.8%` → 67.0% TP / 66.7% FP suppression. Useful as an **adjunct** but not a primary discriminator.

---

## 3. Signal-Quality Index (SQI)

| Metric | TP median | FP median | p-value |
|---|---|---|---|
| flat_pct (%) | 0.0 | 0.0 | 0.64 ns |
| baseline_drift | 0.011 | 0.001 | 1.2e-12 *** |
| **kurt** | **3.73 (3.15–4.30)** | **7.97 (6.37–11.11)** | 1.8e-19 *** |
| snr_proxy | 0.49 | 1.31 | 1.6e-06 *** |
| clip_pct (%) | 1.01 | 1.01 | 5.8e-05 *** |

### Interpretation
- **Kurtosis is the standout SQI feature** — and almost as discriminative as `mean_rr`. FP events have kurtosis ~8 (very peaky) vs TP ~4 (more normal). High kurtosis here marks the **few, isolated R-peaks** of bradycardic FPs against a quiet baseline — the same population separation we see in `mean_rr`. It is essentially redundant with `mean_rr`.
- **`baseline_drift`** is *higher* in TP than FP — counter-intuitive at first, but consistent with the picture: real AFib events in this dataset come from tracings with normal physiological breathing/respiratory variation; FPs come from artificially "clean" but bradycardic strips.
- **`flat_pct`** is non-informative (≈0 for both cohorts).
- **`snr_proxy`** higher in FP just reflects the same "few big R-peaks against quiet baseline" geometry.

### Single-feature operating point
| Filter | TP retention | FP suppression |
|---|---|---|
| `kurt ≤ 4.52` | 88.3% | 90.5% |

Useful primarily as a **confirmatory** check alongside `mean_rr`.

---

## 4. Multi-Window Persistence

Windowing parameters: 10 s windows, 5 s hop (≈5 windows per event); flag = RMSSD > 100 ms.

| Metric | TP median (p25–p75) | FP median (p25–p75) | p-value |
|---|---|---|---|
| persistence_pct (%) | 100 (95–100) | 100 (40–100) | 9.4e-03 ** |
| longest_run | 5 (4.75–5) | 5 (2–5) | 1.7e-02 * |
| persistence_score | 1.00 (0.91–1.00) | 1.00 (0.16–1.00) | 1.2e-02 * |

### Interpretation
- Medians coincide because **both AFib and bradyarrhythmias produce high per-window RMSSD** (long RR gaps trigger the flag).
- The discriminator lives in the **lower quartile of FP** — about 25% of FPs show ≤40% persistence, meaning their "AFib look" doesn't hold across all windows. TP almost never drops below 95%.
- **Best single threshold** `persistence_score ≥ 0.46` → 86% TP / 47.6% FP suppression — weak.
- This feature is **most useful as a tie-breaker** for borderline `mean_rr` cases (650–850 ms region), not a primary filter.

---

## 5. Recommended Production Filter

### Tier 1 — Primary rule (achieves 90/90 on its own)
```
KEEP if  mean_rr ≤ 700 ms  AND  n_peaks ≥ 8 in 25 s window
REJECT otherwise.
```
| Performance | Value |
|---|---|
| TP retention | **91.0%** (182 / 200) |
| FP suppression | **97.8%** (45 / 46 rejected) |
| Min(TP_ret, FP_sup) | 91.0 |

### Tier 2 — "Bradycardic AFib safety net" (optional)
```
KEEP also if  mean_rr ≤ 1000 ms  AND  rmssd ≥ 80 ms  AND  samp_en ≥ 1.3
```
This second clause rescues genuine bradycardic AFib (slow but irregular). On this dataset it adds 1–2 TPs but also re-admits 1 FP — net wash. Recommended only if external evidence suggests bradycardic-AFib prevalence in the target population.

| Combined Tier 1 + Tier 2 | TP retention | FP suppression |
|---|---|---|
| | 91.5% | 95.7% |

### Tier 3 — Full ECG + Motion ensemble (highest performance, more complex)
```
score = +2.6*z(mean_rr) - 0.9*z(samp_en) - 0.9*z(cv_rr) + 0.9*z(kurt)
        + motion_filter_R_score
REJECT if score > 0.5 (logistic regression cutoff)
```
| Combined ECG + Motion LR | TP retention | FP suppression |
|---|---|---|
| best operating point | ~98% | ~100% |

(Coefficients pre-computed by the design script; calibrate on a held-out cohort before deployment.)

---

## 6. Why This Dataset Behaves the Way It Does

The "False Positive" cohort here is **not motion artifacts** as we initially hypothesized — it is **mis-classified bradyarrhythmias**:

- Mean HR ~53 bpm (vs ~100 bpm in TP)
- High RMSSD driven by long pauses, not beat-to-beat chaos
- Low sample entropy = repetitive pattern, not AFib's signature randomness
- Low baseline drift = clean but slow signal

The arrhythmia detector flagged these as AFib because they passed an "irregular RR" heuristic, but a clinician removed them because they lack the rapid, chaotic character of true AFib. **A simple heart-rate floor (`HR ≥ 85 bpm`, equivalent to `mean_rr ≤ 700 ms`) is the dominant correction.**

Motion remains a useful **secondary** filter for the small residual of FPs that are tachycardic but motion-corrupted (1 of 46 here passes the HR rule but fails motion checks).

---

## 7. Practical Recommendation

| Deployment context | Use |
|---|---|
| **Production AFib alarm filter** | **Tier 1 only** (`mean_rr ≤ 700 ms` + `n_peaks ≥ 8`). Simple, transparent, exceeds 90/90 target. |
| Bradycardic-prevalent population | Tier 1 + Tier 2 |
| Research / maximum-accuracy | Tier 3 ensemble |
| Inputs unavailable (no ECG signal) | Fall back to Motion Filter R (92.9% / 80.4%) |

The 90/90 target is reachable with **a single ECG feature** computed from data already present in every event's JSON — no additional sensor or new compute pipeline required.

---

## Files Produced
- `res/motion_analysis/data/afib_tp_ecg_features.csv` — 200 TP feature rows
- `res/motion_analysis/data/afib_fp_ecg_features.csv` — 46 FP feature rows
- `res/motion_analysis/data/afib_ecg_feature_comparison.csv` — per-metric statistics
- `ecg_feature_analysis.py` — extraction pipeline
- `ecg_filter_design.py` — filter optimization

*Generated 2026-05-26.*
