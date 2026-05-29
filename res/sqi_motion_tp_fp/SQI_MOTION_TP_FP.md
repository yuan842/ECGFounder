# SQI & Motion Characterization — TP vs FP cohorts (no detection)

**Date**: 2026-05-29
**Scope**: pure **signal** characterization — gravity-removed motion + the full SQI panel
(`sqi.compute_sqi`). **No model inference, no detection.**
**Cohorts** (stratified seed 42, ≤500/class): **TP** = `ecg_tp_fzark` (clinician-confirmed) vs
**FP** = `ecg_fp_doctor removed1` (clinician-removed false positives). TP n=3,022 · FP n=5,427 windows.
**Reader**: ECG /magnification → mV; ACC /2048 → g (fzark device), then `compute_sqi`.
**Script**: [scripts/sqi_motion_tp_fp.py](../../scripts/sqi_motion_tp_fp.py) · CSVs in `res/sqi_motion_tp_fp/`.
**Effect size**: Cohen's d = (mean_TP − mean_FP)/pooled_sd. |d|≥0.5 medium, ≥0.8 large.

---

## 1. Overall — TP vs FP (median; d on means)

| metric | TP median | FP median | Cohen's d | reading |
|---|---|---|---|---|
| **mean_motion_mg** | 11.06 | 11.92 | **−0.39** | FP higher motion (means 12.1 vs 18.2) |
| std_motion_mg | 49.2 | 49.7 | −0.84 | FP more motion variability |
| max_motion_mg | 399.7 | 399.9 | −0.65 | FP larger motion spikes |
| dynamic_range_mv | 0.87 | 1.12 | −0.06 | ~equal |
| flat_pct | 0.0 | 0.0 | −0.15 | rarely flat (resting gel data) |
| clip_pct | 0.0 | 0.0 | −0.15 | rarely clips |
| baseline_drift | 0.002 | 0.003 | −0.18 | FP drifts more |
| **hf_noise_ratio** | 0.001 | 0.008 | **−0.25** | FP noisier (HF) |
| **snr_proxy** | 1.42 | 1.02 | **+0.41** | **TP cleaner** |
| kurt | 13.8 | 7.6 | +0.06 | TP sharper QRS (noisy metric) |
| pct_physiologic_hr | 100 | 100 | −0.10 | ~equal |
| mean_hr_bpm | 76.8 | 91.4 | −0.44 | FP higher rate |
| **sqi_score_ecg** | 0.82 | 0.76 | **+0.48** | **TP higher quality** |
| **sqi_score_amb** | 0.72 | 0.64 | **+0.65** | **TP higher (motion-fused)** |

**Headline**: across the pooled cohorts, **TP windows are cleaner and lower-motion than FP** — higher
SNR (+0.41), higher composite SQI (+0.48 ECG / +0.65 motion-fused), lower HF noise, lower motion
variability/spikes. Raw saturation/flat metrics don't separate (resting gel ECG rarely clips). The
separating axes are **SNR, HF-noise, baseline-drift, motion energy, and the composites** — not clip/flat.

---

## 2. Per-event — which axis separates TP from FP

median TP vs FP; **motion d** = mean_motion_mg effect, **amb d** = sqi_score_amb effect.

| event | n (TP/FP) | motion TP→FP | motion d | SNR TP→FP | HF TP→FP | amb d | dominant separating axis |
|---|---|---|---|---|---|---|---|
| **Atrial Fibrillation** | 500/500 | **0.32 → 15.8** | **−1.55** | 1.79→0.57 | 0.000→0.006 | **+2.81** | **motion AND SQI (both huge)** |
| **Bradycardia** | 500/500 | 9.98→11.1 | −0.64 | 1.20→0.63 | 0.007→0.029 | **+1.44** | **SQI (SNR, HF-noise)** |
| **Pause** | 82/500 | 9.0→9.8 | −0.25 | 0.97→0.65 | **0.000→0.140** | **+1.00** | **SQI — HF-noise (d≈−1.0)** |
| Ventricular Run | 500/500 | 12.7→15.2 | −0.62 | 1.41→1.02 | 0.003→0.004 | +1.10 | motion + SQI (moderate) |
| Isolated Ventricular Beat | 500/500 | 11.4→13.5 | −0.32 | 2.85→1.66 | 0.000→0.010 | +0.94 | SQI (SNR) |
| Multiple Event | 116/500 | 3.8→11.3 | −0.67 | 0.97→0.67 | 0.014→0.020 | +1.00 | motion + SQI |
| Prolonged RR Interval | 169/17 | 10.9→11.5 | −0.76 | 3.44→1.16 | 0.000→0.012 | +0.79 | SQI (SNR) |
| Isolated Supraventricular Beat | 500/500 | 11.6→10.9 | +0.18 | 2.43→1.86 | 0.000→0.006 | +0.26 | weak |
| Supraventricular Run | 26/500 | 13.6→11.5 | +0.08 | 0.99→0.75 | 0.000→0.011 | +0.37 | weak |
| Ventricular Couplet | 36/500 | 13.9→12.6 | −0.12 | 1.94→1.77 | 0.000→0.004 | +0.37 | weak/SQI |
| **Supraventricular Trigeminy** | 76/500 | **35.1 → 11.5** | **+4.21** | 1.48→1.19 | 0.001→0.010 | **−2.37** | **INVERTED motion (TP high-motion)** |
| **Ventricular Trigeminy** | 16/500 | **34.0 → 12.3** | **+1.10** | 1.80→0.85 | 0.001→0.006 | −0.60 | **INVERTED motion** + SQI |
| **Supraventricular Couplet** | 500/500 | 10.8→11.5 | −0.34 | **0.64→2.14** | 0.012→0.005 | **−0.46** | **INVERTED SQI (TP worse SNR)** |

---

## 3. What this means for the split motion / SQI suppressor

The TP/FP separation axis is **event-specific** — exactly the case for two independently-tuned algos:

1. **AFib → motion gate is ideal (and SQI corroborates).** True AFib is essentially **resting (median 0.32 mG)**; false AFib is **15.8 mG** (d=−1.55) — the cleanest separation in the whole study, and the textbook justification for the production AFib motion gate (≤5 mG). SQI also separates strongly (amb d=+2.81), so an SQI gate would be a good *second* line.
2. **Bradycardia & Pause → SQI gate, not motion.** Their motion barely differs (d≈−0.6/−0.25), but they separate on **signal quality**: false Bradycardia is noisier (HF d=−0.70, SNR d=+1.38); **false Pause is an HF-noise artifact** — HF-noise 0.000 (TP) → 0.140 (FP), d≈−1.0. The production Bradycardia HR-gate ignores this; an **HF-noise/SNR SQI gate would target these FPs directly**.
3. **SV-/V-Trigeminy → INVERTED motion gate.** True trigeminy windows are **high-motion** (35 vs 12 mG, d=+4.2 for SV-Trig) — so the gate must keep *high*-motion and drop low-motion, matching the production inverted gates (SV-Trig motion≥15, V-Trig motion≥24). These events are currently out of detection scope.
4. **SV-Couplet → inverted SQI (hard).** True SV-Couplet has *worse* SNR than false (0.64 vs 2.14) — no simple quality gate helps; out of scope.

This maps cleanly onto `fp_suppression/` device profiles: **motion gate for AFib**, **SQI (HF/SNR) gate for Bradycardia + Pause**, **inverted motion for trigeminy**.

---

## Caveats
- `std_motion_mg` / `max_motion_mg` medians (~49 / ~400 mG) are dominated by transient acc spikes near the device's range ceiling — `mean_motion_mg` is the informative, production-aligned motion metric.
- Raw `clip_pct` / `flat_pct` are ~0 in both cohorts (resting gel ECG) — unlike MOVE, they carry no TP/FP signal here.
- Small-n events (SV-Bigeminy FP n=1, Prolonged-RR FP n=17, V-Trig TP n=16, V-Couplet TP n=36, SV-Run TP n=26) → unstable d (some NaN); read directionally only.
- fzark device units (mG via /2048); thresholds are device-specific.
- HR / rr_cv are physiologic descriptors included for completeness, not detection.
