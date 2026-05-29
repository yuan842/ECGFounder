# Per-Header SQI & Motion Characterization — MOVE

**Date**: 2026-05-29
**Scope**: MOVE chest ECG-gel + chest accelerometer, 5,158 non-padded windows (all rhythm-negative).
**Source**: [scripts/move/characterize_heads.py](../../scripts/move/characterize_heads.py); model = base backbone (= DualHead on these heads).
**Motion** = gravity-removed `mean_motion_mg` (production-comparable). Bins: rest <0.5 (n=1470), low .5–1 (1102), mod 1–2 (1601), high 2–5 (897), extreme >5 (88).

Every head-fire on this cohort is a **false positive** (healthy subjects, no labelled arrhythmia). Exception: rate heads (Sin Tach / VT / SVT) during exercise reflect genuine physiology — see §caveat.

---

## A. How the SQI panel itself behaves vs motion (dataset-wide)

| motion bin | snr_proxy | flat_pct | clip_pct | hf_noise | kurt | pct_phys_HR | sqi_score_ecg |
|---|---|---|---|---|---|---|---|
| rest <0.5 | 0.00 | **99.9** | 0.9 | 0.000 | 0.77 | 43.1 | 0.26 |
| low .5–1 | 1.17 | 7.4 | 8.8 | 0.016 | 10.8 | 98.7 | 0.71 |
| mod 1–2 | 1.30 | 3.7 | 6.3 | 0.029 | 11.0 | 98.7 | 0.69 |
| high 2–5 | 1.04 | 1.3 | 5.2 | 0.024 | 5.5 | 98.9 | 0.66 |
| extreme >5 | 0.22 | 3.7 | **29.1** | 0.013 | 4.0 | 96.8 | 0.52 |

**Non-monotonic, U-shaped quality vs motion** — the central insight:
- **At rest (<0.5 mG): SQI is *lowest*** (sqi 0.26, flat 99.9 %, snr 0). The **gel electrode** (`ecg:gel` — the analyzed channel) is near-flat / poor contact when the subject is still — a real wearable failure mode, not noise. (Notable: this is the clinical-grade *gel* electrode, not the dry one.)
- **Light–moderate motion (0.5–2 mG): SQI is *highest*** (sqi 0.69–0.71). Some movement seats the electrode and produces a live, detectable QRS.
- **Extreme motion (>5 mG): SQI degrades** the expected way (clip 29 %, sqi 0.52).

So for this chest gel electrode (`ecg:gel`), *quality is best at mild activity, worst at dead rest and at violent motion* — neither end of the motion axis is "clean."

---

## B. Per-header summary

| idx | head | FP % | motion(fire) | motion(¬fire) | corr(P,motion) | corr(P,snr) | corr(P,flat) |
|---|---|---|---|---|---|---|---|
| 4 | Bradycardia | 0.29 | 2.12 | 1.37 | −0.36 | −0.54 | **+0.85** |
| 5 | Atrial Fibrillation | 13.3 | 2.28 | 1.23 | +0.34 | −0.14 | −0.27 |
| 6 | Sinus Tachycardia | 57.0 | 1.91 | 0.66 | +0.47 | +0.45 | −0.64 |
| 9 | IVB / V-Couplet | 7.3 | 2.18 | 1.31 | +0.26 | −0.37 | +0.01 |
| 16 | ISB / SV-Trig / SV-Big | 0.31 | 0.88 | 1.37 | −0.13 | −0.23 | +0.23 |
| 19 | SV-Couplet | 0.17 | 0.75 | 1.37 | −0.20 | −0.55 | +0.64 |
| 68 | ST Elevation | 0.00 | — | 1.37 | — | — | — |
| 93 | SV Run (SVT) | 35.6 | 1.90 | 1.08 | +0.40 | +0.58 | −0.59 |
| 98 | Ventricular Run (VT) | 38.9 | 2.03 | 0.96 | +0.47 | +0.53 | −0.61 |
| 142 | Pause | 0.00 | — | 1.37 | — | — | — |

---

## C. FP rate (%) by motion bin — the "against motion" core

| head | rest <0.5 | low .5–1 | mod 1–2 | high 2–5 | extreme >5 |
|---|---|---|---|---|---|
| 4 Bradycardia | 0.0 | 0.5 | 0.4 | 0.2 | 1.1 |
| **5 Atrial Fibrillation** | **0.0** | 14.2 | 17.6 | 20.0 | **77.3** |
| 6 Sinus Tachycardia | 0.4 | 49.1 | 91.4 | 95.1 | 85.2 |
| 9 IVB / V-Couplet | 0.5 | 9.4 | 8.6 | 11.8 | 22.7 |
| 16 ISB / SV-Trig | 0.0 | 1.4 | 0.0 | 0.1 | 0.0 |
| 19 SV-Couplet | 0.0 | 0.8 | 0.0 | 0.0 | 0.0 |
| **93 SV Run (SVT)** | **0.0** | 13.1 | 66.5 | 69.3 | 6.8 |
| **98 Ventricular Run (VT)** | **0.0** | 13.1 | 69.7 | 79.5 | 37.5 |

**Two head families:**
- **Motion-driven FP heads** (AFib 5, VT 98, SVT 93, Sin Tach 6, IVB 9): FP rate climbs monotonically with motion — **0 % at rest**, rising to 20–95 % at high motion. These are exactly the heads a motion gate should suppress. AFib is the cleanest: 0 % at rest → 77 % at extreme motion.
- **Motion-immune heads** (Bradycardia 4, ISB/SV-Trig 16, SV-Couplet 19, ST Elevation 68, Pause 142): FP stays < 1.5 % across all motion bins. A motion gate adds nothing here (and the SV-Trig inverted gate is actively wrong on this device — see motion-gate report).

---

## D. SQI of FIRING vs non-firing windows (`fire | ¬fire`)

| head | snr_proxy | flat_pct | clip_pct | sqi_ecg |
|---|---|---|---|---|
| 5 AFib | 0.49 \| 0.89 | 5.5 \| 35.5 | **19.3 \| 3.4** | 0.58 \| 0.56 |
| 6 Sin Tach | 1.25 \| 0.28 | 2.1 \| 70.4 | 5.7 \| 5.1 | 0.69 \| 0.40 |
| 9 IVB | 0.34 \| 0.88 | 17.2 \| 32.6 | **26.8 \| 3.8** | 0.52 \| 0.57 |
| 93 SV Run | 1.51 \| 0.46 | 0.6 \| 48.6 | 1.7 \| 7.6 | 0.73 \| 0.48 |
| 98 V Run | 1.42 \| 0.47 | 0.7 \| 51.1 | 2.7 \| 7.2 | 0.71 \| 0.47 |

**Signature of a motion-artifact false positive:** AFib and IVB fire on windows with **6–7× higher clip_pct** (19–27 % vs 3–4 %) — i.e. the false alerts ride on saturated/clipped motion spikes. `clip_pct` is the single most discriminating SQI for these FPs (more so than snr or the composite).

The rate heads (Sin Tach, VT, SV Run) fire on *higher-snr, lower-flat* windows — because those windows are during exercise (live signal, elevated HR), so their "FP" is partly true physiology, and SQI looks *good* there.

---

## Key takeaways

1. **Quality vs motion is U-shaped** — worst at dead rest (flat gel-electrode contact) AND at extreme motion; best at mild activity. A single "low-motion = clean" assumption is wrong for this device.
2. **Heads split into motion-driven vs motion-immune.** Motion gates should target {AFib, VT, SVT, IVB}; they are pointless or harmful on {Bradycardia, ISB/SV-Trig, SV-Couplet, Pause, ST}.
3. **`clip_pct` is the best FP discriminator** for the morphology heads (AFib/IVB fire at 6–7× baseline clip) — a clip-based SQI gate would complement the motion gate.
4. **Rate-head "FP" during exercise is largely physiology** (high SNR, high HR), so motion/clip gating must not blindly suppress Sin Tach/VT/SVT during activity without HR context.

## Artifacts
- [res/move_eval/move_head_characterization.csv](move_head_characterization.csv) — §B table
- [res/move_eval/move_head_fp_by_motion.csv](move_head_fp_by_motion.csv) — §C grid
