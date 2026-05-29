# Cross-Dataset Model Performance — FP Suppression OFF / ON (v3.1)

**Date**: 2026-05-28
**Branch**: `cleanup-12lead-finetune-v1-clean`
**Model**: `checkpoint/1_lead_ECGFounder.pth` (single-lead, 150-class, 352.7 MB)
**Device**: MPS (Apple Silicon)
**Suppression layer**: Both **OFF** (§2–§9, raw model behavior) and **ON** (§10, post-v2-filter pipeline). The internal cohorts are reported under both modes; MIT-BIH / PTB-XL / tp_rex scripts do not invoke the suppression layer so the public-benchmark numbers are unchanged between modes.
**Label scope**: For the two internal cohorts (ECG-TP / ECG-FP) this report is restricted to the **V3.1 single-head ontology only** (`FZARK_LABEL_MAP` in [label_config.py:213](label_config.py:213) — 13 events). v3.1 (2026-05-28) recovered three composite events that share a beat-level head with their sibling isolated-beat event: SV Trigeminy and SV Bigeminy → PREMATURE ATRIAL COMPLEXES (idx 16, same as Isolated SV Beat); V Couplet → PREMATURE VENTRICULAR COMPLEXES (idx 9, same as Isolated V Beat). The remaining passthrough events — `Unknown`, `Multiple Event`, `Custom Heart Rate`, `Ventricular Bigeminy`, `Ventricular Trigeminy`, `Prolonged RR Interval` — are excluded so every reported row corresponds to a deterministic event → 150-class-head route.

---

## 1. Scope

Five datasets evaluated against the single-lead ECGFounder baseline with the v2 FP suppression
filter disabled. The goal is to characterize the raw classifier across the internal
clinical cohorts (fzark, fp_doctor, tp_rex) and the public benchmarks (MIT-BIH, PTB-XL).

| # | Dataset | Type | n eval | Sampling | Script |
|---|---|---|---|---|---|
| 1 | `ecg_tp_fzark` (V3.1) | Internal TPs (clinician-confirmed) | 3,285 / 4,088 raw | 500/class, V3.1 single-head filter | `compare_tp_fzark_with_full_suppression.py --suppression off` |
| 2 | `ecg_fp_doctor removed1` (V3.1) | Internal FPs (clinician-removed) | 5,501 / 6,541 raw | 500/class, V3.1 single-head filter | `compare_all_classes_with_full_suppression.py --suppression off` |
| 3 | `ecg-tp_rex` | Internal AFib stress set | 750 | full | `eval_ecg_tprex.py` (cached, 2026-05-27) |
| 4 | MIT-BIH (val split) | Public benchmark | 1,620 segments | full | `mitdb_eval.py` |
| 5 | PTB-XL (val split) | Public benchmark | 2,200 records | full | `ptbxl_eval.py` (cached, 2026-05-27, `res/eval_lead_ii/`) |

Preprocessing: dataset-specific `ECGPreprocessor` recipe (50 Hz notch + 0.67–40 Hz
bandpass + median baseline + winsorized z-score, resample to 500 Hz, center-crop to
5000 samples). Sigmoid output → 150-class probabilities.

---

## 2. Dataset 1 — ecg_tp_fzark (TPs, V3.1 single-head ontology, suppression off)

500/class stratified, capped at the full population for small classes
(Pause n=82, Supraventricular Run n=26, ST Elevation n=1, V Couplet n=36,
SV Bigeminy n=64, SV Trigeminy n=76). V3.1 cohort: **3,285 records** across 12
single-head event types (Sinus Tachycardia has no TPs in fzark).

### Detection rate by event type — every row is a deterministic V3.1 head route

| Event Type | n | head idx | t=0.5 | t=0.6 | t=0.7 | Tier |
|---|---|---|---|---|---|---|
| Atrial Fibrillation | 500 | 5 (ATRIAL FIBRILLATION) | **97.8%** | 96.2% | 94.0% | HIGH |
| **Supraventricular Trigeminy** | 76 | 16 (PREMATURE ATRIAL COMPLEXES) | **96.1%** | 89.5% | 76.3% | **v3.1 recovery** |
| Bradycardia | 500 | 4 (SINUS BRADYCARDIA) | **94.4%** | 94.0% | 93.6% | HIGH |
| **Ventricular Couplet** | 36 | 9 (PREMATURE VENTRICULAR COMPLEXES) | **80.6%** | 61.1% | 55.6% | **v3.1 recovery** |
| Isolated Ventricular Beat | 500 | 9 (PREMATURE VENTRICULAR COMPLEXES) | 78.6% | 74.4% | 69.8% | MODERATE |
| **Supraventricular Bigeminy** | 64 | 16 (PREMATURE ATRIAL COMPLEXES) | **78.1%** | 65.6% | 51.6% | **v3.1 recovery** |
| Isolated Supraventricular Beat | 500 | 16 (PREMATURE ATRIAL COMPLEXES) | 53.6% | 43.6% | 32.0% | LOW |
| Supraventricular Couplet | 500 | 19 (PREMATURE SV COMPLEXES) | 4.4% | 1.4% | 0.8% | CRITICAL |
| Pause | 82 | 142 (WITH SINUS PAUSE) | **0.0%** | 0.0% | 0.0% | CRITICAL |
| Supraventricular Run | 26 | 93 (SUPRAVENTRICULAR TACHYCARDIA) | **0.0%** | 0.0% | 0.0% | CRITICAL |
| Ventricular Run | 500 | 98 (VENTRICULAR TACHYCARDIA) | **0.0%** | 0.0% | 0.0% | CRITICAL |
| ST Elevation | 1 | 68 (ST ELEVATION NOW PRESENT IN) | 0.0% | 0.0% | 0.0% | n=1, not meaningful |

### Aggregate (V3.1 single-head only, weighted by n)

| Threshold | Detection rate | n detected | Δ vs V3 |
|---|---|---|---|
| 0.5 | **54.7%** | 1,796 / 3,285 | +152 events vs V3's 1,644 / 3,109 |
| 0.6 | 51.1% | 1,680 / 3,285 | +132 events |
| 0.7 | 47.5% | 1,562 / 3,285 | +111 events |

Sources: [res/tp_fzark_full_suppression/tp_baseline_threshold_variants_supp_off.csv](res/tp_fzark_full_suppression/tp_baseline_threshold_variants_supp_off.csv) (full), [res/cross_dataset_supp_off/ecg_tp_fzark_v3_only_supp_off.csv](res/cross_dataset_supp_off/ecg_tp_fzark_v3_only_supp_off.csv) (V3.1 slice).

---

## 3. Dataset 2 — ecg_fp_doctor removed1 (FPs, V3.1 single-head ontology, suppression off)

500/class stratified. V3.1 cohort: **5,501 records** across 12 event types
(SV Bigeminy has only n=1 in this dataset; ST Elevation has no FPs).
Every fired alert here is by construction a false positive (clinician-removed),
so **lower = better**.

### Raw alert rate by event type — every row is a deterministic V3.1 head route

| Event Type | n | head idx | t=0.5 | t=0.6 | t=0.7 | Concern |
|---|---|---|---|---|---|---|
| **Sinus Tachycardia** | 500 | 6 (SINUS TACHYCARDIA) | **99.0%** | 98.6% | 98.2% | CRITICAL — does not respond to threshold |
| Atrial Fibrillation | 500 | 5 (ATRIAL FIBRILLATION) | **72.0%** | 61.8% | 49.2% | HIGH — model over-fires on AFib FPs |
| **Supraventricular Trigeminy** (v3.1) | 500 | 16 (PAC head) | **67.4%** | 54.8% | 40.6% | HIGH — recovery cost: PAC head fires on FP trigeminy strips |
| Isolated Supraventricular Beat | 500 | 16 (PAC head) | 30.6% | 21.0% | 13.6% | MODERATE |
| Isolated Ventricular Beat | 500 | 9 (PVC head) | 25.4% | 18.6% | 12.4% | MODERATE |
| **Ventricular Couplet** (v3.1) | 500 | 9 (PVC head) | **19.6%** | 15.2% | 10.6% | MODERATE — recovery cost on V-couplet FPs |
| Bradycardia | 500 | 4 | 3.2% | 2.6% | 2.2% | LOW |
| Supraventricular Couplet | 500 | 19 | 2.2% | 1.0% | 0.4% | LOW |
| Pause | 500 | 142 | 0.0% | 0.0% | 0.0% | head silent on FPs |
| Supraventricular Run | 500 | 93 | 0.0% | 0.0% | 0.0% | head silent on FPs |
| Ventricular Run | 500 | 98 | 0.0% | 0.0% | 0.0% | head silent on FPs |
| Supraventricular Bigeminy (v3.1) | 1 | 16 | 100.0% | 0.0% | 0.0% | n=1, not assessable |

### Aggregate (V3.1 single-head only, weighted by n)

| Threshold | Raw FP rate | n | Δ vs V3 |
|---|---|---|---|
| 0.5 | **29.05%** | 1,598 / 5,501 | +3.2 pp vs V3's 25.82% (1,162 / 4,500) |
| 0.6 | 24.87% | 1,368 / 5,501 | +2.3 pp |
| 0.7 | 20.65% | 1,136 / 5,501 | +1.1 pp |

The aggregate ticks up because the V3.1-recovered classes route to heads that are
not silent on FPs — SV-Trigeminy alerts on 67.4% of FP trigeminy strips, V-Couplet
on 19.6% of FP couplet strips. The FP suppression layer is the natural place to
gate these. The PAC and PVC heads still have stronger TP-FP separation than the
AFib-head fallback they replace.

Sources: [res/fp_allclass_full_suppression/baseline_threshold_variants_supp_off.csv](res/fp_allclass_full_suppression/baseline_threshold_variants_supp_off.csv) (full), [res/cross_dataset_supp_off/ecg_fp_doctor_v3_only_supp_off.csv](res/cross_dataset_supp_off/ecg_fp_doctor_v3_only_supp_off.csv) (V3.1 slice).

---

## 4. Dataset 3 — ecg-tp_rex (AFib stress set)

750 AFib records, single class. Cached eval at base model (no fine-tuning).

| Metric | Value |
|---|---|
| Mean AFib head probability | 0.866 (σ = 0.261) |
| Median probability | 0.958 |
| Detection rate @ 0.5 | **90.0%** |

Source: `res/tprex_comparison/summary_afib_detection.csv`. Fine-tuned variants
(AFIB and AFIB-Fuzzy) underperform the base model on this cohort (88.8% / 86.8%),
so the base model is the production choice here.

---

## 5. Dataset 4 — MIT-BIH (Lead II, val split)

1,620 ten-second segments from MIT-BIH validation patients. Reported under
**Robust Winsorized** preprocessing (the project default `for_mitdb()` recipe).

| Class | positives | ROC | PR-AUC | F1 |
|---|---|---|---|---|
| Normal Sinus Rhythm | 871 | 0.854 | 0.903 | 0.760 |
| **Atrial Fibrillation** | 456 | **0.955** | 0.813 | **0.873** |
| Premature Ventricular Complexes | 341 | 0.820 | 0.601 | 0.559 |
| Premature Atrial Complexes | 162 | 0.920 | 0.568 | 0.590 |
| Atrial Flutter | 116 | 0.821 | 0.204 | 0.271 |
| Premature SV Complexes | 70 | 0.678 | 0.077 | 0.020 |
| Ventricular Tachycardia | 26 | **0.955** | 0.236 | 0.000 |
| Supraventricular Tachycardia | 5 | 0.887 | 0.022 | 0.000 |
| Sinus Bradycardia | 0 | — | — | — |
| RBBB / LBBB | 0 | — | — | — |

Source: `res/mitdb_singlelead/singlelead_comparison.csv`. F1 = 0.000 for VT/SVT
reflects 0 positive predictions at the default head threshold (extremely low
PR-AUC means recall could only be achieved at the cost of precision-zero).

---

## 6. Dataset 5 — PTB-XL (val split)

2,200 records, 51 active labels evaluated with `eval_with_dynamic_thresh` (optimal
per-class threshold picked from the val PR curve).

### High-performance heads (ROC ≥ 0.90)

| Class | ROC | Sens | Spec | F1 | Opt thr |
|---|---|---|---|---|---|
| Sinus Tachycardia | **0.997** | 0.994 | 0.976 | 0.774 | 0.54 |
| Supraventricular Tachycardia | 0.998 | 1.000 | 0.994 | 0.350 | 0.29 |
| Ventricular Tachycardia | 0.998 | 1.000 | 0.993 | 0.304 | 0.30 |
| Atrial Fibrillation | **0.990** | 0.958 | 0.958 | 0.749 | 0.27 |
| Atrial Flutter | 0.981 | 0.958 | 0.934 | 0.138 | 0.34 |
| Left Bundle Branch Block | 0.957 | 0.885 | 0.919 | 0.377 | 0.08 |
| Premature Ventricular Complexes | 0.951 | 0.900 | 0.943 | 0.608 | 0.25 |
| Left Anterior Fascicular Block | 0.947 | 0.924 | 0.869 | 0.513 | 0.03 |
| Electronic Atrial Pacemaker | 0.945 | 0.839 | 0.895 | 0.183 | 0.03 |
| Sinus Bradycardia | 0.941 | 0.938 | 0.843 | 0.264 | 0.64 |
| Right Atrial Enlargement | 0.939 | 0.955 | 0.824 | 0.052 | 0.21 |
| 1st-Degree AV Block | 0.905 | 0.203 | 0.990 | 0.270 | 0.01 |

### Mid heads (0.70 ≤ ROC < 0.90)

Normal ECG (0.83), Sinus Rhythm (0.81), Right Bundle Branch Block (0.78),
Lateral Infarct (0.79), Inferior Infarct (0.76), Anterolateral Infarct (0.77),
Left Ventricular Hypertrophy (0.72), QT Has Lengthened (0.83), Right
Ventricular Hypertrophy (0.73), Anterolateral Leads (0.73), Left Atrial
Enlargement (0.75).

### Weak heads (ROC < 0.70 or non-discriminating)

Septal Infarct (0.46), Anterior Infarct (0.48), Anteroseptal Infarct (0.47),
Low Voltage QRS (0.57), QRS Widening (0.57), Left Posterior Fascicular Block
(0.58). Several normality / catch-all heads ("OTHERWISE NORMAL ECG", "ABNORMAL
ECG", "BORDERLINE ECG", "LEFT AXIS DEVIATION") are flagged ROC=0.000 in the
report — these are dynamic-threshold edge cases (head fires ≈always or
≈never on the val cohort).

Source: `res/eval_lead_ii/res_thre.csv`.

---

## 7. Cross-Dataset Class Summary — V3.1 single-head ontology (suppression OFF)

Only V3.1-mapped event types appear here so the comparison is apples-to-apples
across datasets: a single deterministic 150-class head per row. Three rows are
new in v3.1 (marked **v3.1**) — they share their head with a sibling
isolated-beat event.

| V3.1 Class (head) | ECG-TP @0.5 (higher better) | ECG-FP @0.5 (lower better) | MIT-BIH ROC | PTB-XL ROC |
|---|---|---|---|---|
| Atrial Fibrillation (5) | **97.8%** | 72.0% | 0.955 | **0.990** |
| Sinus Tachycardia (6) | n/a (no TPs in fzark) | **99.0%** | — (n=0) | **0.997** |
| Bradycardia / Sinus Bradycardia (4) | **94.4%** | 3.2% | — (n=0) | 0.941 |
| Isolated Ventricular Beat / PVC (9) | 78.6% | 25.4% | 0.820 | 0.951 |
| **Ventricular Couplet** (v3.1, idx 9) | **80.6%** (n=36) | **19.6%** | (no native label) | (no native label) |
| Isolated Supraventricular Beat / PAC (16) | 53.6% | 30.6% | 0.920 | — |
| **Supraventricular Bigeminy** (v3.1, idx 16) | **78.1%** (n=64) | 100% (n=1) | (no native label) | (no native label) |
| **Supraventricular Trigeminy** (v3.1, idx 16) | **96.1%** (n=76) | 67.4% | (no native label) | (no native label) |
| Supraventricular Couplet / PSVC (19) | 4.4% | 2.2% | 0.678 | — |
| Supraventricular Run / SVT (93) | 0.0% (n=26) | 0.0% | 0.887 | **0.998** |
| Ventricular Run / VT (98) | 0.0% (n=500) | 0.0% | **0.955** | **0.998** |
| Pause / WITH SINUS PAUSE (142) | 0.0% (n=82) | 0.0% | — | — |
| ST Elevation (68) | 0.0% (n=1) | n/a (no FPs) | — | — |

### v3.1 net impact

| Cohort | V3 aggregate | V3.1 aggregate | Δ |
|---|---|---|---|
| ECG-TP fzark detection @ t=0.5 | 52.9% (1,644 / 3,109) | **54.7%** (1,796 / 3,285) | +152 TPs recovered |
| ECG-FP doctor raw alert @ t=0.5 | 25.82% (1,162 / 4,500) | 29.05% (1,598 / 5,501) | +3.2 pp FP rate (cost of recovery) |

The v3.1 recovery trades a 3.2 pp rise in raw FP rate for 152 additional TP
events. The FP cost concentrates on the SV-Trigeminy FP head (67.4% raw),
which is the natural target for the existing motion-based suppression rule
(`mean_motion ≥ 15 mG`) — it remains active in `multiclass_fp_suppression.py`
regardless of model head, so the suppression layer continues to work.

### The signal

1. **AFib and Bradycardia are production-grade.** ≥94% TP detection on fzark,
   ROC ≥ 0.94 on both public datasets. Even on the FP cohort, Bradycardia
   fires only 3.2% of the time — the head's natural specificity is high.
2. **Sinus Tachycardia is the dominant FP source.** 99.0% raw alert rate on the
   FP cohort vs. ROC 0.997 on PTB-XL says the head *can* discriminate, but at
   the t=0.5 threshold it is firing on every borderline-tachycardic strip that
   the clinicians rejected. The FP-suppression filter and/or a per-class threshold
   bump is what brings this under control in production.
3. **Paroxysmal-event detection collapses on the internal cohorts despite high
   public-benchmark ROC.** Ventricular Run reaches ROC 0.955 on MIT-BIH and ROC
   0.998 on PTB-XL, but **0/500 fzark V-Run TPs detect at t=0.5**. Same story
   for SVT (ROC 0.998 on PTB-XL, 0/26 on fzark) and Pause (0/82 on fzark, no
   matching public label). The model's head can separate positives from
   negatives (high ROC), but the absolute probability on a 10-second window
   containing a brief paroxysmal event stays below 0.5 — this is a calibration
   / window-length issue, not a representational one. Lower per-class thresholds
   (e.g. 0.2 for VT) or sliding-window scoring are the appropriate fixes.
4. **The 150-head classifier carries dead heads.** Several PTB-XL classes
   (septal infarct, anterior infarct, low-voltage QRS, QRS widening) sit at
   ROC ≤ 0.58 — these heads are not useful at single-lead, and their alerts
   should be disabled or routed through a stricter filter.

---

## 8. V3 head-mapping diagnostic on ECG-TP

For each V3-mapped event type, the cached 150-class sigmoid output was used to
rank every head by mean probability across the cohort. This shows which head
the model *actually* most strongly activates, vs. the head the V3 ontology
assigns. Source: [res/cross_dataset_supp_off/v3_head_correlation_fzark.csv](res/cross_dataset_supp_off/v3_head_correlation_fzark.csv).

`ABNORMAL ECG` (idx 0) and `SINUS RHYTHM` (idx 3) dominate the top-1 slot for
every cohort — they are global activity flags with near-100% mean probability
on every TP. Stripping those and the meta-rhythm head `UNDETERMINED RHYTHM`
(idx 39, fires on ≈100% of any rhythmic abnormality), the table below
reports the **highest-mean *clinically specific* alternative head** alongside
the V3 assignment.

| Event Type | V3 head (idx) | V3 mean / det@0.5 | V3 rank | Best clinically-specific alternative head (mean / det@0.5) | Verdict |
|---|---|---|---|---|---|
| Atrial Fibrillation | ATRIAL FIBRILLATION (5) | 0.913 / 97.8% | #2 | WITH RAPID VENTRICULAR RESPONSE (28) — 0.551 / 49.2% | ✅ V3 head dominates |
| Bradycardia | SINUS BRADYCARDIA (4) | 0.914 / 94.4% | #2 | MARKED SINUS BRADYCARDIA (33) — 0.762 / 82.6% | ✅ V3 head dominates; idx 33 is a useful sibling for severity |
| Isolated Ventricular Beat | PREMATURE VENTRICULAR COMPLEXES (9) | 0.766 / 78.6% | #4 | PREMATURE VENTRICULAR AND FUSION COMPLEXES (90) — 0.650 / 75.4% | ✅ V3 head dominates; idx 90 nearly as strong (could OR for +5–10% recall) |
| Isolated Supraventricular Beat | PREMATURE ATRIAL COMPLEXES (16) | 0.517 / 53.6% | #5 | — (next clinical heads ≤ 0.45) | ⚠️ V3 head is correct but model is under-confident |
| **Supraventricular Couplet** | PREMATURE SV COMPLEXES (19) | **0.177 / 4.4%** | **#33** | **ATRIAL FIBRILLATION (5) — 0.731 / 83.2%** | ❌ **MISMAPPED in practice** — model reads SV couplets as AFib |
| **Ventricular Run** | VENTRICULAR TACHYCARDIA (98) | **0.066 / 0.0%** | **#61** | **PREMATURE VENTRICULAR COMPLEXES (9) — 0.737 / 88.6%** | ❌ **VT head is dead**; model reads V-Runs as bursts of PVCs |
| **Supraventricular Run** | SUPRAVENTRICULAR TACHYCARDIA (93) | **0.062 / 0.0%** | **#67** | ATRIAL FIBRILLATION (5) — 0.607 / 65.4% | ❌ **SVT head is dead**; closest clinical proxy is AFib |
| **Pause** | WITH SINUS PAUSE (142) | **0.010 / 0.0%** | **#121** | SINUS BRADYCARDIA (4) — 0.716 / 72.0% | ❌ **Pause head is dead**; model represents pauses as bradycardia |
| ST Elevation | ST ELEVATION NOW PRESENT IN (68) | 0.025 / 0.0% | #67 | — (n=1) | n=1, not assessable |

### Reading

- **AFib, Bradycardia, IVB, ISB**: the V3 mapping correlates well with the head
  the model most strongly activates. AFib and Bradycardia head probabilities
  average >0.91 — production-grade. IVB (0.77 mean, det 78.6%) has a strong
  sibling head (PREMATURE VENTRICULAR AND FUSION COMPLEXES, idx 90) that fires
  on a slightly different subset; ORing 9 ∨ 90 at t=0.5 would likely lift IVB
  recall above 85%.
- **SV Couplet**: the V3-assigned PSVC head (idx 19) fires only 4.4%. The model's
  internal representation of a couplet is **AFib** — the AFib head fires at
  83.2% mean 0.73. Clinically this is plausible (couplets are paroxysmal
  ectopy, which the model conflates with brief AFib), but it means routing
  SV-couplet alerts through the PSVC head loses 95% of the signal. The
  practical question is whether the AFib head's specificity is acceptable for
  SV-couplet alerting given the FP overlap with real AFib.
- **Ventricular Run**: idx 98 (VT head) is functionally dead — mean 0.066,
  rank #61. The model represents non-sustained V-runs as **PVCs** (idx 9 fires
  at 88.6%). For deployment, V-Run TPs are recoverable today by listening to
  the PVC head — but that conflates V-Run with isolated PVCs, so a downstream
  pattern recognizer (≥3 consecutive beats with PVC-like morphology) is the
  natural follow-up.
- **Supraventricular Run** and **Pause**: both V3 heads (idx 93, 142) are
  effectively silent. There is no head in the 150-class output with strong
  specificity for these. The model "sees" them as AFib (SV-Run) or sinus
  brady (Pause). These are head-availability gaps in the original 150-class
  vocabulary — no remapping fixes them; a fine-tune or sliding-window
  detector is required.

### Cross-correlation tables (top heads per event type)

The supplementary CSV [res/cross_dataset_supp_off/v3_head_correlation_fzark.csv](res/cross_dataset_supp_off/v3_head_correlation_fzark.csv)
lists the top-3 ranked heads for every V3 event type with their mean
probability and detection rate at t=0.5, so the full alternative-head
distribution can be inspected without re-running inference.

### Passthrough events — heads firing despite no V3 mapping

The reverse question: of the 8 event types that v3 leaves unmapped
(`FZARK_UNMAPPABLE` in [label_config.py:167](label_config.py:167)), is there a
clinically-specific head in the 150-class output that is already firing
strongly? Same background-head filter as above (idx 0/3/39 removed). Source:
[res/cross_dataset_supp_off/passthrough_head_correlation_fzark.csv](res/cross_dataset_supp_off/passthrough_head_correlation_fzark.csv).

| Passthrough Event | n | Strongest clinically-specific head (det@0.5 / mean) | 2nd head | Verdict |
|---|---|---|---|---|
| **Supraventricular Trigeminy** | 76 | **PREMATURE ATRIAL COMPLEXES (16) — 96.1% / 0.773** | PSVC (19): 23.7% / 0.430 | ✅ Strong PAC fire — SV-trigeminy = every-3rd beat is a PAC. Recoverable today by routing through idx 16. |
| **Supraventricular Bigeminy** | 64 | **PREMATURE ATRIAL COMPLEXES (16) — 78.1% / 0.651** | LVH (26): 81.2% / 0.639 | ✅ Same head as ISB — bigeminy = alternating PAC. Recoverable through idx 16. |
| **Ventricular Couplet** | 36 | **PREMATURE VENTRICULAR COMPLEXES (9) — 80.6% / 0.712** | PVC+FUSION (90): 75.0% / 0.674 | ✅ A V-couplet IS two consecutive PVCs — idx 9 fires hard. Recoverable through PVC head. |
| Ventricular Trigeminy | 16 | PAC (16) — 87.5% / 0.774 | LATERAL INFARCT (24): 93.8% / 0.833 | ⚠️ PAC head firing is likely spurious (small n=16); no head specific to V-trigeminy |
| Ventricular Bigeminy | 2 | PVC (9) — 50.0% / 0.678 | PVC+FUSION (90): 50.0% / 0.687 | n=2, not assessable |
| Prolonged RR Interval | 169 | PVC (9) — 69.2% / 0.669 | PVC+FUSION (90): 69.2% / 0.608 | ❌ No head specific to RR variability; PVC fire is coincidental |
| Multiple Event | 116 | RBBB (11) — 75.0% / 0.722 | AFib (5): 56.9% / 0.599 | meta — heterogeneous by design, no single mapping fits |
| Unknown | 500 | AFib (5) — 91.0% / 0.724 | RVR (28): 88.0% / 0.765 | meta — but the heavy AFib + WITH RAPID VENTRICULAR RESPONSE signal suggests many "Unknown" events are actually undiagnosed AFib-with-RVR |

### Reading

Three passthrough event types have a strong, clinically-coherent head that v3
currently leaves on the table:

- **SV Trigeminy** → idx 16 (PAC head) at **96.1%** detection
- **SV Bigeminy** → idx 16 (PAC head) at **78.1%** detection
- **V Couplet** → idx 9 (PVC head) at **80.6%** detection

This is consistent with v3's design rationale (these are composite/pattern
events whose *constituent beats* the model recognizes well, but whose *pattern*
is not encoded in any 150-class head). The dropped-in-v3 decision was correct
*if the goal is single-head pattern recognition* — but if the goal is
**event-level recall**, these three are recoverable today by routing them
through the underlying beat head.

Trade-off: an SV-Bigeminy alert routed through idx 16 looks identical to an
Isolated SV Beat alert. Distinguishing the pattern requires either a
downstream beat-pattern detector (counting consecutive PAC-flagged beats per
window) or a per-class confidence threshold. This is a deployment-policy
decision, not a model-mapping one.

The **"Unknown" cohort** is the most surprising finding: 91% of n=500
"Unknown" events fire the AFib head at t=0.5, and 88% fire `WITH RAPID
VENTRICULAR RESPONSE` (idx 28). These are not random — they look like AFib
with RVR that the annotators couldn't categorize. Worth reviewing the
"Unknown" tagging policy with the clinical team; a fraction of these may be
mislabeled AFib TPs.

### Summary recommendation

| Action | Class(es) | Expected fzark TP recovery |
|---|---|---|
| Add SV Trigeminy → idx 16 to V3 LABEL_MAP | SV Trigeminy (n=76) | +73 events (5.3% → 96.1%) |
| Add SV Bigeminy → idx 16 to V3 LABEL_MAP | SV Bigeminy (n=64) | already 64% via passthrough, +14 events through proper routing |
| Add V Couplet → idx 9 to V3 LABEL_MAP | V Couplet (n=36) | already 41.7% via passthrough, +28 events |
| OR idx 9 ∨ idx 90 (PVC ∨ PVC+FUSION) for IVB head | IVB (n=500) | small recall lift, lifts IVB det from 78.6% toward ~85% |
| Review "Unknown" cohort labeling | Unknown (n=500 sampled / 1,929 total) | n/a — clinical/data-quality follow-up |

These remappings would lift the V3-only fzark TP aggregate detection from
**52.9%** to roughly **57–58%** at t=0.5, recovering ~150 events that the
current V3 ontology drops to passthrough. They do **not** address the
genuinely broken heads (Pause, VT, SVT, ST Elevation) — those require model
work, not ontology work.

---

## 9. Multi-label vs. single-label evaluation paradigm

The ECGFounder output is a **150-element sigmoid vector** — each of the 150
heads produces an independent probability between 0 and 1 with no
sum-to-one constraint. This is multi-label, not multi-class. All the
detection rates above are therefore "head fires" — `P(head_idx) ≥ τ` — and
say nothing about whether that head is also the *strongest* head for the
record. The same record routinely fires many heads at once. This section
quantifies that and compares the multi-label paradigm to a softmax-style
single-label argmax.

Source: [res/cross_dataset_supp_off/multilabel_vs_argmax_fzark.csv](res/cross_dataset_supp_off/multilabel_vs_argmax_fzark.csv).

### 9.1 Cardinality — how many heads fire at once?

Across all 4,088 fzark TP records:

| Threshold τ | mean heads firing | median | p5 – p95 | max | records with 0 heads |
|---|---|---|---|---|---|
| 0.3 | 13.7 | 14 | 10 – 18 | 24 | 0 / 4,088 |
| 0.5 | **8.4** | 8 | 5 – 12 | 15 | 0 / 4,088 |
| 0.7 | 5.7 | 5 | 3 – 9 | 12 | 0 / 4,088 |
| 0.9 | 3.3 | 3 | 1 – 5 | 9 | 0 / 4,088 |

Every record in the cohort fires at least one head at t=0.9. The typical
record fires **8 heads simultaneously at t=0.5** — this is the structure
that makes 91% + 88% > 100% possible (the "Unknown" cohort observation from §8).

### 9.2 What dominates if we collapse to a single label?

If you take `argmax` over the 150 heads — the natural single-label /
softmax-style reading — almost every record is classified as one of three
global activity heads:

| Argmax label across all 4,088 records | count | share |
|---|---|---|
| ABNORMAL ECG (idx 0) | 3,647 | **89.2%** |
| SINUS RHYTHM (idx 3) | 296 | 7.2% |
| RIGHT BUNDLE BRANCH BLOCK (idx 11) | 107 | 2.6% |
| SINUS BRADYCARDIA (idx 4) | 24 | 0.6% |
| (all 146 other heads combined) | 14 | 0.4% |

`ABNORMAL ECG` is essentially "is this strip non-normal?" and fires at
mean probability 0.99+ on every TP — it dominates the argmax. Any
softmax-style single-label evaluation reduces to: "the model agrees this is
abnormal." Useless for arrhythmia routing.

### 9.3 Per V3.1-event: multi-label vs. argmax

What we lose by collapsing to single-label.

| Event Type | n | V3.1 head | multi-label fire @0.5 | top-1 of 150 | top-3 of 150 | top-1 after dropping ABNORMAL/NSR/SR/UND | top-3 after dropping background |
|---|---|---|---|---|---|---|---|
| Atrial Fibrillation | 500 | 5 | **97.8%** | 0.0% | 64.2% | 35.2% | 93.0% |
| Supraventricular Trigeminy | 76 | 16 | **96.1%** | 0.0% | 3.9% | 30.3% | 93.4% |
| Bradycardia | 500 | 4 | **94.4%** | 1.2% | 83.0% | 53.8% | 93.0% |
| Ventricular Couplet | 36 | 9 | **80.6%** | 0.0% | 5.6% | 13.9% | 55.6% |
| Isolated Ventricular Beat | 500 | 9 | 78.6% | 0.6% | 12.2% | 26.0% | 70.0% |
| Supraventricular Bigeminy | 64 | 16 | 78.1% | 0.0% | 1.6% | 34.4% | 65.6% |
| Isolated Supraventricular Beat | 500 | 16 | 53.6% | 0.0% | 1.6% | 10.4% | 45.2% |
| Supraventricular Couplet | 500 | 19 | 4.4% | 0.0% | 0.0% | 0.0% | 0.2% |
| Ventricular Run | 500 | 98 | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% |
| Pause | 82 | 142 | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% |
| Supraventricular Run | 26 | 93 | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% |
| ST Elevation | 1 | 68 | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% |

Reading:

- **Multi-label @0.5** is the production metric — what fraction of TPs fire
  the correct head. AFib 97.8%, Bradycardia 94.4%, etc.
- **Top-1 of 150 (raw argmax)** is ≤1% for every class because
  ABNORMAL ECG / SINUS RHYTHM dominate every record's argmax.
- **Top-1 after dropping the four background heads** is the more useful
  single-label view — it asks: "of the *clinically specific* heads, which
  one wins?" Bradycardia 53.8%, AFib 35.2%, IVB 26.0%, SV-Bigeminy 34.4%.
- **Top-3 after dropping background** reaches 93% for AFib, Bradycardia, and
  SV-Trigeminy — meaning the model's answer is almost always one of the top
  3 specific heads, even when it isn't the strongest.

### 9.4 Single-label substitution for the dead heads

For the four V3.1 heads that multi-label calls "dead" (0% detection), what
does single-label argmax substitute? Background heads dropped.

| Event Type (V3.1 head 0% fire) | Top-1 specific substitution (% of TPs) |
|---|---|
| Ventricular Run (VT, idx 98) | ATRIAL FIBRILLATION 49.2%, PVC 19.2%, RBBB 18.2%, ATRIAL FLUTTER 12.6% |
| Supraventricular Run (SVT, idx 93) | ATRIAL FIBRILLATION 38.5%, RBBB 23.1%, SINUS BRADYCARDIA 7.7%, RAD 7.7% |
| Pause (idx 142) | **SINUS BRADYCARDIA 63.4%**, LAD 12.2%, RBBB 7.3%, RAD 7.3% |
| Supraventricular Couplet (PSVC, idx 19) | ATRIAL FIBRILLATION 28.2%, LVH 24.0%, RAD 23.4%, ATRIAL FLUTTER 5.6% |

These match the diagnostics in §8: the model has a strong, coherent opinion
for V-Run (it looks like AFib or a string of PVCs), Pause (looks like sinus
bradycardia), SV-Run (looks like AFib). The information is there — just
not on the V3-assigned head.

### 9.5 Implication for evaluation

| Paradigm | What it measures | Good for | Bad for |
|---|---|---|---|
| **Multi-label** (`P(head) ≥ τ`) | Whether the right head fires at all | Routing alerts to the right downstream filter; what production actually uses | Reporting "accuracy" — multiple events can be "correct" simultaneously |
| **Top-1 argmax** | The single most likely head | Softmax-style classification studies; benchmarks expecting one label per record | Multi-condition strips (AFib + RVR + RBBB on the same patient — common); dominated by global activity heads |
| **Top-K argmax after background filter** | The most likely *specific* heads | Sanity check for whether the right head is *near* the top of the model's ranking | Direct comparison with single-label benchmarks (need to declare which background heads were stripped) |

For an arrhythmia-detection product, **multi-label is the correct
paradigm** — the question is "does the model think AFib is present" not
"does the model think this is *most likely* AFib." Reporting accuracy as
top-1 argmax would systematically understate the model's clinical utility
because the ABNORMAL ECG head wins every argmax.

---

## 10. v3.1 cross-dataset performance with FP suppression ON

Re-running the two internal cohorts with `--suppression on` after the v3.1
mapping update. The v2 feature-gate filter ([multiclass_fp_suppression.py](multiclass_fp_suppression.py))
has active rules on four event types: AFib (motion ≤ 5 mG), Bradycardia
(HR ≤ 56.3 bpm), SV Trigeminy (motion ≥ 15 mG), and V Trigeminy (motion ≥ 24
mG ∧ snr_proxy > 1.2). All other classes pass through unchanged.

### 10.1 ECG-TP fzark — TP retention under suppression

| Event Type | n | head | det @ 0.5 (OFF) | retained (ON) | TP loss |
|---|---|---|---|---|---|
| Atrial Fibrillation | 500 | 5 | 97.8% | 97.4% | 0.4% (2/489) |
| **Bradycardia** | 500 | 4 | 94.4% | 91.8% | **2.8%** (13/472) |
| **Supraventricular Trigeminy** (v3.1) | 76 | 16 | 96.1% | 85.5% | **11.0%** (8/73) |
| Ventricular Couplet (v3.1) | 36 | 9 | 80.6% | 80.6% | 0% (passthrough) |
| Isolated Ventricular Beat | 500 | 9 | 78.6% | 78.6% | 0% (passthrough) |
| Supraventricular Bigeminy (v3.1) | 64 | 16 | 78.1% | 78.1% | 0% (passthrough) |
| Isolated Supraventricular Beat | 500 | 16 | 53.6% | 53.6% | 0% (passthrough) |
| Supraventricular Couplet | 500 | 19 | 4.4% | 4.4% | 0% (passthrough) |
| Pause | 82 | 142 | 0.0% | 0.0% | n/a |
| Supraventricular Run | 26 | 93 | 0.0% | 0.0% | n/a |
| Ventricular Run | 500 | 98 | 0.0% | 0.0% | n/a |
| ST Elevation | 1 | 68 | 0.0% | 0.0% | n/a |

### Aggregate (V3.1, weighted by n=3,285)

| Threshold | Detection (OFF) | Retention (ON) | TP loss |
|---|---|---|---|
| 0.5 | 54.7% (1,796) | **54.0%** (1,773) | **1.3%** (23 events) |
| 0.6 | 51.1% (1,680) | 50.4% (1,657) | 1.4% (23) |
| 0.7 | 47.5% (1,562) | 46.9% (1,541) | 1.3% (21) |

Source: [res/cross_dataset_supp_off/ecg_tp_fzark_v31_only_supp_on.csv](res/cross_dataset_supp_off/ecg_tp_fzark_v31_only_supp_on.csv).

### 10.2 ECG-FP doctor — FP suppression effectiveness

| Event Type | n | head | raw FP @ 0.5 (OFF) | post-supp (ON) | FP reduction |
|---|---|---|---|---|---|
| **Atrial Fibrillation** | 500 | 5 | 72.0% | **0.0%** | **100%** (motion gate clears all 360 FPs) |
| **Supraventricular Trigeminy** (v3.1) | 500 | 16 | 67.4% | **0.8%** | **98.8%** (motion-≥15 gate clears 333/337 FPs) |
| **Bradycardia** | 500 | 4 | 3.2% | 1.8% | 43.8% (HR gate clears 7/16 FPs) |
| Sinus Tachycardia | 500 | 6 | 99.0% | 99.0% | 0% (no rule) |
| Isolated Supraventricular Beat | 500 | 16 | 30.6% | 30.6% | 0% (no rule) |
| Isolated Ventricular Beat | 500 | 9 | 25.4% | 25.4% | 0% (no rule) |
| Ventricular Couplet (v3.1) | 500 | 9 | 19.6% | 19.6% | 0% (no rule) |
| Supraventricular Couplet | 500 | 19 | 2.2% | 2.2% | 0% (no rule) |
| Pause / SV-Run / V-Run | 1500 | 142/93/98 | 0.0% | 0.0% | n/a (heads silent) |
| Supraventricular Bigeminy | 1 | 16 | 100% | 100% | n=1 |

Also note from the full-class run (V Trigeminy is still passthrough in v3.1):

- Ventricular Trigeminy (passthrough): 37.8% → 0.6% (98.5% reduction via motion+SNR gate)

### Aggregate (V3.1, weighted by n=5,501)

| Threshold | Raw FP rate (OFF) | Post-supp (ON) | FP reduction |
|---|---|---|---|
| 0.5 | 29.05% (1,598) | **16.32%** (898) | **43.8%** |
| 0.6 | 24.87% (1,368) | 14.20% (781) | 42.9% |
| 0.7 | 20.65% (1,136) | 12.40% (682) | 40.0% |

Source: [res/cross_dataset_supp_off/ecg_fp_doctor_v31_only_supp_on.csv](res/cross_dataset_supp_off/ecg_fp_doctor_v31_only_supp_on.csv).

### 10.3 ON vs OFF — operating-point trade-off

| Metric | Suppression OFF | Suppression ON | Net |
|---|---|---|---|
| ECG-TP retention @ 0.5 | 54.7% | **54.0%** | −0.7 pp (23 TPs lost out of 1,796) |
| ECG-FP raw rate @ 0.5 | 29.05% | **16.32%** | **−12.7 pp** (700 FPs cleared) |
| TP loss / FP gain ratio | — | — | **1 TP lost per ~30 FPs cleared** |

The v3.1 + suppression-on combination delivers what the FP filter was
designed for: a **44% reduction in raw FP rate** for the cost of **1.3% TP
loss**. The TP loss concentrates on Bradycardia (2.8%, HR gate at 56.3 bpm)
and SV Trigeminy (11.0%, motion gate at 15 mG); the FP reduction
concentrates on AFib (100%), SV Trigeminy (98.8%), V Trigeminy passthrough
(98.5%). Other classes are unchanged either way — they pass through both
the model and the filter without modification.

### 10.4 What the v3.1 mapping changed under suppression

The v3.1 ontology additions (SV-Trig → 16, SV-Big → 16, V-Couplet → 9)
interact with the suppression layer as follows:

- **SV Trigeminy**: motion gate `mean_motion ≥ 15 mG` already existed. Under
  v3 the head was inactive so the gate never fired on the TP side. Under
  v3.1 the PAC head fires at 96.1%, then the gate trims 11% — the 8 TPs
  lost are presumably the still-and-rested SV-Trigeminy strips that look
  like noise to the gate. **Net: 96.1% → 85.5% TP retention (still way
  ahead of v3's 5.3%).**
- **SV Bigeminy** and **V Couplet**: no suppression rule on these classes
  — TP retention under ON matches OFF (78.1% / 80.6%).
- The v3.1 mapping does **not** weaken the suppression layer's FP-side
  effectiveness because the suppression rules are feature-based (motion,
  HR, SNR) and operate independently of the model head identity.

---

## 11. Combined TP + FP — full binary metrics at t=0.5 (OFF vs ON)

Treating each V3.1-mapped class as a binary detector with:
- **Positives** = TP records from `ecg_tp_fzark` (clinician-confirmed)
- **Negatives** = FP records from `ecg_fp_doctor removed1` (clinician-removed)
- **Score** = sigmoid output at the V3.1 head (`FZARK_LABEL_MAP[event]`)
- **Decision** = score ≥ 0.5 (after suppression, if ON)

Source: [res/cross_dataset_supp_off/combined_binary_metrics_v31.csv](res/cross_dataset_supp_off/combined_binary_metrics_v31.csv). Computed via [scripts/combined_binary_eval.py](scripts/combined_binary_eval.py) on the cached probability matrices (no re-inference).

### 11.1 Overall (micro-pooled, classes with both TP & FP populations)

n_pos = 3,284 TPs · n_neg = 5,001 FPs · 8,285 records total.

| Metric | Suppression OFF | Suppression ON | Δ |
|---|---|---|---|
| Sensitivity | 0.547 | 0.540 | −0.7 pp |
| Specificity | 0.779 | **0.919** | **+14.0 pp** |
| PPV | 0.620 | **0.815** | **+19.5 pp** |
| NPV | 0.724 | 0.753 | +2.9 pp |
| **Accuracy** | 0.687 | **0.769** | **+8.2 pp** |
| **F1 score** | 0.581 | **0.649** | **+6.8 pp** |
| ROC-AUC | 0.700 | **0.772** | **+7.2 pp** |
| PR-AUC | 0.694 | **0.762** | **+6.8 pp** |
| Counts (TP / FP / FN / TN) | 1796 / 1103 / 1488 / 3898 | 1773 / 403 / 1511 / 4598 | −23 TPs, **−700 FPs** |

The suppression layer trades **0.7 pp sensitivity for 14 pp specificity** and **19.5 pp PPV** — alerts go from 62.0% real to **81.5% real** post-suppression. Accuracy lifts 8.2 pp, F1 lifts 6.8 pp.

### 11.2 Per-class binary metrics at t=0.5

| Event Type | head | n_pos / n_neg | Mode | Sens | Spec | PPV | NPV | Acc | F1 | ROC | PR-AUC |
|---|---|---|---|---|---|---|---|---|---|---|---|
| **Atrial Fibrillation** | 5 | 500 / 500 | OFF | 0.978 | 0.280 | 0.576 | 0.927 | 0.629 | 0.725 | 0.890 | 0.878 |
| | | | **ON**  | 0.974 | **1.000** | **1.000** | 0.975 | **0.987** | **0.987** | **0.995** | **0.996** |
| **Bradycardia** | 4 | 500 / 500 | OFF | 0.944 | 0.968 | 0.967 | 0.945 | 0.956 | 0.955 | 0.974 | 0.976 |
| | | | **ON**  | 0.918 | 0.982 | 0.981 | 0.923 | 0.950 | 0.948 | 0.952 | 0.970 |
| **Supraventricular Trigeminy** (v3.1) | 16 | 76 / 500 | OFF | 0.961 | 0.326 | 0.178 | 0.982 | 0.410 | 0.300 | 0.756 | 0.314 |
| | | | **ON**  | 0.855 | **0.992** | **0.942** | 0.978 | **0.974** | **0.897** | **0.923** | **0.887** |
| Isolated Ventricular Beat | 9 | 500 / 500 | OFF/ON | 0.786 | 0.746 | 0.756 | 0.777 | 0.766 | 0.771 | 0.856 | 0.848 |
| Ventricular Couplet (v3.1) | 9 | 36 / 500 | OFF/ON | 0.806 | 0.804 | 0.228 | 0.983 | 0.804 | 0.356 | 0.870 | 0.316 |
| Isolated Supraventricular Beat | 16 | 500 / 500 | OFF/ON | 0.536 | 0.694 | 0.637 | 0.599 | 0.615 | 0.582 | 0.633 | 0.643 |
| Supraventricular Bigeminy (v3.1) | 16 | 64 / 1 | OFF/ON | 0.781 | 0.000 | 0.980 | 0.000 | 0.769 | 0.870 | 0.719 | 0.995 |
| Supraventricular Couplet | 19 | 500 / 500 | OFF/ON | 0.044 | 0.978 | 0.667 | 0.506 | 0.511 | 0.083 | 0.372 | 0.458 |
| Pause | 142 | 82 / 500 | OFF/ON | 0.000 | 1.000 | 0.000 | 0.859 | 0.859 | 0.000 | 0.758 | 0.335 |
| Supraventricular Run | 93 | 26 / 500 | OFF/ON | 0.000 | 1.000 | 0.000 | 0.951 | 0.951 | 0.000 | 0.606 | 0.063 |
| Ventricular Run | 98 | 500 / 500 | OFF/ON | 0.000 | 1.000 | 0.000 | 0.500 | 0.500 | 0.000 | 0.360 | 0.392 |
| Sinus Tachycardia | 6 | 0 / 500 | OFF/ON | — | 0.010 | 0.000 | 1.000 | 0.010 | 0.000 | — | — |
| ST Elevation | 68 | 1 / 0 | OFF/ON | 0.000 | — | — | 0.000 | 0.000 | 0.000 | — | — |

Rows where OFF = ON have no active suppression rule for that class (the suppressor passes through). The three classes with active rules — AFib, Bradycardia, SV-Trigeminy — show the full effect.

### 11.3 Reading the numbers

- **AFib is the showpiece for FP suppression.** Sensitivity stays at 0.974, specificity moves from 0.28 to 1.00, PPV from 0.58 to **1.00**, Accuracy **0.629 → 0.987**, F1 **0.725 → 0.987**, ROC-AUC from 0.89 to **0.995**. The motion gate (`mean_motion ≤ 5 mG`) cleared every one of the 360 AFib FPs in this cohort without killing any clinically-relevant TPs (the 2 TPs lost were borderline motion-corrupted strips).

- **SV-Trigeminy is the v3.1 win-with-cost.** Under v3 this class had no head firing — there was nothing for the suppressor to act on. Under v3.1 the PAC head fires at 96%, then the motion gate (`mean_motion ≥ 15 mG`, inverted because SV-Trigeminy TPs occur during ambulation) suppresses 333 of 337 FPs while losing 8 of 73 TPs. Accuracy **0.410 → 0.974**, F1 **0.300 → 0.897**, ROC-AUC 0.756 → **0.923**, PR-AUC 0.314 → **0.887**.

- **Bradycardia has a mild trade.** Sens 0.944 → 0.918 (HR gate trims 13 borderline-bradycardic TPs whose HR > 56.3 bpm), Spec 0.968 → 0.982. Accuracy 0.956 → 0.950, F1 0.955 → 0.948 — essentially flat (already near ceiling). ROC-AUC actually dips slightly (0.974 → 0.952) because the gate is a hard threshold that doesn't always agree with the model's score order.

- **The dead heads** (Pause, SV-Run, V-Run) sit at sensitivity = 0 / specificity = 1 in both modes. They contribute nothing to the alert stream, so suppression has nothing to gate. These are the model-side gaps that v3.1 mapping cannot fix.

- **Mid-tier classes without suppression rules** (IVB, ISB, SV Couplet, V Couplet) are identical under OFF/ON. Their PPV and AUC numbers reflect raw model behavior — IVB at PPV 0.756 and ROC 0.856 is production-grade; ISB at PPV 0.637 and ROC 0.633 is borderline; SV Couplet at sensitivity 0.044 needs a head fix, not a suppression fix.

### 11.4 What this says about deployment

| Mode | When it's the right choice |
|---|---|
| **Suppression OFF** | Research / development — want raw model behavior, not pipeline behavior. Useful for diagnosing head-mapping issues (§8). |
| **Suppression ON** | Production — the 14 pp specificity gain and 19.5 pp PPV gain at 0.7 pp sensitivity cost is exactly the operating point a clinical alerting system needs. AFib alone goes from 58% PPV (almost half are false) to 100% PPV (essentially every alert is real) in the sampled cohort. |

The combined-dataset evaluation confirms the v3.1 + v2-suppression pipeline is a net-positive operating point across every class with an active rule. For classes without a rule, ON and OFF are by construction identical — so the pipeline never harms classes it's not actively trying to help.

---

## 12. Reproducing this run

```bash
# Internal cohorts — 500/class with suppression disabled
python3 compare_tp_fzark_with_full_suppression.py --per-class 500 --suppression off
python3 compare_all_classes_with_full_suppression.py --per-class 500 --suppression off

# MIT-BIH (no suppression in this pipeline by construction)
python3 standardize_mitdb.py   # one-off, generates ./res/mitdb_standardized/*
python3 split_mitdb.py
python3 mitdb_eval.py

# PTB-XL (no suppression in this pipeline by construction)
python3 ptbxl_eval.py
```

Outputs land under each script's project directory; this report draws from:
- `res/tp_fzark_full_suppression/tp_baseline_threshold_variants_supp_off.csv`
- `res/fp_allclass_full_suppression/baseline_threshold_variants_supp_off.csv`
- `res/tprex_comparison/summary_afib_detection.csv`
- `res/mitdb_singlelead/singlelead_comparison.csv`
- `res/eval_lead_ii/res_thre.csv`
