# Label Reclassification Plan: Multi-Dataset ↔ ECGFounder 150-class

**Date**: 2026-05-27
**Model**: ECGFounder Net1D (`./checkpoint/1_lead_ECGFounder.pth`, n_classes=150)
**Label source**: `tasks.txt` (0-indexed, 150 lines)

**Datasets covered**:
- **ecg-tp-fzark** (37,288 records, 17 event types) — clinician-confirmed true positives
- **ecg-fp-doctor removed1** (102,634 records, 17 event types) — clinician-removed false positives
- **PTB-XL** (21,799 records, 31 active classes) — standard 12-lead diagnostic benchmark
- **MIT-BIH** (8,640 segments, 6 target classes) — arrhythmia beat annotation benchmark

---

## 1. Problem Statement

Three separate label mapping systems exist across the codebase, each with different conventions and issues:

1. **ecg-tp-fzark / ecg-fp LABEL_MAP** — used in 11 scripts, has **5 off-by-one index errors** and **8 unmapped event types**
2. **PTB-XL label vectors** — pre-computed 150-element binary vectors in `csv/ptbxl_label.csv`, maps **31 of 150 classes** via diagnostic text matching against `tasks.txt`
3. **MIT-BIH segment labels** — constructed at runtime from WFDB beat annotations, maps **6 target classes** via hardcoded symbol-to-index dictionaries

There is no shared label configuration module. Each dataset uses its own mapping logic, making it easy for errors to diverge undetected.

---

## 2. ECGFounder 150-class Label Reference

Complete index-to-label mapping from `tasks.txt` (0-based indexing):

| Index | ECGFounder Class Label |
|---|---|
| 0 | ABNORMAL ECG |
| 1 | NORMAL SINUS RHYTHM |
| 2 | NORMAL ECG |
| 3 | SINUS RHYTHM |
| 4 | SINUS BRADYCARDIA |
| 5 | ATRIAL FIBRILLATION |
| 6 | SINUS TACHYCARDIA |
| 7 | otherwise normal ecg |
| 8 | LEFT AXIS DEVIATION |
| 9 | PREMATURE VENTRICULAR COMPLEXES |
| 10 | BORDERLINE ECG |
| 11 | RIGHT BUNDLE BRANCH BLOCK |
| 12 | SEPTAL INFARCT |
| 13 | LEFT ATRIAL ENLARGEMENT |
| 14 | NONSPECIFIC T WAVE ABNORMALITY |
| 15 | LOW VOLTAGE QRS |
| 16 | PREMATURE ATRIAL COMPLEXES |
| 17 | ANTERIOR INFARCT |
| 18 | INCOMPLETE RIGHT BUNDLE BRANCH BLOCK |
| 19 | PREMATURE SUPRAVENTRICULAR COMPLEXES |
| 20 | LEFT BUNDLE BRANCH BLOCK |
| 21 | NONSPECIFIC T WAVE ABNORMALITY NOW EVIDENT IN |
| 22 | NONSPECIFIC T WAVE ABNORMALITY NO LONGER EVIDENT IN |
| 23 | T WAVE INVERSION NOW EVIDENT IN |
| 24 | LATERAL INFARCT |
| 25 | NONSPECIFIC ST ABNORMALITY |
| 26 | LEFT VENTRICULAR HYPERTROPHY |
| 27 | T WAVE INVERSION NO LONGER EVIDENT IN |
| 28 | WITH RAPID VENTRICULAR RESPONSE |
| 29 | QT HAS SHORTENED |
| 30 | QT HAS LENGTHENED |
| 31 | FUSION COMPLEXES |
| 32 | ATRIAL FLUTTER |
| 33 | MARKED SINUS BRADYCARDIA |
| 34 | WITH SINUS ARRHYTHMIA |
| 35 | NONSPECIFIC ST AND T WAVE ABNORMALITY |
| 36 | LEFT ANTERIOR FASCICULAR BLOCK |
| 37 | RIGHT AXIS DEVIATION |
| 38 | ECTOPIC ATRIAL RHYTHM |
| 39 | UNDETERMINED RHYTHM |
| 40 | ANTEROSEPTAL INFARCT |
| 41 | RIGHTWARD AXIS |
| 42 | ST NOW DEPRESSED IN |
| 43 | WITH SHORT PR |
| 44 | WITH MARKED SINUS ARRHYTHMIA |
| 45 | ST NO LONGER DEPRESSED IN |
| 46 | INVERTED T WAVES HAVE REPLACED NONSPECIFIC T WAVE ABNORMALITY IN |
| 47 | NON-SPECIFIC CHANGE IN ST SEGMENT IN |
| 48 | NONSPECIFIC T WAVE ABNORMALITY HAS REPLACED INVERTED T WAVES IN |
| 49 | JUNCTIONAL RHYTHM |
| 50 | ELECTRONIC ATRIAL PACEMAKER |
| 51 | ABERRANT CONDUCTION |
| 52 | ELECTRONIC VENTRICULAR PACEMAKER |
| 53 | T WAVE INVERSION LESS EVIDENT IN |
| 54 | ANTEROLATERAL INFARCT |
| 55 | WITH REPOLARIZATION ABNORMALITY |
| 56 | RSR' OR QR PATTERN IN V1 SUGGESTS RIGHT VENTRICULAR CONDUCTION DELAY |
| 57 | T WAVE INVERSION MORE EVIDENT IN |
| 58 | WIDE QRS RHYTHM |
| 59 | WITH PREMATURE VENTRICULAR OR ABERRANTLY CONDUCTED COMPLEXES |
| 60 | RIGHT ATRIAL ENLARGEMENT |
| 61 | INFERIOR INFARCT |
| 62 | INCOMPLETE LEFT BUNDLE BRANCH BLOCK |
| 63 | VOLTAGE CRITERIA FOR LEFT VENTRICULAR HYPERTROPHY |
| 64 | OR DIGITALIS EFFECT |
| 65 | BIFASCICULAR BLOCK |
| 66 | ST NO LONGER ELEVATED IN |
| 67 | WITH SLOW VENTRICULAR RESPONSE |
| 68 | ST ELEVATION NOW PRESENT IN |
| 69 | PREMATURE ECTOPIC COMPLEXES |
| 70 | LEFT POSTERIOR FASCICULAR BLOCK |
| 71 | T WAVE AMPLITUDE HAS DECREASED IN |
| 72 | WITH A COMPETING JUNCTIONAL PACEMAKER |
| 73 | RIGHT SUPERIOR AXIS DEVIATION |
| 74 | BIATRIAL ENLARGEMENT |
| 75 | VENTRICULAR-PACED RHYTHM |
| 76 | ATRIAL-PACED RHYTHM |
| 77 | T WAVE AMPLITUDE HAS INCREASED IN |
| 78 | WITH QRS WIDENING |
| 79 | WITH 1ST DEGREE AV BLOCK |
| 80 | PROLONGED QT |
| 81 | WITH PROLONGED AV CONDUCTION |
| 82 | RIGHT VENTRICULAR HYPERTROPHY |
| 83 | WITH QRS WIDENING AND REPOLARIZATION ABNORMALITY |
| 84 | ATRIAL-SENSED VENTRICULAR-PACED RHYTHM |
| 85 | AV SEQUENTIAL OR DUAL CHAMBER ELECTRONIC PACEMAKER |
| 86 | PULMONARY DISEASE PATTERN |
| 87 | ACUTE MI / STEMI |
| 88 | INFERIOR-POSTERIOR INFARCT |
| 89 | NONSPECIFIC INTRAVENTRICULAR CONDUCTION DELAY |
| 90 | PREMATURE VENTRICULAR AND FUSION COMPLEXES |
| 91 | IN A PATTERN OF BIGEMINY |
| 92 | AV DUAL-PACED RHYTHM |
| 93 | SUPRAVENTRICULAR TACHYCARDIA |
| 94 | VENTRICULAR-PACED COMPLEXES |
| 95 | WIDE QRS TACHYCARDIA |
| 96 | RSR' PATTERN IN V1 |
| 97 | ST LESS DEPRESSED IN |
| 98 | VENTRICULAR TACHYCARDIA |
| 99 | EARLY REPOLARIZATION |
| 100 | ST MORE DEPRESSED IN |
| 101 | ANTEROLATERAL LEADS |
| 102 | ELECTRONIC DEMAND PACING |
| 103 | RBBB AND LEFT ANTERIOR FASCICULAR BLOCK |
| 104 | LATERAL INJURY PATTERN |
| 105 | BIVENTRICULAR PACEMAKER DETECTED |
| 106 | SUSPECT UNSPECIFIED PACEMAKER FAILURE |
| 107 | WOLFF-PARKINSON-WHITE |
| 108 | WITH VENTRICULAR ESCAPE COMPLEXES |
| 109 | INFERIOR INJURY PATTERN |
| 110 | CONSIDER RIGHT VENTRICULAR INVOLVEMENT IN ACUTE INFERIOR INFARCT |
| 111 | ST ELEVATION HAS REPLACED ST DEPRESSION IN |
| 112 | NONSPECIFIC INTRAVENTRICULAR BLOCK |
| 113 | MASKED BY FASCICULAR BLOCK |
| 114 | PEDIATRIC ECG ANALYSIS |
| 115 | BLOCKED |
| 116 | WITH UNDETERMINED RHYTHM IRREGULARITY |
| 117 | LEFTWARD AXIS |
| 118 | WITH 2ND DEGREE SA BLOCK MOBITZ I |
| 119 | ACUTE |
| 120 | ABNORMAL LEFT AXIS DEVIATION |
| 121 | WITH COMPLETE HEART BLOCK |
| 122 | NO P-WAVES FOUND |
| 123 | ST LESS ELEVATED IN |
| 124 | WITH RETROGRADE CONDUCTION |
| 125 | ST MORE ELEVATED IN |
| 126 | JUNCTIONAL BRADYCARDIA |
| 127 | WITH VARIABLE AV BLOCK |
| 128 | ANTERIOR INJURY PATTERN |
| 129 | WITH JUNCTIONAL ESCAPE COMPLEXES |
| 130 | ACUTE MI |
| 131 | ACUTE PERICARDITIS |
| 132 | POSTERIOR INFARCT |
| 133 | IDIOVENTRICULAR RHYTHM |
| 134 | WITH 2ND DEGREE SA BLOCK MOBITZ II |
| 135 | R IN AVL |
| 136 | SINUS/ATRIAL CAPTURE |
| 137 | AV DUAL-PACED COMPLEXES |
| 138 | INFEROLATERAL INJURY PATTERN |
| 139 | RBBB AND LEFT POSTERIOR FASCICULAR BLOCK |
| 140 | ANTEROLATERAL INJURY PATTERN |
| 141 | ATRIAL-PACED COMPLEXES |
| 142 | WITH SINUS PAUSE |
| 143 | BIVENTRICULAR HYPERTROPHY |
| 144 | ABNORMAL RIGHT AXIS DEVIATION |
| 145 | SUPRAVENTRICULAR COMPLEXES |
| 146 | WITH 2ND DEGREE AV BLOCK MOBITZ I |
| 147 | WITH 2:1 AV CONDUCTION |
| 148 | WITH AV DISSOCIATION |
| 149 | MULTIFOCAL ATRIAL TACHYCARDIA |

---

## 3. Off-by-one Bug Fixes (5 classes)

All five indices are exactly +1 from the correct value — a 1-based vs 0-based indexing mistake when the map was built from `tasks.txt`.

| Fzark Event Type | Current Idx | Currently Reading (WRONG) | Fixed Idx | Correct 150-class Label | TP n |
|---|---|---|---|---|---|
| Supraventricular Couplet | 20 | LEFT BUNDLE BRANCH BLOCK | **19** | PREMATURE SUPRAVENTRICULAR COMPLEXES | 2,021 |
| Ventricular Run | 99 | EARLY REPOLARIZATION | **98** | VENTRICULAR TACHYCARDIA | 5,212 |
| Ventricular Couplet | 91 | IN A PATTERN OF BIGEMINY | **90** | PREMATURE VENTRICULAR AND FUSION COMPLEXES | 36 |
| Pause | 143 | BIVENTRICULAR HYPERTROPHY | **142** | WITH SINUS PAUSE | 82 |
| Prolonged RR Interval | 81 | WITH PROLONGED AV CONDUCTION | **80** | PROLONGED QT *(imperfect — see §7)* | 169 |

**Impact**: These 5 classes show 0% or near-0% detection in current reports because the model reads the probability of a completely unrelated class. Fixing these should dramatically increase detection rates for Supraventricular Couplet (n=2,021) and Ventricular Run (n=5,212).

---

## 4. New Mappings for Currently Unmapped Types (8 classes)

| Fzark Event Type | Proposed Idx | 150-class Label | Match Quality | TP n |
|---|---|---|---|---|
| Bradycardia | **4** | SINUS BRADYCARDIA | ✅ Exact | 2,155 |
| ST Elevation | **68** | ST ELEVATION NOW PRESENT IN | ✅ Exact | 1 |
| Supraventricular Run | **93** | SUPRAVENTRICULAR TACHYCARDIA | ✅ Good (SV run = brief SVT) | 26 |
| Ventricular Bigeminy | **9 + 91** | PVC + IN A PATTERN OF BIGEMINY | ⚠️ Multi-head | 2 |
| Supraventricular Bigeminy | **16 + 91** | PAC + IN A PATTERN OF BIGEMINY | ⚠️ Multi-head | 64 |
| Supraventricular Trigeminy | **16** | PREMATURE ATRIAL COMPLEXES | ⚠️ Partial (no trigeminy head) | 76 |
| Ventricular Trigeminy | **9** | PREMATURE VENTRICULAR COMPLEXES | ⚠️ Partial (no trigeminy head) | 16 |
| Multiple Event | — | No mapping | ❌ Meta-category | 116 |
| Unknown | — | No mapping | ❌ Meta-category | 1,929 |

---

## 5. Multi-head Detection Strategy

Some fzark event types combine a morphology component and a pattern component. The 150-class model has separate heads for each:

| Fzark Event | Morphology Head | Pattern Head | Proposed Detection Logic |
|---|---|---|---|
| Ventricular Bigeminy | idx 9: PVC | idx 91: IN A PATTERN OF BIGEMINY | `max(p[9], p[91])` |
| Supraventricular Bigeminy | idx 16: PAC | idx 91: IN A PATTERN OF BIGEMINY | `max(p[16], p[91])` |
| Ventricular Couplet *(corrected)* | idx 9: PVC | idx 90: PVC+FUSION | `max(p[9], p[90])` |

**Recommendation**: Start with `max()` (either head detecting = alert). Evaluate TP retention and FP rate, then tighten to AND logic if FP rate is too high.

---

## 6. Proposed Corrected LABEL_MAP

```python
LABEL_MAP = {
    # ── Correctly mapped (no change) ──────────────────────────────
    "Atrial Fibrillation":              5,  # ATRIAL FIBRILLATION
    "Sinus Tachycardia":                6,  # SINUS TACHYCARDIA
    "Isolated Ventricular Beat":        9,  # PREMATURE VENTRICULAR COMPLEXES
    "Isolated Supraventricular Beat":  16,  # PREMATURE ATRIAL COMPLEXES

    # ── Off-by-one fixes ──────────────────────────────────────────
    "Supraventricular Couplet":        19,  # PREMATURE SUPRAVENTRICULAR COMPLEXES  (was 20: LBBB)
    "Prolonged RR Interval":           80,  # PROLONGED QT                          (was 81: PROLONGED AV CONDUCTION)
    "Ventricular Couplet":             90,  # PREMATURE VENTRICULAR AND FUSION COMPLEXES  (was 91: BIGEMINY)
    "Ventricular Run":                 98,  # VENTRICULAR TACHYCARDIA               (was 99: EARLY REPOLARIZATION)
    "Pause":                          142,  # WITH SINUS PAUSE                      (was 143: BIVENTRICULAR HYPERTROPHY)

    # ── New mappings ──────────────────────────────────────────────
    "Bradycardia":                      4,  # SINUS BRADYCARDIA
    "ST Elevation":                    68,  # ST ELEVATION NOW PRESENT IN
    "Supraventricular Run":            93,  # SUPRAVENTRICULAR TACHYCARDIA
    "Supraventricular Trigeminy":      16,  # PREMATURE ATRIAL COMPLEXES            (no trigeminy head)
    "Ventricular Trigeminy":            9,  # PREMATURE VENTRICULAR COMPLEXES       (no trigeminy head)
    "Supraventricular Bigeminy":       16,  # PREMATURE ATRIAL COMPLEXES            (+ check idx 91 for bigeminy pattern)
    "Ventricular Bigeminy":             9,  # PREMATURE VENTRICULAR COMPLEXES       (+ check idx 91 for bigeminy pattern)
}

# Multi-head secondary indices (for composite event types)
MULTI_HEAD_MAP = {
    "Ventricular Bigeminy":        [9, 91],   # PVC + BIGEMINY pattern
    "Supraventricular Bigeminy":  [16, 91],   # PAC + BIGEMINY pattern
    "Ventricular Couplet":         [9, 90],   # PVC + PVC+FUSION
}

# No mapping — meta-categories without a model head
UNMAPPABLE_EVENTS = {
    "Multiple Event",    # composite of multiple arrhythmia types
    "Unknown",           # unclassified events
    "Custom Heart Rate", # threshold-based detection, not morphological
}
```

---

## 7. Semantic Match Quality Assessment

### Full mapping quality matrix

| Fzark Event Type | Proposed Idx | 150-class Label | Semantic Match | Notes |
|---|---|---|---|---|
| Atrial Fibrillation | 5 | ATRIAL FIBRILLATION | ✅ Exact | No change |
| Sinus Tachycardia | 6 | SINUS TACHYCARDIA | ✅ Exact | No change |
| Isolated Ventricular Beat | 9 | PREMATURE VENTRICULAR COMPLEXES | ✅ Exact | PVC = isolated ventricular beat |
| Isolated Supraventricular Beat | 16 | PREMATURE ATRIAL COMPLEXES | ✅ Exact | PAC = isolated supraventricular beat |
| Bradycardia | 4 | SINUS BRADYCARDIA | ✅ Exact | New mapping |
| ST Elevation | 68 | ST ELEVATION NOW PRESENT IN | ✅ Exact | New mapping |
| Supraventricular Couplet | 19 | PREMATURE SUPRAVENTRICULAR COMPLEXES | ✅ Good | Couplet = paired premature complexes |
| Supraventricular Run | 93 | SUPRAVENTRICULAR TACHYCARDIA | ✅ Good | SV run = brief episode of SVT |
| Ventricular Run | 98 | VENTRICULAR TACHYCARDIA | ✅ Good | V run = brief episode of VT |
| Pause | 142 | WITH SINUS PAUSE | ✅ Good | Direct clinical match |
| Ventricular Couplet | 90 | PREMATURE VENTRICULAR AND FUSION COMPLEXES | ⚠️ Approximate | Couplet ≈ consecutive PVCs |
| Supraventricular Bigeminy | 16 + 91 | PAC + BIGEMINY | ⚠️ Multi-head | Requires combined detection |
| Ventricular Bigeminy | 9 + 91 | PVC + BIGEMINY | ⚠️ Multi-head | Requires combined detection |
| Supraventricular Trigeminy | 16 | PREMATURE ATRIAL COMPLEXES | ⚠️ Partial | No trigeminy-specific head exists |
| Ventricular Trigeminy | 9 | PREMATURE VENTRICULAR COMPLEXES | ⚠️ Partial | No trigeminy-specific head exists |
| Prolonged RR Interval | 80 | PROLONGED QT | ❌ Semantic mismatch | QT ≠ RR; no RR-specific head exists |
| Multiple Event | — | (none) | ❌ No match | Meta-category |
| Unknown | — | (none) | ❌ No match | Meta-category |
| Custom Heart Rate | — | (none) | ❌ No match | Threshold-based, not morphological |

### Notes on problematic mappings

**Prolonged RR Interval → PROLONGED QT (idx 80)**: This is the original off-by-one intent (was 81, corrected to 80), but clinically QT prolongation and RR prolongation are distinct phenomena. QT measures ventricular repolarization duration; RR measures inter-beat interval. The ECGFounder 150-class vocabulary has no direct "Prolonged RR Interval" or "Bradyarrhythmia with pauses" class. This mapping will likely remain low-performing regardless of index correction. Consider:
- Using Bradycardia head (idx 4) as a proxy (low rate → long RR)
- Using Sinus Pause head (idx 142) as a secondary check
- Accepting that this event type cannot be reliably detected by the current model

**Trigeminy events**: The 150-class model has no concept of trigeminy (every-3rd-beat pattern). Mapping to PVC/PAC heads captures the morphology but not the pattern. Detection rates will reflect whether the underlying beat type is present, not whether it appears in a trigeminal pattern.

---

## 8. Affected Scripts

All scripts below use an identical copy of the old `LABEL_MAP` and must be updated:

| Script | Purpose |
|---|---|
| `compare_tp_fzark_with_full_suppression.py` | TP retention evaluation |
| `compare_all_classes_with_full_suppression.py` | FP suppression evaluation |
| `eval_ecg_tprex.py` | TP dataset evaluation |
| `eval_doctor_removed.py` | FP dataset evaluation |
| `eval_ambulatory_model.py` | Ambulatory model evaluation |
| `eval_comprehensive_tprex.py` | Comprehensive TP evaluation |
| `eval_models_tprex_fixed.py` | Multi-model TP comparison |
| `eval_compare_models_tprex.py` | Model comparison on TP data |
| `compare_all_classes_on_fp_dataset.py` | All-class FP comparison |
| `compare_models_on_fp_dataset.py` | Model comparison on FP data |
| `optimize_motion_thresholds.py` | Motion threshold optimization |

**Recommendation**: Extract `LABEL_MAP`, `MULTI_HEAD_MAP`, and `UNMAPPABLE_EVENTS` into a shared module (e.g. `label_config.py`) to eliminate duplication and prevent drift.

---

## 9. Validation Plan

After applying the corrected map, re-run both comparison scripts and compare detection rates:

| Event Type | Current Detection | Expected After Fix | Basis |
|---|---|---|---|
| Supraventricular Couplet | 0.5% | **Significant increase** | Was reading LBBB probability |
| Ventricular Run | 0.0% | **Significant increase** | Was reading Early Repolarization |
| Ventricular Couplet | 0.0% | **Possible increase** | Was reading Bigeminy (n=36) |
| Pause | 0.0% | **Possible increase** | Was reading Biventricular Hypertrophy (n=82) |
| Prolonged RR Interval | 0.0% | **Uncertain** | PROLONGED QT still imperfect match |
| Bradycardia | 11.5% (no head) | **Significant increase** | Now has correct SINUS BRADYCARDIA head (idx 4) |
| Supraventricular Run | unmapped | **New detection** | SVT head (idx 93) |
| ST Elevation | unmapped | **New detection** | ST ELEVATION head (idx 68), n=1 |
| SV/V Bigeminy | unmapped | **New detection** | Multi-head composite |
| SV/V Trigeminy | mapped to PAC/PVC | **Partial detection** | Captures morphology, not pattern |

**Priority**: The off-by-one fixes for Ventricular Run (5,212 TPs) and Supraventricular Couplet (2,021 TPs) are the highest-impact corrections.

---

## 10. Dataset Event Type Distribution

### ecg-tp-fzark (True Positives)

| Event Type | Records | % of Total |
|---|---|---|
| Atrial Fibrillation | 19,566 | 52.5% |
| Ventricular Run | 5,212 | 14.0% |
| Isolated Supraventricular Beat | 3,534 | 9.5% |
| Isolated Ventricular Beat | 2,283 | 6.1% |
| Bradycardia | 2,155 | 5.8% |
| Supraventricular Couplet | 2,021 | 5.4% |
| Unknown | 1,929 | 5.2% |
| Prolonged RR Interval | 169 | 0.5% |
| Multiple Event | 116 | 0.3% |
| Pause | 82 | 0.2% |
| Supraventricular Trigeminy | 76 | 0.2% |
| Supraventricular Bigeminy | 64 | 0.2% |
| Ventricular Couplet | 36 | 0.1% |
| Supraventricular Run | 26 | 0.1% |
| Ventricular Trigeminy | 16 | <0.1% |
| Ventricular Bigeminy | 2 | <0.1% |
| ST Elevation | 1 | <0.1% |
| **Total** | **37,288** | |

### ecg-fp-doctor removed1 (False Positives)

| Event Type | Records | % of Total |
|---|---|---|
| Isolated Supraventricular Beat | 44,156 | 43.0% |
| Supraventricular Run | 21,263 | 20.7% |
| Isolated Ventricular Beat | 10,875 | 10.6% |
| Supraventricular Couplet | 10,061 | 9.8% |
| Bradycardia | 5,941 | 5.8% |
| Ventricular Run | 3,170 | 3.1% |
| Ventricular Couplet | 2,002 | 2.0% |
| Atrial Fibrillation | 1,793 | 1.7% |
| Multiple Event | 885 | 0.9% |
| Pause | 786 | 0.8% |
| Ventricular Trigeminy | 608 | 0.6% |
| Sinus Tachycardia | 530 | 0.5% |
| Supraventricular Trigeminy | 523 | 0.5% |
| Custom Heart Rate | 22 | <0.1% |
| Prolonged RR Interval | 17 | <0.1% |
| Supraventricular Bigeminy | 1 | <0.1% |
| Ventricular Tachycardia | 1 | <0.1% |
| **Total** | **102,634** | |

---

## Appendix A: Old vs New LABEL_MAP Comparison

| Fzark Event Type | Old Idx | Old 150-class (WRONG) | New Idx | New 150-class (CORRECT) | Change |
|---|---|---|---|---|---|
| Atrial Fibrillation | 5 | ATRIAL FIBRILLATION | 5 | ATRIAL FIBRILLATION | — |
| Sinus Tachycardia | 6 | SINUS TACHYCARDIA | 6 | SINUS TACHYCARDIA | — |
| Isolated Ventricular Beat | 9 | PREMATURE VENTRICULAR COMPLEXES | 9 | PREMATURE VENTRICULAR COMPLEXES | — |
| Isolated Supraventricular Beat | 16 | PREMATURE ATRIAL COMPLEXES | 16 | PREMATURE ATRIAL COMPLEXES | — |
| Supraventricular Couplet | 20 | LEFT BUNDLE BRANCH BLOCK | **19** | PREMATURE SUPRAVENTRICULAR COMPLEXES | off-by-one fix |
| Prolonged RR Interval | 81 | WITH PROLONGED AV CONDUCTION | **80** | PROLONGED QT | off-by-one fix |
| Ventricular Couplet | 91 | IN A PATTERN OF BIGEMINY | **90** | PREMATURE VENTRICULAR AND FUSION COMPLEXES | off-by-one fix |
| Ventricular Run | 99 | EARLY REPOLARIZATION | **98** | VENTRICULAR TACHYCARDIA | off-by-one fix |
| Pause | 143 | BIVENTRICULAR HYPERTROPHY | **142** | WITH SINUS PAUSE | off-by-one fix |
| Bradycardia | — | (unmapped) | **4** | SINUS BRADYCARDIA | new |
| ST Elevation | — | (unmapped) | **68** | ST ELEVATION NOW PRESENT IN | new |
| Supraventricular Run | — | (unmapped) | **93** | SUPRAVENTRICULAR TACHYCARDIA | new |
| Supraventricular Trigeminy | — | (unmapped) | **16** | PREMATURE ATRIAL COMPLEXES | new |
| Ventricular Trigeminy | — | (unmapped) | **9** | PREMATURE VENTRICULAR COMPLEXES | new |
| Supraventricular Bigeminy | — | (unmapped) | **16** | PREMATURE ATRIAL COMPLEXES (+idx 91) | new, multi-head |
| Ventricular Bigeminy | — | (unmapped) | **9** | PREMATURE VENTRICULAR COMPLEXES (+idx 91) | new, multi-head |
| Multiple Event | — | (unmapped) | — | (no match) | meta-category |
| Unknown | — | (unmapped) | — | (no match) | meta-category |
| Custom Heart Rate | — | (unmapped) | — | (no match) | threshold-based |

---

## 11. PTB-XL Dataset Label Mapping

### 11.1 Overview

- **Source**: `csv/ptbxl_label.csv` (21,799 records)
- **Splits**: `csv/ptbxl_train.csv` (17,423), `csv/ptbxl_val.csv` (4,376) — patient-level split
- **Label format**: Pre-computed 150-element binary vector in `label` column (JSON list)
- **Mapping method**: Diagnostic text terms from PTB-XL `diagnosis` column matched against `tasks.txt` entries (case-insensitive)
- **Active classes**: **31 of 150** have positive samples; remaining 119 are all-zero
- **Evaluation model**: 12-lead (`12_lead_ECGFounder.pth`) and 1-lead (`1_lead_ECGFounder.pth`)

### 11.2 PTB-XL Active Classes (31 classes with positive samples)

| Index | 150-class Label | PTB-XL Source | Count (n=21,799) | Prevalence |
|---|---|---|---|---|
| 2 | NORMAL ECG | diagnostic_subclass | 9,514 | 43.6% |
| 3 | SINUS RHYTHM | rhythm | 16,748 | 76.8% |
| 4 | SINUS BRADYCARDIA | rhythm | 637 | 2.9% |
| 5 | ATRIAL FIBRILLATION | rhythm | 1,514 | 6.9% |
| 6 | SINUS TACHYCARDIA | rhythm | 826 | 3.8% |
| 9 | PREMATURE VENTRICULAR COMPLEXES | form | 1,143 | 5.2% |
| 11 | RIGHT BUNDLE BRANCH BLOCK | diagnostic_subclass | 1,658 | 7.6% |
| 12 | SEPTAL INFARCT | diagnostic_subclass | 2,357 | 10.8% |
| 13 | LEFT ATRIAL ENLARGEMENT | diagnostic_subclass | 426 | 2.0% |
| 15 | LOW VOLTAGE QRS | form | 3,327 | 15.3% |
| 17 | ANTERIOR INFARCT | diagnostic_subclass | 353 | 1.6% |
| 20 | LEFT BUNDLE BRANCH BLOCK | diagnostic_subclass | 613 | 2.8% |
| 24 | LATERAL INFARCT | diagnostic_subclass | 1,018 | 4.7% |
| 26 | LEFT VENTRICULAR HYPERTROPHY | diagnostic_subclass | 2,354 | 10.8% |
| 30 | QT HAS LENGTHENED | diagnostic_subclass / form | 117 | 0.5% |
| 32 | ATRIAL FLUTTER | rhythm | 73 | 0.3% |
| 36 | LEFT ANTERIOR FASCICULAR BLOCK | diagnostic_subclass | 1,623 | 7.4% |
| 40 | ANTEROSEPTAL INFARCT | diagnostic_subclass | 2,357 | 10.8% |
| 50 | ELECTRONIC ATRIAL PACEMAKER | rhythm | 294 | 1.3% |
| 54 | ANTEROLATERAL INFARCT | diagnostic_subclass | 288 | 1.3% |
| 60 | RIGHT ATRIAL ENLARGEMENT | diagnostic_subclass | 99 | 0.5% |
| 61 | INFERIOR INFARCT | diagnostic_subclass | 2,676 | 12.3% |
| 70 | LEFT POSTERIOR FASCICULAR BLOCK | diagnostic_subclass | 177 | 0.8% |
| 78 | WITH QRS WIDENING | form | 3,327 | 15.3% |
| 79 | WITH 1ST DEGREE AV BLOCK | diagnostic_subclass | 793 | 3.6% |
| 82 | RIGHT VENTRICULAR HYPERTROPHY | diagnostic_subclass | 126 | 0.6% |
| 93 | SUPRAVENTRICULAR TACHYCARDIA | rhythm | 42 | 0.2% |
| 98 | VENTRICULAR TACHYCARDIA | rhythm | 42 | 0.2% |
| 101 | ANTEROLATERAL LEADS | diagnostic_subclass | 659 | 3.0% |
| 107 | WOLFF-PARKINSON-WHITE | diagnostic_subclass | 79 | 0.4% |
| 112 | NONSPECIFIC INTRAVENTRICULAR BLOCK | diagnostic_subclass | 787 | 3.6% |

### 11.3 PTB-XL Diagnostic Superclass Mapping

PTB-XL organizes labels into 5 superclasses. Their relationship to the 150-class indices:

| Superclass | Description | 150-class Indices |
|---|---|---|
| **NORM** | Normal | 2 |
| **MI** | Myocardial Infarction | 12, 17, 24, 40, 54, 61 |
| **STTC** | ST/T Change | 30, 101 |
| **CD** | Conduction Disturbance | 11, 20, 36, 70, 79, 107, 112 |
| **HYP** | Hypertrophy | 13, 26, 60, 82 |

### 11.4 PTB-XL Unmapped Terms (not in 150-class vocabulary)

Several high-frequency PTB-XL diagnostic terms have no matching entry in `tasks.txt` and are silently dropped during label generation:

| PTB-XL Term | Frequency | Notes |
|---|---|---|
| non-diagnostic T abnormalities | 3,650 | Idx 14 (NONSPECIFIC T WAVE ABNORMALITY) is close but text doesn't match |
| non-specific ST changes | 1,534 | Idx 25 (NONSPECIFIC ST ABNORMALITY) is close but text doesn't match |
| non-specific ischemic | 1,272 | No match |
| INRIGHT BUNDLE BRANCH BLOCK | 1,118 | Formatting mismatch — idx 18 is "INCOMPLETE RIGHT BUNDLE BRANCH BLOCK" |
| non-specific ST depression | 1,009 | No match |
| voltage criteria for LVH | 875 | Idx 63 is similar but text doesn't match exactly |
| SINUS ARRHYTHMIA | 772 | Idx 34 is "WITH SINUS ARRHYTHMIA" — "WITH" prefix prevents match |

**Impact**: These unmapped terms affect 6,000+ records where valid diagnostic information is lost. The formatting mismatches (e.g. missing "WITH" prefix, "IN" prefix on RBBB) are fixable.

### 11.5 PTB-XL Evaluation Scripts

| Script | Model | Notes |
|---|---|---|
| `ptbxl_eval.py` | 12-lead | Main evaluation; all 150 classes; per-class AUROC/F1 |
| `ptbxl_eval_lead_ii.py` | 1-lead | Lead II only; same metrics |
| `ptbxl_eval_subset.py` | 12-lead | 5-record subset for debugging |
| `compare_150class_ptbxl.py` | 1-lead × 3 | Compares base/finetuned/fuzzy; macro AUROC on 31 valid classes |

---

## 12. MIT-BIH Arrhythmia Database Label Mapping

### 12.1 Overview

- **Source**: 48 MIT-BIH records in `data/mitdb/` (WFDB format)
- **Segments**: 8,640 total (6,840 train / 1,800 val), 10-second windows at 360 Hz
- **Label format**: 150-element binary vector constructed at runtime from WFDB beat + rhythm annotations
- **Mapping method**: Hardcoded symbol-to-index dictionary in `get_segment_labels()` function
- **Target classes**: **6 classes** evaluated; only **4 have validation data**
- **Evaluation model**: 12-lead (`12_lead_ECGFounder.pth`) and 1-lead (`1_lead_ECGFounder.pth`)

### 12.2 MITDB Beat Symbol → 150-class Mapping

| MITDB Symbol | Beat Type | 150-class Index | 150-class Label |
|---|---|---|---|
| `V` | Premature ventricular contraction | **9** | PREMATURE VENTRICULAR COMPLEXES |
| `L` | Left bundle branch block beat | **20** | LEFT BUNDLE BRANCH BLOCK |
| `R` | Right bundle branch block beat | **11** | RIGHT BUNDLE BRANCH BLOCK |
| `A` | Atrial premature beat | **16** | PREMATURE ATRIAL COMPLEXES |
| `a` | Aberrated atrial premature beat | **16** | PREMATURE ATRIAL COMPLEXES |
| `S` | Supraventricular premature beat | **16** | PREMATURE ATRIAL COMPLEXES |
| `j` | Nodal (junctional) premature beat | **16** | PREMATURE ATRIAL COMPLEXES |

### 12.3 MITDB Rhythm Annotation → 150-class Mapping

| MITDB Rhythm | 150-class Index | 150-class Label |
|---|---|---|
| `(AFIB` | **5** | ATRIAL FIBRILLATION |

### 12.4 MITDB Default/Normal Assignment

If a 10-second segment contains **no AFib rhythm** AND **no abnormal beats** (none of V, L, R, A, a, S, j):
- Index **1** = 1.0 (NORMAL SINUS RHYTHM)
- Index **2** = 1.0 (NORMAL ECG)

### 12.5 MITDB Target Classes for Evaluation

| Index | 150-class Label | Val Positives (n=1,800) | Notes |
|---|---|---|---|
| 1 | NORMAL SINUS RHYTHM | 1,076 | Assigned by exclusion |
| 5 | ATRIAL FIBRILLATION | 456 | From `(AFIB` rhythm annotation |
| 9 | PREMATURE VENTRICULAR COMPLEXES | 343 | From `V` beats |
| 11 | RIGHT BUNDLE BRANCH BLOCK | 0 | No RBBB records in val split |
| 16 | PREMATURE ATRIAL COMPLEXES | 179 | From `A`, `a`, `S`, `j` beats |
| 20 | LEFT BUNDLE BRANCH BLOCK | 0 | No LBBB records in val split |

### 12.6 MITDB Unmapped Annotations

Several MIT-BIH beat types present in the database are **not mapped** and silently ignored:

| Symbol | Beat Type | Notes |
|---|---|---|
| `N` | Normal beat | Implicitly handled by default assignment |
| `F` | Fusion of ventricular and normal | Could map to idx 31 (FUSION COMPLEXES) or idx 90 |
| `e` | Atrial escape beat | No obvious 150-class match |
| `J` | Nodal escape beat | Could map to idx 129 (JUNCTIONAL ESCAPE) |
| `E` | Ventricular escape beat | Could map to idx 108 (VENTRICULAR ESCAPE) |
| `/` | Paced beat | Could map to idx 75 (VENTRICULAR-PACED) or idx 76 (ATRIAL-PACED) |
| `f` | Fusion of paced and normal | No obvious match |
| `Q` | Unclassifiable beat | No match |

Unmapped rhythm annotations:

| Rhythm | Description | Possible 150-class Match |
|---|---|---|
| `(VT` | Ventricular tachycardia | idx 98 (VENTRICULAR TACHYCARDIA) |
| `(AFL` | Atrial flutter | idx 32 (ATRIAL FLUTTER) |
| `(B` | Ventricular bigeminy | idx 91 (IN A PATTERN OF BIGEMINY) |
| `(T` | Ventricular trigeminy | No direct match |
| `(SVTA` | Supraventricular tachyarrhythmia | idx 93 (SUPRAVENTRICULAR TACHYCARDIA) |
| `(SBR` | Sinus bradycardia | idx 4 (SINUS BRADYCARDIA) |
| `(N` | Normal sinus rhythm | idx 1 (NORMAL SINUS RHYTHM) — already default |

### 12.7 MITDB Evaluation Scripts

| Script | Model | Approach |
|---|---|---|
| `mitdb_eval.py` | 12-lead | Pre-processed `.pth` segments; MLII at lead index 1 |
| `mitdb_eval_standard.py` | 12-lead | Raw WFDB on-the-fly; 60 Hz notch; UniversalECGAdapter |
| `mitdb_eval_singlelead.py` | 1-lead + 12-lead | Compares 4 configs: {12-lead, 1-lead} × {standard, robust} |
| `compare_mitdb_models.py` | 1-lead | AFib-only (idx 5); compares 3 checkpoints |

### 12.8 MITDB Design Decision: PAC Grouping

The scripts group 4 different beat types (`A`, `a`, `S`, `j`) under index 16 (PREMATURE ATRIAL COMPLEXES), rather than using index 19 (PREMATURE SUPRAVENTRICULAR COMPLEXES). This is a **conservative** choice — mapping all supraventricular premature beats to the more common PAC head. However, `S` (supraventricular premature) and `j` (junctional premature) are clinically closer to index 19 (PREMATURE SUPRAVENTRICULAR COMPLEXES). This should be re-evaluated for consistency with the fzark mapping where Supraventricular Couplet → idx 19.

---

## 13. Cross-Dataset 150-class Usage Matrix

Which of the 150 classes are active in each dataset:

| Index | 150-class Label | ecg-tp-fzark | PTB-XL | MIT-BIH | Usage |
|---|---|---|---|---|---|
| 1 | NORMAL SINUS RHYTHM | — | — | ✅ default | MITDB only |
| 2 | NORMAL ECG | — | ✅ | ✅ default | PTB-XL + MITDB |
| 3 | SINUS RHYTHM | — | ✅ | — | PTB-XL only |
| 4 | SINUS BRADYCARDIA | ✅ new | ✅ | — | fzark + PTB-XL |
| 5 | ATRIAL FIBRILLATION | ✅ | ✅ | ✅ | **All 3** |
| 6 | SINUS TACHYCARDIA | ✅ | ✅ | — | fzark + PTB-XL |
| 9 | PREMATURE VENTRICULAR COMPLEXES | ✅ | ✅ | ✅ | **All 3** |
| 11 | RIGHT BUNDLE BRANCH BLOCK | — | ✅ | ✅ | PTB-XL + MITDB |
| 12 | SEPTAL INFARCT | — | ✅ | — | PTB-XL only |
| 13 | LEFT ATRIAL ENLARGEMENT | — | ✅ | — | PTB-XL only |
| 15 | LOW VOLTAGE QRS | — | ✅ | — | PTB-XL only |
| 16 | PREMATURE ATRIAL COMPLEXES | ✅ | — | ✅ | fzark + MITDB |
| 17 | ANTERIOR INFARCT | — | ✅ | — | PTB-XL only |
| 19 | PREMATURE SUPRAVENTRICULAR COMPLEXES | ✅ fix | — | — | fzark only |
| 20 | LEFT BUNDLE BRANCH BLOCK | — | ✅ | ✅ | PTB-XL + MITDB |
| 24 | LATERAL INFARCT | — | ✅ | — | PTB-XL only |
| 26 | LEFT VENTRICULAR HYPERTROPHY | — | ✅ | — | PTB-XL only |
| 30 | QT HAS LENGTHENED | — | ✅ | — | PTB-XL only |
| 32 | ATRIAL FLUTTER | — | ✅ | — | PTB-XL only |
| 36 | LEFT ANTERIOR FASCICULAR BLOCK | — | ✅ | — | PTB-XL only |
| 40 | ANTEROSEPTAL INFARCT | — | ✅ | — | PTB-XL only |
| 50 | ELECTRONIC ATRIAL PACEMAKER | — | ✅ | — | PTB-XL only |
| 54 | ANTEROLATERAL INFARCT | — | ✅ | — | PTB-XL only |
| 60 | RIGHT ATRIAL ENLARGEMENT | — | ✅ | — | PTB-XL only |
| 61 | INFERIOR INFARCT | — | ✅ | — | PTB-XL only |
| 68 | ST ELEVATION NOW PRESENT IN | ✅ new | — | — | fzark only |
| 70 | LEFT POSTERIOR FASCICULAR BLOCK | — | ✅ | — | PTB-XL only |
| 78 | WITH QRS WIDENING | — | ✅ | — | PTB-XL only |
| 79 | WITH 1ST DEGREE AV BLOCK | — | ✅ | — | PTB-XL only |
| 80 | PROLONGED QT | ✅ fix | — | — | fzark only (imperfect) |
| 82 | RIGHT VENTRICULAR HYPERTROPHY | — | ✅ | — | PTB-XL only |
| 90 | PREMATURE VENTRICULAR AND FUSION COMPLEXES | ✅ fix | — | — | fzark only |
| 91 | IN A PATTERN OF BIGEMINY | ✅ new | — | — | fzark only |
| 93 | SUPRAVENTRICULAR TACHYCARDIA | ✅ new | ✅ | — | fzark + PTB-XL |
| 98 | VENTRICULAR TACHYCARDIA | ✅ fix | ✅ | — | fzark + PTB-XL |
| 101 | ANTEROLATERAL LEADS | — | ✅ | — | PTB-XL only |
| 107 | WOLFF-PARKINSON-WHITE | — | ✅ | — | PTB-XL only |
| 112 | NONSPECIFIC INTRAVENTRICULAR BLOCK | — | ✅ | — | PTB-XL only |
| 142 | WITH SINUS PAUSE | ✅ fix | — | — | fzark only |

**Summary**: Of the 150 classes, **38 are used** across all datasets. Only **2 indices** (5, 9) are shared by all 3 datasets.

---

## 14. Cross-Dataset Consistency Issues

### 14.1 Index conflicts and alignment

| 150-class Index | fzark Uses For | MITDB Uses For | PTB-XL Uses For | Consistent? |
|---|---|---|---|---|
| 5 | Atrial Fibrillation | AFib rhythm `(AFIB` | ATRIAL FIBRILLATION | ✅ |
| 9 | Isolated Ventricular Beat | PVC beat `V` | PREMATURE VENTRICULAR COMPLEXES | ✅ |
| 16 | Isolated Supraventricular Beat | PAC beats `A`,`a`,`S`,`j` | — | ✅ |
| 4 | Bradycardia (new) | — | SINUS BRADYCARDIA | ✅ |
| 6 | Sinus Tachycardia | — | SINUS TACHYCARDIA | ✅ |
| 93 | Supraventricular Run (new) | — | SUPRAVENTRICULAR TACHYCARDIA | ✅ |
| 98 | Ventricular Run (fixed) | — | VENTRICULAR TACHYCARDIA | ✅ |
| 11 | — | RBBB beat `R` | RIGHT BUNDLE BRANCH BLOCK | ✅ |
| 20 | — | LBBB beat `L` | LEFT BUNDLE BRANCH BLOCK | ✅ |

No cross-dataset conflicts exist — all shared indices are used consistently.

### 14.2 MITDB PAC grouping inconsistency

MITDB groups `S` (supraventricular premature) and `j` (junctional premature) under index 16 (PAC). After the fzark fix, Supraventricular Couplet maps to index 19 (PREMATURE SUPRAVENTRICULAR COMPLEXES). These are related but distinct indices:
- Index 16: PREMATURE **ATRIAL** COMPLEXES — atrial-origin ectopics
- Index 19: PREMATURE **SUPRAVENTRICULAR** COMPLEXES — broader category (atrial + junctional)

**Recommendation**: Consider splitting MITDB mapping so `A`/`a` → idx 16 (atrial) and `S`/`j` → idx 19 (supraventricular), matching the more granular fzark mapping.

### 14.3 PTB-XL label generation gaps

The PTB-XL label vectors are pre-computed offline. Several high-frequency diagnostic terms fail to match `tasks.txt` due to formatting differences:

| PTB-XL Term | Closest tasks.txt Entry | Issue |
|---|---|---|
| INRIGHT BUNDLE BRANCH BLOCK | INCOMPLETE RIGHT BUNDLE BRANCH BLOCK (idx 18) | Missing "COMPLETE " prefix |
| SINUS ARRHYTHMIA | WITH SINUS ARRHYTHMIA (idx 34) | Missing "WITH " prefix |
| voltage criteria for LVH | VOLTAGE CRITERIA FOR LEFT VENTRICULAR HYPERTROPHY (idx 63) | Case + abbreviation mismatch |
| non-diagnostic T abnormalities | NONSPECIFIC T WAVE ABNORMALITY (idx 14) | Different wording |
| non-specific ST changes | NONSPECIFIC ST ABNORMALITY (idx 25) | Different wording |

**Impact**: ~6,000+ PTB-XL records have valid diagnostic labels dropped during label generation. Fixing text matching could increase the number of active classes from 31 to ~36.

---

## 15. Unified Label Configuration Proposal

### 15.1 Shared module: `label_config.py`

```python
"""
label_config.py — Unified label mapping for ECGFounder 150-class model.

Single source of truth for all dataset → 150-class index mappings.
Used by: ecg-tp-fzark, ecg-fp, PTB-XL, and MIT-BIH evaluation scripts.
"""

# ── Load 150-class vocabulary ─────────────────────────────────────
TASKS_FILE = "tasks.txt"

def load_tasks(path=TASKS_FILE):
    with open(path) as f:
        return [line.strip() for line in f]

# ── ecg-tp-fzark / ecg-fp event type → 150-class index ───────────
FZARK_LABEL_MAP = {
    "Atrial Fibrillation":              5,
    "Sinus Tachycardia":                6,
    "Isolated Ventricular Beat":        9,
    "Isolated Supraventricular Beat":  16,
    "Supraventricular Couplet":        19,
    "Prolonged RR Interval":           80,   # imperfect semantic match
    "Ventricular Couplet":             90,
    "Ventricular Run":                 98,
    "Pause":                          142,
    "Bradycardia":                      4,
    "ST Elevation":                    68,
    "Supraventricular Run":            93,
    "Supraventricular Trigeminy":      16,   # no trigeminy head
    "Ventricular Trigeminy":            9,   # no trigeminy head
    "Supraventricular Bigeminy":       16,   # + check idx 91
    "Ventricular Bigeminy":             9,   # + check idx 91
}

FZARK_MULTI_HEAD = {
    "Ventricular Bigeminy":        [9, 91],
    "Supraventricular Bigeminy":  [16, 91],
    "Ventricular Couplet":         [9, 90],
}

FZARK_UNMAPPABLE = {"Multiple Event", "Unknown", "Custom Heart Rate"}

# ── MIT-BIH beat symbol → 150-class index ─────────────────────────
MITDB_BEAT_MAP = {
    "V":  9,   # PVC
    "L": 20,   # LBBB
    "R": 11,   # RBBB
    "A": 16,   # Atrial premature
    "a": 16,   # Aberrated atrial premature
    "S": 16,   # Supraventricular premature (consider → 19)
    "j": 16,   # Junctional premature (consider → 19)
}

MITDB_RHYTHM_MAP = {
    "(AFIB": 5,   # Atrial fibrillation
    # Proposed additions:
    # "(VT":   98,  # Ventricular tachycardia
    # "(AFL":  32,  # Atrial flutter
    # "(B":    91,  # Ventricular bigeminy
    # "(SVTA": 93,  # SVT
    # "(SBR":   4,  # Sinus bradycardia
}

MITDB_DEFAULT_NORMAL = [1, 2]  # NSR + Normal ECG when no abnormality

MITDB_TARGET_CLASSES = {
    1:  "NORMAL SINUS RHYTHM",
    5:  "ATRIAL FIBRILLATION",
    9:  "PREMATURE VENTRICULAR COMPLEXES",
    11: "RIGHT BUNDLE BRANCH BLOCK",
    16: "PREMATURE ATRIAL COMPLEXES",
    20: "LEFT BUNDLE BRANCH BLOCK",
}

# ── PTB-XL active classes (31 with positive samples) ─────────────
PTBXL_ACTIVE_CLASSES = {
    2, 3, 4, 5, 6, 9, 11, 12, 13, 15, 17, 20, 24, 26, 30, 32,
    36, 40, 50, 54, 60, 61, 70, 78, 79, 82, 93, 98, 101, 107, 112,
}
```

### 15.2 Migration steps

1. Create `label_config.py` with unified mappings
2. Update all 11 fzark/fp scripts to `from label_config import FZARK_LABEL_MAP`
3. Update 3 MITDB scripts to `from label_config import MITDB_BEAT_MAP, MITDB_RHYTHM_MAP`
4. Update PTB-XL label generation to fix text-matching gaps
5. Re-run all evaluation pipelines and compare results

---

## Appendix B: PTB-XL Full Label Reference

### Active 150-class indices with PTB-XL diagnostic mapping

| Index | 150-class Label | PTB-XL Source Column | PTB-XL Superclass |
|---|---|---|---|
| 2 | NORMAL ECG | diagnostic_subclass | NORM |
| 3 | SINUS RHYTHM | rhythm | — |
| 4 | SINUS BRADYCARDIA | rhythm | — |
| 5 | ATRIAL FIBRILLATION | rhythm | — |
| 6 | SINUS TACHYCARDIA | rhythm | — |
| 9 | PREMATURE VENTRICULAR COMPLEXES | form | — |
| 11 | RIGHT BUNDLE BRANCH BLOCK | diagnostic_subclass | CD |
| 12 | SEPTAL INFARCT | diagnostic_subclass | MI |
| 13 | LEFT ATRIAL ENLARGEMENT | diagnostic_subclass | HYP |
| 15 | LOW VOLTAGE QRS | form | — |
| 17 | ANTERIOR INFARCT | diagnostic_subclass | MI |
| 20 | LEFT BUNDLE BRANCH BLOCK | diagnostic_subclass | CD |
| 24 | LATERAL INFARCT | diagnostic_subclass | MI |
| 26 | LEFT VENTRICULAR HYPERTROPHY | diagnostic_subclass | HYP |
| 30 | QT HAS LENGTHENED | diagnostic_subclass / form | STTC |
| 32 | ATRIAL FLUTTER | rhythm | — |
| 36 | LEFT ANTERIOR FASCICULAR BLOCK | diagnostic_subclass | CD |
| 40 | ANTEROSEPTAL INFARCT | diagnostic_subclass | MI |
| 50 | ELECTRONIC ATRIAL PACEMAKER | rhythm | — |
| 54 | ANTEROLATERAL INFARCT | diagnostic_subclass | MI |
| 60 | RIGHT ATRIAL ENLARGEMENT | diagnostic_subclass | HYP |
| 61 | INFERIOR INFARCT | diagnostic_subclass | MI |
| 70 | LEFT POSTERIOR FASCICULAR BLOCK | diagnostic_subclass | CD |
| 78 | WITH QRS WIDENING | form | — |
| 79 | WITH 1ST DEGREE AV BLOCK | diagnostic_subclass | CD |
| 82 | RIGHT VENTRICULAR HYPERTROPHY | diagnostic_subclass | HYP |
| 93 | SUPRAVENTRICULAR TACHYCARDIA | rhythm | — |
| 98 | VENTRICULAR TACHYCARDIA | rhythm | — |
| 101 | ANTEROLATERAL LEADS | diagnostic_subclass | STTC |
| 107 | WOLFF-PARKINSON-WHITE | diagnostic_subclass | CD |
| 112 | NONSPECIFIC INTRAVENTRICULAR BLOCK | diagnostic_subclass | CD |

## Appendix C: MIT-BIH Full Annotation Reference

### All MIT-BIH beat annotation symbols

| Symbol | Description | Mapped Idx | Status |
|---|---|---|---|
| `N` | Normal beat | — | Default (exclusion-based) |
| `L` | Left BBB beat | 20 | ✅ Mapped |
| `R` | Right BBB beat | 11 | ✅ Mapped |
| `A` | Atrial premature beat | 16 | ✅ Mapped |
| `a` | Aberrated atrial premature | 16 | ✅ Mapped |
| `J` | Nodal (junctional) premature | 16 | ✅ Mapped (consider → 19) |
| `S` | Supraventricular premature | 16 | ✅ Mapped (consider → 19) |
| `V` | Premature ventricular contraction | 9 | ✅ Mapped |
| `F` | Fusion of ventricular and normal | — | ❌ Unmapped (→ idx 31 or 90) |
| `e` | Atrial escape beat | — | ❌ Unmapped |
| `j` | Nodal (junctional) escape beat | — | ❌ Unmapped (→ idx 129) |
| `E` | Ventricular escape beat | — | ❌ Unmapped (→ idx 108) |
| `/` | Paced beat | — | ❌ Unmapped (→ idx 75/76) |
| `f` | Fusion of paced and normal | — | ❌ Unmapped |
| `Q` | Unclassifiable beat | — | ❌ Unmapped |

### All MIT-BIH rhythm annotations

| Rhythm | Description | Mapped Idx | Status |
|---|---|---|---|
| `(N` | Normal sinus rhythm | 1 | ✅ Via default |
| `(AFIB` | Atrial fibrillation | 5 | ✅ Mapped |
| `(AFL` | Atrial flutter | — | ❌ Unmapped (→ idx 32) |
| `(VT` | Ventricular tachycardia | — | ❌ Unmapped (→ idx 98) |
| `(B` | Ventricular bigeminy | — | ❌ Unmapped (→ idx 91) |
| `(T` | Ventricular trigeminy | — | ❌ Unmapped |
| `(SVTA` | SVT / atrial tachycardia | — | ❌ Unmapped (→ idx 93) |
| `(SBR` | Sinus bradycardia | — | ❌ Unmapped (→ idx 4) |
| `(IVR` | Idioventricular rhythm | — | ❌ Unmapped (→ idx 133) |
| `(AB` | Atrial bigeminy | — | ❌ Unmapped (→ idx 91) |
| `(PREX` | Pre-excitation (WPW) | — | ❌ Unmapped (→ idx 107) |
| `(NOD` | Nodal rhythm | — | ❌ Unmapped (→ idx 49) |
| `(P` | Paced rhythm | — | ❌ Unmapped (→ idx 75) |
| `(VFL` | Ventricular flutter | — | ❌ Unmapped |

---

*Generated from `tasks.txt`, `ecg_tp_fzark/summary.csv`, `csv/ptbxl_label.csv`, and MIT-BIH WFDB annotation analysis.*
