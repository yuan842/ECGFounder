# LABEL_MAP Off-by-One Bug Report

**Date discovered**: 2026-05-27
**Root cause**: 1-based line number used as 0-based array index
**Introduced in**: Commit `6e7282c57e` (2026-05-20, "Single Lead Model Finetune")
**Author**: yuan842 <yuan@vivalink.com>
**Scope**: 5 of 9 event-type mappings incorrect across 11 scripts
**Severity**: High — 5 event types report 0% detection due to reading wrong model head

---

## 1. Summary

The `LABEL_MAP` dictionary in `eval_ecg_tprex.py` (and 10 copies in other scripts) maps ecg-tp-fzark event type names to ECGFounder 150-class model output indices. Five of the nine entries have an off-by-one error: the raw 1-based line number from `tasks.txt` was used directly as a 0-based Python array index, causing the model to read the probability of a completely unrelated class.

---

## 2. Root Cause

`tasks.txt` contains 150 class names, one per line, with no header. Text editors display lines starting from 1 (line 1 = "ABNORMAL ECG"), but Python lists are 0-indexed (index 0 = "ABNORMAL ECG").

The LABEL_MAP was hand-coded by looking up class names in a text editor. For 4 entries with small indices, the author correctly subtracted 1 to convert from line number to array index. For the other 5 entries, the author forgot to subtract 1 and wrote the raw line number.

No generation script exists — the mapping was entirely manual.

---

## 3. Correctly Mapped Entries (no error)

These 4 entries had the line-to-index conversion done correctly:

| Event Type | Editor Line # | Index Written | 150-class at Index | Correct? |
|---|---|---|---|---|
| Atrial Fibrillation | 6 | **5** | ATRIAL FIBRILLATION | ✅ |
| Sinus Tachycardia | 7 | **6** | SINUS TACHYCARDIA | ✅ |
| Isolated Ventricular Beat | 10 | **9** | PREMATURE VENTRICULAR COMPLEXES | ✅ |
| Isolated Supraventricular Beat | 17 | **16** | PREMATURE ATRIAL COMPLEXES | ✅ |

---

## 4. Off-by-One Errors (5 entries)

These 5 entries used the 1-based line number directly without subtracting 1:

| Event Type | Editor Line # | Index Written (WRONG) | Class at Wrong Index | Correct Index | Class at Correct Index | TP Records Affected |
|---|---|---|---|---|---|---|
| Supraventricular Couplet | 20 | **20** | LEFT BUNDLE BRANCH BLOCK | **19** | PREMATURE SUPRAVENTRICULAR COMPLEXES | 2,021 |
| Prolonged RR Interval | 81 | **81** | WITH PROLONGED AV CONDUCTION | **80** | PROLONGED QT | 169 |
| Ventricular Couplet | 91 | **91** | IN A PATTERN OF BIGEMINY | **90** | PREMATURE VENTRICULAR AND FUSION COMPLEXES | 36 |
| Ventricular Run | 99 | **99** | EARLY REPOLARIZATION | **98** | VENTRICULAR TACHYCARDIA | 5,212 |
| Pause | 143 | **143** | BIVENTRICULAR HYPERTROPHY | **142** | WITH SINUS PAUSE | 82 |

---

## 5. Timeline

| Date | Event |
|---|---|
| **2025-06-09** | `tasks.txt` created by upstream author (Shenda Hong, commit `296602f14b`, "Add files via upload"). 150 lines, no header, never modified after this. |
| **2026-05-20** | `eval_ecg_tprex.py` created with hand-coded LABEL_MAP containing all 5 off-by-one errors (commit `6e7282c57e`, "Single Lead Model Finetune"). This is the first and only commit introducing the map. |
| **2026-05-25** | `eval_doctor_removed.py` created with identical LABEL_MAP copy-pasted from `eval_ecg_tprex.py` (commit `5ade4384ef`, "Single Lead Model Finetune"). |
| **2026-05-25+** | 9 additional scripts created (untracked) by copying the same LABEL_MAP. The bug propagated to all files. |
| **2026-05-27** | Bug discovered during label reclassification analysis. |

---

## 6. Git Evidence

### tasks.txt history

```
$ git log --all --oneline -- tasks.txt
296602f14b Add files via upload        (2025-06-09, Shenda Hong)
8ca3b64b29 150-class classification    (2025-08-05, NickLJLee — byte-identical content)
```

The file has never been modified. It has exactly 150 lines, no header, no blank lines.

### LABEL_MAP introduction

```
$ git log --all --oneline -- eval_ecg_tprex.py
6e7282c57e Single Lead Model Finetune  (2026-05-20, yuan842)
```

`git blame` on the LABEL_MAP lines in `eval_ecg_tprex.py` attributes every line to commit `6e7282c57e`. The mapping has never been changed.

### No generation script

`grep -r` for any script that programmatically builds LABEL_MAP from `tasks.txt` found nothing. The mapping was entirely hand-coded.

---

## 7. Affected Scripts

All 11 scripts below contain an identical copy of the erroneous LABEL_MAP:

| Script | Source |
|---|---|
| `eval_ecg_tprex.py` | Original (committed) |
| `eval_doctor_removed.py` | Copy (committed) |
| `compare_tp_fzark_with_full_suppression.py` | Copy (untracked) |
| `compare_all_classes_with_full_suppression.py` | Copy (untracked) |
| `eval_ambulatory_model.py` | Copy (untracked) |
| `eval_comprehensive_tprex.py` | Copy (untracked) |
| `eval_models_tprex_fixed.py` | Copy (untracked) |
| `eval_compare_models_tprex.py` | Copy (untracked) |
| `compare_all_classes_on_fp_dataset.py` | Copy (untracked) |
| `compare_models_on_fp_dataset.py` | Copy (untracked) |
| `optimize_motion_thresholds.py` | Copy (untracked) |

---

## 8. Why It Went Unnoticed

The 5 affected event types all show **0% or near-0% detection** in evaluation reports:

| Event Type | Reported Detection | Explanation |
|---|---|---|
| Supraventricular Couplet | 0.5% | Reading LBBB probability — unrelated to SV premature beats |
| Ventricular Run | 0.0% | Reading Early Repolarization probability — unrelated to VT |
| Ventricular Couplet | 0.0% | Reading Bigeminy probability — tangentially related but wrong |
| Pause | 0.0% | Reading Biventricular Hypertrophy probability — unrelated |
| Prolonged RR Interval | 0.0% | Reading Prolonged AV Conduction — adjacent but wrong |

These zero-detection results were likely attributed to "the model doesn't support these event types" rather than questioning whether the index mapping was correct. The model may actually have reasonable detection capability for Ventricular Run (via VT head, idx 98) and Supraventricular Couplet (via premature SV complexes head, idx 19) — this will only be revealed after the fix.

---

## 9. Corrected LABEL_MAP

```python
LABEL_MAP = {
    # Unchanged (correctly mapped)
    "Atrial Fibrillation":              5,  # ATRIAL FIBRILLATION
    "Sinus Tachycardia":                6,  # SINUS TACHYCARDIA
    "Isolated Ventricular Beat":        9,  # PREMATURE VENTRICULAR COMPLEXES
    "Isolated Supraventricular Beat":  16,  # PREMATURE ATRIAL COMPLEXES

    # Fixed (was off by +1)
    "Supraventricular Couplet":        19,  # was 20 (LBBB) → now PREMATURE SUPRAVENTRICULAR COMPLEXES
    "Prolonged RR Interval":           80,  # was 81 (PROLONGED AV CONDUCTION) → now PROLONGED QT
    "Ventricular Couplet":             90,  # was 91 (BIGEMINY) → now PREMATURE VENTRICULAR AND FUSION COMPLEXES
    "Ventricular Run":                 98,  # was 99 (EARLY REPOLARIZATION) → now VENTRICULAR TACHYCARDIA
    "Pause":                          142,  # was 143 (BIVENTRICULAR HYPERTROPHY) → now WITH SINUS PAUSE
}
```

---

## 10. Recommended Fix

1. Create a shared `label_config.py` module with the corrected LABEL_MAP as the single source of truth.
2. Update all 11 scripts to import from `label_config.py` instead of maintaining local copies.
3. Re-run TP and FP comparison scripts with corrected indices.
4. Expected impact: Significant detection rate increases for Ventricular Run (5,212 TPs) and Supraventricular Couplet (2,021 TPs).

---

## 11. Lesson Learned

When mapping class labels from a text file to array indices, always verify with programmatic lookup rather than manual line counting:

```python
tasks = [line.strip() for line in open('tasks.txt')]
# Correct: search by name, get 0-based index
idx = tasks.index("VENTRICULAR TACHYCARDIA")  # returns 98, not 99
```

---

*See also: `res/label_reclassification_plan.md` for the full multi-dataset reclassification plan.*
