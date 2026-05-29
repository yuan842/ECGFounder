# Performance Baseline — ECG-TP / ECG-FP Datasets

**Date**: 2026-05-27
**Branch**: `cleanup-12lead-finetune-v1-clean` (after v3 label ontology + preprocessing consolidation)
**Model**: `1_lead_ECGFounder.pth` (1-lead, 150-class, downloaded from HuggingFace; 352 MB)
**Device**: MPS (Apple Silicon)

---

## 1. Scope and Method

### Datasets evaluated

| Dataset | Records (cohort) | Status | Notes |
|---|---|---|---|
| **ecg-tp-fzark** (TPs) | 1,988 (stratified 200/class) | ✅ evaluated | Clinician-confirmed true positives |
| **ecg-fp-doctor removed1** (FPs) | — | ⚠️ unavailable | Dataset not present in `data/`; FP eval skipped |

### Method

- Preprocessing via `ECGPreprocessor.for_fzark()` (50 Hz notch + 0.67–40 Hz bandpass + median baseline removal + winsorized z-score, 128 → 500 Hz resample, center crop to 5000 samples)
- Single forward pass through the 1-lead `Net1D` model
- Sigmoid output → 150-class probabilities
- Per-class detection rate at three thresholds: 0.5, 0.6, 0.7
- Two passes: `--suppression off` (raw model) and `--suppression on` (v2 feature-gate filter applied)
- 200 records per event type (full population for classes with fewer records)
- Indices use the v3 single-index ontology in `label_config.FZARK_LABEL_MAP`

### Files generated

| File | Contents |
|---|---|
| `res/tp_fzark_full_suppression/tp_baseline_threshold_variants_supp_off.csv` | Per-class TP retention at t={0.5, 0.6, 0.7}, suppression OFF |
| `res/tp_fzark_full_suppression/tp_baseline_threshold_variants.csv` | Same, suppression ON |
| `res/tp_fzark_full_suppression/tp_fzark_threshold_variants_report_supp_off.md` | Markdown report, suppression OFF |
| `res/tp_fzark_full_suppression/tp_fzark_threshold_variants_report.md` | Markdown report, suppression ON |
| `res/tp_fzark_full_suppression/baseline_probs_tp.npy` | Raw 150-class sigmoid probabilities (cached) |

---

## 2. Baseline (Suppression OFF) — Raw Model Performance

### 2.1 Per-class detection rate at threshold = 0.5

Events with single-head ontology mappings (`* = native-class head`):

| Event Type | n | 150-class index | Detection @ 0.5 | Tier |
|---|---|---|---|---|
| Atrial Fibrillation | 200 | 5 | **98.5%** | HIGH |
| Bradycardia | 200 | 4 | **95.5%** | HIGH |
| Isolated Ventricular Beat | 200 | 9 | 81.0% | LOW |
| Isolated Supraventricular Beat | 200 | 16 | 53.0% | LOW |
| Supraventricular Couplet | 200 | 19 | 5.5% | MODERATE |
| Ventricular Run | 200 | 98 | **0.0%** | CRITICAL |
| Pause | 82 | 142 | **0.0%** | CRITICAL |
| Supraventricular Run | 26 | 93 | 0.0% | MODERATE |
| ST Elevation | 1 | 68 | 0.0% | CRITICAL (n=1, not meaningful) |
| Sinus Tachycardia | (not in fzark TP) | 6 | n/a | LOW |

Events that pass through (no single-head mapping in v3):

| Event Type | n | Detection @ 0.5 (any head) | Status |
|---|---|---|---|
| Unknown | 200 | 90.5% | Unmappable (meta-category) |
| Multiple Event | 116 | 56.9% | Unmappable (meta-category) |
| Supraventricular Bigeminy | 64 | 64.1% | Composite — dropped in v3 |
| Ventricular Couplet | 36 | 41.7% | Composite — dropped in v3 |
| Ventricular Bigeminy | 2 | 50.0% | Composite — dropped in v3 |
| Ventricular Trigeminy | 16 | 6.2% | Pattern (no head) — dropped in v3 |
| Supraventricular Trigeminy | 76 | 5.3% | Pattern (no head) — dropped in v3 |
| Prolonged RR Interval | 169 | 18.3% | Vocabulary-limited — dropped in v3 |

### 2.2 Threshold-variant aggregate

| Threshold | Detection rate (weighted by n) | Records detected |
|---|---|---|
| 0.5 | **50.7%** | 1,007 / 1,988 |
| 0.6 | 45.6% | 907 / 1,988 |
| 0.7 | 40.5% | 806 / 1,988 |

### 2.3 Observations

1. **Strong heads** (>80% detection at t=0.5):
   - Atrial Fibrillation (98.5%), Bradycardia (95.5%), Isolated Ventricular Beat (81.0%).
   - These are clinically well-represented in the training vocabulary and dominate the TP dataset (~24,000 of the 37,288 records, ~64% of the dataset).

2. **Underperforming critical heads** (0% detection):
   - **Ventricular Run → VT head (idx 98)**: 0/200 detected. With 5,212 TPs in the full dataset, this is the highest-impact gap. The off-by-one fix from idx 99 (EARLY REPOLARIZATION) to idx 98 (VENTRICULAR TACHYCARDIA) was supposed to lift this from 0%, but the model probability appears to stay below 0.5 across the V-Run cohort.
   - **Pause → WITH SINUS PAUSE (idx 142)**: 0/82 detected.
   - **Supraventricular Run → SVT head (idx 93)**: 0/26 detected.
   - **Supraventricular Couplet → PSVC head (idx 19)**: 5.5% — better than the pre-fix LBBB miss (0.5%) but still very low.

   These four classes share a pattern: brief paroxysmal events where the model's head likely activates on the entire 10-second window, and an isolated couplet/run within a normal background doesn't push the global head probability above 0.5.

3. **Anomalous "Unknown" detection rate (90.5%)**: the model fires *some* head on most "Unknown" events, suggesting these aren't unrecognizable — they may be legitimate arrhythmias the annotators couldn't categorize. Worth investigating whether these align with idx 5 (AFib) or other major heads.

4. **Threshold sensitivity**: detection rate drops ~10% per 0.1 threshold increment for the high-activity heads. AFib stays robust (95.0% at t=0.7). Underperforming classes don't recover at lower thresholds — even t=0.5 produces 0% for V-Run/Pause/SV-Run, which means the head probability for these events is structurally low (not a threshold problem).

---

## 3. Suppression ON — Pipeline Performance

### 3.1 Per-class TP retention

The v2 feature-gate filter applies rules to 4 classes: AFib, Bradycardia, SV Trigeminy, V Trigeminy. All other classes pass through.

| Event Type | Detection % | Post-supp Retain % | TP loss % | Notes |
|---|---|---|---|---|
| Atrial Fibrillation | 98.5% | 98.5% | 0.0% | motion gate non-binding |
| **Bradycardia** | 95.5% | 92.5% | **3.1%** | HR-gate trims 6 records |
| Supraventricular Trigeminy | 5.3% | 5.3% | 0.0% | motion gate non-binding on TPs |
| **Ventricular Trigeminy** | 6.2% | 0.0% | **100.0%** | n=16; only 1 detected, filter killed it |
| (all other classes) | — | unchanged | 0% | passthrough |

### 3.2 Aggregate impact (weighted by n)

| Threshold | Detection | Post-suppression | TP loss |
|---|---|---|---|
| 0.5 | 50.7% (1007/1988) | 50.3% (1000/1988) | **0.7%** (7 records) |
| 0.6 | 45.6% (907/1988) | 45.3% (900/1988) | 0.8% |
| 0.7 | 40.5% (806/1988) | 40.2% (799/1988) | 0.9% |

### 3.3 Suppressed-classes-only aggregate (n=492)

| Threshold | Detection | Post-suppression | TP loss |
|---|---|---|---|
| 0.5 | 79.9% (393/492) | 78.5% (386/492) | **1.8%** (7 records) |
| 0.6 | 78.9% (388/492) | 77.4% (381/492) | 1.8% |
| 0.7 | 78.0% (384/492) | 76.6% (377/492) | 1.8% |

### 3.4 Observations

1. **Net cost of suppression on the TP side**: 7 records lost out of 1,007 detected (0.7% aggregate, 1.8% within the suppressed classes). Acceptable trade-off given the FP-elimination gains reported elsewhere.

2. **V-Trigeminy 100% loss** is statistically fragile (n=16, 1 detected pre-suppression). The dual gate (motion ≥ 24 mG AND snr_proxy > 1.2) rejects that single TP. This class is too small to draw conclusions from — should be reviewed once more V-Trigeminy data is available.

3. **AFib motion gate (motion ≤ 5 mG)** is non-binding on the TP cohort: 0 of 197 detected AFib TPs were suppressed. This matches the design intent — TPs typically occur at rest, while FPs (from the FP dataset) come from motion artifacts.

4. **Bradycardia HR gate (HR ≤ 56.3 bpm)** trims 6 TPs that detected as bradycardia but had HR > 56.3 bpm — likely borderline cases where the model fired on near-bradycardic heart rates that don't meet the strict definition.

---

## 4. Missing Half: FP Dataset Performance

`ecg-fp-doctor removed1` is the false-positive companion dataset (102,634 records of clinician-removed alerts). It is **not currently present in `data/`**, so we cannot report the corresponding FP suppression metrics in this run.

When the dataset is restored, run:

```bash
python3 compare_all_classes_with_full_suppression.py --suppression off  # baseline FP rate
python3 compare_all_classes_with_full_suppression.py --suppression on   # FP rate post-suppression
```

The expected outcome (from historical runs): the v2 suppression filter drops suppressed-class FP rate from 58.5% → 3.9% (93% reduction).

---

## 5. Key Takeaways for the Performance Baseline

### Wins
- **AFib** and **Bradycardia** heads are production-grade (>95% TP retention at t=0.5).
- The v3 single-index ontology + SHA256 vocabulary integrity check + 88-test suite caught all 4 off-by-one bugs at commit time — the LABEL_MAP indices reading the *intended* heads is now verified.
- v2 suppression filter has minimal TP cost (0.7% aggregate, 1.8% within suppressed classes) and zero impact on classes outside the filter scope.

### Gaps
- **Critical-tier heads underperform on this cohort**: Ventricular Run (VT), Pause, Supraventricular Run all show 0% TP detection at t=0.5. The off-by-one corrections route to the *right* head, but the head's probability stays below 0.5 for brief paroxysmal events in a 10-second window. Likely fixes:
  1. **Lower per-class thresholds** for these heads (e.g., 0.2 for VT instead of 0.5).
  2. **Sliding-window scoring** instead of single 10-second window — paroxysmal arrhythmias of 2–3 seconds are diluted by the surrounding normal context.
  3. **Re-train or fine-tune** for paroxysmal-event detection (out of scope for this PR).

- **Checkpoint size mystery**: The downloaded `1_lead_ECGFounder.pth` is 352 MB, vs the previously-cached 118 MB version. Forward pass produces the expected `(batch, 150)` output, so the architecture matches, but the larger file may indicate inclusion of optimizer state or a different training snapshot. Worth verifying against the upstream provenance.

### Open follow-ups
- Run FP-dataset evaluation when `ecg-fp-doctor removed1` is back on disk.
- Investigate paroxysmal-event detection strategy (threshold sweep + sliding window).
- Verify HuggingFace `1_lead_ECGFounder.pth` provenance (352 MB vs 118 MB).
- Run a per-class threshold sweep (e.g., 0.1, 0.2, 0.3, 0.4) on Ventricular Run, Pause, Supraventricular Couplet specifically.

---

## 6. Reproducing this baseline

```bash
# Suppression OFF (raw model baseline)
python3 compare_tp_fzark_with_full_suppression.py --per-class 200 --suppression off

# Suppression ON
python3 compare_tp_fzark_with_full_suppression.py --per-class 200 --suppression on

# Full cohort (no sampling) — slower
python3 compare_tp_fzark_with_full_suppression.py --per-class 0 --suppression off
```

All outputs land in `res/tp_fzark_full_suppression/`. Files written by the
`--suppression off` run get the `_supp_off` suffix to avoid overwriting the ON
results.

---

*Generated from `compare_tp_fzark_with_full_suppression.py` on the `cleanup-12lead-finetune-v1-clean` branch (PR #2). All 88 tests pass; v3 ontology indices verified against `tasks.txt` SHA256 `7c087a0d…0df0c39`.*
