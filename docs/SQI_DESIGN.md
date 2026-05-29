# Signal Quality Index (SQI) — Integration Design

**Status**: design (this doc) → prototype on MOVE (validation) → integrate into `preprocessing.py`.
**Scope**: add a comprehensive, per-window SQI panel to the ECG pipeline without disturbing the model-input path.
**Companion code today**: scattered quality functions in [ecg_feature_analysis.py](../ecg_feature_analysis.py) (`sqi_features`, `sample_entropy`, `detect_r_peaks`, `bandpass`) and [compare_tp_fp_sqi_motion.py](../compare_tp_fp_sqi_motion.py).

---

## 1. Why & what

The current pipeline ([preprocessing.py](../preprocessing.py) `ECGPreprocessor.process()`) returns only a `(1, 5000)` model-ready tensor. It produces **no quality signal**, so downstream code (FP suppression, demo, MOVE motion analysis) cannot tell a clean strip from a motion-corrupted one except via the separate, ad-hoc functions in `ecg_feature_analysis.py`.

Goal: a single **comprehensive SQI panel** computed per window, returned alongside the tensor, consolidating the existing scattered metrics + a composite score, so any consumer can gate on quality uniformly.

---

## 2. The one non-negotiable constraint: tap the signal at the CORRECT stage

`process()` is a destructive 7-stage pipeline (select → notch → bandpass → baseline-removal → resample → crop → winsorize+zscore). **Computing SQI on the final tensor is wrong** — the pipeline erases the very artifacts SQI must measure:

| SQI | If computed on final tensor | Correct tap |
|---|---|---|
| `clip_pct` / saturation | winsorize already clipped extremes → reads ~0 | **RAW** (pre-filter) |
| `flat_pct` / lead-off | bandpass smooths flat regions → understated | **RAW** |
| `dynamic_range_mv`, amplitude | z-score destroyed absolute mV | **RAW** |
| `baseline_drift` | stage-4 removed the baseline → ~0 by construction | **RAW** (pre baseline-removal) |
| `powerline_residual` | notch already removed 50/60 Hz | **RAW** (measures notch necessity) |
| `snr_proxy` (QRS band ratio) | defined on the QRS band; bandpass-OK | **post-bandpass** |
| `kurt`, `sample_entropy` | z-score is scale-invariant → survive | either; cleanest **pre-normalize** |
| R-peak / HR / RR metrics | need clean QRS | **post-bandpass** |

**Rule**: amplitude/saturation/flatline/baseline/powerline SQIs run on the RAW lead; morphology/SNR/rhythm SQIs run on the filtered (post-bandpass) signal. Nothing meaningful is computed on the winsorized+z-scored output.

---

## 3. The SQI panel ("comprehensive")

Per window. ★ = already implemented in `ecg_feature_analysis.py`; ○ = new.

| Category | Metric | Tap | Status |
|---|---|---|---|
| **Amplitude / saturation** | `clip_pct` ★ — % samples at extremes | RAW | ★ |
| | `flat_pct` ★ — % flat-line (rolling 100 ms std < ε) | RAW | ★ |
| | `dynamic_range_mv` ○ — p99.5 − p0.5 in mV | RAW | ○ |
| **Noise / SNR** | `snr_proxy` ★ — var(5–25 Hz) / var(residual) | filtered | ★ |
| | `hf_noise_ratio` ○ — power >40 Hz / total | RAW | ○ |
| | `powerline_residual` ○ — power in ±1 Hz of 50/60 / total | RAW | ○ |
| **Baseline** | `baseline_drift` ★ — std of <0.5 Hz envelope | RAW | ★ |
| **Morphology** | `kurt` ★ — excess kurtosis (R-peak peakedness) | filtered | ★ |
| | `sample_entropy` ★ — complexity / noise | filtered | ★ |
| | `qrs_template_corr` ○ — mean beat-to-median-template correlation | filtered | ○ |
| **Rhythm plausibility** | `n_peaks`, `mean_hr_bpm`, `rr_cv` ★ (via `detect_r_peaks` + `rr_irregularity_features`) | filtered | ★ |
| | `pct_physiologic_hr` ○ — HR in [40, 200] bpm | filtered | ○ |
| **Motion (multi-sensor only)** | `chest_motion_mg` ★ (MOVE: already in segmentation metadata) | ACC channel | ★ |
| **Composite** | `sqi_score ∈ [0,1]` ○, `sqi_class ∈ {good, acceptable, poor}` ○ | derived | ○ |

~70% already exist; the work is consolidation + the ○ additions + the composite.

---

## 4. Composite score & class

Downstream code shouldn't reason over 14 raw metrics — it gates on one number + one label.

```
sqi_score = weighted, clamped blend of normalized sub-metrics, e.g.
    0.30 * f(snr_proxy)        # higher better
  + 0.20 * (1 - clip_pct/100)  # lower clip better
  + 0.20 * (1 - flat_pct/100)  # lower flat better
  + 0.15 * g(kurt)             # moderate-high better
  + 0.15 * h(pct_physiologic_hr)
  → clamp to [0, 1]

sqi_class = good        if sqi_score ≥ 0.7
            acceptable  if 0.4 ≤ sqi_score < 0.7
            poor        if sqi_score < 0.4
```

Exact weights/thresholds are **calibrated empirically** — that's what the MOVE validation (§6) is for: thresholds should separate `baseline` (clean rest) from `run` (motion-corrupted) windows. Weights are config, not hard-coded magic numbers.

---

## 5. API design — quality-aware `process()` (backward-compatible)

**Option A (chosen): optional return.**

```python
def process(self, signal, fs_in, source_leads=None, *, return_sqi=False):
    x_raw = self._select_lead(signal, source_leads)         # tap: RAW
    sqi = compute_raw_sqi(x_raw, fs_in) if return_sqi else None
    x = self._notch(x_raw); x = self._bandpass(x)           # tap: filtered
    if return_sqi:
        sqi.update(compute_band_sqi(x, fs_in))
        sqi.update(score_and_class(sqi))                    # composite
    x = self._baseline(x); x = self._resample(x, fs_in)
    x = self._crop_or_pad(x); x = self._normalize(x)
    tensor = torch.from_numpy(x.astype(np.float32))[None, :]
    return (tensor, sqi) if return_sqi else tensor
```

- Default `return_sqi=False` → **signature & behavior unchanged** for every existing caller (fzark/PTB-XL/MIT-BIH evals untouched).
- One pass; each metric tapped at the right stage.
- A new module `sqi.py` holds `compute_raw_sqi`, `compute_band_sqi`, `score_and_class` — consolidating the functions currently scattered in `ecg_feature_analysis.py` (which becomes a thin re-export for back-compat).

**Rejected — Option B (standalone `quality(raw)` method)**: cleaner separation but re-runs notch/bandpass. Kept as a fallback only if a caller needs SQI without the tensor.

---

## 6. MOVE validation plan (the prototype — implemented next)

MOVE is the ideal proving ground: the segmentation step already saved **raw** ECG windows + a **time-synchronized chest-ACC channel** + ground-truth **activity labels**. So we can test whether SQI actually measures quality, against an independent truth:

**Hypotheses** (if SQI works, these must hold):
1. `snr_proxy` and `sqi_score` are **highest during `baseline`** (still rest), **lowest during `run`** (motion).
2. `clip_pct`, `flat_pct`, `hf_noise_ratio`, `baseline_drift` **rise during `run`/`walk`**.
3. ECG-intrinsic SQI **correlates with the independent `chest_motion_mg`** (more motion → worse SQI) — cross-sensor confirmation the SQI isn't measuring noise of its own invention.

**Method** (prototype `scripts/move/prototype_sqi.py`):
- Load `move_window_metadata.csv`, exclude padded windows (`ecg_pad_fraction == 0`).
- Compute the SQI panel on each raw ECG `.npy` window (correct-stage taps).
- Group by `activity_label`; report each SQI's mean ± std per activity.
- Correlate each SQI against `chest_motion_mg`.
- Verdict: does quality monotonically degrade rest → motion?

If the hypotheses hold, the composite weighting/thresholds in §4 get calibrated from these distributions and SQI is promoted into `preprocessing.py`. If they don't, the metric set / taps get revised before any integration.

---

## 7. Integration sequence (after validation passes)

1. Create `sqi.py` consolidating the panel (raw + band + composite).
2. Add `return_sqi` to `ECGPreprocessor.process()` (Option A).
3. Calibrate composite weights/thresholds from the MOVE distributions.
4. Re-export the old `ecg_feature_analysis.sqi_features` from `sqi.py` (back-compat; add deprecation note).
5. Optional: a `--with-sqi` flag on the MOVE preprocessing step to write a `sqi.csv` joined to `move_window_metadata.csv`.

No change to the model-input tensor at any step — SQI is purely additive metadata.

---

## 8. Open questions

1. **Composite weights** — calibrate from MOVE (§6) or adopt a published SQI scheme (e.g. bSQI/kSQI/pSQI from Clifford et al.)? Recommend: start data-driven from MOVE, cross-check against literature.
2. **R-peak detector robustness** — `detect_r_peaks` is a simple find_peaks; under heavy `run` motion it may over/under-detect, which is itself a quality signal but complicates `mean_hr_bpm`. Flag low-confidence HR rather than trusting it.
3. **Per-modality SQI** — this doc covers ECG SQI. PPG has its own SQI family (template-matching, perfusion index); out of scope for v1, notable for the cross-modality work.
4. **Where SQI gates** — does a `poor` window get dropped, flagged-and-kept, or down-weighted? A pipeline policy decision, not an SQI-computation one; defer to the consuming task.

---

## 9. MOVE validation findings (prototype run, 2026-05-29)

Ran [scripts/move/prototype_sqi.py](../scripts/move/prototype_sqi.py) on 5,158 non-padded MOVE windows. **The prototype works — SQI is strongly discriminative and cross-sensor-correlated — but it overturned the naive hypothesis and exposed two metric bugs and one architectural conclusion.**

### Mean SQI by activity (selected)

| activity | sqi_score | snr_proxy | clip_pct | flat_pct | dyn_range_mv | chest_motion_mg |
|---|---|---|---|---|---|---|
| baseline (rest) | **0.312** | 0.288 | 74.7 | 74.3 | **0.70** | 1271 |
| walk_before | 0.537 | 0.801 | 41.8 | 41.0 | 1.50 | 923 |
| run | **0.578** | 0.886 | 27.4 | 25.3 | 2.44 | 761 |
| walk_after | **0.654** | 1.057 | 8.2 | 6.5 | 2.88 | 506 |

### H1 — REJECTED (and that's the finding)

The naive hypothesis "rest = cleanest, run = noisiest" is **false for this wearable data** (analyzed channel: `ecg:gel`, the gel electrode). `baseline` scores the *lowest* quality (sqi 0.31, flat/clip ≈ 74 %, dyn-range 0.70 mV) and `run`/`walk_after` score *highest*. Why:

- At rest, the **gel-electrode ECG is low-amplitude / often flat** (poor skin contact before the subject moves; dyn-range 0.70 mV). The flat-detector correctly fires. (This is the clinical-grade *gel* electrode — not the dry one — which makes the rest-contact failure more striking.)
- During exercise, **motion artifact ADDS amplitude and 5–25 Hz energy** (dyn-range 2.4–2.9 mV), which `snr_proxy` (var(5–25 Hz)/var(residual)) mistakes for *signal*. So the motion-corrupted windows look "cleaner" by an ECG-intrinsic SNR.

### H3 — CONFIRMED (strongly)

SQI correlates with the independent chest-motion channel: `corr(motion, clip_pct)=+0.95`, `corr(motion, sqi_score)=−0.84`. The SQI is measuring something real and cross-sensor-consistent — it just measures *amplitude/liveness*, not *cleanliness*, when used on ECG alone.

### Three concrete outcomes for the integration (before touching `preprocessing.py`)

1. **`clip_pct` mis-fires on flat signals** — when a window is mostly flat, the few non-flat samples become percentile "extremes" so clip ≈ flat (both ≈74 % at baseline). **Fix**: gate `clip_pct` on `dynamic_range_mv`; report 0 clip when the signal is flat. Decouple the two metrics.
2. **Motion proxy must be `std(|acc|)`, not `mean(|acc|)`** — `chest_motion_mg` (mean magnitude) is gravity-dominated and *inversely* tracked activity (baseline 1271 > run 761), which is a chest-orientation artifact, not motion intensity. **Fix**: the segmentation metadata's motion field should be the std of the gravity-removed magnitude. (This is a fix to the segmentation `chest_motion_mg`, applied where motion is used for gating.)
3. **ECG-intrinsic SQI alone is fooled by motion → the composite MUST fuse the synchronized ACC channel.** This is the architectural finding: a single-lead SNR cannot separate a big QRS from a big motion artifact. The whole reason we time-synchronized the ACC channel in segmentation now pays off — `sqi_score` for wearable data must include a motion-penalty term from the independent accelerometer, not just ECG-band ratios. For clinical resting ECG (PTB-XL etc., no ACC) the ECG-intrinsic panel is fine; for ambulatory/wearable (MOVE, fzark) the motion-fused composite is required.

### Net verdict

The prototype validated the *machinery* (compute, taps, cross-sensor join all work) and **invalidated the naive composite weighting** — exactly what a prototype is for. Before promotion into `preprocessing.py`:
- fix metrics (1) and (2),
- split the composite into an **ECG-intrinsic `sqi_score`** (clinical resting data) and a **motion-fused `sqi_score_amb`** (wearable data, penalized by `std(|acc|)`),
- recalibrate thresholds against these distributions.

Artifacts: [res/move_segmented/move_sqi_prototype.csv](../res/move_segmented/move_sqi_prototype.csv) (per-window SQI panel, joined to activity + motion).

---

## 10. Fixes implemented — [sqi.py](../sqi.py) (2026-05-29)

All three issues from §9 are resolved in the consolidated `sqi.py` module. **Bug #2's fix was corrected** after verification (the original "use std" was a mis-diagnosis):

| Fix | Implemented as | Post-fix MOVE result |
|---|---|---|
| **#1 — clip ≠ flat** | `compute_raw_sqi`: clip is **rail-sticking, run-length ≥ 3, flat-gated** (returns 0 when `dynamic_range < 0.05 mV`); isolated R-peak tips excluded | baseline `clip_pct` **74 % → 4.3 %** while `flat_pct` stays 74 % — fully decoupled |
| **#2 — motion (CORRECTED)** | `gravity_removed_motion`: `mean(\|vm − moving_avg(vm)\|)·1000` — **byte-for-byte the production `multiclass_fp_suppression.extract_features` definition**, NOT std-of-magnitude | baseline **0.49 → run 1.81 mG** (rises with exertion, on the 5/15/24 mG gate scale). Also wired into `segment_move.py` so the segmentation metadata's `chest_motion_mg` is now production-comparable. **Does NOT touch the FP algo** (separate code path / data). |
| **#3 — dual composite** | `sqi_score_ecg` (intrinsic, clinical resting data) + `sqi_score_amb` (motion-penalized, wearable data) + `sqi_class` | both present; near-equal on MOVE (motion <2 mG → ~2 % penalty, correct) — they diverge on high-motion cohorts like fzark FPs |

**Why bug #2's original "std" diagnosis was wrong**: `std(\|acc\|)` is high at baseline (12.9 mG, electrode-settling noise) and does *not* cleanly track activity. The production gravity-removed **mean** is the correct, FP-gate-comparable metric. Verified empirically before changing anything.

**Validation note**: MOVE's `baseline` (rest) remaining the *lowest*-SQI phase is a genuine gel-electrode-contact finding (`ecg:gel`, 74 % flat, 0.70 mV range), not a metric bug — SQI correctly flags it.

**Module API**: `compute_sqi(sig, fs, acc_xyz=None) → panel dict`. Pass `acc_xyz` for the motion-fused composite (wearable); omit for clinical resting ECG. Stage discipline enforced (raw-stage vs filtered-stage taps). `scripts/move/prototype_sqi.py` now imports this module (single source of truth).

**Still pending** (not requested in this round): wire `return_sqi=True` into `ECGPreprocessor.process()` per §5, and empirically calibrate the composite weights/thresholds. The metric set + taps are now validated and frozen.

---

*Design + MOVE validation + fixes (#1, #2-corrected, #3) complete in `sqi.py`. The `process()` integration (§5) remains the only un-done step, deferred until needed.*
