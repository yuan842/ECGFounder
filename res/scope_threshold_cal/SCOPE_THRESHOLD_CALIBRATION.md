# Per-head threshold calibration — scope heads 93 / 98 / 142

> **⚠️ DECISION UPDATE (2026-05-29, later same day): the overrides below were REVERTED to 0.5.**
> The 93→0.040 and 142→0.006 thresholds calibrated here are noise-floor values — fragile and
> device/cohort-specific. `label_config.HEAD_THRESHOLDS` is now `{}` (all 6 scope heads at 0.5).
> Consequence: SV-Run/V-Run/Pause stay effectively silent at single-lead; detecting them needs the
> fine-tuned head, not a low threshold. **This document is retained as the calibration record only.**

**Date**: 2026-05-29
**Why**: heads 93 (SV Run), 98 (V Run/VT), 142 (Pause) were added to `DETECTION_SCOPE` but
have **0 % sensitivity at the default 0.5** — their base-model scores sit in the noise floor.
**Method**: sweep each head's threshold on cached base probs — fzark TP positives vs
fp_doctor matched negatives (seed 42, ≤500/class, same alignment as `combined_binary_eval.py`).
**Script**: [scripts/calibrate_scope_thresholds.py](../../scripts/calibrate_scope_thresholds.py) · sweeps in `res/scope_threshold_cal/sweep_head*.csv`.

## Results

| head | event | n_pos | n_neg | AUROC | score range (pos) | @0.50 sens | Youden thr | sens / spec / PPV @Youden | verdict |
|---|---|---|---|---|---|---|---|---|---|
| **142** | Pause | 82 | 500 | **0.758** | ≤0.030 | 0 % | **0.006** | 0.71 / 0.69 / 0.27 | **calibratable** ✅ |
| 93 | Supraventricular Run | 26 | 500 | 0.606 | ≤0.173 | 0 % | **0.040** | 0.85 / 0.45 / 0.07 | marginal ⚠️ |
| 98 | Ventricular Run (VT) | 500 | 500 | **0.360** | ≤0.196 | 0 % | 0.037 | 0.95 / 0.12 / 0.52 | **worse than chance** ❌ |

All three score far below 0.5 (Pause positives top out at 0.030) — which is exactly why the
0.5 default never fires.

## Decisions (wired into `label_config.HEAD_THRESHOLDS`)

- **142 Pause → 0.006.** AUROC 0.76; the head genuinely ranks pauses above negatives. At 0.006
  it recovers sens 0.71 at spec 0.69 — a defensible operating point for a CRITICAL event.
- **93 SV Run → 0.040.** AUROC 0.61, only marginal. Recall is recoverable (sens 0.85) but PPV is
  0.07 (lots of FP). Included so the head can fire at all; flagged low-confidence.
- **98 VT → NOT overridden (stays 0.5).** AUROC 0.36 is *worse than chance* — fp_doctor negatives
  outscore true V-Run (neg median 0.081 > pos 0.060). No threshold makes head 98 a valid VT
  detector; lowering it just fires on ~88 % of windows. VT should use the fine-tuned / fuzzy head
  instead. At 0.5 it stays silent rather than flooding false positives.

## Caveats

- Thresholds are **device/cohort-specific** (fzark base-model scores) — recalibrate per device,
  exactly like the motion gates.
- Negatives are the per-event fp_doctor windows; head 98 also fires across *other* events, so its
  real-world FP rate at a lowered threshold would be even worse than the spec here.
- Small positive n for SV Run (26) and Pause (82) → wide sampling noise; treat thresholds as a
  starting operating point, not a final spec.
- These are **base-model** thresholds. The fine-tuned heads (DualHead) would shift them.
