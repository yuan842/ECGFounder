# Scope-limited re-evaluation (DETECTION_SCOPE + per-head thresholds)

**Date**: 2026-05-29
**What**: the eval scripts and FP suppressor now honour `label_config.DETECTION_SCOPE`
({4,5,6,93,98,142}) and `head_threshold()` (0.5 except 142→0.006, 93→0.040) instead of
iterating all of `FZARK_LABEL_MAP` at a hard-coded 0.5.

- `scripts/combined_binary_eval.py` → `res/cross_dataset_supp_off/combined_binary_metrics_scope.csv`
- `scripts/confusion_mapping_off.py` → `res/confusion_off_scope/` (full 13-event version preserved in `res/confusion_off/`)
- `multiclass_fp_suppression`: `suppress_alert` returns `out_of_scope` passthrough for
  events whose head ∉ DETECTION_SCOPE; `supported_classes()` → `['Atrial Fibrillation','Bradycardia']`.

## Combined binary metrics — 6 in-scope events (base, t = per-head)

| event | head | thr | OFF sens/spec/ppv | ON sens/spec/ppv |
|---|---|---|---|---|
| Atrial Fibrillation | 5 | 0.5 | 0.978 / 0.280 / 0.576 | **0.974 / 1.000 / 1.000** |
| Bradycardia | 4 | 0.5 | 0.944 / 0.968 / 0.967 | 0.918 / 0.982 / 0.981 |
| Pause | 142 | 0.006 | 0.732 / 0.652 / 0.256 | = OFF (no rule) |
| Supraventricular Run | 93 | 0.040 | 0.808 / 0.470 / 0.073 | = OFF (no rule) |
| Ventricular Run | 98 | 0.5 | **0.000** / 1.000 / 0.000 | = OFF |
| Sinus Tachycardia | 6 | 0.5 | n_pos=0 (no TP in cohort) | — |
| **OVERALL (micro)** | | | **0.648 / 0.674 / 0.561 / acc 0.664** | **0.639 / 0.821 / 0.696 / acc 0.750** |

- **FP suppression now only fires on in-scope active heads** (AFib, Bradycardia). AFib ON:
  spec 0.28→1.00, PPV 0.58→1.00. Overall ON: spec 0.67→0.82, PPV 0.56→0.70, acc 0.66→0.75.
- **Pause is now detectable** at its 0.006 threshold (sens 0.73) — it was 0% at 0.5.
- **VT (98) stays silent** (0.5, not overridden) — by design; its base head is worse-than-chance.

## Confusion (scope heads, per-head thresholds)

The low calibrated thresholds make heads 93/142 fire **promiscuously** — e.g. WITH SINUS
PAUSE(142) co-fires on 78% of AFib, 91% of V-Run, 96% of SV-Run windows; SVT(93) on 62% of
AFib. This is the visible cost of recovering their recall (PPV 0.07–0.26) and is why the
argmax confusion still routes most windows to AFib/Bradycardia (those heads clear their 0.5
threshold by a large margin, whereas 93/142 barely clear their noise-floor thresholds).

VT(98) column is all-zero (never fires at 0.5), confirming it needs the fine-tuned head, not
a threshold.

## Caveat
These scope/threshold values are base-model + fzark/fp_doctor specific; recalibrate per device.
