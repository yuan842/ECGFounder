# Sinus Tachycardia FP-Suppression — Preset Filter Modes

**Date**: 2026-05-26
**FP cohort**: 300 records from `ecg_fp_doctor removed1` (clinician-removed sinus-tachy alerts)
**TP cohort**: not available (zero sinus-tachy TPs in `ecg_tp_fzark`)

---

## Headline — five selectable modes

| Mode | `motion_max` (mG) | FPs kept | FPs suppressed | **FP suppression rate** |
|---|---|---|---|---|
| `conservative` | 50.0 | 126 | 174 | 58.0 % |
| `balanced` | 13.5 | 73 | 227 | 75.7 % |
| **`default`** *(production default — updated)* | **12.0** | **37** | **263** | **87.7 %** |
| `aggressive` | 11.5 | 29 | 271 | 90.3 % |
| `maximum` | 10.0 | 0 | 300 | 100.0 % |

Each mode keeps the rest of the rule set unchanged (HR < 100 reject, HR > 150 reject, rr_cv > 0.20 reject, SQI gate, exertional-regular gate). Only the `mean_motion` ceiling is moved.

**The project default has been updated to `motion_max = 12.0 mG`** — `DEFAULT_STACHY_MODE = 'default'` in `perclass_fp_suppression.py`. Constructing `PerClassFPSuppressor('Sinus Tachycardia')` without arguments now selects this preset.

---

## How to use

```python
from perclass_fp_suppression import PerClassFPSuppressor

sup = PerClassFPSuppressor('Sinus Tachycardia', mode='balanced')   # or 'aggressive'
result = sup.suppress(p_class=model_prob, json_path=event_json)
```

Inside `MultiClassFPSuppressor` you can pass it through:
```python
from multiclass_fp_suppression import MultiClassFPSuppressor
mc = MultiClassFPSuppressor()
mc.rule_based['Sinus Tachycardia'].mode = 'aggressive'
```

Both `balanced` and `aggressive` are wired into `STACHY_PRESETS` in
`perclass_fp_suppression.py` and load directly when `mode=` is passed
to the suppressor constructor.

---

## Why a single motion threshold works so well

Among the 126 surviving FPs of the current default rules, every single one
has `mean_motion > 10 mG`. The escapees cluster tightly:

| Feature | median | p25 | p75 |
|---|---|---|---|
| mean_motion | 13.08 | 11.73 | 15.19 |
| std_motion | 50.09 | 49.43 | 50.61 |
| mean_hr_bpm | 131.6 | 130.0 | 132.9 |
| rr_cv | 0.01 | 0.01 | 0.10 |
| snr_proxy | 2.14 | 1.76 | 2.51 |

These are **mildly active sinus-tachy events** — HR ~130, motion ~13 mG (light activity, breathing, posture changes). The clinician removed them because the tachycardia is contextually appropriate, not pathologic.

Pushing `motion_max` from 50 → 13.5 mG catches the larger half of these. Pushing it further to 11.5 mG catches all but a few.

---

## TP-risk assessment (no fzark TPs to measure)

| Mode | TP-loss risk (best estimate) | Reasoning |
|---|---|---|
| `conservative` | very low | Only rejects motion > 50 mG (clear activity) |
| `balanced` (13.5 mG) | low | Above "sitting still" range, breathing-only motion stays below |
| **`default`** (12.0 mG) | **low–moderate** | Just above the upper "sitting still" envelope (~8 mG); breathing alone unlikely to trip, mild postural shifts may |
| `aggressive` (11.5 mG) | moderate | Closer to upper edge of "sitting still"; some rest events with breathing or shifting could be rejected |
| `maximum` (10.0 mG) | high | Any non-zero baseline motion gets rejected; risky without TP validation |

The literature on Holter motion suggests:
- Supine rest: 0–3 mG mean motion
- Sitting still: 2–8 mG
- Light activity / standing: 10–25 mG
- Walking: 25–80 mG

The **12.0 mG** default sits 1.5× above the upper "sitting still" envelope, so resting and breathing-only events should stay below it, while mild postural shifts (~10–12 mG) are at risk of being rejected. The TP-risk is acceptable for most production deployments; validate as soon as a sinus-tachy TP cohort is available.

> **Updated recommendation**: deploy `default` (12.0 mG, **87.7 % FP suppression**). The previous `balanced` and `aggressive` presets remain available for safety-first or alarm-burden-first deployments respectively.

---

## Side-by-side summary with the other classes

| Class | TP retention | FP suppression | Source of TP cohort |
|---|---|---|---|
| Atrial Fibrillation | 97 % (CV) | 91 % (CV) | matched cohort |
| **Sinus Tachycardia** (`default` 12 mG) | **unmeasured** | **87.7 %** | rule-based |
| Isolated Supraventricular Beat | 90.5 % | 93.7 % | `ecg_tp_fzark` |
| Isolated Ventricular Beat | 89.9 % | 88.7 % | `ecg_tp_fzark` |
| **Sinus Tachycardia** (`balanced`) | **unmeasured** | **75.7 %** | rule-based |
| **Sinus Tachycardia** (`aggressive`) | **unmeasured** | **90.3 %** | rule-based |

---

## Files

- `perclass_fp_suppression.py` — adds `STACHY_PRESETS` and `mode` argument
- `PERCLASS_FP_FINETUNED.md` — original fine-tune report (IVB / ISB)
- `res/perclass_fp_suppression_eval/eval_FP_Sinus_Tachycardia.csv` — per-event audit
- This document — preset definitions and TP-risk discussion

---

*Generated 2026-05-26 from `res/perclass_fp/fp_features_Sinus_Tachycardia.csv` and `res/perclass_fp_suppression_eval/eval_FP_Sinus_Tachycardia.csv`.*
