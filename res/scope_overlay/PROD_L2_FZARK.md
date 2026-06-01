# Production (L1 fuzzySL + L2 on) — fzark TP & FP evaluation

fzark TP = confirmed events (↑sensitivity good); FP = clinician-removed false alarms (↓retention good). Baseline = raw head @0.5. Production = routed L1 + per-head policy thresholds (PTB-XL-fuzzy-calibrated) + GT-matched L2 (default ON). External validation — fzark is out-of-distribution from the PTB-XL calibration.

## fzark TP — sensitivity (per event, baseline → production+L2)

| event | head | n | base sens | prod+L2 sens | Δ |
|---|---|---:|---:|---:|---:|
| Atrial Fibrillation | 5 | 500 | 0.978 | 0.848 | -0.130 |
| Bradycardia | 4 | 500 | 0.944 | 0.842 | -0.102 |
| Sinus Tachycardia | 6 | 0 | — | — | — (no TP) |
| Supraventricular Run | 93 | 26 | 0.000 | 0.000 | +0.000 |
| Ventricular Run | 98 | 500 | 0.000 | 0.000 | +0.000 |
| Pause | 142 | 82 | 0.000 | 0.000 | +0.000 |

## fzark FP — false-alarm retention (per event, baseline → production+L2; lower better)

| event | head | n | base retention | prod+L2 retention | Δ (reduction) |
|---|---|---:|---:|---:|---:|
| Atrial Fibrillation | 5 | 500 | 0.720 | 0.408 | -0.312 |
| Bradycardia | 4 | 500 | 0.032 | 0.004 | -0.028 |
| Sinus Tachycardia | 6 | 500 | 0.990 | 0.284 | -0.706 |
| Supraventricular Run | 93 | 500 | 0.000 | 0.000 | +0.000 |
| Ventricular Run | 98 | 500 | 0.000 | 0.000 | +0.000 |
| Pause | 142 | 500 | 0.000 | 0.000 | +0.000 |

## Verdict

Two distinct regimes on the production cohort:

**Heads that fire at single-lead (AFib, Brady, Sinus-Tachy) — large false-alarm reductions at a moderate sensitivity cost:**

| event | TP sens (base→prod) | FP retention (base→prod) | net |
|---|---|---|---|
| **Sinus-Tachy** | — (no TP) | **0.990 → 0.284** (−71 pp) | massive false-alarm cut |
| **AFib** | 0.978 → 0.848 (−13 pp) | **0.720 → 0.408** (−31 pp) | ~43% fewer false AF alarms |
| **Bradycardia** | 0.944 → 0.842 (−10 pp) | 0.032 → 0.004 (−3 pp) | already-clean, cleaner |

This is exactly the policy intent — **spec/false-alarm improvement traded for bounded sensitivity loss** — now validated on the real TP/FP cohort. The wins are big (Sinus-Tachy false alarms cut to ~1/4, AFib to ~4/7), driven by the raised per-head thresholds + L2 (NSR-contradiction + AFib▸Tachy exclusion).

**Heads silent at single-lead (SV-Run 93, V-Run 98, Pause 142) — 0 sensitivity AND 0 false alarms:**
- At 0.5 the base heads never fire on fzark (consistent with the CLAUDE global rule: 93/98/142 are effectively silent at single-lead). L1 can't help — it was trained on PTB-XL where these heads also barely fire, and they route to base. So production neither detects nor false-alarms on them. **Detecting runs/pause needs the fine-tuned head, not threshold/L2 tuning.**

## Caveats
- **Sensitivity loss exceeds the 5 pp budget** (AFib −13, Brady −10) because the thresholds were calibrated on **PTB-XL fuzzy**, and **fzark is out-of-distribution** (ambulatory patch vs derived 12-lead). To restore the 5 pp budget on fzark, recalibrate per-head thresholds on an fzark validation split (`calibrate_l1_policy` adapted to fzark) — the FP reductions would shrink somewhat but sensitivity would recover.
- TP Sinus-Tachy is unmeasurable (fzark TP has 0 Sinus-Tachy events); only its FP retention is shown.
- Cohort = seed-fixed sample, ≤500 records per (cohort, event); backbone run fresh + cached under res/scope_overlay/feat_fzark/.
