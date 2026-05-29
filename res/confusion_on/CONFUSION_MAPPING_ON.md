# Baseline Model — FP-algo ON — Accuracy / FP Analysis + Confusion Mapping

**Date**: 2026-05-29
**Model**: base ECGFounder (`1_lead_ECGFounder.pth`), single-lead, **FP suppression ON** (v2 motion/HR gates).
**Detection scope**: `DETECTION_SCOPE` = **{4, 5, 6, 93, 98, 142}** · per-head thresholds (0.5 except 142→0.006, 93→0.040).
**Cohorts**: `ecg_tp_fzark` (positives) vs `ecg_fp_doctor removed1` (negatives), stratified seed 42, ≤500/class.
**Script**: [scripts/confusion_mapping_on.py](../../scripts/confusion_mapping_on.py) · CSVs in `res/confusion_on/`.
**Counterpart**: the OFF report is `res/confusion_off/CONFUSION_MAPPING_OFF.md`.

## What suppression actually touches

Within the scope the only **in-scope active rules** are:
- **AFib (head 5)** — motion gate: keep iff `mean_motion ≤ 5 mG`
- **Bradycardia (head 4)** — HR gate: keep iff `mean_hr_bpm ≤ 56.3`

Heads 6/93/98/142 have no rule → **ON ≡ OFF for them**. So the entire ON-vs-OFF difference lives in the AFib and Bradycardia columns. (SV-Trig/V-Trig rules exist in v2 but are out of scope → suppressor passes them through.)

---

## 1. Per-event accuracy / FP metrics — OFF → ON

| event | head | sens | spec | PPV | acc | F1 | AUROC |
|---|---|---|---|---|---|---|---|
| **Atrial Fibrillation** | 5 | 0.978→**0.974** | 0.280→**1.000** | 0.576→**1.000** | 0.629→**0.987** | 0.725→**0.987** | 0.890→**0.995** |
| **Bradycardia** | 4 | 0.944→0.918 | 0.968→0.982 | 0.967→0.981 | 0.956→0.950 | 0.955→0.948 | 0.974→0.952 |
| Pause (thr 0.006) | 142 | 0.732 | 0.652 | 0.256 | 0.663 | 0.380 | 0.758 | *(no rule — unchanged)* |
| Supraventricular Run (thr 0.040) | 93 | 0.808 | 0.470 | 0.073 | 0.487 | 0.135 | 0.606 | *(unchanged)* |
| Ventricular Run | 98 | 0.000 | 1.000 | 0.000 | 0.500 | 0.000 | 0.360 | *(unchanged)* |
| Sinus Tachycardia | 6 | — | 0.010 | — | — | — | — | *(n_pos=0)* |
| **OVERALL (micro)** | | 0.648→**0.639** | 0.674→**0.821** | 0.561→**0.696** | 0.664→**0.750** | 0.601→**0.666** | 0.768→**0.828** |

- **AFib is the win**: the motion gate removes **all 360 FP (→ 0)** at a cost of just **2 TP** (489→487). Specificity 0.28→1.00, PPV 0.58→1.00, accuracy 0.63→0.99, AUROC 0.89→0.995.
- **Bradycardia**: HR gate trims FP 16→9 (spec 0.968→0.982) but also drops 13 TP (sens 0.944→0.918) — a mild, expected trade.
- **Overall**: specificity 0.67→**0.82**, PPV 0.56→**0.70**, accuracy 0.66→**0.75**, AUROC 0.77→**0.83**; sensitivity barely moves (0.648→0.639). Suppression buys large precision for negligible recall loss.

---

## 2. Confusion mapping — single-label argmax (TP cohort), OFF → ON

| true event ↓ | OFF prediction | ON prediction |
|---|---|---|
| **Atrial Fibrillation** (500) | 475 AFib | 477 AFib (clean) |
| **Bradycardia** (500) | 464 Brady | 459 Brady (clean) |
| **Ventricular Run** (500) | **491 → AFib** | **0 AFib** → 233 Pause(142) / 212 SVT(93) / 28 NONE |
| **Pause** (82) | 58 Brady | 40 Brady / 12 Pause / 20 SVT |
| Supraventricular Run (26) | 15 AFib | scattered (10 SVT / 10 Pause / 1 AFib) |

**The biggest confusion change**: true **Ventricular Run no longer reads as AFib** — V-Run windows are high-motion, so the AFib motion gate suppresses them (AFib argmax wins 491→**0**). The catch: with AFib gone, the argmax falls onto the low-threshold noise-floor heads (Pause 142 / SVT 93), so V-Run is still mis-mapped — just to a different wrong head. The VT head (98) remains silent throughout.

---

## 3. Multi-label co-firing (TP) — OFF → ON (AFib/Brady columns only change)

% of each true event's windows firing each scope head. Own head **bold**; values shown OFF→ON where they differ.

| true event ↓ | Brady(4) | AFib(5) | STach(6) | SVT(93) | VT(98) | Pause(142) |
|---|---|---|---|---|---|---|
| Atrial Fibrillation | 7.0→3.8 | **97.8→97.4** | 34.4 | 62.0 | 0.0 | 78.0 |
| Bradycardia | **94.4→91.8** | 14.2→2.6 | 0.0 | 3.0 | 0.0 | 44.6 |
| Ventricular Run | 10.8→3.8 | **99.2→0.0** | 2.2 | 69.8 | 0.0 | 90.8 |
| Pause | **72.0→48.8** | 34.1→0.0 | 8.5 | 47.6 | 0.0 | **73.2** |

- **True AFib retains its head (97.8→97.4)** — real AFib is low-motion, so the gate keeps it. Meanwhile the AFib head's *promiscuous* firing collapses on other rhythms: V-Run 99.2→**0.0**, Pause 34.1→**0.0**, Bradycardia 14.2→2.6. The motion gate is doing exactly its job — killing motion-driven AFib false fires while sparing true resting AFib.
- The uncalibrated low-threshold heads (93, 142) are unaffected (no rule) and still fire promiscuously.

---

## 4. False-positive analysis — argmax on confirmed negatives, OFF → ON

| predicted on NEGATIVE windows (n=6,541) | OFF | ON |
|---|---|---|
| **Atrial Fibrillation** | 3,096 (47.3 %) | **0 (0 %)** |
| Sinus Tachycardia | 1,409 | 2,270 |
| Supraventricular Tachycardia (93) | 418 | 1,411 |
| WITH SINUS PAUSE (142) | 654 | 1,146 |
| Bradycardia | 280 | 232 |
| **NONE (correctly silent)** | 684 (10.5 %) | **1,482 (22.7 %)** |

- **The AFib motion gate eliminates AFib as a false-positive label entirely** (3,096 → 0 argmax wins) and **doubles the correctly-silent fraction** (10.5 % → 22.7 %).
- But the FP "mass" partly **redistributes** to the uncalibrated heads: with AFib suppressed, Sinus Tachycardia (no gate) and the low-threshold SVT/Pause heads now win more negative-window argmaxes. Net specificity still improves (overall 0.67→0.82), but it highlights that **Sinus Tachycardia and the 93/142 thresholds are the next FP targets** once AFib is handled.

---

## Key findings

1. **AFib motion suppression is the headline**: 360→0 FP for −2 TP → spec 0.28→1.00, PPV 0.58→1.00, AUROC 0.89→0.995. Bradycardia HR gate gives a smaller, favorable trade (spec +0.014, sens −0.026).
2. **Overall: spec 0.67→0.82, PPV 0.56→0.70, acc 0.66→0.75, AUROC 0.77→0.83**, sensitivity essentially flat (0.648→0.639).
3. **Confusion improves where it's gated**: V-Run→AFib (491) disappears; true AFib keeps its head (low motion). But V-Run then falls onto the noise-floor heads (Pause/SVT), and the VT head stays silent — confirming VT needs the fine-tuned head, not suppression.
4. **FP burden shifts, not just shrinks**: AFib FP→0 and silent-rate doubles, but Sinus Tachycardia (un-gated) and the low-threshold 93/142 heads absorb some redistributed mass — the next-priority suppression/threshold targets.

## Caveats
- Suppression is **motion/HR-based and device-specific** (fzark mG units, 5 mG / 56.3 bpm gates). Other heads (Pause/SV-Run/VT/Sinus-Tachy) have no rule in scope.
- Single stratified sample (seed 42); small-n events (SV-Run 26, Pause 82) noisy; Sinus Tachycardia has no TP windows.
- Per-event OFF↔ON numbers match `res/cross_dataset_supp_off/combined_binary_metrics_scope.csv`.
