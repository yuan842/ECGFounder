# Baseline Model — FP-algo OFF — Accuracy / FP Analysis + Confusion Mapping

**Date**: 2026-05-29 (updated for `DETECTION_SCOPE` + per-head thresholds)
**Model**: base ECGFounder (`1_lead_ECGFounder.pth`), single-lead, **FP suppression OFF** (raw sigmoid heads).
**Detection scope**: `label_config.DETECTION_SCOPE` = **{4, 5, 6, 93, 98, 142}** — six in-scope heads only.
**Thresholds**: per-head (`head_threshold`) — **0.5** everywhere except calibrated scope heads **142 → 0.006**, **93 → 0.040** (see `res/scope_threshold_cal/`).
**Cohorts** (stratified `random_state=42`, ≤500/class):
- **Positives** = `ecg_tp_fzark` · **Negatives** = `ecg_fp_doctor removed1`

**Script**: [scripts/confusion_mapping_off.py](../../scripts/confusion_mapping_off.py) · scope CSVs in `res/confusion_off_scope/`.
> The pre-scope, full-13-event-at-0.5 analysis (and its CSVs) is preserved in this same folder's history / `res/confusion_off/` CSVs; this report now reflects the **scope-limited** configuration.

---

## Scope heads (tasks.txt head → fzark event)

| idx | head (tasks.txt) | fzark event | thr |
|---|---|---|---|
| 4 | SINUS BRADYCARDIA | Bradycardia | 0.5 |
| 5 | ATRIAL FIBRILLATION | Atrial Fibrillation | 0.5 |
| 6 | SINUS TACHYCARDIA | Sinus Tachycardia | 0.5 |
| 93 | SUPRAVENTRICULAR TACHYCARDIA | Supraventricular Run | **0.040** |
| 98 | VENTRICULAR TACHYCARDIA | Ventricular Run | 0.5 (not overridden) |
| 142 | WITH SINUS PAUSE | Pause | **0.006** |

Out-of-scope heads (PVC 9, PAC 16, SV-Couplet 19, ST-Elev 68, …) are no longer evaluated or reported.

---

## 1. Per-event accuracy / FP metrics (base, FP-OFF, per-head thresholds)

| event | head | n_pos | n_neg | sens | spec | PPV | NPV | acc | F1 | AUROC | PR-AUC |
|---|---|---|---|---|---|---|---|---|---|---|---|
| **Atrial Fibrillation** | 5 | 500 | 500 | **0.978** | 0.280 | 0.576 | 0.927 | 0.629 | 0.725 | 0.890 | 0.878 |
| **Bradycardia** | 4 | 500 | 500 | 0.944 | **0.968** | **0.967** | 0.945 | **0.956** | **0.955** | **0.974** | 0.976 |
| **Pause** (thr 0.006) | 142 | 82 | 500 | 0.732 | 0.652 | 0.256 | 0.937 | 0.663 | 0.380 | 0.758 | 0.335 |
| Supraventricular Run (thr 0.040) | 93 | 26 | 500 | 0.808 | 0.470 | 0.073 | 0.979 | 0.487 | 0.135 | 0.606 | 0.063 |
| **Ventricular Run** | 98 | 500 | 500 | **0.000** | 1.000 | 0.000 | 0.500 | 0.500 | 0.000 | 0.360 | 0.392 |
| Sinus Tachycardia | 6 | 0 | 500 | — | 0.010 | — | — | — | — | — | — |
| **OVERALL (micro)** | | **1608** | **2500** | **0.648** | **0.674** | **0.561** | 0.749 | **0.664** | 0.601 | 0.768 | 0.767 |

Reading by head:
- **AFib (5)** — high recall 0.98, poor specificity 0.28 (PPV 0.58): the prime FP-suppression target.
- **Bradycardia (4)** — production-grade unaided (acc 0.96, AUROC 0.97).
- **Pause (142)** — calibrated threshold **0.006 recovers it from 0% to sens 0.73** (spec 0.65); AUROC 0.76 means the head genuinely ranks pauses, it just lives in the noise floor.
- **SV Run (93)** — threshold 0.040 gives sens 0.81 but PPV **0.073** (marginal / low-confidence).
- **VT (98)** — **0% sensitivity, left at 0.5 by design.** AUROC **0.360** is *worse than chance* (fp_doctor negatives outscore true V-Run); no threshold yields a valid detector — needs the fine-tuned head.
- **Sinus Tachycardia (6)** — no TP windows in the cohort (FP-side only).
- Micro overall: **sens 0.648 / spec 0.674 / acc 0.664 / AUROC 0.768.**

---

## 2. Confusion mapping — single-label argmax (TP cohort)

Predicted = the in-scope head with the largest margin over **its own** threshold, else **NONE**. (Margin-based, since heads use different thresholds.)

| true event ↓ \ predicted → | Brady(4) | AFib(5) | STach(6) | SVT(93) | VT(98) | Pause(142) | NONE |
|---|---|---|---|---|---|---|---|
| **Atrial Fibrillation** (500) | 9 | **475** | 11 | 1 | 0 | 2 | 2 |
| **Bradycardia** (500) | **464** | 26 | 0 | 3 | 0 | 2 | 5 |
| **Ventricular Run** (500) | 5 | **491** | 1 | 1 | **0** | 1 | 1 |
| Supraventricular Run (26) | 3 | 15 | 3 | 3 | 0 | 2 | 0 |
| **Pause** (82) | **58** | 10 | 4 | 5 | 0 | 4 | 1 |

- AFib (475/500) and Bradycardia (464/500) are clean.
- **Ventricular Run → 491 AFib** — VT is read as AFib (head 98 never wins; column is all-zero).
- **Pause → 58 Bradycardia** — pauses collapse onto the slow-rate head; the Pause head wins argmax only 4× despite firing on 73% of windows (its 0.006-margin is tiny vs Bradycardia's 0.5-margin).

---

## 3. Multi-label co-firing (TP) — the cost of the low thresholds

% of each **true** event's windows firing each scope head (multi-label → rows can exceed 100%). Own head **bold**.

| true event ↓ | Brady(4) | AFib(5) | STach(6) | SVT(93) | VT(98) | Pause(142) | none |
|---|---|---|---|---|---|---|---|
| Atrial Fibrillation | 7.0 | **97.8** | 34.4 | 62.0 | 0.0 | 78.0 | 0.4 |
| Bradycardia | **94.4** | 14.2 | 0.0 | 3.0 | 0.0 | 44.6 | 1.0 |
| Ventricular Run | 10.8 | 99.2 | 2.2 | 69.8 | **0.0** | 90.8 | 0.2 |
| Supraventricular Run | 11.5 | 65.4 | 23.1 | **80.8** | 0.0 | 96.2 | 0.0 |
| Pause | 72.0 | 34.1 | 8.5 | 47.6 | 0.0 | **73.2** | 1.2 |

**The low calibrated thresholds make heads 93 and 142 fire promiscuously**: WITH SINUS PAUSE(142) co-fires on 78 % of AFib, 91 % of V-Run, 96 % of SV-Run windows; SVT(93) on 62 % of AFib. This is the visible cost of recovering their recall (PPV 0.07–0.26) and why argmax (§2) still routes most windows to AFib/Bradycardia. VT(98) is silent everywhere.

---

## 4. False-positive analysis — what fires on confirmed negatives

Argmax over the all-negative pool (6,541 clinician-removed windows):

| predicted on NEGATIVE windows | count | share |
|---|---|---|
| **Atrial Fibrillation** | **3,096** | **47.3 %** |
| Sinus Tachycardia | 1,409 | 21.5 % |
| WITH SINUS PAUSE (142) | 654 | 10.0 % |
| Supraventricular Tachycardia (93) | 418 | 6.4 % |
| Bradycardia | 280 | 4.3 % |
| Ventricular Tachycardia (98) | 0 | 0.0 % |
| **NONE (correctly silent)** | **684** | **10.5 %** |

Per-head residual firing on negatives is high across the board (AFib 53–72 %, Pause head 32–47 %, SVT head 53–77 %) — only **10.5 %** of confirmed-negative windows stay fully silent. **AFib remains the dominant FP generator** (47 % of the argmax), with Sinus Tachycardia #2 — pinpointing AFib + Sinus-Tachy as the highest-value motion/SQI suppression targets. (VT (98) contributes 0 FPs precisely because it never fires at 0.5.)

---

## 5. Full-150 cross-firing — background heads (caveat)

`head_0` (~100 %) and `SINUS RHYTHM` (head 3, ~97–100 %) fire on essentially all windows regardless of label — non-discriminative background heads, excluded from the scope set. With the new low thresholds, `WITH SINUS PAUSE` now also appears as a top co-firer on most events (e.g. 78 % of AFib, 91 % of V-Run). Raw head probabilities are not mutually exclusive — the scope + per-head thresholds are what make the output actionable.

---

## Key findings

1. **Scope = 6 heads; overall FP-OFF: sens 0.648 / spec 0.674 / acc 0.664 / AUROC 0.768** (micro).
2. **Two heads are production-grade unaided**: Bradycardia (acc 0.96) and AFib recall (0.98, pre-suppression).
3. **Pause is now detectable** at its calibrated 0.006 threshold (sens 0.73, AUROC 0.76) — recovered from 0 %.
4. **SV Run fires but is low-confidence** (sens 0.81 / PPV 0.07).
5. **VT (98) is still effectively silent (0 %)** and intentionally left at 0.5 — its base head is worse-than-chance; it needs the fine-tuned head, not a threshold.
6. **AFib is the dominant confusion sink and FP source** — VT reads as AFib, and AFib wins 47 % of confirmed-negative windows → the top suppression target.
7. **The low thresholds trade PPV for recall** on 93/142, visible as promiscuous co-firing (§3).

## Caveats
- Per-head thresholds + scope are **base-model + fzark/fp_doctor specific**; recalibrate per device.
- Single stratified sample (seed 42). Small-n events (SV Run n_pos=26, Pause n_pos=82) carry wide sampling noise; Sinus Tachycardia has no TP windows.
- FP-OFF only. The OFF vs ON (motion+SQI suppression) comparison is in `combined_binary_metrics_scope.csv` / `res/confusion_off_scope/SCOPE_EVAL_NOTE.md`; suppression now touches only the in-scope active heads (AFib, Bradycardia).
