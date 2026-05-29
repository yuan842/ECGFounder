# Baseline Model — FP-algo OFF — Accuracy / FP Analysis + Confusion Mapping

**Date**: 2026-05-29
**Model**: base ECGFounder (`1_lead_ECGFounder.pth`), single-lead, **FP suppression OFF** (raw sigmoid heads, threshold 0.5).
**Cohorts** (stratified `random_state=42`, ≤500/class, grouped by Event Type — same alignment as `combined_binary_eval.py`):
- **Positives** = `ecg_tp_fzark` (3,284 clinician-confirmed true events)
- **Negatives** = `ecg_fp_doctor removed1` (5,001 / 6,541 clinician-removed false positives)

**Script**: [scripts/confusion_mapping_off.py](../../scripts/confusion_mapping_off.py) · CSVs in `res/confusion_off/`.
This uses the cached baseline probability matrices (no re-inference) — identical to the production OFF run.

---

## 1. Per-event accuracy / FP metrics (base, FP-OFF, t=0.5)

| event | n_pos | n_neg | sens | spec | PPV | NPV | acc | F1 | AUROC | PR-AUC |
|---|---|---|---|---|---|---|---|---|---|---|
| **Atrial Fibrillation** | 500 | 500 | **0.978** | 0.280 | 0.576 | 0.927 | 0.629 | 0.725 | 0.890 | 0.878 |
| **Bradycardia** | 500 | 500 | 0.944 | **0.968** | **0.967** | 0.945 | **0.956** | **0.955** | **0.974** | 0.976 |
| Isolated Ventricular Beat (PVC) | 500 | 500 | 0.786 | 0.746 | 0.756 | 0.777 | 0.766 | 0.771 | 0.856 | 0.848 |
| Isolated Supraventricular Beat (PAC) | 500 | 500 | 0.536 | 0.694 | 0.637 | 0.599 | 0.615 | 0.582 | 0.633 | 0.643 |
| Ventricular Couplet | 36 | 500 | 0.806 | 0.804 | 0.228 | 0.983 | 0.804 | 0.356 | 0.870 | 0.316 |
| Supraventricular Trigeminy | 76 | 500 | 0.961 | 0.326 | 0.178 | 0.982 | 0.410 | 0.300 | 0.756 | 0.314 |
| Supraventricular Couplet | 500 | 500 | 0.044 | 0.978 | 0.667 | 0.506 | 0.511 | 0.083 | 0.372 | 0.458 |
| Supraventricular Bigeminy | 64 | 1 | 0.781 | — | 0.980 | — | 0.769 | 0.870 | 0.719 | 0.995 |
| **Ventricular Run (VT)** | 500 | 500 | **0.000** | 1.000 | 0.000 | 0.500 | 0.500 | 0.000 | 0.360 | 0.392 |
| **Supraventricular Run (SVT)** | 26 | 500 | **0.000** | 1.000 | 0.000 | 0.951 | 0.951 | 0.000 | 0.606 | 0.063 |
| **Pause** | 82 | 500 | **0.000** | 1.000 | 0.000 | 0.859 | 0.859 | 0.000 | 0.758 | 0.335 |
| Sinus Tachycardia | 0 | 500 | — | 0.010 | — | — | — | — | — | — |
| ST Elevation | 1 | 0 | — | — | — | — | — | — | — | — |
| **OVERALL (micro)** | **3284** | **5001** | **0.547** | **0.779** | **0.620** | 0.724 | **0.687** | 0.581 | 0.700 | 0.694 |

Three tiers emerge with the FP algo OFF:
- **Works well unaided**: Bradycardia (acc 0.956, AUROC 0.974), PVC (acc 0.77, AUROC 0.86).
- **High sensitivity, poor specificity** → the FP-suppression targets: AFib (sens 0.98 but spec 0.28 → PPV 0.58), SV-Trigeminy (sens 0.96 / spec 0.33).
- **Dead heads — 0 sensitivity at 0.5**: **Ventricular Run (VT, head 98), Supraventricular Run (SVT, head 93), Pause (head 142)** never cross 0.5. VT's AUROC is 0.36 (*worse than chance* — negatives outscore positives on head 98). These three events are not detected by their assigned heads at single-lead; instead they are **mis-mapped onto other heads** (see §2–3).

---

## 2. Confusion mapping — single-label argmax (TP cohort)

Predicted = argmax over the 10 fzark detector heads if its prob ≥ 0.5, else **NONE**. Rows sum to n (true count). Diagonal (own head) = correct.

| true event ↓ \ predicted → | Brady(4) | AFib(5) | STach(6) | PVC(9) | PAC(16) | NONE | reading |
|---|---|---|---|---|---|---|---|
| **Atrial Fibrillation** (500) | 9 | **475** | 11 | 1 | 2 | 2 | clean |
| **Bradycardia** (500) | **463** | 27 | 0 | 2 | 0 | 8 | clean |
| Isolated Ventricular Beat (500) | 8 | 38 | 24 | **353** | 60 | 17 | mostly correct; bleeds to PAC/AFib |
| **Ventricular Run** (500) | 3 | **338** | 1 | 158 | 0 | 0 | **VT read as AFib (338) / PVC (158); never VT** |
| **Supraventricular Couplet** (500) | 49 | **321** | 3 | 51 | 74 | 2 | **read as AFib, not its own head** |
| Isolated Supraventricular Beat (500) | 52 | 90 | 6 | 85 | **207** | 57 | weak diagonal; spreads to AFib/PVC |
| Supraventricular Trigeminy (76) | 0 | 4 | 2 | 2 | **68** | 0 | correct (PAC head) |
| Supraventricular Bigeminy (64) | 1 | 25 | 0 | 2 | **35** | 1 | PAC, with AFib leakage |
| **Pause** (82) | **55** | 10 | 3 | 4 | 3 | 7 | **Pause read as Bradycardia** (slow rate) |
| Supraventricular Run (26) | 3 | 14 | 2 | 2 | 2 | 3 | **read as AFib**; never SVT |

(Columns 19/68/93/98/142 are ~all zero — those heads essentially never win the argmax.)

**The dominant confusion axis is "→ AFib".** Ventricular Run, SV Couplet, and SV Run are all read as Atrial Fibrillation. Pause collapses onto Bradycardia. The AFib head is both the strongest true detector *and* the strongest sink for other rhythms.

---

## 3. Multi-label co-firing (the >100% phenomenon)

% of each **true** event's windows that fire each fzark head (multi-label → rows can exceed 100%). Own head **bold**.

| true event ↓ | Brady(4) | AFib(5) | STach(6) | PVC(9) | PAC(16) | none |
|---|---|---|---|---|---|---|
| Atrial Fibrillation | 7.0 | **97.8** | 34.4 | 4.0 | 1.2 | 0.4 |
| Bradycardia | **94.4** | 14.2 | 0.0 | 14.2 | 1.2 | 1.6 |
| Isolated Ventricular Beat | 3.2 | 8.4 | 8.2 | **78.6** | 13.6 | 3.4 |
| Ventricular Couplet | 5.6 | 41.7 | 16.7 | **80.6** | 0.0 | 0.0 |
| Ventricular Run | 10.8 | **99.2** | 2.2 | 88.6 | 0.0 | 0.0 |
| Isolated Supraventricular Beat | 15.0 | 23.0 | 3.0 | 22.2 | **53.6** | 11.4 |
| Supraventricular Couplet | 24.2 | 83.2 | 13.4 | 24.6 | **34.6** | 0.4 |
| Supraventricular Trigeminy | 0.0 | 5.3 | 14.5 | 3.9 | **96.1** | 0.0 |
| Pause | 72.0 | 34.1 | 8.5 | 26.8 | 17.1 | 8.5 |

The model is genuinely multi-label: e.g. true **Ventricular Run** fires AFib 99% *and* PVC 89% simultaneously — the head-98 (VT) detector is silent but the rhythm lights up AFib+PVC together. True **AFib** also co-fires Sinus Tachycardia 34% (rate overlap). This is why per-window argmax (§2) and the binary per-head metrics (§1) tell complementary stories.

---

## 4. False-positive analysis — what fires on confirmed negatives

Argmax over the all-negative pool (6,541 clinician-removed windows):

| predicted on NEGATIVE windows | count | share |
|---|---|---|
| **Atrial Fibrillation** | **2,793** | **42.7 %** |
| Sinus Tachycardia | 1,310 | 20.0 % |
| PVC (Ventricular Beat) | 961 | 14.7 % |
| PAC (Supraventricular Beat) | 496 | 7.6 % |
| Bradycardia | 262 | 4.0 % |
| **NONE (correctly silent)** | **710** | **10.9 %** |

**The AFib head is the single largest false-positive generator** — it wins the argmax on 43 % of confirmed-negative windows, and (from the residual-firing matrix) fires ≥0.5 on **53–86 %** of negatives across nearly every event category. Sinus Tachycardia is the #2 FP source. With the FP algo OFF, the model emits *some* alert on ~89 % of windows that clinicians had ruled out. **This is precisely the burden the motion + SQI suppression algos exist to remove** — and AFib is exactly the head the motion gate (`mean_motion ≤ 5`) targets.

---

## 5. Full-150 cross-firing — the always-on background heads (caveat)

Across the full 150-class space, two heads fire on ~100 % of *all* windows regardless of label: `head_0` (100 %) and `SINUS RHYTHM` (head 3, ~97–100 %). These are non-discriminative background/rhythm heads — they do **not** enter the fzark detector set, so they don't affect the binary metrics, but they confirm the model keeps many heads active at once. Other frequent co-fires (`head_1`, `head_39`, `RIGHT BUNDLE BRANCH BLOCK`) ride along on most rhythms. Takeaway: **raw head probabilities are not mutually exclusive**; the fzark ontology's choice of one specific head per event is what makes detection tractable.

---

## Key findings

1. **Overall FP-OFF operating point: sens 0.547, spec 0.779, accuracy 0.687, AUROC 0.700** (micro-pooled). High recall on the heads that work, dragged down by dead heads and FP burden.
2. **Two heads are production-grade unaided**: Bradycardia (acc 0.96, AUROC 0.97) and PVC (AUROC 0.86).
3. **Three heads are effectively non-functional at single-lead (0 % sensitivity @0.5)**: Ventricular Run (VT/98), Supraventricular Run (SVT/93), Pause (142). VT is *worse than chance* (AUROC 0.36). Their true events are mis-mapped — VT/SV-Run → AFib, Pause → Bradycardia.
4. **AFib is the dominant confusion sink and FP source** — true VT, SV-Couplet, and SV-Run all read as AFib, and AFib wins 43 % of confirmed-negative windows. This pinpoints AFib (and Sinus Tachycardia) as the highest-value targets for the motion/SQI suppression split.
5. **The model is multi-label, not single-label** — rows exceed 100 % co-firing; argmax and per-head binary metrics are complementary views, not contradictory.

## Caveats
- Threshold fixed at 0.5; single stratified sample (seed 42). Small-n events (ST Elev n=1, SV-Bigeminy n_neg=1, V-Couplet n_pos=36, SV-Run n_pos=26) have unstable metrics.
- "Confusion" here is over the 10 fzark **detector heads**; the 150-head model also fires many non-fzark heads (§5).
- FP-OFF only, as requested. The ON comparison (motion+SQI suppression) lives in `combined_binary_metrics_v31.csv` and the split-algo package.
