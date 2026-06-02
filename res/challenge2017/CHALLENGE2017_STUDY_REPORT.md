# Out-of-Box Validation of the Production Single-Lead AFib Detector on PhysioNet/CinC Challenge 2017

**Study date:** 2026-06-02 · **Model:** ECGFounder 1-lead backbone + fzark production overlay (L1 + L2) · **Dataset:** PhysioNet/CinC Challenge 2017 (AliveCor, 300 Hz, single-lead)

---

## 1. Executive summary

We evaluated the **production atrial-fibrillation (AFib) detector** — exactly as deployed on the fzark device, with **no retraining or tuning** — against the public PhysioNet/CinC Challenge 2017 dataset, a completely independent device and patient population.

**Headline result (held-out test set, AFib vs Normal):**

| Metric | Value | Plain-language meaning |
|---|---:|---|
| **Sensitivity** | **88.2%** | of true AFib recordings are caught |
| **Specificity** | **93.5%** | of normal recordings are correctly left alone |
| **ROC-AUC** | **0.95** | strong overall discrimination |
| **F1** | **0.76** | balanced precision/recall |

**Bottom line:** a model calibrated only on our own ambulatory single-lead data generalizes to an unseen device with **0.95 AUC** for AFib — and the production post-processing (L1 calibration + L2 arbitration) converts a raw model that fires indiscriminately (F1 0.50) into a **deployable detector (F1 0.76)** with clinically usable specificity.

---

## 2. Why this study

- The production AFib head was calibrated on the **fzark** ambulatory single-lead cohort. Before relying on it, we need evidence it holds up on **out-of-distribution** data — a different recording device, patient mix, and noise profile.
- Challenge 2017 is an ideal external benchmark: a large, publicly labeled single-lead AFib dataset (AliveCor handheld, 300 Hz) with no overlap with our training data.
- Because the model was **never trained on Challenge 2017**, this is a true zero-leakage, out-of-box generalization test.

---

## 3. Data and method

**Dataset.** 8,528 single-lead recordings, 30–60 s each, four reference labels:

| Label | Meaning | n | Role in this study |
|---|---|---:|---|
| N | Normal sinus | 5,076 | AFib-negative (clean) |
| A | Atrial fibrillation | 758 | AFib-positive (target) |
| O | Other rhythm | 2,415 | context only (mixed) |
| ~ | Noisy / poor quality | 279 | context only |

Only **A** maps to a production scope event (AFib, head 5). **N** is the clean negative; **O** and **~** are reported for context but excluded from binary metrics (Other contains non-AFib arrhythmias and some AF-like rhythms; ~ indexes signal quality).

**Split.** Stratified **80 / 10 / 10** (train / val / test), per-class, seed-fixed, frozen as a manifest (`csv/challenge2017_split.csv`). **Validation** is reported as tuning context; **test** is the held-out report set. (The model is not trained on this data, so the split exists to give a clean held-out number, not to prevent leakage.)

**Detection.** Faithful to the production recording-report pipeline: each recording is tiled into non-overlapping **10 s windows**, every window is preprocessed (300→500 Hz resample, 60 Hz notch, band-pass, baseline removal, winsorized normalization) and scored; a recording is flagged AFib if **any** window fires. Recording-level score = max window probability.

---

## 4. The production model (what is being tested)

A three-stage pipeline on a frozen foundation backbone:

1. **Backbone** — ECGFounder 1-lead, 150-head, frozen. Produces raw per-event probabilities.
2. **L1 — learned calibration** (`fzarkSL`): a routed projection that re-calibrates the data-rich heads (AFib, Bradycardia, Sinus-Tachy) to device-specific operating points; rare heads route straight from the backbone.
3. **L2 — deterministic arbiter**: GT-matched rules that suppress physiologically/empirically impossible co-firings (mutually-exclusive rate rhythms, NSR contradiction), reducing false alarms without a learned model.

This study isolates each stage's contribution (Section 6).

---

## 5. Primary results — held-out TEST (n = 852; A = 76, N = 507)

| Variant | Sensitivity (A) | Specificity (N) | PPV | F1 | ROC-AUC | PR-AUC |
|---|---:|---:|---:|---:|---:|---:|
| Raw head @0.5 | 0.987 | 0.708 | 0.336 | 0.502 | 0.983 | 0.922 |
| + L1 calibration (L2-off) | 0.895 | 0.876 | 0.519 | 0.657 | 0.946 | 0.803 |
| **Full production (L1 + L2)** | **0.882** | **0.935** | **0.670** | **0.761** | 0.946 | 0.803 |

**Context fire-rates** (lower = better; these are not clean AFib): Other 0.564 → 0.282 → **0.178**; Noisy 0.750 → 0.214 → **0.214** across raw → L1 → full production.

---

## 6. Stage-by-stage contribution

**base → L1 (calibration):** specificity **+16.8 pp** (0.708 → 0.876) for a **−9.2 pp** sensitivity cost. This is the dominant clean-up step.

**L1 → L2 (arbitration):** specificity **+5.9 pp** (0.876 → 0.935) and Other fire-rate **−10.4 pp**, for only **−1.3 pp** sensitivity. Near-free false-alarm reduction.

| Metric | L2-off | L2-on | Δ |
|---|---:|---:|---:|
| Sensitivity (A) | 0.895 | 0.882 | −0.013 |
| Specificity (N) | 0.876 | 0.935 | **+0.059** |
| Fire-rate Other | 0.282 | 0.178 | **−0.104** |
| Fire-rate Noisy | 0.214 | 0.214 | 0.000 |

---

## 7. Key technical finding — calibration, not ranking

The corrected per-variant AUC shows the **raw head ranks A-vs-N slightly *better*** (test ROC 0.983) than the L1-calibrated score (0.946) — yet the raw head is **useless at its default threshold** (fires on 29% of normals). 

The value of the production overlay is therefore **calibration of the decision point, not ranking**: L1 maps the raw score to a usable operating threshold (specificity 0.876 at τ = 0.54), and L2 adds specificity on top (→ 0.935). ROC/PR-AUC are identical L2-off vs L2-on by construction — L2 changes the binary decision, not the score.

> Engineering implication: there is headroom to recover sensitivity by re-fitting L1 toward this distribution's ranking, *if* a multi-device operating point is desired.

---

## 8. Validation consistency (tuning context)

VAL (n = 854) and TEST (n = 852) agree closely — the result is stable, not an artifact of the split:

| | VAL | TEST |
|---|---:|---:|
| Full-production sensitivity | 0.908 | 0.882 |
| Full-production specificity | 0.919 | 0.935 |
| Full-production F1 | 0.742 | 0.761 |
| ROC-AUC (L1) | 0.952 | 0.946 |

---

## 9. Clinical interpretation

- **Sensitivity 88%** at **specificity 94%** on an unseen handheld device means the detector misses roughly 1 in 9 AFib recordings while correctly clearing ~19 of every 20 normal recordings — a favorable balance for an ambulatory screening / triage context where false-alarm burden drives clinician fatigue.
- The **−23 pp false-positive reduction** on normals (vs the raw model) directly translates to fewer spurious AFib alerts reaching clinical review.
- "Other rhythm" recordings still fire ~18% of the time — expected, since that class contains AF-like and other arrhythmias; this is not a pure false-positive rate.

---

## 10. Limitations

- **Single mappable label.** Only AFib has a 1:1 production scope event; N/O/~ cannot be scored as distinct targets. Other arrhythmia heads were not evaluable here.
- **"Other"/"Noisy" are not clean negatives** — excluded from binary metrics, reported as context.
- **Recording-level, any-window-fires** rollup is sensitive to recording length; very short recordings yield a single window.
- **Modest positive count** (76 AFib in test) — confidence intervals are non-trivial; consistency with val mitigates this.

---

## 11. Conclusions & recommendations

1. **The production AFib detector generalizes out-of-box** to an independent single-lead device at **0.95 AUC, 88% / 94% sensitivity/specificity** — evidence the fzark calibration is not overfit to our own device.
2. **The L1 + L2 overlay is doing essential work** — without it the raw model is undeployable (fires on 29% of normals). L2 specifically buys **+5.9 pp specificity at ~zero sensitivity cost**.
3. **Recommended next steps:** (a) optionally re-fit L1 for a multi-device operating point to recover the ranking AUC headroom; (b) extend evaluation to additional external single-lead datasets; (c) report confidence intervals with a larger positive sample.

---

*Artifacts: `res/challenge2017/PROD_AFIB.md` (metrics), `csv/challenge2017_split.csv` (frozen split), `scripts/eval_challenge2017_prod.py` (reproducible eval).*
