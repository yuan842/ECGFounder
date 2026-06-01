# ECGFounder co-firing on PTB-XL — comprehensive report

**Scope:** 6 fzark detection heads `{4 Brady, 5 AFib, 6 Sinus Tachy, 93 SVT-Run, 98 V-Run, 142 Pause}`. **NORMAL SINUS RHYTHM (1) and NORMAL ECG (2) were removed from scope 2026-06-01** — they are normal-state backbone heads, not detection targets (head 1 had 0 PTB-XL GT / 100% FP; head 2 was always subsumed by head 1). **Model:** base backbone `1_lead_ECGFounder.pth`, lead II @ 500 Hz, t=0.5, FP-suppression off (no accelerometer on PTB-XL). Split-exempt — the base backbone was not trained on PTB-XL, so all **21,799 records** are evaluated. **Inputs:** `res/ptbxl_baseline_sqi/base_probs_full.npy` (sigmoid probs) + `csv/ptbxl_label.csv` `label` column (GT). Alignment verified against `run.log`.

**Generators (this directory):** [ptbxl_cofiring_analysis.py](../../scripts/ptbxl_cofiring_analysis.py) · [ptbxl_cofiring_scope_detail.py](../../scripts/ptbxl_cofiring_scope_detail.py) · [ptbxl_cofiring_scores.py](../../scripts/ptbxl_cofiring_scores.py) · [ptbxl_cofiring_realness.py](../../scripts/ptbxl_cofiring_realness.py) · [ptbxl_cofiring_gt.py](../../scripts/ptbxl_cofiring_gt.py). **Data tables (CSVs):** every table below has a matching CSV in this directory.

---

## 0. Headline findings

1. **Real co-firing is almost nonexistent in GT.** GT puts ≥2 scope labels on only **42 of 21,799 segments (0.19%)** — mean 0.14 scope labels/segment. Among the 6 detection heads, the events are essentially mutually exclusive.
2. **The only real GT co-occurrence is SVT-Run ≡ V-Run** (42 records, **Jaccard 1.0** — perfectly co-extensive). Brady, AFib, Sinus-Tachy, Pause **never co-occur with each other** in GT.
3. **The model over-fires co-occurrences.** It co-fires a mean of **0.39** scope heads/segment vs GT's 0.14, and produces co-firings GT never sanctions: **AFib+Sinus-Tachy 436×** (GT 0), Brady+AFib 190× (GT 0), Brady+Tachy 102× (GT 0).
4. **SVT-Run ↔ V-Run is NOT a confusion to fix — it's a real GT identity** the model under-represents (model Jaccard 0.68 vs GT 1.0). ⚠ Likely a PTB-XL label-encoding artifact: one run-type SCP code mapping to both heads.
5. **Co-fires are low-confidence and mostly false.** When a head rides along as a secondary detection its score collapses toward the 0.5 floor (median 0.62–0.72 vs 0.91–0.99 as main), and its annotation precision craters (Sinus-Tachy co-fire **1.0%**, AFib co-fire **14.7%**).
6. **NSR's score is still available as an FP-suppression feature** even though head 1 is no longer a detection target — see §F rule 2.

---

## 1. Setup and definitions

- **Universe.** Scope-restricted to the 6 detection heads; every number is **scope-vs-scope only**. The 144 out-of-scope heads (incl. the descriptor scaffold ABNORMAL ECG/SINUS RHYTHM/LAE and the de-scoped NSR/Normal-ECG) fire on most records but are excluded from co-firing analysis. (Mean firings/segment across all 150 heads is 6.29; only the 6 scope heads matter for detection.)
- **Main fire vs co-fire.** Per segment, among fired scope heads (prob ≥ 0.5) the **highest-scoring** is the *main* fire; every other fired scope head is a *co-fire*.
- **Solo.** A scope head fires "solo" when it is the only scope head firing on the segment.
- **Score.** Raw sigmoid probability — no calibration applied.

---

## A. Ground truth — what GT sanctions per 10-s segment

CSVs: [gt_labels_per_segment.csv](gt_labels_per_segment.csv) · [gt_cofire_counts.csv](gt_cofire_counts.csv) · [gt_cofire_jaccard.csv](gt_cofire_jaccard.csv) · [gt_conditional_directional.csv](gt_conditional_directional.csv)

### A.1 GT scope-label load per segment

| #scope GT labels | segments | share |
|---:|---:|---:|
| 0 | 18,783 | 86.2% |
| 1 | 2,974 | 13.6% |
| 2 | 39 | 0.18% |
| 3 | 3 | 0.01% |

86% of records carry **no** scope-event label (they are normal / out-of-scope rhythms). Of those that do, all but 42 carry exactly one. **Real co-firing is a 0.19% edge case.**

### A.2 Every nonzero GT co-occurrence

| GT pair | count | Jaccard | reading |
|---|---:|---:|---|
| **SVT-Run (93) ≡ V-Run (98)** | **42** | **1.000** | the same 42 records carry both labels |
| AFib (5) ↔ SVT/V-Run | 2 / 2 | 0.001 | trivial |
| Sinus Tachy (6) ↔ SVT/V-Run | 1 / 1 | 0.001 | trivial |

**Everything else is exactly zero** — Brady↔AFib, Brady↔Tachy, AFib↔Tachy, and all Pause pairs (head 142 has 0 GT positives in PTB-XL).

> ⚠ **The SVT≡VT identity (Jaccard 1.0) is almost certainly a PTB-XL encoding artifact** — implausible that all 42 patients had both sustained SVT and sustained VT; far likelier a single run-type SCP code mapped to both heads when the 150-vector was pre-computed. Read it as "PTB-XL does not distinguish the two run heads."

### A.3 What "should" co-fire — answer

Within the 6 detection heads, GT sanctions **essentially one** co-fire pattern: **SVT-Run + V-Run together** (PTB-XL does not separate the two run heads). AFib, Brady, Sinus-Tachy and Pause are **strictly single-label events** in GT — they do not co-occur with each other.

---

## B. Predicted co-firing — per-head profile

CSVs: [scope_per_head_profile.csv](scope_per_head_profile.csv) · [scope_accompaniment_dist.csv](scope_accompaniment_dist.csv) · [scope_conditional_directional.csv](scope_conditional_directional.csv) · [cofire_counts_scope.csv](cofire_counts_scope.csv) · [cofire_jaccard_scope.csv](cofire_jaccard_scope.csv) · [heads_per_segment.csv](heads_per_segment.csv).

### B.1 Master table

| head | name | n_gt | n_pred | sens | ppv | **% solo** | mean other scope | top scope companion (rate) |
|---:|---|---:|---:|---:|---:|---:|---:|---|
| 4 | SINUS BRADYCARDIA | 637 | 3745 | 0.970 | 0.165 | 92.9% | 0.078 | AFib (5.1%) |
| 5 | ATRIAL FIBRILLATION | 1514 | 2135 | 0.976 | 0.692 | 71.9% | 0.309 | Sinus Tachy (20.4%) |
| 6 | SINUS TACHYCARDIA | 826 | 2528 | 0.981 | 0.320 | 77.6% | 0.263 | AFib (17.2%) |
| 93 | SUPRAVENTRICULAR TACHYCARDIA | 42 | 55 | 0.571 | 0.436 | **0.0%** | 2.200 | Sinus Tachy (100%) |
| 98 | VENTRICULAR TACHYCARDIA | 42 | 71 | 0.643 | 0.380 | **0.0%** | 1.972 | Sinus Tachy (100%) |
| 142 | WITH SINUS PAUSE | 0 | 0 | — | — | — | — | never fires |

Mean **0.39** scope heads/segment overall (median 0, max 4). **14,093 of 21,799 (64.6%)** segments fire no scope head.

### B.2 Two regimes

- **Mostly-solo arrhythmia heads (4 Brady, 5 AFib, 6 Sinus-Tachy)** — fire 72–93% solo; behave as independent detectors.
- **Never-solo cluster (93 SVT-Run, 98 V-Run)** — 0% solo, mean ~2 other scope heads always firing alongside (always Sinus-Tachy, usually each other).
- **Pause (142)** — never fires (0 GT, 0 prediction).

### B.3 Directional conditionals — predicted P(B fires | A fires)

| given ↓ \ also fires → | 4 Brady | 5 AFib | 6 Tachy | 93 SVT | 98 VT | 142 Pause |
|---|---:|---:|---:|---:|---:|---:|
| **4 BRADY** | — | 0.051 | 0.027 | 0.000 | 0.000 | 0 |
| **5 AFIB** | 0.089 | — | **0.204** | 0.007 | 0.008 | 0 |
| **6 SINUS TACHY** | 0.040 | **0.173** | — | 0.022 | 0.028 | 0 |
| **93 SVT-RUN** | 0.000 | 0.273 | **1.000** | — | **0.927** | 0 |
| **98 V-RUN** | 0.000 | 0.254 | **1.000** | **0.718** | — | 0 |
| **142 PAUSE** | 0 | 0 | 0 | 0 | 0 | — |

P(Sinus Tachy | SVT-Run) = P(Sinus Tachy | V-Run) = **1.00** — the runs always co-fire tachy. P(V-Run | SVT-Run) = 0.93, P(SVT-Run | V-Run) = 0.72 — asymmetric but the cluster is essentially one entangled detection.

---

## C. Score distributions: main fire vs co-fire

CSVs: [scope_score_main_vs_cofire.csv](scope_score_main_vs_cofire.csv) · [scope_pair_cofire_scores.csv](scope_pair_cofire_scores.csv).

### C.1 Per-head — main vs co-fire score percentiles

| head | role | n | mean | p5 | p10 | p25 | **median** | p75 | p90 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 4 Bradycardia | main | 3663 | 0.94 | 0.76 | 0.84 | 0.93 | **0.97** | 0.98 | 0.99 |
| | co-fire | 82 | 0.66 | 0.53 | 0.54 | 0.58 | **0.63** | 0.73 | 0.80 |
| 5 AFib | main | 1911 | 0.91 | 0.58 | 0.67 | 0.91 | **0.97** | 0.99 | 0.99 |
| | co-fire | 224 | 0.66 | 0.51 | 0.51 | 0.55 | **0.64** | 0.78 | 0.85 |
| 6 Sinus Tachy | main | 2126 | 0.83 | 0.53 | 0.56 | 0.65 | **0.91** | 0.996 | 0.999 |
| | co-fire | 402 | 0.65 | 0.52 | 0.53 | 0.56 | **0.62** | 0.71 | 0.81 |
| 93 SVT-Run | main | 4 | 0.97 | 0.94 | 0.94 | 0.96 | **0.97** | 0.98 | 0.98 |
| | co-fire | 51 | 0.72 | 0.52 | 0.54 | 0.59 | **0.72** | 0.84 | 0.90 |
| 98 V-Run | main | 2 | 0.99 | 0.98 | 0.98 | 0.99 | **0.99** | 0.99 | 0.99 |
| | co-fire | 69 | 0.73 | 0.51 | 0.55 | 0.60 | **0.74** | 0.84 | 0.93 |
| 142 Pause | — | 0 | never fires | | | | | | |

**Universal rule: co-fire scores collapse toward the 0.5 threshold** — every head's median falls 0.25–0.34 between main and co-fire role, with co-fire mass hugging the floor (p5 = 0.51–0.53).

### C.2 Two regimes for the collapse

- **Regime A — weak ride-alongs (Brady, AFib, Sinus-Tachy):** demoted to co-fire → median 0.62–0.64, ~0.3 below their main-fire score. Low-confidence shadows; suppressible.
- **Regime B — confident cluster (SVT-Run, V-Run):** never fire solo, but co-fire scores stay high (median 0.72–0.74) — near-peers of each other. No weak member to suppress; one entangled detection.

### C.3 Directional pairwise — A main, B co-fires → B's score

| main A | co-fire B | n | B median (co-fire) | B median (standalone) | reading |
|---|---|---:|---:|---:|---|
| **AFib** | Sinus Tachy | 310 | **0.64** | 0.91 | weak shadow — supports "Tachy is FP when AFib+Tachy co-fire" |
| **Sinus Tachy** | AFib | 87 | **0.72** | 0.97 | weak shadow |
| Bradycardia | AFib | 110 | 0.62 | 0.97 | weak |
| Sinus Tachy | SVT-Run | 41 | 0.73 | 0.97 | moderate |
| Sinus Tachy | V-Run | 58 | 0.74 | 0.99 | moderate |
| **SVT-Run** | V-Run | 4 | **0.97** | 0.99 | **strong — triad fires as a confident block** |
| **SVT-Run** | Sinus Tachy | 4 | **0.93** | 0.91 | **strong — co-fire ≈ main** |
| V-Run | SVT-Run | 2 | 0.97 | 0.97 | strong |

Regime-A co-fires score ~0.6 (weak shadows ~0.3 below their as-main score); the Regime-B triad co-fires score 0.9+ (tied with the main — nothing to suppress).

---

## D. Realness — are co-fires annotated? Does total firing match GT?

CSVs: [scope_realness_by_role.csv](scope_realness_by_role.csv) · [scope_cofire_tp_vs_fp.csv](scope_cofire_tp_vs_fp.csv) · [scope_count_vs_gt.csv](scope_count_vs_gt.csv).

### D.1 Precision by role — are the firings real?

| head | **main precision** | **co-fire precision** | reading |
|---|---:|---:|---|
| 4 Bradycardia | 0.166 (609/3663) | **0.110** (9/82) | both low; FP problem is rate-threshold, not co-firing |
| **5 AFib** | **0.756** (1444/1911) | **0.147** (33/224) | main 76% real → co-fire **15% real** |
| **6 Sinus Tachy** | **0.379** (806/2126) | **0.010** (4/402) | main 38% real → co-fire **1.0% real** |
| 93 SVT-Run | 1.0 (4/4) | 0.392 (20/51) | co-fires partly real — the head's only firing mode |
| 98 V-Run | 0.5 (1/2) | 0.377 (26/69) | co-fires partly real |
| 142 Pause | — | — | never fires |

**Co-fires are mostly spurious for the arrhythmia heads.** Sinus-Tachy co-fire = **1.0% precision** (4 of 402) is the extreme: a co-firing tachy is almost always a false label. AFib drops 76%→15%. **For SVT/VT the order is reversed** — co-fire is ~40% precision and is essentially the head's *only* firing mode (SVT fires as main just 4×, V-Run 2×); suppressing run co-fires would destroy the real signal.

### D.2 TP-vs-FP co-firing split

| head | TP mean others | FP mean others | TP top companion | FP top companion |
|---|---:|---:|---|---|
| 4 Brady | 0.10 | 0.07 | AFib (5.7%) | AFib (5.0%) |
| **5 AFib** | 0.24 | **0.46** | Sinus Tachy (19.9%) | Bradycardia (22.9%) |
| **6 Sinus Tachy** | 0.05 | **0.36** | AFib (3.6%) | AFib (23.7%) |
| 93 SVT-Run | 2.04 | 2.32 | Sinus Tachy (100%) | Sinus Tachy (100%) |
| 98 V-Run | 1.93 | 2.00 | Sinus Tachy (100%) | Sinus Tachy (100%) |

False AFib co-fires more (0.46 vs 0.24 others) and skews toward Brady; **false Sinus-Tachy co-fires AFib 23.7% vs 3.6% for true** — a within-scope FP flag (the AFib+Tachy pair). For Brady the TP/FP profiles are identical (co-firing carries no signal); for SVT/VT identical (intrinsic clustering).

### D.3 Does counting all firings match the annotation count?

| head | n_gt | n_main | n_cofire | n_total | total/gt | recall | verdict |
|---|---:|---:|---:|---:|---:|---:|---|
| 4 Bradycardia | 637 | 3663 | 82 | 3745 | **5.88×** | 0.970 | over-counts ~6× |
| 5 AFib | 1514 | 1911 | 224 | 2135 | **1.41×** | 0.976 | roughly matches |
| 6 Sinus Tachy | 826 | 2126 | 402 | 2528 | **3.06×** | 0.981 | over-counts 3× |
| 93 SVT-Run | 42 | 4 | 51 | 55 | 1.31× | 0.571 | count close, records don't |
| 98 V-Run | 42 | 2 | 69 | 71 | 1.69× | 0.643 | count close, records don't |
| 142 Pause | 0 | 0 | 0 | 0 | — | — | GT 0, fires 0 |

**Counting all firings does not reproduce the annotation.** Brady over-counts 6×, Sinus-Tachy 3×; only AFib is close (1.41×, the reliable head). **The trap:** SVT-Run total 55 vs GT 42 (1.31×) and V-Run 71 vs 42 (1.69×) *look* close, but recall is only 57%/64% and precision ~40% — near-equal counts are missed positives roughly cancelling added false positives, not a real correspondence.

---

## E. GT vs predicted — the contrast

CSV: [gt_vs_pred_cofire.csv](gt_vs_pred_cofire.csv).

### E.1 Pure model confusion — GT NEVER sanctions, model fires anyway

| pair | GT count | model count | model Jaccard | disposition |
|---|---:|---:|---:|---|
| **AFib ↔ Sinus Tachy** | **0** | **436** | 0.103 | **suppress — Sinus-Tachy is the FP** |
| Bradycardia ↔ AFib | 0 | 190 | 0.033 | suppress |
| Bradycardia ↔ Sinus Tachy | 0 | 102 | 0.017 | suppress (physiologically impossible) |

### E.2 Model over-couples relative to GT

| pair | GT count | model count | factor |
|---|---:|---:|---:|
| Sinus Tachy ↔ V-Run | 1 | 71 | 71× |
| Sinus Tachy ↔ SVT-Run | 1 | 55 | 55× |
| AFib ↔ V-Run | 2 | 18 | 9× |
| AFib ↔ SVT-Run | 2 | 15 | 7.5× |

Runs ride on a tachycardic base rate far more than GT supports — the "tachy-run" entanglement of §B.3.

### E.3 Model UNDER-couples the one real GT co-occurrence

| pair | GT count | GT Jaccard | model count | model Jaccard |
|---|---:|---:|---:|---:|
| **SVT-Run ↔ V-Run** | 42 | **1.000** | 51 | 0.68 |

SVT-Run↔V-Run is **not a confusion to fix** — it's a real GT identity (same 42 records) the model under-represents. The model is closer to right than confused, subject to the §A artifact caveat.

---

## F. Actionable findings (see [MULTI_LABEL_RULES.md](MULTI_LABEL_RULES.md) for the policy)

1. **Suppress AFib + Sinus-Tachy double-labels.** GT sanctions zero; model produces 436. The **Sinus-Tachy is the false member** (co-fire score median 0.64 vs 0.91 standalone; co-fire precision 1.0%). Likely AF-RVR double-counted — keep AFib, drop Sinus-Tachy.
2. **NSR (head 1) is out of scope but its raw score remains an FP-suppression feature.** Head 1 is no longer detected (detect→None), but the backbone still computes its probability. A high head-1 score alongside an AFib/Tachy firing is a strong "this arrhythmia is false" signal (the NSR-contradiction effect measured when head 1 was in scope). Use it inside the FP-suppression layer, not as an emitted label.
3. **Suppress Brady+Tachy and Brady+AFib co-fires.** Both 0 GT count; model produces 102 and 190. Physiologically contradictory / unsanctioned.
4. **Do NOT suppress SVT-Run ↔ V-Run co-firing.** GT identifies the two heads (Jaccard 1.0 on the same 42 records); the model's strong co-firing is closer to GT than weaker would be. ⚠ Reflects PTB-XL encoding, not clinical fact — distinguishing the run heads needs a dataset that separates them + a fine-tuned head.
5. **Counting firings ≠ counting annotations.** Brady 5.9×, Tachy 3.1× over; SVT/VT counts coincide but record-match is poor. Use record-level recall/precision.
6. **SVT/VT remain ungeneralizable at single-lead** — co-fire scores stay high (median 0.72–0.97); no threshold/co-firing rule can split the SVT/VT/Sinus-Tachy cluster. Treat any 93/98 firing as "fast tachyarrhythmia of uncertain origin."
7. **Pause (142) is invisible on PTB-XL** — 0 GT, 0 predictions. Validation needs CINC2015 asystole or fzark Pause events.
