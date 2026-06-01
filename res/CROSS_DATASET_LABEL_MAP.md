# Cross-dataset label map — all 150 Founder heads, every dataset

**Companion CSV (machine-readable, all 150 rows):** [cross_dataset_label_map.csv](cross_dataset_label_map.csv).
**Regenerate with:** `python3 -m scripts.build_cross_dataset_label_map`.
**Status:** This document accompanies [GLOBAL_LABEL_MAP.md](GLOBAL_LABEL_MAP.md) — it lists every head and every dataset side-by-side, with one justification per mapping. If anything here contradicts `GLOBAL_LABEL_MAP.md`, that document wins on the **hard-rule scope** and `label_config.py` wins on the **actual routing**.

---

## 1. Purpose

ECGFounder emits 150 independent probabilities from a fixed canonical vocabulary ([tasks.txt](../tasks.txt), SHA-pinned). Eight datasets feed labels into that vocabulary, each with a different native scheme:

| Dataset | Native scheme | Granularity | Mapping module |
|---|---|---|---|
| Challenge 2017 | 4-class string (`N`/`A`/`O`/`~`) | record | *none committed* (trivial — listed §7.2 in GLOBAL_LABEL_MAP) |
| CINC2015 | ICU alarm token + true/false verdict | record | [scripts/cinc2015/cinc2015_label_map.py](../scripts/cinc2015/cinc2015_label_map.py) |
| ecg_tp | 14 fzark event names | strip (10 s) | [label_config.py](../label_config.py) `FZARK_LABEL_MAP` |
| ecg_fp | 14 fzark event names (hard-negative semantic) | strip | same `FZARK_LABEL_MAP` |
| MIMIC-IV-ECG | MUSE free-text diagnostic phrases | record | [scripts/mimic/mimic_label_map.py](../scripts/mimic/mimic_label_map.py) |
| MIT-BIH | beat symbols + rhythm tokens | beat / rhythm | [label_config.py](../label_config.py) `MITDB_BEAT_MAP`, `MITDB_RHYTHM_MAP` |
| MOVE | activity tags only | continuous | *no rhythm labels — FP-only* |
| PTB-XL | pre-computed 150-vector from SCP codes | record | [csv/ptbxl_label.csv](../csv/ptbxl_label.csv) + `PTBXL_ACTIVE_CLASSES` |

The map answers two questions per head: **(a)** which datasets carry a label that routes here, and **(b)** why that routing is correct. **36 of 150 heads** receive at least one dataset label; the remaining 114 are heads the model can fire but for which we have no labelled positives anywhere.

---

## 2. Legend (column conventions)

- **idx / head_name** — position and canonical name from `tasks.txt`.
- **in_scope** — `yes` if the head is one of the 7 detection-scope heads (Normal-ECG, Brady, AFib, Sinus Tachy, SVT-Run, V-Run, Pause). Anything else is *mapped for labelling only*, never *detected*.
- **challenge2017** — the cinc17 class string that routes here. Mapping is trivial-by-design but not implemented in code; included for completeness.
- **cinc2015** — alarm-token substring (prefixed `alarm:`). A *true* alarm sets `vec[head]=1`; a *false* alarm yields an all-zero vector (hard negative).
- **ecg_tp_fzark / ecg_fp_fzark** — fzark `Event Type` name(s) that route to this head. The two columns are identical content because ecg_tp and ecg_fp share the same ontology; they differ only in **verdict semantic** (TP = positive, FP = hard negative).
- **mimic** — substring patterns (lowercased) that the MIMIC text-router treats as positive for this head. `;`-separated.
- **mitdb** — `beat:<symbol>` for beat-level AAMI annotations, `rhythm:<token>` for rhythm spans, `default-normal` for the no-event fallback indices (1, 2).
- **move** — always empty by design (no rhythm ground truth).
- **ptbxl_active** — `yes` if the head has ≥1 positive sample in `csv/ptbxl_label.csv` (n=21,799). PTB-XL labels are pre-computed 150-vectors; no per-SCP regex lives in this repo.

---

## 3. Mapping methodology (why these routes exist)

Three rules govern every routing decision:

1. **Single-head per event.** Each native label routes to exactly one Founder idx (v3 invariant). No multi-head fan-out, no operator logic. Composite events that can't pick one head (`Ventricular Bigeminy`, MIT-BIH `(B`/`(T`/`(AB`) are left **unmapped** and handled by the downstream FP suppressor.

2. **Match the head's tasks.txt phrase clinically, not lexically.** Several heads have awkward fragment names from the original Marquette 12SL vocabulary (e.g. head 142 = `"WITH SINUS PAUSE"`, head 78 = `"WITH QRS WIDENING"`). The route is justified by the head's empirical firing pattern + clinical intent (head 142 fires on asystole/pause events), not by string match.

3. **Specificity-first text routing.** Where free text (MIMIC) or substring tokens (CINC2015) drive labels, more-specific phrases are checked first and broad fallbacks last. MIMIC additionally masks `supraventricular` before the V-pattern heads (9, 98) run, because `"ventricular tachycardia" ⊂ "supraventricular tachycardia"`.

All routings have been ratified against `Dataset Classification Cross-Mapping.xlsx` (2026-05-28) and the ontology test suite ([tests/test_ontology.py](../tests/test_ontology.py), 52 tests).

---

## 4. The 36 mapped heads — side-by-side

Heads not listed below are in the CSV with empty per-dataset columns. The 7 in-scope heads are **bold**.

| idx | head | in_scope | challenge2017 | cinc2015 | fzark (tp/fp) | mimic | mitdb | ptbxl |
|---:|---|:---:|---|---|---|---|---|:---:|
| 1 | NORMAL SINUS RHYTHM | — | | | | | default-normal | |
| **2** | **NORMAL ECG** | **yes** | N | | Normal ECG | | default-normal | yes |
| 3 | SINUS RHYTHM | — | | | | | | yes |
| **4** | **SINUS BRADYCARDIA** | **yes** | | bradycardia | Bradycardia | sinus bradycardia / marked sinus bradycardia / bradycardia | (SBR | yes |
| **5** | **ATRIAL FIBRILLATION** | **yes** | A | | Atrial Fibrillation | atrial fibrillation / afib / a-fib | (AFIB | yes |
| **6** | **SINUS TACHYCARDIA** | **yes** | | tachycardia ⚠ | Sinus Tachycardia | sinus tachycardia / sinus tach | | yes |
| 9 | PREMATURE VENTRICULAR COMPLEXES | — | | | Isolated Ventricular Beat; Ventricular Couplet | pvc / ventricular ectopic / bigeminy / trigeminy / couplet | beat:V | yes |
| 11 | RIGHT BUNDLE BRANCH BLOCK | — | | | | | beat:R | yes |
| 12 | SEPTAL INFARCT | — | | | | | | yes |
| 13 | LEFT ATRIAL ENLARGEMENT | — | | | | | | yes |
| 15 | LOW VOLTAGE QRS | — | | | | | | yes |
| 16 | PREMATURE ATRIAL COMPLEXES | — | | | Isolated SV Beat; SV Bigeminy; SV Trigeminy | pac / atrial premature / ectopic / bigeminy / trigeminy | beat:A, beat:a | |
| 17 | ANTERIOR INFARCT | — | | | | | | yes |
| 19 | PREMATURE SUPRAVENTRICULAR COMPLEXES | — | | | Supraventricular Couplet | psvc / supraventricular couplet / ectopic / premature | beat:S, beat:j | |
| 20 | LEFT BUNDLE BRANCH BLOCK | — | | | | | beat:L | yes |
| 24 | LATERAL INFARCT | — | | | | | | yes |
| 26 | LEFT VENTRICULAR HYPERTROPHY | — | | | | | | yes |
| 30 | QT HAS LENGTHENED | — | | | | | | yes |
| 32 | ATRIAL FLUTTER | — | | | | | (AFL | yes |
| 36 | LEFT ANTERIOR FASCICULAR BLOCK | — | | | | | | yes |
| 40 | ANTEROSEPTAL INFARCT | — | | | | | | yes |
| 50 | ELECTRONIC ATRIAL PACEMAKER | — | | | | | | yes |
| 54 | ANTEROLATERAL INFARCT | — | | | | | | yes |
| 60 | RIGHT ATRIAL ENLARGEMENT | — | | | | | | yes |
| 61 | INFERIOR INFARCT | — | | | | | | yes |
| 68 | ST ELEVATION NOW PRESENT IN | — | | | ST Elevation | | | |
| 70 | LEFT POSTERIOR FASCICULAR BLOCK | — | | | | | | yes |
| 78 | WITH QRS WIDENING | — | | | | | | yes |
| 79 | WITH 1ST DEGREE AV BLOCK | — | | | | | | yes |
| 82 | RIGHT VENTRICULAR HYPERTROPHY | — | | | | | | yes |
| **93** | **SUPRAVENTRICULAR TACHYCARDIA** | **yes** | | | Supraventricular Run | svt / supraventricular tachycardia / avnrt / avrt | (SVTA | yes |
| **98** | **VENTRICULAR TACHYCARDIA** | **yes** | | ventricular_tachycardia / vtach | Ventricular Run | vt / v-tach / nsvt / nonsustained vt | (VT | yes |
| 101 | ANTEROLATERAL LEADS | — | | | | | | yes |
| 107 | WOLFF-PARKINSON-WHITE | — | | | | | | yes |
| 112 | NONSPECIFIC INTRAVENTRICULAR BLOCK | — | | | | | | yes |
| **142** | **WITH SINUS PAUSE** | **yes** | | asystole ⚠ | Pause | sinus pause / sinoatrial pause / sinus arrest / asystole | | |

⚠ = mapping is a clinical proxy or relaxation — see §5 justification.

---

## 5. Per-head justification

Heads are grouped by clinical category. Each entry explains *why* every non-trivial route lands here.

### 5.1 Rhythm reference (heads 1, 2, 3)

- **idx 1 — NORMAL SINUS RHYTHM.** MIT-BIH default-normal fallback only. PTB-XL does not carry positives here (it uses head 2 + head 3 for the "normal" cluster); MIMIC text routing has no entry because "normal sinus rhythm" lands implicitly in head 3 via PTB-XL's pre-computed labels — there is no live MIMIC mapping into this head.
- **idx 2 — NORMAL ECG ✅ scope.** The backbone "normal" head. PTB-XL has 9,514 positives (43.6%). MIT-BIH sets it default-on. Challenge 2017 `N` and fzark `Normal ECG` both route here exactly. CINC2015/MIMIC do **not** explicitly set head 2; their negative-cases route to an all-zero vector by design (see §7.3 of GLOBAL_LABEL_MAP for the implication on Normal-ECG metrics).
- **idx 3 — SINUS RHYTHM.** PTB-XL active. Distinct from head 2 in tasks.txt; we do not route into it from any other dataset because none of the others carry a "sinus rhythm without further qualifier" label cleanly separable from "normal ECG."

### 5.2 Rate disorders (heads 4, 6)

- **idx 4 — SINUS BRADYCARDIA ✅ scope.** Routes are exact across all datasets except CINC2015. The CINC2015 alarm token is `"bradycardia"` (extreme rate, ICU monitor — not necessarily sinus origin); we accept the relaxation because rate-only alarms are the dominant labelled bradycardia source and the FP cost lands in head-4 calibration.
- **idx 6 — SINUS TACHYCARDIA ✅ scope.** ecg_tp/fp + MIMIC route on explicit "sinus" phrases (correct). **CINC2015 ⚠**: the `"tachycardia"` alarm token routes here even though ICU extreme tachy may be SVT/AT/AF-with-RVR — head 6 is *specifically* sinus tachy. Documented relaxation in [scripts/cinc2015/cinc2015_label_map.py:36](../scripts/cinc2015/cinc2015_label_map.py:36). Consequence: head 6 fzark TPs = 0; PPV = 0.0% (see §2 in GLOBAL_LABEL_MAP). Tightening this route (require `"sinus" in token`) is recommendation #3 in the cross-dataset review and remains open.

### 5.3 AFib + flutter (heads 5, 32)

- **idx 5 — ATRIAL FIBRILLATION ✅ scope.** The strongest, most-validated head in the system (91.6% fzark PPV). Every dataset that carries an AFib label routes here exactly: cinc17 `A`, fzark `Atrial Fibrillation`, MIMIC `"atrial fibrillation"`/`afib`/`a-fib`, MIT-BIH `(AFIB`.
- **idx 32 — ATRIAL FLUTTER.** MIT-BIH `(AFL` only; PTB-XL active. Not in detection scope (the 7-label hard rule excludes AFL because production fzark has insufficient flutter data). The head still receives labels for completeness.

### 5.4 Ventricular ectopy + runs (heads 9, 98)

- **idx 9 — PREMATURE VENTRICULAR COMPLEXES.** The "PVC bucket" head. ecg_tp/fp combine `Isolated Ventricular Beat` (the canonical PVC) **+** `Ventricular Couplet` (v3.1 recovery — two PVCs in a row are still PVCs by head semantics, just patterned). MIMIC routes the entire V-ectopy substring family here (`pvc`, `ventricular ectopic`, `ventricular bigeminy/trigeminy/couplet` — all PVC patterns). MIT-BIH `beat:V` is the AAMI symbol for PVC. PTB-XL active. Out of detection scope (single-PVCs are not alerting events at the strip level).
- **idx 98 — VENTRICULAR TACHYCARDIA ✅ scope.** The "3+ consecutive PVCs" alerting head. ecg_tp/fp `Ventricular Run` → exact. MIMIC routes `vt`, `v-tach`, `nsvt`, `nonsustained ventricular tachycardia`. CINC2015 `ventricular_tachycardia` / `vtach` alarms route here; CINC2015 **`ventricular_flutter` / `ventricular_fib` are explicitly left unmapped** because they are clinically distinct from VT (faster, polymorphic, no QRS structure) and would distort the head's training distribution. MIT-BIH `(VT`. PTB-XL active.

### 5.5 Atrial / supraventricular ectopy (heads 16, 19, 93)

- **idx 16 — PREMATURE ATRIAL COMPLEXES.** ecg_tp/fp routes three fzark events here: `Isolated Supraventricular Beat`, `Supraventricular Bigeminy`, `Supraventricular Trigeminy`. Rationale: PAC is the constituent beat; the patterned forms (bigeminy/trigeminy) fire the same model head because the head responds to per-beat morphology, not to the repetition pattern (which the FP suppressor handles). MIMIC routes the broad PAC family. MIT-BIH `A` and `a` (atrial premature, aberrated atrial premature).
- **idx 19 — PREMATURE SUPRAVENTRICULAR COMPLEXES.** The "couplet / paired ectopy" head — narrower than 16. ecg_tp/fp `Supraventricular Couplet` → exact. MIMIC `supraventricular couplet`, `psvc`. **MIT-BIH `S` and `j` route here, not to 16** — this was the v3 fix: `S` (supraventricular premature) and `j` (junctional premature) align with PSVC, not PAC. Test-asserted at [tests/test_ontology.py](../tests/test_ontology.py).
- **idx 93 — SUPRAVENTRICULAR TACHYCARDIA ✅ scope.** ecg_tp/fp `Supraventricular Run` → good match (brief SVT). MIMIC routes `svt`, `avnrt`, `avrt` and the full phrase. MIT-BIH `(SVTA` (supraventricular tachyarrhythmia). PTB-XL active (n=42). Detection head; production PPV is severe (0.1%) due to specificity collapse, not labelling — the routing itself is clean.

### 5.6 Pause (head 142)

- **idx 142 — WITH SINUS PAUSE ✅ scope.** Head name is a fragment from the parent Marquette label; clinical meaning is "Pause / asystole ≥2 s." ecg_tp/fp `Pause` and MIMIC `sinus pause`/`sinoatrial pause`/`sinus arrest`/`asystole` route here straightforwardly. **CINC2015 `asystole` ⚠**: asystole >2 s is the *clinical* meaning of Pause, but the tasks.txt phrase mentions "sinus" while ICU asystole is rate-independent. The route is preserved on clinical grounds — it's the best available source of pause labels outside the production cohort. Documented in [scripts/cinc2015/cinc2015_label_map.py:10](../scripts/cinc2015/cinc2015_label_map.py:10). PTB-XL has **0 positives** for head 142 (SCP codes have no pause entry).

### 5.7 Conduction blocks (heads 11, 20, 36, 70, 79, 112)

PTB-XL is the sole source for these labels (`RIGHT BUNDLE BRANCH BLOCK`, `LEFT BUNDLE BRANCH BLOCK`, `LEFT ANTERIOR FASCICULAR BLOCK`, `LEFT POSTERIOR FASCICULAR BLOCK`, `WITH 1ST DEGREE AV BLOCK`, `NONSPECIFIC INTRAVENTRICULAR BLOCK`). MIT-BIH adds beat-level RBBB (`beat:R` → 11) and LBBB (`beat:L` → 20) — same head as the rhythm-level PTB-XL label, because a single bundle-block beat carries the same morphology that drives the head. Not in detection scope.

### 5.8 Ischemia / infarct (heads 12, 17, 24, 40, 54, 61, 68, 101)

PTB-XL only, except head 68 (`ST ELEVATION NOW PRESENT IN`) which also receives fzark `ST Elevation`. Head 68 is the only ischemia head with any fzark labelled data — and that data is sparse (1 TP). All other infarct heads (`SEPTAL INFARCT`, `ANTERIOR INFARCT`, `LATERAL INFARCT`, `ANTEROSEPTAL INFARCT`, `ANTEROLATERAL INFARCT`, `INFERIOR INFARCT`, `ANTEROLATERAL LEADS`) come from PTB-XL's SCP-derived 150-vectors. None in detection scope; none routed from MIMIC/MIT-BIH/CINC2015/MOVE because their native vocabularies do not carry distinct infarct-location labels.

### 5.9 Chamber enlargement / hypertrophy (heads 13, 26, 60, 82)

PTB-XL only. SCP codes carry `LAE`/`RAE`/`LVH`/`RVH` as discrete labels; no other dataset annotates these.

### 5.10 Other PTB-XL-only heads (15, 30, 50, 78, 107)

`LOW VOLTAGE QRS`, `QT HAS LENGTHENED`, `ELECTRONIC ATRIAL PACEMAKER`, `WITH QRS WIDENING`, `WOLFF-PARKINSON-WHITE`. All sourced exclusively from PTB-XL's pre-computed labels. None in detection scope; none mapped from other datasets.

---

## 6. The 114 unmapped heads

114 of 150 heads have **no labelled positive in any of our eight datasets**. They fall into three groups:

1. **PTB-XL non-active heads** — 119 of 150 PTB-XL idx have zero positives; the 119 minus the 5 routed-from-other-datasets-only heads (1, 16, 19, 68, 142) leaves the bulk of the unmapped 114. These are diagnoses in the canonical Marquette vocabulary that simply did not appear in PTB-XL's 21,799 records (e.g. specific axis deviations, lead-specific T-wave abnormalities).
2. **Heads no production data uses** — heads like `EARLY REPOLARIZATION`, `JUVENILE T WAVES`, `BORDERLINE`, etc. — the canonical vocabulary anticipates them but none of our cohorts carry the corresponding native label.
3. **Sentence-fragment continuation heads** — several head names in tasks.txt are sentence fragments (e.g. `"OR ANTEROLATERAL LEADS"`, `"OR LATERAL LEADS"`, `"WITH PROBABLY OLD MYOCARDIAL INFARCTION"`) that originate from the source EMR's "diagnosis-continuation" text and have no native equivalent in any external dataset.

The model can still fire these heads at inference, but with zero positive supervision from our labelled datasets they cannot be evaluated for PPV/recall on our cohorts. Adding new datasets (Chapman-Shaoxing, ICBEB, etc.) is the only way to extend coverage.

---

## 7. Asymmetries and design notes

- **ecg_tp ↔ ecg_fp share routing**, differ only in **verdict**. The CSV duplicates the fzark column intentionally so the matrix stays rectangular; a single column with a separate "semantic" column would obscure that ecg_fp positives at a head become **negatives** for that head's calibration.
- **MOVE has no native rhythm labels**, so its column is empty by design — adding any routing would be fabricating ground truth. MOVE drives **FP-behavior** evaluation only, via inference-time predictions + the motion/SQI gate.
- **Normal-ECG (head 2) coverage is asymmetric** — only PTB-XL and MIT-BIH set head 2 explicitly. CINC2015 / MIMIC / ecg_fp produce implicit zeros at head 2 (no-match → all-zeros). Normal-ECG metrics are only valid on PTB-XL + MIT-BIH.
- **Composite events are intentionally unmapped** — `Ventricular Bigeminy`/`Trigeminy`, `Prolonged RR Interval`, MIT-BIH `(B`/`(T`/`(AB` all leave the model un-supervised at the strip level. Pattern detection lives in the downstream FP suppressor, not in a head.
- **Challenge 2017 routing is documented but not coded.** If we need to evaluate ECGFounder on cinc17 the trivial mapper takes three lines (`A → 5`, `N → 2`, `O/~ → None`); we have not committed it because cinc17 is currently used only for the retrained Stanford baseline.

---

## 8. How to extend / modify

Adding a new dataset:

1. Build a mapping module under `scripts/<dataset>/` that imports head names from `label_config.load_tasks()` — never hard-code head names.
2. Document the mapping in [GLOBAL_LABEL_MAP.md](GLOBAL_LABEL_MAP.md) §6 (or a new sub-section).
3. Extend [scripts/build_cross_dataset_label_map.py](../scripts/build_cross_dataset_label_map.py) with the new inverter + column, regenerate `cross_dataset_label_map.csv`.
4. Add a sub-section to §5 here justifying any non-exact routes.

Adding a new head to detection scope: edit `SCOPE_EVENT_TO_HEAD` **and** `DETECTION_SCOPE` together in `label_config.py` (the import-time assertion enforces agreement), then update the hard-rule block in [GLOBAL_LABEL_MAP.md](GLOBAL_LABEL_MAP.md) and [CLAUDE.md](../CLAUDE.md).
