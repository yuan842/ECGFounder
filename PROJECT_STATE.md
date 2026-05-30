# Project State Snapshot

**Date**: 2026-05-28
**Purpose**: Single landing page that captures every decision, every canonical file, and every open question across the work streams active in this codebase. **Read this first before any new task** — it replaces the need to re-read prior session transcripts.

> ⛔ **GLOBAL HARD RULE (2026-05-29): detection scope = 7 labels (6 events + Normal ECG).** The
> system detects only **Atrial Fibrillation (5), Bradycardia (4), Sinus Tachycardia (6),
> Supraventricular Run (93), Ventricular Run (98), Pause (142), Normal ECG (2).** Source of truth =
> `label_config.SCOPE_EVENT_TO_HEAD` / `SCOPE_EVENTS` / `DETECTION_SCOPE`, enforced by an import-time
> assertion (Normal ECG is a backbone head, exempt from the fzark check). Anything else is out of
> scope everywhere (detect → None, FP suppressor passes through, eval scripts skip). See
> `res/GLOBAL_LABEL_MAP.md` § hard rule.

---

## 0. The four work streams

| Stream | Status | Canonical doc | Living code |
|---|---|---|---|
| 1. **Label mapping** (v3.1 fzark ontology) | ✅ shipped | [res/GLOBAL_LABEL_MAP.md](res/GLOBAL_LABEL_MAP.md) | [label_config.py](label_config.py) |
| 2. **Fine-tune experiments** | ✅ concluded (all attempts retired; production stack unchanged) | retired strategy docs in [res/_archived/](res/_archived/); diagnostic snapshot in §3 below | [dual_head_ecgfounder.py](dual_head_ecgfounder.py) (the production-ready output) |
| 3. **MIMIC partial download** | 📋 planned, not yet credentialed | [data/mimic-iv-ecg/README.md](data/mimic-iv-ecg/README.md) | [scripts/mimic/partial_download.py](scripts/mimic/partial_download.py) + [scripts/mimic/mimic_label_map.py](scripts/mimic/mimic_label_map.py) |
| 4. **Cloud demo (Streamlit MVP)** | 📋 planning complete; phase 1 implementation pending | [docs/cloud_demo/](docs/cloud_demo/) — read order: `BEGINNER.md` → `SETUP.md` → `STREAMLIT_APP.md` | none yet (will live at `app/app.py`) |

---

## 1. Label mapping — what's canonical

### V3.1 ontology (13 fzark events → 10 unique Founder heads)

`FZARK_LABEL_MAP` in [label_config.py](label_config.py) is the **single source of truth**. Every event maps to exactly one head; composite events that share a head (SV-Trig+SV-Big+ISB → idx 16, V-Couplet+IVB → idx 9) are explicitly allowed.

| idx | Head | Events |
|---|---|---|
| 4 | SINUS BRADYCARDIA | Bradycardia |
| 5 | ATRIAL FIBRILLATION | Atrial Fibrillation |
| 6 | SINUS TACHYCARDIA | Sinus Tachycardia |
| 9 | PVC | Isolated Ventricular Beat + Ventricular Couplet (v3.1 recovery) |
| 16 | PAC | Isolated Supraventricular Beat + Supraventricular Trigeminy + Supraventricular Bigeminy (v3.1 recoveries) |
| 19 | PSVC | Supraventricular Couplet |
| 68 | ST ELEVATION | ST Elevation (n=1, insufficient data) |
| 93 | SVT | Supraventricular Run |
| 98 | VT | Ventricular Run |
| 142 | WITH SINUS PAUSE | Pause |

Excluded from the live ontology (in `FZARK_UNMAPPABLE`): `Multiple Event`, `Unknown`, `Custom Heart Rate`, `Ventricular Bigeminy`, `Ventricular Trigeminy`, `Prolonged RR Interval`.

### Reliability metadata (added 2026-05-28 from Excel)

Every `FZARK_ONTOLOGY` node now carries `fzark_ppv_pct`, `clinical_reliability` (RELIABLE / MODERATE_FP / SEVERE_FP / INSUFFICIENT_DATA), and `clinical_support` (clinical description). Source: [res/_archived/Dataset Classification Cross-Mapping.xlsx](res/_archived/Dataset Classification Cross-Mapping.xlsx).

### Tests guarding the ontology

[tests/test_ontology.py](tests/test_ontology.py) — 40 tests. All pass. Guards: tasks.txt SHA256, V3.1 event count = 13, composite-rhythm exclusion, v3.1 head-sharing recoveries.

### Datasets touching this ontology

The 9 "production-critical" heads are validated in both PTB-XL and Fzark. The cross-dataset reliability table is in [res/GLOBAL_LABEL_MAP.md](res/GLOBAL_LABEL_MAP.md) §8.

---

## 2. Cross-dataset performance (suppression OFF and ON)

Canonical report: [res/cross_dataset_supp_off/cross_dataset_performance_supp_off.md](res/cross_dataset_supp_off/cross_dataset_performance_supp_off.md). Headline:

| Cohort | Metric | OFF | ON | Δ |
|---|---|---|---|---|
| ECG-TP fzark | aggregate detection @ 0.5 | 54.7% | 54.0% | −0.7 pp |
| ECG-FP doctor | raw FP rate @ 0.5 | 29.05% | 16.32% | **−12.7 pp** (44% reduction) |
| Combined | PPV @ 0.5 | 0.620 | 0.815 | +19.5 pp |
| Combined | ROC-AUC @ 0.5 | 0.700 | 0.772 | +7.2 pp |

The production stack is **base model + v2 FP suppression + v3.1 ontology**. This is what's shipping.

---

## 3. Fine-tune experiments — verdict and lessons

**All fine-tune attempts on the fuzzylead2 / PTB-XL cohort were retired.** The production stack uses the unmodified base model. Five attempts ran:

| # | Approach | Outcome | Why retired |
|---|---|---|---|
| 1 | Vanilla full fine-tune on fuzzylead2 (10 active classes) | fzark V3.1 detection collapsed 54.7% → 0.2% | Catastrophic forgetting — 140 heads with 0 positives got pushed to 0 by BCE |
| 2 | Masked-loss FT (gradient zeroed on 140 inactive heads) | fzark agg 62.3% at t=0.5; AFib head broke (97.8% → 6.8%); calibration drifted | Backbone still drifted; AFib head was *in* the active set so couldn't be protected |
| 3 | State-dict surgery B1 (base backbone + 10 fuzzy head rows) | fzark bit-identical to base; val_75deg gain almost zero | Confirmed the gain lives in the backbone, not the head rows |
| 4 | State-dict surgery B2 (fuzzy backbone + 140 base head rows) | Matches fuzzy on val_75deg; breaks fzark (54.7% → 2.4%) | Mirror of #3; classifier rows entangled with their training backbone |
| 5 | 6-head linear probe v2 (unweighted BCE on PTB-XL 4-angle) | fzark TP detection regressed 15–40 pp; FP rate dropped dramatically | Calibration shift — PTB-XL class imbalance moved logits down at t=0.5 |
| 5b | 6-head linear probe v3 (pos_weight BCE) | TP regressed only 1–5 pp; FP rate increased on PVC/V-Couplet | Swung the other way; failed regression gate |

**Production-ready output that survived this work:** [dual_head_ecgfounder.py](dual_head_ecgfounder.py) — two-checkpoint inference wrapper. Routes 8 PTB-XL-specific labels through a fine-tuned checkpoint, all others through base. Default routing tested and bit-identical on fzark.

**Checkpoints remaining on disk:**

| File | Status |
|---|---|
| `checkpoint/1_lead_ECGFounder.pth` | **Production base** — unchanged from upstream |
| `checkpoint/1_lead_ECGFounder_fuzzy.pth` | Vanilla FT (attempt #1) — destroys fzark; do not use alone |
| `checkpoint/1_lead_ECGFounder_fuzzy_masked.pth` | Masked-loss FT (attempt #2) — partially preserves fzark; v3.1 routing-compatible |
| `checkpoint/1_lead_ECGFounder_fuzzy_B1.pth` | State-dict surgery B1 (attempt #3) — diagnostic only |
| `checkpoint/1_lead_ECGFounder_fuzzy_B2.pth` | State-dict surgery B2 (attempt #4) — diagnostic only |
| `checkpoint/1_lead_ECGFounder_6head_v2.pth` | 6-head LP unweighted (attempt #5) — failed regression gate |
| `checkpoint/1_lead_ECGFounder_6head_v3_posw.pth` | 6-head LP pos_weight (attempt #5b) — failed regression gate |

**Detailed diagnostic tables** for attempts 1-5 live in [res/_archived/](res/_archived/) (the original training reports — preserved but explicitly marked superseded).

**Lessons that govern any future fine-tune attempt:**

1. The 150-head BCE loss is a poison pill on sparse-label data — heads with zero positives get pushed to 0.
2. Even masking the loss isn't enough — the backbone still drifts and the calibration shifts.
3. State-dict surgery on classifier rows alone doesn't transfer gains; the head rows are entangled with their training backbone.
4. **The cleanest "preserve both worlds" answer is to NOT touch the deep model**. Use two-checkpoint inference (`DualHeadECGFounder`) or downstream gates (per-class LR rankers, see [PERCLASS_FP_FINETUNED.md](PERCLASS_FP_FINETUNED.md)) instead.

**Paper-aligned analysis: see [FINETUNE.md](FINETUNE.md).** The ECGFounder paper's own fine-tune strategy (§2.5) explicitly **discards** the 150-class head on every fine-tune — adapt-to-new-task, not preserve-original-task. So the paper doesn't claim to address our preservation problem, and any future fine-tune should follow §4 of FINETUNE.md: new head, new task, new checkpoint, route via `DualHeadECGFounder`.

---

## 4. MIMIC partial-download pipeline (not yet executed)

**Goal**: pull 10k positives + 10k negatives per head for the 9 unified Founder heads from MIMIC-IV-ECG. Auto-interpretation labels from `machine_measurements.csv`.

**Status**: scripts written, ready to run once PhysioNet credentialed access exists.

**Critical clarification**: the URL the user originally cited (`mimiciv/3.1/`) is MIMIC-IV CORE (hospital tables, no waveforms). The pipeline targets `mimic-iv-ecg/1.0/` which has the ECG `.dat` files.

**4-step pipeline** (`scripts/mimic/partial_download.py`):
1. Download metadata (machine_measurements.csv + record_list.csv) — ~250 MB
2. Apply 9-head label map to free-text reports; select 10k+10k per head
3. Download only the selected ~125k waveforms — ~150 GB
4. Derive 4-angle (45/60/75/90°) npz files matching the existing fuzzylead2 schema

**Critical detail in the label mapper** ([scripts/mimic/mimic_label_map.py](scripts/mimic/mimic_label_map.py)): "ventricular tachycardia" is a substring of "supraventricular tachycardia". The mapper masks "supraventricular" with a sentinel before checking V-pattern keywords. **10/10 unit tests pass.**

**Prerequisites for the operator**: PhysioNet account → CITI training → signed MIMIC DUA → credentialing approval. Takes ~1 day.

---

## 5. Cloud demo — Streamlit MVP (planning complete, phase 1 pending)

**Documents (read in order):**
1. [docs/cloud_demo/README.md](docs/cloud_demo/README.md) — orchestration / navigation
2. [docs/cloud_demo/BEGINNER.md](docs/cloud_demo/BEGINNER.md) — first-timer guide; recommends Phase 0 (local Streamlit, skip AWS)
3. [docs/cloud_demo/SETUP.md](docs/cloud_demo/SETUP.md) — step-by-step install + ops + troubleshooting + cost summary
4. [docs/cloud_demo/STREAMLIT_APP.md](docs/cloud_demo/STREAMLIT_APP.md) — design doc (full code stubs, ~150 LoC budget)

**Key decisions baked in:**

| Decision | Rationale |
|---|---|
| Framework: **Streamlit** (single-file monolithic) | After review, chose time-to-first-demo over backend/frontend separation. Pivot is documented and reversible — the original FastAPI plan lives at [docs/cloud_demo/_archived/](docs/cloud_demo/_archived/). |
| Instance: **t3.large** (NOT t3.medium) | RAM budget shows ~3.6 GB worst-case; t3.medium leaves only ~300 MB headroom → OOM risk |
| Model: **`DualHeadECGFounder(routing='ptbxl_specific')`** | Production stack, no further fine-tuning |
| Waveform render: **`st.line_chart`** | Interactive (pan/zoom/hover); drops matplotlib from deps |
| File support: **CSV / JSON / ZIP (.dat+.hea)** | Multi-format; auto-extract Lead II from multi-lead with banner |
| Suppression toggle: **rejected** | Keep minimal — demo shows raw model output |
| Sample-data buttons: **accepted** | 4 buttons sidebar, sourced from existing fzark dataset |
| Auth / TLS / DB / S3: **none** | Internal MVP only; VPN-gated |

**Phase 1 implementation when greenlit**: write `app/app.py` (~150 LoC), assemble 4 sample JSONs in `app/sample_data/`, drop `deploy/ecg-demo.service`, then follow [SETUP.md](docs/cloud_demo/SETUP.md) Phases A-F to deploy.

**Open questions deferred to phase 1**:
1. Branding (logo? watermark?)
2. WFDB ZIP layout confirmation with one operator
3. Specific 4 fzark records to use as samples (selection criteria in `STREAMLIT_APP.md` §15.3)

---

## 6. Top-level repo doc map

Doc-by-doc current status. Read the canonical one and skip the rest unless investigating a specific past decision.

### Currently authoritative

| File | Purpose | Status |
|---|---|---|
| [PROJECT_STATE.md](PROJECT_STATE.md) | This file — session-spanning state snapshot | **start here** |
| [res/GLOBAL_LABEL_MAP.md](res/GLOBAL_LABEL_MAP.md) | All label mappings (Founder ↔ Fzark ↔ PTB-XL ↔ MIT-BIH ↔ fuzzylead2) | canonical |
| [res/cross_dataset_supp_off/cross_dataset_performance_supp_off.md](res/cross_dataset_supp_off/cross_dataset_performance_supp_off.md) | Performance across all 5 datasets, OFF + ON | canonical |
| [docs/cloud_demo/README.md](docs/cloud_demo/README.md) | Cloud demo orchestration | canonical |
| [README.md](README.md) | Repo top-level README | (pre-existing; may need update to reference PROJECT_STATE.md) |

### Older single-purpose docs (not retired, still referenced)

| File | What it covers | Re-read when |
|---|---|---|
| [PERCLASS_FP_FINETUNED.md](PERCLASS_FP_FINETUNED.md) | The LR-ranker per-class FP suppressor (orthogonal to deep model FT) | Reviewing the "downstream gate" path that succeeded where deep-model FT failed |
| [ARCHITECTURE.md](ARCHITECTURE.md) | Repo high-level architecture | Onboarding |
| [DEVICE_POLICY.md](DEVICE_POLICY.md) | MPS / CUDA / CPU resolution | Debugging device-specific issues |
| [FP_SUPPRESSION_IMPLEMENTATION.md](FP_SUPPRESSION_IMPLEMENTATION.md) | v2 motion/HR/SNR feature gates | Reviewing why suppression-on works |
| [MOTION_ANALYSIS_README.md](MOTION_ANALYSIS_README.md), [accelerometer_ecg_alignment_evaluation.md](accelerometer_ecg_alignment_evaluation.md), [motion_comparison_tp_vs_fp.md](motion_comparison_tp_vs_fp.md) | Motion-feature analysis | Investigating motion-related FPs |
| [SINUS_TACHY_FILTER_PRESETS.md](SINUS_TACHY_FILTER_PRESETS.md), [AFIB_FILTER_RECOMMENDATIONS.md](AFIB_FILTER_RECOMMENDATIONS.md), [AFIB_NONMOTION_FEATURE_ANALYSIS.md](AFIB_NONMOTION_FEATURE_ANALYSIS.md) | Class-specific filter tuning | Adjusting v2 filter parameters |
| [ALLCLASS_FP_COMPARISON.md](ALLCLASS_FP_COMPARISON.md), [ALLCLASS_FULL_SUPPRESSION_COMPARISON.md](ALLCLASS_FULL_SUPPRESSION_COMPARISON.md), [MODEL_COMPARISON_FP_DATASET.md](MODEL_COMPARISON_FP_DATASET.md) | Historical comparison runs | Auditing older fzark numbers |
| [ambulatory_performance_evaluation.md](ambulatory_performance_evaluation.md) | Production-distribution eval | Comparing fzark to other ambulatory datasets |
| [res/baseline_performance_report.md](res/baseline_performance_report.md) | Baseline (pre-V3.1) performance | Auditing the pre-recovery numbers |
| [res/preprocessing_consolidation_plan.md](res/preprocessing_consolidation_plan.md), [res/stage1_vs_existing_plan_comparison.md](res/stage1_vs_existing_plan_comparison.md), [res/system_cleanup_plan.md](res/system_cleanup_plan.md) | Older planning docs (mostly executed) | Historical reference |

### Archived (DO NOT consult for current behavior)

[res/_archived/](res/_archived/) — six retired strategy docs + the source Excel. See its README for the manifest.
[docs/cloud_demo/_archived/](docs/cloud_demo/_archived/) — the FastAPI alternative plan (re-promote if separation becomes a hard requirement again).

---

## 7. Open questions across all streams

Two work streams have unresolved open questions; everything else is decided.

### Cloud demo (3 deferred to phase 1)

1. Branding (logo, watermark, primary color)
2. WFDB ZIP upload layout confirmation
3. Specific 4 fzark records for `app/sample_data/`

### MIMIC download (3 operational)

1. PhysioNet credentialing — not yet done by anyone on the team
2. ICD-coded labels via MIMIC-IV CORE join — out of scope for now; can be added later
3. Disk for ~150 GB waveforms — needs a designated machine with space

### Fine-tune work — **no open questions**. All approaches retired; production stack is fixed.

---

## 8. Recommended entry points for next session

By role / intent:

| If you want to… | Start with |
|---|---|
| Pick up the cloud demo | [docs/cloud_demo/BEGINNER.md](docs/cloud_demo/BEGINNER.md) — try Phase 0 first |
| Implement `app/app.py` (phase 1) | [docs/cloud_demo/STREAMLIT_APP.md](docs/cloud_demo/STREAMLIT_APP.md) §7 has full pseudocode |
| Run the MIMIC download | [data/mimic-iv-ecg/README.md](data/mimic-iv-ecg/README.md) — credential first |
| Understand the model behavior | [res/GLOBAL_LABEL_MAP.md](res/GLOBAL_LABEL_MAP.md) + [res/cross_dataset_supp_off/cross_dataset_performance_supp_off.md](res/cross_dataset_supp_off/cross_dataset_performance_supp_off.md) |
| Audit why fine-tuning was retired | §3 of this doc + [res/_archived/](res/_archived/) |
| Add a new fzark event to the ontology | [label_config.py](label_config.py) + add a test in [tests/test_ontology.py](tests/test_ontology.py) |
| Tune the v2 FP suppressor | [multiclass_fp_suppression.py](multiclass_fp_suppression.py) + [PERCLASS_FP_FINETUNED.md](PERCLASS_FP_FINETUNED.md) |

---

## 9. Things to NOT redo

These were considered and explicitly rejected (or already executed and retired). Don't burn time re-investigating without a new signal:

- Vanilla full fine-tune on PTB-XL-derived single-lead → catastrophic forgetting (see §3 attempt #1)
- Masked-loss FT alone → backbone drift breaks active heads (§3 attempt #2)
- State-dict surgery on classifier rows → gains live in backbone, not heads (§3 attempts #3/#4)
- FastAPI + Vanilla JS cloud demo → pivoted to Streamlit for time-to-first-demo (preserved at `docs/cloud_demo/_archived/`)
- `t3.medium` for the demo instance → OOM risk; use `t3.large` (`docs/cloud_demo/STREAMLIT_APP.md` §12)
- Suppression toggle in the demo UI → keep minimal (`STREAMLIT_APP.md` §14)
- Pre-staging sample files from non-fzark sources → format-compatibility risk; use fzark JSONs

---

*This document is the session-context replacement. Future sessions can start fresh by reading this + whichever §6 doc is relevant to the immediate task.*
