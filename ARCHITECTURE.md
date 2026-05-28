# System Architecture

**Last updated**: 2026-05-27
**Branch**: `cleanup-12lead-finetune-v1`
**Model surface**: ECGFounder Net1D (single 1-lead checkpoint, 150-class sigmoid output)

---

## 1. TL;DR

Single-lead ECG arrhythmia evaluation pipeline built around one model
(`1_lead_ECGFounder.pth`, 124 MB), one preprocessing pipeline (`preprocessing.ECGPreprocessor`),
one label ontology (`label_config`), and one optional FP-suppression filter
(`multiclass_fp_suppression`).

Three external datasets are supported uniformly: **ecg-tp-fzark** /
**ecg-fp-doctor removed1** (Vivalink Holter JSON @ 128 Hz), **PTB-XL**
(WFDB @ 500 Hz), and **MIT-BIH** (WFDB @ 360 Hz). All routes converge on
`(1, 5000)` float32 tensors @ 500 Hz for inference.

```
                ┌───────────────────────────────────────────────┐
                │              Raw ECG recording                  │
                │  (JSON sidecar, WFDB record, or live stream)    │
                └────────────────────────┬────────────────────────┘
                                         │
                ┌────────────────────────▼────────────────────────┐
                │           preprocessing.ECGPreprocessor          │
                │  notch → bandpass → baseline → resample →        │
                │  crop/pad → winsorize-zscore  → (1, 5000) float  │
                └────────────────────────┬────────────────────────┘
                                         │
                ┌────────────────────────▼────────────────────────┐
                │                   net1d.Net1D                    │
                │       7-stage CNN + SE blocks (loaded via         │
                │       checkpoints.load_ecgfounder)                │
                └────────────────────────┬────────────────────────┘
                                         │   150-dim sigmoid
                ┌────────────────────────▼────────────────────────┐
                │   label_config: map(event_name → idx) → prob[idx] │
                └────────────────────────┬────────────────────────┘
                                         │
                ┌────────────────────────▼────────────────────────┐
                │       multiclass_fp_suppression                   │
                │  Feature-gate filter (motion / HR / SNR).         │
                │  Toggle: enabled=True / False.                    │
                └────────────────────────┬────────────────────────┘
                                         │
                                         ▼
                                  Per-class metrics
```

---

## 2. Module Layer

Eight first-party modules at repo root. Dependency direction is strictly
downward (no cycles).

```
                     ┌───────────────────────┐
                     │  device_utils.py      │   resolve_device() → MPS/CUDA/CPU
                     └───────────┬───────────┘
                                 │
                     ┌───────────▼───────────┐
                     │  net1d.py             │   Net1D architecture
                     └───────────┬───────────┘
                                 │
       ┌─────────────────────────┼─────────────────────────┐
       │                         │                         │
┌──────▼───────┐         ┌───────▼────────┐       ┌────────▼──────────┐
│ preprocessing│         │ checkpoints.py │       │ label_config.py    │
│   .py        │         │  load_ecgfounder│      │  FZARK_ONTOLOGY    │
│              │         │  ensure_checkpoint│    │  MITDB_*_MAP       │
└──────┬───────┘         └───────┬────────┘       │  PTBXL_ACTIVE_*    │
       │                         │                │  TASKS_SHA256      │
       │                         │                └────────┬──────────┘
       │                         │                         │
       │                         │              ┌──────────▼────────────┐
       │                         │              │ multiclass_fp_        │
       │                         │              │  suppression.py        │
       │                         │              │  (uses fzark class     │
       │                         │              │   names only; no       │
       │                         │              │   import dependency)   │
       │                         │              └──────────┬────────────┘
       └─────────────────────────┴──────────────────────────┘
                                 │
                                 ▼
                       Eval / compare scripts
                       (6 entry points)
```

| Module | Lines | Responsibility | External deps |
|---|---|---|---|
| `net1d.py` | 350 | `Net1D` 1D-CNN with bottleneck residual blocks + SE attention | torch, numpy |
| `device_utils.py` | 137 | `resolve_device()` → MPS/CUDA/CPU resolution | torch |
| `preprocessing.py` | 340 | `ECGPreprocessor` unified preprocessing class + factories | scipy, torch |
| `checkpoints.py` | 200 | Download + load ECGFounder weights; canonical config dict | requests, torch |
| `label_config.py` | 240 | v3 single-index ontology + MIT-BIH/PTB-XL maps + SHA256 pin | (stdlib only) |
| `multiclass_fp_suppression.py` | 420 | v2 feature-gate FP filter with on/off toggle | numpy |
| `util.py` | 700 | Eval utilities (dynamic thresholds, bootstrap CIs) | torch, sklearn |

---

## 3. Core Components

### 3.1 Model: `net1d.Net1D` + `checkpoints.load_ecgfounder`

**Single canonical config** lives in `checkpoints.ECGFOUNDER_NET1D_CONFIG`:

```python
ECGFOUNDER_NET1D_CONFIG = dict(
    in_channels=1, base_filters=64, ratio=1,
    filter_list=[64, 160, 160, 400, 400, 1024, 1024],
    m_blocks_list=[2, 2, 2, 3, 3, 4, 4],
    kernel_size=16, stride=2, groups_width=16,
    n_classes=150,
)
```

Network: 7 residual stages, bottleneck 1×1 → grouped 1×k → 1×1 blocks with
Squeeze-Excitation gating, Swish activation, no BN, no dropout. Output is 150
independent class probabilities matching `tasks.txt`.

**One entry point** for loading:

```python
from checkpoints import load_ecgfounder
model = load_ecgfounder(device)                      # auto-download if missing
model = load_ecgfounder(device, download_if_missing=False)  # fail fast
```

`load_ecgfounder()` calls `ensure_checkpoint()` first, which:
1. Returns immediately if `checkpoint/1_lead_ECGFounder.pth` exists and is ≥ 100 MB
2. Otherwise parallel-downloads from HuggingFace (8 chunks, ~118 MB)
3. Rejects truncated downloads (sanity floor on file size)

### 3.2 Preprocessing: `preprocessing.ECGPreprocessor`

One class, three factory methods, one pipeline:

```
load → select_lead → notch → bandpass → baseline → resample
     → crop_pad → normalize → torch.float32 (1, 5000)
```

| Factory | fs_in | Powerline | Use case |
|---|---|---|---|
| `ECGPreprocessor.for_fzark()` | 128 Hz | 50 Hz | Vivalink Holter JSON |
| `ECGPreprocessor.for_ptbxl()` | 500 Hz | 50 Hz | German 12-lead WFDB |
| `ECGPreprocessor.for_mitdb()` | 360 Hz | 60 Hz | US 2-lead WFDB (MLII) |

Format-specific loaders:

```python
prep = ECGPreprocessor.for_fzark()
tensor = prep.from_fzark_json(json_path)       # (1, 5000)

prep = ECGPreprocessor.for_mitdb()
tensor = prep.from_wfdb(record_path)           # auto-selects MLII as Lead II
```

Lead selection logic: exact-match on lead name, with `MLII == II` as a
recognized alias for MIT-BIH compatibility.

Filter coefficients are cached per `(fs_in, powerline_hz)` via `lru_cache`.
Stage toggles (`apply_notch`, `apply_bandpass`, `apply_baseline`, `normalize`) and
constants (`TARGET_FS=500`, `TARGET_LEN=5000`, `WIN_LIMITS=(1.5, 98.5)`)
are exposed but rarely overridden in practice.

### 3.3 Label ontology: `label_config`

**v3 single-index design**: every supported event maps to exactly one 150-class
index. No multi-head logic, no MAX/AND operators.

**Schema**:

```python
@dataclass(frozen=True)
class ClinicalOntologyNode:
    canonical_name: str
    ecgfounder_index: int
    risk_tier: ClinicalRiskTier      # CRITICAL | HIGH | MODERATE | LOW
    semantic_match: str              # "exact" | "good"
    notes: str = ""
```

**Three dataset-specific tables**:

| Constant | Purpose | Size |
|---|---|---|
| `FZARK_ONTOLOGY` | fzark / ECG-FP event-name → ontology node | 10 entries |
| `FZARK_LABEL_MAP` | Back-compat flat dict (event → idx) | 10 entries |
| `FZARK_UNMAPPABLE` | Events that pass through with no prediction | 9 entries |
| `MITDB_BEAT_MAP` | WFDB beat symbol → index | 7 entries |
| `MITDB_RHYTHM_MAP` | WFDB rhythm token → index | 5 entries |
| `MITDB_DEFAULT_NORMAL` | Indices set when no abnormality detected | (1, 2) |
| `PTBXL_ACTIVE_CLASSES` | The 31 indices with any positive sample in PTB-XL | 31 entries |

**Integrity guard**: `TASKS_SHA256` pins
`7c087a0d383c7ea9ae9b818ad00a551b029334edfa5f1fbb5036bfbaa0df0c39`. Any drift in
`tasks.txt` (e.g. row reordering) makes `load_tasks()` raise at import time.

**Detection helpers**:

```python
from label_config import detect, get_index, is_supported

idx = get_index("Atrial Fibrillation")        # 5
is_supported("Ventricular Couplet")           # False (composite — passthrough)
detect(probs, "Pause", threshold=0.5)         # True | False | None
```

`detect()` returns `None` for unsupported events. Callers treat `None` as "no
detection available" rather than logging a false negative.

### 3.4 FP suppression: `multiclass_fp_suppression`

The v2 feature-gate filter sits **between the model and the alert system**. It
does not look at the model output — it looks at side-channel features
(accelerometer, R-peaks, SQI).

**Toggleable**:

```python
suppressor = MultiClassFPSuppressor(enabled=True)      # default
suppressor = MultiClassFPSuppressor(enabled=False)     # A/B no-op
```

When `enabled=False`, `suppress_alert()` short-circuits with
`reason="disabled"` **before** any JSON I/O — verified by tests via mocked
`extract_features` call counts.

**Per-class rules** (`ACTIVE_RULES`):

| Event | Gate | Tier |
|---|---|---|
| Atrial Fibrillation | `mean_motion ≤ 5.0 mG` | MAX_TP_RETENTION |
| Bradycardia | `mean_hr_bpm ≤ 56.3 bpm` | MAX_TP_RETENTION |
| Supraventricular Trigeminy | `mean_motion ≥ 15.0 mG` (inverted) | BALANCED |
| Ventricular Trigeminy | `mean_motion ≥ 24.0` AND `snr_proxy > 1.2` | BALANCED |

All other event types are in `PASSTHROUGH_CLASSES` — alerts pass unchanged.

**Aggregate empirical impact**: suppressed-class FP rate drops 58.5% → 3.9%
with 0.7% TP loss on the validation cohort.

**Fail-open semantics**: if feature extraction errors (missing accelerometer,
bad JSON), the alert is kept. `keep=True` with `reason="no_features (fail-open)"`.

### 3.5 Device resolution: `device_utils.resolve_device`

```python
device = resolve_device()         # auto: MPS → CUDA → CPU
device = resolve_device("cpu")    # force
```

Used uniformly across all eval scripts. On Apple Silicon resolves to `mps`,
on Linux+CUDA boxes to `cuda:0`, falls back to CPU.

---

## 4. Data Layer

| Dataset | Source format | Records | Active classes | Loader |
|---|---|---|---|---|
| **ecg-tp-fzark** | Vivalink JSON @ 128 Hz, 1-lead | 37,288 TPs | 10 events (v3) | `prep.from_fzark_json` |
| **ecg-fp-doctor removed1** | Same as fzark | 102,634 FPs | Same | Same |
| **PTB-XL** | WFDB @ 500 Hz, 12-lead | 21,799 records | 31 of 150 | `prep.from_wfdb` (slices Lead II) |
| **MIT-BIH** | WFDB @ 360 Hz, 2-lead | 8,640 segments | 11 (v3, was 6) | `prep.from_wfdb` (selects MLII) |

The JSON datasets also carry **3-axis accelerometer @ 5 Hz** alongside the ECG
— this is what the v2 suppression filter consumes. PTB-XL and MIT-BIH have no
accelerometer; suppression overlay is therefore not applied to those datasets.

---

## 5. Entry Points (6 scripts)

Each script wires the same component stack with dataset-specific glue:

```
device_utils.resolve_device
        ↓
ECGPreprocessor.for_<dataset>()
        ↓
load_ecgfounder(device)
        ↓
forward pass → probs
        ↓
label_config.FZARK_LABEL_MAP / MITDB_*_MAP
        ↓
(optional) MultiClassFPSuppressor(enabled=…)
        ↓
metrics + reports
```

| Script | Dataset | Suppression overlay | CLI flags |
|---|---|---|---|
| `eval_ecg_tprex.py` | fzark TP | no | (none) |
| `eval_doctor_removed.py` | fzark FP | no | (none) |
| `compare_tp_fzark_with_full_suppression.py` | fzark TP | yes | `--per-class`, `--device`, `--seed`, `--suppression {on,off}` |
| `compare_all_classes_with_full_suppression.py` | fzark FP | yes | Same |
| `ptbxl_eval.py` | PTB-XL | no | (none) |
| `mitdb_eval.py` | MIT-BIH | no | (none) |

When `--suppression off` is passed to a compare script, output filenames get a
`_supp_off` suffix so A/B runs don't overwrite each other.

---

## 6. Inference Pipeline (end-to-end)

A single fzark TP record traveling through the system:

```
data/ecg-tp_rex/Atrial Fibrillation/True/middelares-...json
        │
        │  ECGPreprocessor.for_fzark().from_fzark_json(json_path)
        │   1. Concatenate JSON frames → int array
        │   2. Divide by 1000 → mV
        │   3. 50 Hz notch (Q=30)
        │   4. 0.67–40 Hz Butterworth bandpass (order 4, zero-phase)
        │   5. Median baseline removal (0.4s window)
        │   6. Linear resample 128 → 500 Hz
        │   7. Center-crop to 5000 samples
        │   8. Winsorize [1.5, 98.5] then z-score
        │   → torch.float32 (1, 5000)
        ▼
load_ecgfounder(device)  # cached after first call
        │   forward pass
        ▼
sigmoid(logits)  →  probs ∈ [0,1]^150
        │
        │  label_config.FZARK_LABEL_MAP["Atrial Fibrillation"] = 5
        ▼
p = probs[5]      # AFib probability
        │
        │  MultiClassFPSuppressor(enabled=True).suppress_alert(
        │      "Atrial Fibrillation", p, json_path
        │  )
        ▼
SuppressionResult(keep, final_prob, reason, features)
        │
        ▼
Aggregated per-class metrics (TP retention, FP elimination)
```

---

## 7. Configuration Surfaces

Three places to configure behavior:

| Surface | Examples | Who controls |
|---|---|---|
| **Factory methods** | `ECGPreprocessor.for_mitdb()` | Dataset choice → preprocessing knobs |
| **Constructor args** | `MultiClassFPSuppressor(enabled=False)` | Runtime toggle |
| **CLI flags** | `--suppression off`, `--per-class 500` | Per-run overrides |

**Constants** (single source of truth):
- `preprocessing.TARGET_FS = 500` Hz
- `preprocessing.TARGET_LEN = 5000` samples
- `preprocessing.WIN_LIMITS = (1.5, 98.5)`
- `checkpoints.CHECKPOINT_URL` (HuggingFace)
- `checkpoints.CHECKPOINT_PATH = "./checkpoint/1_lead_ECGFounder.pth"`
- `checkpoints.ECGFOUNDER_NET1D_CONFIG`
- `label_config.TASKS_SHA256`
- `multiclass_fp_suppression.ACTIVE_RULES`

---

## 8. Test Coverage

**88 tests** in 4 files, run with `pytest tests/`:

| File | Tests | Coverage |
|---|---|---|
| `tests/test_preprocessing.py` | 32 | Shape/dtype invariants, NaN robustness, factory presets, lead alignment, stage toggles, **byte-parity with legacy `robust_preprocess`** (tolerance 1e-5), loader smoke tests, deprecation shims |
| `tests/test_checkpoints.py` | 9 | Existing-file fast path, too-small-file redownload, missing-file download, truncated-download rejection, fail-fast when `download_if_missing=False`, real-checkpoint forward pass, config contract guard |
| `tests/test_ontology.py` | 36 | tasks.txt SHA256 integrity, per-event index resolution (would have caught the 4 off-by-one bugs), v3 single-head invariant guard, label-map / ontology consistency, MIT-BIH S/j→19 routing, PTB-XL active-class set |
| `tests/test_suppression_toggle.py` | 11 | Default-is-enabled, disabled returns `keep=True` for every `ACTIVE_RULES` class, `extract_features` is NEVER called when disabled, enabled path still extracts features |

Run: `venv/bin/python3 -m pytest tests/`

---

## 9. Repository Layout

```
ECGFounder/
├── ARCHITECTURE.md                       (this file)
├── README.md
├── tasks.txt                             150-class vocabulary, SHA256-pinned
│
├── net1d.py                              Net1D model
├── device_utils.py                       Device resolution
├── preprocessing.py                      Unified ECG preprocessing
├── checkpoints.py                        Download + load model
├── label_config.py                       v3 ontology + dataset maps
├── multiclass_fp_suppression.py          v2 feature-gate filter
├── util.py                               Eval utilities
│
├── eval_ecg_tprex.py                     fzark TP eval (raw model)
├── eval_doctor_removed.py                fzark FP eval (raw model)
├── compare_tp_fzark_with_full_suppression.py    fzark TP with suppression overlay
├── compare_all_classes_with_full_suppression.py fzark FP with suppression overlay
├── ptbxl_eval.py                         PTB-XL eval (1-lead, Lead II)
├── mitdb_eval.py                         MIT-BIH eval (1-lead, MLII)
│
├── standardize_mitdb.py                  MIT-BIH preprocessing CLI
├── split_mitdb.py                        MIT-BIH train/val split + standardize
├── split_dataset.py, split_ecg_tprex.py
├── fast_download.py                      Checkpoint download CLI
├── fast_download_ptbxl.py                PTB-XL data download
│
├── motion_analysis_utilities.py          Motion analysis toolchain
├── motion_comparison_stats.py            (5 scripts total — analysis only,
├── motion_visualization.py                not part of inference path)
├── motion_analysis_report.py
├── comprehensive_motion_analysis.py
├── compare_tp_fp_motion.py
├── compare_tp_fp_sqi_motion.py
├── explore_afib_motion.py
├── optimize_motion_thresholds.py
├── ecg_feature_analysis.py
├── ecg_filter_design.py
├── eval_combo_aligned.py
├── evaluate_combo_filter.py
├── plot_examples.py
├── visualize_db.py
├── explore_ptbxl_db.py
├── list_classes.py
├── list_total_classes.py
├── system_performance.py
│
├── checkpoint/
│   └── 1_lead_ECGFounder.pth             118 MB — only model in the system
│
├── tests/
│   ├── test_preprocessing.py             32 tests
│   ├── test_checkpoints.py               9 tests
│   ├── test_ontology.py                  36 tests
│   └── test_suppression_toggle.py        11 tests
│
├── data/                                 (external; not committed)
│   ├── ecg-tp_rex/
│   ├── ecg_tp_fzark/
│   ├── ecg_fp_doctor removed1/
│   ├── mitdb/
│   └── ptb-xl-…/
│
└── res/                                  Reports and analysis output
    ├── label_reclassification_plan.md
    ├── stage1_vs_existing_plan_comparison.md
    ├── system_cleanup_plan.md
    ├── preprocessing_consolidation_plan.md
    ├── label_map_off_by_one_bug_report.md
    ├── tp_fzark_full_suppression/        Per-class TP retention reports
    ├── fp_allclass_full_suppression/     Per-class FP suppression reports
    ├── eval_lead_ii/                     PTB-XL eval results
    └── mitdb_singlelead/                 MIT-BIH eval results
```

---

## 10. v3 Label Ontology (canonical reference)

10 supported events (single-head), 9 unmappable events (composite or meta):

| Event | Idx | 150-class label | Risk tier | Status |
|---|---|---|---|---|
| Atrial Fibrillation | 5 | ATRIAL FIBRILLATION | HIGH | exact |
| Sinus Tachycardia | 6 | SINUS TACHYCARDIA | LOW | exact |
| Bradycardia | **4** | SINUS BRADYCARDIA | HIGH | exact (new) |
| ST Elevation | **68** | ST ELEVATION NOW PRESENT IN | CRITICAL | exact (new) |
| Isolated Ventricular Beat | 9 | PREMATURE VENTRICULAR COMPLEXES | LOW | exact |
| Isolated Supraventricular Beat | 16 | PREMATURE ATRIAL COMPLEXES | LOW | exact |
| Supraventricular Couplet | **19** | PREMATURE SUPRAVENTRICULAR COMPLEXES | MODERATE | good (off-by-one fix from 20) |
| Ventricular Run | **98** | VENTRICULAR TACHYCARDIA | CRITICAL | good (off-by-one fix from 99) |
| Pause | **142** | WITH SINUS PAUSE | CRITICAL | good (off-by-one fix from 143) |
| Supraventricular Run | **93** | SUPRAVENTRICULAR TACHYCARDIA | MODERATE | good (new) |

**Unmappable** (no model prediction; downstream filter handles pattern detection):
Ventricular Couplet, Ventricular Bigeminy, Supraventricular Bigeminy,
Ventricular Trigeminy, Supraventricular Trigeminy, Prolonged RR Interval,
Multiple Event, Unknown, Custom Heart Rate.

---

## 11. Cross-Dataset Index Consistency

Indices used in more than one dataset (no conflicts):

| Idx | fzark | MIT-BIH | PTB-XL |
|---|---|---|---|
| 4 | Bradycardia | `(SBR` rhythm | SINUS BRADYCARDIA |
| 5 | Atrial Fibrillation | `(AFIB` rhythm | ATRIAL FIBRILLATION |
| 9 | Isolated Ventricular Beat | `V` beat | PREMATURE VENTRICULAR COMPLEXES |
| 11 | — | `R` beat | RIGHT BUNDLE BRANCH BLOCK |
| 16 | Isolated Supraventricular Beat | `A` / `a` beats | — |
| 19 | Supraventricular Couplet | `S` / `j` beats | — |
| 20 | — | `L` beat | LEFT BUNDLE BRANCH BLOCK |
| 32 | — | `(AFL` rhythm | ATRIAL FLUTTER |
| 93 | Supraventricular Run | `(SVTA` rhythm | SUPRAVENTRICULAR TACHYCARDIA |
| 98 | Ventricular Run | `(VT` rhythm | VENTRICULAR TACHYCARDIA |

---

## 12. Resolved Issues (originally flagged in the architecture walkthrough)

| Issue | Resolution |
|---|---|
| **LABEL_MAP off-by-one bugs** (5 wrong indices in 11 scripts) | Consolidated into `label_config.FZARK_LABEL_MAP` with all 4 surviving fixes; SHA256-pinned tasks.txt + 36 ontology tests prevent regression |
| **No shared preprocessing module** (4 inline copies + 2 adapter classes) | Replaced by `preprocessing.ECGPreprocessor` with 32 tests including byte-parity with legacy |
| **Empty `checkpoint/` will fail eval on fresh clone** | `checkpoints.ensure_checkpoint()` auto-downloads; integrated into `load_ecgfounder()` so eval just works |
| **PTB-XL bypassed filtering** (silently different from training pipeline) | `ECGPreprocessor.for_ptbxl()` now applies the same notch/bandpass/baseline as everything else; brings PTB-XL onto the training manifold (metrics will shift — this is expected and desired) |
| **5 copies of `build_model()`** | Consolidated into `checkpoints.load_ecgfounder(device)` with one canonical `ECGFOUNDER_NET1D_CONFIG` dict |
| **MITDB S/j routed to PAC (idx 16) instead of PSVC (idx 19)** | Fixed in `MITDB_BEAT_MAP`; aligned with fzark Supraventricular Couplet mapping |
| **MITDB missing rhythm tokens** (VT, AFL, SVTA, SBR silently dropped) | All 4 added to `MITDB_RHYTHM_MAP`; `TARGET_CLASSES` expanded from 6 → 11 |

---

## 13. Open Items / Future Work

| Item | Notes |
|---|---|
| **No FP suppression on PTB-XL / MIT-BIH** | Suppression filter needs accelerometer; those datasets are ECG-only. Could add a degraded "no-motion" version. |
| **Multi-head composite events deferred** | Bigeminy / trigeminy / couplets / Prolonged RR are intentionally left to the suppression filter (motion + SNR rules). v3 explicitly does not try to detect them at the model layer. |
| **No GPU-side preprocessing** | Filtering runs on CPU via scipy; inference dominates wall time so this isn't currently a bottleneck. |
| **Deprecation shims** (`UniversalECGAdapter`, `RobustECGAdapter`) | Live in `preprocessing.py`; emit `DeprecationWarning`. Remove after one release. |
| **PTB-XL text-matching gaps** (~6,000 records with unmapped diagnostic terms) | Listed in v3 plan §9.3; not yet implemented. Would lift active classes from 31 → ~35. |
| **Multi-dataset unified eval** | Each dataset has its own entry-point script; no single CLI runs the full suite. |

---

## 14. How To...

### Run all evals

```bash
# Auto-downloads checkpoint on first use
python3 eval_ecg_tprex.py
python3 eval_doctor_removed.py
python3 ptbxl_eval.py
python3 mitdb_eval.py
```

### Compare with/without FP suppression

```bash
python3 compare_tp_fzark_with_full_suppression.py --suppression on
python3 compare_tp_fzark_with_full_suppression.py --suppression off
# Output files get _supp_off suffix in the second run
diff res/tp_fzark_full_suppression/tp_fzark_threshold_variants_report.md \
     res/tp_fzark_full_suppression/tp_fzark_threshold_variants_report_supp_off.md
```

### Add a new dataset

1. Add a factory in `preprocessing.py`: `ECGPreprocessor.for_<dataset>()` with the right `powerline_hz` and `target_lead`.
2. Add the dataset → 150-class mapping in `label_config.py` (single-index per the v3 invariant).
3. Add a CI test in `tests/test_ontology.py` enforcing that the indices resolve to expected `tasks.txt` substrings.
4. Add an eval script using `load_ecgfounder()` + `ECGPreprocessor.for_<dataset>()` + the new label map.

### Add a new FP suppression rule

1. Add an entry to `multiclass_fp_suppression.ACTIVE_RULES` keyed by the canonical event name.
2. Ensure the feature you're gating on is computed in `extract_features()`.
3. Add a tier comment indicating MAX_TP_RETENTION / BALANCED / etc.

### Run tests

```bash
venv/bin/python3 -m pytest tests/                # all 88
venv/bin/python3 -m pytest tests/ -v             # verbose
venv/bin/python3 -m pytest tests/test_ontology.py  # specific suite
```

### Verify tasks.txt hasn't drifted

```bash
shasum -a 256 tasks.txt
# Must match: 7c087a0d383c7ea9ae9b818ad00a551b029334edfa5f1fbb5036bfbaa0df0c39
```

---

*Generated from the current state of branch `cleanup-12lead-finetune-v1`. Run
`pytest tests/` (88 tests) to verify all components are functional after any
change.*
