# Project rules — ECGFounder (read first)

## ⛔ GLOBAL HARD RULE — detection scope = 6 fzark events

The system detects **EXACTLY these 6 labels. Nothing else is a valid detection
target — anywhere in code, evals, or reports:**

| label | ECGFounder head | per-head threshold |
|---|---|---|
| Atrial Fibrillation | 5 | 0.5 |
| Bradycardia | 4 | 0.5 |
| Sinus Tachycardia | 6 | 0.5 |
| Supraventricular Run | 93 | 0.5 |
| Ventricular Run | 98 | 0.5 |
| Pause | 142 | 0.5 |

- **Source of truth**: `label_config.SCOPE_EVENT_TO_HEAD` (label→head), with `SCOPE_EVENTS`
  and `DETECTION_SCOPE`, kept consistent by an **import-time assertion**
  (`_assert_scope_consistency`). All 6 scope events are fzark arrhythmia events.
  `NON_FZARK_SCOPE_EVENTS` is now **empty** — NORMAL SINUS RHYTHM (1) and NORMAL ECG (2)
  were **removed from scope 2026-06-01** (head 1 had 0 PTB-XL GT / 100% FP; head 2 was
  always subsumed by head 1). They remain valid label-MAPPING targets, just not detection
  targets.
- All 6 heads use the 0.5 default (`HEAD_THRESHOLDS = {}`). Earlier noise-floor overrides
  (93→0.040, 142→0.006) were reverted as fragile; at single-lead, 93/98/142 stay effectively
  silent — detecting them needs the fine-tuned head, not a low threshold.

- **Source of truth**: `label_config.SCOPE_EVENTS` (event names) and `DETECTION_SCOPE`
  (head→label), kept consistent by an **import-time assertion** (`_assert_scope_consistency`).
  The module fails to import if the two ever drift.
- Helpers: `scope_events()`, `scope_indices()`, `event_in_scope()`, `require_in_scope()`,
  `in_scope()`, `head_threshold()`.
- Everything out of scope is inert: `detect()`/`detect_index()` return `None` (out-of-scope,
  not a false negative); the FP suppressor passes out-of-scope events through; eval scripts
  iterate `scope_indices()` only.
- `FZARK_ONTOLOGY` (13 events) is retained **only for label MAPPING**, not detection.
- **To change the scope**: edit `SCOPE_EVENTS` **and** `DETECTION_SCOPE` together, then update
  `res/GLOBAL_LABEL_MAP.md` (§ hard rule) and this file.

**Detection scope ≠ final-classification scope.** The 6-head detection scope above is the
hard rule and is unchanged. For terminal per-sample classification there is a separate
`label_config.FINAL_CLASSIFICATION_SCOPE = DETECTION_SCOPE + {150: "Noisy"}` (helpers
`final_classification_scope()`, `in_final_scope()`). **Noisy (150) is a signal-STATE** emitted
by the S0 Signal-Quality Gate (`overlay.signal_quality_gate`, index 150) when a window is
uninterpretable and detection is skipped — it is **not** a backbone detection head and never
enters `DETECTION_SCOPE`. The import assertion enforces that any `FINAL_STATE_CLASSES` entry is
a `SIGNAL_STATE_LABELS` index (≥150), never a head, so the 6-head rule cannot be diluted. To add
another terminal state later (e.g. High Motion 151), add it to `FINAL_STATE_CLASSES` only.

## ⛔ GLOBAL RULE — never commit or push datasets

Raw datasets, signal dumps, and model weights **must never be committed or pushed.**
They live in `data/` (gitignored) or are regenerable artifacts.

- **Blocked patterns**: anything under `data/`, and `*.npy *.npz *.pth *.pt *.ckpt *.edf
  *.dat *.hea *.mat *.h5 *.hdf5 *.wav`, plus any blob > 50 MB.
- **Enforced two ways**: `.gitignore` (excludes them), **and** a committed pre-commit hook
  `scripts/git-hooks/pre-commit` wired via `git config core.hooksPath scripts/git-hooks` —
  it aborts the commit if a dataset/large binary is staged.
- **Fresh clone**: run `git config core.hooksPath scripts/git-hooks` once to activate the hook.
- **Deliberate exception only**: `ALLOW_DATA=1 git commit ...` (reviewed cases).
- When staging, prefer `git add <specific paths>`; never blanket-add data directories.

## ⛔ GLOBAL RULE — PTB-XL splitting (folds 1–8 / 9 / 10)

**All PTB-XL training/eval uses the authors' patient-stratified fold convention — nothing else:**
**folds 1–8 = train (17,418), fold 9 = validation (2,183), fold 10 = test (2,198).**

- **Validation (9) is for tuning/model-selection; test (10) is reported once, never tuned on** —
  they must stay separate so the test number is an unbiased estimate.
- **Source of truth**: `csv/ptbxl_fold_split.csv` (frozen manifest: ecg_id, patient_id, strat_fold,
  split) + `ptbxl_splits.py` (`mask_for`, `split_of`, `counts`, `assert_no_leakage`). Integrity is
  **asserted at import** (`_assert_split_integrity`) — the module fails to load if the manifest
  drifts from the 1-8/9/10 rule or develops patient leakage.
- Fuzzy loaders partition the existing angle npz by ecg_id: train/val from `train_*deg.npz`
  (folds 1-9) masked to 1-8 / 9; test = `val_*deg.npz` (fold 10). 0 patient leakage across splits.
- **Exception**: a model NOT trained on PTB-XL (e.g. the base backbone) may be evaluated on ALL
  records — no leakage to worry about. Fine-tuned models MUST respect the split.
- To change it: edit the manifest + `FOLD_TO_SPLIT` together (the import assertion enforces agreement).

## Canonical docs
- Label mapping + hard rule: `res/GLOBAL_LABEL_MAP.md` (sole guide).
- Project state / decisions: `PROJECT_STATE.md`.
- Active branch for v3.1 work: `v3.1-single-lead`.
