# Project rules — ECGFounder (read first)

## ⛔ GLOBAL HARD RULE — detection scope = 7 labels (6 events + Normal ECG)

The system detects **EXACTLY these 7 labels. Nothing else is a valid detection
target — anywhere in code, evals, or reports:**

| label | ECGFounder head | per-head threshold |
|---|---|---|
| Atrial Fibrillation | 5 | 0.5 |
| Bradycardia | 4 | 0.5 |
| Sinus Tachycardia | 6 | 0.5 |
| Supraventricular Run | 93 | 0.5 |
| Ventricular Run | 98 | 0.5 |
| Pause | 142 | 0.5 |
| Normal ECG | 2 | 0.5 |

- **Source of truth**: `label_config.SCOPE_EVENT_TO_HEAD` (label→head), with `SCOPE_EVENTS`
  and `DETECTION_SCOPE`, kept consistent by an **import-time assertion**
  (`_assert_scope_consistency`). Normal ECG is a backbone head (not a fzark event,
  not in `FZARK_LABEL_MAP`) and is exempt from the fzark-consistency check.
- All 7 heads use the 0.5 default (`HEAD_THRESHOLDS = {}`). Earlier noise-floor overrides
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

## Canonical docs
- Label mapping + hard rule: `res/GLOBAL_LABEL_MAP.md` (sole guide).
- Project state / decisions: `PROJECT_STATE.md`.
- Active branch for v3.1 work: `v3.1-single-lead`.
