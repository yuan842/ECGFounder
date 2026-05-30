# Global Label Map — Solo Guide

**Authoritative source**: this document.
**Backing code**: [label_config.py](../label_config.py) (`FZARK_ONTOLOGY` + helpers).
**Backing data**: [Dataset Classification Cross-Mapping.xlsx](../res/_archived/Dataset%20Classification%20Cross-Mapping.xlsx) (2026-05-28).
**Status**: This is the **sole** label-mapping reference for the project. All prior strategy / relabeling docs are archived under [res/_archived/](_archived/) and should not be consulted for current behavior — they're kept only as historical record. If anything below contradicts an older document, **this document wins**.

---

## ⛔ GLOBAL HARD RULE — detection scope = 7 labels (6 events + Normal ECG)

The system detects **EXACTLY these 7 labels — nothing else is a valid detection target, anywhere:**

| label | head | tasks.txt head label | threshold |
|---|---|---|---|
| Atrial Fibrillation | 5 | ATRIAL FIBRILLATION | 0.5 |
| Bradycardia | 4 | SINUS BRADYCARDIA | 0.5 |
| Sinus Tachycardia | 6 | SINUS TACHYCARDIA | 0.5 |
| Supraventricular Run | 93 | SUPRAVENTRICULAR TACHYCARDIA | 0.5 |
| Ventricular Run | 98 | VENTRICULAR TACHYCARDIA | 0.5 |
| Pause | 142 | WITH SINUS PAUSE | 0.5 |
| **Normal ECG** | **2** | **NORMAL ECG** | **0.5** |

> **Normal ECG (2026-05-29)** is a backbone head (the "normal" reference, 43.6% of PTB-XL), **not a fzark arrhythmia event** — it is in `SCOPE_EVENT_TO_HEAD`/`DETECTION_SCOPE` but **not** in `FZARK_LABEL_MAP`, and is exempt from the fzark-consistency check.
> **All 7 heads use the 0.5 default.** The earlier noise-floor overrides (93→0.040, 142→0.006) were reverted — fragile/device-specific. Consequence: at single-lead, heads 93/98/142 stay effectively silent (0% sensitivity); detecting SV-Run/V-Run/Pause needs the fine-tuned head, not a low threshold.

- **Source of truth**: `label_config.SCOPE_EVENT_TO_HEAD` (label→head) / `SCOPE_EVENTS` / `DETECTION_SCOPE`, enforced by an **import-time assertion** (`_assert_scope_consistency`) — the module fails to import if the scope ever drifts.
- `detect()` / `detect_index()` return `None` (out-of-scope, *not* a false negative) for any other head; the FP suppressor passes out-of-scope events through; eval scripts iterate `scope_indices()` only.
- The full `FZARK_ONTOLOGY` (13 events) is retained **only for label MAPPING** — it is **not** the detection set. The §-tables below describe mapping; detection is bounded by the 7 labels above.
- To change the scope: edit `SCOPE_EVENT_TO_HEAD` **and** `DETECTION_SCOPE` together (the assertion enforces they agree) — and update this section.

---

## 0. What the ECGFounder model emits

`model(x)` returns `(batch, 150)` raw logits from a single `nn.Linear(1024, 150)` head. No softmax, no sigmoid, no internal classification — this is a **multi-label** classifier. Callers apply `torch.sigmoid` externally to obtain 150 independent probabilities, each in [0, 1] with no sum-to-one constraint. A single record routinely fires multiple heads simultaneously (mean ≈ 8 heads at t=0.5 on fzark TPs).

All clinical interpretation — which head means what, which dataset's events map where, how to combine multi-head outputs, when to suppress — lives in `label_config.py` and the downstream filters. Not in the model weights.

---

## 1. The 13 fzark events → 10 unique 150-class heads (v3.1)

The Excel cross-mapping in §1 ratifies this exact routing. Every fzark `Event Type` either maps to one head (single-head route) or is in `FZARK_UNMAPPABLE` (passthrough — model has no useful head for it). No multi-head logic, no operators.

### 1.1 Single-head routes (13 events → 10 heads)

| Founder idx | Founder head (= PTB-XL label) | Fzark Event Type(s) | Risk tier | Mapping note |
|---|---|---|---|---|
| 4 | SINUS BRADYCARDIA | Bradycardia | HIGH | Exact match |
| 5 | ATRIAL FIBRILLATION | Atrial Fibrillation | HIGH | Exact match |
| 6 | SINUS TACHYCARDIA | Sinus Tachycardia | LOW | Exact match |
| 9 | PREMATURE VENTRICULAR COMPLEXES | Isolated Ventricular Beat **+** Ventricular Couplet (v3.1) | LOW / MODERATE | Shared PVC head — fires on constituent beats |
| 16 | PREMATURE ATRIAL COMPLEXES | Isolated Supraventricular Beat **+** Supraventricular Trigeminy (v3.1) **+** Supraventricular Bigeminy (v3.1) | LOW | Shared PAC head |
| 19 | PREMATURE SUPRAVENTRICULAR COMPLEXES | Supraventricular Couplet | MODERATE | Good match — broad ectopy bucket |
| 68 | ST ELEVATION NOW PRESENT IN | ST Elevation | CRITICAL | Exact match |
| 93 | SUPRAVENTRICULAR TACHYCARDIA | Supraventricular Run | MODERATE | Good match — brief SVT |
| 98 | VENTRICULAR TACHYCARDIA | Ventricular Run | CRITICAL | Good match — brief NSVT |
| 142 | WITH SINUS PAUSE | Pause | CRITICAL | Exact match |

### 1.2 Passthrough (no model head — handled downstream by FP suppression / pattern detection)

`FZARK_UNMAPPABLE` (in [label_config.py](../label_config.py)):
- Meta categories: `Multiple Event`, `Unknown`, `Custom Heart Rate`
- Composite events with no head firing strongly on their constituent beats or pattern: `Ventricular Bigeminy`, `Ventricular Trigeminy`, `Prolonged RR Interval`

These produce no model prediction at inference time. Pattern detection (if any) is delegated to the v2 FP-suppression filter and downstream rules.

---

## 2. Empirical reliability per head (Excel-derived)

PPV = true positives / (true positives + false positives), measured on the production fzark + ECG-FP-doctor-removed cohorts. Tiering:

- **Reliable** : PPV ≥ 80%
- **Moderate FP** : 20% ≤ PPV < 80%
- **Severe FP** : PPV < 20%
- **Insufficient Data** : n < 5 in fzark TPs

| idx | Head | TP | FP | PPV | Reliability | Risk |
|---:|---|---:|---:|---:|---|---|
| 5 | ATRIAL FIBRILLATION | 19,566 | 1,793 | **91.6%** | **Reliable** ✅ | HIGH |
| 98 | VENTRICULAR TACHYCARDIA | 5,212 | 3,170 | 62.2% | Moderate FP ⚠️ | CRITICAL |
| 4 | SINUS BRADYCARDIA | 2,155 | 5,941 | 26.6% | Moderate FP ⚠️ | HIGH |
| 19 | PREMATURE SUPRAVENTRICULAR COMPLEXES | 2,021 | 10,061 | 16.7% | Severe FP ❌ | MODERATE |
| 9 | PVC (IVB + V Couplet) | 2,319 | 12,877 | 15.3% | Severe FP ❌ | LOW–MODERATE |
| 142 | WITH SINUS PAUSE | 82 | 786 | 9.4% | Severe FP ❌ | CRITICAL |
| 16 | PAC (ISB + SV Trig + SV Big) | 3,674 | 44,680 | 7.6% | Severe FP ❌ | LOW |
| 93 | SUPRAVENTRICULAR TACHYCARDIA | 26 | 21,263 | 0.1% | Severe FP ❌ | MODERATE |
| 6 | SINUS TACHYCARDIA | 0 (no TPs in fzark) | 530 | 0.0% | Severe FP ❌ | LOW |
| 68 | ST ELEVATION NOW PRESENT IN | 1 | 0 | 100% | Insufficient Data ⚠️ | CRITICAL |

**Reading the reliability column:** PPV is the headline metric for whether a given alert from this head is real. Only AFib qualifies as a "Reliable" head; VT and Bradycardia are usable with downstream FP gating; the rest emit far more false alerts than real ones at t=0.5 and must be paired with the v2 FP-suppression layer or a per-class threshold tighter than 0.5 before serving end-users.

---

## 3. Scientific / clinical descriptions per head

Sourced verbatim from the Excel "scientific_clinical_support" column.

| idx | Head | Clinical description |
|---:|---|---|
| 4 | SINUS BRADYCARDIA | SA node pacing at a rate <60 bpm. Clinically maps directly to the generalized "Bradycardia" alerting event. |
| 5 | ATRIAL FIBRILLATION | Irregularly irregular atrial rhythm lacking distinct P waves. Maps directly to "Atrial Fibrillation" continuous rhythm alerts. |
| 6 | SINUS TACHYCARDIA | SA node pacing at a rate >100 bpm. Clinically maps directly to "Sinus Tachycardia" physiological arousal or stress alerts. |
| 9 | PREMATURE VENTRICULAR COMPLEXES | Ectopic beats originating from ventricles. Mapped to "Isolated Ventricular Beat" and "Ventricular Couplet" as these are the exact temporal event manifestations of PVC burden. |
| 16 | PREMATURE ATRIAL COMPLEXES | Ectopic supraventricular beats. Mapped to "Isolated Supraventricular Beat", "Bigeminy", and "Trigeminy", as these represent the isolated and patterned temporal variations of atrial ectopy. |
| 19 | PREMATURE SUPRAVENTRICULAR COMPLEXES | Broad category for early beats above the ventricles. Specifically mapped to "Supraventricular Couplet" to capture paired ectopic firing before escalating to a run/tachycardia. |
| 68 | ST ELEVATION NOW PRESENT IN | Acute transmural myocardial ischemia/injury. Clinically maps directly to the acute "ST Elevation" Fzark alert signaling potential STEMI. |
| 93 | SUPRAVENTRICULAR TACHYCARDIA | Rapid rhythm (usually >150 bpm) originating above the ventricles. Mapped to "Supraventricular Run" which is the algorithmic detection of 3+ consecutive SVTs. |
| 98 | VENTRICULAR TACHYCARDIA | Potentially lethal rapid rhythm originating from ventricles. Mapped to "Ventricular Run", the exact algorithmic trigger for 3+ consecutive PVCs. |
| 142 | WITH SINUS PAUSE | Failure of the SA node to fire for >2 seconds. Mapped to "Pause", which is the critical algorithmic alert for transient asystole. |

---

## 4. MIT-BIH single-index map

Same v3.1 single-head invariant applies. See `MITDB_BEAT_MAP` and `MITDB_RHYTHM_MAP` in [label_config.py](../label_config.py).

### 4.1 Beat-annotation symbols

| Symbol | Founder idx | Head |
|---|---|---|
| `V` | 9 | PREMATURE VENTRICULAR COMPLEXES |
| `L` | 20 | LEFT BUNDLE BRANCH BLOCK |
| `R` | 11 | RIGHT BUNDLE BRANCH BLOCK |
| `A`, `a` | 16 | PREMATURE ATRIAL COMPLEXES |
| `S`, `j` | 19 | PREMATURE SUPRAVENTRICULAR COMPLEXES |

### 4.2 Rhythm-annotation tokens

| Token | Founder idx | Head |
|---|---|---|
| `(AFIB` | 5 | ATRIAL FIBRILLATION |
| `(VT` | 98 | VENTRICULAR TACHYCARDIA |
| `(AFL` | 32 | ATRIAL FLUTTER |
| `(SVTA` | 93 | SUPRAVENTRICULAR TACHYCARDIA |
| `(SBR` | 4 | SINUS BRADYCARDIA |

Composite rhythms `(B`, `(T`, `(AB` are **intentionally omitted** (v3 invariant — bigeminy/trigeminy never resolve to a single head).

### 4.3 Default-normal indices

`MITDB_DEFAULT_NORMAL = (1, 2)` — set to 1.0 when a segment has no AFib rhythm AND no abnormal beats:
- idx 1 = NORMAL SINUS RHYTHM
- idx 2 = NORMAL ECG

---

## 5. PTB-XL active class set

`PTBXL_ACTIVE_CLASSES` in [label_config.py](../label_config.py) — 31 of 150 indices with ≥1 positive sample in `csv/ptbxl_label.csv` (n=21,799).

Active indices: `{2, 3, 4, 5, 6, 9, 11, 12, 13, 15, 17, 20, 24, 26, 30, 32, 36, 40, 50, 54, 60, 61, 70, 78, 79, 82, 93, 98, 101, 107, 112}`.

PTB-XL labels are pre-computed as binary 150-vectors in the `label` column of `csv/ptbxl_label.csv`. v3 introduced no on-the-fly mapping changes — this set is exposed for metric averaging restricted to active classes.

### 5.1 PTB-XL ∩ Fzark — the six production-critical heads

These are the Founder heads with labeled data in **both** PTB-XL and fzark. They are the only heads with parallel validation across the two cohorts:

| idx | Head | PTB-XL positives | Fzark TPs | Fzark FPs | PPV | Reliability |
|---:|---|---:|---:|---:|---:|---|
| 5 | ATRIAL FIBRILLATION | 1,514 | 19,566 | 1,793 | 91.6% | Reliable |
| 4 | SINUS BRADYCARDIA | 637 | 2,155 | 5,941 | 26.6% | Moderate FP |
| 6 | SINUS TACHYCARDIA | 826 | 0 | 530 | 0.0% | Severe FP |
| 9 | PVC | 1,143 | 2,319 | 12,877 | 15.3% | Severe FP |
| 93 | SVT | 42 | 26 | 21,263 | 0.1% | Severe FP |
| 98 | VT | 42 | 5,212 | 3,170 | 62.2% | Moderate FP |

This is the **6-head core** — the natural scope for any fine-tune that wants paired labels in both clinical (PTB-XL) and ambulatory (fzark) distributions.

---

## 6. Helper API

From [label_config.py](../label_config.py):

```python
from label_config import (
    FZARK_ONTOLOGY, FZARK_LABEL_MAP, FZARK_UNMAPPABLE,
    MITDB_BEAT_MAP, MITDB_RHYTHM_MAP, MITDB_DEFAULT_NORMAL,
    PTBXL_ACTIVE_CLASSES,
    ClinicalRiskTier, ClinicalReliability, ClinicalOntologyNode,
    get_index, is_supported, detect,
)

# Event → Founder idx
get_index("Atrial Fibrillation")      # → 5
get_index("Ventricular Couplet")      # → 9 (v3.1 shared PVC head)
get_index("Prolonged RR Interval")    # → None (in FZARK_UNMAPPABLE)

# Per-event metadata (incl. PPV, reliability, clinical description)
node = FZARK_ONTOLOGY["Atrial Fibrillation"]
node.fzark_ppv_pct           # 91.6
node.clinical_reliability    # ClinicalReliability.RELIABLE
node.clinical_support        # "Irregularly irregular atrial rhythm…"
node.risk_tier               # ClinicalRiskTier.HIGH

# Threshold detection
detect(probs, "Atrial Fibrillation", threshold=0.5)   # True / False / None
```

### CI guarantees ([tests/test_ontology.py](../tests/test_ontology.py), 40 tests)

- `tasks.txt` SHA256 pinned to canonical hash; mismatch raises.
- V3.1 commits to exactly **13 supported events** mapping to 10 unique heads.
- Every mapped index is `0 ≤ idx < 150` and lookup returns a head whose name contains the expected substring.
- MIT-BIH `S` / `j` route to idx 19 (PSVC), not idx 16 (PAC) — v3 fix.
- Composite rhythms `(B`, `(T`, `(AB` are absent from `MITDB_RHYTHM_MAP`.
- V3.1 head-sharing recoveries are explicitly tested (SV Trig → 16, SV Big → 16, V Couplet → 9).

---

## 7. Glossary of supporting code locations

| Concern | File |
|---|---|
| Single source of truth — mappings + reliability | [label_config.py](../label_config.py) |
| 150-class vocabulary | [tasks.txt](../tasks.txt) |
| Net1D architecture (raw-logit forward) | [net1d.py](../net1d.py) |
| Checkpoint loading | [checkpoints.py](../checkpoints.py) |
| Preprocessing (per dataset) | [preprocessing.py](../preprocessing.py) |
| FP suppression v2 (motion/HR/SNR feature gates) | [multiclass_fp_suppression.py](../multiclass_fp_suppression.py) |
| Two-checkpoint inference wrapper | [dual_head_ecgfounder.py](../dual_head_ecgfounder.py) |
| Multi-checkpoint per-head router | [multi_head_router.py](../multi_head_router.py) |
| Ontology test suite | [tests/test_ontology.py](../tests/test_ontology.py) |

---

## 8. Archived documents

Historical strategy / planning / experimental documents have been moved to [res/_archived/](_archived/). They include the v3 reclassification plan, the off-by-one bug report, two iterations of fine-tune strategy, the vanilla and masked-loss fuzzylead2 training reports, and similar materials. **They are kept for historical reference only and are not consulted for current behavior** — this document is the sole guide.

If you need to revisit a past decision, [res/_archived/README.md](_archived/README.md) gives the manifest with dates and a one-line summary of each.
