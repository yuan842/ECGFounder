# Reclassification Plan Comparison: Stage 1 vs Existing Plan

**Date**: 2026-05-27
**Purpose**: Evaluate the Stage 1 reclassification plan (`label_reclassification_plan_stage1.md`) against the existing multi-dataset plan (`res/label_reclassification_plan.md`) and provide recommendations.

---

## 1. Executive Summary

The Stage 1 plan is a **disciplined, production-ready subset** of the existing plan. It intentionally narrows scope to the 10 event types with unambiguous single-head mappings, while introducing significant architectural improvements (typed ontology, CI tests, integrity checks, rollback plan). The existing plan is broader (19 event types, 3 datasets) but ships everything in one pass without staged validation gates.

**Recommendation**: **Adopt the Stage 1 plan as the implementation vehicle**, with three amendments:
1. Include V Couplet idx 90 as a simple off-by-one fix (not multi-head) in Stage 1
2. Include Prolonged RR idx 80 as an off-by-one fix in Stage 1, but flag it as `semantic_match="poor"` and add a NOTE that it is vocabulary-limited
3. Port the PTB-XL and MIT-BIH sections from the existing plan into a Stage 3 scope

---

## 2. Scope Comparison

| Dimension | Stage 1 Plan | Existing Plan |
|---|---|---|
| **Datasets** | fzark only | fzark + PTB-XL + MIT-BIH |
| **Event types mapped** | 10 (single-head only) | 19 (including multi-head, partial) |
| **Off-by-one fixes** | 3 of 5 | All 5 |
| **New mappings** | 3 (Bradycardia, ST Elevation, SV Run) | 8 (adds bigeminy, trigeminy, V Couplet multi-head) |
| **Deferred events** | 9 (explicit passthrough set) | 3 (unmappable meta-categories only) |
| **Multi-head strategy** | Deferred to Stage 2 | Included (MAX operator for bigeminy/couplet) |
| **Cross-dataset analysis** | No | Yes (usage matrix, consistency audit) |
| **PTB-XL unmapped terms** | Not addressed | Identified (~6,000 records affected) |
| **MIT-BIH annotation gaps** | Not addressed | Identified (8 beat types, 10 rhythms unmapped) |
| **TP coverage** | 93.5% (34,880 / 37,288) | ~98% (all but meta-categories) |

---

## 3. Off-by-One Fix Comparison

The bug report identified **5 off-by-one errors**. The two plans handle them differently:

| Event Type | Bug Report | Stage 1 Plan | Existing Plan | Analysis |
|---|---|---|---|---|
| SV Couplet (20 -> 19) | Confirmed | **Fixed in Stage 1** | Fixed | Agreement |
| V Run (99 -> 98) | Confirmed | **Fixed in Stage 1** | Fixed | Agreement |
| Pause (143 -> 142) | Confirmed | **Fixed in Stage 1** | Fixed | Agreement |
| V Couplet (91 -> 90) | Confirmed | **Deferred to Stage 2** | Fixed as simple off-by-one | Disagreement |
| Prolonged RR (81 -> 80) | Confirmed | **Deferred to Stage 2** | Fixed with "imperfect" caveat | Disagreement |

### 3.1 V Couplet Disagreement

**Stage 1 rationale**: Defers V Couplet to Stage 2 as a multi-head composite (idx 9 + idx 90, MAX operator). Argues that a couplet (two consecutive PVCs) is better captured by combining the PVC head with the PVC+Fusion head.

**Existing plan rationale**: Treats V Couplet as a straightforward off-by-one fix (91 -> 90). Also proposes multi-head [9, 90] as a secondary enrichment.

**Assessment**: The Stage 1 rationale is **overly conservative here**. The off-by-one correction (91 -> 90) is a pure bug fix that replaces "IN A PATTERN OF BIGEMINY" with "PREMATURE VENTRICULAR AND FUSION COMPLEXES" -- the latter is a substantially better semantic match for ventricular couplets. The multi-head composite [9, 90] is an optional enhancement, not a prerequisite. With only n=36 TPs, even the single-head fix is worth shipping immediately -- reading idx 90 instead of idx 91 is strictly better. **Recommend including in Stage 1.**

### 3.2 Prolonged RR Interval Disagreement

**Stage 1 rationale**: Defers as "vocabulary-limited proxy" because PROLONGED QT (idx 80) is semantically different from Prolonged RR Interval. QT measures repolarization duration; RR measures inter-beat interval.

**Existing plan rationale**: Fixes the off-by-one (81 -> 80) but flags it as "imperfect semantic match" and suggests alternatives (Bradycardia idx 4, Sinus Pause idx 142).

**Assessment**: The Stage 1 plan is **clinically correct** that PROLONGED QT != Prolonged RR. However, the off-by-one fix is still an improvement: idx 80 (PROLONGED QT) is closer to "long cardiac interval" than idx 81 (WITH PROLONGED AV CONDUCTION), which describes conduction delay specifically. The existing plan's multi-head suggestion (idx 142 + idx 4, MAX) is the better long-term solution. **Recommend fixing the off-by-one in Stage 1 with `semantic_match="poor"` annotation, and revisiting with multi-head in Stage 2.**

---

## 4. New Mapping Comparison

Both plans agree on 3 new mappings. The existing plan proposes 5 additional ones:

| Event Type | Stage 1 | Existing Plan | Agreement? |
|---|---|---|---|
| Bradycardia -> idx 4 | Included | Included | Yes |
| ST Elevation -> idx 68 | Included | Included | Yes |
| SV Run -> idx 93 | Included | Included | Yes |
| SV Bigeminy -> idx 16 + 91 | Deferred (multi-head) | Included | Stage 1 deferral is reasonable |
| V Bigeminy -> idx 9 + 91 | Deferred (multi-head) | Included | Stage 1 deferral is reasonable |
| SV Trigeminy -> idx 16 | Deferred (no trigeminy head) | Included (partial) | Stage 1 deferral is reasonable |
| V Trigeminy -> idx 9 | Deferred (no trigeminy head) | Included (partial) | Stage 1 deferral is reasonable |
| Prolonged RR -> idx 80 | Deferred | Included (imperfect) | See section 3.2 |

**Assessment**: The Stage 1 plan correctly identifies bigeminy and trigeminy as qualitatively different from single-head mappings. The existing plan's inclusion of partial-match mappings (trigeminy -> PVC/PAC head) captures morphology but not pattern, which could produce misleadingly high detection rates for the wrong reason. Staging these separately is the right call.

---

## 5. Architecture Comparison

### 5.1 Data Model

| Feature | Stage 1 Plan | Existing Plan |
|---|---|---|
| **Configuration format** | `ClinicalOntologyNode` dataclass | Plain Python `dict` |
| **Type safety** | `DetectionOperator` + `ClinicalRiskTier` enums | No type constraints |
| **Semantic match tracking** | `semantic_match` field per mapping | Comment-only annotations |
| **Multi-head support** | Reserved `MAX`/`AND` operators | Separate `MULTI_HEAD_MAP` dict |
| **Risk classification** | 4-tier enum (CRITICAL/HIGH/MODERATE/LOW) | Not present |
| **Notes/provenance** | `notes` field per mapping | Inline comments |
| **Extensibility** | Stage 2 adds nodes to same structure | Stage 2 would need refactoring |

**Verdict**: Stage 1's ontology dataclass is a **significant improvement**. It makes the mapping self-documenting, type-safe, and extensible. The risk tier classification is especially valuable for prioritizing alert severity in production. The existing plan's plain-dict approach works but would need refactoring when multi-head composites are added.

### 5.2 Detection Logic

| Feature | Stage 1 Plan | Existing Plan |
|---|---|---|
| **Detection function** | `detect(probs, event_type, threshold)` -> `bool | None` | Direct indexing `p[LABEL_MAP[event_type]] > threshold` |
| **Unknown event handling** | Returns `None` for passthrough events | `KeyError` for unmapped events |
| **Multi-head detection** | Deferred (SINGLE only) | `max(p[idx1], p[idx2])` |

**Verdict**: Stage 1's `detect()` function with `None` return for passthrough events is cleaner. The existing plan's direct indexing is simpler but doesn't handle partial-coverage gracefully.

### 5.3 Integrity & Safety

| Feature | Stage 1 Plan | Existing Plan |
|---|---|---|
| **tasks.txt hash pinning** | SHA256 check in `load_tasks()` | Not present |
| **CI tests** | 7 pytest tests | Not present |
| **Rollback plan** | Explicit 3-scenario plan | Not present |
| **Validation criteria** | Per-class min TP/FP + acceptance thresholds | General "re-run and compare" |
| **Passthrough safety** | Explicit `STAGE1_PASSTHROUGH` frozenset | `UNMAPPABLE_EVENTS` set (smaller) |

**Verdict**: Stage 1 is **substantially more production-ready**. The CI tests alone would have prevented the original off-by-one bug. The SHA256 hash pinning guards against vocabulary drift. The rollback plan provides a safety net for regressions.

---

## 6. Clinical Risk Assessment Comparison

The Stage 1 plan introduces clinical risk tiers. The existing plan does not classify risk. Here is the complete risk classification from Stage 1:

| Risk Tier | Events | Clinical Rationale |
|---|---|---|
| **CRITICAL** | Ventricular Run, Pause, ST Elevation | Life-threatening if missed |
| **HIGH** | Atrial Fibrillation, Bradycardia | Significant clinical events requiring intervention |
| **MODERATE** | SV Couplet, SV Run | Actionable but lower urgency |
| **LOW** | Isolated V Beat, Isolated SV Beat, Sinus Tachycardia | Benign or contextual |

**Assessment**: This risk stratification is clinically sound and valuable for:
- Prioritizing which mappings to validate first
- Setting different detection thresholds by risk tier (lower threshold for CRITICAL)
- Alerting severity in production systems
- Regression testing priority (CRITICAL classes get stricter acceptance criteria)

The existing plan has no equivalent, which is a gap.

---

## 7. Validation & Deployment Comparison

| Aspect | Stage 1 Plan | Existing Plan |
|---|---|---|
| **Acceptance criteria** | Per-class TP retention + FP suppression thresholds | "Re-run and compare" |
| **Sample size awareness** | Marks n<50 as "report-only" | Lists TP counts but no threshold |
| **Expected outcomes** | Per-class pre/post predictions | Per-class directional predictions |
| **Rollback triggers** | Regression on unchanged classes | Not specified |
| **Rollback procedure** | Revert 11 scripts, keep ontology file | Not specified |
| **Migration steps** | 8 sequenced steps with dependencies | 5 steps, less detailed |

**Verdict**: Stage 1's validation plan is significantly more rigorous. It distinguishes between statistically meaningful validation (n >= 100 TP) and report-only tracking (SV Run n=26, ST Elevation n=1), which prevents false confidence from small-sample metrics.

---

## 8. Coverage Gap Analysis

### 8.1 What Stage 1 Misses from the Existing Plan

| Gap | Impact | Recommendation |
|---|---|---|
| V Couplet off-by-one fix | 36 TPs remain on wrong head | Include in Stage 1 (see section 3.1) |
| Prolonged RR off-by-one fix | 169 TPs remain on wrong head | Include in Stage 1 with "poor" match flag |
| Bigeminy multi-head | 66 TPs (2 V + 64 SV) unmapped | Acceptable Stage 2 deferral |
| Trigeminy partial map | 92 TPs (16 V + 76 SV) unmapped | Acceptable Stage 2 deferral |
| PTB-XL unmapped terms | ~6,000 records with lost labels | Stage 3 scope |
| MIT-BIH annotation gaps | 8 beat types + 10 rhythms unmapped | Stage 3 scope |
| Cross-dataset consistency | MITDB PAC grouping inconsistency | Stage 3 scope |

### 8.2 What Existing Plan Misses

| Gap | Impact | Present in Stage 1 |
|---|---|---|
| CI test suite | No automated guard against future indexing errors | 7 pytest tests |
| Vocabulary integrity check | No detection if tasks.txt changes | SHA256 hash pinning |
| Clinical risk classification | No prioritization framework | 4-tier risk enum |
| Typed configuration | Refactoring needed for Stage 2 features | Dataclass with enums |
| Rollback plan | No recovery procedure for regressions | 3-scenario rollback |
| Per-class acceptance criteria | No quantitative validation gates | Min TP/FP + thresholds |
| Explicit passthrough handling | `KeyError` for Stage-2 events | `None` return from `detect()` |

---

## 9. Detailed Mapping Reconciliation

### 9.1 Complete mapping comparison (all 19 fzark event types)

| Event Type | Stage 1 Index | Existing Plan Index | Match | Recommended |
|---|---|---|---|---|
| Atrial Fibrillation | 5 | 5 | Identical | 5 |
| Sinus Tachycardia | 6 | 6 | Identical | 6 |
| Isolated Ventricular Beat | 9 | 9 | Identical | 9 |
| Isolated Supraventricular Beat | 16 | 16 | Identical | 16 |
| Supraventricular Couplet | 19 | 19 | Identical | 19 |
| Ventricular Run | 98 | 98 | Identical | 98 |
| Pause | 142 | 142 | Identical | 142 |
| Bradycardia | 4 | 4 | Identical | 4 |
| ST Elevation | 68 | 68 | Identical | 68 |
| Supraventricular Run | 93 | 93 | Identical | 93 |
| Ventricular Couplet | *deferred* | 90 (+ multi [9,90]) | **Divergent** | **90** (Stage 1, single-head) |
| Prolonged RR Interval | *deferred* | 80 (flagged imperfect) | **Divergent** | **80** (Stage 1, match="poor") |
| Ventricular Bigeminy | *deferred* | 9 (+ multi [9,91]) | Consistent | Stage 2 |
| Supraventricular Bigeminy | *deferred* | 16 (+ multi [16,91]) | Consistent | Stage 2 |
| Ventricular Trigeminy | *deferred* | 9 (partial) | Consistent | Stage 2 |
| Supraventricular Trigeminy | *deferred* | 16 (partial) | Consistent | Stage 2 |
| Multiple Event | passthrough | unmappable | Consistent | No mapping |
| Unknown | passthrough | unmappable | Consistent | No mapping |
| Custom Heart Rate | passthrough | unmappable | Consistent | No mapping |

### 9.2 Index verification against tasks.txt (all recommended Stage 1 mappings)

| Index | tasks.txt Line (0-based) | Class Label | Fzark Event | Verified |
|---|---|---|---|---|
| 4 | Line 5 | SINUS BRADYCARDIA | Bradycardia | Correct |
| 5 | Line 6 | ATRIAL FIBRILLATION | Atrial Fibrillation | Correct |
| 6 | Line 7 | SINUS TACHYCARDIA | Sinus Tachycardia | Correct |
| 9 | Line 10 | PREMATURE VENTRICULAR COMPLEXES | Isolated Ventricular Beat | Correct |
| 16 | Line 17 | PREMATURE ATRIAL COMPLEXES | Isolated SV Beat | Correct |
| 19 | Line 20 | PREMATURE SUPRAVENTRICULAR COMPLEXES | SV Couplet | Correct |
| 68 | Line 69 | ST ELEVATION NOW PRESENT IN | ST Elevation | Correct |
| 80 | Line 81 | PROLONGED QT | Prolonged RR Interval | Correct (semantic mismatch flagged) |
| 90 | Line 91 | PREMATURE VENTRICULAR AND FUSION COMPLEXES | Ventricular Couplet | Correct |
| 93 | Line 94 | SUPRAVENTRICULAR TACHYCARDIA | SV Run | Correct |
| 98 | Line 99 | VENTRICULAR TACHYCARDIA | Ventricular Run | Correct |
| 142 | Line 143 | WITH SINUS PAUSE | Pause | Correct |

All 12 indices verified against `tasks.txt` (150 lines, 0-based).

---

## 10. Minor Issues Found in Stage 1 Plan

| Issue | Location | Severity | Detail |
|---|---|---|---|
| Off-by-one count discrepancy | Section 1 vs Section 2.2 | Low | Section 1 says "4 off-by-one index errors" but Section 2.2 lists only 3 fixes. The 4th may refer to V Couplet (deferred). Should clarify. |
| "7 changes" count discrepancy | TL;DR | Low | "Bold rows are the 7 changes" but only 6 rows are bold (rows 5-10). Count should be 6. |
| TASKS_SHA256 placeholder | Section 3, code | Medium | Left as `"<pin to known-good hash>"`. Must be computed before shipping. |
| No `Custom Heart Rate` in TP note | TL;DR table | Low | Sinus Tachycardia is marked "FP-only: 530" but Custom Heart Rate (FP-only: 22) is not listed in the TL;DR; it's only in section 7. |
| Detection operator not used | Section 3, code | Low | `DetectionOperator.SINGLE` is set for all 10 events, making it redundant in Stage 1. Justified by Stage 2 extensibility. |

---

## 11. Recommendations

### 11.1 Primary Recommendation: Adopt Stage 1 with Amendments

Use the Stage 1 plan as the implementation framework. Its ontology dataclass, CI tests, and deployment rigor are substantial improvements over the existing plan's flat-dict approach. Make these amendments:

**Amendment 1 — Include V Couplet off-by-one fix (idx 91 -> 90):**
```python
"Ventricular Couplet": ClinicalOntologyNode(
    canonical_name="PREMATURE VENTRICULAR AND FUSION COMPLEXES",
    ecgfounder_index=90,
    operator=DetectionOperator.SINGLE,
    risk_tier=ClinicalRiskTier.MODERATE,
    semantic_match="good",
    notes="Off-by-one fix: was idx 91 (BIGEMINY). Stage 2 may add multi-head [9, 90].",
),
```
**Rationale**: Pure bug fix. Single-head idx 90 is strictly better than idx 91. Multi-head enrichment is a separate concern for Stage 2.

**Amendment 2 — Include Prolonged RR Interval off-by-one fix (idx 81 -> 80):**
```python
"Prolonged RR Interval": ClinicalOntologyNode(
    canonical_name="PROLONGED QT",
    ecgfounder_index=80,
    operator=DetectionOperator.SINGLE,
    risk_tier=ClinicalRiskTier.MODERATE,
    semantic_match="poor",
    notes="Off-by-one fix: was idx 81 (PROLONGED AV CONDUCTION). "
          "PROLONGED QT is semantically imperfect for RR prolongation. "
          "Stage 2 should evaluate multi-head [142, 4] (MAX) as better proxy.",
),
```
**Rationale**: Even an imperfect match (PROLONGED QT) is better than the current state (PROLONGED AV CONDUCTION). The `semantic_match="poor"` annotation explicitly flags the limitation. The Stage 2 multi-head approach remains the long-term solution.

**Amendment 3 — Update event counts:**
- Update `FZARK_ONTOLOGY` to 12 events (was 10)
- Update `STAGE1_PASSTHROUGH` to remove V Couplet and Prolonged RR (7 events, was 9)
- Update CI test `test_event_set_matches_spec` to expect 12 events
- Update `EXPECTED_LABEL_SUBSTRINGS` with 2 new entries:
  - `"Ventricular Couplet": "premature ventricular"`
  - `"Prolonged RR Interval": "prolonged qt"`

**Amendment 4 — Add `semantic_match="poor"` to quality enum:**
Add a third quality tier for vocabulary-limited mappings that are technically the best available option but have known semantic gaps. The CI test `test_all_stage1_are_exact_or_good` should become `test_all_stage1_are_exact_good_or_poor`.

### 11.2 Staging Roadmap

| Stage | Scope | Events | Key Deliverable |
|---|---|---|---|
| **Stage 1** (this PR) | All off-by-one fixes + exact/good new mappings | 12 events | `label_config.py` with ontology, CI tests, hash pinning |
| **Stage 2** | Multi-head composites + partial matches | +5 events (bigeminy, trigeminy, Prolonged RR multi-head) | `DetectionOperator.MAX`/`AND` implementation |
| **Stage 3** | Cross-dataset alignment | PTB-XL text matching fixes, MIT-BIH annotation expansion | Unified cross-dataset label_config |

### 11.3 Implementation Priority

1. **Immediate** (Stage 1 PR):
   - Create `label_config.py` with amended 12-event ontology
   - Create `tests/test_ontology.py` with 7+ tests
   - Pin `TASKS_SHA256`
   - Update 11 scripts to import from `label_config.py`
   - Re-run TP and FP evaluations

2. **Short-term** (Stage 2):
   - Implement multi-head detection (`DetectionOperator.MAX`)
   - Add bigeminy, trigeminy, enhanced Prolonged RR mappings
   - Evaluate multi-head vs single-head on held-out split

3. **Medium-term** (Stage 3):
   - Fix PTB-XL text matching (~6,000 records recovered)
   - Expand MIT-BIH rhythm annotations (VT, AFL, SBR, etc.)
   - Split MIT-BIH `S`/`j` beats from idx 16 to idx 19
   - Unified cross-dataset validation suite

---

## 12. Amended Stage 1 Summary Table

With the two amendments, Stage 1 covers all 5 off-by-one fixes and all 3 exact/good new mappings:

| # | Event Type | Old Idx | New Idx | Change | Match | Risk | TP n |
|---|---|---|---|---|---|---|---|
| 1 | Atrial Fibrillation | 5 | 5 | unchanged | exact | HIGH | 19,566 |
| 2 | Sinus Tachycardia | 6 | 6 | unchanged | exact | LOW | 0 |
| 3 | Isolated Ventricular Beat | 9 | 9 | unchanged | exact | LOW | 2,283 |
| 4 | Isolated Supraventricular Beat | 16 | 16 | unchanged | exact | LOW | 3,534 |
| 5 | **Supraventricular Couplet** | 20 | **19** | off-by-one fix | good | MODERATE | 2,021 |
| 6 | **Ventricular Run** | 99 | **98** | off-by-one fix | good | CRITICAL | 5,212 |
| 7 | **Pause** | 143 | **142** | off-by-one fix | good | CRITICAL | 82 |
| 8 | **Ventricular Couplet** | 91 | **90** | off-by-one fix | good | MODERATE | 36 |
| 9 | **Prolonged RR Interval** | 81 | **80** | off-by-one fix | poor | MODERATE | 169 |
| 10 | **Bradycardia** | -- | **4** | new mapping | exact | HIGH | 2,155 |
| 11 | **ST Elevation** | -- | **68** | new mapping | exact | CRITICAL | 1 |
| 12 | **Supraventricular Run** | -- | **93** | new mapping | good | MODERATE | 26 |

**Coverage**: 35,085 / 37,288 = **94.1%** of fzark TPs (up from 93.5% in original Stage 1).

### Remaining passthrough events (7):

| Event Type | TP n | Stage 2 Strategy |
|---|---|---|
| Ventricular Bigeminy | 2 | Multi-head [9, 91] MAX |
| Supraventricular Bigeminy | 64 | Multi-head [16, 91] MAX |
| Ventricular Trigeminy | 16 | idx 9 only (no trigeminy head) |
| Supraventricular Trigeminy | 76 | idx 16 only (no trigeminy head) |
| Multiple Event | 116 | No mapping (meta-category) |
| Unknown | 1,929 | No mapping (meta-category) |
| Custom Heart Rate | 0 | No mapping (threshold-based) |

---

## 13. Conclusion

The Stage 1 plan represents a **maturation of the reclassification approach** from a "fix all bugs at once" strategy to a "staged, validated, reversible" deployment. Its key innovations -- typed ontology, CI tests, risk tiers, and hash pinning -- address systemic issues that the existing plan's flat-dict approach cannot prevent.

The two plans **agree on all 10 shared mappings** with zero index conflicts. The only disagreements are about scope (what to defer vs. include), not about correctness. The recommended amendments bring two additional off-by-one fixes into Stage 1, covering all 5 known index errors while maintaining the Stage 1 plan's architectural discipline.

**Bottom line**: The Stage 1 plan is the better implementation vehicle. The existing plan remains the authoritative reference for cross-dataset coverage (PTB-XL, MIT-BIH) and Stage 2 multi-head strategies.

---

*Comparison generated from `label_reclassification_plan_stage1.md` (v2.0) and `res/label_reclassification_plan.md` (2026-05-27).*
