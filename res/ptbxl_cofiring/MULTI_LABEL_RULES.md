# Multi-label rules — quick reference

**Source:** §F of [REPORT.md](REPORT.md) — derived from PTB-XL co-firing analysis (21,799 records, base backbone, single-lead, t=0.5).
**Scope:** 6 fzark detection heads `{4 Brady, 5 AFib, 6 Sinus Tachy, 93 SVT-Run, 98 V-Run, 142 Pause}`. NORMAL SINUS RHYTHM (1) and NORMAL ECG (2) are **out of scope** (removed 2026-06-01) — not detection targets, though head 1's raw score is still usable as an FP feature (rule 2).
**Frozen on:** 2026-06-01. Regenerate the numbers with `python3 -m scripts.ptbxl_cofiring_{analysis,scope_detail,scores,realness,gt}`.

---

## Suppression / FP-gate rules

1. **Suppress AFib + Sinus-Tachy double-labels.** GT sanctions **zero** AFib+Tachy co-occurrence; the model produces **436** at t=0.5. When both fire, the **Sinus-Tachy is the false member** (co-fire score median 0.64 vs 0.91 standalone; co-fire precision **1.0%**, 4/402). Almost certainly AF-with-rapid-ventricular-response being double-counted — keep AFib, drop Sinus-Tachy.

2. **Use NSR (head 1) as a contradiction flag — as a feature, not a detection.** Head 1 is out of scope (`detect()` returns None), but the backbone still computes its probability. When head 1 scores high alongside an AFib or Sinus-Tachy firing, that arrhythmia is likely false (real "normal sinus rhythm" contradicts an arrhythmia). When head 1 *was* in scope this showed up as: false-AFib co-fired NSR 62%, false-Sinus-Tachy 68%, vs ~5% for true firings. Consume head-1 score inside the FP-suppression layer; never emit it as a label.

3. **Suppress Brady+Tachy and Brady+AFib co-fires.** Both pairs have **0 GT count**; the model produces 102 and 190 respectively. Physiologically contradictory (Brady+Tachy) or unsanctioned (Brady+AFib); drop without losing any GT signal.

4. **Do NOT suppress SVT-Run ↔ V-Run co-firing.** GT identifies the two heads (Jaccard **1.0** on the same 42 records); the model's strong co-firing (Jaccard 0.68) is closer to GT than weaker co-firing would be. ⚠ This likely reflects PTB-XL encoding (one run-type SCP code → both heads) rather than clinical fact — distinguishing the two run heads needs a dataset that separates them.

---

## Reporting / interpretation rules

5. **Counting firings ≠ counting annotations.** No firing-count-to-GT ratio is reliable on PTB-XL:
   - Brady: 5.9× over · Sinus-Tachy: 3.1× over · AFib: 1.41× (close) · SVT-Run / V-Run: 1.31× / 1.69× (close, but record-level match is poor — recall 57%/64%)
   Use record-level recall/precision, never firing totals.

6. **Report per-head precision by role.** Main-role precision is the meaningful number (AFib 0.76, Sinus-Tachy 0.38, Brady 0.17); co-fire precision is the FP rate (AFib 0.15, Sinus-Tachy **0.01**). Aggregated all-fire precision (AFib 0.69, Tachy 0.32) blends the two and is misleading.

---

## Detection-design rules

7. **SVT-Run / V-Run remain ungeneralizable at single-lead.** Co-fire scores stay high (median 0.72 / 0.74, and 0.93–0.97 within the triad) — **no threshold or co-firing rule can split the SVT/VT/Sinus-Tachy cluster**. Distinguishing origin needs the fine-tuned head. Treat any 93 or 98 firing as "fast tachyarrhythmia of uncertain origin," not an origin-specific call.

8. **Pause (142) is invisible on PTB-XL.** 0 GT, 0 predictions. Validation needs another dataset — CINC2015 asystole alarms (asystole→142 mapper exists), fzark Pause events (ecg_tp/ecg_fp).

---

## Quick lookup — pair-level disposition (6-head scope)

| pair | GT count | model count | rule |
|---|---:|---:|---|
| SVT-Run + V-Run | 42 (Jaccard 1.0) | 51 | **keep** (real identity; model ≈right) |
| AFib + Sinus-Tachy | 0 | 436 | **suppress — Tachy is the FP** |
| Brady + AFib | 0 | 190 | **suppress** |
| Brady + Sinus-Tachy | 0 | 102 | **suppress (physiologically impossible)** |
| Sinus-Tachy + V/SVT-Run | 1 / 1 | 71 / 55 | model over-couples, but inseparable at single-lead |
| AFib + V/SVT-Run | 2 / 2 | 18 / 15 | model over-couples, low absolute count |
| any arrhythmia + high NSR(1) score | n/a (1 out of scope) | — | **FP flag — suppress the arrhythmia (rule 2)** |

> **Removed from this policy (2026-06-01):** the former rules about NSR/Normal-ECG *as scope heads* — "NSR internal-only detection," "increase Normal-ECG sensitivity," "Normal-ECG should co-fire rate findings." With heads 1 and 2 out of scope they are no longer detection targets. The single surviving NSR use is rule 2 (head-1 score as an FP feature).
