# Overlay design — scoped detection + multi-label arbitration (Option B)

**Status:** design (no training/implementation yet). **Scope:** 6 fzark detection heads `{4 Brady, 5 AFib, 6 Sinus Tachy, 93 SVT-Run, 98 V-Run, 142 Pause}` (NSR 1 / Normal-ECG 2 out of scope as of 2026-06-01). **Decision:** two learned-vs-deterministic layers after a frozen backbone, with signal-quality moved **upstream of the CNN**.

---

## 1. Pipeline (revised 2026-06-01)

```
raw ECG (10 s, lead II)
      │
      ▼
┌───────────────────────────────┐
│ Preprocessing                  │  resample 500 Hz, filter, normalize
└───────────────────────────────┘
      │
      ▼
┌───────────────────────────────┐
│ SQG  Signal-Quality Gate       │  ★ DEFAULT OFF ★   (SQI + Motion filter)
│  • computes Noisy(150),         │  runs on the INPUT SIGNAL, before the CNN
│    High-Motion(151) flags       │  ON  → gate/flag uninterpretable segments
│  • OFF → pure pass-through      │  OFF → every segment proceeds to the CNN
└───────────────────────────────┘
      │ (signal + optional quality flags)
      ▼
┌───────────────────────────────┐
│ Net1D backbone  (FROZEN, CNN)  │  150 logits — never modified
└───────────────────────────────┘
      │ 150 logits/probs (incl. descriptor scaffold + head-1 NSR)
      ▼
┌───────────────────────────────┐
│ L1  Scope projection (LEARNED) │  150 → 6 calibrated scope probs
│  • frozen backbone             │  pure per-head scorer, NO inter-label logic
└───────────────────────────────┘
      │ 6 calibrated probs + pass-through: head-1 NSR score, (quality flags)
      ▼
┌───────────────────────────────┐
│ L2  Rule arbiter (DETERMINISTIC)│ 6 probs → 6 decisions (+audit)
│  • inter-label logic only       │ enforces MULTI_LABEL_RULES.md
│  • suppress / merge only        │ NO signal-quality stage (moved to SQG)
└───────────────────────────────┘
      │ 6 decisions {fired, score, reason}
      ▼  alerts
```

**Three independent stages, three governance models:** SQG = signal policy (toggle + thresholds), L1 = learned checkpoint, L2 = deterministic rule config. Each updates on its own cadence.

---

## 2. SQG — Signal-Quality Gate  (moved here 2026-06-01; default OFF)

**What moved.** Signal-quality logic (SQI + accelerometer motion) was previously a *post-detection* per-head suppression stage inside `multiclass_fp_suppression.py`. It is relocated to **between preprocessing and the CNN** and runs on the input signal.

**Rationale.** Signal interpretability is a property of the *input*, not of individual label outputs. Deciding up front "is this 10 s interpretable?" is cleaner than detecting on bad signal and second-guessing per head afterward.

**Semantics.**
- **OFF (default, now):** pure pass-through. Quality metrics may still be *computed and logged* (Noisy 150 / High-Motion 151 flags), but nothing is gated — every segment reaches the CNN. This preserves current behavior while the gate is validated.
- **ON (future):** a segment failing the SQI/motion threshold is **not sent to detection** — emit "uninterpretable / low-quality," no scope decision. (Gate = skip, not flag-and-detect, since the CNN output on corrupted signal is untrustworthy.)

**Behavioral change to record.** The old in-filter design used **per-head** motion thresholds (AFib gated at motion ≤ 5 mG, VT at ≥ 24 mG, etc.). A pre-CNN gate is **segment-level, all-or-nothing** — one quality verdict for the whole window, not per label. This is simpler and more defensible but loses per-head motion tuning. Acceptable because default is OFF; revisit thresholds when enabling.

**HR-plausibility split-out.** The old filter also had an HR rule (Bradycardia → `mean_hr_bpm ≤ 56.3`). That is **not** signal quality — it is physiological plausibility and belongs in **L2** (context rule), not the SQG. So `multiclass_fp_suppression.py` decomposes into: motion/SNR → SQG; HR plausibility → L2.

**Inputs.** Preprocessed signal + (when available) accelerometer stream. On PTB-XL there is no accelerometer, so motion is undefined → SQG OFF is the only meaningful setting there anyway.

---

## 3. L1 — learned scope projection

**Contract.** Input = full 150-dim backbone output. Output = 6 independent **calibrated** probabilities for the scope heads, plus pass-through of head-1 (NSR) raw score for L2.

**Consume all 150, not just the 6 scope rows** — the descriptor scaffold (ABNORMAL ECG, SINUS RHYTHM, LAE …) and NSR carry correlated signal; a projection over all 150 can recover sensitivity the raw heads collapse.

**Architecture — start simple:**
- (a) **Linear probe 150→6** (≈906 params) — default; interpretable, reuses `finetune_scope_linprobe.py` scaffolding.
- (b) MLP 150→128→6 — only if (a)'s val PR-AUC is short (esp. SVT/VT).
- (c) classifier-chain — avoid unless (a)/(b) plateau (blurs the L1/L2 split).

**Loss / calibration.** Per-head `BCEWithLogitsLoss`, multi-label (no softmax); per-head `pos_weight` for rare heads (SVT/VT n=42, Pause 0); **post-hoc per-head temperature/Platt calibration** so 0.5 is meaningful (current heads over-fire at 0.5: Brady 5.9×, Tachy 3.1× GT).

**Training data — multi-dataset union** (join via `cross_dataset_label_map.csv`):
- **Pause (142) has 0 PTB-XL GT → must come from fzark + CINC2015.**
- SVT/VT augmented from MIMIC + MIT-BIH.
- Respect PTB-XL fold split (1–8 / 9 / 10); other datasets partition by patient ID.

**Must not** do inter-label logic — pure per-head scorer.

**Acceptance gate** (extend `eval_6head_regression_gate.py`): adopt only if per-head calibrated PR-AUC ≥ base and FP-rate ≤ base; assert backbone weights untouched.

---

## 4. L2 — deterministic rule arbiter

**Configurable; DEFAULT OFF (2026-06-01).** `arbitrate(s, config)` / `to_alerts(d, config)` take an `ArbiterConfig` whose master switch `enabled` defaults False. When OFF, L2 is pure pass-through (candidate firings at `fire_threshold` flow through unchanged — no suppression, no merge); each rule (`nsr_contradiction`, `afib_over_tachy`, `mutual_exclusion`, `hr_plausibility`, `merge_runs`) and threshold is independently togglable when enabled. `ScopedDetector` defaults to L2 OFF. Kept OFF for now.

**Contract.** Input = `{6 calibrated probs, nsr_score (head 1), context (hr_bpm, rr_irregularity), optional quality flags}`. Output = 6 `Decision{fired, score, reason}`. Pure function, no weights, unit-tested; thresholds are versioned constants. **Suppress / merge only — never promotes.**

**Rule pipeline — strict precedence** (signal-quality stage REMOVED — handled by SQG):

| # | stage | rule | source |
|---|---|---|---|
| 1 | Validity mask | drop heads invalid for the segment (e.g. Pause needs adequate clean length) | new |
| 2 | **NSR contradiction** | if `nsr_score ≥ τ_nsr` AND arrhythmia fired with `score < τ_weak` → suppress that arrhythmia | realness: false-AFib 62% / false-Tachy 68% co-fire NSR |
| 3 | **AFib ▸ Tachy** | AFib + Sinus-Tachy both fired → drop Tachy, keep AFib (AF-RVR; Tachy co-fire precision 1.0%) | MULTI_LABEL_RULES r1 |
| 4 | **Brady ⊕ Tachy / Brady ⊕ AFib** | mutually-exclusive → keep higher-confidence, drop other | r3 (GT count 0) |
| 5 | **HR plausibility** | Bradycardia requires `hr_bpm ≤ ~56`; (moved from old SQI filter) | reclassified |
| 6 | **Run cluster** | SVT(93)+V-Run(98): keep both (never suppress); optionally merge to one "Run — origin uncertain" alert | r4 (GT Jaccard 1.0) |
| 7 | Emit | decisions + per-head `reason` audit string | — |

Precedence rationale: validity → FP contradiction → pairwise arbitration → plausibility → clustering. L2 never invents positives (promotion = L1's job).

---

## 5. Interfaces

```python
@dataclass(frozen=True)
class ScopeScores:                 # L1 output
    probs: dict[int, float]        # {4,5,6,93,98,142} → calibrated prob
    nsr_score: float               # head-1 raw prob — FP feature, NOT a detection
    context: dict[str, float]      # hr_bpm, rr_irregularity, ...
    quality: dict[int, bool] = {}  # {150 Noisy, 151 High-Motion} — informational when SQG OFF

@dataclass(frozen=True)
class Decision:                    # L2 output, one per scope head
    head: int; fired: bool; score: float; reason: str

# SQG:  gate(signal, accel=None, enabled=False) -> (signal, quality_flags, passed: bool)
# L2 :  arbitrate(s: ScopeScores) -> dict[int, Decision]
```

---

## 6. Where it plugs into the repo

| component | existing file | change |
|---|---|---|
| **SQG** | `multiclass_fp_suppression.py` (motion/SNR parts), `sqi.py` | extract signal-quality into a pre-CNN gate module (e.g. `signal_quality_gate.py`), `enabled=False` default |
| L1 projection | `scripts/finetune_scope_linprobe.py` | generalize to a 150→6 projection head (`scope_overlay.py`) |
| L1 acceptance | `scripts/eval_6head_regression_gate.py` | PR-AUC / FP gate on 6 heads |
| L2 arbiter | `multiclass_fp_suppression.py` (logic parts) + HR rule | rebuild as inter-label pipeline (stages 1–7), no signal-quality stage |
| inference wrapper | `dual_head_ecgfounder.py` / `multi_head_router.py` | wire preprocessing → SQG → backbone → L1 → L2 |
| rule config + tests | `label_config.py` + `tests/` | thresholds as constants; `test_arbiter.py` |

---

## 7. Open decisions

1. **L1 form** — true 150→6 projection vs in-place row-tuning of the 6 scope rows.
2. **L2 Stage 6** — merge SVT/VT into one "Run (origin uncertain)" alert vs keep both.
3. **SQG enable criteria** — what SQI/motion thresholds, and skip-vs-flag semantics, when it is eventually turned ON (moot while OFF).
