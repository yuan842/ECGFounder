# Proposal — a Noisy-only SQI gate (calibrated on Challenge 2017)

**Status: PROPOSAL — not implemented.** This document recommends thresholds and a
gate form; no production code is changed. Validate on the target device before
enabling.

## 1. Objective

Flag **Noisy** (uninterpretable) windows so they are not sent to the detector,
while passing clean recordings. This is a **pure signal-quality gate** — it does
**not** attempt to separate AFib from Normal (both are "clean"). A flagged window
is marked uninterpretable rather than producing a detection on corrupted signal.

## 2. Where it fits

`overlay/signal_quality_gate.py` already defines this gate (`SignalQualityGate`,
**DEFAULT OFF**). Its Noisy flag today is `snr_proxy < min_snr` with an
**uncalibrated placeholder `min_snr = 1.0`**. On Challenge 2017 that threshold is
badly mis-set — Normal's median SNR is 0.84, so `min_snr = 1.0` would flag **~50%
of Normal recordings as Noisy**. This proposal re-calibrates it and adds one
complementary feature.

## 3. Feature selection (data-driven)

Discrimination of Noisy vs clean (AFib+Normal) at the raw stage (S0), AUC:

| feature | AUC | verdict |
|---|---:|---|
| `baseline_drift` (σ of <0.5 Hz) | **0.844** | ✅ strong — captures wander/motion noise |
| `snr_proxy` (band/residual var) | **0.826** | ✅ strong — captures HF/EMG noise |
| `sqi_score_ecg` (composite) | 0.789 | ok, but redundant with the two above |
| `kurt` (QRS peakedness) | 0.653 | weak — drop |
| `hf_noise_ratio` (>40 Hz) | **0.424** | ❌ **inverted** — Noisy median is *lower*; only its tail is high. Including it hurts the gate. |

**Recommended features: `snr_proxy` (low ⇒ noisy) and `baseline_drift`
(high ⇒ noisy).** They are complementary — SNR catches high-frequency/EMG noise,
baseline drift catches low-frequency wander/motion — so an **OR** of the two beats
either alone. `hf_noise_ratio` and `kurt` are excluded.

## 4. Recommended gate form

```
NOISY  ⇔  (snr_proxy < τ_snr)  OR  (baseline_drift > τ_drift)
```

Computed at the raw (S0) window, before the backbone. A combined 4-feature OR
(adding hf/kurt) was tested and **over-flags** — it reaches the same Noisy recall
only by flagging far more clean recordings — so the 2-feature form is preferred.

## 5. Operating points (measured on Challenge 2017)

n = AFib 500, Normal 500, Noisy 279. "Noisy✓" = Noisy correctly flagged (recall);
"clean✗" = clean recordings wrongly flagged (lost to the detector).

| operating point | τ SNR | τ drift | **Noisy✓** | AFib✗ | Normal✗ | **clean✗** |
|---|---:|---:|---:|---:|---:|---:|
| **Conservative (recommended)** | 0.115 | 96 | 56.3% | 5.0% | 9.4% | **7.2%** |
| Balanced | 0.196 | 69 | 72.0% | 11.0% | 17.2% | 14.1% |
| Sensitive | 0.32 | 53 | 82.1% | 17.0% | 25.6% | 21.3% |
| Aggressive | 0.41 | 44 | 86.0% | 22.6% | 34.0% | 28.3% |

(Thresholds set at clean-cohort percentile cut-offs: conservative = SNR p5 / drift
p95; balanced = p10 / p90; sensitive = p15 / p85; aggressive = p20 / p80.)

## 6. Recommendation

**Adopt the Conservative point: `NOISY ⇔ snr_proxy < 0.12 OR baseline_drift > 96`.**

- Catches the **majority (~56%) of Noisy** at a **low ~7% clean-loss cost**.
- For a gate, false-flagging a clean recording costs **lost sensitivity** (a good
  signal is dropped), whereas missing a noisy one is partly absorbed downstream by
  the L1+L2 false-positive suppression. So we should bias toward protecting clean
  recordings — the conservative point does exactly that.
- **AFib is flagged less than Normal at every point** (5.0% vs 9.4% conservative),
  because AFib here has higher SNR and lower drift — i.e. the gate is gentle on the
  target arrhythmia, which is desirable.

Move to Balanced only if downstream false-alarm burden from noisy windows proves
more costly than the ~14% clean loss.

## 7. Key caveats (read before implementing)

1. **Noisy and clean genuinely overlap.** ~30–40% of "Noisy" recordings have
   clean-looking SQI (Noisy SNR p90 = 1.04, in the clean range). **No threshold
   flags all Noisy without heavy clean loss** — 86% recall already costs 28% clean.
   A perfect Noisy gate is not achievable from these SQI features alone.
2. **Scale portability.** `snr_proxy` is a variance *ratio* (scale-free) → its
   threshold (~0.12) should transfer across devices. `baseline_drift` is an
   absolute amplitude (here in mV after the AliveCor ADC÷1000 convention) → **its
   threshold must be re-derived per device/gain**. Consider normalizing drift by
   signal amplitude before fixing a portable threshold.
3. **Per-window vs recording-level.** Rates above are per 10 s window. Detection is
   recording-level (any-window-fires); a recording-level Noisy policy (e.g. flag if
   ≥k noisy windows, or if the whole recording's median SQI is low) needs its own
   calibration.
4. **External-only labels.** Challenge 2017's "~" is a single noisy class on one
   handheld device. Re-validate on the actual target device's clean/noisy set
   before enabling — these thresholds are a starting point, not final.

## 8. Proposed wiring (NOT done here)

When validated, the change is small and local to `overlay/signal_quality_gate.py`:
- Re-calibrate `DEFAULT_MIN_SNR`: **1.0 → ~0.12**.
- Add a `baseline_drift > τ_drift` OR-term to the Noisy flag in `_flags()` (today
  it only reads `snr_proxy`); source `baseline_drift` from the `sqi` panel.
- Keep the gate **DEFAULT OFF** until validated on the target device.

## 9. Next steps before implementation

1. Reproduce these rates on the target device's labeled clean/noisy data.
2. Decide per-window vs recording-level policy and re-tune.
3. Make `baseline_drift` scale-invariant (normalize by amplitude) for portability.
4. Pick the final operating point with clinical input on the clean-loss vs
   noise-pass trade-off, then wire §8 and add a regression test.

---

*Supporting analysis: `scripts/eval_sqi_noisy_gate_challenge2017.py`,
`scripts/eval_sqi_stages_challenge2017.py`; cut-offs in
`res/challenge2017/SQI_STAGES_cutoffs.csv`; figure
`docs/sqi_cutoffs_challenge2017.png`.*
