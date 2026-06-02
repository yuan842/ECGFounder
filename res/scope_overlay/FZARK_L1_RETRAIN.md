# fzark L1 — device config (max-sens under per-head FP cap)

**What:** the L1 ScopeProjection (150→6) trained on the **fzark TP/FP** cohorts (frozen backbone, frozen L2), thresholds set by a **max-sensitivity-under-FP-cap** policy, persisted as a **single device-specific config**. Trainer: [scripts/train_l1_fzark.py](../../scripts/train_l1_fzark.py); checkpoint `fzarkSL.pth`; config `device_configs/fzark.json`; loader `overlay/device_config.py`.

## Policy (fzark device profile) — joint FP-cap + sens-budget

Per head: **τ = min(τ_fp, τ_sens)** —
- `τ_fp` = lowest threshold keeping **FP-retention < per-head cap** (on the fzark FP cohort),
- `τ_sens` = highest threshold keeping **sensitivity loss < 4 pp** vs the base head (vacuous for silent heads with base-sens ≈ 0),
- both with a 0.02 val→test margin.

When the two conflict (a tight FP cap would cost >4% sensitivity — AFib), the **sens budget wins** and FP is allowed above its cap.

| head | FP cap | sens-loss cap | route |
|---|---:|---:|---|
| 4 Bradycardia | 0.10 | 4% | fzark-L1 |
| 5 AFib | 0.05 | 4% | fzark-L1 |
| 93 SV-Run | **0.40** | 4% | fzark-L1 |
| 98 V-Run | 0.05 | 4% | fzark-L1 |
| 142 Pause | **0.40** | 4% | fzark-L1 |
| 6 Sinus-Tachy | — | — | **base @0.5** (0 fzark TP events) |

**Global device rule:** settings are device-specific — keep one config file per device, update it when a new device is identified, and **any head with 0 calibration events → base @0.5**.

## Held-out fzark TEST — base @0.5 → fzark-L1 (device thresholds)

| head | thr | TP n | sens base→L1 (Δ) | FP n | FP-retention base→L1 | caps |
|---|---:|---:|---|---:|---|---|
| 4 Bradycardia | 0.092 | 382 | 0.950→**0.950** (−0.0%) | 263 | 0.034→0.099 | FP<0.10 ✅ sens<4% ✅ |
| 5 AFib | 0.541 | 923 | 0.963→**0.934** (−2.9%) | 265 | 0.736→**0.049** | sens<4% ✅ FP<0.05 ✅ |
| 93 SV-Run | 0.049 | 7 | 0.000→**0.714** | 247 | 0.000→0.312 | FP<0.40 ✅ |
| 98 V-Run | 0.450 | 330 | 0.000→**0.979** | 215 | 0.000→**0.019** | FP<0.05 ✅ |
| 142 Pause | 0.074 | 25 | 0.000→**0.600** | 80 | 0.000→0.275 | FP<0.40 ✅ |
| 6 Sinus-Tachy | base 0.5 | 0 | — | 197 | base 0.995 | (base; 0 events) |

**All heads meet every cap on held-out test:**
- **AFib sens loss now 2.9% (< 4%)** — the sens budget bound the threshold (binding=`sens_budget`); FP still landed at 0.049 (< 0.05) on test. (Was −9.5% sens loss under the FP-only policy.)
- **Bradycardia: full sensitivity preserved (0.950), FP 0.099 < 0.10** — regression fixed.
- **V-Run recovered to 0.979 sens at 1.9% FP** (base 0% — undetectable single-lead). The headline.
- **Looser SV-Run / Pause caps (0.40) recover more sensitivity**: SV-Run 0→**0.714** (was 0.571 at 0.20), Pause 0→**0.600** (was 0.520) — small n (7 / 25 TP, noisy).

## The single device config

[device_configs/fzark.json](device_configs/fzark.json) records, per head: `source` (fzark-L1 / base), `threshold`, `fp_limit`, and the achieved test sens / FP-retention. Loaded by `overlay.device_config.load_device_config`; consumed by `ScopedDetector(device_config=...)` — routes {4,5,93,98,142} through fzarkSL at their FP-capped thresholds and Sinus-Tachy to base @0.5, with L2 default-ON. The **PTB profile is untouched and separate** (`fuzzySL.pth` + `POLICY_MAX_SENS_DROP`, `DEFAULT_CKPT` unchanged).

## Caveats
- **Sinus-Tachy** on fzark = base @0.5 (raw head, FP-retention ~0.99) — unusable until a device with Tachy events is calibrated. By design (0 events → base).
- **SV-Run (n=7) and Pause (n=25)** TP test counts are small → sens estimates noisy; FP caps are on larger FP sets (247 / 80).
- Weights are unchanged from the prior fzark training (BCE is policy-independent); the change is the FP-capped thresholds + the device config.
- Thresholds are device-specific; re-run `scripts/train_l1_fzark.py` (or a calibrator) per new device.
