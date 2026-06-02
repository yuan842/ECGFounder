# fzark TP/FP — base-routed-head L2 refinement evaluation

**Recommendation under test:** on a device where a head is base-routed (uncalibrated, base@0.5), exclude it from L2 rate-exclusion arbitration (`exclude`) — or disable it outright (`disable`). On the fzark config the only base-routed scope head is **SINUS TACHYCARDIA** (head 6). It shares the rate-exclusion group {Brady 4, AFib 5, Tachy 6}, so the change can move decisions for the uncalibrated head AND its calibrated peers.

Held-out fzark TEST split (seed-fixed, grouped by Source Data File). TP→sensitivity, FP→retention/specificity, pooled PPV/F1 **relative** (cohorts separately capped-sampled). All prod variants use the fzark device config + L2 GT-matched (ON).

Test records: 4541.  Rate-exclusion-group co-fires where the uncalibrated head (6) fired alongside a calibrated peer pre-arbitration: **460**.  Records whose final decisions differ (current→exclude): **457**.

| head | source | TP n | FP n | metric | base | current | **exclude (a)** | disable (b) |
|---|---|---:|---:|---|---:|---:|---:|---:|
| 4 SINUS BRADYCARDIA | fzark-L1 | 382 | 263 | Sensitivity | 0.950 | 0.950 | **0.950** | 0.950 |
|  |  |  |  | Specificity | 0.966 | 0.920 | **0.920** | 0.920 |
|  |  |  |  | FP-retention | 0.034 | 0.080 | **0.080** | 0.080 |
|  |  |  |  | PPV* | 0.976 | 0.945 | **0.945** | 0.945 |
|  |  |  |  | F1* | 0.963 | 0.948 | **0.948** | 0.948 |
| 5 ATRIAL FIBRILLATION | fzark-L1 | 923 | 265 | Sensitivity | 0.963 | 0.901 | **0.901** | 0.901 |
|  |  |  |  | Specificity | 0.264 | 0.951 | **0.951** | 0.951 |
|  |  |  |  | FP-retention | 0.736 | 0.049 | **0.049** | 0.049 |
|  |  |  |  | PPV* | 0.820 | 0.985 | **0.985** | 0.985 |
|  |  |  |  | F1* | 0.886 | 0.941 | **0.941** | 0.941 |
| 6 SINUS TACHYCARDIA | base ★ | 0 | 197 | Sensitivity | — | — | **—** | — |
|  |  |  |  | Specificity | 0.005 | 0.010 | **0.005** | 1.000 |
|  |  |  |  | FP-retention | 0.995 | 0.990 | **0.995** | 0.000 |
|  |  |  |  | PPV* | 0.000 | 0.000 | **0.000** | — |
|  |  |  |  | F1* | — | — | **—** | — |
| 93 SUPRAVENTRICULAR TACHYCARDIA | fzark-L1 | 7 | 247 | Sensitivity | 0.000 | 0.714 | **0.714** | 0.714 |
|  |  |  |  | Specificity | 1.000 | 0.688 | **0.688** | 0.688 |
|  |  |  |  | FP-retention | 0.000 | 0.312 | **0.312** | 0.312 |
|  |  |  |  | PPV* | — | 0.061 | **0.061** | 0.061 |
|  |  |  |  | F1* | — | 0.112 | **0.112** | 0.112 |
| 98 VENTRICULAR TACHYCARDIA | fzark-L1 | 330 | 215 | Sensitivity | 0.000 | 0.979 | **0.979** | 0.979 |
|  |  |  |  | Specificity | 1.000 | 0.981 | **0.981** | 0.981 |
|  |  |  |  | FP-retention | 0.000 | 0.019 | **0.019** | 0.019 |
|  |  |  |  | PPV* | — | 0.988 | **0.988** | 0.988 |
|  |  |  |  | F1* | — | 0.983 | **0.983** | 0.983 |
| 142 WITH SINUS PAUSE | fzark-L1 | 25 | 80 | Sensitivity | 0.000 | 0.600 | **0.600** | 0.600 |
|  |  |  |  | Specificity | 1.000 | 0.725 | **0.725** | 0.725 |
|  |  |  |  | FP-retention | 0.000 | 0.275 | **0.275** | 0.275 |
|  |  |  |  | PPV* | — | 0.405 | **0.405** | 0.405 |
|  |  |  |  | F1* | — | 0.484 | **0.484** | 0.484 |

★ = base-routed (uncalibrated) head on this device.

## Interpretation

- **`current`** lets the uncalibrated head suppress / be suppressed by calibrated peers — those decisions are effectively arbitrary, since base@0.5 on OOD single-lead is not a meaningful operating point.
- **`exclude (a)`** removes the uncalibrated head from rate-exclusion only; it still fires on its own threshold. Calibrated peers (Brady/AFib) are no longer perturbed by it. This is the implemented device-config refinement.
- **`disable (b)`** drops the uncalibrated head's alerts entirely. On fzark the head has 0 TP events, so disabling costs **zero** sensitivity while removing its false alarms — the strongest specificity option when a head is uncalibrated.
