# Challenge 2017 — AFib performance WITH vs WITHOUT the S0 SQI gate

Routing = **device-specific AliveCor-fine-tuned L1** (AFib head fit on Challenge 2017, others frozen at fzark; AFib τ=0.396) + L2.

Held-out TEST (n=852: A=76, N=507, O=241, ~=28). Gate = S0 SQI, device `alivecor`→default (NOISY ⇔ snr_proxy <= 0.1; baseline-drift gating off). Per-window gate; recording fires AFib if any non-gated window fires. Binary metrics: clean A vs N.

## AFib detection — with vs without gate

| metric | without gate | with gate | Δ |
|---|---:|---:|---:|
| Sensitivity (A) | 0.987 | 0.974 | -0.013 |
| Specificity (N) | 0.941 | 0.945 | +0.004 |
| PPV | 0.714 | 0.725 | +0.011 |
| F1 | 0.829 | 0.831 | +0.003 |
| Fire-rate Other ↓ | 0.307 | 0.303 | -0.004 |
| Fire-rate Noisy ↓ | 0.536 | 0.429 | -0.107 |

## Gate activity (per group)

| group | windows gated Noisy | recordings fully uninterpretable |
|---|---:|---:|
| AFib | 2.9% | 1.3% |
| Normal | 7.1% | 1.2% |
| Other | 4.1% | 1.2% |
| Noisy | 36.5% | 21.4% |

## Read

- The gate skips noisy windows before detection. On **Noisy (~)** it gates 37% of windows, cutting their AFib fire-rate 0.536→0.429 and Other 0.307→0.303.
- AFib sensitivity 0.987→0.974, Normal specificity 0.941→0.945 (clean groups are barely gated: AFib 2.9%, Normal 7.1% of windows).
- 'Uninterpretable' = every window of a recording gated Noisy → no detection emitted (reported, not counted as a positive).
