# challenge2017 — production AFib eval: val + test, L2-on vs L2-off

Split: stratified **80/10/10** (`csv/challenge2017_split.csv`). VAL = tuning context; TEST = held-out report. Production stack on the base 1-lead backbone. Three variants: **base@0.5** (raw head) → **L2-off** (fzarkSL L1 routed + per-head τ, no arbiter) → **L2-on** (+ GT-matched L2). Recording-level: tile into 10 s windows, fire if any window fires; prob = max window. Binary metrics use clean **A vs N** only; O/~ fire-rates shown for context.

AFib head-5 device threshold τ = 0.5412.

## VAL  (n=854, A=76, N=508, windows=2704)

| variant | sens (A) | spec (N) | PPV | F1 | ROC-AUC | PR-AUC | fire O | fire ~ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| base@0.5 | 0.961 | 0.675 | 0.307 | 0.465 | 0.967 | 0.882 | 0.533 | 0.786 |
| prod L2-off (L1+τ) | 0.961 | 0.866 | 0.518 | 0.673 | 0.952 | 0.726 | 0.227 | 0.143 |
| **prod L2-on** | 0.908 | 0.919 | 0.627 | 0.742 | 0.952 | 0.726 | 0.165 | 0.143 |

## TEST  (n=852, A=76, N=507, windows=2710)

| variant | sens (A) | spec (N) | PPV | F1 | ROC-AUC | PR-AUC | fire O | fire ~ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| base@0.5 | 0.987 | 0.708 | 0.336 | 0.502 | 0.983 | 0.922 | 0.564 | 0.750 |
| prod L2-off (L1+τ) | 0.895 | 0.876 | 0.519 | 0.657 | 0.946 | 0.803 | 0.282 | 0.214 |
| **prod L2-on** | 0.882 | 0.935 | 0.670 | 0.761 | 0.946 | 0.803 | 0.178 | 0.214 |

## L2 contribution (TEST: L2-off → L2-on)

| metric | L2-off | L2-on | Δ |
|---|---:|---:|---:|
| Sensitivity (A) | 0.895 | 0.882 | -0.013 |
| Specificity (N) | 0.876 | 0.935 | +0.059 |
| Fire-rate O | 0.282 | 0.178 | -0.104 |
| Fire-rate ~ | 0.214 | 0.214 | +0.000 |

L2-off and L2-on share the same routed prob, so their ROC/PR-AUC are identical — L2 only moves the binary fire decision, not the ranking. The AUC change is at **base→L1**: the raw head actually *ranks* A-vs-N slightly better here (test ROC 0.983 vs L1's 0.946), but at its default 0.5 it fires on everything (spec 0.708). fzarkSL L1 trades a little ranking AUC for a usable operating point (spec 0.876 at τ=0.541); L2 then adds specificity on top (→0.935). I.e. L1's value here is **calibration of the decision point**, not ranking.

> O ('Other') = non-AFib arrhythmias + some AF-like rhythms; ~ = noisy. Excluded from binary metrics.
