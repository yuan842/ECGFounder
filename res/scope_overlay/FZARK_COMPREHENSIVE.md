# fzark TP/FP — comprehensive production evaluation

Production = fzark device config (fzarkSL routed + per-head FP-cap/4%-sens thresholds; Sinus-Tachy→base@0.5) → L2 (GT-matched, default ON). Held-out fzark TEST split (seed-fixed, grouped by Source Data File). TP→sensitivity; FP→retention/specificity; pooled PPV/F1 are **relative** (TP/FP cohorts are separately capped-sampled, not natural prevalence). `prod` = full pipeline (L2 on).

| head | source | TP n | FP n | metric | base | prod(no L2) | **prod (L2)** |
|---|---|---:|---:|---|---:|---:|---:|
| 4 SINUS BRADYCARDIA | fzark-L1 | 382 | 263 | Sensitivity | 0.950 | 0.950 | **0.950** |
|  |  |  |  | Specificity | 0.966 | 0.901 | **0.920** |
|  |  |  |  | FP-retention | 0.034 | 0.099 | **0.080** |
|  |  |  |  | PPV* | 0.976 | 0.933 | **0.945** |
|  |  |  |  | F1* | 0.963 | 0.942 | **0.948** |
| 5 ATRIAL FIBRILLATION | fzark-L1 | 923 | 265 | Sensitivity | 0.963 | 0.934 | **0.901** |
|  |  |  |  | Specificity | 0.264 | 0.951 | **0.951** |
|  |  |  |  | FP-retention | 0.736 | 0.049 | **0.049** |
|  |  |  |  | PPV* | 0.820 | 0.985 | **0.985** |
|  |  |  |  | F1* | 0.886 | 0.959 | **0.941** |
| 6 SINUS TACHYCARDIA | base | 0 | 197 | Sensitivity | — | — | **—** |
|  |  |  |  | Specificity | 0.005 | 0.005 | **0.010** |
|  |  |  |  | FP-retention | 0.995 | 0.995 | **0.990** |
|  |  |  |  | PPV* | 0.000 | 0.000 | **0.000** |
|  |  |  |  | F1* | — | — | **—** |
| 93 SUPRAVENTRICULAR TACHYCARDIA | fzark-L1 | 7 | 247 | Sensitivity | 0.000 | 0.714 | **0.714** |
|  |  |  |  | Specificity | 1.000 | 0.688 | **0.688** |
|  |  |  |  | FP-retention | 0.000 | 0.312 | **0.312** |
|  |  |  |  | PPV* | — | 0.061 | **0.061** |
|  |  |  |  | F1* | — | 0.112 | **0.112** |
| 98 VENTRICULAR TACHYCARDIA | fzark-L1 | 330 | 215 | Sensitivity | 0.000 | 0.979 | **0.979** |
|  |  |  |  | Specificity | 1.000 | 0.981 | **0.981** |
|  |  |  |  | FP-retention | 0.000 | 0.019 | **0.019** |
|  |  |  |  | PPV* | — | 0.988 | **0.988** |
|  |  |  |  | F1* | — | 0.983 | **0.983** |
| 142 WITH SINUS PAUSE | fzark-L1 | 25 | 80 | Sensitivity | 0.000 | 0.600 | **0.600** |
|  |  |  |  | Specificity | 1.000 | 0.725 | **0.725** |
|  |  |  |  | FP-retention | 0.000 | 0.275 | **0.275** |
|  |  |  |  | PPV* | — | 0.405 | **0.405** |
|  |  |  |  | F1* | — | 0.484 | **0.484** |
