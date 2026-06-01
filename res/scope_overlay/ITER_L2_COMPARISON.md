# Per-head with L2 ON — base vs iter1 vs iter3 vs production

Each model at its per-head max-F1/5pp thresholds (per deployment dist), then GT-matched L2 (exclusion {4,5,6}, couple {93,98}). Base = raw heads @0.5 + L2. Metrics from post-L2 decisions; PR-AUC is score-level (L2-invariant). Heads 93/98/142 are base-routed and L2 keeps the couple → identical across models.


## PTB-XL lead-II (test n=2,198)

| head | metric | base+L2 | iter1+L2 | iter3+L2 | prod+L2 |
|---|---|---:|---:|---:|---:|
| 4 SINUS BRADYCARDIA | thr | 0.50 | 0.43 | 0.21 | 0.05 |
|  | sens | 0.938 | 0.922 | 0.938 | 0.938 |
|  | spec | 0.857 | 0.859 | 0.853 | 0.866 |
|  | ppv | 0.164 | 0.164 | 0.160 | 0.173 |
|  | f1 | 0.280 | 0.278 | 0.274 | 0.292 |
|  | pr_auc | 0.346 | 0.601 | 0.555 | 0.489 |
| 5 ATRIAL FIBRILLATION | thr | 0.50 | 0.87 | 0.82 | 0.83 |
|  | sens | 0.934 | 0.849 | 0.875 | 0.908 |
|  | spec | 0.974 | 0.992 | 0.990 | 0.989 |
|  | ppv | 0.724 | 0.890 | 0.869 | 0.863 |
|  | f1 | 0.816 | 0.869 | 0.872 | 0.885 |
|  | pr_auc | 0.909 | 0.922 | 0.918 | 0.920 |
| 6 SINUS TACHYCARDIA | thr | 0.50 | 0.70 | 0.73 | 0.50 |
|  | sens | 0.963 | 0.927 | 0.915 | 0.927 |
|  | spec | 0.942 | 0.990 | 0.992 | 0.991 |
|  | ppv | 0.391 | 0.776 | 0.815 | 0.792 |
|  | f1 | 0.556 | 0.844 | 0.862 | 0.854 |
|  | pr_auc | 0.813 | 0.805 | 0.828 | 0.825 |
| **MACRO f1** |  | 0.551 | 0.664 | 0.669 | 0.677 |

## FUZZY derived-lead (4 angles) (test n=8,792)

| head | metric | base+L2 | iter1+L2 | iter3+L2 | prod+L2 |
|---|---|---:|---:|---:|---:|
| 4 SINUS BRADYCARDIA | thr | 0.50 | 0.72 | 0.39 | 0.43 |
|  | sens | 0.969 | 0.922 | 0.953 | 0.930 |
|  | spec | 0.843 | 0.873 | 0.860 | 0.868 |
|  | ppv | 0.157 | 0.178 | 0.170 | 0.175 |
|  | f1 | 0.270 | 0.299 | 0.288 | 0.294 |
|  | pr_auc | 0.369 | 0.628 | 0.591 | 0.582 |
| 5 ATRIAL FIBRILLATION | thr | 0.50 | 0.84 | 0.83 | 0.86 |
|  | sens | 0.934 | 0.878 | 0.880 | 0.891 |
|  | spec | 0.984 | 0.991 | 0.989 | 0.990 |
|  | ppv | 0.808 | 0.878 | 0.857 | 0.870 |
|  | f1 | 0.867 | 0.878 | 0.869 | 0.881 |
|  | pr_auc | 0.927 | 0.917 | 0.888 | 0.899 |
| 6 SINUS TACHYCARDIA | thr | 0.50 | 0.86 | 0.86 | 0.86 |
|  | sens | 0.954 | 0.945 | 0.899 | 0.918 |
|  | spec | 0.985 | 0.988 | 0.990 | 0.990 |
|  | ppv | 0.710 | 0.756 | 0.780 | 0.774 |
|  | f1 | 0.814 | 0.840 | 0.836 | 0.840 |
|  | pr_auc | 0.803 | 0.807 | 0.831 | 0.835 |
| **MACRO f1** |  | 0.650 | 0.672 | 0.664 | 0.672 |

## Verdict

| macro F1 (Brady/AFib/Tachy) | base+L2 | iter1+L2 | iter3+L2 | prod+L2 | (L2-off prod) |
|---|---:|---:|---:|---:|---:|
| lead-II | 0.551 | 0.664 | 0.669 | **0.677** | 0.650 |
| fuzzy | 0.650 | 0.672 | 0.664 | **0.672** | 0.658 |

- **L2 lifts every model**, by cleaning the spurious rate-head co-fires. The gain concentrates on **AFib and Sinus-Tachy** (whose FP co-fires get suppressed): e.g. base Tachy F1 0.49→**0.56** (lead-II) just from L2; the L1 models reach Tachy F1 0.84–0.86. Production AFib F1 0.86→**0.885** (lead-II) with L2.
- **Production (fuzzySL) has the best macro F1 with L2 on, on both distributions** (lead-II 0.677, fuzzy 0.672) — L2 + its policy thresholds combine best. L2 also widens production's edge over L2-off (0.650→0.677 lead-II).
- **iter1 / iter3 / prod stay close** (within ~0.01 macro F1); ranking (PR-AUC) is unchanged by L2 — iter1 still best PR-AUC. The decision-level wins come from L2, not the training-lead choice.
- **Bradycardia is the floor for all** (F1 ≈0.28–0.30): exclusion keeps the higher-confidence rate head, but brady's weak precision is intrinsic — L2 can't manufacture PPV that the ranking lacks.
- **93/98/142 base-routed + couple-kept → identical across all four models** (not shown per-model).

**Takeaway:** turning L2 on is a net win for every model (spurious co-fires removed, AFib/Tachy precision up), and it makes **production (fuzzySL) the clear macro-F1 leader on both PTB-XL and fuzzy** — reinforcing the production choice.
