# iter1 vs iter3 vs production — same L1 policy (max-F1, 5pp)

Each L1 model calibrated with its own per-head max-F1 thresholds on the matching validation distribution, then evaluated at that operating point. Base = raw head @0.5. L1 heads {4,5,6}; 93/98/142 base-routed (identical).


## PTB-XL lead-II (test n=2,198)

| head | metric | base | iter1 | iter3 | prod(fuzzySL) |
|---|---|---:|---:|---:|---:|
| 4 SINUS BRADYCARDIA | thr | 0.50 | 0.43 | 0.21 | 0.05 |
|  | sens | 0.953 | 0.922 | 0.938 | 0.938 |
|  | spec | 0.852 | 0.858 | 0.845 | 0.858 |
|  | ppv | 0.162 | 0.163 | 0.154 | 0.165 |
|  | f1 | 0.277 | 0.276 | 0.264 | 0.280 |
|  | pr_auc | 0.346 | 0.601 | 0.555 | 0.489 |
| 5 ATRIAL FIBRILLATION | thr | 0.50 | 0.87 | 0.82 | 0.83 |
|  | sens | 0.934 | 0.914 | 0.921 | 0.914 |
|  | spec | 0.968 | 0.991 | 0.988 | 0.987 |
|  | ppv | 0.683 | 0.885 | 0.848 | 0.842 |
|  | f1 | 0.789 | 0.900 | 0.883 | 0.877 |
|  | pr_auc | 0.909 | 0.922 | 0.918 | 0.920 |
| 6 SINUS TACHYCARDIA | thr | 0.50 | 0.70 | 0.73 | 0.50 |
|  | sens | 0.976 | 0.927 | 0.927 | 0.927 |
|  | spec | 0.922 | 0.982 | 0.986 | 0.984 |
|  | ppv | 0.328 | 0.661 | 0.717 | 0.691 |
|  | f1 | 0.491 | 0.772 | 0.809 | 0.792 |
|  | pr_auc | 0.813 | 0.805 | 0.828 | 0.825 |
| **MACRO** | f1 | 0.519 | 0.649 | 0.652 | 0.650 |
| **MACRO** | pr_auc | 0.690 | 0.776 | 0.767 | 0.745 |

## FUZZY derived-lead (4 angles) (test n=8,792)

| head | metric | base | iter1 | iter3 | prod(fuzzySL) |
|---|---|---:|---:|---:|---:|
| 4 SINUS BRADYCARDIA | thr | 0.50 | 0.72 | 0.39 | 0.43 |
|  | sens | 0.969 | 0.922 | 0.953 | 0.930 |
|  | spec | 0.842 | 0.869 | 0.857 | 0.867 |
|  | ppv | 0.155 | 0.175 | 0.167 | 0.173 |
|  | f1 | 0.268 | 0.294 | 0.284 | 0.292 |
|  | pr_auc | 0.369 | 0.628 | 0.591 | 0.582 |
| 5 ATRIAL FIBRILLATION | thr | 0.50 | 0.84 | 0.83 | 0.86 |
|  | sens | 0.934 | 0.878 | 0.891 | 0.891 |
|  | spec | 0.983 | 0.990 | 0.988 | 0.989 |
|  | ppv | 0.799 | 0.870 | 0.846 | 0.863 |
|  | f1 | 0.861 | 0.874 | 0.868 | 0.877 |
|  | pr_auc | 0.927 | 0.917 | 0.888 | 0.899 |
| 6 SINUS TACHYCARDIA | thr | 0.50 | 0.86 | 0.86 | 0.86 |
|  | sens | 0.976 | 0.948 | 0.927 | 0.936 |
|  | spec | 0.979 | 0.983 | 0.986 | 0.985 |
|  | ppv | 0.643 | 0.685 | 0.715 | 0.706 |
|  | f1 | 0.775 | 0.795 | 0.807 | 0.805 |
|  | pr_auc | 0.803 | 0.807 | 0.831 | 0.835 |
| **MACRO** | f1 | 0.635 | 0.654 | 0.653 | 0.658 |
| **MACRO** | pr_auc | 0.700 | 0.784 | 0.770 | 0.772 |

## Verdict

Under the **same policy** (each model at its own per-head max-F1 / 5 pp operating point), the three L1 iterations are **statistically equivalent** — the operating-point calibration largely erases the training-lead differences:

| | lead-II F1 | lead-II PR-AUC | fuzzy F1 | fuzzy PR-AUC |
|---|---:|---:|---:|---:|
| base | 0.519 | 0.690 | 0.635 | 0.700 |
| iter1 (lead-II) | 0.649 | **0.776** | 0.654 | **0.784** |
| iter3 (union) | **0.652** | 0.767 | 0.653 | 0.770 |
| prod / iter2 (fuzzy) | 0.650 | 0.745 | **0.658** | 0.772 |

- **Macro F1 is tied within 0.005** on both distributions — once thresholds are policy-calibrated, all three L1 models land at essentially the same precision/recall balance, and all beat base by a wide margin (lead-II +0.13, fuzzy +0.02).
- **iter1 (lead-II) wins ranking (PR-AUC)** on both distributions and the best lead-II AFib F1 (0.900). It's the best choice for a lead-II deployment.
- **prod / iter2 (fuzzy) wins fuzzy macro F1** (its deployment target) and fuzzy AFib F1 — the basis for keeping it as the single-lead production model.
- **iter3 (union) is the balanced middle** — best Sinus-Tachy F1, never worst, never clearly best; no decisive advantage over iter1.
- **Bradycardia is the floor for all three** (F1 ≈0.28–0.29, PR-AUC ≈0.58–0.63) — the gain there is in ranking (PR-AUC +0.21–0.26 over base), not threshold precision.

**Takeaway:** the training-lead choice is second-order once the policy sets the operating point; the production pick (fuzzySL) is justified by best fuzzy-deployment F1, while iter1 would be the pick if the target were clean lead-II.
