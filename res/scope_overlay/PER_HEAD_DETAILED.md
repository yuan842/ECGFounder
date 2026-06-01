# Per-head detail — baseline vs production (fuzzySL, max-F1 / 5% policy)

Production = overlay routing (L1 for {4,5,6}, base for {93,98,142}), L2 OFF, per-head thresholds from the L1 policy (max-F1 s.t. sens loss <5pp), calibrated **per deployment distribution**. Baseline = raw backbone head at 0.5. PTB-XL fold-10 test. `thr` = decision threshold used.

Production thresholds — lead-II: { 4:0.049, 5:0.830, 6:0.500 }; fuzzy: { 4:0.431, 5:0.863, 6:0.860 }.

## PTB-XL lead-II (n=2,198)

| head | model | thr | Sens | Spec | PPV | NPV | F1 | AUROC | PR-AUC |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 4 SINUS BRADYCARDIA (L1) | base | 0.50 | 0.953 | 0.852 | 0.162 | 0.998 | 0.277 | 0.944 | 0.346 |
| 4 SINUS BRADYCARDIA (L1) | **prod** | 0.05 | 0.938 | 0.858 | 0.165 | 0.998 | 0.280 | 0.939 | 0.489 |
| | Δ | | -0.016 | +0.005 | +0.003 | -0.001 | +0.003 | -0.005 | +0.143 |
| 5 ATRIAL FIBRILLATION (L1) | base | 0.50 | 0.934 | 0.968 | 0.683 | 0.995 | 0.789 | 0.976 | 0.909 |
| 5 ATRIAL FIBRILLATION (L1) | **prod** | 0.83 | 0.914 | 0.987 | 0.842 | 0.994 | 0.877 | 0.977 | 0.920 |
| | Δ | | -0.020 | +0.020 | +0.160 | -0.001 | +0.088 | +0.002 | +0.011 |
| 6 SINUS TACHYCARDIA (L1) | base | 0.50 | 0.976 | 0.922 | 0.328 | 0.999 | 0.491 | 0.986 | 0.813 |
| 6 SINUS TACHYCARDIA (L1) | **prod** | 0.50 | 0.927 | 0.984 | 0.691 | 0.997 | 0.792 | 0.985 | 0.825 |
| | Δ | | -0.049 | +0.061 | +0.363 | -0.002 | +0.301 | -0.001 | +0.011 |
| 93 SUPRAVENTRICULAR TACHYCARDIA (base) | base | 0.50 | 0.200 | 0.998 | 0.200 | 0.998 | 0.200 | 0.996 | 0.319 |
| 93 SUPRAVENTRICULAR TACHYCARDIA (base) | **prod** | 0.50 | 0.200 | 0.998 | 0.200 | 0.998 | 0.200 | 0.996 | 0.319 |
| 98 VENTRICULAR TACHYCARDIA (base) | base | 0.50 | 0.600 | 0.997 | 0.300 | 0.999 | 0.400 | 0.997 | 0.341 |
| 98 VENTRICULAR TACHYCARDIA (base) | **prod** | 0.50 | 0.600 | 0.997 | 0.300 | 0.999 | 0.400 | 0.997 | 0.341 |
| 142 WITH SINUS PAUSE | — | — | — | 1.000 | — | 1.000 | — | — | — | (0 positives) |

## FUZZY derived-lead (4 angles pooled) (n=8,792)

| head | model | thr | Sens | Spec | PPV | NPV | F1 | AUROC | PR-AUC |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 4 SINUS BRADYCARDIA (L1) | base | 0.50 | 0.969 | 0.842 | 0.155 | 0.999 | 0.268 | 0.949 | 0.369 |
| 4 SINUS BRADYCARDIA (L1) | **prod** | 0.43 | 0.930 | 0.867 | 0.173 | 0.998 | 0.292 | 0.947 | 0.582 |
| | Δ | | -0.039 | +0.025 | +0.018 | -0.001 | +0.024 | -0.002 | +0.213 |
| 5 ATRIAL FIBRILLATION (L1) | base | 0.50 | 0.934 | 0.983 | 0.799 | 0.995 | 0.861 | 0.979 | 0.927 |
| 5 ATRIAL FIBRILLATION (L1) | **prod** | 0.86 | 0.891 | 0.989 | 0.863 | 0.992 | 0.877 | 0.982 | 0.899 |
| | Δ | | -0.043 | +0.007 | +0.064 | -0.003 | +0.016 | +0.003 | -0.027 |
| 6 SINUS TACHYCARDIA (L1) | base | 0.50 | 0.976 | 0.979 | 0.643 | 0.999 | 0.775 | 0.991 | 0.803 |
| 6 SINUS TACHYCARDIA (L1) | **prod** | 0.86 | 0.936 | 0.985 | 0.706 | 0.997 | 0.805 | 0.991 | 0.835 |
| | Δ | | -0.040 | +0.006 | +0.063 | -0.002 | +0.030 | -0.001 | +0.031 |
| 93 SUPRAVENTRICULAR TACHYCARDIA (base) | base | 0.50 | 0.650 | 0.997 | 0.333 | 0.999 | 0.441 | 0.997 | 0.330 |
| 93 SUPRAVENTRICULAR TACHYCARDIA (base) | **prod** | 0.50 | 0.650 | 0.997 | 0.333 | 0.999 | 0.441 | 0.997 | 0.330 |
| 98 VENTRICULAR TACHYCARDIA (base) | base | 0.50 | 0.700 | 0.996 | 0.311 | 0.999 | 0.431 | 0.996 | 0.310 |
| 98 VENTRICULAR TACHYCARDIA (base) | **prod** | 0.50 | 0.700 | 0.996 | 0.311 | 0.999 | 0.431 | 0.996 | 0.310 |
| 142 WITH SINUS PAUSE | — | — | — | 1.000 | — | 1.000 | — | — | — | (0 positives) |
