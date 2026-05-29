#!/usr/bin/env python3
"""
ecg_filter_design.py

Design optimal filters from non-motion ECG features and combine with motion filter
to achieve >90% TP retention AND >90% FP suppression on AFib events.
"""
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.preprocessing import StandardScaler

tp = pd.read_csv('res/motion_analysis/data/afib_tp_ecg_features.csv')
fp = pd.read_csv('res/motion_analysis/data/afib_fp_ecg_features.csv')
print(f"TP={len(tp)}, FP={len(fp)}")
N_TP, N_FP = len(tp), len(fp)

# Drop rows with too many NaN
metrics = ['rmssd','pnn50','samp_en','cv_rr','mean_rr','p_amp_mean','p_present_pct',
           'p_consistency','baseline_drift','kurt','snr_proxy','persistence_score']
tp = tp.dropna(subset=metrics).reset_index(drop=True)
fp = fp.dropna(subset=metrics).reset_index(drop=True)
N_TP, N_FP = len(tp), len(fp)
print(f"After NaN drop: TP={N_TP}, FP={N_FP}")

def perf(keep_tp, keep_fp):
    tp_r = keep_tp.sum() / N_TP * 100
    fp_s = (1 - keep_fp.sum() / N_FP) * 100
    return tp_r, fp_s

# ──────────────────────────────────────────────────────────────────
# 1. Single-feature ROC sweep for each ECG feature
# ──────────────────────────────────────────────────────────────────
print("\n" + "="*80)
print("SINGLE-FEATURE BEST OPERATING POINTS")
print("="*80)
print(f"{'metric':>18}  {'direction':>12}  {'thresh':>10}  {'TP_ret%':>8}  {'FP_sup%':>8}  {'min':>5}")

best_single = {}
for m in metrics:
    tp_v = tp[m].values; fp_v = fp[m].values
    combined = np.concatenate([tp_v, fp_v])
    candidates = np.unique(np.percentile(combined, np.arange(0.5, 100, 0.5)))
    # Try both directions: KEEP if x<=thresh, KEEP if x>=thresh
    best = (None, -1, None, None)
    for direction in ['<=', '>=']:
        for thr in candidates:
            if direction == '<=':
                keep_tp = tp_v <= thr; keep_fp = fp_v <= thr
            else:
                keep_tp = tp_v >= thr; keep_fp = fp_v >= thr
            tp_r, fp_s = perf(keep_tp, keep_fp)
            score = min(tp_r, fp_s)
            if score > best[1]:
                best = (direction, score, thr, (tp_r, fp_s))
    best_single[m] = best
    d, sc, th, (tr, fs) = best
    flag = "  **90/90**" if tr >= 90 and fs >= 90 else ""
    print(f"{m:>18}  {('KEEP '+d):>12}  {th:>10.3f}  {tr:>8.1f}  {fs:>8.1f}  {sc:>5.1f}{flag}")

# ──────────────────────────────────────────────────────────────────
# 2. Logistic regression — full ECG feature stack
# ──────────────────────────────────────────────────────────────────
print("\n" + "="*80)
print("LOGISTIC REGRESSION ON ECG FEATURES (class-balanced)")
print("="*80)
X = pd.concat([tp[metrics], fp[metrics]], ignore_index=True)
y = np.array([0]*N_TP + [1]*N_FP)
sc = StandardScaler(); Xs = sc.fit_transform(X)
lr = LogisticRegression(max_iter=5000, class_weight='balanced', C=1.0)
lr.fit(Xs, y)
scores = lr.predict_proba(Xs)[:, 1]
print("Feature weights (positive = pushes toward FP):")
for f, c in zip(metrics, lr.coef_[0]):
    print(f"  {f:>18}  {c:+8.3f}")
print(f"  {'intercept':>18}  {lr.intercept_[0]:+8.3f}")
print()
print(f"{'P(FP) thresh':>13}  {'TP_ret%':>8}  {'FP_sup%':>8}  {'min':>5}")
best_lr = (None, -1, None)
for t in np.linspace(0.05, 0.95, 91):
    pred = scores >= t
    keep_tp = pred[:N_TP] == False
    keep_fp = pred[N_TP:] == False
    tp_r, fp_s = perf(keep_tp, keep_fp)
    if min(tp_r, fp_s) > best_lr[1]:
        best_lr = (t, min(tp_r, fp_s), (tp_r, fp_s))
    if t in [0.30, 0.40, 0.50, 0.55, 0.60, 0.70, 0.80]:
        flag = "  **90/90**" if tp_r>=90 and fp_s>=90 else ""
        print(f"{t:>13.2f}  {tp_r:>8.1f}  {fp_s:>8.1f}  {min(tp_r,fp_s):>5.1f}{flag}")
print(f"\nBest LR operating point: thresh={best_lr[0]:.3f}  TP_ret={best_lr[2][0]:.1f}%  FP_sup={best_lr[2][1]:.1f}%")

# ──────────────────────────────────────────────────────────────────
# 3. Decision tree
# ──────────────────────────────────────────────────────────────────
print("\n" + "="*80)
print("DECISION TREE ON ECG FEATURES")
print("="*80)
for depth in [2, 3, 4, 5]:
    dt = DecisionTreeClassifier(max_depth=depth, class_weight='balanced',
                                min_samples_leaf=5, random_state=42)
    dt.fit(X, y)
    pred = dt.predict(X)
    keep_tp = pred[:N_TP] == 0; keep_fp = pred[N_TP:] == 0
    tp_r, fp_s = perf(keep_tp, keep_fp)
    flag = "  ** 90/90 **" if tp_r >= 90 and fp_s >= 90 else ""
    print(f"  depth={depth}:  TP_ret={tp_r:.1f}%  FP_sup={fp_s:.1f}%{flag}")
print()
dt = DecisionTreeClassifier(max_depth=3, class_weight='balanced',
                            min_samples_leaf=5, random_state=42)
dt.fit(X, y)
print("Best tree (depth=3):")
print(export_text(dt, feature_names=metrics))

# ──────────────────────────────────────────────────────────────────
# 4. Combined ECG + Motion filter
# ──────────────────────────────────────────────────────────────────
print("\n" + "="*80)
print("COMBINED ECG + MOTION FILTER")
print("="*80)
# Merge motion features
motion_fp = pd.read_csv('res/motion_analysis/data/fp_motion_features.csv')
motion_tp = pd.read_csv('res/motion_analysis/data/tp_motion_features.csv')
motion_fp = motion_fp[motion_fp['event_type'] == 'Atrial Fibrillation'].reset_index(drop=True)
motion_tp = motion_tp[motion_tp['event_type'] == 'Atrial Fibrillation'].reset_index(drop=True)

# Join on path basename
def basename(p): return p.split('/')[-1] if isinstance(p, str) else ''
motion_fp['bn'] = motion_fp['json_path'].apply(basename)
motion_tp['bn'] = motion_tp['json_path'].apply(basename)
fp['bn'] = fp['path'].apply(basename)
tp['bn'] = tp['path'].apply(basename)

fp_full = fp.merge(motion_fp[['bn','mean_motion','max_motion','std_motion','median_motion',
                              'peak_ratio','zero_motion_pct']], on='bn', how='inner')
tp_full = tp.merge(motion_tp[['bn','mean_motion','max_motion','std_motion','median_motion',
                              'peak_ratio','zero_motion_pct']], on='bn', how='inner')
print(f"Joined: TP={len(tp_full)}, FP={len(fp_full)}")
N_TP2, N_FP2 = len(tp_full), len(fp_full)

all_feats = metrics + ['mean_motion','max_motion','std_motion','median_motion',
                       'peak_ratio','zero_motion_pct']
X2 = pd.concat([tp_full[all_feats], fp_full[all_feats]], ignore_index=True)
y2 = np.array([0]*N_TP2 + [1]*N_FP2)

# Combined logistic regression
sc2 = StandardScaler(); X2s = sc2.fit_transform(X2)
lr2 = LogisticRegression(max_iter=5000, class_weight='balanced')
lr2.fit(X2s, y2)
scores2 = lr2.predict_proba(X2s)[:, 1]

print("\nFeature weights (combined ECG+motion):")
for f, c in zip(all_feats, lr2.coef_[0]):
    print(f"  {f:>18}  {c:+8.3f}")
print()
best_c = (None, -1, None)
for t in np.linspace(0.05, 0.95, 181):
    pred = scores2 >= t
    keep_tp = pred[:N_TP2] == False
    keep_fp = pred[N_TP2:] == False
    tp_r = keep_tp.sum()/N_TP2*100
    fp_s = (1 - keep_fp.sum()/N_FP2)*100
    if min(tp_r, fp_s) > best_c[1]:
        best_c = (t, min(tp_r, fp_s), (tp_r, fp_s))
print(f"\nBest combined LR operating point: thresh={best_c[0]:.3f}  TP_ret={best_c[2][0]:.1f}%  FP_sup={best_c[2][1]:.1f}%")

# Sweep at clinically relevant operating points
print(f"\n{'P(FP) thresh':>13}  {'TP_ret%':>8}  {'FP_sup%':>8}  {'min':>5}")
for t in [0.30, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80]:
    pred = scores2 >= t
    keep_tp = pred[:N_TP2] == False; keep_fp = pred[N_TP2:] == False
    tp_r = keep_tp.sum()/N_TP2*100; fp_s = (1-keep_fp.sum()/N_FP2)*100
    flag = "  **90/90**" if tp_r>=90 and fp_s>=90 else ""
    print(f"{t:>13.2f}  {tp_r:>8.1f}  {fp_s:>8.1f}  {min(tp_r,fp_s):>5.1f}{flag}")

# Decision tree on combined
print("\nCombined Decision Tree:")
for depth in [2, 3, 4, 5, 6]:
    dt = DecisionTreeClassifier(max_depth=depth, class_weight='balanced',
                                min_samples_leaf=5, random_state=42)
    dt.fit(X2, y2)
    pred = dt.predict(X2)
    tp_r = (pred[:N_TP2]==0).sum()/N_TP2*100
    fp_s = (1 - (pred[N_TP2:]==0).sum()/N_FP2)*100
    flag = "  ** 90/90 **" if tp_r >= 90 and fp_s >= 90 else ""
    print(f"  depth={depth}:  TP_ret={tp_r:.1f}%  FP_sup={fp_s:.1f}%{flag}")

# Best tree
dt = DecisionTreeClassifier(max_depth=4, class_weight='balanced',
                            min_samples_leaf=5, random_state=42)
dt.fit(X2, y2)
print(f"\nBest tree structure (depth=4):")
print(export_text(dt, feature_names=all_feats))

# Manual rule from top discriminators
print("\n" + "="*80)
print("HAND-CRAFTED COMBINED RULE")
print("="*80)
# Top discriminators by single-feature analysis
# Then AND-combine
print("\nRule: KEEP if  mean_rr<=900  AND  kurt<=6.5  AND  std_motion<=50  AND  samp_en>=1.2")
print("(Lower than active-motion AFib pattern with bradycardic+peaky+motion-heavy FPs)")
rule_tp = (tp_full['mean_rr']<=900) & (tp_full['kurt']<=6.5) & (tp_full['std_motion']<=50) & (tp_full['samp_en']>=1.2)
rule_fp = (fp_full['mean_rr']<=900) & (fp_full['kurt']<=6.5) & (fp_full['std_motion']<=50) & (fp_full['samp_en']>=1.2)
tp_r = rule_tp.sum()/N_TP2*100; fp_s = (1-rule_fp.sum()/N_FP2)*100
print(f"  → TP_ret={tp_r:.1f}%  FP_sup={fp_s:.1f}%  (TP kept={rule_tp.sum()}, FP kept={rule_fp.sum()})")

# Grid-search the simple hand rule
print("\nGrid search on (mean_rr, kurt) two-feature rule:")
best = []
for trr in np.arange(700, 1100, 50):
    for tk in np.arange(4.0, 9.0, 0.25):
        keep_tp = (tp_full['mean_rr']<=trr) & (tp_full['kurt']<=tk)
        keep_fp = (fp_full['mean_rr']<=trr) & (fp_full['kurt']<=tk)
        tp_r = keep_tp.sum()/N_TP2*100; fp_s = (1-keep_fp.sum()/N_FP2)*100
        if tp_r>=90 and fp_s>=90:
            best.append((trr, tk, tp_r, fp_s))
best.sort(key=lambda r: -(r[2]+r[3]))
print(f"Found {len(best)} 2-feature rules meeting BOTH >=90%")
for b in best[:15]:
    print(f"  KEEP if mean_rr<={b[0]:.0f}ms AND kurt<={b[1]:.2f}  =>  TP={b[2]:.1f}%  FP_sup={b[3]:.1f}%")

# Three feature
print("\nGrid search on (mean_rr, kurt, samp_en):")
best3 = []
for trr in [800, 850, 900, 950, 1000]:
    for tk in np.arange(4.5, 8.0, 0.25):
        for tse in np.arange(0.8, 2.0, 0.1):
            keep_tp = (tp_full['mean_rr']<=trr) & (tp_full['kurt']<=tk) & (tp_full['samp_en']>=tse)
            keep_fp = (fp_full['mean_rr']<=trr) & (fp_full['kurt']<=tk) & (fp_full['samp_en']>=tse)
            tp_r = keep_tp.sum()/N_TP2*100; fp_s = (1-keep_fp.sum()/N_FP2)*100
            if tp_r>=90 and fp_s>=90:
                best3.append((trr, tk, tse, tp_r, fp_s))
best3.sort(key=lambda r: -(r[3]+r[4]))
print(f"Found {len(best3)} 3-feature rules meeting BOTH >=90%")
for b in best3[:15]:
    print(f"  KEEP if mean_rr<={b[0]:.0f} AND kurt<={b[1]:.2f} AND samp_en>={b[2]:.2f}  =>  TP={b[3]:.1f}%  FP_sup={b[4]:.1f}%")
