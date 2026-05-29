#!/usr/bin/env python3
"""
system_performance.py

Evaluate the AFib FP-suppression system as an integrated pipeline,
combining all developed filters in cascade / ensemble configurations.

Filters under test:
  F1 - Motion Filter R (3-tier decision tree on motion only)
  F2 - ECG Tier-1 (mean_rr <= 700, n_peaks >= 8)
  F3 - User combo  (kurt < 6 AND mean_rr < 850 AND mean_motion < 30)
  F4 - P-wave gate (p_present_pct <= 30)
  F5 - Sample-entropy gate (samp_en >= 1.0)
  F6 - Persistence gate (persistence_pct >= 60)

System variants:
  S1 - Motion-only baseline (F1)
  S2 - ECG-only baseline (F2)
  S3 - Cascade: F2 → F1 (ECG first, then motion sanity check)
  S4 - AND of all primary thresholds (F3 + F4 + F5)
  S5 - Tiered system: zero-motion bypass + ECG + motion + P-wave
  S6 - Ensemble (logistic-regression-weighted score on full feature stack)
"""
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold

tp = pd.read_csv('res/motion_analysis/data/afib_tp_combined_features.csv')
fp = pd.read_csv('res/motion_analysis/data/afib_fp_combined_features.csv')
N_TP, N_FP = len(tp), len(fp)
print(f"Aligned cohort: TP={N_TP}, FP={N_FP}")

# ─── Filter definitions ────────────────────────────────────────────────────
def F1_motion_R(df):
    """Motion Filter R — 3-tier decision tree."""
    std_m = df['std_motion'].fillna(0)
    mean_m = df['mean_motion'].fillna(0)
    median_m = df['median_motion'].fillna(0)
    peak_r = df['peak_ratio'].fillna(0)
    keep = pd.Series([False]*len(df), index=df.index)
    # Tier 1: std <= 48.20 → KEEP
    t1 = std_m <= 48.20
    keep |= t1
    # Tier 2: 48.20 < std <= 50.28
    t2 = (std_m > 48.20) & (std_m <= 50.28)
    keep |= t2 & (median_m <= 2.65) & (peak_r <= 36.29)
    keep |= t2 & (median_m > 2.65) & (peak_r > 32.70)
    # Tier 3: std > 50.28 → KEEP if mean in (11.45, 15.77]
    t3 = std_m > 50.28
    keep |= t3 & (mean_m > 11.45) & (mean_m <= 15.77)
    # Otherwise REJECT
    return keep

def F2_ecg_tier1(df):
    return ((df['mean_rr'].fillna(9999) <= 700) &
            (df['n_peaks'].fillna(0) >= 8))

def F3_user_combo(df):
    return ((df['kurt'].fillna(9999) < 6) &
            (df['mean_rr'].fillna(9999) < 850) &
            (df['mean_motion'].fillna(9999) < 30))

def F4_pwave(df):
    return df['p_present_pct'].fillna(100) <= 30

def F5_samp_en(df):
    return df['samp_en'].fillna(0) >= 1.0

def F6_persist(df):
    return df['persistence_pct'].fillna(0) >= 60

# ─── Evaluation helper ─────────────────────────────────────────────────────
def perf(keep_tp, keep_fp, name=""):
    tk = int(keep_tp.sum()); fk = int(keep_fp.sum())
    tp_ret = 100*tk/N_TP
    fp_sup = 100*(1 - fk/N_FP)
    return tp_ret, fp_sup, tk, fk

# ═══════════════════════════════════════════════════════════════════════════
# INDIVIDUAL FILTER PERFORMANCE
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "="*80)
print("INDIVIDUAL FILTER PERFORMANCE")
print("="*80)
print(f"{'Filter':>30}  {'TP_ret%':>8}  {'FP_sup%':>8}  {'TP_kept':>8}  {'FP_kept':>8}  {'90/90':>6}")
filters = [
    ('F1 — Motion Filter R',         F1_motion_R),
    ('F2 — ECG Tier-1 (HR)',         F2_ecg_tier1),
    ('F3 — User combo (kurt+rr+mot)', F3_user_combo),
    ('F4 — P-wave gate',              F4_pwave),
    ('F5 — Sample entropy',           F5_samp_en),
    ('F6 — Persistence',              F6_persist),
]
individual_results = {}
for name, fn in filters:
    tr, fs, tk, fk = perf(fn(tp), fn(fp))
    flag = "✓" if tr>=90 and fs>=90 else ""
    print(f"{name:>30}  {tr:>7.1f}%  {fs:>7.1f}%  {tk:>8}  {fk:>8}  {flag:>6}")
    individual_results[name] = (tr, fs)

# ═══════════════════════════════════════════════════════════════════════════
# SYSTEM CONFIGURATIONS
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "="*80)
print("INTEGRATED SYSTEM CONFIGURATIONS")
print("="*80)

systems = []

# S1 — Motion only
def S1(df): return F1_motion_R(df)
systems.append(('S1 — Motion-only baseline', S1))

# S2 — ECG only (Tier-1)
def S2(df): return F2_ecg_tier1(df)
systems.append(('S2 — ECG-only baseline', S2))

# S3 — Cascade: ECG first, then motion sanity check
def S3(df): return F2_ecg_tier1(df) & F1_motion_R(df)
systems.append(('S3 — Cascade ECG→Motion (AND)', S3))

# S4 — User combo + P-wave + sample entropy
def S4(df): return F3_user_combo(df) & F4_pwave(df) & F5_samp_en(df)
systems.append(('S4 — AND(F3,F4,F5)', S4))

# S5 — Tiered: zero-motion bypass + ECG + motion
def S5(df):
    # Tier 1: zero motion → KEEP unconditionally
    quiet = df['mean_motion'].fillna(9999) < 1.0
    # Tier 2: tachycardic AFib with bounded motion
    tachy = ((df['mean_rr'].fillna(9999) < 850) &
             (df['kurt'].fillna(9999) < 6) &
             (df['mean_motion'].fillna(9999) < 30))
    # Tier 3: bradycardic AFib rescue (irregular + chaotic)
    brady_rescue = ((df['mean_rr'].fillna(9999) < 1100) &
                    (df['samp_en'].fillna(0) >= 1.3) &
                    (df['p_present_pct'].fillna(100) <= 30))
    return quiet | tachy | brady_rescue
systems.append(('S5 — Tiered (quiet|tachy|rescue)', S5))

# S6 — Ensemble (logistic regression on all features)
feats = ['mean_motion','max_motion','std_motion','median_motion','peak_ratio','zero_motion_pct',
         'mean_rr','rmssd','pnn50','samp_en','cv_rr','kurt','baseline_drift','snr_proxy',
         'p_present_pct','p_consistency','persistence_pct','n_peaks']
feats = [f for f in feats if f in tp.columns and f in fp.columns]
X = pd.concat([tp[feats], fp[feats]], ignore_index=True).fillna(0)
y = np.array([0]*N_TP + [1]*N_FP)
sc = StandardScaler(); Xs = sc.fit_transform(X)
lr = LogisticRegression(max_iter=5000, class_weight='balanced', C=1.0)
lr.fit(Xs, y)
scores_lr = lr.predict_proba(Xs)[:, 1]

# Pick threshold maximizing min(TP_ret, FP_sup)
best = (-1, None)
for t in np.linspace(0.05, 0.95, 181):
    pred_keep = scores_lr < t
    tk = pred_keep[:N_TP].sum(); fk = pred_keep[N_TP:].sum()
    tr = 100*tk/N_TP; fs = 100*(1-fk/N_FP)
    if min(tr, fs) > best[0]:
        best = (min(tr, fs), (t, tr, fs))
t_opt, tr_opt, fs_opt = best[1]
print(f"\n  (S6 LR optimal threshold: {t_opt:.3f}  → TP={tr_opt:.1f}%, FP_sup={fs_opt:.1f}%)")

def S6(df):
    Xt = sc.transform(df[feats].fillna(0).values)
    s = lr.predict_proba(Xt)[:, 1]
    return pd.Series(s < t_opt, index=df.index)
systems.append(('S6 — LR Ensemble (in-sample)', S6))

# Run all systems
print(f"\n{'System':>40}  {'TP_ret%':>8}  {'FP_sup%':>8}  {'TP_kept':>8}  {'FP_kept':>8}  {'90/90':>6}")
sys_results = {}
for name, fn in systems:
    keep_tp = fn(tp); keep_fp = fn(fp)
    tr, fs, tk, fk = perf(keep_tp, keep_fp)
    flag = "✓" if tr>=90 and fs>=90 else ""
    print(f"{name:>40}  {tr:>7.1f}%  {fs:>7.1f}%  {tk:>8}  {fk:>8}  {flag:>6}")
    sys_results[name] = (tr, fs, tk, fk)

# ═══════════════════════════════════════════════════════════════════════════
# CROSS-VALIDATED S6 (held-out generalization estimate)
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "="*80)
print("S6 — 5-FOLD CROSS-VALIDATED PERFORMANCE (held-out)")
print("="*80)
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_tr, cv_fs = [], []
for fold, (tr_idx, te_idx) in enumerate(skf.split(X, y), 1):
    sc2 = StandardScaler(); Xtr = sc2.fit_transform(X.iloc[tr_idx]); Xte = sc2.transform(X.iloc[te_idx])
    lr2 = LogisticRegression(max_iter=5000, class_weight='balanced'); lr2.fit(Xtr, y[tr_idx])
    s = lr2.predict_proba(Xte)[:, 1]
    # Find optimal threshold on training fold
    best_local = (-1, 0.5)
    for t in np.linspace(0.05, 0.95, 91):
        s_train = lr2.predict_proba(Xtr)[:, 1]
        keep = s_train < t
        tk = (keep & (y[tr_idx]==0)).sum() / max(1, (y[tr_idx]==0).sum())
        fk = (keep & (y[tr_idx]==1)).sum() / max(1, (y[tr_idx]==1).sum())
        m = min(tk*100, (1-fk)*100)
        if m > best_local[0]:
            best_local = (m, t)
    t_use = best_local[1]
    keep = s < t_use
    tk_test = (keep & (y[te_idx]==0)).sum()
    fk_test = (keep & (y[te_idx]==1)).sum()
    n_tp_test = (y[te_idx]==0).sum(); n_fp_test = (y[te_idx]==1).sum()
    tr_f = 100*tk_test/max(1,n_tp_test); fs_f = 100*(1-fk_test/max(1,n_fp_test))
    cv_tr.append(tr_f); cv_fs.append(fs_f)
    print(f"  Fold {fold}: thresh={t_use:.2f}  TP_ret={tr_f:.1f}%  FP_sup={fs_f:.1f}%  "
          f"(test n_tp={n_tp_test}, n_fp={n_fp_test})")
print(f"\n  Mean CV:  TP_ret = {np.mean(cv_tr):.1f}% ± {np.std(cv_tr):.1f}")
print(f"            FP_sup = {np.mean(cv_fs):.1f}% ± {np.std(cv_fs):.1f}")
cv_min = min(np.mean(cv_tr), np.mean(cv_fs))
print(f"  → Realistic out-of-sample 90/90: {'YES ✓' if cv_min>=90 else 'NO'}")

# ═══════════════════════════════════════════════════════════════════════════
# SYSTEM-LEVEL CLINICAL METRICS
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "="*80)
print("SYSTEM-LEVEL CLINICAL METRICS")
print("="*80)
print("Confusion matrices and derived metrics for each system.")
print()

# True prevalence in production data (use the original dataset proportions: 750 TP : 102,634 FP)
# Or for AFib-specific: 750 TP vs 46 FP in our cohort. Production may differ.
prev_tp = 750; prev_fp = 102634  # FP dataset total
print(f"Production prevalence assumed: TP rate = {prev_tp/(prev_tp+prev_fp)*100:.2f}%")
print(f"FP-to-TP ratio in production: {prev_fp/prev_tp:.1f}:1")
print()
print(f"{'System':>40}  {'PPV%':>5}  {'NPV%':>5}  {'Alerts/day':>11}  {'Missed/day':>11}")
print(f"  (assumes 1 patient-day = 1 detection event; rescale by your event rate)")

baseline_alarms = prev_tp + prev_fp  # all detected events per "day"
for name, (tr, fs, tk, fk) in sys_results.items():
    # In production-scaled cohort:
    prod_tp_kept = prev_tp * (tr/100)
    prod_fp_kept = prev_fp * (1 - fs/100)
    prod_tp_rej  = prev_tp * (1 - tr/100)
    if prod_tp_kept + prod_fp_kept > 0:
        ppv = 100 * prod_tp_kept / (prod_tp_kept + prod_fp_kept)
    else:
        ppv = 0
    if prev_tp - prod_tp_kept + prev_fp - prod_fp_kept > 0:
        npv = 100 * (prev_fp - prod_fp_kept) / (prev_fp - prod_fp_kept + prev_tp - prod_tp_kept)
    else:
        npv = 0
    alerts = prod_tp_kept + prod_fp_kept
    missed = prod_tp_rej
    print(f"{name:>40}  {ppv:>4.1f}  {npv:>4.1f}  {alerts:>11.0f}  {missed:>11.0f}")

print("\nPPV = positive predictive value = fraction of alerts that are real")
print("NPV = negative predictive value = fraction of rejected events that were truly FP")
print("Alerts/day = how many alarms the clinician still sees")
print("Missed/day = how many true AFib events are silently rejected")
