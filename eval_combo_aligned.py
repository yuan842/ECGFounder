#!/usr/bin/env python3
"""
eval_combo_aligned.py

Properly evaluate the combination filter by extracting ECG features on the
EXACT same 46 FP records that have motion features.
"""
import pandas as pd, numpy as np, os, sys
sys.path.insert(0, '.')
from ecg_feature_analysis import extract_features_for_event

# Use the EXACT json_path values from the motion CSVs
motion_fp = pd.read_csv('res/motion_analysis/data/fp_motion_features.csv')
motion_tp = pd.read_csv('res/motion_analysis/data/tp_motion_features.csv')
motion_fp = motion_fp[motion_fp['event_type'] == 'Atrial Fibrillation'].reset_index(drop=True)
motion_tp = motion_tp[motion_tp['event_type'] == 'Atrial Fibrillation'].reset_index(drop=True)

def normpath(p):
    p = str(p).replace('\\', '/')
    # strip leading "._" macOS metadata prefix from basename
    parts = p.split('/')
    parts[-1] = parts[-1].replace('._', '', 1) if parts[-1].startswith('._') else parts[-1]
    return '/'.join(parts)

print(f"Motion FP records: {len(motion_fp)}")
print(f"Motion TP records: {len(motion_tp)}")

# Extract ECG features for every FP and (capped) TP record
print("\nExtracting ECG features for FP records...")
fp_rows = []
for i, row in motion_fp.iterrows():
    p = normpath(row['json_path'])
    if not os.path.exists(p):
        # try without ._ strip
        p2 = str(row['json_path']).replace('\\', '/')
        if os.path.exists(p2): p = p2
        else: continue
    feats = extract_features_for_event(p)
    if feats:
        feats['idx'] = i
        feats['mean_motion'] = row['mean_motion']
        feats['max_motion']  = row['max_motion']
        feats['std_motion']  = row['std_motion']
        feats['median_motion'] = row['median_motion']
        feats['peak_ratio']  = row['peak_ratio']
        feats['zero_motion_pct'] = row['zero_motion_pct']
        fp_rows.append(feats)
fp = pd.DataFrame(fp_rows)
print(f"  FP joined records: {len(fp)}")

# For TP, only do up to 200 to keep runtime manageable
print("\nExtracting ECG features for TP records (cap 300)...")
tp_rows = []
sampled_tp = motion_tp.head(300)
for i, row in sampled_tp.iterrows():
    p = normpath(row['json_path'])
    if not os.path.exists(p):
        p2 = str(row['json_path']).replace('\\', '/')
        if os.path.exists(p2): p = p2
        else: continue
    feats = extract_features_for_event(p)
    if feats:
        feats['idx'] = i
        feats['mean_motion'] = row['mean_motion']
        feats['max_motion']  = row['max_motion']
        feats['std_motion']  = row['std_motion']
        feats['median_motion'] = row['median_motion']
        feats['peak_ratio']  = row['peak_ratio']
        feats['zero_motion_pct'] = row['zero_motion_pct']
        tp_rows.append(feats)
    if (i+1) % 50 == 0:
        print(f"  Processed {i+1}/{len(sampled_tp)}")
tp = pd.DataFrame(tp_rows)
print(f"  TP joined records: {len(tp)}")

tp.to_csv('res/motion_analysis/data/afib_tp_combined_features.csv', index=False)
fp.to_csv('res/motion_analysis/data/afib_fp_combined_features.csv', index=False)

# ──────────────────────────────────────────────────────────────────
# EVALUATE COMBO FILTER
# ──────────────────────────────────────────────────────────────────
N_TP, N_FP = len(tp), len(fp)
print("\n" + "="*80)
print(f"COMBINATION FILTER EVALUATION  (TP n={N_TP}, FP n={N_FP})")
print("="*80)
print("Rule: KEEP if  kurt < 6  AND  mean_rr < 850 ms  AND  mean_motion < 12 mG")

# Helper: treat NaN as fail (do not keep)
def rule(df, kurt_max=6, mrr_max=850, mm_max=12):
    return ((df['kurt'].fillna(9999) < kurt_max) &
            (df['mean_rr'].fillna(9999) < mrr_max) &
            (df['mean_motion'].fillna(9999) < mm_max))

keep_tp = rule(tp); keep_fp = rule(fp)
tp_ret = 100*keep_tp.sum()/N_TP
fp_sup = 100*(1 - keep_fp.sum()/N_FP)
print(f"\nResult:")
print(f"  TP kept:        {keep_tp.sum()}/{N_TP} = {tp_ret:.1f}%")
print(f"  FP kept:        {keep_fp.sum()}/{N_FP} = {100*keep_fp.sum()/N_FP:.1f}%")
print(f"  TP retention:   {tp_ret:.1f}%")
print(f"  FP suppression: {fp_sup:.1f}%")
flag = "✓ MEETS 90/90" if tp_ret >= 90 and fp_sup >= 90 else "✗ does NOT meet 90/90"
print(f"  Status: {flag}")

# Per-condition contribution
print("\n── Per-condition pass rates ──")
print(f"{'Condition':>35}  {'TP_pass':>9}  {'FP_pass':>9}  {'FP_supp':>8}")
conds = {
    'kurt < 6':        (tp['kurt'].fillna(9999) < 6,        fp['kurt'].fillna(9999) < 6),
    'mean_rr < 850':   (tp['mean_rr'].fillna(9999) < 850,   fp['mean_rr'].fillna(9999) < 850),
    'mean_motion < 12':(tp['mean_motion'].fillna(9999) < 12, fp['mean_motion'].fillna(9999) < 12),
}
for name, (kt, kf) in conds.items():
    tr = 100*kt.sum()/N_TP; fp_p = 100*kf.sum()/N_FP
    print(f"{name:>35}  {tr:>8.1f}%  {fp_p:>8.1f}%  {100-fp_p:>7.1f}%")

# Pairwise AND
print("\n── Pairwise AND combinations ──")
pairs = {
    'kurt<6 AND mean_rr<850':
        ((tp['kurt'].fillna(9999)<6) & (tp['mean_rr'].fillna(9999)<850),
         (fp['kurt'].fillna(9999)<6) & (fp['mean_rr'].fillna(9999)<850)),
    'kurt<6 AND mean_motion<12':
        ((tp['kurt'].fillna(9999)<6) & (tp['mean_motion'].fillna(9999)<12),
         (fp['kurt'].fillna(9999)<6) & (fp['mean_motion'].fillna(9999)<12)),
    'mean_rr<850 AND mean_motion<12':
        ((tp['mean_rr'].fillna(9999)<850) & (tp['mean_motion'].fillna(9999)<12),
         (fp['mean_rr'].fillna(9999)<850) & (fp['mean_motion'].fillna(9999)<12)),
}
print(f"{'Rule':>40}  {'TP_ret':>7}  {'FP_supp':>8}  {'90/90':>6}")
for name, (kt, kf) in pairs.items():
    tr = 100*kt.sum()/N_TP; fs = 100*(1-kf.sum()/N_FP)
    flag = " ✓ " if tr>=90 and fs>=90 else " ✗ "
    print(f"{name:>40}  {tr:>6.1f}%  {fs:>7.1f}%  {flag:>6}")

# Triple combo
print(f"\n{'TRIPLE: kurt<6 AND mean_rr<850 AND mean_motion<12':>60}  {tp_ret:>6.1f}%  {fp_sup:>7.1f}%")

# Why is TP retention low? Which condition is failing on TPs?
print("\n── TP rejections by reason ──")
all_pass_tp = rule(tp)
rejected_tp = tp[~all_pass_tp].copy()
print(f"Total TPs rejected: {len(rejected_tp)}/{N_TP}")
# Categorize each rejection
fail_kurt = (rejected_tp['kurt'].fillna(9999) >= 6) | rejected_tp['kurt'].isna()
fail_rr   = (rejected_tp['mean_rr'].fillna(9999) >= 850) | rejected_tp['mean_rr'].isna()
fail_mot  = rejected_tp['mean_motion'].fillna(9999) >= 12
print(f"  rejected by kurt:         {fail_kurt.sum()}")
print(f"  rejected by mean_rr:      {fail_rr.sum()}")
print(f"  rejected by mean_motion:  {fail_mot.sum()}  ← motion is the heaviest TP rejector")

# How TPs rejected ONLY because of motion (would have passed otherwise)
only_mot = (~rejected_tp.index.isin(rejected_tp[fail_kurt].index)) & \
           (~rejected_tp.index.isin(rejected_tp[fail_rr].index)) & \
           fail_mot.values
print(f"  rejected ONLY by motion:  {only_mot.sum()}  (these would pass the ECG checks)")
