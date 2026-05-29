#!/usr/bin/env python3
"""
evaluate_combo_filter.py

Evaluate the user-specified combination filter on AFib events:
  KEEP if  kurt < 6  AND  mean_rr < 850 ms  AND  mean_motion < 12 mG
"""
import pandas as pd
import numpy as np
import os

# Load ECG features
tp_ecg = pd.read_csv('res/motion_analysis/data/afib_tp_ecg_features.csv')
fp_ecg = pd.read_csv('res/motion_analysis/data/afib_fp_ecg_features.csv')

# Load motion features
motion = pd.read_csv('res/motion_analysis/data/fp_motion_features.csv')
motion_fp = motion[motion['event_type'] == 'Atrial Fibrillation'].reset_index(drop=True)
motion_tp = pd.read_csv('res/motion_analysis/data/tp_motion_features.csv')
motion_tp = motion_tp[motion_tp['event_type'] == 'Atrial Fibrillation'].reset_index(drop=True)

print(f"ECG cohort: TP={len(tp_ecg)}, FP={len(fp_ecg)}")
print(f"Motion cohort: TP={len(motion_tp)}, FP={len(motion_fp)}")

# Join ECG and motion on basename
def basename(p): return os.path.basename(str(p)) if isinstance(p, str) else ''
tp_ecg['bn'] = tp_ecg['path'].apply(basename)
fp_ecg['bn'] = fp_ecg['path'].apply(basename)
motion_tp['bn'] = motion_tp['json_path'].apply(basename)
motion_fp['bn'] = motion_fp['json_path'].apply(basename)

# Try different basename normalization for cross-platform paths
def norm_bn(p):
    p = str(p).replace('\\', '/')
    return p.split('/')[-1]
tp_ecg['bn'] = tp_ecg['path'].apply(norm_bn)
fp_ecg['bn'] = fp_ecg['path'].apply(norm_bn)
motion_tp['bn'] = motion_tp['json_path'].apply(norm_bn)
motion_fp['bn'] = motion_fp['json_path'].apply(norm_bn)

# Strip "._" prefixes (macOS metadata files) from motion side
motion_fp['bn_clean'] = motion_fp['bn'].str.replace(r'^\._', '', regex=True)
motion_tp['bn_clean'] = motion_tp['bn'].str.replace(r'^\._', '', regex=True)

tp_joined = tp_ecg.merge(
    motion_tp[['bn_clean', 'mean_motion', 'max_motion', 'std_motion',
               'median_motion', 'peak_ratio', 'zero_motion_pct']],
    left_on='bn', right_on='bn_clean', how='inner')
fp_joined = fp_ecg.merge(
    motion_fp[['bn_clean', 'mean_motion', 'max_motion', 'std_motion',
               'median_motion', 'peak_ratio', 'zero_motion_pct']],
    left_on='bn', right_on='bn_clean', how='inner')

print(f"\nJoined ECG+motion: TP={len(tp_joined)}, FP={len(fp_joined)}")

# If join fails for FP, evaluate ECG and motion separately and report each
if len(fp_joined) == 0:
    print("WARNING: FP join produced 0 rows — filename mismatch between ECG and motion CSVs.")
    print("Evaluating each component independently against the full cohorts.\n")

N_TP_ECG = len(tp_ecg); N_FP_ECG = len(fp_ecg)
N_TP_MOT = len(motion_tp); N_FP_MOT = len(motion_fp)

# ──────────────────────────────────────────────────────────────────
# COMPONENT-WISE EVALUATION (each rule applied independently)
# ──────────────────────────────────────────────────────────────────
print("="*80)
print("PER-COMPONENT EVALUATION (each rule, independent cohort)")
print("="*80)

# Rule 1: kurt < 6
keep_tp = (tp_ecg['kurt'] < 6) & tp_ecg['kurt'].notna()
keep_fp = (fp_ecg['kurt'] < 6) & fp_ecg['kurt'].notna()
print(f"\nRule 1: kurt < 6")
print(f"  TP retention: {keep_tp.sum()}/{N_TP_ECG} = {100*keep_tp.sum()/N_TP_ECG:.1f}%")
print(f"  FP kept:      {keep_fp.sum()}/{N_FP_ECG} = {100*keep_fp.sum()/N_FP_ECG:.1f}%  → suppression = {100*(1-keep_fp.sum()/N_FP_ECG):.1f}%")

# Rule 2: mean_rr < 850 ms
keep_tp = (tp_ecg['mean_rr'] < 850) & tp_ecg['mean_rr'].notna()
keep_fp = (fp_ecg['mean_rr'] < 850) & fp_ecg['mean_rr'].notna()
print(f"\nRule 2: mean_rr < 850 ms")
print(f"  TP retention: {keep_tp.sum()}/{N_TP_ECG} = {100*keep_tp.sum()/N_TP_ECG:.1f}%")
print(f"  FP kept:      {keep_fp.sum()}/{N_FP_ECG} = {100*keep_fp.sum()/N_FP_ECG:.1f}%  → suppression = {100*(1-keep_fp.sum()/N_FP_ECG):.1f}%")

# Rule 3: mean_motion < 12 mG
keep_tp = motion_tp['mean_motion'] < 12
keep_fp = motion_fp['mean_motion'] < 12
print(f"\nRule 3: mean_motion < 12 mG")
print(f"  TP retention: {keep_tp.sum()}/{N_TP_MOT} = {100*keep_tp.sum()/N_TP_MOT:.1f}%")
print(f"  FP kept:      {keep_fp.sum()}/{N_FP_MOT} = {100*keep_fp.sum()/N_FP_MOT:.1f}%  → suppression = {100*(1-keep_fp.sum()/N_FP_MOT):.1f}%")

# ──────────────────────────────────────────────────────────────────
# COMBINED FILTER (where ECG+motion are joined)
# ──────────────────────────────────────────────────────────────────
print("\n" + "="*80)
print("COMBINED AND-FILTER: kurt < 6  AND  mean_rr < 850  AND  mean_motion < 12")
print("="*80)

if len(tp_joined) > 0 and len(fp_joined) > 0:
    N_TP_J, N_FP_J = len(tp_joined), len(fp_joined)
    rule_tp = ((tp_joined['kurt'] < 6) &
               (tp_joined['mean_rr'] < 850) &
               (tp_joined['mean_motion'] < 12))
    rule_fp = ((fp_joined['kurt'] < 6) &
               (fp_joined['mean_rr'] < 850) &
               (fp_joined['mean_motion'] < 12))
    tp_kept = int(rule_tp.sum()); fp_kept = int(rule_fp.sum())
    tp_ret = 100*tp_kept/N_TP_J
    fp_sup = 100*(1 - fp_kept/N_FP_J)
    print(f"\nJoined cohort (n_TP={N_TP_J}, n_FP={N_FP_J}):")
    print(f"  TP kept:        {tp_kept}/{N_TP_J} = {tp_ret:.1f}%")
    print(f"  FP kept:        {fp_kept}/{N_FP_J} = {100*fp_kept/N_FP_J:.1f}%")
    print(f"  TP retention:   {tp_ret:.1f}%")
    print(f"  FP suppression: {fp_sup:.1f}%")
    flag = " ✓ MEETS 90/90" if tp_ret >= 90 and fp_sup >= 90 else " ✗ does not meet 90/90"
    print(f"  Status:        {flag}")

    # Break down how each rule contributes to TP loss
    print("\nTP rejection breakdown (which condition fails):")
    fails_kurt = ((tp_joined['kurt'] >= 6) | tp_joined['kurt'].isna()).sum()
    fails_rr   = ((tp_joined['mean_rr'] >= 850) | tp_joined['mean_rr'].isna()).sum()
    fails_mot  = (tp_joined['mean_motion'] >= 12).sum()
    print(f"  kurt >= 6:        {fails_kurt}/{N_TP_J} ({100*fails_kurt/N_TP_J:.1f}%)")
    print(f"  mean_rr >= 850:   {fails_rr}/{N_TP_J} ({100*fails_rr/N_TP_J:.1f}%)")
    print(f"  mean_motion >= 12:{fails_mot}/{N_TP_J} ({100*fails_mot/N_TP_J:.1f}%)")

    print("\nFP rejection breakdown:")
    f_kurt = ((fp_joined['kurt'] >= 6) | fp_joined['kurt'].isna()).sum()
    f_rr   = ((fp_joined['mean_rr'] >= 850) | fp_joined['mean_rr'].isna()).sum()
    f_mot  = (fp_joined['mean_motion'] >= 12).sum()
    print(f"  kurt >= 6:        {f_kurt}/{N_FP_J} ({100*f_kurt/N_FP_J:.1f}%)")
    print(f"  mean_rr >= 850:   {f_rr}/{N_FP_J} ({100*f_rr/N_FP_J:.1f}%)")
    print(f"  mean_motion >= 12:{f_mot}/{N_FP_J} ({100*f_mot/N_FP_J:.1f}%)")
else:
    print("\n⚠ Could not join ECG + motion features.")
    print("  Estimating combined performance from independent cohorts (upper bound).\n")
    # Independent estimate
    keep_tp_ecg = ((tp_ecg['kurt'] < 6) & (tp_ecg['mean_rr'] < 850) &
                   tp_ecg['kurt'].notna() & tp_ecg['mean_rr'].notna())
    keep_fp_ecg = ((fp_ecg['kurt'] < 6) & (fp_ecg['mean_rr'] < 850) &
                   fp_ecg['kurt'].notna() & fp_ecg['mean_rr'].notna())
    keep_tp_mot = (motion_tp['mean_motion'] < 12)
    keep_fp_mot = (motion_fp['mean_motion'] < 12)
    # Assuming independence
    tp_ecg_pass_rate = keep_tp_ecg.sum() / N_TP_ECG
    fp_ecg_pass_rate = keep_fp_ecg.sum() / N_FP_ECG
    tp_mot_pass_rate = keep_tp_mot.sum() / N_TP_MOT
    fp_mot_pass_rate = keep_fp_mot.sum() / N_FP_MOT
    print(f"  ECG sub-rule (kurt<6 AND mean_rr<850):")
    print(f"    TP pass rate: {100*tp_ecg_pass_rate:.1f}%")
    print(f"    FP pass rate: {100*fp_ecg_pass_rate:.1f}%  (suppression={100*(1-fp_ecg_pass_rate):.1f}%)")
    print(f"  Motion sub-rule (mean_motion<12):")
    print(f"    TP pass rate: {100*tp_mot_pass_rate:.1f}%")
    print(f"    FP pass rate: {100*fp_mot_pass_rate:.1f}%  (suppression={100*(1-fp_mot_pass_rate):.1f}%)")
    print(f"\n  ESTIMATED COMBINED (independence assumption):")
    tp_comb = tp_ecg_pass_rate * tp_mot_pass_rate * 100
    fp_comb = fp_ecg_pass_rate * fp_mot_pass_rate * 100
    print(f"    TP retention (est):   {tp_comb:.1f}%")
    print(f"    FP suppression (est): {100-fp_comb:.1f}%")
