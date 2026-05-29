"""
Motion Gating Threshold Optimization
=====================================
Sweeps multiple threshold values for each rhythm class, computing:
  - FP Suppression Rate  (higher is better - we want to gate false alarms)
  - TP Preservation Rate  (higher is better - we must not gate real events)
Uses both the FP dataset (ecg_fp_doctor removed1) and TP dataset (ecg-tp_rex).
"""
import os, json, sys
import numpy as np
import pandas as pd
from collections import defaultdict

FP_DIR = "./data/ecg_fp_doctor removed1"
FP_CSV = "./data/ecg_fp_doctor removed1/summary.csv"
TP_DIR = "./data/ecg-tp_rex"
TP_CSV = "./data/ecg-tp_rex/summary.csv"
SAVED_DIR = "./res/threshold_optimization"
os.makedirs(SAVED_DIR, exist_ok=True)

# Classes we evaluate (must exist in both datasets for full analysis)
LABEL_MAP = {
    "Atrial Fibrillation":             5,
    "Isolated Ventricular Beat":       9,
    "Prolonged RR Interval":          81,
    "Ventricular Couplet":            91,
    "Ventricular Run":                99,
    "Pause":                         143,
    "Sinus Tachycardia":               6,
    "Supraventricular Couplet":       20,
    "Isolated Supraventricular Beat": 16,
}

def extract_motion_features(json_path: str):
    """Extract motion features from a single JSON record."""
    try:
        with open(json_path) as f:
            records = json.load(f)
    except Exception:
        return None

    raw_acc = []
    for r in records:
        data = r['data']
        if 'acc' in data:
            for frame in data['acc']:
                raw_acc.append([frame.get('x', 0), frame.get('y', 0), frame.get('z', 0)])

    if len(raw_acc) == 0:
        return None

    acc = np.array(raw_acc, dtype=np.float32) / 2048.0
    vm = np.sqrt(acc[:, 0]**2 + acc[:, 1]**2 + acc[:, 2]**2)
    vm_dc = pd.Series(vm).rolling(window=5, min_periods=1, center=True).mean().values
    vm_dyn = np.abs(vm - vm_dc) * 1000.0  # mG

    mean_motion = float(np.mean(vm_dyn))
    max_motion  = float(np.max(vm_dyn))
    std_motion  = float(np.std(vm_dyn))

    # Peak-to-baseline ratio
    if std_motion > 0:
        peak_ratio = max_motion / std_motion
    else:
        peak_ratio = 0.0

    # Dominant frequency (for bradycardia / tachycardia stride-lock detection)
    fs_acc = 5  # 5 Hz native ACC rate
    if len(vm_dyn) > 10:
        fft_vals = np.abs(np.fft.rfft(vm_dyn))
        fft_freqs = np.fft.rfftfreq(len(vm_dyn), d=1.0/fs_acc)
        if len(fft_vals) > 1:
            dom_freq = fft_freqs[np.argmax(fft_vals[1:]) + 1]
        else:
            dom_freq = 0.0
    else:
        dom_freq = 0.0

    return {
        'mean_motion': mean_motion,
        'max_motion': max_motion,
        'std_motion': std_motion,
        'peak_ratio': peak_ratio,
        'dom_freq': dom_freq,
    }

def load_features(csv_path, data_dir, event_types, max_per_class=200):
    """Load motion features for each event type from a dataset."""
    df = pd.read_csv(csv_path)
    results = defaultdict(list)
    for et in event_types:
        sub = df[df['Event Type'] == et]
        if len(sub) == 0:
            continue
        sample = sub.sample(min(max_per_class, len(sub)), random_state=42)
        for _, row in sample.iterrows():
            path = os.path.join(data_dir, row['JSON File'].replace('\\', '/'))
            feat = extract_motion_features(path)
            if feat:
                results[et].append(feat)
    return results

def gating_decision(feat, event_type, thresholds):
    """Apply gating rule for a given event type using supplied thresholds."""
    if event_type == "Atrial Fibrillation":
        return feat['mean_motion'] > thresholds.get('afib_mean', 150.0)

    elif event_type in ["Pause", "Prolonged RR Interval"]:
        return feat['max_motion'] > thresholds.get('pause_max', 400.0)

    elif event_type in ["Sinus Tachycardia", "Ventricular Run"]:
        return feat['max_motion'] > thresholds.get('tachy_max', 300.0)

    elif "Beat" in event_type or "Couplet" in event_type:
        return feat['peak_ratio'] > thresholds.get('ectopic_ratio', 2.5)

    return False

def sweep():
    print("Loading FP features ...")
    fp_feats = load_features(FP_CSV, FP_DIR, LABEL_MAP.keys(), max_per_class=200)
    print("Loading TP features ...")
    tp_feats = load_features(TP_CSV, TP_DIR, LABEL_MAP.keys(), max_per_class=200)

    print(f"\nFP classes loaded: {[(k, len(v)) for k, v in fp_feats.items()]}")
    print(f"TP classes loaded: {[(k, len(v)) for k, v in tp_feats.items()]}")

    # ---------------------------------------------------------------
    # Sweep configurations per gating parameter
    # ---------------------------------------------------------------
    sweep_configs = {
        'afib_mean': {
            'values': [5, 8, 10, 12, 15, 20, 25, 30, 40, 50, 75, 100, 150],
            'affected_classes': ['Atrial Fibrillation'],
            'label': 'AFib Mean Dynamic Motion (mG)',
        },
        'pause_max': {
            'values': [30, 50, 67, 80, 100, 150, 200, 300, 400],
            'affected_classes': ['Pause', 'Prolonged RR Interval'],
            'label': 'Pause / Prolonged RR Max Spike (mG)',
        },
        'tachy_max': {
            'values': [30, 50, 67, 80, 100, 150, 200, 300, 400],
            'affected_classes': ['Sinus Tachycardia', 'Ventricular Run'],
            'label': 'Tachycardia / VT Run Max Motion (mG)',
        },
        'ectopic_ratio': {
            'values': [1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5, 3.0, 4.0],
            'affected_classes': [
                'Isolated Ventricular Beat', 'Isolated Supraventricular Beat',
                'Supraventricular Couplet', 'Ventricular Couplet',
            ],
            'label': 'Ectopic Peak-to-Baseline Ratio',
        },
    }

    all_results = []

    for param, cfg in sweep_configs.items():
        print(f"\n{'='*70}")
        print(f"SWEEPING: {cfg['label']}")
        print(f"{'='*70}")

        for threshold_val in cfg['values']:
            thresholds = {param: threshold_val}

            for et in cfg['affected_classes']:
                fp_list = fp_feats.get(et, [])
                tp_list = tp_feats.get(et, [])

                if len(fp_list) == 0:
                    continue

                fp_gated = sum(1 for f in fp_list if gating_decision(f, et, thresholds))
                fp_suppression = fp_gated / len(fp_list) * 100.0

                if len(tp_list) > 0:
                    tp_gated = sum(1 for f in tp_list if gating_decision(f, et, thresholds))
                    tp_preserved = (1.0 - tp_gated / len(tp_list)) * 100.0
                    tp_lost = tp_gated / len(tp_list) * 100.0
                else:
                    tp_preserved = None
                    tp_lost = None

                row = {
                    'Parameter': cfg['label'],
                    'Threshold': threshold_val,
                    'Event Type': et,
                    'FP Count': len(fp_list),
                    'FP Gated': fp_gated,
                    'FP Suppression (%)': fp_suppression,
                    'TP Count': len(tp_list) if tp_list else 0,
                    'TP Lost': tp_gated if tp_list else 'N/A',
                    'TP Preserved (%)': tp_preserved if tp_preserved is not None else 'N/A',
                }
                all_results.append(row)

                tp_str = f"TP Preserved: {tp_preserved:.1f}%" if tp_preserved is not None else "TP: N/A"
                print(f"  {et:<35} Threshold={threshold_val:<8} | FP Suppression: {fp_suppression:5.1f}%  |  {tp_str}")

    results_df = pd.DataFrame(all_results)
    results_df.to_csv(os.path.join(SAVED_DIR, "threshold_sweep_results.csv"), index=False)
    print(f"\nSaved sweep results to {SAVED_DIR}/threshold_sweep_results.csv")

    # ---------------------------------------------------------------
    # Select optimal thresholds (maximize FP suppression with TP >= 95%)
    # ---------------------------------------------------------------
    print("\n" + "="*70)
    print("OPTIMAL THRESHOLD RECOMMENDATIONS (Target: TP Preserved >= 95%)")
    print("="*70)

    optimal = {}
    for param, cfg in sweep_configs.items():
        for et in cfg['affected_classes']:
            sub = results_df[(results_df['Parameter'] == cfg['label']) & (results_df['Event Type'] == et)]
            if len(sub) == 0:
                continue

            # Filter for TP >= 95% (or no TP data available)
            safe = sub[(sub['TP Preserved (%)'] == 'N/A') | (sub['TP Preserved (%)'].apply(lambda x: float(x) >= 95.0 if x != 'N/A' else True))]
            if len(safe) > 0:
                best = safe.loc[safe['FP Suppression (%)'].idxmax()]
            else:
                # If no threshold achieves 95% TP preservation, pick the one closest
                best = sub.loc[sub['FP Suppression (%)'].idxmax()]

            optimal[et] = {
                'param': param,
                'threshold': best['Threshold'],
                'fp_suppression': best['FP Suppression (%)'],
                'tp_preserved': best['TP Preserved (%)'],
            }
            print(f"  {et:<35} -> {param}={best['Threshold']:<8}  |  FP Suppression: {best['FP Suppression (%)']:.1f}%  |  TP Preserved: {best['TP Preserved (%)']}")

    # Save optimal thresholds
    opt_df = pd.DataFrame([
        {'Event Type': k, 'Parameter': v['param'], 'Optimal Threshold': v['threshold'],
         'FP Suppression (%)': v['fp_suppression'], 'TP Preserved (%)': v['tp_preserved']}
        for k, v in optimal.items()
    ])
    opt_df.to_csv(os.path.join(SAVED_DIR, "optimal_thresholds.csv"), index=False)
    print(f"\nSaved optimal thresholds to {SAVED_DIR}/optimal_thresholds.csv")

if __name__ == '__main__':
    sweep()
