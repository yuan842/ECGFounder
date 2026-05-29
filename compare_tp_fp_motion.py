import os
import json
import numpy as np
import pandas as pd

FP_DIR = "./data/ecg_fp_doctor removed1"
FP_CSV = "./data/ecg_fp_doctor removed1/summary.csv"

TP_DIR = "./data/ecg-tp_rex"
TP_CSV = "./data/ecg-tp_rex/summary.csv"

def extract_motion(json_path: str):
    with open(json_path) as f:
        records = json.load(f)
    raw_acc = []
    for r in records:
        data = r['data']
        if 'acc' in data:
            for frame in data['acc']:
                raw_acc.append([frame.get('x', 0), frame.get('y', 0), frame.get('z', 0)])
    if len(raw_acc) == 0:
        return None
    acc_arr = np.array(raw_acc, dtype=np.float32) / 2048.0
    vm = np.sqrt(acc_arr[:, 0]**2 + acc_arr[:, 1]**2 + acc_arr[:, 2]**2)
    vm_dc = pd.Series(vm).rolling(window=5, min_periods=1, center=True).mean().values
    vm_dynamic = np.abs(vm - vm_dc) * 1000.0
    return np.mean(vm_dynamic), np.max(vm_dynamic)

def run_comparison():
    print("Loading datasets...")
    fp_df = pd.read_csv(FP_CSV)
    fp_afib = fp_df[fp_df['Event Type'] == 'Atrial Fibrillation'].sample(100, random_state=42)
    
    tp_df = pd.read_csv(TP_CSV)
    tp_afib = tp_df[tp_df['Event Type'] == 'Atrial Fibrillation'].sample(100, random_state=42)
    
    fp_means, fp_maxes = [], []
    for _, row in fp_afib.iterrows():
        path = os.path.join(FP_DIR, row['JSON File'].replace('\\', '/'))
        res = extract_motion(path)
        if res:
            fp_means.append(res[0])
            fp_maxes.append(res[1])
            
    tp_means, tp_maxes = [], []
    for _, row in tp_afib.iterrows():
        path = os.path.join(TP_DIR, row['JSON File'].replace('\\', '/'))
        res = extract_motion(path)
        if res:
            tp_means.append(res[0])
            tp_maxes.append(res[1])
            
    print("\n" + "="*80)
    print("MOTION CHARACTERISTICS COMPARISON: AFIB FALSE POSITIVES VS. TRUE POSITIVES")
    print("="*80)
    
    print("\n[Average Dynamic Motion per Event (mG)]")
    print(f"  - False Positives (Clinician-Removed): Mean = {np.mean(fp_means):.2f} mG | Median = {np.median(fp_means):.2f} mG | 95th Percentile = {np.percentile(fp_means, 95):.2f} mG")
    print(f"  - True Positives (Clinician-Confirmed): Mean = {np.mean(tp_means):.2f} mG | Median = {np.median(tp_means):.2f} mG | 95th Percentile = {np.percentile(tp_means, 95):.2f} mG")
    
    print("\n[Peak Dynamic Motion per Event (mG)]")
    print(f"  - False Positives (Clinician-Removed): Mean = {np.mean(fp_maxes):.2f} mG | Median = {np.median(fp_maxes):.2f} mG | 95th Percentile = {np.percentile(fp_maxes, 95):.2f} mG")
    print(f"  - True Positives (Clinician-Confirmed): Mean = {np.mean(tp_maxes):.2f} mG | Median = {np.median(tp_maxes):.2f} mG | 95th Percentile = {np.percentile(tp_maxes, 95):.2f} mG")
    
if __name__ == '__main__':
    run_comparison()
