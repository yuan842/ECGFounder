import os
import json
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d

DATA_DIR = "./data/ecg_fp_doctor removed1"
VAL_CSV  = "./data/ecg_fp_doctor removed1/summary.csv"
FS_IN    = 128

def get_afib_motion_stats():
    df = pd.read_csv(VAL_CSV)
    afib_df = df[df['Event Type'] == 'Atrial Fibrillation']
    print(f"Total AFib False Positive records found: {len(afib_df)}")
    
    # We will sample up to 300 records to get a robust distribution quickly
    sample_df = afib_df.sample(min(300, len(afib_df)), random_state=42)
    
    means = []
    maxes = []
    
    for idx, row in sample_df.iterrows():
        json_rel = row['JSON File'].replace('\\', '/')
        json_path = os.path.join(DATA_DIR, json_rel)
        
        try:
            with open(json_path) as f:
                records = json.load(f)
            
            raw_acc = []
            for r in records:
                data = r['data']
                if 'acc' in data:
                    for frame in data['acc']:
                        raw_acc.append([frame.get('x', 0), frame.get('y', 0), frame.get('z', 0)])
            
            if len(raw_acc) == 0:
                continue
                
            acc_arr = np.array(raw_acc, dtype=np.float32) / 2048.0
            
            # Compute Vector Magnitude (VM)
            vm = np.sqrt(acc_arr[:, 0]**2 + acc_arr[:, 1]**2 + acc_arr[:, 2]**2)
            # Simple DC removal using rolling mean
            vm_dc = pd.Series(vm).rolling(window=5, min_periods=1, center=True).mean().values # ACC is 5Hz, so 1s window = 5 samples
            vm_dynamic = np.abs(vm - vm_dc) * 1000.0 # Convert to mG
            
            means.append(np.mean(vm_dynamic))
            maxes.append(np.max(vm_dynamic))
        except Exception as e:
            continue
            
    means = np.array(means)
    maxes = np.array(maxes)
    
    print("\n" + "="*50)
    print("ATRIAL FIBRILLATION DYNAMIC ACCELERATION DISTRIBUTION")
    print("="*50)
    
    stats_df = pd.DataFrame({
        "Metric": ["Mean (Average Motion per Event)", "Max (Peak Motion per Event)"],
        "Count": [len(means), len(maxes)],
        "Mean (mG)": [np.mean(means), np.mean(maxes)],
        "Std Dev (mG)": [np.std(means), np.std(maxes)],
        "Min (mG)": [np.min(means), np.min(maxes)],
        "25% (mG)": [np.percentile(means, 25), np.percentile(maxes, 25)],
        "50% / Median (mG)": [np.percentile(means, 50), np.percentile(maxes, 50)],
        "75% (mG)": [np.percentile(means, 75), np.percentile(maxes, 75)],
        "90% (mG)": [np.percentile(means, 90), np.percentile(maxes, 90)],
        "95% (mG)": [np.percentile(means, 95), np.percentile(maxes, 95)],
        "Max (mG)": [np.max(means), np.max(maxes)]
    })
    
    print(stats_df.to_string(index=False))

if __name__ == '__main__':
    get_afib_motion_stats()
