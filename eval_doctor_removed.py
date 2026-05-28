import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from device_utils import resolve_device
from preprocessing import ECGPreprocessor, TARGET_LEN
from checkpoints import load_ecgfounder
from label_config import FZARK_LABEL_MAP as LABEL_MAP

# Config
DATA_DIR      = "./data/ecg_fp_doctor removed1"
VAL_CSV       = "./data/ecg_fp_doctor removed1/summary.csv"
CKPT_BASE     = "./checkpoint/1_lead_ECGFounder.pth"
SAVED_DIR     = "./res/doctor_removed_eval"
os.makedirs(SAVED_DIR, exist_ok=True)

BATCH_SIZE    = 64

# LABEL_MAP is imported from `label_config` (v3 single-index ontology).

class DoctorRemovedDataset(Dataset):
    def __init__(self, df, data_dir):
        self.df = df
        self.data_dir = data_dir
        self.prep = ECGPreprocessor.for_fzark()

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        json_rel = row['JSON File'].replace('\\', '/')
        event = row['Event Type']
        json_path = os.path.join(self.data_dir, json_rel)

        # All of these are human-removed false positives, so actual ground truth is negative (0)
        label = np.zeros(150, dtype=np.float32)

        try:
            tensor = self.prep.from_fzark_json(json_path)
        except Exception:
            tensor = torch.zeros(1, TARGET_LEN)

        return tensor, label, event

def run_eval():
    device = resolve_device()        # project default: MPS → CUDA → CPU
    print(f"Using device: {device}")
    
    # Load summary and filter for relevant events
    df = pd.read_csv(VAL_CSV)
    df = df[df['Event Type'].isin(LABEL_MAP.keys())]
    
    # To run quickly, let's take all Atrial Fibrillation events (1793) and a sample of 200 from each other class
    print("Sampling dataset...")
    afib_df = df[df['Event Type'] == 'Atrial Fibrillation']
    other_dfs = []
    for et in LABEL_MAP.keys():
        if et == 'Atrial Fibrillation':
            continue
        sub_df = df[df['Event Type'] == et]
        if len(sub_df) > 200:
            other_dfs.append(sub_df.sample(200, random_state=42))
        else:
            other_dfs.append(sub_df)
    
    sampled_df = pd.concat([afib_df] + other_dfs).reset_index(drop=True)
    print(f"Total sampled evaluation records: {len(sampled_df)}")
    print(sampled_df['Event Type'].value_counts())
    
    dataset = DoctorRemovedDataset(sampled_df, DATA_DIR)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    # Evaluate the single-lead base model
    models = {
        "Base (1_lead_ECGFounder)": CKPT_BASE,
    }

    results = {}
    for name, path in models.items():
        print(f"\nEvaluating model: {name}")
        model = load_ecgfounder(device, ckpt_path=path)
        
        all_preds = []
        all_events = []
        
        with torch.no_grad():
            for x, _, events in tqdm(loader, desc=f"Inference - {name}"):
                probs = torch.sigmoid(model(x.to(device))).cpu().numpy()
                all_preds.append(probs)
                all_events.extend(events)
                
        preds = np.concatenate(all_preds)
        results[name] = (preds, all_events)
        
    # Analyze and compare
    # Since these are all false positives, we want the prediction for the corresponding class to be as close to 0 as possible
    # We will print the average predicted probability and FP rate (at 0.5 threshold) for each model on each class
    
    print("\n" + "="*80)
    print("DOCTOR-REMOVED FALSE POSITIVES EVALUATION RESULTS")
    print("="*80)
    
    summary_data = []
    
    for name, (preds, events) in results.items():
        print(f"\nModel: {name}")
        print("-" * 50)
        for event, cls_idx in LABEL_MAP.items():
            # Get predictions for this class when the event type was this class
            indices = [i for i, ev in enumerate(events) if ev == event]
            if len(indices) == 0:
                continue
            
            # Since these are false positives, we want this specific class score to be low
            class_preds = preds[indices, cls_idx]
            avg_score = np.mean(class_preds)
            fp_rate_05 = np.mean(class_preds >= 0.5)
            fp_rate_07 = np.mean(class_preds >= 0.7)
            fp_rate_08 = np.mean(class_preds >= 0.8)
            fp_rate_01 = np.mean(class_preds >= 0.1) # at a lower clinical warning threshold
            
            print(f"  {event:<32} (n={len(indices)}): Avg Score: {avg_score:.4f} | FP Rate (@0.5): {fp_rate_05*100:.2f}% | FP Rate (@0.7): {fp_rate_07*100:.2f}% | FP Rate (@0.8): {fp_rate_08*100:.2f}% | FP Rate (@0.1): {fp_rate_01*100:.2f}%")
            
            summary_data.append({
                "Model": name,
                "Event Type": event,
                "Class Index": cls_idx,
                "Sample Count": len(indices),
                "Average Score": avg_score,
                "FP Rate (@0.5)": fp_rate_05,
                "FP Rate (@0.7)": fp_rate_07,
                "FP Rate (@0.8)": fp_rate_08,
                "FP Rate (@0.1)": fp_rate_01
            })
            
    # Save results
    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv(os.path.join(SAVED_DIR, "doctor_removed_eval_comparison.csv"), index=False)
    print(f"\nSaved CSV results to {SAVED_DIR}/doctor_removed_eval_comparison.csv")

if __name__ == '__main__':
    run_eval()
