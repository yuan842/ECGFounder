import os
import glob
import numpy as np
import pandas as pd
import wfdb
import torch
from sklearn.model_selection import train_test_split
from preprocessing import ECGPreprocessor

def main():
    print("==================================================")
    print("MIT-BIH Arrhythmia Dataset: Standardization & Split")
    print("==================================================")

    mitdb_dir = "./data/mitdb"
    out_dir = "./res/mitdb_standardized"
    os.makedirs(out_dir, exist_ok=True)
    
    # 1. Search for available records in mitdb directory
    hea_files = glob.glob(os.path.join(mitdb_dir, "*.hea"))
    if not hea_files:
        print(f"Error: No WFDB records found in '{mitdb_dir}'.")
        return
        
    records = sorted([os.path.splitext(os.path.basename(f))[0] for f in hea_files])
    print(f"Found {len(records)} raw MIT-BIH records.")

    # 2. Recording-Level 80/20 Split (Patient-Level Split)
    # Using a fixed random state for reproducibility
    train_records, val_records = train_test_split(records, test_size=0.20, random_state=42)
    print(f"\nRecording-Level Split (80% Train, 20% Val):")
    print(f"  - Training Records ({len(train_records)}): {sorted(train_records)}")
    print(f"  - Validation Records ({len(val_records)}): {sorted(val_records)}")

    # Create directory structure
    train_dir = os.path.join(out_dir, "train")
    val_dir = os.path.join(out_dir, "val")
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)

    # 3. Initialize unified preprocessor for MIT-BIH (60 Hz powerline, MLII).
    prep = ECGPreprocessor.for_mitdb()

    metadata = []
    total_segments = 0

    print("\nStarting pipeline processing across all records...")
    for idx, record in enumerate(records):
        split = "train" if record in train_records else "val"
        target_dir = train_dir if split == "train" else "val_dir"
        target_path = train_dir if split == "train" else val_dir
        
        record_path = os.path.join(mitdb_dir, record)
        
        try:
            signals, fields = wfdb.rdsamp(record_path)
            fs_in = fields['fs']
            source_leads = fields['sig_name']
            
            # Transpose to (channels, sequence_length)
            signals = np.transpose(signals, (1, 0))
            
            segment_len_seconds = 10
            segment_samples_in = int(segment_len_seconds * fs_in)
            num_segments = signals.shape[1] // segment_samples_in
            
            print(f"[{idx+1}/{len(records)}] Record {record} ({split}) -> Processing {num_segments} segments...")
            
            for i in range(num_segments):
                start = i * segment_samples_in
                end = start + segment_samples_in
                raw_segment = signals[:, start:end]
                
                # Run the unified preprocessing pipeline (winsorized z-score,
                # 60 Hz notch baked into for_mitdb()).
                std_tensor = prep.process(
                    signal=raw_segment,
                    fs_in=fs_in,
                    source_leads=source_leads,
                )
                
                # Save standardized tensor file
                segment_name = f"{record}_seg_{i:03d}.pth"
                file_path = os.path.join(target_path, segment_name)
                torch.save(std_tensor, file_path)
                
                # Append metadata info
                metadata.append({
                    "segment_id": f"{record}_seg_{i:03d}",
                    "parent_record": record,
                    "segment_index": i,
                    "split": split,
                    "file_path": file_path
                })
                total_segments += 1
                
        except Exception as e:
            print(f"  Warning: Failed to process record {record}: {e}")

    # 4. Save metadata summary index CSV
    df_meta = pd.DataFrame(metadata)
    meta_csv_path = os.path.join(out_dir, "mitdb_split_metadata.csv")
    df_meta.to_csv(meta_csv_path, index=False)
    
    print("\n==================================================")
    print("MIT-BIH Processing & Split Complete!")
    print(f"Total Standardized Tensors Saved: {total_segments}")
    print(f"  - Training set segments:   {len(df_meta[df_meta['split'] == 'train'])}")
    print(f"  - Validation set segments: {len(df_meta[df_meta['split'] == 'val'])}")
    print(f"Master metadata catalog saved at: '{meta_csv_path}'")
    print("==================================================")

if __name__ == "__main__":
    main()
