import os
import glob
import numpy as np
import pandas as pd
import wfdb
import torch
import matplotlib.pyplot as plt
from preprocessing import ECGPreprocessor

def main():
    print("--------------------------------------------------")
    print("MIT-BIH Arrhythmia Database Standardization Utility")
    print("--------------------------------------------------")

    mitdb_dir = "./data/mitdb"
    out_dir = "./res/mitdb_standardized"
    os.makedirs(out_dir, exist_ok=True)
    
    # 1. Search for available records in mitdb directory
    hea_files = glob.glob(os.path.join(mitdb_dir, "*.hea"))
    if not hea_files:
        print(f"Error: No WFDB records found in '{mitdb_dir}'.")
        print("Please make sure the MIT-BIH dataset download has completed successfully.")
        return
        
    records = sorted([os.path.splitext(os.path.basename(f))[0] for f in hea_files])
    print(f"Found {len(records)} MIT-BIH records: {records}")

    # 2. Configure unified preprocessor for MIT-BIH (60 Hz powerline, MLII lead).
    # Target length is 5000 samples, target fs is 500 Hz (defaults).
    prep = ECGPreprocessor.for_mitdb()

    # 3. Process the first record (e.g., '100') as a primary showcase demo
    demo_record = records[0]
    record_path = os.path.join(mitdb_dir, demo_record)
    
    print(f"\nProcessing demo record: {demo_record}...")
    try:
        # Load raw signal and metadata fields
        signals, fields = wfdb.rdsamp(record_path)
        fs_in = fields['fs']
        source_leads = fields['sig_name']
        print(f"Record {demo_record} loaded successfully:")
        print(f"  - Sampling frequency: {fs_in} Hz")
        print(f"  - Total samples: {signals.shape[0]}")
        print(f"  - Lead names: {source_leads}")
        
        # Transpose to (channels, sequence_length)
        signals = np.transpose(signals, (1, 0))
        
        # Segment 30-minute record into 10-second non-overlapping segments
        # 10 seconds at 360 Hz is 3600 samples
        segment_len_seconds = 10
        segment_samples_in = int(segment_len_seconds * fs_in)
        num_segments = signals.shape[1] // segment_samples_in
        
        print(f"Segmenting 30-minute recording into {num_segments} non-overlapping 10-second windows...")
        
        # Create output subfolder for this record
        record_out_dir = os.path.join(out_dir, demo_record)
        os.makedirs(record_out_dir, exist_ok=True)
        
        # Denoise and standardize segments
        standardized_count = 0
        
        # We will extract a middle segment for our beautiful comparison plot
        plot_segment_idx = num_segments // 2
        raw_demo_segment = None
        std_demo_segment = None
        
        for i in range(num_segments):
            start = i * segment_samples_in
            end = start + segment_samples_in
            raw_segment = signals[:, start:end]
            
            # Run the unified preprocessing pipeline
            std_tensor = prep.process(
                signal=raw_segment,
                fs_in=fs_in,
                source_leads=source_leads,
            )
            
            # Save standardized segment tensor
            segment_filename = os.path.join(record_out_dir, f"segment_{i:03d}.pth")
            torch.save(std_tensor, segment_filename)
            standardized_count += 1
            
            if i == plot_segment_idx:
                raw_demo_segment = raw_segment
                std_demo_segment = std_tensor.numpy()
                
        print(f"Successfully standardized and saved {standardized_count} segments under '{record_out_dir}/'.")
        
        # 4. Generate high-fidelity Raw vs. Standardized comparison plot
        if raw_demo_segment is not None and std_demo_segment is not None:
            print("\nGenerating visual raw vs. standardized comparison plot...")
            
            # Find which channel in raw_demo_segment corresponds to Lead II / MLII
            lead_idx = 0
            for j, s_lead in enumerate(source_leads):
                if 'ii' in s_lead.lower():
                    lead_idx = j
                    break
            
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=False)
            
            # Plot Raw Waveform (10 seconds @ 360 Hz = 3600 samples)
            time_raw = np.linspace(0, segment_len_seconds, raw_demo_segment.shape[1])
            ax1.plot(time_raw, raw_demo_segment[lead_idx, :], color='#e056fd', linewidth=1.2, label=f"Raw {source_leads[lead_idx]} (360 Hz)")
            ax1.set_title(f"MIT-BIH Record {demo_record} (Segment {plot_segment_idx}): Raw Signal", fontsize=12, fontweight='bold')
            ax1.set_ylabel("Amplitude (mV)")
            ax1.grid(True, linestyle='--', alpha=0.5)
            ax1.legend(loc="upper right")
            
            # Plot Standardized Waveform (10 seconds @ 500 Hz = 5000 samples)
            time_std = np.linspace(0, segment_len_seconds, std_demo_segment.shape[1])
            ax2.plot(time_std, std_demo_segment[0, :], color='#0984e3', linewidth=1.2, label="Standardized Lead II (500 Hz)")
            ax2.set_title("Standardized Waveform (Notched 60Hz, Bandpassed 0.67-40Hz, Median Baseline Removed, Z-scored)", fontsize=12, fontweight='bold')
            ax2.set_xlabel("Time (seconds)")
            ax2.set_ylabel("Standardized Amplitude (z)")
            ax2.grid(True, linestyle='--', alpha=0.5)
            ax2.legend(loc="upper right")
            
            plt.tight_layout()
            plot_path = os.path.join(out_dir, f"record_{demo_record}_comparison.png")
            plt.savefig(plot_path, dpi=150)
            plt.close()
            print(f"Comparison plot successfully saved at '{plot_path}'.")
            
        print("\nMIT-BIH Standardization Task Complete! All systems verified.")
        print("--------------------------------------------------")
            
    except Exception as e:
        print(f"Error occurred during record standardization: {e}")

if __name__ == "__main__":
    main()
