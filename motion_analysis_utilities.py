"""
motion_analysis_utilities.py

Core utilities for motion/accelerometer analysis.
Consolidates motion extraction logic from:
  - optimize_motion_thresholds.py
  - compare_tp_fp_motion.py
  - explore_afib_motion.py

Provides unified, reusable functions for:
  - Dataset loading and validation
  - Accelerometer data extraction
  - Motion feature computation
  - Statistical analysis helpers
"""

import os
import json
import numpy as np
import pandas as pd
from scipy.signal import welch
from pathlib import Path
from tqdm import tqdm
import warnings

warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

ACC_SCALE_FACTOR = 2048.0  # Raw accelerometer values to G units
ACC_SAMPLING_RATE = 5.0     # Hz (5 frames per second)
DC_REMOVAL_WINDOW = 5       # Sample rolling mean for DC component removal
MOTION_UNITS = 1000.0       # Convert G to mG

# ─────────────────────────────────────────────────────────────────────────────
# Dataset Loading
# ─────────────────────────────────────────────────────────────────────────────

def load_dataset(csv_path, data_dir):
    """
    Load dataset metadata from CSV and validate file existence.

    Args:
        csv_path (str): Path to summary.csv
        data_dir (str): Root data directory

    Returns:
        pd.DataFrame: Dataset metadata with validation status
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Dataset CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)
    print(f"✓ Loaded {len(df)} records from {os.path.basename(csv_path)}")

    # Normalize column names to lowercase for consistency
    df.columns = df.columns.str.lower().str.replace(' ', '_')

    # Validate that JSON files exist
    valid_records = []
    for idx, row in df.iterrows():
        json_file_rel = row.get('json_file', '')
        if json_file_rel:
            # Handle both Windows-style (backslash) and Unix-style (forward slash) paths
            json_file_rel = json_file_rel.replace('\\', '/')
            # Construct full path by joining with data_dir
            json_path = os.path.join(data_dir, json_file_rel)
            if os.path.exists(json_path):
                row['json_file'] = json_path  # Store full path for later use
                valid_records.append(row)

    df_valid = pd.DataFrame(valid_records).reset_index(drop=True)
    print(f"  → {len(df_valid)} records with valid JSON files ({100*len(df_valid)/len(df):.1f}%)")

    return df_valid


# ─────────────────────────────────────────────────────────────────────────────
# Accelerometer Data Extraction
# ─────────────────────────────────────────────────────────────────────────────

def extract_accelerometer_raw(json_path):
    """
    Extract raw accelerometer data (x, y, z) from JSON file.

    Args:
        json_path (str): Path to JSON ECG record

    Returns:
        np.ndarray: Shape (N, 3) with x, y, z acceleration values
               or None if data is missing/corrupted

    Raises:
        FileNotFoundError: If JSON file doesn't exist
    """
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"JSON file not found: {json_path}")

    try:
        with open(json_path, 'r') as f:
            records = json.load(f)

        # Handle both single record and list of records
        if isinstance(records, dict):
            records = [records]

        acc_all = []
        for record in records:
            if 'data' not in record or 'acc' not in record['data']:
                continue

            acc_data = record['data']['acc']
            for frame in acc_data:
                if isinstance(frame, dict) and 'x' in frame and 'y' in frame and 'z' in frame:
                    acc_all.append([float(frame['x']), float(frame['y']), float(frame['z'])])

        if len(acc_all) == 0:
            return None

        return np.array(acc_all, dtype=np.float32)

    except Exception as e:
        warnings.warn(f"Error loading {json_path}: {str(e)}")
        return None


def compute_vector_magnitude(acc_array, scale_factor=ACC_SCALE_FACTOR):
    """
    Compute vector magnitude (scalar motion) from 3-axis accelerometer data.
    Includes DC removal (gravity removal) using rolling mean.

    Args:
        acc_array (np.ndarray): Shape (N, 3) with x, y, z acceleration
        scale_factor (float): Scale factor to convert raw values to G units

    Returns:
        np.ndarray: Shape (N,) with dynamic motion in mG (milliG)
                   or None if input is invalid
    """
    if acc_array is None or len(acc_array) == 0:
        return None

    # Normalize to G units
    acc_g = acc_array / scale_factor

    # Compute vector magnitude
    vm = np.linalg.norm(acc_g, axis=1)

    # DC removal: subtract rolling mean to remove gravity component
    # This gives us dynamic acceleration (motion)
    dc_offset = np.convolve(vm, np.ones(DC_REMOVAL_WINDOW) / DC_REMOVAL_WINDOW, mode='same')
    dynamic_motion = vm - dc_offset

    # Ensure non-negative values (take absolute value of dynamic component)
    dynamic_motion = np.abs(dynamic_motion)

    # Convert to mG (milliG)
    dynamic_motion_mg = dynamic_motion * MOTION_UNITS

    return dynamic_motion_mg


# ─────────────────────────────────────────────────────────────────────────────
# Motion Feature Extraction
# ─────────────────────────────────────────────────────────────────────────────

def extract_motion_features(json_path, verbose=False):
    """
    Comprehensive motion feature extraction from a single JSON ECG record.

    Args:
        json_path (str): Path to JSON ECG record
        verbose (bool): Print debug information

    Returns:
        dict: Motion features including:
            - mean_motion: Mean dynamic motion (mG)
            - max_motion: Peak dynamic motion (mG)
            - std_motion: Standard deviation (mG)
            - median_motion: Median motion (mG)
            - peak_ratio: max_motion / mean_motion
            - zero_motion_pct: Percentage below 1 mG
            - dom_freq: Dominant frequency (Hz)
            - n_samples: Number of accelerometer samples
        or None if extraction fails
    """
    try:
        # Extract raw accelerometer data
        acc_raw = extract_accelerometer_raw(json_path)
        if acc_raw is None or len(acc_raw) < 2:
            return None

        # Compute dynamic motion
        motion = compute_vector_magnitude(acc_raw)
        if motion is None or len(motion) < 2:
            return None

        # Basic statistics
        mean_motion = float(np.mean(motion))
        max_motion = float(np.max(motion))
        std_motion = float(np.std(motion))
        median_motion = float(np.median(motion))

        # Ratios
        peak_ratio = max_motion / (mean_motion + 1e-6)  # Avoid division by zero

        # Zero-motion percentage (< 1 mG)
        zero_motion_pct = float(100.0 * np.mean(motion < 1.0))

        # Dominant frequency (via FFT)
        dom_freq = compute_dominant_frequency(motion, ACC_SAMPLING_RATE)

        features = {
            'mean_motion': mean_motion,
            'max_motion': max_motion,
            'std_motion': std_motion,
            'median_motion': median_motion,
            'peak_ratio': peak_ratio,
            'zero_motion_pct': zero_motion_pct,
            'dom_freq': dom_freq,
            'n_samples': len(motion)
        }

        if verbose:
            print(f"  Mean: {mean_motion:.2f} mG | Max: {max_motion:.2f} mG | "
                  f"Median: {median_motion:.2f} mG | Dom Freq: {dom_freq:.2f} Hz")

        return features

    except Exception as e:
        warnings.warn(f"Feature extraction failed for {json_path}: {str(e)}")
        return None


def compute_dominant_frequency(motion_signal, sampling_rate):
    """
    Compute dominant frequency of motion signal using FFT.

    Args:
        motion_signal (np.ndarray): 1D motion array
        sampling_rate (float): Sampling rate in Hz

    Returns:
        float: Dominant frequency in Hz
    """
    try:
        if len(motion_signal) < 4:
            return 0.0

        # Compute FFT
        fft_vals = np.abs(np.fft.fft(motion_signal - np.mean(motion_signal)))
        freq_bins = np.fft.fftfreq(len(motion_signal), 1/sampling_rate)

        # Find dominant frequency (excluding DC component)
        positive_freqs = freq_bins[:len(freq_bins)//2]
        positive_fft = fft_vals[:len(fft_vals)//2]

        # Exclude DC (index 0)
        if len(positive_freqs) > 1:
            dom_idx = np.argmax(positive_fft[1:]) + 1
            return float(positive_freqs[dom_idx])

        return 0.0

    except Exception:
        return 0.0


# ─────────────────────────────────────────────────────────────────────────────
# Batch Processing
# ─────────────────────────────────────────────────────────────────────────────

def extract_motion_features_batch(df_records, json_col='json_file',
                                   event_col='event_type', sample_size=None,
                                   stratified=True):
    """
    Extract motion features for a batch of records.

    Args:
        df_records (pd.DataFrame): Records with JSON file paths
        json_col (str): Column name for JSON file paths
        event_col (str): Column name for event type (for stratification)
        sample_size (int): Limit number of samples (None = use all)
        stratified (bool): Stratify sampling by event type

    Returns:
        pd.DataFrame: Motion features for each record
        list: Failed indices (for error tracking)
    """
    # Apply sampling if requested
    if sample_size is not None and len(df_records) > sample_size:
        if stratified and event_col in df_records.columns:
            # Stratified sampling by event type
            df_sampled = df_records.groupby(event_col, group_keys=False).apply(
                lambda x: x.sample(min(len(x), max(1, sample_size // x[event_col].nunique())),
                                   random_state=42)
            ).reset_index(drop=True)
            # Resample if needed to reach exact sample_size
            if len(df_sampled) > sample_size:
                df_sampled = df_sampled.sample(sample_size, random_state=42)
            df_to_process = df_sampled
        else:
            df_to_process = df_records.sample(sample_size, random_state=42).reset_index(drop=True)
    else:
        df_to_process = df_records

    results = []
    failed_indices = []

    print(f"Extracting motion features from {len(df_to_process)} records...")
    for idx, row in tqdm(df_to_process.iterrows(), total=len(df_to_process)):
        json_path = row.get(json_col, '')
        if not json_path:
            failed_indices.append(idx)
            continue

        features = extract_motion_features(json_path)
        if features is None:
            failed_indices.append(idx)
            continue

        # Add metadata
        features['record_index'] = idx
        features['json_path'] = json_path
        if event_col in row.index:
            features['event_type'] = row[event_col]

        results.append(features)

    df_features = pd.DataFrame(results)

    print(f"✓ Successfully extracted features from {len(df_features)} records")
    if failed_indices:
        print(f"⚠ Failed to extract from {len(failed_indices)} records")

    return df_features, failed_indices


# ─────────────────────────────────────────────────────────────────────────────
# Statistical Helpers
# ─────────────────────────────────────────────────────────────────────────────

def compute_percentiles(data, percentiles=(25, 50, 75, 90, 95, 99)):
    """
    Compute percentiles for a data array.

    Args:
        data (np.ndarray or list): 1D array of values
        percentiles (tuple): Percentiles to compute

    Returns:
        dict: Percentile names and values
    """
    result = {}
    for p in percentiles:
        result[f'p{p}'] = float(np.percentile(data, p))
    return result


def compute_distribution_stats(data):
    """
    Compute comprehensive distribution statistics.

    Args:
        data (np.ndarray or list): 1D array of values

    Returns:
        dict: Comprehensive statistics
    """
    data = np.asarray(data)
    data = data[~np.isnan(data)]  # Remove NaN values

    if len(data) == 0:
        return None

    stats = {
        'count': len(data),
        'mean': float(np.mean(data)),
        'median': float(np.median(data)),
        'std': float(np.std(data)),
        'min': float(np.min(data)),
        'max': float(np.max(data)),
        'q25': float(np.percentile(data, 25)),
        'q75': float(np.percentile(data, 75)),
        'iqr': float(np.percentile(data, 75) - np.percentile(data, 25)),
    }

    # Add additional percentiles
    for p in [10, 50, 90, 95, 99]:
        stats[f'p{p}'] = float(np.percentile(data, p))

    return stats


# ─────────────────────────────────────────────────────────────────────────────
# Validation & Verification
# ─────────────────────────────────────────────────────────────────────────────

def verify_motion_features(features_df, min_samples=10):
    """
    Verify motion features for completeness and validity.

    Args:
        features_df (pd.DataFrame): Motion features
        min_samples (int): Minimum samples required for valid analysis

    Returns:
        dict: Validation report
    """
    report = {
        'total_records': len(features_df),
        'complete_features': 0,
        'missing_values': {},
        'event_types': {},
        'data_quality': 'PASS'
    }

    # Check for missing values
    for col in features_df.columns:
        missing = features_df[col].isna().sum()
        if missing > 0:
            report['missing_values'][col] = missing

    # Count complete records
    report['complete_features'] = len(features_df.dropna())

    # Event type distribution
    if 'event_type' in features_df.columns:
        report['event_types'] = dict(features_df['event_type'].value_counts())

    # Overall quality
    if report['missing_values']:
        report['data_quality'] = 'WARNING'

    return report


if __name__ == "__main__":
    print("Motion Analysis Utilities Module")
    print(f"  ACC Scale Factor: {ACC_SCALE_FACTOR}")
    print(f"  ACC Sampling Rate: {ACC_SAMPLING_RATE} Hz")
    print(f"  DC Removal Window: {DC_REMOVAL_WINDOW} samples")
    print(f"  Motion Units: mG (milliG = {MOTION_UNITS})")
