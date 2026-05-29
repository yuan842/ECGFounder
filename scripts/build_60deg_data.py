"""Build train_60deg.npz + val_60deg_split.npz from PTB-XL records500.

60° in fuzzylead2's hexaxial reference coincides with the standard 12-lead
Lead II (derivelead2.md, §3.B: "At +60°, the lead perfectly overlaps with
Lead II"). So we extract Lead II from the raw WFDB records for the same
ecg_ids that appear in the existing train_45deg.npz / val_45deg.npz files
and save them with matching shape/dtype.

Output:
  data/fuzzylead2/ptbxl/train_60deg.npz   ← matches train_45deg (n=19,601)
  data/fuzzylead2/ptbxl/val_60deg_split.npz ← matches val_45deg (n=2,198)

The existing val_60deg.npz is left untouched — it contains a different
record set (4,376) and is not used in our 4-angle training pipeline.
"""
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import wfdb
from tqdm import tqdm

PTBXL_ROOT  = "data/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3"
FUZZY_DIR   = "data/fuzzylead2/ptbxl"
TARGET_LEN  = 5000   # 10 seconds @ 500 Hz


def ecg_id_to_record_path(eid: int) -> str:
    folder = (eid // 1000) * 1000
    return f"{PTBXL_ROOT}/records500/{folder:05d}/{eid:05d}_hr"


def extract_lead_ii(eid: int) -> np.ndarray:
    """Return Lead II (idx 1 in PTB-XL signal order) at 500 Hz, shape (5000,)."""
    rec = wfdb.rdrecord(ecg_id_to_record_path(int(eid)))
    assert rec.fs == 500, f"unexpected fs={rec.fs} on ecg_id={eid}"
    assert rec.sig_name[1] == 'II', f"unexpected lead order {rec.sig_name[:3]} on ecg_id={eid}"
    lead_ii = rec.p_signal[:, 1].astype(np.float32)
    # Crop or pad to TARGET_LEN
    if lead_ii.shape[0] >= TARGET_LEN:
        return lead_ii[:TARGET_LEN]
    out = np.zeros(TARGET_LEN, dtype=np.float32)
    out[:lead_ii.shape[0]] = lead_ii
    return out


def build(split_name: str, src_npz: str, out_npz: str):
    src = np.load(src_npz)
    n = src['ecg_ids'].shape[0]
    print(f"\nBuilding {split_name} (n={n}) → {out_npz}")
    ecg_out = np.zeros((n, 1, TARGET_LEN), dtype=np.float32)
    for i in tqdm(range(n), desc=f"  {split_name}"):
        eid = int(src['ecg_ids'][i])
        ecg_out[i, 0, :] = extract_lead_ii(eid)
    payload = dict(ecg=ecg_out, ecg_ids=src['ecg_ids'])
    # Carry labels from the reference angle file too — for compatibility with
    # the existing FuzzylNpzDataset loader. We'll re-label with full PTB-XL
    # labels at train time.
    if 'labels' in src:
        payload['labels'] = src['labels']
    np.savez(out_npz, **payload)
    sz = os.path.getsize(out_npz) / 1e6
    print(f"  wrote {out_npz}  ({sz:.1f} MB)")


if __name__ == '__main__':
    build('train_60', f"{FUZZY_DIR}/train_45deg.npz",  f"{FUZZY_DIR}/train_60deg.npz")
    build('val_60',   f"{FUZZY_DIR}/val_45deg.npz",    f"{FUZZY_DIR}/val_60deg_split.npz")
    print("\nDone.")
