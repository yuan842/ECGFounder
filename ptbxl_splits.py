"""PTB-XL train/val/test split — the recommended 1-8 / 9 / 10 fold convention.

Single source of truth: csv/ptbxl_fold_split.csv (frozen manifest, committed) —
columns: ecg_id, patient_id, strat_fold, split. Folds are PTB-XL's native
patient-stratified 10-fold; the convention maps:

    folds 1-8 → train   (17,418 records)   model fitting
    fold  9   → val     ( 2,183 records)   model selection / threshold tuning
    fold  10  → test    ( 2,198 records)   reported once, never tuned on

The manifest decouples the split from the (gitignored, non-redistributable) raw
PTB-XL database, so the split is reproducible on a fresh clone.
"""
from __future__ import annotations
import functools, os
import numpy as np
import pandas as pd

SPLIT_CSV = os.path.join(os.path.dirname(__file__), "csv", "ptbxl_fold_split.csv")
FOLD_TO_SPLIT = {**{f: "train" for f in range(1, 9)}, 9: "val", 10: "test"}
SPLITS = ("train", "val", "test")


@functools.lru_cache(maxsize=1)
def _table() -> pd.DataFrame:
    df = pd.read_csv(SPLIT_CSV)
    df["ecg_id"] = df["ecg_id"].astype(int)
    return df


@functools.lru_cache(maxsize=1)
def _split_map() -> dict:
    df = _table()
    return dict(zip(df["ecg_id"], df["split"]))


def split_of(ecg_id: int) -> str:
    """Return 'train' | 'val' | 'test' for an ecg_id (KeyError if unknown)."""
    return _split_map()[int(ecg_id)]


def mask_for(ecg_ids, split: str) -> np.ndarray:
    """Boolean mask over `ecg_ids` selecting rows belonging to `split`."""
    if split not in SPLITS:
        raise ValueError(f"split must be one of {SPLITS}, got {split!r}")
    m = _split_map()
    return np.array([m.get(int(i)) == split for i in ecg_ids], dtype=bool)


def counts() -> dict:
    return _table()["split"].value_counts().reindex(SPLITS).to_dict()


def assert_no_leakage() -> None:
    """Hard check: no patient_id spans more than one split."""
    df = _table()
    spanning = df.groupby("patient_id")["split"].nunique()
    bad = int((spanning > 1).sum())
    if bad:
        raise AssertionError(f"patient-level leakage: {bad} patients span >1 split")


if __name__ == "__main__":
    assert_no_leakage()
    print("PTB-XL split (folds 1-8 / 9 / 10):", counts(), "| 0 patient leakage ✓")
