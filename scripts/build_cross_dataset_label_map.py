"""Generate res/cross_dataset_label_map.csv — the cross-dataset 150-head mapping.

One row per Founder head (idx 0..149). Per-dataset columns hold the native label
text (or empty), so the CSV is a single source of truth for "what does each
dataset call this head?". Run from repo root:

    python3 -m scripts.build_cross_dataset_label_map

Regenerate whenever any of these change:
  - tasks.txt (and TASKS_SHA256)
  - FZARK_LABEL_MAP / MITDB_*_MAP / PTBXL_ACTIVE_CLASSES in label_config.py
  - scripts/cinc2015/cinc2015_label_map.py ALARM_TO_HEAD
  - scripts/mimic/mimic_label_map.py HEAD_SPEC
"""
from __future__ import annotations
import csv
import os
from collections import defaultdict

from label_config import (
    load_tasks,
    FZARK_LABEL_MAP,
    MITDB_BEAT_MAP, MITDB_RHYTHM_MAP, MITDB_DEFAULT_NORMAL,
    PTBXL_ACTIVE_CLASSES,
    DETECTION_SCOPE,
)
from scripts.cinc2015.cinc2015_label_map import ALARM_TO_HEAD
from scripts.mimic.mimic_label_map import HEAD_SPEC as MIMIC_SPEC

OUT_PATH = "res/cross_dataset_label_map.csv"


def _invert_fzark() -> dict[int, list[str]]:
    out: dict[int, list[str]] = defaultdict(list)
    for ev, idx in FZARK_LABEL_MAP.items():
        out[idx].append(ev)
    return out


def _invert_mitdb() -> dict[int, list[str]]:
    out: dict[int, list[str]] = defaultdict(list)
    for sym, idx in MITDB_BEAT_MAP.items():
        out[idx].append(f"beat:{sym}")
    for tok, idx in MITDB_RHYTHM_MAP.items():
        out[idx].append(f"rhythm:{tok}")
    for idx in MITDB_DEFAULT_NORMAL:
        out[idx].append("default-normal")
    return out


def _invert_cinc2015() -> dict[int, list[str]]:
    out: dict[int, list[str]] = defaultdict(list)
    for tok, idx in ALARM_TO_HEAD:
        if idx is not None:
            out[idx].append(f"alarm:{tok}")
    return out


def _invert_mimic() -> dict[int, list[str]]:
    return {idx: list(kws) for idx, kws in MIMIC_SPEC}


# Challenge 2017 is not in label_config (no committed mapper). Hard-coded here
# from the trivial 4-class routing documented in GLOBAL_LABEL_MAP.md §7.2.
CHALLENGE2017: dict[int, str] = {
    2: "N (normal)",
    5: "A (atrial fibrillation)",
}


def main() -> None:
    tasks = load_tasks()
    fz = _invert_fzark()
    mi = _invert_mitdb()
    ci = _invert_cinc2015()
    mm = _invert_mimic()
    ptbxl = PTBXL_ACTIVE_CLASSES
    scope = set(DETECTION_SCOPE)

    rows: list[list[str]] = []
    header = [
        "idx",
        "head_name",
        "in_scope",
        "challenge2017",
        "cinc2015",
        "ecg_tp_fzark",
        "ecg_fp_fzark",
        "mimic",
        "mitdb",
        "move",
        "ptbxl_active",
    ]
    rows.append(header)

    for idx, name in enumerate(tasks):
        # ecg_fp uses the same FZARK ontology as ecg_tp — same event names,
        # opposite verdict semantic (hard negative). The mapping is identical.
        fzark_events = "; ".join(sorted(fz.get(idx, [])))
        rows.append([
            str(idx),
            name,
            "yes" if idx in scope else "",
            CHALLENGE2017.get(idx, ""),
            "; ".join(ci.get(idx, [])),
            fzark_events,
            fzark_events,
            "; ".join(mm.get(idx, [])),
            "; ".join(mi.get(idx, [])),
            "",  # MOVE has no native rhythm labels
            "yes" if idx in ptbxl else "",
        ])

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    with open(OUT_PATH, "w", newline="") as f:
        csv.writer(f).writerows(rows)

    touched = sum(
        1 for r in rows[1:]
        if any(r[i] for i in (3, 4, 5, 6, 7, 8, 10))
    )
    print(f"wrote {OUT_PATH}: {len(rows)-1} rows ({touched} with ≥1 dataset mapping)")


if __name__ == "__main__":
    main()
