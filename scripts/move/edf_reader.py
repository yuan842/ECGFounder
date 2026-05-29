"""Dependency-free EDF / EDF+ reader (header + signals + annotations).

Pure stdlib + numpy — avoids the pyedflib dependency. Sufficient for the
MOVE dataset (EDF+C, continuous). Handles:
  - fixed 256-byte header + per-signal header
  - int16 little-endian data records → physical units via the EDF affine map
  - the "EDF Annotations" channel → list of (onset_s, label) TALs

Not a general EDF library — no support for EDF+D discontinuous records beyond
detecting them, no BDF, no logical/physical edge cases outside MOVE's profile.
"""
from __future__ import annotations
import re
from dataclasses import dataclass

import numpy as np

ANNOT_LABEL = "EDF Annotations"


@dataclass
class EdfHeader:
    start_seconds: float          # seconds since midnight from the starttime field
    n_records: int
    record_dur: float
    ns: int
    labels: list[str]
    phys_dim: list[str]
    phys_min: list[float]
    phys_max: list[float]
    dig_min: list[float]
    dig_max: list[float]
    n_samps: list[int]            # samples per record, per signal
    nbytes_hdr: int
    reserved: str                 # 'EDF+C' / 'EDF+D' / ''

    @property
    def total_duration_s(self) -> float:
        return self.n_records * self.record_dur

    def fs(self, sig_index: int) -> float:
        return self.n_samps[sig_index] / self.record_dur


def _starttime_to_seconds(starttime: str) -> float:
    """'HH.MM.SS' → seconds since midnight."""
    hh, mm, ss = starttime.strip().split(".")
    return int(hh) * 3600 + int(mm) * 60 + int(ss)


def read_header(path: str) -> EdfHeader:
    with open(path, "rb") as f:
        f.read(8)                                   # version
        f.read(80)                                  # patient
        f.read(80)                                  # recording
        f.read(8)                                   # startdate (placeholder in MOVE)
        starttime = f.read(8).decode("latin-1")
        nbytes_hdr = int(f.read(8))
        reserved = f.read(44).decode("latin-1").strip()
        n_records = int(f.read(8))
        record_dur = float(f.read(8))
        ns = int(f.read(4))
        labels   = [f.read(16).decode("latin-1").strip() for _ in range(ns)]
        [f.read(80) for _ in range(ns)]             # transducer
        phys_dim = [f.read(8).decode("latin-1").strip() for _ in range(ns)]
        phys_min = [float(f.read(8)) for _ in range(ns)]
        phys_max = [float(f.read(8)) for _ in range(ns)]
        dig_min  = [float(f.read(8)) for _ in range(ns)]
        dig_max  = [float(f.read(8)) for _ in range(ns)]
        [f.read(80) for _ in range(ns)]             # prefilter
        n_samps  = [int(f.read(8)) for _ in range(ns)]
        f.read(32 * ns)                             # reserved per-signal
    return EdfHeader(
        start_seconds=_starttime_to_seconds(starttime),
        n_records=n_records, record_dur=record_dur, ns=ns,
        labels=labels, phys_dim=phys_dim,
        phys_min=phys_min, phys_max=phys_max, dig_min=dig_min, dig_max=dig_max,
        n_samps=n_samps, nbytes_hdr=nbytes_hdr, reserved=reserved,
    )


def _read_data_block(path: str, hdr: EdfHeader) -> np.ndarray:
    """Return the raw int16 data as (n_records, record_len) array."""
    record_len = sum(hdr.n_samps)
    with open(path, "rb") as f:
        f.seek(hdr.nbytes_hdr)
        raw = np.frombuffer(f.read(), dtype="<i2")
    expected = hdr.n_records * record_len
    if raw.size < expected:                          # tolerate a short final record
        raw = raw[: (raw.size // record_len) * record_len]
    return raw[: (raw.size // record_len) * record_len].reshape(-1, record_len)


def read_signal(path: str, hdr: EdfHeader, label: str) -> tuple[np.ndarray, float]:
    """Return (physical-unit 1-D signal, fs) for a named non-annotation channel."""
    if label not in hdr.labels:
        raise KeyError(f"{label!r} not in {path}: have {hdr.labels}")
    s = hdr.labels.index(label)
    block = _read_data_block(path, hdr)
    off = sum(hdr.n_samps[:s])
    dig = block[:, off: off + hdr.n_samps[s]].reshape(-1).astype(np.float64)
    # EDF affine: phys = (dig - dig_min) * gain + phys_min
    span_dig = (hdr.dig_max[s] - hdr.dig_min[s]) or 1.0
    gain = (hdr.phys_max[s] - hdr.phys_min[s]) / span_dig
    phys = (dig - hdr.dig_min[s]) * gain + hdr.phys_min[s]
    return phys.astype(np.float32), hdr.fs(s)


def read_annotations(path: str, hdr: EdfHeader) -> list[tuple[float, str]]:
    """Parse the EDF+ annotation channel → list of (onset_s, label).

    Onset is seconds relative to the recording start (the EDF+ convention).
    Only returns TALs that carry a non-empty, non-timekeeping text label.
    """
    if ANNOT_LABEL not in hdr.labels:
        return []
    s = hdr.labels.index(ANNOT_LABEL)
    record_len_bytes = sum(hdr.n_samps) * 2
    off_bytes = sum(hdr.n_samps[:s]) * 2
    len_bytes = hdr.n_samps[s] * 2
    out: list[tuple[float, str]] = []
    with open(path, "rb") as f:
        f.seek(hdr.nbytes_hdr)
        data = f.read()
    n_rec = len(data) // record_len_bytes
    for r in range(n_rec):
        region = data[r * record_len_bytes + off_bytes:
                      r * record_len_bytes + off_bytes + len_bytes]
        # TALs are separated by 0x00; within a TAL: onset[0x15 duration]0x14 text 0x14
        for tal in region.split(b"\x00"):
            if not tal:
                continue
            txt = tal.decode("latin-1", errors="ignore")
            # split annotation text off the onset/duration prefix
            m = re.match(r"([+\-]\d+(?:\.\d+)?)(?:\x15\d+(?:\.\d+)?)?\x14(.*)", txt)
            if not m:
                continue
            onset = float(m.group(1))
            for label in m.group(2).split("\x14"):
                label = label.strip()
                if label:                            # skip the empty timekeeping TAL
                    out.append((onset, label))
    return out
