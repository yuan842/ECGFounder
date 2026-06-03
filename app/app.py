"""Vivalink E2E Fusion Model — internal Streamlit demo.

Single-file synchronous demo of the production DualHeadECGFounder + v3.1
ontology. See docs/cloud_demo/STREAMLIT_APP.md for the design contract.
Internal use only; no PHI, no auth, no persistence.
"""
from __future__ import annotations

import io
import re
import sys
import tempfile
import time
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from device_utils import resolve_device
from dual_head_ecgfounder import DualHeadECGFounder
from label_config import (
    FZARK_ONTOLOGY,
    ClinicalReliability,
    ClinicalRiskTier,
)
from preprocessing import ECGPreprocessor

TARGET_FS = 500
TARGET_LEN = 5000


# ─── one-time setup (cached across reruns and sessions) ──────────────────

@st.cache_resource
def get_model():
    device = resolve_device()
    model = DualHeadECGFounder(device, routing="ptbxl_specific")
    return model, device


@st.cache_resource
def get_preprocessor():
    return ECGPreprocessor.for_fzark()


@st.cache_resource
def get_detector():
    """Production v3.1 stack: backbone + L1 + L2 FP-suppression + Signal-Quality Gate.

    Used by Evaluation mode so demo metrics match the offline eval_vvl10min.py.
    """
    from overlay.inference import ScopedDetector
    return ScopedDetector()


# ─── parsers ─────────────────────────────────────────────────────────────

def parse_file(name: str, raw: bytes) -> tuple[torch.Tensor, dict]:
    """Parse uploaded bytes into a model-ready tensor of shape (1, 1, 5000).

    Returns (tensor, metadata). Metadata fields:
      filename, format, n_samples, fs_hz, duration_s, lead_used,
      multi_lead_detected (bool, optional).
    """
    ext = name.rsplit(".", 1)[-1].lower()
    meta = {"filename": name, "format": ext, "fs_hz": TARGET_FS, "lead_used": "II"}

    if ext == "csv":
        df = read_csv_robust(raw)
        sig, lead = extract_lead_ii_from_dataframe(df)
        meta["lead_used"] = lead
        meta["multi_lead_detected"] = df.shape[1] > 1
        sig = enforce_length_5000(sig)
    elif ext == "json":
        sig = preprocess_fzark_json(raw)  # → (5000,) float32 at 500 Hz
    elif ext == "zip":
        sig = preprocess_wfdb_zip(raw)
        meta["multi_lead_detected"] = True
    elif ext == "npz":
        sig = preprocess_npz(raw, meta)   # → (5000,) float32 at 500 Hz; mutates meta
    else:
        raise ValueError(f"Unsupported file extension: .{ext}")

    meta["n_samples"] = int(sig.shape[0])
    meta["duration_s"] = meta["n_samples"] / meta["fs_hz"]
    tensor = torch.from_numpy(sig.astype(np.float32)).unsqueeze(0).unsqueeze(0)
    return tensor, meta


def enforce_length_5000(sig: np.ndarray) -> np.ndarray:
    """Truncate or zero-pad to exactly 5000 samples (10 s @ 500 Hz)."""
    if sig.shape[0] >= TARGET_LEN:
        return sig[:TARGET_LEN]
    return np.pad(sig, (0, TARGET_LEN - sig.shape[0]))


def read_csv_robust(raw: bytes) -> pd.DataFrame:
    """Auto-detect headered vs headerless CSV.

    Pandas treats the first row as headers by default. If a headerless file
    is read that way, the first data point becomes a column name and
    downstream float coercion corrupts. Heuristic: read once, then if every
    column name parses as a float, re-read with header=None.
    """
    df = pd.read_csv(io.BytesIO(raw))
    try:
        [float(c) for c in df.columns]
        return pd.read_csv(io.BytesIO(raw), header=None)
    except ValueError:
        return df


_LEAD_II_RE = re.compile(r"(?:^|_)(ii|lead.?ii|lead.?2)$", re.IGNORECASE)


def extract_lead_ii_from_dataframe(df: pd.DataFrame) -> tuple[np.ndarray, str]:
    """Return (1-D float32 array, human-readable lead label)."""
    if df.shape[1] == 1:
        return df.iloc[:, 0].to_numpy(dtype=np.float32), "II (single column)"
    for i, c in enumerate(df.columns):
        if _LEAD_II_RE.search(str(c)):
            return df.iloc[:, i].to_numpy(dtype=np.float32), f"II (column '{c}')"
    return df.iloc[:, 1].to_numpy(dtype=np.float32), "II (column index 1, defaulted)"


def preprocess_fzark_json(raw: bytes) -> np.ndarray:
    """Run the fzark preprocessor on uploaded JSON bytes. Returns (5000,)."""
    prep = get_preprocessor()
    with tempfile.NamedTemporaryFile(suffix=".json", delete=True) as tmp:
        tmp.write(raw)
        tmp.flush()
        tensor = prep.from_fzark_json(tmp.name)   # (1, 5000)
    return tensor.squeeze().cpu().numpy()


# Keys we recognize as the signal / sampling-rate when an .npz doesn't follow
# the Vivalink convention. Checked in order; first hit wins.
_NPZ_SIGNAL_KEYS = ("signal", "ecg", "data", "x", "waveform", "lead_ii", "ii")
_NPZ_FS_KEYS = ("fs", "sampling_rate", "sample_rate", "sampling_frequency", "freq")


def _pick_npz_signal(npz) -> tuple[np.ndarray, str]:
    """Pick the 1-D signal array from a loaded .npz.

    Prefers a known key; otherwise falls back to the largest numeric array
    (by element count). Reduces a 2-D array to a single lead — Lead II if a
    column index 1 exists, else the first channel.
    """
    keys = list(npz.keys())
    chosen = next((k for k in _NPZ_SIGNAL_KEYS if k in npz), None)
    if chosen is None:
        numeric = [(k, npz[k]) for k in keys
                   if np.issubdtype(npz[k].dtype, np.number) and npz[k].ndim in (1, 2)]
        if not numeric:
            raise ValueError(
                f".npz has no recognizable numeric signal array (keys: {keys})"
            )
        chosen = max(numeric, key=lambda kv: kv[1].size)[0]
    arr = np.asarray(npz[chosen])
    if arr.ndim == 2:
        # Orient so the long axis is time, then pick a lead.
        if arr.shape[0] < arr.shape[1]:   # (channels, samples)
            arr = arr.T
        lead = 1 if arr.shape[1] > 1 else 0
        return arr[:, lead].astype(np.float64), f"{chosen}[:, {lead}]"
    return arr.ravel().astype(np.float64), chosen


def preprocess_npz(raw: bytes, meta: dict) -> np.ndarray:
    """Load an .npz, resample/crop via the fzark preprocessor. Returns (5000,).

    Recognizes the Vivalink VVL10min convention (`signal` int16 + `fs`), and
    falls back to the largest numeric array + 500 Hz when keys are unknown.
    Mutates `meta` with native fs/length and a crop note (the model analyzes a
    single 10 s window — long records are center-cropped after resampling).
    """
    prep = get_preprocessor()
    with np.load(io.BytesIO(raw), allow_pickle=False) as npz:
        sig, sig_key = _pick_npz_signal(npz)
        native_fs = next((int(npz[k]) for k in _NPZ_FS_KEYS
                          if k in npz and np.ndim(npz[k]) == 0), None)

    if native_fs is None:
        native_fs = TARGET_FS
        meta["fs_assumed"] = True
    meta["native_fs_hz"] = native_fs
    meta["native_n_samples"] = int(sig.shape[0])
    meta["native_duration_s"] = sig.shape[0] / native_fs
    meta["npz_signal_key"] = sig_key
    meta["multi_lead_detected"] = "[:" in sig_key
    # process() resamples → center-crops to 5000 (10 s). Flag if we dropped signal.
    if meta["native_duration_s"] > TARGET_LEN / TARGET_FS + 0.5:
        meta["cropped_to_center_10s"] = True

    tensor = prep.process(sig, fs_in=native_fs)   # (1, 5000) @ 500 Hz
    return tensor.squeeze().cpu().numpy()


def preprocess_wfdb_zip(raw: bytes) -> np.ndarray:
    """Extract .dat + .hea from an uploaded zip and run WFDB preprocessing."""
    prep = get_preprocessor()
    with tempfile.TemporaryDirectory() as td:
        with zipfile.ZipFile(io.BytesIO(raw)) as zf:
            zf.extractall(td)
        hea_files = list(Path(td).rglob("*.hea"))
        if not hea_files:
            raise ValueError("ZIP does not contain a .hea file")
        record_path = str(hea_files[0].with_suffix(""))
        tensor = prep.from_wfdb(record_path)      # (1, 5000)
    return tensor.squeeze().cpu().numpy()


# ─── inference ───────────────────────────────────────────────────────────

def infer(tensor: torch.Tensor, model, device) -> dict:
    t0 = time.perf_counter()
    with torch.no_grad():
        probs = torch.sigmoid(model(tensor.to(device))).cpu().numpy().ravel()
    inference_ms = int((time.perf_counter() - t0) * 1000)

    findings: list[dict] = []
    for event_name, node in FZARK_ONTOLOGY.items():
        p = float(probs[node.ecgfounder_index])
        if p < 0.5:
            continue
        findings.append({
            "event_label": event_name,
            "founder_head_idx": node.ecgfounder_index,
            "founder_head_name": node.canonical_name,
            "probability": p,
            "risk_tier": node.risk_tier.value,
            "reliability": (
                node.clinical_reliability.value
                if node.clinical_reliability is not None
                else "insufficient_data"
            ),
            "ppv_pct": node.fzark_ppv_pct,
            "clinical_support": node.clinical_support,
        })
    findings.sort(key=lambda f: (-f["probability"], _risk_rank(f["risk_tier"])))

    all_heads: dict[str, float] = {}
    seen_idx: set[int] = set()
    for _, node in FZARK_ONTOLOGY.items():
        if node.ecgfounder_index in seen_idx:
            continue
        seen_idx.add(node.ecgfounder_index)
        all_heads[node.canonical_name] = float(probs[node.ecgfounder_index])

    return {
        "findings": findings,
        "all_v31_head_probabilities": all_heads,
        "inference_ms": inference_ms,
    }


# ─── 10-min block evaluation (VVL10min annotation) ─────────────────────────
#
# A VVL10min .npz carries its own ground truth: `seg_majority` labels each of
# 20 contiguous 30-s segments (0 Has Afibs · 1 Too Noisy · 2 No Afibs · 3 Others).
# The annotation is AFib-only, so the evaluation scores the AFib head (5) only.
# Positive = Has Afibs; negative = No Afibs. Too Noisy and Others are excluded
# (uninterpretable / not a clean non-AFib reference). Mirrors eval_vvl10min.py.

AFIB_HEAD = 5
SEG_STATES = ["Has Afibs", "Too Noisy", "No Afibs", "Others"]  # indices 0..3
_POS_IDX = 0       # Has Afibs == positive
_NEG_IDX = 2       # No Afibs  == negative
_NOISY_IDX = 1     # Too Noisy == excluded (uninterpretable)
_OTHERS_IDX = 3    # Others    == excluded (not a clean non-AFib reference)
SCORABLE_IDX = (_POS_IDX, _NEG_IDX)
SEG_SECONDS = 30.0
WIN_SECONDS = 10.0


def npz_has_annotation(raw: bytes) -> bool:
    """True if the .npz is a VVL10min-style annotated block (signal + seg_majority)."""
    try:
        with np.load(io.BytesIO(raw), allow_pickle=False) as npz:
            return "seg_majority" in npz and "signal" in npz
    except Exception:
        return False


def _split(arr: np.ndarray, fs: int, seconds: float) -> list[np.ndarray]:
    n = int(round(seconds * fs))
    return [arr[i * n:(i + 1) * n] for i in range(len(arr) // n)]


def evaluate_block(raw: bytes, prep, detector, progress=None) -> tuple[list[dict], np.ndarray, int]:
    """Score every 30-s segment of a 10-min block against its `seg_majority` GT.

    Each segment = 3 contiguous 10-s windows; segment AFib prob = max over
    windows, segment fires if any window fires. Returns (rows, signal, fs).
    """
    with np.load(io.BytesIO(raw), allow_pickle=False) as d:
        sig = d["signal"].astype(np.float32)
        fs = int(d["fs"])
        seg_majority = d["seg_majority"].astype(int)

    segs = _split(sig, fs, SEG_SECONDS)
    n_seg = min(len(segs), len(seg_majority))
    rows: list[dict] = []
    for s_idx in range(n_seg):
        seg, gt = segs[s_idx], int(seg_majority[s_idx])
        windows = _split(seg, fs, WIN_SECONDS)
        probs, fired, gated = [], [], 0
        for w in windows:
            x = prep.process(w, fs_in=fs)                 # (1, 5000) float32
            decisions = detector.detect(x)
            if decisions is None:                         # SQG gated this window
                gated += 1
                continue
            dec = decisions.get(AFIB_HEAD)
            if dec is None:
                continue
            probs.append(float(dec.score))
            fired.append(bool(dec.fired))
        rows.append(dict(
            seg_idx=s_idx,
            gt_idx=gt,
            gt_state=SEG_STATES[gt] if 0 <= gt < len(SEG_STATES) else str(gt),
            n_windows=len(windows),
            n_gated=gated,
            seg_prob=(max(probs) if probs else float("nan")),
            seg_fire=(any(fired) if fired else False),
        ))
        if progress is not None:
            progress((s_idx + 1) / n_seg)
    return rows, sig, fs


def eval_confusion(rows: list[dict]) -> dict:
    """AFib confusion. Positive = Has Afibs, negative = No Afibs.

    Too Noisy, Others, and fully-gated segments are excluded from scoring.
    """
    keep = [r for r in rows
            if r["gt_idx"] in SCORABLE_IDX
            and not (isinstance(r["seg_prob"], float) and np.isnan(r["seg_prob"]))]
    tp = sum(1 for r in keep if r["gt_idx"] == _POS_IDX and r["seg_fire"])
    fn = sum(1 for r in keep if r["gt_idx"] == _POS_IDX and not r["seg_fire"])
    fp = sum(1 for r in keep if r["gt_idx"] == _NEG_IDX and r["seg_fire"])
    tn = sum(1 for r in keep if r["gt_idx"] == _NEG_IDX and not r["seg_fire"])

    def _ratio(num, den):
        return num / den if den else float("nan")

    return dict(
        tp=tp, fn=fn, fp=fp, tn=tn, n_scored=len(keep),
        n_drop_noisy=sum(1 for r in rows if r["gt_idx"] == _NOISY_IDX),
        n_drop_others=sum(1 for r in rows if r["gt_idx"] == _OTHERS_IDX),
        n_drop_gated=sum(1 for r in rows
                         if r["gt_idx"] in SCORABLE_IDX
                         and isinstance(r["seg_prob"], float) and np.isnan(r["seg_prob"])),
        sens=_ratio(tp, tp + fn),
        spec=_ratio(tn, tn + fp),
        ppv=_ratio(tp, tp + fp),
        npv=_ratio(tn, tn + fn),
        f1=_ratio(2 * tp, 2 * tp + fp + fn),
    )


# ─── render helpers ──────────────────────────────────────────────────────

_RISK_DOT = {
    "critical": "🔴",
    "high":     "🟠",
    "moderate": "🟡",
    "low":      "🟢",
}
_RISK_RANK = {"critical": 0, "high": 1, "moderate": 2, "low": 3}


def _risk_dot(tier: str) -> str:
    return _RISK_DOT.get(tier, "⚪")


def _risk_rank(tier: str) -> int:
    return _RISK_RANK.get(tier, 99)


def _chip(text: str, bg: str, fg: str = "#fff") -> str:
    return (
        f"<span style='background:{bg};color:{fg};padding:2px 8px;"
        f"border-radius:10px;font-size:0.75rem;font-weight:600;'>{text}</span>"
    )


_TIER_BG = {
    "critical": "#dc2626",
    "high":     "#ea580c",
    "moderate": "#ca8a04",
    "low":      "#16a34a",
}
_RELIABILITY_BG = {
    "reliable":          "#16a34a",
    "moderate_fp":       "#ca8a04",
    "severe_fp":         "#dc2626",
    "insufficient_data": "#6b7280",
}


def render_waveform(sig: np.ndarray, steps: list[str]) -> None:
    st.caption("Preprocessed waveform · " + " · ".join(steps))
    df = pd.DataFrame({"amplitude": sig}, index=np.arange(sig.shape[0]) / TARGET_FS)
    df.index.name = "time (s)"
    st.line_chart(df, height=200)


def render_findings(findings: list[dict]) -> None:
    n = len(findings)
    st.subheader(f"Findings · {n} event{'' if n == 1 else 's'} ≥ 0.5")
    if not findings:
        st.info("No findings ≥ 0.5. See the full-head chart below for raw probabilities.")
        return
    for f in findings:
        cols = st.columns([0.08, 0.42, 0.15, 0.15, 0.2])
        cols[0].markdown(_risk_dot(f["risk_tier"]))
        cols[1].markdown(f"**{f['event_label']}**  \n*{f['founder_head_name']}*")
        cols[2].markdown(f"`{f['probability']:.3f}`")
        cols[3].markdown(
            _chip(f["risk_tier"].upper(), _TIER_BG.get(f["risk_tier"], "#6b7280")),
            unsafe_allow_html=True,
        )
        ppv = f["ppv_pct"]
        ppv_text = f" · PPV {ppv:.0f}%" if ppv is not None else ""
        cols[4].markdown(
            _chip(
                f["reliability"].replace("_", " ") + ppv_text,
                _RELIABILITY_BG.get(f["reliability"], "#6b7280"),
            ),
            unsafe_allow_html=True,
        )
        if f["clinical_support"]:
            st.caption(f["clinical_support"])
        st.divider()


def render_bar_chart(all_heads: dict) -> None:
    st.subheader("All 10 V3.1 heads")
    sorted_heads = dict(sorted(all_heads.items(), key=lambda kv: -kv[1]))
    st.bar_chart(sorted_heads, horizontal=True, x_label="probability")


# ─── evaluation render ─────────────────────────────────────────────────────

def render_overview_waveform(sig: np.ndarray, fs: int, max_points: int = 4000) -> None:
    """Decimated full-record overview (a 10-min block is too dense to plot raw)."""
    step = max(1, sig.shape[0] // max_points)
    dec = sig[::step]
    df = pd.DataFrame({"amplitude": dec},
                      index=np.arange(dec.shape[0]) * step / fs)
    df.index.name = "time (s)"
    st.caption(f"Full record overview · {sig.shape[0] / fs:.0f} s @ {fs} Hz "
               f"(decimated {step}×)")
    st.line_chart(df, height=160)


def _seg_chip(label: str, bg: str, error: bool) -> str:
    border = "2px solid #111" if error else "1px solid rgba(0,0,0,0.15)"
    return (f"<div style='flex:1;min-width:26px;text-align:center;background:{bg};"
            f"color:#fff;border:{border};border-radius:4px;padding:3px 0;"
            f"font-size:0.7rem;font-weight:600;'>{label}</div>")


def render_segment_timeline(rows: list[dict]) -> None:
    """Two aligned strips of 20 cells: ground truth (top) vs prediction (bottom)."""
    st.markdown("**Segment timeline** · 20 × 30 s · ⬛ outline = disagreement")
    gt_bg = {0: "#dc2626", 1: "#6b7280", 2: "#16a34a", 3: "#0ea5e9"}  # afib/noisy/clean/other
    gt_cells, pred_cells = [], []
    for r in rows:
        gt_pos = r["gt_idx"] == _POS_IDX
        gated = isinstance(r["seg_prob"], float) and np.isnan(r["seg_prob"])
        scorable = r["gt_idx"] in SCORABLE_IDX
        # disagreement only counts on scorable (Has Afibs / No Afibs) segments
        err = (scorable and not gated and (gt_pos != r["seg_fire"]))
        gt_cells.append(_seg_chip(str(r["seg_idx"] + 1),
                                  gt_bg.get(r["gt_idx"], "#6b7280"), err))
        if gated:
            pred_bg = "#6b7280"
        else:
            pred_bg = "#dc2626" if r["seg_fire"] else "#16a34a"
        pred_cells.append(_seg_chip(str(r["seg_idx"] + 1), pred_bg, err))

    st.markdown("<div style='font-size:0.75rem;color:#666;margin-bottom:2px'>Ground truth</div>"
                "<div style='display:flex;gap:3px'>" + "".join(gt_cells) + "</div>",
                unsafe_allow_html=True)
    st.markdown("<div style='font-size:0.75rem;color:#666;margin:6px 0 2px'>Model prediction</div>"
                "<div style='display:flex;gap:3px'>" + "".join(pred_cells) + "</div>",
                unsafe_allow_html=True)
    st.caption("GT: 🔴 Has Afibs · 🟢 No Afibs · 🔵 Others (excluded) · ⚪ Too Noisy (excluded)   "
               "Pred: 🔴 AFib fired · 🟢 no fire · ⚪ quality-gated")


def render_eval_summary(rows: list[dict], m: dict) -> None:
    st.subheader("AFib detection performance · 30-s segments")
    st.caption(
        f"{len(rows)} segments · scored {m['n_scored']} "
        f"(positive=Has Afibs, negative=No Afibs) · "
        f"excluded {m['n_drop_noisy']} Too-Noisy + {m['n_drop_others']} Others + "
        f"{m['n_drop_gated']} quality-gated · "
        "production ScopedDetector (backbone + L1/L2 + SQG)"
    )

    c = st.columns(5)
    c[0].metric("Sensitivity", _pct(m["sens"]))
    c[1].metric("Specificity", _pct(m["spec"]))
    c[2].metric("PPV", _pct(m["ppv"]))
    c[3].metric("NPV", _pct(m["npv"]))
    c[4].metric("F1", "—" if np.isnan(m["f1"]) else f"{m['f1']:.3f}")

    conf = pd.DataFrame(
        [[m["tp"], m["fn"]], [m["fp"], m["tn"]]],
        index=["GT AFib", "GT not-AFib"],
        columns=["Pred AFib", "Pred not-AFib"],
    )
    st.markdown("**Confusion matrix** (segment-level)")
    st.table(conf)
    if m["tn"] + m["fp"] == 0:
        st.info("No negative (non-AFib) segments in this block — specificity/PPV "
                "are undefined. This recording is essentially pure-AFib.")


def render_segment_table(rows: list[dict]) -> None:
    with st.expander("Per-segment detail"):
        df = pd.DataFrame([{
            "segment": r["seg_idx"] + 1,
            "t_start_s": int(r["seg_idx"] * SEG_SECONDS),
            "ground_truth": r["gt_state"],
            "AFib_prob": (None if isinstance(r["seg_prob"], float) and np.isnan(r["seg_prob"])
                          else round(r["seg_prob"], 3)),
            "fired": r["seg_fire"],
            "windows_gated": r["n_gated"],
        } for r in rows])
        st.dataframe(df, use_container_width=True, hide_index=True)


def _pct(x: float) -> str:
    return "—" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x * 100:.1f}%"


# ─── main entry point ────────────────────────────────────────────────────

def run_detection_view(name: str, raw: bytes, model, device) -> None:
    """Single 10-s window detection across all 10 heads (CSV/JSON/ZIP/short NPZ)."""
    try:
        with st.spinner("Analyzing…"):
            tensor, meta = parse_file(name, raw)
            result = infer(tensor, model, device)
    except (ValueError, RuntimeError) as exc:
        st.error(f"Could not process file: {exc}")
        st.stop()

    if meta.get("multi_lead_detected"):
        st.info(f"🔍 Multi-lead input detected — using Lead {meta['lead_used']} for inference.")
    if meta.get("cropped_to_center_10s"):
        st.info(
            f"✂️ Long record ({meta['native_duration_s']:.0f} s @ {meta['native_fs_hz']} Hz) — "
            "analyzing the center 10 s window only."
        )
    if meta.get("fs_assumed"):
        st.warning(f"⚠️ No sampling rate in the .npz — assuming {TARGET_FS} Hz.")

    sig_np = tensor.squeeze().cpu().numpy()
    render_waveform(
        sig_np,
        steps=["50 Hz notch", "0.67–40 Hz bandpass", "median baseline", "winsorized zscore"],
    )
    render_findings(result["findings"])
    render_bar_chart(result["all_v31_head_probabilities"])

    native = ""
    if meta.get("native_fs_hz") and meta["native_fs_hz"] != meta["fs_hz"]:
        native = (f" · native {meta['native_n_samples']} samples @ "
                  f"{meta['native_fs_hz']} Hz ({meta['native_duration_s']:.1f} s)")
    st.caption(
        f"{meta['filename']} · {meta['format']} · {meta['n_samples']} samples "
        f"@ {meta['fs_hz']} Hz · {meta['duration_s']:.1f} s{native} · "
        f"inference {result['inference_ms']} ms"
    )


def run_evaluation_view(name: str, raw: bytes) -> None:
    """Whole-record AFib evaluation of an annotated 10-min VVL block vs its GT."""
    st.info("📋 Annotated 10-min block detected — running full-record AFib evaluation.")
    prep = get_preprocessor()
    detector = get_detector()
    bar = st.progress(0.0, text="Scoring 30-s segments…")
    try:
        rows, sig, fs = evaluate_block(
            raw, prep, detector,
            progress=lambda frac: bar.progress(frac, text=f"Scoring segments… {frac*100:.0f}%"),
        )
    except (ValueError, RuntimeError, KeyError) as exc:
        bar.empty()
        st.error(f"Could not evaluate block: {exc}")
        st.stop()
    bar.empty()

    m = eval_confusion(rows)
    render_eval_summary(rows, m)
    render_segment_timeline(rows)
    render_overview_waveform(sig, fs)
    render_segment_table(rows)
    st.caption(
        f"{name} · npz · {sig.shape[0]} samples @ {fs} Hz · "
        f"{sig.shape[0] / fs:.0f} s · {len(rows)} segments × {WIN_SECONDS:.0f}-s windows · "
        "AFib head (5) only"
    )


def main() -> None:
    st.set_page_config(page_title="Vivalink E2E Fusion Model", layout="centered")
    st.title("Vivalink E2E Fusion Model — internal demo")
    st.caption(
        "Single-lead DualHeadECGFounder + v3.1 ontology · "
        "internal use only · not for clinical or PHI workloads"
    )

    with st.sidebar:
        st.markdown("### Status")
        model, device = get_model()
        st.markdown(f"● **model ready** ({device})")
        st.markdown("---")
        st.markdown("### Accepted formats")
        st.markdown(
            "- **CSV** — single Lead II column, or multi-lead with `II` "
            "named or at column index 1. Assumed 500 Hz.\n"
            "- **JSON** — fzark/Vivalink sidecar (128 Hz).\n"
            "- **ZIP** — WFDB `.dat` + `.hea` pair.\n"
            "- **NPZ** — NumPy archive with a `signal` array (+ optional `fs`); "
            "e.g. VVL10min blocks. Long records use the center 10 s.\n"
            "  - With a `seg_majority` annotation → **full-record AFib evaluation**."
        )

    uploaded = st.file_uploader(
        "Drop or browse an ECG file",
        type=["csv", "json", "zip", "npz"],
        accept_multiple_files=False,
        help="CSV (raw signal, 500 Hz), JSON (fzark sidecar), ZIP (.dat + .hea), "
             "or NPZ (NumPy archive with a 'signal' array; annotated VVL10min "
             "blocks run a full-record evaluation). Max 25 MB.",
    )
    if uploaded is None:
        st.stop()

    raw = uploaded.getvalue()
    if uploaded.name.lower().endswith(".npz") and npz_has_annotation(raw):
        run_evaluation_view(uploaded.name, raw)
    else:
        run_detection_view(uploaded.name, raw, model, device)


if __name__ == "__main__":
    main()
