"""ECG Founder — internal Streamlit demo.

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


# ─── main entry point ────────────────────────────────────────────────────

def main() -> None:
    st.set_page_config(page_title="ECG Founder Demo", layout="centered")
    st.title("ECG Founder — internal demo")
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
            "- **ZIP** — WFDB `.dat` + `.hea` pair."
        )

    uploaded = st.file_uploader(
        "Drop or browse an ECG file",
        type=["csv", "json", "zip"],
        accept_multiple_files=False,
        help="CSV (raw signal, 500 Hz), JSON (fzark sidecar), or ZIP (.dat + .hea). Max 25 MB.",
    )
    if uploaded is None:
        st.stop()

    try:
        with st.spinner("Analyzing…"):
            tensor, meta = parse_file(uploaded.name, uploaded.getvalue())
            result = infer(tensor, model, device)
    except (ValueError, RuntimeError) as exc:
        st.error(f"Could not process file: {exc}")
        st.stop()

    if meta.get("multi_lead_detected"):
        st.info(f"🔍 Multi-lead input detected — using Lead {meta['lead_used']} for inference.")

    sig_np = tensor.squeeze().cpu().numpy()
    render_waveform(
        sig_np,
        steps=["50 Hz notch", "0.67–40 Hz bandpass", "median baseline", "winsorized zscore"],
    )
    render_findings(result["findings"])
    render_bar_chart(result["all_v31_head_probabilities"])

    st.caption(
        f"{meta['filename']} · {meta['format']} · {meta['n_samples']} samples "
        f"@ {meta['fs_hz']} Hz · {meta['duration_s']:.1f} s · "
        f"inference {result['inference_ms']} ms"
    )


if __name__ == "__main__":
    main()
