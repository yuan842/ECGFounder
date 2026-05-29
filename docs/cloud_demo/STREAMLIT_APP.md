# Streamlit App Design — ECG Cloud Demo

**Layer**: single-file Streamlit application
**Status**: planning only
**Top-level**: [README.md](README.md)
**Supersedes**: [_archived/BACKEND.md](_archived/BACKEND.md) + [_archived/FRONTEND.md](_archived/FRONTEND.md)

---

## 1. Purpose & scope

Internal MVP web app for **synchronous, on-demand ECG inference** using the current production model stack (`DualHeadECGFounder` + v3.1 ontology). The operator drags an ECG file onto the page; within 1–2 seconds, the page shows:

1. The preprocessed waveform
2. A findings list with risk + reliability badges (sourced from `FZARK_ONTOLOGY`)
3. A 10-head probability bar chart

Single-tenant, no auth, no persistence. **Demo / algorithm evaluation only — not for PHI or clinical use.**

## 2. Non-goals

- ❌ No multi-tenancy, user accounts, JWT, session auth
- ❌ No async job queue; everything synchronous
- ❌ No database, cache, or object storage
- ❌ No HIPAA / PHI handling — demo data only
- ❌ No FastAPI, no JSON API, no JS, no CSS framework
- ❌ No request-level metrics endpoint — `journalctl` is enough

## 3. Architecture

```
[Browser]  ──── HTTPS ──── (internal VPN) ────>  [Streamlit on EC2 :8000]
                                                       │
                                                       ├─ on cold start (~30 s):
                                                       │    @st.cache_resource → DualHeadECGFounder
                                                       │    @st.cache_resource → ECGPreprocessor
                                                       │
                                                       └─ per file upload (~1 s):
                                                            parse_file()  →  Tensor (1, 5000)
                                                            infer()       →  result dict
                                                            render_*()    →  st.pyplot, st.bar_chart, …
```

**Why monolithic Streamlit is acceptable here**: see [README.md §"Why Streamlit"](README.md#why-streamlit-and-not-the-prior-fastapi--vanilla-js-plan). Briefly — the demo audience is internal, the codebase is small, and the time-to-first-demo halves vs the FastAPI plan. The trade-off is real (no JSON contract, no UI/model separation), and the [archived FastAPI plan](_archived/) is preserved for re-promotion if requirements change.

## 4. User flow

```
1. Page loads
   ├─ Streamlit runs app.py top-to-bottom
   ├─ Model loads on first run via @st.cache_resource (one-time 30 s cold start)
   └─ Empty drop zone + status indicator in sidebar ("● model ready" / "🟡 loading…")

2. User drags ECG file onto the file uploader (or clicks "Browse files")
   ├─ Streamlit re-runs the script, this time with file in session state
   ├─ Spinner appears: "Analyzing…"
   └─ parse_file → infer → render

3. Result panel renders top-to-bottom
   ├─ (Optional) "12-lead detected — Lead II used" banner
   ├─ Waveform plot (st.pyplot)
   ├─ Findings list — one row per finding ≥ 0.5, with risk + reliability chips
   ├─ All-10-heads bar chart (st.bar_chart with sorted dict)
   └─ Metadata footer: filename, format, fs_hz, duration_s, inference_ms

4. User clicks "Run another"
   └─ st.session_state.uploaded_file = None → re-run → back to step 1
```

## 5. Wireframe (Streamlit components rendered top-to-bottom)

```
╔═══════════ sidebar ═══════════╗  ═════════════════════════════════════════════════════════════════════
║ Status                        ║    ECG Founder — internal demo
║ ● model ready (mps)           ║  ═════════════════════════════════════════════════════════════════════
║                               ║
║ Sample files                  ║    📂 Drop or browse an ECG file
║ ┌───────────────────────────┐ ║    ┌──────────────────────────────────────────────────────────────┐
║ │ AFib TP                   │ ║    │   Drag and drop file here                Browse files        │
║ ├───────────────────────────┤ ║    │   Limit 25 MB per file • CSV, JSON, ZIP (.dat+.hea)          │
║ │ PVC burden                │ ║    └──────────────────────────────────────────────────────────────┘
║ ├───────────────────────────┤ ║
║ │ Multi-finding strip       │ ║    ─── after upload ───
║ ├───────────────────────────┤ ║
║ │ Sin Tach false positive   │ ║
║ └───────────────────────────┘ ║
╚═══════════════════════════════╝

  ℹ️ 12-lead detected — Lead II used for inference        (banner, optional)

  Waveform (preprocessed: notch + bandpass + median + zscore)
  ╱╲      ╱╲      ╱╲      ╱╲      ╱╲                    (st.line_chart, interactive)
 ╱  ╲────╱  ╲────╱  ╲────╱  ╲────╱  ╲

  Findings  •  2 events ≥ 0.5
  ┌──────────────────────────────────────────────────────────────┐
  │ 🟠 ATRIAL FIBRILLATION    0.962  ▢HIGH      ▢reliable        │
  │     Irregularly irregular atrial rhythm lacking distinct…   │
  ├──────────────────────────────────────────────────────────────┤
  │ 🟡 PVC (Isolated V Beat)  0.78   ▢LOW       ▢severe-fp       │
  │     Ectopic beats originating from ventricles…              │
  └──────────────────────────────────────────────────────────────┘

  All 10 V3.1 heads (probability)                      (st.bar_chart)
  ▮▮▮▮▮▮▮▮▮▮▮▮▮▮▮▮▮▮▮▮ ATRIAL FIBRILLATION   0.96
  ▮▮▮▮▮▮▮▮▮▮▮▮▮▮▮      PVC                   0.78
  ▮▮                   ISB / PAC              0.10
  ▮                    Bradycardia            0.04
  …

  sample_001.csv · csv · 5000 samples @ 500 Hz · 10.0s · 287 ms

  [ Run another ]
═════════════════════════════════════════════════════════════════════
```

The Streamlit look (white background, grey header bar) is the default theme and is acceptable for internal use. The `.streamlit/config.toml` will only set `primaryColor` to match VivaLink brand if desired.

## 6. File layout

```
app/
├── app.py                ~150 LoC — the entire application (incl. ~10 LoC for sample-data buttons)
├── requirements.txt      streamlit + repo's existing deps (numpy, pandas, torch, wfdb)
│                         # matplotlib NOT required — st.line_chart renders the waveform
├── sample_data/          4 pre-staged fzark-format JSONs (~50–200 KB each, <1 MB total)
│   ├── sample_afib_tp.json           # AFib TP from data/ecg_tp_fzark/Atrial Fibrillation/
│   ├── sample_pvc_burden.json        # PVC-rich TP demonstrating v3.1 head-sharing
│   ├── sample_multi_finding.json     # strip that fires 2-3 heads simultaneously
│   └── sample_sintach_fp.json        # Sin Tach FP from data/ecg_fp_doctor removed1/ — 99% FP rate head
└── .streamlit/
    └── config.toml       theme + server port + maxUploadSize=25
```

`app.py` is the only Python file we write. It imports the existing algorithm modules in-place:

```python
# app/app.py
from device_utils import resolve_device
from dual_head_ecgfounder import DualHeadECGFounder
from label_config import FZARK_ONTOLOGY, FZARK_LABEL_MAP
from preprocessing import ECGPreprocessor
```

## 7. Function-level design (~150 LoC budget)

The script reads top-to-bottom on each Streamlit re-run; only `@st.cache_resource`-decorated functions retain state across re-runs.

```python
# Pseudocode — actual implementation in phase 1

# ─── module-level imports (top of file) ──────────────────────────────────
import io, time, uuid
from pathlib import Path
import numpy as np, pandas as pd, torch
import streamlit as st     # matplotlib dropped — using st.line_chart instead

from device_utils import resolve_device
from dual_head_ecgfounder import DualHeadECGFounder
from label_config import FZARK_ONTOLOGY
from preprocessing import ECGPreprocessor

# ─── one-time setup (cached across reruns and sessions) ──────────────────
@st.cache_resource
def get_model():
    device = resolve_device()
    model = DualHeadECGFounder(device, routing='ptbxl_specific')
    return model, device

@st.cache_resource
def get_preprocessor():
    return ECGPreprocessor.for_fzark()

# ─── parsers ─────────────────────────────────────────────────────────────
def parse_file(name: str, raw: bytes) -> tuple[torch.Tensor, dict]:
    """Return (tensor of shape (1, 5000), metadata dict).

    Accepts raw bytes + filename so the same function works for both
    user-uploaded files (UploadedFile.getvalue()) AND pre-staged sample
    files (open(path, "rb").read()) — no UploadedFile-specific API used.

    Metadata fields:
      filename, format, n_samples, fs_hz, duration_s, lead_used,
      multi_lead_detected (bool, optional)
    """
    ext = name.rsplit(".", 1)[-1].lower()
    meta = {"filename": name, "format": ext, "fs_hz": 500, "lead_used": "II"}
    if ext == "csv":
        df = read_csv_robust(raw)                              # handles headered vs headerless
        sig, lead = extract_lead_ii_from_dataframe(df)         # returns (np.ndarray, str)
        meta["lead_used"] = lead
        meta["multi_lead_detected"] = (df.shape[1] > 1)
        sig = enforce_length_5000(sig)
    elif ext == "json":
        prep = get_preprocessor()
        sig = prep.from_fzark_json(io.BytesIO(raw)).squeeze().numpy()
        sig = enforce_length_5000(sig)
    elif ext == "zip":  # .dat + .hea bundle
        sig = parse_wfdb_zip(raw)                              # extracts Lead II
        meta["multi_lead_detected"] = True
        sig = enforce_length_5000(sig)
    else:
        raise ValueError(f"Unsupported file extension: .{ext}")
    meta["n_samples"] = sig.shape[0]
    meta["duration_s"] = sig.shape[0] / meta["fs_hz"]
    return torch.from_numpy(sig.astype(np.float32)).unsqueeze(0).unsqueeze(0), meta

def enforce_length_5000(sig: np.ndarray) -> np.ndarray:
    """Truncate or zero-pad to exactly 5000 samples (10 s @ 500 Hz).
    Eliminates the shape-mismatch class of inference errors at the parser
    layer (per reviewer round-1 fix #3)."""
    if sig.shape[0] >= 5000: return sig[:5000]
    return np.pad(sig, (0, 5000 - sig.shape[0]))

def read_csv_robust(raw: bytes) -> pd.DataFrame:
    """Read CSV with auto-detection of headered vs headerless files.

    Pandas's default `read_csv` consumes the first row as column names. If the
    file is headerless (just numeric data), the first data point becomes a
    string column name and downstream `.to_numpy(dtype=float32)` corrupts.

    Heuristic (reviewer round-2 suggestion #2 — accepted): read with the
    default, then check whether every resulting column name parses as a float.
    If yes → it was actually headerless → re-read with `header=None`.
    """
    df = pd.read_csv(io.BytesIO(raw))
    try:
        [float(c) for c in df.columns]
        return pd.read_csv(io.BytesIO(raw), header=None)
    except ValueError:
        return df

def extract_lead_ii_from_dataframe(df: pd.DataFrame) -> tuple[np.ndarray, str]:
    """Return (signal as 1-D float32 array, human-readable lead label).

    Priority order:
      1. Single column → treat that column as Lead II.
      2. Multi-column with a name matching /(^|_)(II|lead.?ii|lead.?2)$/i → use that.
      3. Multi-column with no name match → default to column index 1
         (assumes the common `[timestamp, lead_i, lead_ii, ...]` layout — per
         reviewer round-2 suggestion #2).
    """
    if df.shape[1] == 1:
        return df.iloc[:, 0].to_numpy(dtype=np.float32), "II (single column)"
    name_col = _find_lead_ii_by_name(df.columns)
    if name_col is not None:
        return df.iloc[:, name_col].to_numpy(dtype=np.float32), f"II (column '{df.columns[name_col]}')"
    return df.iloc[:, 1].to_numpy(dtype=np.float32), "II (column index 1, defaulted)"

# ─── inference (sync) ────────────────────────────────────────────────────
def infer(tensor: torch.Tensor, model, device) -> dict:
    t0 = time.perf_counter()
    with torch.no_grad():
        probs = torch.sigmoid(model(tensor.to(device))).cpu().numpy().ravel()
    inference_ms = int((time.perf_counter() - t0) * 1000)

    # Findings list — one entry per V3.1 event whose head fires ≥ 0.5
    findings = []
    for event_name, node in FZARK_ONTOLOGY.items():
        p = float(probs[node.ecgfounder_index])
        if p < 0.5: continue
        findings.append({
            "event_label": event_name,
            "founder_head_idx": node.ecgfounder_index,
            "founder_head_name": node.canonical_name,
            "probability": p,
            "risk_tier": node.risk_tier.value,
            "reliability": node.clinical_reliability.value,
            "ppv_pct": node.fzark_ppv_pct,
            "clinical_support": node.clinical_support,
        })
    # Descending by probability; ties broken by risk tier rank
    findings.sort(key=lambda f: (-f["probability"], _risk_rank(f["risk_tier"])))

    # All 10 unique-head probabilities for the bar chart
    all_heads = {}
    seen_idx = set()
    for event_name, node in FZARK_ONTOLOGY.items():
        if node.ecgfounder_index in seen_idx: continue
        seen_idx.add(node.ecgfounder_index)
        all_heads[node.canonical_name] = float(probs[node.ecgfounder_index])

    return {
        "findings": findings,
        "all_v31_head_probabilities": all_heads,
        "inference_ms": inference_ms,
    }

# ─── rendering ───────────────────────────────────────────────────────────
def render_waveform(sig: np.ndarray, preprocessing_steps: list[str]):
    # Use Streamlit's native Altair-backed chart: interactive (pan, zoom, hover)
    # and ~5× faster than rasterizing through matplotlib. No figure lifecycle
    # to manage. (Reviewer round-2 suggestion #1 — accepted.)
    st.caption("Preprocessed waveform · " + " · ".join(preprocessing_steps))
    df = pd.DataFrame({"amplitude": sig}, index=np.arange(sig.shape[0]) / 500.0)
    df.index.name = "time (s)"
    st.line_chart(df, height=200)

def render_findings(findings: list[dict]):
    st.subheader(f"Findings · {len(findings)} event{'' if len(findings)==1 else 's'} ≥ 0.5")
    if not findings:
        st.info("No findings ≥ 0.5. See the full-head chart below for raw probabilities.")
        return
    for f in findings:
        cols = st.columns([0.1, 0.4, 0.15, 0.15, 0.2])
        cols[0].markdown(_risk_dot(f["risk_tier"]))
        cols[1].markdown(f"**{f['event_label']}**  \n*{f['founder_head_name']}*")
        cols[2].markdown(f"`{f['probability']:.3f}`")
        cols[3].markdown(_chip(f["risk_tier"]))
        cols[4].markdown(_reliability_chip(f["reliability"], f["ppv_pct"]))
        st.caption(f["clinical_support"])
        st.divider()

def render_bar_chart(all_heads: dict):
    st.subheader("All 10 V3.1 heads")
    sorted_heads = dict(sorted(all_heads.items(), key=lambda kv: -kv[1]))
    st.bar_chart(sorted_heads, horizontal=True, x_label="probability")

# ─── main entry point ────────────────────────────────────────────────────
def main():
    st.set_page_config(page_title="ECG Founder Demo", layout="centered")
    st.title("ECG Founder — internal demo")

    # Sidebar — status indicator + sample-data buttons
    SAMPLES = {
        "AFib TP":                  "sample_data/sample_afib_tp.json",
        "PVC burden":               "sample_data/sample_pvc_burden.json",
        "Multi-finding strip":      "sample_data/sample_multi_finding.json",
        "Sin Tach false positive":  "sample_data/sample_sintach_fp.json",
    }
    with st.sidebar:
        st.markdown("### Status")
        model, device = get_model()                          # cached after first call
        st.markdown(f"● **model ready** ({device})")
        st.markdown("### Sample files")
        st.caption("Pre-staged fzark JSONs from the existing repo dataset.")
        for label, rel in SAMPLES.items():
            if st.button(label, use_container_width=True, key=f"sample_{label}"):
                with open(Path(__file__).parent / rel, "rb") as f:
                    st.session_state["sample_bytes"] = f.read()
                    st.session_state["sample_name"]  = Path(rel).name
                st.rerun()

    uploaded = st.file_uploader(
        "Drop or browse an ECG file",
        type=["csv", "json", "zip"],
        accept_multiple_files=False,
        help="CSV (raw signal), JSON (fzark-format), or ZIP (.dat + .hea pair). Max 25 MB.",
    )
    # Resolve input source: user upload takes priority, else last-clicked sample
    if uploaded is not None:
        file_bytes, file_name = uploaded.getvalue(), uploaded.name
    elif "sample_bytes" in st.session_state:
        file_bytes = st.session_state["sample_bytes"]
        file_name  = st.session_state["sample_name"]
        st.caption(f"📌 Using sample: **{file_name}**")
    else:
        st.stop()

    try:
        with st.spinner("Analyzing…"):
            tensor, meta = parse_file(file_name, file_bytes)
            result = infer(tensor, model, device)
    except (ValueError, RuntimeError) as exc:
        st.error(f"Could not process file: {exc}")    # surface exception_message (reviewer fix #3)
        st.stop()

    # Multi-lead banner (Option A)
    if meta.get("multi_lead_detected"):
        st.info(f"🔍 Multi-lead input detected — using Lead {meta['lead_used']} for inference.")

    sig_np = tensor.squeeze().numpy()
    render_waveform(sig_np, preprocessing_steps=[
        "50 Hz notch", "0.67–40 Hz bandpass", "median baseline", "winsorized zscore",
    ])
    render_findings(result["findings"])
    render_bar_chart(result["all_v31_head_probabilities"])

    # Metadata footer
    st.caption(
        f"{meta['filename']} · {meta['format']} · {meta['n_samples']} samples "
        f"@ {meta['fs_hz']} Hz · {meta['duration_s']:.1f} s · "
        f"inference {result['inference_ms']} ms"
    )

if __name__ == "__main__":
    main()
```

## 8. Visual language

Inherited verbatim from the archived FRONTEND.md §6 — Streamlit doesn't change clinical semantics, only the rendering primitives.

| Element | Mapping | Streamlit primitive |
|---|---|---|
| Risk-tier dot | Pulled from `FZARK_ONTOLOGY[event].risk_tier` | `st.markdown` with colored emoji or HTML `<span>` |
| Reliability pill | Pulled from `FZARK_ONTOLOGY[event].clinical_reliability` | `st.markdown` with inline `<span style="background:...">` |
| Probability bar | `findings[i].probability` × 100% | `st.progress(...)` or HTML `<progress>` |
| All-heads chart | `result["all_v31_head_probabilities"]` | `st.bar_chart(dict, horizontal=True)` |
| Multi-lead banner | `meta["multi_lead_detected"] == True` | `st.info("🔍 Multi-lead input detected — using Lead II for inference.")` |

**Helper functions** (`_risk_dot`, `_chip`, `_reliability_chip`, `_risk_rank`) are 3-5 lines each — total ~20 LoC.

## 9. Error handling

Per reviewer fix #3, **surface exception messages** for an internal-only audience:

| Condition | Streamlit response | Operator sees |
|---|---|---|
| Unsupported extension | `st.error("Could not process file: Unsupported file extension: .xyz")` | inline red box at top of result area |
| Parse failure (bad CSV / corrupt JSON) | `st.error("Could not process file: <pandas error>")` | inline red box with the actual error |
| Bad signal length (caught in parser) | `enforce_length_5000` truncates/pads silently — eliminates this error class | (resolved by design) |
| Model inference RuntimeError | `st.error("Could not process file: <torch error>")` | inline red box with shape mismatch / device error |
| File > 25 MB | Streamlit's `st.file_uploader` rejects natively | "File must be ≤ 25 MB" toast (built-in) |

The `try/except (ValueError, RuntimeError)` wrapper around `parse_file` + `infer` is the only error path. Anything else (network, OS, OOM) crashes Streamlit and the systemd unit restarts within 5 s.

## 10. Async / blocking concerns

**N/A under Streamlit** — this was reviewer's critical risk #1 for the FastAPI plan, but it doesn't apply here:

- Streamlit runs the script top-to-bottom on each rerun, in a Tornado coroutine per session.
- CPU-bound work (`model(tensor)`) inside a session **does not block other sessions** — they run on separate coroutines.
- `st.cache_resource` makes the model singleton thread-safe by serializing first-load across sessions.
- For a single-operator internal demo, contention is moot anyway.

The reviewer's `async def predict` trap is a FastAPI-specific footgun that Streamlit's programming model sidesteps.

## 11. Deployment runbook

Target host: clean Ubuntu 22.04 **`t3.large`** (8 GB RAM — per reviewer fix #2, NOT t3.medium).

```bash
# 1. System packages
sudo apt update && sudo apt install -y python3-pip git

# 2. Application
git clone <repo> && cd ECGFounder
pip3 install -r requirements.txt streamlit  # matplotlib NOT needed — st.line_chart used instead

# 3. Warm the checkpoint cache (avoids first-request 30 s download)
python3 -c "from checkpoints import load_ecgfounder; from device_utils import resolve_device; load_ecgfounder(resolve_device())"

# 4. systemd unit
sudo cp deploy/ecg-demo.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now ecg-demo

# 5. Verify
curl -I http://localhost:8000      # 200 expected
sudo journalctl -u ecg-demo -f
```

### `deploy/ecg-demo.service`

```ini
[Unit]
Description=ECG demo Streamlit app
After=network-online.target

[Service]
WorkingDirectory=/home/ubuntu/ECGFounder
Environment=PYTHONPATH=/home/ubuntu/ECGFounder
ExecStart=/usr/bin/python3 -m streamlit run app/app.py \
    --server.address 0.0.0.0 \
    --server.port 8000 \
    --server.maxUploadSize 25 \
    --browser.gatherUsageStats false \
    --server.headless true
Restart=always
RestartSec=5
User=ubuntu
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
```

### `app/.streamlit/config.toml`

```toml
[server]
port = 8000
address = "0.0.0.0"
headless = true
maxUploadSize = 25

[theme]
# minimal branding — adjust later if needed
primaryColor = "#2563eb"
backgroundColor = "#ffffff"
secondaryBackgroundColor = "#f3f4f6"

[browser]
gatherUsageStats = false
```

## 12. Sizing

Per reviewer fix #2, **bumped from t3.medium to t3.large**:

| Resource | Value | Rationale |
|---|---|---|
| Instance | `t3.large` (2 vCPU, **8 GB RAM**) | Baseline RAM budget: OS 600 MB + Streamlit/Python 800 MB + 2 cached model checkpoints in DualHead ~1.8 GB + pandas reshape spike on 25 MB CSV ~400 MB = **~3.6 GB worst case**. 4 GB on t3.medium leaves only ~300 MB headroom = OOM-killer risk. 8 GB gives ~4 GB headroom. |
| Concurrency | 1 active session at a time (internal demo) | Streamlit handles multiple sessions via Tornado coroutines, but the single GPU/MPS instance serializes inference. |
| Inbound | TCP 8000 from VPN CIDR only | No TLS / auth in scope. |
| Outbound | HTTPS for first-run checkpoint download (one-time) | After warm-up, no outbound required. |
| Disk | 15 GB | Repo ~500 MB, models 470 MB (base + v2/v3 fuzzy), Python deps ~2 GB, OS + logs minimal. |

## 13. Security (demo-grade, NOT production)

- ✅ EC2 security group: inbound 8000/tcp from internal VPN CIDR only
- ✅ Instance profile only — no long-lived AWS keys on the box (and the demo doesn't need AWS perms at all)
- ✅ `User=ubuntu` in systemd unit (not root)
- ❌ No TLS — internal HTTP only
- ❌ No authentication — IP allowlist via security group is the only gate
- ❌ No rate limiting
- ❌ No request audit logging beyond journald

If any of the ❌ items needs to become ✅, escalate per [README.md §"Out of scope"](README.md#out-of-scope--when-to-not-use-this-stack) — those changes mean re-promoting the [archived FastAPI plan](_archived/) (which has the middleware ecosystem to support them).

## 14. Reviewer feedback resolution table

### Round 1 (2026-05-28, FastAPI plan review → Streamlit pivot)

| Reviewer point | Resolution |
|---|---|
| **Risk 1 — async blocking trap** | N/A under Streamlit (§10). FastAPI-specific footgun avoided by framework choice. |
| **Risk 2 — t3.medium OOM** | **t3.large** baseline (§12) with explicit RAM budget table. |
| **Risk 3 — masking 500 errors in UI** | `st.error(str(exc))` surfaces full exception message (§9). Internal-only audience, no PII concern. |
| **Alternative — Streamlit** | **Adopted** (§3 + README.md §"Why Streamlit"). |
| **Multi-lead UX** | **Option A** — auto-extract Lead II, banner via `st.info(...)` (§4, §5, §8). New `meta["lead_used"]` + `meta["multi_lead_detected"]` fields in parser output. |

### Round 2 (2026-05-28, Streamlit-plan review)

| Reviewer point | Decision | Resolution |
|---|---|---|
| **Suggestion 1 — `st.line_chart` over `st.pyplot`** | ✅ Accept | §7 `render_waveform` rewritten; matplotlib dropped from `requirements.txt` (§6, §11); waveform now interactive (pan / zoom / hover) and ~5× faster to render. |
| **Suggestion 2 — robust CSV header detection** | ✅ Accept | §7 adds `read_csv_robust()` + `extract_lead_ii_from_dataframe()` with the heuristic: re-read with `header=None` iff every column name parses as a float. Multi-lead fallback defaults to column index 1. |
| **Suggestion 3 — sidebar suppression toggle** | ❌ Reject | Keep minimalist. Demo shows **raw model output only** — no `MultiClassFPSuppressor` import, no per-finding gate evaluation, no toggle widget. The v2 suppression effect is documented elsewhere (cross-dataset performance report); not needed in the live demo path. |
| **Suggestion 4 — pre-staged sample buttons** | ✅ Accept | Samples are sourced **from the existing fzark dataset** (`data/ecg_tp_fzark/` + `data/ecg_fp_doctor removed1/`) — the same format internal operators upload, so the format-compatibility risk doesn't apply. Implementation in §7; file layout in §6. |

## 15. Open questions for the next review pass

1. **Branding** — VivaLink logo at the top? "DEMO ONLY" watermark? `primaryColor` in `.streamlit/config.toml` is the only branding hook used so far. Default to no branding for the internal MVP unless the demo audience requests it.
2. **WFDB ZIP layout** — the parser expects `.dat` + `.hea` in a single zip. Worth confirming with one operator before phase 1 that this is an acceptable upload mechanism (vs. providing the two files separately).
3. **Specific sample-record picks** — phase 1 needs to choose four representative fzark JSONs from the repo. **Selection criteria** (decide during phase 1 by inspecting per-record predictions):
   - **AFib TP** — pick from `data/ecg_tp_fzark/Atrial Fibrillation/` where the AFib head fires ≥ 0.95. Demonstrates the "Reliable" head + PPV 91.6% badge lighting green.
   - **PVC burden** — pick from `data/ecg_tp_fzark/Isolated Ventricular Beat/` (or `Ventricular Couplet/` if found) where PVC head fires ≥ 0.85. Demonstrates v3.1 head-sharing (IVB + V Couplet both surface as findings off the same idx-9 head).
   - **Multi-finding strip** — pick a fzark record where ≥ 3 heads fire ≥ 0.5 simultaneously. Demonstrates the multi-label nature of the model (mean 8.4 heads firing per record at t=0.5 from §9 of GLOBAL_LABEL_MAP).
   - **Sin Tach false positive** — pick from `data/ecg_fp_doctor removed1/Sinus Tachycardia/` where the Sin Tach head fires ≥ 0.9. Demonstrates the "Severe FP" reliability chip (PPV 0.0%); also opens the conversation about why the v2 suppression layer exists.
   - Each file is copied to `app/sample_data/` (≤ 200 KB per file, < 1 MB total, safe to commit).

### Retired by round-2 review

- ~~CSV format policy~~ → resolved by `read_csv_robust()` + `extract_lead_ii_from_dataframe()` in §7 (reviewer suggestion 2 accepted).
- ~~Suppression layer toggle~~ → **rejected** (reviewer suggestion 3). Demo shows raw model output only.
- ~~Sample-data buttons~~ → **accepted** after re-review (reviewer suggestion 4). Sourced from existing fzark repo data — same format, no compatibility risk. See §7 for implementation and §6 for file layout; specific record picks deferred to phase 1 (item 3 above).

---

## 16. Reference — algorithm dependencies

The Streamlit app uses these existing repo modules **unchanged** (no fork, no copy):

| Module | What the app uses |
|---|---|
| [device_utils.py](../../device_utils.py) | `resolve_device()` |
| [checkpoints.py](../../checkpoints.py) | indirectly via `DualHeadECGFounder` |
| [dual_head_ecgfounder.py](../../dual_head_ecgfounder.py) | `DualHeadECGFounder` constructor + `forward` |
| [label_config.py](../../label_config.py) | `FZARK_ONTOLOGY`, `ClinicalReliability`, `ClinicalRiskTier` |
| [preprocessing.py](../../preprocessing.py) | `ECGPreprocessor.for_fzark()`, `from_fzark_json()` |

If any of these modules change shape, only `app/app.py` may need adjustment — and the existing test suite ([tests/test_ontology.py](../../tests/test_ontology.py)) guards the ontology surface.

---

*This is a planning document. No code committed under `app/` yet — implementation in phase 1 per [README.md §"Implementation phases"](README.md#implementation-phases).*
