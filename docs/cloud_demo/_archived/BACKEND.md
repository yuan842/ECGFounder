# Backend Design — ECG Cloud Demo

**Layer**: backend (FastAPI service on EC2)
**Status**: planning only
**Companion**: [FRONTEND.md](FRONTEND.md)
**Top-level**: [README.md](README.md)

---

## 1. Purpose & scope

Internal MVP HTTP API for synchronous, on-demand ECG inference using the **current production model stack**: `DualHeadECGFounder` (base + supplementary v2/v3 per-head routing) plus the v3.1 fzark ontology defined in [res/GLOBAL_LABEL_MAP.md](../../res/GLOBAL_LABEL_MAP.md).

Single-tenant, no auth, no persistence. **Demo / algorithm evaluation only — not for PHI or clinical use.**

## 2. Non-goals (explicit "do nots")

- ❌ No multi-tenancy, user accounts, JWT, or session state
- ❌ No async job queue, background workers, or polling
- ❌ No database, cache, or object storage
- ❌ No HIPAA / PHI handling — demo data only
- ❌ No S3, Lambda, API Gateway, or CloudFront
- ❌ No multi-worker scaling — `uvicorn --workers 1` is the production setting
- ❌ No request-level metrics endpoint — `journalctl` is enough

## 3. Service architecture

| Component | Responsibility | File | Approx LoC |
|---|---|---|---|
| HTTP server | FastAPI + uvicorn, port 8000 | `backend/app.py` | ~80 |
| Model service | Load `DualHeadECGFounder` once at startup; expose `predict(tensor) → dict` | `backend/model_service.py` | ~80 |
| Schemas | Pydantic request / response models | `backend/schemas.py` | ~40 |
| File parsers | Dispatch by extension to CSV / WFDB / JSON loaders | `backend/parsers.py` | ~70 |
| Static frontend mount | Serves `frontend/` at `/` via `StaticFiles` | (in `app.py`) | ~5 |

**Lifecycle**:

```
process start
   │
   ├─ FastAPI startup event
   │     ├─ resolve_device()  → "mps" | "cuda" | "cpu"
   │     ├─ DualHeadECGFounder(device, routing='ptbxl_specific')  ← takes ~30s
   │     └─ Pre-warm one inference on a (1, 1, 5000) zero tensor
   │
   ├─ for each request:
   │     ├─ parse multipart → torch.Tensor (1, 1, 5000)
   │     ├─ model.forward() under torch.no_grad()
   │     ├─ shape result via FZARK_ONTOLOGY metadata
   │     └─ render waveform PNG (matplotlib, 600×150) → base64
   │
   └─ SIGTERM → graceful shutdown (uvicorn handles)
```

Single warm process, single inference at a time. MPS / CUDA does not share across uvicorn workers, so additional workers would each load their own 350 MB copy — wasted RAM for no concurrency benefit on this workload.

## 4. API contract

This section is the **single source of truth** for the JSON shape that both backend and frontend depend on. If you change it, update both layers in the same commit.

### 4.1 `GET /healthz`

Liveness + readiness probe. Frontend calls on page load to show a green/red badge.

```http
GET /healthz
```

Response (200):
```json
{
  "status": "ok",
  "model": "1_lead_ECGFounder.pth",
  "device": "mps",
  "uptime_s": 142
}
```

While model is still loading (first ~30 s after process start):
```json
HTTP 503
{
  "status": "warming",
  "uptime_s": 8
}
```

### 4.2 `GET /` — serves `frontend/index.html`

Mounted via FastAPI `StaticFiles(directory='frontend', html=True)` at root. Static assets at `/app.js`, `/styles.css`, etc.

### 4.3 `POST /predict` — synchronous inference

**Request** — `multipart/form-data`:
- `file` (required): the ECG file. Accepted extensions:
  - `.csv` — single-column raw signal at 500 Hz (or whatever `fs` field a header row declares)
  - `.json` — fzark-format JSON (passed through `ECGPreprocessor.from_fzark_json`)
  - `.dat` paired with a `.hea` — WFDB record (uploaded as a `.zip` or two separate parts; see §6.2)
- Max body size: **25 MB** (enforced by uvicorn `--limit-max-body-size` and FastAPI dependency).

**Response (200)** — `application/json`:

```json
{
  "request_id": "550e8400-e29b-41d4-a716-446655440000",
  "received_at_utc": "2026-05-28T18:34:11.231Z",
  "inference_ms": 287,
  "input": {
    "filename": "sample_001.csv",
    "format": "csv",
    "n_samples": 5000,
    "fs_hz": 500,
    "duration_s": 10.0
  },
  "findings": [
    {
      "rank": 1,
      "event_label": "Atrial Fibrillation",
      "founder_head_idx": 5,
      "founder_head_name": "ATRIAL FIBRILLATION",
      "probability": 0.962,
      "risk_tier": "high",
      "reliability": "reliable",
      "ppv_pct": 91.6,
      "clinical_support": "Irregularly irregular atrial rhythm lacking distinct P waves. Maps directly to \"Atrial Fibrillation\" continuous rhythm alerts."
    }
  ],
  "all_v31_head_probabilities": {
    "Atrial Fibrillation":          0.962,
    "Bradycardia":                  0.041,
    "Sinus Tachycardia":            0.012,
    "Isolated Ventricular Beat":    0.103,
    "Isolated Supraventricular Beat": 0.058,
    "Supraventricular Couplet":     0.007,
    "Supraventricular Run":         0.003,
    "Ventricular Run":              0.001,
    "Pause":                        0.000,
    "ST Elevation":                 0.012
  },
  "waveform_png_b64": "iVBORw0KGgoAAAANSUhEUgAAA...",
  "preprocessing_applied": [
    "50 Hz notch",
    "0.67–40 Hz bandpass",
    "median baseline removal",
    "winsorized z-score"
  ]
}
```

**Response (4xx / 5xx)** — same envelope, with `error` instead of `findings`:

```json
HTTP 400
{
  "request_id": "uuid",
  "received_at_utc": "...",
  "error": {
    "code": "unsupported_format",
    "message": "File extension '.txt' is not supported. Use .csv, .json, or .dat.",
    "details": { "filename": "raw.txt" }
  }
}
```

### 4.4 Error codes

| HTTP | `code` | Cause |
|---|---|---|
| 400 | `unsupported_format` | File extension not in `{.csv, .json, .dat}` |
| 400 | `parse_error` | CSV / JSON / WFDB failed to decode into a numeric array |
| 400 | `bad_signal_length` | Decoded array has `< 500` or `> 50,000` samples |
| 413 | `payload_too_large` | Request body > 25 MB |
| 500 | `inference_error` | Anything raised by `model_service.predict()` — full stack trace logged but **not returned** |
| 503 | `model_warming` | Model not yet loaded (first ~30 s after process start) |

### 4.5 Result composition policy

These rules govern how the backend transforms the raw 150-element sigmoid vector into the response above. They are NOT negotiable for the frontend — frontend assumes them and does no clinical-decision logic.

| Rule | Behavior |
|---|---|
| **Findings threshold** | Include only events with `probability ≥ 0.5` in the `findings[]` array. Empty array is valid output. |
| **Findings ordering** | Descending by `probability`. Ties broken by `risk_tier` (critical > high > moderate > low) then by head index ascending. |
| **Findings source set** | The 13 events in `FZARK_LABEL_MAP` (v3.1 ontology). Each finding inherits `risk_tier`, `reliability`, `ppv_pct`, `clinical_support` directly from `FZARK_ONTOLOGY[event]` — no app-code defaults. |
| **Head-sharing dedup** | When multiple events route to the same head (e.g. IVB + V Couplet both → idx 9), emit one finding per *event*, not per head, so the UI can show "Isolated Ventricular Beat 0.85" and "Ventricular Couplet 0.85" separately even though the probability is the same. |
| **all_v31_head_probabilities** | Always present. Keyed by the 10 unique head canonical names (not by index). Always 10 entries regardless of threshold. Used by the frontend's full bar chart. |
| **waveform_png_b64** | 600×150 PNG of the preprocessed signal (after `ECGPreprocessor.for_fzark()` runs). Base64-encoded inline so the frontend doesn't need a second HTTP request. |
| **preprocessing_applied** | Verbatim list of preprocessing steps applied, sourced from the `ECGPreprocessor` instance. Lets the frontend caption the waveform. |

## 5. Project layout

```
backend/
├── __init__.py
├── app.py              # FastAPI app + routes + StaticFiles mount
├── model_service.py    # global ModelService singleton (warm, predict)
├── parsers.py          # parse_file(filename, bytes) -> torch.Tensor
├── schemas.py          # Pydantic models: HealthResponse, PredictResponse, ErrorResponse, Finding
└── deploy/
    └── ecg-demo.service
```

Target: each file < 100 lines. Total backend code budget: **<300 lines**.

## 6. Implementation notes

### 6.1 Model service (`model_service.py`)

- Singleton class, instantiated once in `app.py` startup hook.
- Wraps `DualHeadECGFounder(device, routing='ptbxl_specific')` so the demo automatically inherits whatever routing we ship in production.
- Exposes:
  - `is_warm() -> bool` (False until first inference completes; healthz uses this)
  - `predict(tensor: torch.Tensor) -> dict` (does the inference + result composition per §4.5)
- Pre-warm with one zero-tensor inference at startup so the first real request is fast.

### 6.2 Parsers (`parsers.py`)

```python
def parse_file(filename: str, raw: bytes) -> tuple[torch.Tensor, dict[str, Any]]:
    """Return (tensor of shape (1, 5000), metadata dict)."""
```

Dispatch by extension. For `.dat` we require the upload to be a `.zip` containing both `.dat` and `.hea` (browsers can't submit a multi-file form into one field cleanly; zip is the lowest-friction option). Documented in FRONTEND.md.

### 6.3 Schemas (`schemas.py`)

Pydantic v2 models matching the JSON in §4.3 exactly. One model per response variant. `model_config = ConfigDict(populate_by_name=True)` to allow camelCase aliasing if frontend prefers (decision: stick to `snake_case` end-to-end for simplicity).

### 6.4 App (`app.py`)

```python
# Pseudocode — actual implementation in phase 1
app = FastAPI()
service = ModelService()

@app.on_event("startup")
async def warm():
    await service.load()  # blocking ~30 s; uvicorn won't serve until done

@app.get("/healthz")
def healthz(): ...

@app.post("/predict", response_model=PredictResponse, responses={400: ..., 413: ..., 500: ..., 503: ...})
async def predict(file: UploadFile = File(...)): ...

app.mount("/", StaticFiles(directory="frontend", html=True), name="frontend")
```

`StaticFiles` mounts LAST so explicit routes win over the wildcard. Visiting `/` serves `frontend/index.html`.

### 6.5 Logging

Standard library `logging` with `logging.basicConfig` driven by uvicorn's `--log-config`. Structured one-line-per-event format. Per-request fields:

```
request_id, route, status_code, inference_ms, file_extension, n_samples, top_finding, error_code
```

Output goes to stdout/stderr → captured by journald via the systemd unit. Query with:

```bash
sudo journalctl -u ecg-demo -f -o json | jq 'select(.MESSAGE | contains("request_id"))'
```

No log file rotation in app code — journald handles retention via `/etc/systemd/journald.conf`.

## 7. Deployment runbook

Target host: clean Ubuntu 22.04 t3.medium (4 vCPU, 4 GB RAM, 15 GB EBS).

```bash
# 1. System packages
sudo apt update && sudo apt install -y python3-pip git

# 2. Application
git clone <repo> && cd ECGFounder
pip3 install -r requirements.txt fastapi 'uvicorn[standard]' python-multipart

# 3. Warm the checkpoint cache (avoids first-request 30 s download)
python3 -c "from checkpoints import load_ecgfounder; from device_utils import resolve_device; load_ecgfounder(resolve_device())"

# 4. systemd unit
sudo cp backend/deploy/ecg-demo.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now ecg-demo

# 5. Verify
curl http://localhost:8000/healthz
sudo journalctl -u ecg-demo -f
```

### `backend/deploy/ecg-demo.service`

```ini
[Unit]
Description=ECG demo API (frontend + backend)
After=network-online.target

[Service]
WorkingDirectory=/home/ubuntu/ECGFounder
ExecStart=/usr/bin/python3 -m uvicorn backend.app:app \
    --host 0.0.0.0 --port 8000 \
    --workers 1 \
    --limit-max-body-size 26214400
Restart=always
RestartSec=5
User=ubuntu
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
```

## 8. Sizing

| Resource | Value | Rationale |
|---|---|---|
| Instance | `t3.medium` (4 vCPU, 4 GB) | Model is 353 MB on disk, ~1.5 GB resident. 4 GB leaves headroom for the OS + matplotlib + uvicorn. |
| Concurrency | `--workers 1` | MPS / CUDA doesn't share across workers; additional workers each load their own model copy. Sequential inference at ~300 ms is fine for demo. |
| Inbound | TCP 8000, restricted to VPN CIDR | No TLS / auth in scope; not for public exposure. |
| Outbound | HTTPS for first-run checkpoint download (one-time) | After warm-up, no outbound network calls. |
| Disk | 15 GB | Repo ~500 MB, model 350 MB, OS + logs minimal. |

## 9. Security checklist (demo-grade, NOT production)

- ✅ EC2 security group: inbound 8000/tcp from internal VPN CIDR only
- ✅ No long-lived AWS access keys on the box — instance profile only (and the demo doesn't actually need any AWS perms)
- ✅ `User=ubuntu` in the systemd unit (not root)
- ❌ No TLS — internal HTTP only
- ❌ No request authentication — IP allowlist via security group is the only gate
- ❌ No rate limiting — single-tenant demo
- ❌ No request logging beyond journald — no PII to redact

If any of the ❌ rows need to become ✅, escalate per the [README §"Out of scope"](README.md#out-of-scope--when-to-not-use-this-stack) — those changes are a different architecture, not bolt-ons.

## 10. Open questions for the next review pass

These are things the planning document deliberately defers. Resolve before phase 1:

1. **CSV format ambiguity** — single-column raw signal, or column header row with `fs_hz` declared? The current plan assumes single-column. If someone might upload a multi-lead CSV from a Holter export, we need to either auto-detect Lead II or document the required format on the frontend.
2. **WFDB upload mechanism** — `.dat` + `.hea` as a zip is one option; uploading just the `.dat` and having backend re-construct a minimal `.hea` from query parameters is another. Decision deferred.
3. **What to do with the AFib FP suppression layer** — the current `DualHeadECGFounder` does not apply the v2 motion/HR/SNR feature gates. For a demo showing the *real* production output, we should apply them. For a demo showing *raw model behavior*, we should not. Pick one and document it on the result panel.
4. **Should the response include the v2 FP suppression decision per finding?** If we do apply the suppressor, surfacing "AFib detected, suppressed by motion-gate" is a powerful demo moment.
5. **Should `findings[]` ever be empty in the UI**, or do we always show at least the top-5 most-probable events regardless of threshold? Current plan: empty array allowed; frontend shows "No findings ≥ 0.5".

These are UX decisions that change the response contract — better to nail them down before writing code.

---

*This is a planning document. No code committed under `backend/` yet — implementation in phase 1 per [README.md §"Implementation phases"](README.md#implementation-phases-when-greenlit-again).*
